#include "addon/depth_choice.hpp"
#include "addon/frame_cap.hpp"
#include <reshade.hpp>
#include <d3d11.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include "addon/diag.hpp"

namespace fw {
namespace {
using namespace reshade::api;
using Microsoft::WRL::ComPtr;

// Generic Depth's per-runtime data (ReShade's add-on, examples/09-depth; BSD 3-Clause, (c) Patrick Mours):
// found through its identifier, as its own overlay finds it. Only the override is written - what ticking a
// depth buffer in its list does - and only after the texture it says it handed to the effects was seen to
// be the one the effects got (a build of it laid out differently is left alone).
struct __declspec(uuid("7c6363c7-f94e-437a-9160-141782c44a98")) generic_depth_data {
    resource selected_depth_stencil = {0};
    resource override_depth_stencil = {0};
    resource_view selected_shader_resource = {0};
    bool using_backup_texture = false;
};

struct Candidate {
    resource res{0};
    std::uint32_t w = 0, h = 0;
    format fmt = format::unknown;
    std::uint64_t first_frame = 0, last_frame = 0, binds = 0;
    double sum = 0;
    int probes = 0;
    bool multisampled = false;
};

// Depth buffers the game binds, while the feed runs (any thread).
std::atomic<bool> g_track{false};
std::atomic<std::uint64_t> g_frame{0};
std::mutex g_mutex;
std::unordered_map<std::uint64_t, Candidate> g_seen;  // by resource handle
// Depth buffers bound, all of them, per frame (depth_binds_per_frame): over the last 60 frames.
std::atomic<std::uint64_t> g_binds{0};
std::uint64_t g_rate_frame = 0, g_rate_binds = 0;
double g_rate = 0;

void note(command_list* cmd_list, resource_view dsv) {
    if (!g_track.load(std::memory_order_relaxed) || !dsv.handle || !cmd_list) return;
    device* dev = cmd_list->get_device();
    if (!dev || (dev->get_api() != device_api::d3d12 && dev->get_api() != device_api::d3d11 && dev->get_api() != device_api::vulkan)) return;
    const resource res = dev->get_resource_from_view(dsv);
    if (!res.handle) return;
    g_binds.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t frame = g_frame.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (const auto it = g_seen.find(res.handle); it != g_seen.end()) { it->second.last_frame = frame; ++it->second.binds; return; }
    // (a game can go through hundreds of them - shadow maps, probes, one per level: those long out of use go)
    if (g_seen.size() >= 512) {
        for (auto it = g_seen.begin(); it != g_seen.end();) it = it->second.last_frame + 300 < frame ? g_seen.erase(it) : std::next(it);
        if (g_seen.size() >= 512) return;
    }
    const resource_desc desc = dev->get_resource_desc(res);
    if (desc.type != resource_type::texture_2d && desc.type != resource_type::surface) return;
    Candidate c;
    c.res = res; c.w = desc.texture.width; c.h = desc.texture.height; c.fmt = desc.texture.format;
    c.first_frame = c.last_frame = frame; c.binds = 1;
    c.multisampled = desc.texture.samples > 1;  // (would need resolving: left to Generic Depth's own choice)
    g_seen.emplace(res.handle, c);
}

void on_bind(command_list* cmd_list, uint32_t, const resource_view*, resource_view dsv) { note(cmd_list, dsv); }
bool on_begin_pass(command_list* cmd_list, uint32_t, const render_pass_render_target_desc*, const render_pass_depth_stencil_desc* ds,
                   render_pass_flags) {
    if (ds) note(cmd_list, ds->view);
    return false;
}
resource g_twin{0};  // (see on_present)
std::atomic<bool> g_twin_gone{false};
void on_destroy(device*, resource res) {
    if (res.handle == g_twin.handle) g_twin_gone = true;
    if (!g_track.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_seen.erase(res.handle);
}

constexpr int kSettleFrames = 4;    // after an override: Generic Depth switches, copies, the effects render
constexpr int kReadbackFrames = 5;  // after the probe's copy, at least (and the GPU done with it: probe_ready)
constexpr int kRounds = 2;
constexpr double kMinScore = 0.5;   // brightness changes 1.5x stronger across the depth edges than elsewhere: clearly the picture's
// ...or, below that, at least 1.25x - a picture whose depth edges show only faintly (Elden Ring's fog and soft light: its scene depth
// scored 0.32-0.49 and was never taken, the warp had no depth); buffers unrelated to the picture score near 0 (0.00-0.08)
constexpr double kAcceptScore = 0.25;

// Reads XPAR.fx's probe back to the CPU (a few frames after the copy).
struct ProbeReader {
    ComPtr<ID3D12Resource> readback;
    ComPtr<ID3D11Texture2D> staging;  // (Direct3D 11)
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    std::uint64_t readback_bytes = 0;
    std::uint64_t present = 0;  // the present the copy was recorded for (frame_cap_gpu_done)
    int waited = 0;
    // (Vulkan: through ReShade's own calls - a readback buffer, and a fence of its own: frame_cap's present fence is Direct3D
    // only. Signalled on the queue at the next frame's effects, once ReShade has submitted the frame that holds the copy)
    device* vk_device = nullptr;
    resource vk_buffer{0};
    fence vk_fence{0};
    std::uint64_t vk_signalled = 0, vk_target = 0;  // (vk_target: the value that marks this copy done; 0: not signalled yet)
    bool vk_signal = false;                         // (the copy recorded: the fence signalled at the next readiness check)
    std::uint32_t vk_w = 0, vk_h = 0;
    void reset() {
        readback.Reset(); staging.Reset();
        if (vk_device) {
            if (vk_buffer.handle) vk_device->destroy_resource(vk_buffer);
            if (vk_fence.handle) vk_device->destroy_fence(vk_fence);
        }
        vk_device = nullptr; vk_buffer = {0}; vk_fence = {0};
        vk_signalled = vk_target = 0; vk_signal = false; vk_w = vk_h = 0;
    }
};
command_queue* g_queue = nullptr;  // (the effect runtime's, as of this frame - for ProbeReader's Vulkan fence)
// The copy done on the GPU: a fixed few frames were not enough with frames queued on the GPU (RE4 uncapped, 2-3 frames
// behind) - the probe was read before the GPU had written it, every buffer scored as nothing, no depth was ever chosen
// and the camera estimate, without depth, failed a third of the frames. (Read anyway after 120 frames: never stuck.)
bool probe_ready(ProbeReader& r) {
    if (r.vk_buffer.handle) {
        if (r.vk_signal) {
            r.vk_signal = false;
            if (r.vk_fence.handle && g_queue && g_queue->signal(r.vk_fence, r.vk_signalled + 1)) r.vk_target = ++r.vk_signalled;
        }
        // (no fence - not made, or not signalled: a fixed few frames more, as Direct3D 11 without fences did)
        if (r.vk_target ? r.vk_device->get_completed_fence_value(r.vk_fence) >= r.vk_target : r.waited >= 8) return true;
        return ++r.waited > 120;
    }
    if (frame_cap_gpu_done(r.present) || ++r.waited > 120) return true;
    return false;
}

struct Chooser {
    enum Phase { kIdle, kProbe, kSettled } phase = kIdle;
    std::vector<Candidate> cand;
    std::size_t at = 0;
    int round = 0, wait = 0;
    bool copied = false;
    resource chosen{0};
    bool ours = false, users = false, disabled = false, said_missing = false;
    std::uint64_t retry_at = 0, retry_after = 120;
    std::vector<std::uint64_t> tried;  // the buffers the last choice was made among
    std::size_t said_candidates = ~std::size_t(0);
    // Checking that the chosen buffer still fits the picture (another one came into use), while it is used.
    bool verifying = false, verify_copied = false;
    int verify_wait = 0, verify_fails = 0;
    ProbeReader reader;
    // Empty depth in a running game (see empty_depth_verdict): this choice's evidence, and the verdicts so far.
    bool all_flat = true;
    double detail = 0;
    std::vector<float> luma;          // the picture's brightness at the last probe of this choice
    std::vector<float> empty_luma;    // ... at the last verdict of empty depth
    int empty_verdicts = 0;
    std::uint64_t first_empty_frame = 0;
    bool restart_needed = false;
} c;

void say(const char* text) { fw::diag_log(reshade::log::level::info, text); }

// Of the picture's shape, whatever its size (a scene can be rendered at any fraction of the picture and
// scaled up, by an upscaler or by the game itself, or above it); what tells the scene's apart is the probe.
// Generic Depth's own check takes 54% to twice the picture's size only: a game that renders below that has
// its scene's depth buffer outside it, and Generic Depth then settles for a buffer of the picture's size with
// a handful of draw calls in it (Assassin's Creed Black Flag: a 1424x800 scene in a 3840x2160 picture).
bool fits_picture(const Candidate& k, std::uint32_t fw_, std::uint32_t fh) {
    if (k.w < 64 || k.h < 64 || !fw_ || !fh) return false;
    return std::fabs(float(fw_) / float(fh) - float(k.w) / float(k.h)) <= 0.1f;
}

bool copy_probe(ProbeReader& r, device* dev, command_list* cmd_list, resource_view probe) {
    r.present = frame_cap_presents() + 1;  // (recorded during this frame: done once its present is)
    r.waited = 0;
    if (dev->get_api() == device_api::vulkan) {
        const resource tex = dev->get_resource_from_view(probe);
        if (!tex.handle) return false;
        const resource_desc desc = dev->get_resource_desc(tex);
        if (desc.texture.format != format::r32g32_float) return false;
        if (r.vk_device != dev || !r.vk_buffer.handle || r.vk_w != desc.texture.width || r.vk_h != desc.texture.height) {
            r.reset();
            const std::uint64_t bytes = std::uint64_t(desc.texture.width) * desc.texture.height * 8;
            if (!dev->create_resource(resource_desc(bytes, memory_heap::readback, resource_usage::copy_dest), nullptr, resource_usage::copy_dest,
                                      &r.vk_buffer))
                return false;
            r.vk_device = dev;
            if (!dev->create_fence(0, fence_flags::none, &r.vk_fence)) r.vk_fence = {0};
            r.vk_w = desc.texture.width; r.vk_h = desc.texture.height;
        }
        // (ReShade leaves an effect's render targets readable by shaders between passes, as with Direct3D 12)
        cmd_list->barrier(tex, resource_usage::shader_resource, resource_usage::copy_source);
        cmd_list->copy_texture_to_buffer(tex, 0, nullptr, r.vk_buffer, 0);
        cmd_list->barrier(tex, resource_usage::copy_source, resource_usage::shader_resource);
        r.vk_signal = true;
        r.vk_target = 0;
        return true;
    }
    if (dev->get_api() == device_api::d3d11) {
        auto* tex = reinterpret_cast<ID3D11Texture2D*>(dev->get_resource_from_view(probe).handle);
        auto* context = reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native());
        if (!tex || !context) return false;
        D3D11_TEXTURE2D_DESC desc{};
        tex->GetDesc(&desc);
        if (desc.Format != DXGI_FORMAT_R32G32_FLOAT) return false;
        D3D11_TEXTURE2D_DESC have{};
        if (r.staging) r.staging->GetDesc(&have);
        if (!r.staging || have.Width != desc.Width || have.Height != desc.Height) {
            r.staging.Reset();
            ComPtr<ID3D11Device> d3d;
            tex->GetDevice(&d3d);
            D3D11_TEXTURE2D_DESC sd = desc;
            sd.MipLevels = 1; sd.ArraySize = 1; sd.Usage = D3D11_USAGE_STAGING; sd.BindFlags = 0;
            sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ; sd.MiscFlags = 0;
            if (!d3d || FAILED(d3d->CreateTexture2D(&sd, nullptr, &r.staging))) return false;
        }
        context->CopySubresourceRegion(r.staging.Get(), 0, 0, 0, 0, tex, 0, nullptr);
        return true;
    }
    auto* tex = reinterpret_cast<ID3D12Resource*>(dev->get_resource_from_view(probe).handle);
    auto* d3d = reinterpret_cast<ID3D12Device*>(dev->get_native());
    auto* list = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native());
    if (!tex || !d3d || !list) return false;
    const D3D12_RESOURCE_DESC desc = tex->GetDesc();
    if (desc.Format != DXGI_FORMAT_R32G32_FLOAT) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    d3d->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, &total);
    if (!r.readback || r.readback_bytes != total) {
        r.readback.Reset();
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r.readback))))
            return false;
        r.readback_bytes = total;
    }
    r.footprint = fp;
    // ReShade leaves an effect's render targets readable by shaders between passes.
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = tex; b.Transition.Subresource = 0;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dst{r.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src{tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    list->ResourceBarrier(1, &b);
    return true;
}

bool read_probe(ProbeReader& r, std::vector<float>& px, std::uint32_t& w, std::uint32_t& h) {
    if (r.vk_buffer.handle) {
        void* mapped = nullptr;
        const std::uint64_t bytes = std::uint64_t(r.vk_w) * r.vk_h * 8;
        if (!r.vk_device->map_buffer_region(r.vk_buffer, 0, bytes, map_access::read_only, &mapped) || !mapped) return false;
        w = r.vk_w; h = r.vk_h;
        px.resize(std::size_t(w) * h * 2);
        std::memcpy(px.data(), mapped, std::size_t(bytes));
        r.vk_device->unmap_buffer_region(r.vk_buffer);
        return true;
    }
    if (r.staging) {
        ComPtr<ID3D11Device> d3d;
        ComPtr<ID3D11DeviceContext> context;
        r.staging->GetDevice(&d3d);
        if (d3d) d3d->GetImmediateContext(&context);
        D3D11_TEXTURE2D_DESC desc{};
        r.staging->GetDesc(&desc);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (!context || FAILED(context->Map(r.staging.Get(), 0, D3D11_MAP_READ, 0, &mapped)) || !mapped.pData) return false;
        w = desc.Width; h = desc.Height;
        px.resize(std::size_t(w) * h * 2);
        for (std::uint32_t y = 0; y < h; ++y)
            std::memcpy(&px[std::size_t(y) * w * 2], static_cast<const std::uint8_t*>(mapped.pData) + std::size_t(y) * mapped.RowPitch, std::size_t(w) * 8);
        context->Unmap(r.staging.Get(), 0);
        return true;
    }
    if (!r.readback) return false;
    w = r.footprint.Footprint.Width; h = r.footprint.Footprint.Height;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(r.readback_bytes)};
    std::uint8_t* mapped = nullptr;
    if (FAILED(r.readback->Map(0, &range, reinterpret_cast<void**>(&mapped))) || !mapped) return false;
    px.resize(std::size_t(w) * h * 2);
    for (std::uint32_t y = 0; y < h; ++y)
        std::memcpy(&px[std::size_t(y) * w * 2], mapped + r.footprint.Offset + std::size_t(y) * r.footprint.Footprint.RowPitch, std::size_t(w) * 8);
    const D3D12_RANGE none{0, 0};
    r.readback->Unmap(0, &none);
    return true;
}

// Which way round the game stores its depth (depth_orientation_ready). The depth XPAR.fx hands over is
// near = 1 and proportional to 1 / distance, so nearly everything in a picture - and the sky, 0 - lies
// close to zero; the wrong way round, it all bunches up near 1. Three readings in a row decide.
struct Orientation {
    ProbeReader reader;
    bool decided = false, flip = false, copied = false;
    int wait = 1, wrong = 0, right = 0;
    std::uint64_t checks = 0;
} g_orient;

// The game draws a 3D scene into its depth buffers, yet by the end of the frame (when ReShade's effects and
// XPAR read it) every one of them is empty: the game clears or reuses them before that. ReShade's Generic
// Depth can keep a copy from before the clear ("Copy depth buffer before clear operations"), a setting it
// reads when the game starts. Only on firm evidence, so that a menu, a loading screen or an intro never
// sets it: every candidate flat, a 3D scene being drawn (the buffers bound 20 times a frame or more - a menu
// or a video hardly binds them), a picture with detail that keeps changing between the checks, three
// times over 15 seconds at least. Then the setting goes one step up (off -> before clears -> also before
// full-screen draws) and the game has to be restarted once.
void empty_depth_verdict(std::uint64_t frame) {
    bool busy = false;
    for (const auto& k : c.cand)
        if (frame > k.first_frame && double(k.binds) / double(frame - k.first_frame) >= 20.0) busy = true;
    const bool empty = !c.cand.empty() && c.all_flat && busy && c.detail >= 0.03 && !c.luma.empty();
    if (!empty) { c.empty_verdicts = 0; c.empty_luma.clear(); return; }
    bool changing = true;
    if (!c.empty_luma.empty() && c.empty_luma.size() == c.luma.size()) {
        double diff = 0;
        for (std::size_t i = 0; i < c.luma.size(); ++i) diff += std::fabs(double(c.luma[i]) - c.empty_luma[i]);
        changing = diff / double(c.luma.size()) >= 0.02;
    }
    if (c.empty_verdicts > 0 && !changing) return;  // (a still picture: no evidence either way)
    if (c.empty_verdicts == 0) c.first_empty_frame = frame;
    ++c.empty_verdicts;
    c.empty_luma = c.luma;
    if (c.empty_verdicts < 3 || frame < c.first_empty_frame + 900 || c.restart_needed) return;
    int copy = 0;
    reshade::get_config_value(nullptr, "DEPTH", "DepthCopyBeforeClears", copy);
    char text[260];
    if (copy >= 2) {
        say("XPAR: the game's depth buffers still read empty with ReShade copying them before clears and full-screen draws: "
            "the depth has to be set up by hand for this game (ReShade's Add-ons tab, Generic Depth)");
        c.empty_verdicts = -1000000;  // (said once)
        return;
    }
    reshade::set_config_value(nullptr, "DEPTH", "DepthCopyBeforeClears", copy + 1);
    std::snprintf(text, sizeof(text), "XPAR: the game draws into its depth buffers but they read empty by the end of the frame: ReShade's "
                  "'Copy depth buffer before %s' switched on - restart the game once", copy == 0 ? "clear operations" : "full-screen draw calls");
    say(text);
    c.restart_needed = true;
}

// The size the scene's depth is drawn at (scene_depth_size), for the buffer chosen at the time.
std::uint64_t g_scene_of = 0;
std::uint32_t g_scene_w = 0, g_scene_h = 0;
std::vector<std::uint64_t> g_scene_larger;  // the larger buffers the last choice that saw both found holding the same scene

void settle(generic_depth_data* gd, std::uint64_t frame) {
    empty_depth_verdict(frame);
    const Candidate* best = nullptr;
    for (const auto& k : c.cand)
        if (k.probes && (!best || k.sum / k.probes > best->sum / best->probes)) best = &k;
    // Games keep copies of the scene's depth (for water, for effects): those score the same as the buffer the
    // scene is drawn into. Among the ones close to the best, the one the game binds most - the scene's own,
    // complete however late in the frame the copies are taken.
    if (best)
        for (const auto& k : c.cand)
            if (k.probes && k.sum / k.probes >= 0.85 * (best->sum / best->probes) && k.binds > best->binds) best = &k;
    // A reduced copy of the depth can score higher than the scene's own, its edges coarser and so on more of the picture's: Enshrouded's 640x360 at 1.49, the 1280x720 it
    // reduces at 1.17. Of the same shape, larger, drawn into more and at half the score or better (a stopgap): that one.
    if (best)
        for (const auto& k : c.cand)
            if (k.probes && k.w > best->w && k.w * best->h == k.h * best->w && k.binds > best->binds && k.sum / k.probes >= 0.5 * (best->sum / best->probes)) best = &k;
    char text[240];
    if (best && best->sum / best->probes >= kAcceptScore) {
        gd->override_depth_stencil = best->res;
        c.chosen = best->res; c.ours = true; c.retry_after = 120;
        // The scene's depth at the game's render resolution: of the buffers that hold the same scene (close to the best, as above),
        // of the same shape, the smallest - Cyberpunk with XeSS keeps it at 1280x720 and at the picture's 3840x2160, scoring
        // 1.65 and 1.59, either chosen by turns; the feed is handed at the smaller whichever it is (scene_depth_size).
        // A choice made where only the larger is drawn (Cyberpunk's menus: the 1280x720 ones not bound at all) keeps the size the last choice
        // that saw both found, for that same larger buffer: otherwise the feed went back to the larger after every menu.
        const std::uint32_t kept_w = g_scene_w, kept_h = g_scene_h;
        const bool kept_for_best = std::find(g_scene_larger.begin(), g_scene_larger.end(), best->res.handle) != g_scene_larger.end();
        const auto same_scene = [&](const Candidate& k) {
            return k.probes && k.sum / k.probes >= 0.85 * (best->sum / best->probes) && std::uint64_t(k.w) * best->h == std::uint64_t(k.h) * best->w;
        };
        g_scene_of = best->res.handle; g_scene_w = best->w; g_scene_h = best->h;
        std::uint32_t largest = best->w;
        for (const auto& k : c.cand)
            if (same_scene(k)) {
                if (k.w < g_scene_w) { g_scene_w = k.w; g_scene_h = k.h; }
                largest = std::max(largest, k.w);
            }
        if (largest > g_scene_w) {  // (both seen: the larger ones noted)
            g_scene_larger.clear();
            for (const auto& k : c.cand)
                if (same_scene(k) && k.w > g_scene_w) g_scene_larger.push_back(k.res.handle);
        } else if (kept_for_best && kept_w && kept_w < best->w) {
            g_scene_w = kept_w; g_scene_h = kept_h;
        }
        std::snprintf(text, sizeof(text), "XPAR: depth buffer chosen: %ux%u, format %u (score %.2f%s, of %zu tried)", best->w, best->h,
                      unsigned(best->fmt), best->sum / best->probes, best->sum / best->probes < kMinScore ? ": faint, the best there is" : "",
                      std::max(c.tried.size(), c.cand.size()));
    } else {
        // Nothing stands out (a menu, a loading screen, a dark or blurred picture): ReShade's own choice for now.
        gd->override_depth_stencil = {0};
        c.chosen = {0}; c.ours = false;
        // (asked again later, each time after twice as long: every question is a second without reprojection - but not
        // longer after an empty depth: a menu or a loading screen, nothing drawn yet. RE4 spends its first minute there,
        // and the gaps had grown to a minute by the time its game began: depth chosen a minute into the game)
        c.retry_at = frame + c.retry_after;
        if (!c.all_flat) c.retry_after = std::min<std::uint64_t>(c.retry_after * 2, 3600);
        std::snprintf(text, sizeof(text), "XPAR: no depth buffer stands out among %zu (best score %.2f): ReShade's own choice for now", std::max(c.tried.size(), c.cand.size()),
                      best ? best->sum / best->probes : 0.0);
    }
    say(text);
    if (c.tried.empty()) for (const auto& k : c.cand) c.tried.push_back(k.res.handle);
    c.phase = Chooser::kSettled;
}

}  // namespace

// A game can draw each frame into the next of two depth buffers, the other being last frame's for its temporal passes (Enshrouded). The buffer chosen is then
// a frame old every other frame: on half of the frames depth and motion disagreed, and the camera fit and the warp with them. Generic Depth stays on the chosen one;
// its twin - the same size and format, bound the nearest number of times - is copied each frame into a texture of ours, as Generic Depth copies when it
// cannot read a buffer directly, and XPAR.fx sees which of the two changed.
effect_runtime* g_runtime = nullptr;
resource g_twin_copy{0};
resource_view g_twin_view{0};
std::uint64_t g_twin_of = 0;
int g_said_why = 0;

void why(int reason, const char* text) {  // (each reason once, in turn: a game whose twin is not read says what holds it)
    if (reason == g_said_why) return;
    g_said_why = reason;
    say(text);
}
void drop_twin(effect_runtime* runtime) {
    g_twin_gone = false;
    g_twin_of = 0;
    if (runtime) {
        if (const auto v = runtime->find_uniform_variable("XPAR.fx", "XPAR_Twin"); v.handle) runtime->set_uniform_value_bool(v, false);
        if (g_twin_view.handle) runtime->update_texture_bindings("XPAR_TWIN_DEPTH", resource_view{0}, resource_view{0});
        if (g_twin_view.handle) runtime->get_device()->destroy_resource_view(g_twin_view);
        if (g_twin_copy.handle) runtime->get_device()->destroy_resource(g_twin_copy);
    }
    g_twin = {0};
    g_twin_copy = {0};
    g_twin_view = {0};
}
bool bind_twin(effect_runtime* runtime, resource res) {
    device* dev = runtime->get_device();
    drop_twin(runtime);
    resource_desc desc = dev->get_resource_desc(res);
    if ((desc.usage & resource_usage::copy_source) == 0) { why(2, "XPAR: the second depth buffer cannot be copied (made before XPAR could ask for it): restart the game"); return false; }
    const format view_format = format_to_default_typed(desc.texture.format, 0);
    if (dev->get_api() != device_api::vulkan) desc.texture.format = format_to_typeless(desc.texture.format);
    desc.heap = memory_heap::gpu_only;
    desc.usage = resource_usage::shader_resource | resource_usage::copy_dest;
    if (!dev->create_resource(desc, nullptr, resource_usage::shader_resource, &g_twin_copy)) { g_twin_copy = {0}; why(3, "XPAR: no texture for the second depth buffer"); return false; }
    if (!dev->create_resource_view(g_twin_copy, resource_usage::shader_resource, resource_view_desc(view_format), &g_twin_view)) {
        dev->destroy_resource(g_twin_copy); g_twin_copy = {0}; g_twin_view = {0};
        why(4, "XPAR: no view of the second depth buffer's texture");
        return false;
    }
    g_twin = res;
    runtime->update_texture_bindings("XPAR_TWIN_DEPTH", g_twin_view, g_twin_view);
    if (const auto v = runtime->find_uniform_variable("XPAR.fx", "XPAR_Twin"); v.handle) runtime->set_uniform_value_bool(v, true);
    else why(5, "XPAR: this XPAR.fx cannot read a second depth buffer (an older one?)");
    return true;
}
void on_present(command_queue*, swapchain*, const rect*, const rect*, uint32_t, const rect*) {
    if (!g_runtime || !g_track.load(std::memory_order_relaxed)) return;
    if (g_twin_gone.load()) drop_twin(g_runtime);
    const bool settled = c.phase == Chooser::kSettled && c.ours && !c.users && !c.verifying && c.chosen.handle;
    if (!g_runtime->get_private_data<generic_depth_data>()) return;
    if (!settled) {
        if (g_twin.handle) drop_twin(g_runtime);
        return;
    }
    if (g_twin.handle && c.chosen.handle == g_twin_of) return;
    if (g_twin.handle) drop_twin(g_runtime);
    resource twin{0};
    char text[200] = "";
    {
        const std::uint64_t frame = g_frame.load(std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto chosen = g_seen.find(c.chosen.handle);
        if (chosen == g_seen.end() || chosen->second.binds < 60) return;
        // (the one of the same size and format in use, bound the nearest number of times - no gate on how near: Enshrouded's pair was bound 111 and 105 times
        // in one session, past the twentieth asked before, and every other frame's depth was the frame before's (seq_4: 100% identical on alternate frames) -
        // the character held and let go in turn, a shake. Which of the two to read is decided each frame by XPAR.fx (XPAR_PickPS): the twin only when the chosen
        // one did not change and it did, so a buffer bound beside it that is not its twin is never read while the chosen one is drawn)
        std::uint64_t nearest = ~0ull, twin_binds = 0;
        for (const auto& [handle, k] : g_seen)
            if (handle != c.chosen.handle && k.last_frame + 2 >= frame && k.w == chosen->second.w && k.h == chosen->second.h && k.fmt == chosen->second.fmt) {
                const std::uint64_t apart = k.binds > chosen->second.binds ? k.binds - chosen->second.binds : chosen->second.binds - k.binds;
                if (apart < nearest) { nearest = apart; twin = k.res; twin_binds = k.binds; }
            }
        if (!twin.handle) return;
        std::snprintf(text, sizeof(text), "XPAR: a second depth buffer like the chosen one (%ux%u, bound %llu and %llu times): both are read, the one drawn this frame used",
                      chosen->second.w, chosen->second.h, static_cast<unsigned long long>(chosen->second.binds), static_cast<unsigned long long>(twin_binds));
    }
    // (outside the lock: binding destroys the texture before, and on_destroy takes it)
    if (bind_twin(g_runtime, twin)) {
        g_twin_of = c.chosen.handle;
        say(text);
    }
}
void on_begin_effects(effect_runtime*, command_list* cmd_list, resource_view, resource_view) {
    if (!g_twin.handle || !g_twin_copy.handle) return;
    constexpr auto kGame = resource_usage::depth_stencil | resource_usage::shader_resource;  // (as Generic Depth takes the state of its own)
    const resource both[2] = {g_twin, g_twin_copy};
    const resource_usage from[2] = {kGame, resource_usage::shader_resource}, to[2] = {resource_usage::copy_source, resource_usage::copy_dest};
    cmd_list->barrier(2, both, from, to);
    cmd_list->copy_resource(g_twin, g_twin_copy);
    cmd_list->barrier(2, both, to, from);
}
void on_destroy_runtime(effect_runtime* runtime) { if (runtime == g_runtime) { drop_twin(runtime); g_runtime = nullptr; } }

void register_depth_choice_events() {
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy);
    reshade::register_event<reshade::addon_event::present>(on_present);
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_runtime);
    reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
}
void unregister_depth_choice_events() {
    reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
    reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_runtime);
    reshade::unregister_event<reshade::addon_event::present>(on_present);
    reshade::unregister_event<reshade::addon_event::destroy_resource>(on_destroy);
    reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    c.reader.reset();
    g_orient.reader.reset();
}

DepthChoice choose_depth_buffer(effect_runtime* runtime, command_list* cmd_list, resource_view probe, resource_view bound_depth) {
    device* dev = runtime->get_device();
    g_queue = runtime->get_command_queue();
    g_runtime = runtime;
    const std::uint64_t frame = g_frame.load();
    if (c.disabled) return DepthChoice::kUnavailable;
    auto* gd = runtime->get_private_data<generic_depth_data>();
    if (!gd) {
        if (!c.said_missing) { say("XPAR: ReShade's Generic Depth add-on is not active: no depth buffer to choose from"); c.said_missing = true; }
        return DepthChoice::kUnavailable;
    }
    if (bound_depth.handle && gd->selected_shader_resource.handle && bound_depth.handle != gd->selected_shader_resource.handle) {
        say("XPAR: this ReShade's Generic Depth is not the one XPAR knows: the depth buffer is left to its own choice");
        c.disabled = true;
        return DepthChoice::kUnavailable;
    }
    // The picture-shaped depth buffers in use now: bound within the last 30 presented frames (a game may
    // present several times per frame it renders), for 30 at least (one used once while loading is not one).
    std::uint32_t fw_ = 0, fh = 0;
    runtime->get_screenshot_width_and_height(&fw_, &fh);
    std::vector<Candidate> in_use;
    std::size_t seen = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        seen = g_seen.size();
        for (const auto& [handle, k] : g_seen)
            if (k.last_frame + 30 >= frame && frame >= k.first_frame + 30 && !k.multisampled && fits_picture(k, fw_, fh)) in_use.push_back(k);
    }
    switch (c.phase) {
    case Chooser::kSettled: {
        // A depth buffer came into use that the choice was not made among (the game left its menu, loaded a
        // level): choose again, unless the user chose.
        // With a buffer chosen, that one is first checked against the picture (while it keeps being used):
        // only when it no longer fits, twice in a row, is the question asked again - games bring small
        // buffers in and out of use all the time, and every question is a second or two without reprojection.
        if (!c.users && !c.verifying) {
            bool fresh = false, fresh_smaller = false;
            std::uint32_t chosen_w = 0, chosen_h = 0;
            for (const auto& k : in_use)
                if (k.res.handle == c.chosen.handle) { chosen_w = k.w; chosen_h = k.h; }
            for (const auto& k : in_use)
                if (std::find(c.tried.begin(), c.tried.end(), k.res.handle) == c.tried.end()) {
                    fresh = true; c.tried.push_back(k.res.handle);
                    // (smaller than the chosen, of its shape: maybe the scene's depth at the game's render resolution, the chosen one
                    // its copy at the picture's - chosen in a menu, where only that one is drawn (Cyberpunk). Chosen again among both:
                    // the feed is handed at the smaller - scene_depth_size)
                    if (chosen_w && fw_ && fh && chosen_w >= fw_ && chosen_h >= fh &&  // (the chosen one the picture's size: see reshade_feed's use)
                        k.w < chosen_w && std::uint64_t(k.w) * chosen_h == std::uint64_t(k.h) * chosen_w &&
                        std::find(g_scene_larger.begin(), g_scene_larger.end(), c.chosen.handle) == g_scene_larger.end())
                        fresh_smaller = true;
                }
            if (fresh_smaller) {
                say("XPAR: a smaller depth buffer of the chosen one's shape came into use: choosing again among both");
                gd->override_depth_stencil = {0};
                c.chosen = {0}; c.ours = false; c.retry_after = 120; c.phase = Chooser::kIdle;
                return DepthChoice::kSettled;
            }
            if (fresh && in_use.size() >= 2) {
                if (c.chosen.handle) {
                    c.verifying = true; c.verify_copied = false; c.verify_wait = 1; c.verify_fails = 0;
                } else {
                    say("XPAR: another depth buffer came into use: choosing again");
                    gd->override_depth_stencil = {0};
                    c.retry_after = 120; c.phase = Chooser::kIdle;
                    return DepthChoice::kSettled;
                }
            }
        }
        if (c.verifying && !c.users && c.chosen.handle && --c.verify_wait <= 0) {
            if (!c.verify_copied) {
                if (probe.handle && copy_probe(c.reader, dev, cmd_list, probe)) { c.verify_copied = true; c.verify_wait = kReadbackFrames; }
                else c.verifying = false;
            } else if (!probe_ready(c.reader)) {
                c.verify_wait = 1;
            } else {
                std::vector<float> px;
                std::uint32_t w = 0, h = 0;
                const DepthProbeScore s = read_probe(c.reader, px, w, h) ? score_depth_probe(px.data(), w, h) : DepthProbeScore{};
                char text[200];
                if (s.score >= kAcceptScore) {
                    std::snprintf(text, sizeof(text), "XPAR: another depth buffer came into use; the chosen one still fits the picture (score %.2f)", s.score);
                    say(text);
                    c.verifying = false;
                } else if (++c.verify_fails < 2) {
                    c.verify_copied = false; c.verify_wait = 30;  // (a dark or blurred moment: once more, half a second later)
                } else {
                    std::snprintf(text, sizeof(text), "XPAR: another depth buffer came into use and the chosen one no longer fits the picture (score %.2f): choosing again", s.score);
                    say(text);
                    c.verifying = false;
                    gd->override_depth_stencil = {0};
                    c.chosen = {0}; c.ours = false; c.retry_after = 120; c.phase = Chooser::kIdle;
                    return DepthChoice::kSettled;
                }
            }
        }
        if (c.chosen.handle) {
            // Another one ticked in ReShade's list meanwhile: the user's word stands.
            if (!c.users && gd->override_depth_stencil.handle != c.chosen.handle) {
                say("XPAR: another depth buffer was ticked in ReShade's list: that one is used");
                c.users = true; c.ours = false;
            }
            if (c.users) return DepthChoice::kSettled;
            // The chosen buffer is gone or out of use (another resolution, another level): choose again.
            std::uint64_t last = 0;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                if (const auto it = g_seen.find(c.chosen.handle); it != g_seen.end()) last = it->second.last_frame;
            }
            if (frame > last + 120) {
                say("XPAR: the chosen depth buffer is no longer in use: choosing again");
                gd->override_depth_stencil = {0};
                c.chosen = {0}; c.ours = false; c.phase = Chooser::kIdle;
            }
        } else if (frame >= c.retry_at) {
            c.phase = Chooser::kIdle;
        }
        return DepthChoice::kSettled;
    }
    case Chooser::kIdle: {
        if (gd->override_depth_stencil.handle) {  // ticked by the user before XPAR got to it
            say("XPAR: a depth buffer is ticked in ReShade's list: that one is used");
            c.chosen = gd->override_depth_stencil; c.users = true; c.ours = false; c.phase = Chooser::kSettled;
            return DepthChoice::kSettled;
        }
        c.cand = in_use;
        std::sort(c.cand.begin(), c.cand.end(), [](const Candidate& a, const Candidate& b) { return a.binds > b.binds; });
        if (c.cand.size() > 16) c.cand.resize(16);
        for (auto& k : c.cand) { k.sum = 0; k.probes = 0; }
        if (c.cand.empty() || !probe.handle) {
            // None (nothing to choose), or an XPAR.fx without the probe: ReShade's own choice until another
            // comes into use. (A single one is still probed: whether it holds any depth at all.)
            if (c.said_candidates != c.cand.size()) {
                char text[200];
                std::snprintf(text, sizeof(text), "XPAR: %zu depth buffer(s) of the picture's shape in use (of %zu the game binds)%s: ReShade's own choice",
                              c.cand.size(), seen, probe.handle ? "" : ", and XPAR.fx has no probe (an older XPAR.fx?)");
                say(text);
                c.said_candidates = c.cand.size();
                // (what the game binds, for the log: the twelve bound most recently)
                std::vector<Candidate> all;
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    for (const auto& [handle, k] : g_seen) all.push_back(k);
                }
                std::sort(all.begin(), all.end(), [](const Candidate& a, const Candidate& b) { return a.last_frame != b.last_frame ? a.last_frame > b.last_frame : a.binds > b.binds; });
                for (std::size_t i = 0; i < all.size() && i < 12; ++i) {
                    const Candidate& k = all[i];
                    std::snprintf(text, sizeof(text), "XPAR:   %ux%u format %u%s, bound %llu times, first %llu frames ago, last %llu frames ago (picture %ux%u)",
                                  k.w, k.h, unsigned(k.fmt), k.multisampled ? " multisampled" : "", static_cast<unsigned long long>(k.binds),
                                  static_cast<unsigned long long>(frame - k.first_frame), static_cast<unsigned long long>(frame - k.last_frame), fw_, fh);
                    say(text);
                }
            }
            c.tried.clear();
            for (const auto& k : c.cand) c.tried.push_back(k.res.handle);
            c.chosen = {0}; c.retry_at = frame + 600; c.phase = Chooser::kSettled;
            return DepthChoice::kSettled;
        }
        c.said_candidates = ~std::size_t(0);
        c.at = 0; c.round = 0; c.copied = false; c.wait = kSettleFrames; c.tried.clear();
        c.all_flat = true; c.detail = 0; c.luma.clear();
        gd->override_depth_stencil = c.cand[0].res;
        c.phase = Chooser::kProbe;
        return DepthChoice::kChoosing;
    }
    case Chooser::kProbe: {
        if (--c.wait > 0) return DepthChoice::kChoosing;
        if (!c.copied) {
            if (!probe.handle || !copy_probe(c.reader, dev, cmd_list, probe)) {
                gd->override_depth_stencil = {0};
                c.chosen = {0}; c.retry_at = frame + 600; c.phase = Chooser::kSettled;
                return DepthChoice::kChoosing;
            }
            c.copied = true; c.wait = kReadbackFrames;
            return DepthChoice::kChoosing;
        }
        if (!probe_ready(c.reader)) { c.wait = 1; return DepthChoice::kChoosing; }
        std::vector<float> px;
        std::uint32_t w = 0, h = 0;
        Candidate& k = c.cand[c.at];
        if (read_probe(c.reader, px, w, h)) {
            const DepthProbeScore s = score_depth_probe(px.data(), w, h);
            if (s.unlit) {
                // (no picture yet where the depth has edges - fading in from black: nothing judged, the choice starts over a little later)
                char text[160];
                std::snprintf(text, sizeof(text), "XPAR: depth buffer %ux%u probed on a picture without any brightness across its edges (fading in?): choosing again later", k.w, k.h);
                say(text);
                gd->override_depth_stencil = {0};
                c.chosen = {0}; c.ours = false; c.retry_at = frame + 120; c.phase = Chooser::kSettled;
                return DepthChoice::kChoosing;
            }
            k.sum += s.score; ++k.probes;
            c.all_flat = c.all_flat && s.flat;
            c.detail = std::max(c.detail, s.detail);
            c.luma.resize(std::size_t(w) * h);
            for (std::size_t i = 0; i < c.luma.size(); ++i) c.luma[i] = px[i * 2 + 1];
            char text[240];
            std::snprintf(text, sizeof(text), "XPAR: depth buffer %zu/%zu: %ux%u format %u, bound %llu times: covers %.0f%% of the picture, edges %.1f%%, "
                          "brightness changes %.2fx across them (score %.2f)", c.at + 1, c.cand.size(), k.w, k.h, unsigned(k.fmt),
                          static_cast<unsigned long long>(k.binds), s.coverage * 100.0, s.edges * 100.0, s.lift, s.score);
            say(text);
        }
        if (++c.at == c.cand.size()) {
            c.at = 0; ++c.round;
            // (every candidate once; then only the best three again, for a second look at another moment)
            std::sort(c.cand.begin(), c.cand.end(), [](const Candidate& a, const Candidate& b) {
                return a.sum / std::max(a.probes, 1) > b.sum / std::max(b.probes, 1);
            });
            if (c.round < kRounds) {
                c.tried.clear();
                for (const auto& n : c.cand) c.tried.push_back(n.res.handle);
                if (c.cand.size() > 3) c.cand.resize(3);
            }
        }
        if (c.round >= kRounds) { settle(gd, frame); return DepthChoice::kChoosing; }
        gd->override_depth_stencil = c.cand[c.at].res;
        c.copied = false; c.wait = kSettleFrames;
        return DepthChoice::kChoosing;
    }
    }
    return DepthChoice::kSettled;
}

bool depth_orientation_ready(effect_runtime* runtime, command_list* cmd_list, resource_view probe, effect_uniform_variable flip_variable) {
    Orientation& o = g_orient;
    g_queue = runtime->get_command_queue();
    if (!probe.handle || !flip_variable.handle) return true;  // (an older XPAR.fx: ReShade's setting as it is)
    runtime->set_uniform_value_bool(flip_variable, o.flip);
    if (--o.wait > 0) return o.decided;
    device* dev = runtime->get_device();
    if (!o.copied) {
        if (!copy_probe(o.reader, dev, cmd_list, probe)) { o.wait = 600; return true; }
        o.copied = true; o.wait = kReadbackFrames;
        return o.decided;
    }
    if (!probe_ready(o.reader)) { o.wait = 1; return o.decided; }
    o.copied = false;
    // (first quickly, then once in a while: a game can change it between its menu and the game)
    o.wait = o.decided ? 120 : 2;
    std::vector<float> px;
    std::uint32_t w = 0, h = 0;
    if (!read_probe(o.reader, px, w, h) || !w || !h) return o.decided;
    std::vector<float> depth;
    depth.reserve(std::size_t(w) * h);
    for (std::size_t i = 0; i < std::size_t(w) * h; ++i)
        if (std::isfinite(px[i * 2])) depth.push_back(std::clamp(px[i * 2], 0.0f, 1.0f));
    if (depth.size() < 64) return o.decided;
    const auto [lo, hi] = std::minmax_element(depth.begin(), depth.end());
    if (*hi - *lo < 1e-4f) return o.decided;  // (one value everywhere - a menu, a cleared buffer - says nothing)
    std::nth_element(depth.begin(), depth.begin() + depth.size() / 2, depth.end());
    const float median = depth[depth.size() / 2];
    ++o.checks;
    if (median > 0.5f) { ++o.wrong; o.right = 0; } else { ++o.right; o.wrong = 0; }
    if (o.wrong >= 3) {
        o.flip = !o.flip;
        o.wrong = o.right = 0;
        o.decided = false; o.wait = 3;  // (the next readings see it turned)
        runtime->set_uniform_value_bool(flip_variable, o.flip);
        char text[200];
        std::snprintf(text, sizeof(text), "XPAR: the depth comes the other way round (half of it above %.2f): turned%s", median,
                      o.flip ? " (ReShade's 'reversed' setting does not match the game)" : "");
        say(text);
    } else if (o.right >= 3 && !o.decided) {
        o.decided = true;
        char text[160];
        std::snprintf(text, sizeof(text), "XPAR: the depth is the right way round%s (half of it below %.2f)", o.flip ? " once turned" : "", median);
        say(text);
    }
    return o.decided;
}

bool depth_restart_needed() { return c.restart_needed; }

bool depth_usable() { return c.disabled || c.users || c.chosen.handle != 0; }
bool scene_depth_size(std::uint32_t& w, std::uint32_t& h) {
    if (!c.chosen.handle || c.chosen.handle != g_scene_of || !g_scene_w || !g_scene_h) return false;
    w = g_scene_w; h = g_scene_h;
    return true;
}

void track_depth_buffers() {
    g_track = true;
    const std::uint64_t frame = ++g_frame;
    if (frame >= g_rate_frame + 60) {
        const std::uint64_t binds = g_binds.load(std::memory_order_relaxed);
        g_rate = g_rate_frame ? double(binds - g_rate_binds) / double(frame - g_rate_frame) : 0.0;
        g_rate_frame = frame; g_rate_binds = binds;
    }
}

double depth_binds_per_frame() { return g_track ? g_rate : 0.0; }

void choose_depth_again(effect_runtime* runtime) {
    if (c.disabled || c.users || !c.ours || c.phase != Chooser::kSettled) return;
    say("XPAR: the 3D scene is on: the depth buffer chosen again (the choice before it was made while the scene was not shown yet)");
    if (auto* gd = runtime ? runtime->get_private_data<generic_depth_data>() : nullptr) gd->override_depth_stencil = {0};
    c.chosen = {0}; c.ours = false; c.verifying = false; c.retry_after = 120; c.phase = Chooser::kIdle;
}

void release_depth_choice(effect_runtime* runtime) {
    if (!g_track.exchange(false)) return;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_seen.clear();
    }
    drop_twin(runtime);
    if (runtime && !c.disabled && (c.ours || c.phase == Chooser::kProbe))
        if (auto* gd = runtime->get_private_data<generic_depth_data>()) gd->override_depth_stencil = {0};
    c.phase = Chooser::kIdle; c.chosen = {0}; c.ours = false; c.users = false; c.cand.clear(); c.tried.clear();
    c.retry_after = 120; c.said_candidates = ~std::size_t(0); c.verifying = false;
}

}  // namespace fw
