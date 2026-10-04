#include "addon/depth_choice.hpp"
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

void note(command_list* cmd_list, resource_view dsv) {
    if (!g_track.load(std::memory_order_relaxed) || !dsv.handle || !cmd_list) return;
    device* dev = cmd_list->get_device();
    if (!dev || (dev->get_api() != device_api::d3d12 && dev->get_api() != device_api::d3d11)) return;
    const resource res = dev->get_resource_from_view(dsv);
    if (!res.handle) return;
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
void on_destroy(device*, resource res) {
    if (!g_track.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_seen.erase(res.handle);
}

constexpr int kSettleFrames = 4;    // after an override: Generic Depth switches, copies, the effects render
constexpr int kReadbackFrames = 5;  // after the probe's copy: the GPU has done it
constexpr int kRounds = 2;
constexpr double kMinScore = 0.5;   // brightness changes 1.5x stronger across the depth edges than elsewhere, at least

// Reads XPAR.fx's probe back to the CPU (a few frames after the copy).
struct ProbeReader {
    ComPtr<ID3D12Resource> readback;
    ComPtr<ID3D11Texture2D> staging;  // (Direct3D 11)
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    std::uint64_t readback_bytes = 0;
    void reset() { readback.Reset(); staging.Reset(); }
};

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

void say(const char* text) { reshade::log::message(reshade::log::level::info, text); }

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
    char text[240];
    if (best && best->sum / best->probes >= kMinScore) {
        gd->override_depth_stencil = best->res;
        c.chosen = best->res; c.ours = true; c.retry_after = 120;
        std::snprintf(text, sizeof(text), "XPAR: depth buffer chosen: %ux%u, format %u (score %.2f, of %zu tried)", best->w, best->h,
                      unsigned(best->fmt), best->sum / best->probes, std::max(c.tried.size(), c.cand.size()));
    } else {
        // Nothing stands out (a menu, a loading screen, a dark or blurred picture): ReShade's own choice for now.
        gd->override_depth_stencil = {0};
        c.chosen = {0}; c.ours = false;
        // (asked again later, each time after twice as long: every question is a second without reprojection)
        c.retry_at = frame + c.retry_after;
        c.retry_after = std::min<std::uint64_t>(c.retry_after * 2, 3600);
        std::snprintf(text, sizeof(text), "XPAR: no depth buffer stands out among %zu (best score %.2f): ReShade's own choice for now", std::max(c.tried.size(), c.cand.size()),
                      best ? best->sum / best->probes : 0.0);
    }
    say(text);
    if (c.tried.empty()) for (const auto& k : c.cand) c.tried.push_back(k.res.handle);
    c.phase = Chooser::kSettled;
}

}  // namespace

void register_depth_choice_events() {
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy);
}
void unregister_depth_choice_events() {
    reshade::unregister_event<reshade::addon_event::destroy_resource>(on_destroy);
    reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    c.reader.reset();
    g_orient.reader.reset();
}

DepthChoice choose_depth_buffer(effect_runtime* runtime, command_list* cmd_list, resource_view probe, resource_view bound_depth) {
    device* dev = runtime->get_device();
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
            bool fresh = false;
            for (const auto& k : in_use)
                if (std::find(c.tried.begin(), c.tried.end(), k.res.handle) == c.tried.end()) { fresh = true; c.tried.push_back(k.res.handle); }
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
            } else {
                std::vector<float> px;
                std::uint32_t w = 0, h = 0;
                const DepthProbeScore s = read_probe(c.reader, px, w, h) ? score_depth_probe(px.data(), w, h) : DepthProbeScore{};
                char text[200];
                if (s.score >= kMinScore) {
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
        std::vector<float> px;
        std::uint32_t w = 0, h = 0;
        Candidate& k = c.cand[c.at];
        if (read_probe(c.reader, px, w, h)) {
            const DepthProbeScore s = score_depth_probe(px.data(), w, h);
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
    if (!probe.handle || !flip_variable.handle) return true;  // (an older XPAR.fx: ReShade's setting as it is)
    runtime->set_uniform_value_bool(flip_variable, o.flip);
    if (--o.wait > 0) return o.decided;
    device* dev = runtime->get_device();
    if (!o.copied) {
        if (!copy_probe(o.reader, dev, cmd_list, probe)) { o.wait = 600; return true; }
        o.copied = true; o.wait = kReadbackFrames;
        return o.decided;
    }
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

void track_depth_buffers() {
    g_track = true;
    ++g_frame;
}

void release_depth_choice(effect_runtime* runtime) {
    if (!g_track.exchange(false)) return;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_seen.clear();
    }
    if (runtime && !c.disabled && (c.ours || c.phase == Chooser::kProbe))
        if (auto* gd = runtime->get_private_data<generic_depth_data>()) gd->override_depth_stencil = {0};
    c.phase = Chooser::kIdle; c.chosen = {0}; c.ours = false; c.users = false; c.cand.clear(); c.tried.clear();
    c.retry_after = 120; c.said_candidates = ~std::size_t(0); c.verifying = false;
}

}  // namespace fw
