#include "addon/depth_choice.hpp"
#include <reshade.hpp>
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
    if (!dev || dev->get_api() != device_api::d3d12) return;
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
    ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    std::uint64_t readback_bytes = 0;
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

bool copy_probe(device* dev, command_list* cmd_list, resource_view probe) {
    auto* tex = reinterpret_cast<ID3D12Resource*>(dev->get_resource_from_view(probe).handle);
    auto* d3d = reinterpret_cast<ID3D12Device*>(dev->get_native());
    auto* list = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native());
    if (!tex || !d3d || !list) return false;
    const D3D12_RESOURCE_DESC desc = tex->GetDesc();
    if (desc.Format != DXGI_FORMAT_R32G32_FLOAT) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    d3d->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, &total);
    if (!c.readback || c.readback_bytes != total) {
        c.readback.Reset();
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&c.readback))))
            return false;
        c.readback_bytes = total;
    }
    c.footprint = fp;
    // ReShade leaves an effect's render targets readable by shaders between passes.
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = tex; b.Transition.Subresource = 0;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dst{c.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src{tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    list->ResourceBarrier(1, &b);
    return true;
}

bool read_probe(std::vector<float>& px, std::uint32_t& w, std::uint32_t& h) {
    if (!c.readback) return false;
    w = c.footprint.Footprint.Width; h = c.footprint.Footprint.Height;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(c.readback_bytes)};
    std::uint8_t* mapped = nullptr;
    if (FAILED(c.readback->Map(0, &range, reinterpret_cast<void**>(&mapped))) || !mapped) return false;
    px.resize(std::size_t(w) * h * 2);
    for (std::uint32_t y = 0; y < h; ++y)
        std::memcpy(&px[std::size_t(y) * w * 2], mapped + c.footprint.Offset + std::size_t(y) * c.footprint.Footprint.RowPitch, std::size_t(w) * 8);
    const D3D12_RANGE none{0, 0};
    c.readback->Unmap(0, &none);
    return true;
}

void settle(generic_depth_data* gd, std::uint64_t frame) {
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
    c.readback.Reset();
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
                if (probe.handle && copy_probe(dev, cmd_list, probe)) { c.verify_copied = true; c.verify_wait = kReadbackFrames; }
                else c.verifying = false;
            } else {
                std::vector<float> px;
                std::uint32_t w = 0, h = 0;
                const DepthProbeScore s = read_probe(px, w, h) ? score_depth_probe(px.data(), w, h) : DepthProbeScore{};
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
        if (c.cand.size() < 2 || !probe.handle) {
            // A single one or none (nothing to choose), or an XPAR.fx without the probe: ReShade's own choice
            // until another comes into use.
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
        gd->override_depth_stencil = c.cand[0].res;
        c.phase = Chooser::kProbe;
        return DepthChoice::kChoosing;
    }
    case Chooser::kProbe: {
        if (--c.wait > 0) return DepthChoice::kChoosing;
        if (!c.copied) {
            if (!probe.handle || !copy_probe(dev, cmd_list, probe)) {
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
        if (read_probe(px, w, h)) {
            const DepthProbeScore s = score_depth_probe(px.data(), w, h);
            k.sum += s.score; ++k.probes;
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
