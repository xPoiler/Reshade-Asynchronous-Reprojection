#include "addon/ui_layer.hpp"
#include <reshade.hpp>
#include <atomic>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fw::ui_layer {
namespace {
using namespace reshade::api;

std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> g_take, g_take_hudless;
std::mutex g_mutex;
std::vector<std::uint64_t> g_back_buffers;
std::uint32_t g_width = 0, g_height = 0;
// This frame (since the last present): the layers cleared to transparent, and when each was last drawn into (0: not).
std::unordered_map<std::uint64_t, std::uint64_t> g_cleared;  // resource -> draw order
std::unordered_set<std::uint64_t> g_with_depth;               // ...of them, drawn with a depth buffer bound since cleared
std::uint64_t g_order = 0;
bool g_taken = false;
// The layers' states as changed on each command list (native list -> resource -> state after its last change there).
std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, std::uint32_t>> g_states;
// ...and on whichever list it was changed last this frame (a game may make it readable at the end of the list that draws
// it rather than on the one that blends it on).
std::unordered_map<std::uint64_t, std::uint32_t> g_last_state;
// Second pattern: the final pictures (every texture copied onto a back buffer lately - a game may take turns between
// two), each with whether a picture was copied into it since its last HUD-less copy, and its state as changed on each
// list / last anywhere.
struct Final {
    std::uint32_t seen = 0;  // present count when last copied onto a back buffer
    bool armed = false, known = false;
    std::uint32_t last = 0;
    std::unordered_map<std::uint64_t, std::uint32_t> states;  // native list -> state
};
std::unordered_map<std::uint64_t, Final> g_finals;
std::uint32_t g_serial = 0;
std::uint32_t g_logged_finals = 0;
// (for the log, every 600 presents: copies onto a back buffer, pictures copied into a final picture, render-target binds
// of one, of them without a picture copied in first, copies taken, binds skipped for an unknown state)
std::uint32_t g_presents = 0, g_copies_out = 0, g_copies_in = 0, g_final_binds = 0, g_unarmed = 0, g_hudless_taken_n = 0, g_hudless_unknown = 0;
// Back-buffer-sized textures bound as render targets, with the present count then (a copy onto the back buffer from one
// of them is a picture the game drew into - the second pattern's final picture - not the end of a first-pattern frame).
std::unordered_map<std::uint64_t, std::uint32_t> g_rt_bound;
// For the log: what was taken, and how its state was known (on the final list / elsewhere this frame / not at all).
std::uint64_t g_logged = 0;
std::uint32_t g_frames = 0, g_same_list = 0, g_elsewhere = 0, g_skipped = 0, g_not_read = 0, g_at_copy = 0, g_with_depth_n = 0;

bool is_back_buffer(std::uint64_t r) {
    for (const std::uint64_t b : g_back_buffers) if (b == r) return true;
    return false;
}
// D3D12 resource states from ReShade's (the same bits; its special states are D3D12's common/present, 0).
std::uint32_t d3d12_state(resource_usage u) {
    const auto v = static_cast<std::uint32_t>(u);
    return (v & 0x80000000u) ? 0u : v;
}

bool on_clear(command_list* cmd, resource_view rtv, const float color[4], uint32_t, const rect*) {
    if (!cmd || !rtv.handle || !color || color[0] != 0 || color[1] != 0 || color[2] != 0 || color[3] != 0) return false;
    device* dev = cmd->get_device();
    if (!dev || dev->get_api() != device_api::d3d12) return false;
    const resource r = dev->get_resource_from_view(rtv);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!r.handle || !g_width || is_back_buffer(r.handle)) return false;
    const resource_desc d = dev->get_resource_desc(r);
    if (d.texture.width != g_width || d.texture.height != g_height || d.texture.samples > 1) return false;
    g_cleared[r.handle] = 0;
    g_with_depth.erase(r.handle);
    return false;
}
void on_barrier(command_list* cmd, uint32_t count, const resource* resources, const resource_usage*, const resource_usage* new_states) {
    if (!cmd || !count) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_cleared.empty() && g_finals.empty()) return;
    for (uint32_t i = 0; i < count; ++i) {
        if (g_cleared.count(resources[i].handle)) {
            g_states[cmd->get_native()][resources[i].handle] = d3d12_state(new_states[i]);
            g_last_state[resources[i].handle] = d3d12_state(new_states[i]);
        }
        if (const auto f = g_finals.find(resources[i].handle); f != g_finals.end()) {
            f->second.states[cmd->get_native()] = d3d12_state(new_states[i]);
            f->second.last = d3d12_state(new_states[i]);
            f->second.known = true;
        }
    }
}
// The end of a frame (first pattern): of the layers drawn into this frame, the last one the game can read now (a pixel or
// compute shader resource - its state as changed on this very list, else as changed last this frame). One still a render
// target is not what the frame's end blends on (Requiem clears and draws a second such texture in some scenes: taken as
// the HUD layer, the HUD that is not in it warped). Layers drawn with depth only in a frame with no layer drawn without.
// (under the lock; true: layer/state set)
bool pick(device* dev, command_list* cmd, bool at_copy, std::uint64_t& layer, std::uint32_t& state) {
    g_taken = true;
    ++g_frames;
    const auto states = g_states.find(cmd->get_native());
    bool any_plain = false;
    for (const auto& [res, order] : g_cleared)
        if (order && !g_with_depth.count(res)) any_plain = true;
    std::uint64_t best_order = 0;
    bool any_state = false, same_list = false;
    for (const auto& [res, order] : g_cleared) {
        if (!order || (any_plain && g_with_depth.count(res))) continue;
        std::uint32_t st = 0;
        bool here = false;
        if (states != g_states.end() && states->second.count(res)) { st = states->second.at(res); here = true; }
        else if (const auto last = g_last_state.find(res); last != g_last_state.end()) st = last->second;
        else continue;
        any_state = true;
        // (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE)
        if (!(st & 0xC0u) || order <= best_order) continue;
        best_order = order; layer = res; state = st; same_list = here;
    }
    if (!layer) { ++(any_state ? g_not_read : g_skipped); return false; }
    ++(same_list ? g_same_list : g_elsewhere);
    if (at_copy) ++g_at_copy;
    const bool with_depth = g_with_depth.count(layer) != 0;
    if (with_depth) ++g_with_depth_n;
    if (layer != g_logged) {
        g_logged = layer;
        char text[224];
        std::snprintf(text, sizeof(text), "XPAR: the game's HUD layer found (%ux%u, format %u, state 0x%x%s, at %s) - taken as the frame's UI layer",
                      g_width, g_height, static_cast<unsigned>(dev->get_resource_desc(resource{layer}).texture.format), state,
                      with_depth ? ", drawn with depth" : "", at_copy ? "a copy onto the back buffer" : "the final draw");
        reshade::log::message(reshade::log::level::info, text);
    }
    return true;
}
// Copies onto the back buffer: their source is a final picture (second pattern) - or, from a texture not drawn into as
// a render target, the end of a first-pattern frame; copies into a final picture give it its frame's picture.
void copied(command_list* cmd, resource src, resource dst) {
    if (!cmd || !src.handle || !dst.handle) return;
    device* dev = cmd->get_device();
    if (!dev || dev->get_api() != device_api::d3d12) return;
    std::uint64_t layer = 0;
    std::uint32_t state = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (is_back_buffer(dst.handle)) {
            ++g_copies_out;
            const resource_desc d = dev->get_resource_desc(src);
            if (d.texture.width == g_width && d.texture.height == g_height && d.texture.samples <= 1) {
                g_finals[src.handle].seen = g_serial;
                const auto bound = g_rt_bound.find(src.handle);
                const bool drawn = bound != g_rt_bound.end() && g_serial - bound->second <= 1;
                if (!drawn && g_order && !g_taken && !pick(dev, cmd, true, layer, state)) return;
            }
        } else if (const auto f = g_finals.find(dst.handle); f != g_finals.end()) {
            f->second.armed = true;
            ++g_copies_in;
        }
    }
    if (layer && g_take) g_take(layer, state, cmd->get_native());
}
bool on_copy_resource(command_list* cmd, resource src, resource dst) { copied(cmd, src, dst); return false; }
bool on_copy_region(command_list* cmd, resource src, uint32_t, const subresource_box*, resource dst, uint32_t, const subresource_box*, filter_mode) {
    copied(cmd, src, dst);
    return false;
}
void target(command_list* cmd, resource_view first, bool depth) {
    if (!cmd || !first.handle) return;
    device* dev = cmd->get_device();
    if (!dev || dev->get_api() != device_api::d3d12) return;
    const resource r = dev->get_resource_from_view(first);
    std::uint64_t layer = 0, list = 0;
    std::uint32_t state = 0;
    // Second pattern: the final picture bound as a render target after a picture was copied into it: the HUD's turn.
    {
        std::uint64_t hudless = 0;
        std::uint32_t hstate = 0;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            const auto f = depth ? g_finals.end() : g_finals.find(r.handle);
            if (f != g_finals.end()) {
                Final& fin = f->second;
                ++g_final_binds;
                if (!fin.armed) {
                    ++g_unarmed;
                } else {
                    fin.armed = false;
                    const auto it = fin.states.find(cmd->get_native());
                    if (it != fin.states.end()) hstate = it->second;
                    else if (fin.known) hstate = fin.last;
                    else { ++g_hudless_unknown; return; }
                    ++g_hudless_taken_n;
                    hudless = r.handle;
                    if (g_logged_finals < 4) {  // (each final picture's first copy, a few)
                        ++g_logged_finals;
                        char text[160];
                        std::snprintf(text, sizeof(text), "XPAR: the game draws its HUD onto its final picture (%ux%u, format %u, state 0x%x) - a HUD-less copy taken before",
                                      g_width, g_height, static_cast<unsigned>(dev->get_resource_desc(r).texture.format), hstate);
                        reshade::log::message(reshade::log::level::info, text);
                    }
                }
            }
        }
        if (hudless) {
            if (g_take_hudless) g_take_hudless(hudless, hstate, cmd->get_native());
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_width && !is_back_buffer(r.handle)) {
            const resource_desc d = dev->get_resource_desc(r);
            if (d.texture.width == g_width && d.texture.height == g_height) g_rt_bound[r.handle] = g_serial;
        }
        if (const auto it = g_cleared.find(r.handle); it != g_cleared.end()) {
            it->second = ++g_order;
            if (depth) g_with_depth.insert(r.handle);
            return;
        }
        // The final draw (a back buffer bound as the render target).
        if (!is_back_buffer(r.handle) || !g_order || g_taken || !pick(dev, cmd, false, layer, state)) return;
        list = cmd->get_native();
    }
    if (g_take) g_take(layer, state, list);
}
void on_bind(command_list* cmd, uint32_t n, const resource_view* rtvs, resource_view dsv) { if (n && rtvs) target(cmd, rtvs[0], dsv.handle != 0); }
bool on_begin_pass(command_list* cmd, uint32_t n, const render_pass_render_target_desc* rts, const render_pass_depth_stencil_desc* ds, render_pass_flags) {
    if (n && rts) target(cmd, rts[0].view, ds && ds->view.handle);
    return false;
}
void on_reset(command_list* cmd) {
    if (!cmd) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_states.erase(cmd->get_native());
    for (auto& [res, fin] : g_finals) fin.states.erase(cmd->get_native());
}
}  // namespace

void init(std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> take,
          std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> take_hudless) {
    g_take = std::move(take);
    g_take_hudless = std::move(take_hudless);
    reshade::register_event<reshade::addon_event::copy_resource>(on_copy_resource);
    reshade::register_event<reshade::addon_event::copy_texture_region>(on_copy_region);
    reshade::register_event<reshade::addon_event::clear_render_target_view>(on_clear);
    reshade::register_event<reshade::addon_event::barrier>(on_barrier);
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::register_event<reshade::addon_event::reset_command_list>(on_reset);
}
void shutdown() {
    reshade::unregister_event<reshade::addon_event::reset_command_list>(on_reset);
    reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::unregister_event<reshade::addon_event::barrier>(on_barrier);
    reshade::unregister_event<reshade::addon_event::clear_render_target_view>(on_clear);
    reshade::unregister_event<reshade::addon_event::copy_texture_region>(on_copy_region);
    reshade::unregister_event<reshade::addon_event::copy_resource>(on_copy_resource);
    g_take = nullptr;
    g_take_hudless = nullptr;
}

void on_present(swapchain* chain) {
    if (!chain) return;
    device* dev = chain->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_back_buffers.clear();
    for (uint32_t i = 0; i < chain->get_back_buffer_count(); ++i) g_back_buffers.push_back(chain->get_back_buffer(i).handle);
    if (dev && !g_back_buffers.empty()) {
        const resource_desc d = dev->get_resource_desc(resource{g_back_buffers[0]});
        g_width = d.texture.width; g_height = d.texture.height;
    }
    // (every 600 frames, how often the layer could not be taken)
    if (g_frames >= 600) {
        char text[192];
        std::snprintf(text, sizeof(text), "XPAR: HUD layer over %u frames: state from the final list %u, from earlier this frame %u, none readable %u, "
                      "not taken %u; at a copy onto the back buffer %u, drawn with depth %u",
                      g_frames, g_same_list, g_elsewhere, g_not_read, g_skipped, g_at_copy, g_with_depth_n);
        reshade::log::message(reshade::log::level::info, text);
        g_frames = g_same_list = g_elsewhere = g_not_read = g_skipped = g_at_copy = g_with_depth_n = 0;
    }
    // (final pictures not copied onto a back buffer for 300 presents are forgotten)
    ++g_serial;
    for (auto it = g_finals.begin(); it != g_finals.end();)
        it = g_serial - it->second.seen > 300 ? g_finals.erase(it) : std::next(it);
    for (auto it = g_rt_bound.begin(); it != g_rt_bound.end();)
        it = g_serial - it->second > 2 ? g_rt_bound.erase(it) : std::next(it);
    if (++g_presents >= 600) {
        if (g_copies_out) {
            char text[256];
            std::snprintf(text, sizeof(text), "XPAR: final pictures over %u presents: %zu known, copied onto the back buffer %u, pictures copied into them %u, "
                          "drawn into %u times (%u without a picture copied in first), HUD-less copies taken %u, skipped for an unknown state %u",
                          g_presents, g_finals.size(), g_copies_out, g_copies_in, g_final_binds, g_unarmed, g_hudless_taken_n, g_hudless_unknown);
            reshade::log::message(reshade::log::level::info, text);
        }
        g_presents = g_copies_out = g_copies_in = g_final_binds = g_unarmed = g_hudless_taken_n = g_hudless_unknown = 0;
    }
    g_cleared.clear();
    g_with_depth.clear();
    g_last_state.clear();
    g_order = 0;
    g_taken = false;
}
}  // namespace fw::ui_layer
