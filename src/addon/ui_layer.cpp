#include "addon/ui_layer.hpp"
#include <reshade.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>
#include <d3d11.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fw::ui_layer {
namespace {
using namespace reshade::api;

std::function<void(std::uint64_t, std::uint32_t, std::uint64_t, bool)> g_take;  // (bool: alpha inverted - see cleared_to_transparent)
std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> g_take_hudless;
std::function<void(std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t)> g_take_vk;
std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> g_take_before;
std::function<void(std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t)> g_take_hudless_vk;
// Fifth pattern (Vulkan): the back buffer a compute pass wrote this frame (0: none, or its copy taken), whether a HUD layer
// was taken this frame (then that is the HUD, as in DOOM Eternal - no HUD-less copy), and for the log.
std::uint64_t g_vk_finished = 0;
bool g_vk_layer_frame = false, g_vk_hudless_logged = false;
std::uint32_t g_vk_hudless_n = 0;
std::mutex g_mutex;
device_api g_api = device_api::d3d12;  // the game's (its swapchain's device)
bool g_triggered = false;              // a frame's end was seen since the last present
// Third pattern (Direct3D 11): the render target bound on each context, with or without depth; the blending of each
// pipeline (render target 0: 0 overwrite, 1 alpha blend, 2 other); the pictures overwritten since their last HUD copy.
struct Bound { std::uint64_t res = 0; bool depth = false; };
std::unordered_map<std::uint64_t, Bound> g_rt;
std::unordered_map<std::uint64_t, std::uint8_t> g_blend, g_bound_blend;
std::unordered_set<std::uint64_t> g_overwritten;
std::uint64_t g_logged_before = 0;
std::uint64_t g_hud_picture = 0;  // the picture the HUD was last seen drawn into (third pattern)
// This frame: the picture a copy was taken from (0: none), and pictures given up on (overwritten after their copy).
std::uint64_t g_took = 0;
std::unordered_set<std::uint64_t> g_given_up;
std::vector<std::uint64_t> g_back_buffers;
// Fourth pattern (Direct3D 12; Hi-Fi Rush, Unreal's Slate): the picture drawn onto the back buffer by one overwriting draw,
// then the HUD alpha-blended straight onto it. Each back buffer's step this frame (1 overwritten, no HUD yet; 2 HUD-less
// copy taken; 3 given up - overwritten after its copy), whether the pattern was seen (then a frame that leaves the back
// buffer with no HUD drawn hands it over as it is: no HUD), and what the recording thread bound last (draws are many:
// checked without the lock - a command list is recorded on one thread at a time).
std::unordered_map<std::uint64_t, std::uint8_t> g_bb_step;
std::unordered_map<std::uint64_t, std::uint64_t> g_bb_picture_pipeline;  // (the pipeline that drew the picture onto it)
bool g_bb_seen = false;
thread_local std::uint64_t tl_list = 0, tl_back_buffer = 0, tl_pipe_list = 0, tl_pipeline = 0;
thread_local bool tl_depth = false;
thread_local std::uint64_t tl_inverted = 0;  // (Direct3D 12: the render target bound on tl_list, if a candidate cleared to opaque black)
// (the list a render pass is open on, Direct3D 12: no copy may be recorded inside one - see drawn_back_buffer)
thread_local std::uint64_t tl_pass_list = 0;
std::uint32_t g_width = 0, g_height = 0;
// This frame (since the last present): the layers cleared to transparent, and when each was last drawn into (0: not).
std::unordered_map<std::uint64_t, std::uint64_t> g_cleared;  // resource -> draw order
std::unordered_set<std::uint64_t> g_inverted;  // ...of them, cleared to opaque black: alpha 1 is nothing (see cleared_to_transparent)
// ...their draws since cleared: alpha-blended, all with known blending (a HUD is blended in; a picture drawn over is not)
std::unordered_map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>> g_inverted_draws;
// (kept over frames - the game records its command lists on several threads, some a frame ahead: the clear of Armored Core
// VI's HUD layer was often recorded after the draws into it, or in another frame, and the draws counted so far were thrown
// away with it - the layer never reached the 4 draws pick needs, nor its wait. Known once cleared to opaque black, until
// cleared to transparent; its draws counted over frames, halved past 4096)
std::unordered_set<std::uint64_t> g_inverted_seen;
// (textures ever bound together with other render targets: a G-buffer's, never a HUD layer - see target_set)
std::unordered_set<std::uint64_t> g_multi;
std::unordered_map<std::uint64_t, std::uint32_t> g_inverted_state;  // (their last state, over frames - a barrier recorded before the bind)
std::uint32_t g_inv_turned_down = 0, g_inv_alpha = 0, g_inv_draws = 0;  // (for the log every 600 frames)
std::unordered_set<std::uint64_t> g_with_depth;               // ...of them, drawn with a depth buffer bound since cleared
std::uint64_t g_order = 0;
bool g_taken = false;
// Vulkan: the frame's end seen while no layer was readable yet (DOOM Eternal may make it readable just after): taken when
// one becomes readable; dropped when the next frame's layers start, or after two presents.
bool g_end_pending = false;
std::uint32_t g_pending_presents = 0, g_waited = 0;
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

void end_of_frame(device* dev, command_list* cmd, const char* at);
const char kComputeEnd[] = "a compute pass writing the back buffer";
bool is_back_buffer(std::uint64_t r) {
    for (const std::uint64_t b : g_back_buffers) if (b == r) return true;
    return false;
}
// D3D12 resource states from ReShade's (the same bits; its special states are D3D12's common/present, 0).
std::uint32_t d3d12_state(resource_usage u) {
    const auto v = static_cast<std::uint32_t>(u);
    return (v & 0x80000000u) ? 0u : v;
}

// The first pattern's bookkeeping runs in Direct3D 12 and Vulkan (the add-on can hand a layer over in both).
bool layer_api(device* dev) { return dev && (dev->get_api() == device_api::d3d12 || dev->get_api() == device_api::vulkan); }
// A texture the size of the back buffer cleared to transparent: a candidate layer (a clear command, or a render pass that
// starts by clearing it - Vulkan's way, DOOM Eternal). Or cleared to opaque black (inverted, Direct3D 12): a layer whose alpha
// says how much of the picture shows through - Armored Core VI clears its HUD layer to (0, 0, 0, 1) and blends its HUD in;
// its coverage is 1 - alpha (Producer::on_ui_layer). Only textures cleared to transparent counted: the HUD layer was never a
// candidate, the game's own targets of the screen's size cleared to transparent were taken instead.
bool coverage_format(format f) {
    switch (format_to_typeless(f)) {
        case format::r8g8b8a8_typeless: case format::b8g8r8a8_typeless: case format::r16g16b16a16_typeless: case format::r32g32b32a32_typeless:
            return true;
        default:
            return false;
    }
}
// (several render targets bound at once - a G-buffer, Armored Core VI's: none of them is a HUD layer, which is drawn into on
// its own. Remembered for good)
void target_set(command_list* cmd, uint32_t n, const resource_view* views) {
    if (!cmd || n < 2 || !views) return;
    device* dev = cmd->get_device();
    if (!layer_api(dev)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    for (uint32_t i = 0; i < n; ++i) {
        if (!views[i].handle) continue;
        const resource r = dev->get_resource_from_view(views[i]);
        if (!r.handle || is_back_buffer(r.handle)) continue;
        g_multi.insert(r.handle);
        g_cleared.erase(r.handle); g_inverted.erase(r.handle); g_inverted_seen.erase(r.handle);
    }
}
void cleared_to_transparent(device* dev, resource_view rtv, bool inverted = false) {
    const resource r = dev->get_resource_from_view(rtv);
    if (inverted && dev->get_api() != device_api::d3d12) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!r.handle || !g_width || is_back_buffer(r.handle)) return;
    const resource_desc d = dev->get_resource_desc(r);
    if (d.texture.width != g_width || d.texture.height != g_height || d.texture.samples > 1) return;
    // (a HUD layer carries its coverage: an alpha channel of 8 bits or more. Taken in its place before: Armored Core VI's
    // R11G11B10 and R10G10B10A2 3D targets, DOOM Eternal's R11G11B10, Cyberpunk's R32_UINT - the real layers are 8-bit or
    // 16-bit RGBA)
    if (!coverage_format(d.texture.format) || g_multi.count(r.handle)) return;
    if (g_end_pending) {  // (the next frame's layers start: the frame that waited is not taken)
        g_end_pending = false;
        ++g_frames; ++g_not_read;
        g_cleared.clear(); g_with_depth.clear(); g_inverted.clear(); g_last_state.clear(); g_order = 0; g_taken = false;
    }
    g_cleared[r.handle] = 0;
    g_with_depth.erase(r.handle);
    if (inverted) { g_inverted.insert(r.handle); g_inverted_seen.insert(r.handle); }
    else { g_inverted.erase(r.handle); g_inverted_seen.erase(r.handle); g_inverted_draws.erase(r.handle); }
}
bool on_clear(command_list* cmd, resource_view rtv, const float color[4], uint32_t, const rect*) {
    if (!cmd || !rtv.handle || !color || color[0] != 0 || color[1] != 0 || color[2] != 0 || (color[3] != 0 && color[3] != 1)) return false;
    device* dev = cmd->get_device();
    if (layer_api(dev)) cleared_to_transparent(dev, rtv, color[3] == 1);
    return false;
}
void on_barrier(command_list* cmd, uint32_t count, const resource* resources, const resource_usage*, const resource_usage* new_states) {
    if (!cmd || !count) return;
    // A back buffer made writable by compute shaders: the frame's end, written by a compute pass (DOOM Eternal composes
    // its picture and HUD layer into the back buffer that way: no draw onto it, no copy).
    device* dev = cmd->get_device();
    bool compute_end = false, made_readable = false;
    std::uint64_t vk_hudless = 0;
    std::uint64_t no_hud = 0;
    std::uint32_t no_hud_state = 0;
    const bool vulkan = dev && dev->get_api() == device_api::vulkan;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        // (unordered access in Direct3D 12; Vulkan games write it as a storage image in the general layout - DOOM
        // Eternal in game: ReShade's resource_usage::general, 0x80000000 alone)
        for (uint32_t i = 0; i < count; ++i) {
            const std::uint32_t st = static_cast<std::uint32_t>(new_states[i]);
            if ((((st & static_cast<std::uint32_t>(resource_usage::unordered_access)) && !(st & 0x80000000u)) ||
                 (vulkan && st == static_cast<std::uint32_t>(resource_usage::general))) && is_back_buffer(resources[i].handle)) {
                compute_end = true;
                if (vulkan) g_vk_finished = resources[i].handle;
            }
        }
        // (fifth pattern: that back buffer made a render target now - the barrier event comes after the game's barrier, so
        // it is in the colour attachment layout here, outside a render pass: copied in it, put back as it was)
        if (vulkan && g_vk_finished && !g_vk_layer_frame && g_take_hudless_vk)
            for (uint32_t i = 0; i < count; ++i)
                if (resources[i].handle == g_vk_finished &&
                    (static_cast<std::uint32_t>(new_states[i]) & static_cast<std::uint32_t>(resource_usage::render_target)) &&
                    !(static_cast<std::uint32_t>(new_states[i]) & 0x80000000u)) {
                    vk_hudless = g_vk_finished;
                    g_vk_finished = 0;
                    ++g_vk_hudless_n;
                }
        // (this call's own changes first: a game makes its HUD layer readable and the back buffer writable in one
        // barrier - the end below then sees the layer as it is after it)
        for (uint32_t i = 0; i < count && !(g_cleared.empty() && g_finals.empty() && g_inverted_seen.empty()); ++i) {
            if (g_inverted_seen.count(resources[i].handle)) g_inverted_state[resources[i].handle] = d3d12_state(new_states[i]);
            if (g_cleared.count(resources[i].handle)) {
                g_states[cmd->get_native()][resources[i].handle] = d3d12_state(new_states[i]);
                g_last_state[resources[i].handle] = d3d12_state(new_states[i]);
                if (g_cleared.at(resources[i].handle) && (d3d12_state(new_states[i]) & 0xC0u)) made_readable = true;
            }
            if (const auto f = g_finals.find(resources[i].handle); f != g_finals.end()) {
                f->second.states[cmd->get_native()] = d3d12_state(new_states[i]);
                f->second.last = d3d12_state(new_states[i]);
                f->second.known = true;
            }
        }
        made_readable = made_readable && g_end_pending;
        // (fourth pattern: a back buffer no longer a render target - its frame's drawing is over; overwritten with no HUD
        // after it: handed over as it is, no HUD this frame)
        if (!vulkan && !g_bb_step.empty())
            for (uint32_t i = 0; i < count; ++i) {
                const auto s = g_bb_step.find(resources[i].handle);
                if (s == g_bb_step.end() || (d3d12_state(new_states[i]) & 0x4u)) continue;
                if (s->second == 1 && g_bb_seen && !no_hud) { no_hud = resources[i].handle; no_hud_state = d3d12_state(new_states[i]); }
                g_bb_step.erase(s);
            }
    }
    if (no_hud && g_take_hudless) g_take_hudless(no_hud, no_hud_state, cmd->get_native());
    if (vk_hudless) {
        const resource_desc d = dev->get_resource_desc(resource{vk_hudless});
        if (!g_vk_hudless_logged) {
            g_vk_hudless_logged = true;
            char text[192];
            std::snprintf(text, sizeof(text), "XPAR: the game draws its HUD onto the back buffer after finishing it with compute shaders (%ux%u, format %u) - "
                          "a HUD-less copy taken before", d.texture.width, d.texture.height, static_cast<unsigned>(d.texture.format));
            reshade::log::message(reshade::log::level::info, text);
        }
        // (VkImageLayout: COLOR_ATTACHMENT_OPTIMAL)
        g_take_hudless_vk(vk_hudless, static_cast<std::uint32_t>(d.texture.format), d.texture.width, d.texture.height, 2, cmd->get_native());
    }
    if (!layer_api(dev)) return;
    if (compute_end) end_of_frame(dev, cmd, kComputeEnd);
    else if (made_readable) end_of_frame(dev, cmd, "the layer made readable after the frame's end");
}
// The end of a frame (first pattern): of the layers drawn into this frame, the last one the game can read now (a pixel or
// compute shader resource - its state as changed on this very list, else as changed last this frame). One still a render
// target is not what the frame's end blends on (Requiem clears and draws a second such texture in some scenes: taken as
// the HUD layer, the HUD that is not in it warped). Layers drawn with depth only in a frame with no layer drawn without.
// (under the lock; true: layer/state set)
// may_wait: none readable yet - nothing counted or taken, false returned with `waits` set (see g_end_pending).
bool pick(device* dev, command_list* cmd, const char* at, std::uint64_t& layer, std::uint32_t& state, bool may_wait, bool wait_inverted,
          bool& waits) {
    const bool at_copy = at[0] == 'a' && at[2] == 'c';  // ("a copy onto the back buffer")
    waits = false;
    const auto states = g_states.find(cmd->get_native());
    bool any_plain = false;
    for (const auto& [res, order] : g_cleared)
        if (order && !g_with_depth.count(res)) any_plain = true;
    std::uint64_t best_order = 0, unread_inverted = 0;
    bool any_state = false, same_list = false;
    for (const auto& [res, order] : g_cleared) {
        if (!order || (any_plain && g_with_depth.count(res))) continue;
        // (cleared to opaque black: a HUD layer only if what was drawn into it is mostly alpha-blended - at least 4 draws, 60%
        // of them. A picture of the screen's size cleared so and drawn over - a scene, its parts, a post-processing pass -
        // with alpha left below 1 here and there would otherwise be taken, and held there as HUD)
        if (g_inverted.count(res)) {
            const auto n = g_inverted_draws.find(res);
            if (n == g_inverted_draws.end() || n->second.second < 4 || n->second.first * 10 < n->second.second * 6) {
                ++g_inv_turned_down;
                g_inv_alpha = n == g_inverted_draws.end() ? 0 : n->second.first; g_inv_draws = n == g_inverted_draws.end() ? 0 : n->second.second;
                continue;
            }
        }
        std::uint32_t st = 0;
        bool here = false;
        if (states != g_states.end() && states->second.count(res)) { st = states->second.at(res); here = true; }
        else if (const auto last = g_last_state.find(res); last != g_last_state.end()) st = last->second;
        else if (const auto inv = g_inverted.count(res) ? g_inverted_state.find(res) : g_inverted_state.end(); inv != g_inverted_state.end()) st = inv->second;
        else continue;
        any_state = true;
        // (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE)
        if (!(st & 0xC0u) && g_inverted.count(res)) unread_inverted = std::max(unread_inverted, order);
        if (!(st & 0xC0u) || order <= best_order) continue;
        best_order = order; layer = res; state = st; same_list = here;
    }
    if (!layer && any_state && may_wait) { waits = true; return false; }
    // (a layer cleared to opaque black, drawn after every readable one, not readable yet: waited for - Armored Core VI makes
    // its HUD layer readable on the final list after the back buffer is bound there, and a texture of its 3D rendering read
    // earlier was taken in its place, nearly every frame. Only these: Requiem's second layer, cleared to transparent and still
    // a render target at the end, stays left out as before)
    if (wait_inverted && unread_inverted > best_order) { layer = 0; waits = true; return false; }
    g_taken = true;
    ++g_frames;
    if (!layer) { ++(any_state ? g_not_read : g_skipped); return false; }
    ++(same_list ? g_same_list : g_elsewhere);
    if (at_copy) ++g_at_copy;
    const bool with_depth = g_with_depth.count(layer) != 0;
    if (with_depth) ++g_with_depth_n;
    if (layer != g_logged) {
        g_logged = layer;
        char text[224];
        std::snprintf(text, sizeof(text), "XPAR: the game's HUD layer found (%ux%u, format %u, state 0x%x%s%s, at %s) - taken as the frame's UI layer",
                      g_width, g_height, static_cast<unsigned>(dev->get_resource_desc(resource{layer}).texture.format), state,
                      with_depth ? ", drawn with depth" : "", g_inverted.count(layer) ? ", cleared to opaque black: alpha inverted" : "", at);
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
    bool end = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (is_back_buffer(dst.handle)) {
            ++g_copies_out;
            const resource_desc d = dev->get_resource_desc(src);
            if (d.texture.width == g_width && d.texture.height == g_height && d.texture.samples <= 1) {
                g_finals[src.handle].seen = g_serial;
                const auto bound = g_rt_bound.find(src.handle);
                const bool drawn = bound != g_rt_bound.end() && g_serial - bound->second <= 1;
                end = !drawn && g_order && !g_taken;
            }
        } else if (const auto f = g_finals.find(dst.handle); f != g_finals.end()) {
            f->second.armed = true;
            ++g_copies_in;
        }
    }
    if (end) end_of_frame(dev, cmd, "a copy onto the back buffer");
}
bool on_copy_resource(command_list* cmd, resource src, resource dst) { copied(cmd, src, dst); return false; }
bool on_copy_region(command_list* cmd, resource src, uint32_t, const subresource_box*, resource dst, uint32_t, const subresource_box*, filter_mode) {
    copied(cmd, src, dst);
    return false;
}
void target(command_list* cmd, resource_view first, bool depth) {
    if (!cmd || !first.handle) return;
    device* dev = cmd->get_device();
    if (!dev) return;
    const resource r = dev->get_resource_from_view(first);
    if (dev->get_api() == device_api::d3d12) {  // (fourth pattern: a back buffer drawn into)
        tl_list = cmd->get_native();
        tl_depth = depth;
        std::lock_guard<std::mutex> lock(g_mutex);
        tl_back_buffer = is_back_buffer(r.handle) ? r.handle : 0;
        tl_inverted = g_inverted.count(r.handle) || g_inverted_seen.count(r.handle) ? r.handle : 0;
    }
    if (dev->get_api() == device_api::d3d11) {  // (third pattern: what each draw goes into)
        std::lock_guard<std::mutex> lock(g_mutex);
        g_rt[cmd->get_native()] = {r.handle, depth};
        return;
    }
    if (!layer_api(dev)) return;
    // Second pattern: the final picture bound as a render target after a picture was copied into it: the HUD's turn.
    if (dev->get_api() == device_api::d3d12) {
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
        if (g_api == device_api::d3d12 && !g_taken && g_inverted_seen.count(r.handle) && !g_cleared.count(r.handle)) {
            g_cleared[r.handle] = 0;  // (its clear recorded elsewhere - see g_inverted_seen)
            g_inverted.insert(r.handle);
        }
        if (const auto it = g_cleared.find(r.handle); it != g_cleared.end()) {
            it->second = ++g_order;
            if (depth) g_with_depth.insert(r.handle);
            return;
        }
        if (!is_back_buffer(r.handle) || !g_order || g_taken) return;
    }
    // The final draw (a back buffer bound as the render target).
    end_of_frame(dev, cmd, "the final draw");
}
// The frame's end (first pattern): the layer the game can read now handed over - on this command list - and the frame's
// layers start over. Driven by the game's own steps, not by presents: a game recording ahead (DOOM Eternal) records the
// next frame's layer before the present that ends this one (a present only starts over what no frame's end did).
void end_of_frame(device* dev, command_list* cmd, const char* at) {
    std::uint64_t layer = 0;
    std::uint32_t state = 0;
    bool inverted = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_order || g_taken) return;
        // (Vulkan, at a compute pass writing the back buffer: a layer not readable yet is waited for, once)
        const bool may_wait = !g_end_pending && dev->get_api() == device_api::vulkan && std::strcmp(at, kComputeEnd) == 0;
        const bool wait_inverted = !g_end_pending && dev->get_api() == device_api::d3d12;
        bool waits = false;
        const bool picked = pick(dev, cmd, at, layer, state, may_wait, wait_inverted, waits);
        inverted = picked && g_inverted.count(layer) != 0;
        g_triggered = true;
        if (waits) {
            g_end_pending = true;
            g_pending_presents = 0;
            ++g_waited;
            return;
        }
        g_end_pending = false;
        g_cleared.clear();
        g_with_depth.clear(); g_inverted.clear();
        g_last_state.clear();
        g_order = 0;
        g_taken = false;
        if (!picked) return;
    }
    if (dev->get_api() == device_api::vulkan) {
        if (!g_take_vk) return;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_vk_layer_frame = true;
        }
        const resource_desc d = dev->get_resource_desc(resource{layer});
        g_take_vk(layer, static_cast<std::uint32_t>(d.texture.format), d.texture.width, d.texture.height, state, cmd->get_native());
    } else if (g_take) {
        g_take(layer, state, cmd->get_native(), inverted);
    }
}
void on_bind(command_list* cmd, uint32_t n, const resource_view* rtvs, resource_view dsv) {
    if (cmd && tl_pass_list == cmd->get_native()) tl_pass_list = 0;
    target_set(cmd, n, rtvs);
    if (n && rtvs) target(cmd, rtvs[0], dsv.handle != 0);
}
// (what is kept over frames goes with the texture: its handle may be another texture's later)
void on_destroy_resource(device*, resource r) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_multi.erase(r.handle); g_inverted_seen.erase(r.handle); g_inverted_draws.erase(r.handle); g_inverted_state.erase(r.handle);
}
bool on_end_pass(command_list* cmd) { if (cmd && tl_pass_list == cmd->get_native()) tl_pass_list = 0; return false; }
bool on_begin_pass(command_list* cmd, uint32_t n, const render_pass_render_target_desc* rts, const render_pass_depth_stencil_desc* ds, render_pass_flags) {
    if (n && rts && rts[0].load_op == render_pass_load_op::clear && rts[0].view.handle && rts[0].clear_color[0] == 0 &&
        rts[0].clear_color[1] == 0 && rts[0].clear_color[2] == 0 && (rts[0].clear_color[3] == 0 || rts[0].clear_color[3] == 1) && cmd &&
        layer_api(cmd->get_device()))
        cleared_to_transparent(cmd->get_device(), rts[0].view, rts[0].clear_color[3] == 1);
    if (n > 1 && rts) {
        std::vector<resource_view> views(n);
        for (uint32_t i = 0; i < n; ++i) views[i] = rts[i].view;
        target_set(cmd, n, views.data());
    }
    if (n && rts) target(cmd, rts[0].view, ds && ds->view.handle);
    if (cmd) tl_pass_list = cmd->get_native();
    return false;
}
// Third pattern (Direct3D 11; Resident Evil 2): the HUD drawn into the game's picture after its post-processing, then one
// draw converts the picture onto the back buffer - no layer, no copy. In the game's picture (back buffer size, drawn
// without depth) the post-processing overwrites; the HUD blends over it. So the first alpha-blended draw without depth
// into a picture overwritten since: the HUD's turn - a copy of the picture then (the HUD-less picture, in the picture's
// own colours), and at the present the picture with its HUD compared with it: the HUD (see Producer::on_hud_before).
// The scene's own transparencies blend with depth: not taken.
std::uint8_t blend_kind(const blend_desc& b) {
    if (!b.blend_enable[0]) return 0;
    const blend_factor s = b.source_color_blend_factor[0], d = b.dest_color_blend_factor[0];
    return (s == blend_factor::source_alpha || s == blend_factor::one) && d == blend_factor::one_minus_source_alpha ? 1 : 2;
}
void on_init_pipeline(device* dev, pipeline_layout, uint32_t count, const pipeline_subobject* subobjects, pipeline p) {
    if (!p.handle || !dev || (dev->get_api() != device_api::d3d11 && dev->get_api() != device_api::d3d12)) return;
    for (uint32_t i = 0; i < count; ++i)
        if (subobjects[i].type == pipeline_subobject_type::blend_state && subobjects[i].data) {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_blend[p.handle] = blend_kind(*static_cast<const blend_desc*>(subobjects[i].data));
        }
}
void on_destroy_pipeline(device* dev, pipeline p) {
    if (!dev || (dev->get_api() != device_api::d3d11 && dev->get_api() != device_api::d3d12)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_blend.erase(p.handle);
}
void on_bind_pipeline(command_list* cmd, pipeline_stage stages, pipeline p) {
    if (g_api == device_api::d3d12 && cmd && (static_cast<uint32_t>(stages) & static_cast<uint32_t>(pipeline_stage::output_merger))) {
        tl_pipe_list = cmd->get_native();  // (looked up only at a draw into a back buffer)
        tl_pipeline = p.handle;
        return;
    }
    if (g_api != device_api::d3d11 || !cmd || !(static_cast<uint32_t>(stages) & static_cast<uint32_t>(pipeline_stage::output_merger))) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_blend.find(p.handle);
    g_bound_blend[cmd->get_native()] = it != g_blend.end() ? it->second : 0;
}
// Does the draw about to happen on this context read `picture` (a pixel shader resource bound to it)?
bool reads(command_list* cmd, std::uint64_t picture) {
    auto* context = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    ID3D11ShaderResourceView* views[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT] = {};
    context->PSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, views);
    bool found = false;
    for (ID3D11ShaderResourceView* v : views) {
        if (!v) continue;
        ID3D11Resource* r = nullptr;
        v->GetResource(&r);
        if (r && reinterpret_cast<std::uint64_t>(r) == picture) found = true;
        if (r) r->Release();
        v->Release();
    }
    return found;
}
// What each draw means (kinds handed over: 0 the copy before the HUD, 1 no HUD this frame, 2 the copy confirmed - the
// final draw reads its picture, 3 the copy given up). A copy counts only once the draw onto the back buffer reads the
// picture it was taken from (a game's alpha-blended draw into some other picture is never handed over); a picture
// overwritten after its copy (a post-processing pass after an alpha-blended one, or a HUD with opaque parts) is given
// up for the frame - the fallback then, not a HUD that is part of the picture.
// Fourth pattern: a draw into a back buffer (Direct3D 12). Overwriting: the picture (again); the first alpha-blended
// draw without depth after it: the HUD's turn - a HUD-less copy of the back buffer then (render target state), handed
// over as a game's HUD-less picture is (HUD = what differs). The picture drawn again after that copy (the same
// pipeline): given up for the frame.
void drawn_back_buffer(command_list* cmd) {
    const std::uint64_t list = cmd->get_native();
    if (tl_list != list || !tl_back_buffer || tl_depth) return;
    const std::uint64_t bb = tl_back_buffer;
    bool take = false, give_up = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = tl_pipe_list == list ? g_blend.find(tl_pipeline) : g_blend.end();
        if (it == g_blend.end()) return;  // (blending unknown: not counted)
        std::uint8_t& step = g_bb_step[bb];
        if (it->second == 0) {
            // (after the copy, an overwriting draw by the pipeline that drew the picture: the picture drawn again - given
            // up; by another one: an opaque part of the HUD - Hi-Fi Rush draws one among its blended ones)
            if (step == 2) {
                if (tl_pipeline == g_bb_picture_pipeline[bb]) { step = 3; give_up = true; }
            } else if (step != 3) {
                step = 1;
                g_bb_picture_pipeline[bb] = tl_pipeline;
            }
        } else if (it->second == 1 && step == 1) {
            step = 2;
            take = true;
            if (!g_bb_seen) {
                g_bb_seen = true;
                reshade::log::message(reshade::log::level::info,
                                      "XPAR: the game draws its HUD straight onto the back buffer after the picture - a HUD-less copy taken before it");
            }
        }
    }
    if (!g_take_hudless) return;
    if (take && tl_pass_list == list) return;  // (inside a render pass: no copy - the frame has none, as if not found)
    if (take) g_take_hudless(bb, 0x4u /* D3D12_RESOURCE_STATE_RENDER_TARGET */, list);
    if (give_up) g_take_hudless(0, 0, list);
}
void drawn(command_list* cmd) {
    if (g_api == device_api::d3d12 && cmd) {
        // (a draw into a candidate cleared to opaque black: counted by its blending - see pick)
        if (tl_inverted && tl_list == cmd->get_native()) {
            std::lock_guard<std::mutex> lock(g_mutex);
            const auto it = tl_pipe_list == tl_list ? g_blend.find(tl_pipeline) : g_blend.end();
            if (it != g_blend.end()) {
                auto& n = g_inverted_draws[tl_inverted];
                n.first += it->second == 1 ? 1u : 0u;
                if (++n.second > 4096) { n.first /= 2; n.second /= 2; }
            }
        }
        drawn_back_buffer(cmd);
        return;
    }
    if (g_api != device_api::d3d11 || !cmd) return;
    std::uint64_t picture = 0, given_up = 0;
    std::uint32_t kind = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto rt = g_rt.find(cmd->get_native());
        if (rt == g_rt.end() || !rt->second.res || rt->second.depth) return;
        const std::uint64_t res = rt->second.res;
        if (is_back_buffer(res)) {
            if (g_took) {  // (the copy taken: confirmed if this draw reads its picture)
                picture = g_took;
                g_took = 0;
                if (!reads(cmd, picture)) return;
                kind = 2;
                g_hud_picture = picture;
                if (picture != g_logged_before) {
                    g_logged_before = picture;
                    reshade::log::message(reshade::log::level::info,
                                          "XPAR: the game draws its HUD into its picture after the post-processing (Direct3D 11) - a copy taken before it");
                }
            } else {
                // (the picture the HUD went into reaching the screen with no HUD drawn into it since its post-processing:
                // no HUD this frame - Resident Evil 2 hides it unless aiming or hurt. Handed over as an empty HUD, not
                // nothing)
                if (!g_hud_picture || g_given_up.count(g_hud_picture) || !g_overwritten.erase(g_hud_picture) || !reads(cmd, g_hud_picture)) return;
                kind = 1;
            }
        } else {
            if (g_given_up.count(res)) return;
            const auto bb = g_bound_blend.find(cmd->get_native());
            const std::uint8_t blend = bb != g_bound_blend.end() ? bb->second : 0;
            if (blend == 0) {
                if (res == g_took) {  // (overwritten after its copy: given up for this frame)
                    g_given_up.insert(res);
                    g_took = 0;
                    picture = res;
                    kind = 3;
                } else {
                    device* dev = cmd->get_device();
                    if (!dev || !g_width) return;
                    const resource_desc d = dev->get_resource_desc(resource{res});
                    if (d.texture.width == g_width && d.texture.height == g_height && d.texture.samples <= 1) g_overwritten.insert(res);
                    return;
                }
            } else {
                if (blend != 1 || !g_overwritten.erase(res)) return;
                if (g_took) {  // (a second picture: the first given up, this one taken)
                    g_given_up.insert(g_took);
                    given_up = g_took;
                }
                picture = res;
                g_took = res;
                kind = 0;
            }
        }
    }
    if (!g_take_before) return;
    if (given_up) g_take_before(given_up, 3, cmd->get_native());
    g_take_before(picture, kind, cmd->get_native());
}
bool on_draw(command_list* cmd, uint32_t, uint32_t, uint32_t, uint32_t) { drawn(cmd); return false; }
bool on_draw_indexed(command_list* cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t) { drawn(cmd); return false; }
void on_reset(command_list* cmd) {
    if (!cmd) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_states.erase(cmd->get_native());
    for (auto& [res, fin] : g_finals) fin.states.erase(cmd->get_native());
}
}  // namespace

void init(std::function<void(std::uint64_t, std::uint32_t, std::uint64_t, bool)> take,
          std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> take_hudless,
          std::function<void(std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t)> take_vk,
          std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> take_before,
          std::function<void(std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t)> take_hudless_vk) {
    g_take = std::move(take);
    g_take_hudless_vk = std::move(take_hudless_vk);
    g_take_hudless = std::move(take_hudless);
    g_take_vk = std::move(take_vk);
    g_take_before = std::move(take_before);
    reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy_resource);
    reshade::register_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
    reshade::register_event<reshade::addon_event::destroy_pipeline>(on_destroy_pipeline);
    reshade::register_event<reshade::addon_event::bind_pipeline>(on_bind_pipeline);
    reshade::register_event<reshade::addon_event::draw>(on_draw);
    reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::register_event<reshade::addon_event::copy_resource>(on_copy_resource);
    reshade::register_event<reshade::addon_event::copy_texture_region>(on_copy_region);
    reshade::register_event<reshade::addon_event::clear_render_target_view>(on_clear);
    reshade::register_event<reshade::addon_event::barrier>(on_barrier);
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::register_event<reshade::addon_event::end_render_pass>(on_end_pass);
    reshade::register_event<reshade::addon_event::reset_command_list>(on_reset);
}
void shutdown() {
    reshade::unregister_event<reshade::addon_event::reset_command_list>(on_reset);
    reshade::unregister_event<reshade::addon_event::destroy_resource>(on_destroy_resource);
    reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::unregister_event<reshade::addon_event::end_render_pass>(on_end_pass);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::unregister_event<reshade::addon_event::barrier>(on_barrier);
    reshade::unregister_event<reshade::addon_event::clear_render_target_view>(on_clear);
    reshade::unregister_event<reshade::addon_event::copy_texture_region>(on_copy_region);
    reshade::unregister_event<reshade::addon_event::copy_resource>(on_copy_resource);
    reshade::unregister_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::unregister_event<reshade::addon_event::draw>(on_draw);
    reshade::unregister_event<reshade::addon_event::bind_pipeline>(on_bind_pipeline);
    reshade::unregister_event<reshade::addon_event::destroy_pipeline>(on_destroy_pipeline);
    reshade::unregister_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
    g_take = nullptr;
    g_take_hudless = nullptr;
    g_take_vk = nullptr;
    g_take_before = nullptr;
    g_take_hudless_vk = nullptr;
}

void on_present(swapchain* chain) {
    if (!chain) return;
    device* dev = chain->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    // (the back buffers' size looked up only when they change - a new swapchain)
    const uint32_t count = chain->get_back_buffer_count();
    bool same = count == g_back_buffers.size();
    for (uint32_t i = 0; i < count && same; ++i) same = chain->get_back_buffer(i).handle == g_back_buffers[i];
    if (!same) {
        g_back_buffers.clear();
        for (uint32_t i = 0; i < count; ++i) g_back_buffers.push_back(chain->get_back_buffer(i).handle);
        if (dev && !g_back_buffers.empty()) {
            const resource_desc d = dev->get_resource_desc(resource{g_back_buffers[0]});
            g_width = d.texture.width; g_height = d.texture.height;
            g_api = dev->get_api();
        }
    }
    g_overwritten.clear();
    g_took = 0;
    g_given_up.clear();
    g_vk_finished = 0;
    g_vk_layer_frame = false;
    if (g_end_pending && ++g_pending_presents >= 2) {  // (never made readable: not taken)
        g_end_pending = false;
        ++g_frames; ++g_not_read;
        g_cleared.clear(); g_with_depth.clear(); g_inverted.clear(); g_last_state.clear(); g_order = 0; g_taken = false;
    }
    // (every 600 frames, how often the layer could not be taken)
    if (g_frames >= 600) {
        char text[384];
        std::snprintf(text, sizeof(text), "XPAR: HUD layer over %u frames: state from the final list %u, from earlier this frame %u, none readable %u, "
                      "not taken %u; at a copy onto the back buffer %u, drawn with depth %u, waited to become readable %u; layers cleared to "
                      "opaque black known %zu, turned down for their draws %u (last: %u of %u alpha-blended); textures drawn with others "
                      "(not a layer) %zu",
                      g_frames, g_same_list, g_elsewhere, g_not_read, g_skipped, g_at_copy, g_with_depth_n, g_waited, g_inverted_seen.size(),
                      g_inv_turned_down, g_inv_alpha, g_inv_draws, g_multi.size());
        reshade::log::message(reshade::log::level::info, text);
        g_frames = g_same_list = g_elsewhere = g_not_read = g_skipped = g_at_copy = g_with_depth_n = g_waited = g_inv_turned_down = 0;
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
    // (the frame's layers start over at its end - end_of_frame; here only when no end was seen since the last present)
    if (!g_triggered && !g_end_pending) {
        g_cleared.clear();
        g_with_depth.clear(); g_inverted.clear();
        g_last_state.clear();
        g_order = 0;
        g_taken = false;
    }
    g_triggered = false;
}
}  // namespace fw::ui_layer
