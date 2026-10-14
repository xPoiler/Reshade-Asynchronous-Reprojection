#include "addon/hud_learn.hpp"
#include <reshade.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace reshade::api;

namespace fw::hud_learn {
namespace {
Probe g_probe;
std::function<void()> g_none;
// Pipelines: the fingerprint of their pixel shader (0: none), and how render target 0 blends (0 overwrite, 1 alpha, 2
// other, 3 not part of it). Direct3D 12: one per pipeline state; Direct3D 11: pixel shaders and blend states apart.
struct Pipe { std::uint32_t fp = 0; std::uint8_t blend = 3; };
std::shared_mutex g_pipes_mutex;
std::unordered_map<std::uint64_t, Pipe> g_pipes;
// What the recording thread has bound (draws are many: checked without the lock - a list is recorded on one thread at
// a time).
thread_local std::uint64_t tl_list = 0, tl_target = 0;
// (the list a render pass is open on, Direct3D 12: no copy or barrier may be recorded inside one - Cyberpunk draws its
// loading screen in render passes, and a copy taken at a draw there hung the GPU)
thread_local std::uint64_t tl_pass_list = 0;
thread_local bool tl_full = false, tl_watched = false;
thread_local std::uint64_t tl_fp_list = 0, tl_blend_list = 0;
thread_local std::uint32_t tl_fp = 0;
thread_local std::uint8_t tl_blend = 0;

std::mutex g_mutex;
device_api g_api = device_api::d3d12;
std::vector<std::uint64_t> g_bbs;
std::uint32_t g_w = 0, g_h = 0, g_presents = 0;
std::atomic<std::uint64_t> g_last_full{0};  // the last back-buffer-sized texture drawn into (not a back buffer)
// Final pictures: textures that reached the back buffer lately (copied onto it, or drawn into last before its first
// draw), with the present count then. Watched (lock-free check at a bind): those and the back buffers.
struct Final { std::uint64_t res; std::uint32_t seen; };
std::vector<Final> g_finals;
constexpr int kWatch = 8;
std::atomic<std::uint64_t> g_watch[kWatch];
// This frame: draws into watched textures (texture, fingerprint, blend, draws), merged while the same; the final
// picture found; the back buffer drawn into yet; the candidate's copy taken.
struct Entry { std::uint64_t res; std::uint32_t fp; std::uint8_t blend; std::uint32_t draws; };
std::vector<Entry> g_log;
std::uint64_t g_final = 0, g_bb_frame = 0;  // (the back buffer drawn into this frame)
bool g_bb_drawn = false, g_taken = false;
// Direct3D 12: per command list, the last back-buffer-sized texture drawn into, and - if it draws into a back buffer - the
// last one before that; the lists in the order the game hands them to the GPU this frame. The final picture is the one
// drawn into last before the back buffer's draw in that order (Elden Ring records its lists on several threads at once:
// the last one recorded is not the last one the GPU runs). Direct3D 11's immediate context runs as it records.
struct ListInfo { std::uint64_t last_full = 0, before_bb = 0; bool bb = false; };
std::unordered_map<std::uint64_t, ListInfo> g_lists;
std::vector<std::uint64_t> g_exec;
thread_local std::uint64_t tl_full_list = 0, tl_full_res = 0;
// Candidates: fingerprint -> where (0 in the final picture, 1 in the back buffer), presents since last seen. The one
// being tried, and for how many frames more.
struct Candidate { std::uint8_t role = 0; std::uint32_t last = 0; };
std::map<std::uint32_t, Candidate> g_cands;
std::uint32_t g_active = 0, g_run = 0;
// The learned point in use (0: learning), and its role; candidates tried since learning (re)started (a pick is locked
// only after a full round); frames in a row the learned point did not draw (long: learned again); the panel's counter.
std::uint32_t g_locked = 0, g_since_drawn = 0, g_used = 0, g_relearn_seen = 0;
// The profile: its path (set by the add-on), read yet, the locked point came from it, saved.
std::wstring g_profile;
bool g_profile_read = false, g_from_profile = false, g_saved = false;
std::uint8_t g_locked_role = 0;
std::vector<std::uint32_t> g_tried;
std::atomic<std::uint32_t> g_status_locked{0};
// Direct3D 12: the final pictures' states as changed last (for the copy as the back buffer's draw reads it).
std::unordered_map<std::uint64_t, std::uint32_t> g_fstate;
// The picture the candidate's copy was taken from, this frame or the last (the only one copied again at the back
// buffer's draw: one the game draws into now, not a remembered handle the game may have destroyed since).
std::uint64_t g_taken_res = 0, g_taken_res_last = 0;
std::atomic<std::uint32_t> g_status_cands{0}, g_status_active{0};
constexpr std::uint32_t kRun = 12;  // frames per candidate

bool is_bb(std::uint64_t r) { return std::find(g_bbs.begin(), g_bbs.end(), r) != g_bbs.end(); }
void rebuild_watch() {
    int n = 0;
    for (const auto b : g_bbs) if (n < kWatch) g_watch[n++] = b;
    for (const auto& f : g_finals) if (n < kWatch) g_watch[n++] = f.res;
    while (n < kWatch) g_watch[n++] = 0;
}
bool watched(std::uint64_t r) {
    for (auto& w : g_watch) if (w.load(std::memory_order_relaxed) == r) return true;
    return false;
}
void add_final(std::uint64_t r) {  // (under g_mutex)
    for (auto& f : g_finals) if (f.res == r) { f.seen = g_presents; return; }
    if (g_finals.size() >= 4) g_fstate.erase(std::min_element(g_finals.begin(), g_finals.end(), [](const Final& a, const Final& b) { return a.seen < b.seen; })->res);
    if (g_finals.size() >= 4) g_finals.erase(std::min_element(g_finals.begin(), g_finals.end(), [](const Final& a, const Final& b) { return a.seen < b.seen; }));
    g_finals.push_back({r, g_presents});
    rebuild_watch();
}
bool api_ok(device* dev) { return dev && (dev->get_api() == device_api::d3d12 || dev->get_api() == device_api::d3d11); }

std::uint32_t fingerprint(const void* code, std::size_t size) {
    std::uint32_t h = 2166136261u;
    const auto* p = static_cast<const std::uint8_t*>(code);
    for (std::size_t i = 0; i < size; ++i) h = (h ^ p[i]) * 16777619u;
    return h ? h : 1;
}
std::uint8_t blend_kind(const blend_desc& b) {
    if (!b.blend_enable[0]) return 0;
    const blend_factor s = b.source_color_blend_factor[0], d = b.dest_color_blend_factor[0];
    return (s == blend_factor::source_alpha || s == blend_factor::one) && d == blend_factor::one_minus_source_alpha ? 1 : 2;
}
void on_init_pipeline(device* dev, pipeline_layout, uint32_t count, const pipeline_subobject* subobjects, pipeline p) {
    if (!p.handle || !api_ok(dev)) return;
    Pipe pipe;
    bool any = false;
    for (uint32_t i = 0; i < count; ++i) {
        if (!subobjects[i].data) continue;
        if (subobjects[i].type == pipeline_subobject_type::pixel_shader) {
            const auto* sh = static_cast<const shader_desc*>(subobjects[i].data);
            if (sh->code && sh->code_size) { pipe.fp = fingerprint(sh->code, sh->code_size); any = true; }
        } else if (subobjects[i].type == pipeline_subobject_type::blend_state) {
            pipe.blend = blend_kind(*static_cast<const blend_desc*>(subobjects[i].data));
            any = true;
        }
    }
    if (!any) return;
    std::unique_lock lock(g_pipes_mutex);
    g_pipes[p.handle] = pipe;
}
void on_destroy_pipeline(device* dev, pipeline p) {
    if (!api_ok(dev)) return;
    std::unique_lock lock(g_pipes_mutex);
    g_pipes.erase(p.handle);
}
void on_bind_pipeline(command_list* cmd, pipeline_stage stages, pipeline p) {
    if (!cmd) return;
    const auto st = static_cast<uint32_t>(stages);
    const bool ps = st & static_cast<uint32_t>(pipeline_stage::pixel_shader), om = st & static_cast<uint32_t>(pipeline_stage::output_merger);
    if (!ps && !om) return;
    Pipe pipe;
    bool found = false;
    {
        std::shared_lock lock(g_pipes_mutex);
        const auto it = g_pipes.find(p.handle);
        if (it != g_pipes.end()) { pipe = it->second; found = true; }
    }
    const std::uint64_t list = cmd->get_native();
    // (a Direct3D 12 pipeline is both; Direct3D 11 binds a pixel shader and a blend state apart)
    if (ps) { tl_fp_list = list; tl_fp = found ? pipe.fp : 0; }
    if (om) { tl_blend_list = list; tl_blend = found && pipe.blend != 3 ? pipe.blend : 0; }
}
void target(command_list* cmd, resource_view first) {
    if (!cmd) return;
    tl_list = cmd->get_native();
    tl_target = 0; tl_full = tl_watched = false;
    device* dev = cmd->get_device();
    if (!first.handle || !api_ok(dev) || !g_w) return;
    const resource r = dev->get_resource_from_view(first);
    if (!r.handle) return;
    const resource_desc d = dev->get_resource_desc(r);
    tl_target = r.handle;
    tl_full = d.texture.width == g_w && d.texture.height == g_h && d.texture.samples <= 1;
    tl_watched = watched(r.handle);
}
void on_bind(command_list* cmd, uint32_t n, const resource_view* rtvs, resource_view) {
    if (cmd && tl_pass_list == cmd->get_native()) tl_pass_list = 0;
    target(cmd, n && rtvs ? rtvs[0] : resource_view{0});
}
bool on_begin_pass(command_list* cmd, uint32_t n, const render_pass_render_target_desc* rts, const render_pass_depth_stencil_desc*, render_pass_flags) {
    target(cmd, n && rts ? rts[0].view : resource_view{0});
    if (cmd) tl_pass_list = cmd->get_native();
    return false;
}
bool on_end_pass(command_list* cmd) { if (cmd && tl_pass_list == cmd->get_native()) tl_pass_list = 0; return false; }
void drawn(command_list* cmd) {
    if (!cmd || !g_probe) return;
    const std::uint64_t list = cmd->get_native();
    if (tl_list != list || !tl_target) return;
    if (!tl_watched) {
        if (tl_full) {
            g_last_full.store(tl_target, std::memory_order_relaxed);
            if (g_api == device_api::d3d12 && (tl_full_list != list || tl_full_res != tl_target)) {  // (the lock only when it changes)
                tl_full_list = list; tl_full_res = tl_target;
                std::lock_guard<std::mutex> lock(g_mutex);
                g_lists[list].last_full = tl_target;
            }
        }
        return;
    }
    const std::uint32_t fp = tl_fp_list == list ? tl_fp : 0;
    const std::uint8_t blend = tl_blend_list == list ? tl_blend : 0;
    const std::uint64_t res = tl_target;
    bool take = false;
    std::uint64_t end_res = 0;
    std::uint32_t end_state = 0;
    bool in_final = false;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const bool bb = is_bb(res);
        if (!bb && tl_full) {  // (a watched final picture is still the last full-size texture drawn into)
            g_last_full.store(res, std::memory_order_relaxed);
            if (g_api == device_api::d3d12) { g_lists[list].last_full = res; tl_full_list = list; tl_full_res = res; }
        }
        if (bb) {
            g_bb_frame = res;
            if (g_api == device_api::d3d12) {
                auto& li = g_lists[list];
                if (!li.bb) { li.bb = true; li.before_bb = li.last_full; }
            }
        }
        if (bb && !g_bb_drawn && g_api == device_api::d3d12 && g_active) {
            // (a candidate in the final picture: the final picture as the back buffer's draw reads it - the latest one)
            const auto c = g_cands.find(g_active);
            const std::uint64_t from = g_taken_res ? g_taken_res : g_taken_res_last;
            if (c != g_cands.end() && c->second.role == 0 && from && !is_bb(from) &&
                std::any_of(g_finals.begin(), g_finals.end(), [&](const Final& f) { return f.res == from; })) {
                // (only with its state known: a guessed one - shader resource - was wrong while Cyberpunk loaded, and the copy's
                // transition hung the GPU)
                const auto st = g_fstate.find(from);
                if (st != g_fstate.end()) { end_res = from; end_state = st->second; }
            }
        }
        if (bb && !g_bb_drawn) {
            g_bb_drawn = true;
            // (Direct3D 11: the final picture is the texture drawn into last before the back buffer's first draw;
            // Direct3D 12: found at the present, in the order the lists ran)
            const std::uint64_t last = g_last_full.load(std::memory_order_relaxed);
            if (g_api == device_api::d3d11 && last && !g_final) { g_final = last; add_final(last); }
        }
        if (!g_log.empty() && g_log.back().res == res && g_log.back().fp == fp && g_log.back().blend == blend) ++g_log.back().draws;
        else if (g_log.size() < 512) g_log.push_back({res, fp, blend, 1});
        if (g_active && fp == g_active && !g_taken) {
            const auto c = g_cands.find(fp);
            if (c != g_cands.end() && (c->second.role == 1) == bb) { g_taken = true; take = true; if (!bb) g_taken_res = res; in_final = !bb; }
        }
    }
    const bool use = g_locked && fp == g_locked;
    if (tl_pass_list == list) return;  // (inside a render pass: nothing copied - see tl_pass_list)
    if (end_res) g_probe(end_res, false, end_state, list, g_active, true, false);
    // (in the final picture, Direct3D 12: compared only with the final picture as the back buffer's draw reads it - see end)
    if (take) g_probe(res, g_api == device_api::d3d11, 0x4u /* D3D12_RESOURCE_STATE_RENDER_TARGET */ | (in_final ? 0x80000000u : 0u), list, fp, false, use);
}
bool on_draw(command_list* cmd, uint32_t, uint32_t, uint32_t, uint32_t) { drawn(cmd); return false; }
bool on_draw_indexed(command_list* cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t) { drawn(cmd); return false; }
bool on_draw_indirect(command_list* cmd, indirect_command type, resource, uint64_t, uint32_t, uint32_t) {
    if (type == indirect_command::draw || type == indirect_command::draw_indexed) drawn(cmd);
    return false;
}
void copied(command_list* cmd, resource src, resource dst) {
    if (!cmd || !src.handle || !dst.handle) return;
    device* dev = cmd->get_device();
    if (!api_ok(dev)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!is_bb(dst.handle) || is_bb(src.handle)) return;
    const resource_desc d = dev->get_resource_desc(src);
    if (d.texture.width != g_w || d.texture.height != g_h) return;
    g_final = src.handle;
    add_final(src.handle);
}
bool on_copy_resource(command_list* cmd, resource src, resource dst) { copied(cmd, src, dst); return false; }
void on_barrier(command_list* cmd, uint32_t count, const resource* resources, const resource_usage*, const resource_usage* new_states) {
    if (!cmd || g_api != device_api::d3d12) return;
    for (uint32_t i = 0; i < count; ++i) {
        if (!watched(resources[i].handle)) continue;
        std::lock_guard<std::mutex> lock(g_mutex);
        if (is_bb(resources[i].handle)) continue;
        const auto v = static_cast<std::uint32_t>(new_states[i]);
        g_fstate[resources[i].handle] = (v & 0x80000000u) ? 0u : v;  // (as D3D12's: ReShade's special states are its common/present)
    }
}
// (a destroyed texture is forgotten: never copied again)
void on_destroy_resource(device*, resource r) {
    if (!r.handle || !watched(r.handle)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto before = g_finals.size();
    g_finals.erase(std::remove_if(g_finals.begin(), g_finals.end(), [&](const Final& f) { return f.res == r.handle; }), g_finals.end());
    g_fstate.erase(r.handle);
    if (g_taken_res == r.handle) g_taken_res = 0;
    if (g_taken_res_last == r.handle) g_taken_res_last = 0;
    if (g_final == r.handle) g_final = 0;
    if (g_finals.size() != before) rebuild_watch();
}
void on_execute(command_queue*, command_list* cmd) {
    if (!cmd || g_api != device_api::d3d12) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_exec.size() < 1024) g_exec.push_back(cmd->get_native());
}
void on_reset(command_list* cmd) {
    if (!cmd) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_lists.erase(cmd->get_native());
}
bool on_copy_region(command_list* cmd, resource src, uint32_t, const subresource_box*, resource dst, uint32_t, const subresource_box*, filter_mode) {
    copied(cmd, src, dst);
    return false;
}
// The frame's groups into a texture: consecutive draws by one pixel shader.
std::string describe(std::uint64_t res, const char* name) {
    std::string out = std::string(name) + ":";
    char item[48];
    int n = 0;
    for (const auto& e : g_log) {
        if (e.res != res) continue;
        std::snprintf(item, sizeof(item), " %08x/%c%u", e.fp, "oa+-"[std::min<int>(e.blend, 3)], e.draws);
        out += item;
        if (++n >= 24) { out += " ..."; break; }
    }
    return out;
}
}  // namespace

void init(Probe probe, std::function<void()> none) {
    g_probe = std::move(probe);
    g_none = std::move(none);
    for (auto& w : g_watch) w = 0;
    reshade::register_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
    reshade::register_event<reshade::addon_event::destroy_pipeline>(on_destroy_pipeline);
    reshade::register_event<reshade::addon_event::bind_pipeline>(on_bind_pipeline);
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::register_event<reshade::addon_event::end_render_pass>(on_end_pass);
    reshade::register_event<reshade::addon_event::draw>(on_draw);
    reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(on_draw_indirect);
    reshade::register_event<reshade::addon_event::copy_resource>(on_copy_resource);
    reshade::register_event<reshade::addon_event::copy_texture_region>(on_copy_region);
    reshade::register_event<reshade::addon_event::execute_command_list>(on_execute);
    reshade::register_event<reshade::addon_event::barrier>(on_barrier);
    reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy_resource);
    reshade::register_event<reshade::addon_event::reset_command_list>(on_reset);
}
void shutdown() {
    reshade::unregister_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
    reshade::unregister_event<reshade::addon_event::destroy_pipeline>(on_destroy_pipeline);
    reshade::unregister_event<reshade::addon_event::bind_pipeline>(on_bind_pipeline);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::unregister_event<reshade::addon_event::end_render_pass>(on_end_pass);
    reshade::unregister_event<reshade::addon_event::draw>(on_draw);
    reshade::unregister_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::unregister_event<reshade::addon_event::draw_or_dispatch_indirect>(on_draw_indirect);
    reshade::unregister_event<reshade::addon_event::copy_resource>(on_copy_resource);
    reshade::unregister_event<reshade::addon_event::copy_texture_region>(on_copy_region);
    reshade::unregister_event<reshade::addon_event::execute_command_list>(on_execute);
    reshade::unregister_event<reshade::addon_event::barrier>(on_barrier);
    reshade::unregister_event<reshade::addon_event::destroy_resource>(on_destroy_resource);
    reshade::unregister_event<reshade::addon_event::reset_command_list>(on_reset);
    g_probe = nullptr;
    g_none = nullptr;
}

void on_present(swapchain* chain, bool learning, bool paused, std::uint32_t pick, std::uint32_t fail, std::uint32_t relearn) {
    if (!chain) return;
    device* dev = chain->get_device();
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_presents;
    g_bbs.clear();
    for (uint32_t i = 0; i < chain->get_back_buffer_count(); ++i) g_bbs.push_back(chain->get_back_buffer(i).handle);
    if (dev && !g_bbs.empty()) {
        const resource_desc d = dev->get_resource_desc(resource{g_bbs[0]});
        g_w = d.texture.width; g_h = d.texture.height;
        g_api = dev->get_api();
    }
    // (final pictures not seen for 300 presents are forgotten)
    const auto before = g_finals.size();
    g_finals.erase(std::remove_if(g_finals.begin(), g_finals.end(), [](const Final& f) { return g_presents - f.seen > 300; }), g_finals.end());
    if (g_finals.size() != before || g_watch[0].load() != (g_bbs.empty() ? 0 : g_bbs[0])) rebuild_watch();
    // (Direct3D 12: the final picture - unless copied onto the back buffer - from the lists in the order they ran)
    if (g_api == device_api::d3d12 && !g_final) {
        std::uint64_t last = 0;
        for (const std::uint64_t l : g_exec) {
            const auto it = g_lists.find(l);
            if (it == g_lists.end()) continue;
            if (it->second.bb) { if (it->second.before_bb) last = it->second.before_bb; break; }
            if (it->second.last_full) last = it->second.last_full;
        }
        if (last && !is_bb(last)) { g_final = last; add_final(last); }
    }
    g_exec.clear();
    if (g_lists.size() > 4096) g_lists.clear();
    // The frame's candidates: the groups into its final picture, and into the back buffer after its first draw.
    // (the last 16 groups of each: the HUD comes at the end - a final picture the scene is drawn into as well, Elden Ring,
    // has hundreds before it)
    std::vector<std::pair<std::uint32_t, std::uint8_t>> found;
    std::uint32_t seen_bb = 0;
    for (const auto& e : g_log) {
        if (!e.fp) continue;
        const bool bb = is_bb(e.res);
        if (!bb && e.res != g_final) continue;
        if (bb && seen_bb++ == 0) continue;  // (the back buffer's first group: the picture drawn onto it)
        found.push_back({e.fp, bb ? std::uint8_t(1) : std::uint8_t(0)});
    }
    for (std::uint8_t role = 0; role < 2; ++role) {
        int n = 0;
        for (auto it = found.rbegin(); it != found.rend() && n < 16; ++it) {
            if (it->second != role) continue;
            ++n;
            auto& c = g_cands[it->first];
            c.role = role;
            c.last = g_presents;
        }
    }
    for (auto it = g_cands.begin(); it != g_cands.end();)
        it = g_presents - it->second.last > 600 ? g_cands.erase(it) : std::next(it);
    // (every 600 presents, what the frame drawn last looked like)
    if (g_presents % 600 == 0 && g_final) {
        const std::string text = "XPAR HUD learning: " + describe(g_final, "final picture") + " | " + describe(g_bb_frame, "back buffer") +
                                 " | candidates " + std::to_string(g_cands.size());
        reshade::log::message(reshade::log::level::info, text.c_str());
    }
    // The saved point (FrameWarp\profiles\hud_point.txt), once: locked from the start - no learning on every launch.
    if (!g_profile_read && !g_profile.empty()) {
        g_profile_read = true;
        FILE* f = nullptr;
        if (_wfopen_s(&f, g_profile.c_str(), L"r") == 0 && f) {
            unsigned fp = 0, role = 0;
            if (fscanf_s(f, "%x %u", &fp, &role) == 2 && fp && learning) {
                g_locked = fp;
                g_locked_role = static_cast<std::uint8_t>(role ? 1 : 0);
                g_from_profile = g_saved = true;
                g_cands[fp].role = g_locked_role;
                g_cands[fp].last = g_presents;
                char text[128];
                std::snprintf(text, sizeof(text), "XPAR HUD learning: %08x from the profile (in the %s)", fp, g_locked_role ? "back buffer" : "final picture");
                reshade::log::message(reshade::log::level::info, text);
            }
            std::fclose(f);
        }
    }
    // Paused while the game gives its HUD itself (its HUD layer, a HUD-less picture): nothing copied, nothing learned or
    // forgotten meanwhile - the copies were of no use there, and in Cyberpunk, whose saved point is in its final picture, they
    // hung the GPU while it loaded (the picture's state not known, guessed).
    if (paused) {
        g_active = 0;
    } else {
        // Learned again: from the panel (the profile forgotten too), or the presenter found the learned point no longer behaving
        // like a HUD (the same), or - not proven yet (a minute of frames with the HUD from it) - it has not drawn for 600 frames
        // (a point locked on a title screen with the camera moving behind its menu: Elden Ring). A proven point (saved, or from
        // the profile) waits through menus and a HUD hidden for any length of time.
        const bool drew = g_final || g_bb_frame;  // (the game drew its picture this frame: not a loading screen)
        const bool forget = relearn != g_relearn_seen || (g_locked && fail == g_locked);
        if (forget || (g_locked && !g_saved && g_since_drawn > 600) || !learning) {
            if (g_locked && learning) {
                char text[128];
                std::snprintf(text, sizeof(text), "XPAR HUD learning: %08x dropped (%s) - learning again", g_locked,
                              relearn != g_relearn_seen ? "Relearn" : fail == g_locked ? "it no longer behaves like a HUD" : "it has not drawn lately");
                reshade::log::message(reshade::log::level::info, text);
            }
            if (forget && learning && !g_profile.empty()) _wremove(g_profile.c_str());
            if (forget) g_saved = false;
            g_relearn_seen = relearn;
            g_locked = 0;
            g_since_drawn = 0;
            g_used = 0;
            g_from_profile = false;
            g_tried.clear();
        }
        // The learned point: locked once every candidate has been tried and the presenter picks one of them.
        if (learning && !g_locked && pick && g_cands.count(pick) &&
            std::all_of(g_cands.begin(), g_cands.end(), [](const auto& c) { return std::find(g_tried.begin(), g_tried.end(), c.first) != g_tried.end(); })) {
            g_locked = pick;
            g_locked_role = g_cands[pick].role;
            g_since_drawn = 0;
            g_used = 0;
            char text[128];
            std::snprintf(text, sizeof(text), "XPAR HUD learning: %08x learned (in the %s) - the HUD taken from there", pick,
                          g_locked_role ? "back buffer" : "final picture");
            reshade::log::message(reshade::log::level::info, text);
        }
        if (g_locked) {
            // A frame the game drew where the learned point did not: no HUD (the game hides it - for as long as it does).
            if (g_taken) {
                g_since_drawn = 0;
                // (proven in gameplay - a minute of frames with the HUD from it: saved for the next launch)
                if (++g_used >= 3600 && !g_saved && !g_profile.empty()) {
                    g_saved = true;
                    FILE* f = nullptr;
                    if (_wfopen_s(&f, g_profile.c_str(), L"w") == 0 && f) {
                        std::fprintf(f, "%08x %u\n", g_locked, unsigned(g_locked_role));
                        std::fclose(f);
                        char text[128];
                        std::snprintf(text, sizeof(text), "XPAR HUD learning: %08x saved to the profile", g_locked);
                        reshade::log::message(reshade::log::level::info, text);
                    }
                }
            } else if (drew) {
                ++g_since_drawn;
                if (g_none) g_none();
            }
            g_active = g_locked;
        } else if (!learning) {
            g_active = 0;
        } else if (g_run == 0 || !g_cands.count(g_active)) {
            // The next candidate after a run (in fingerprint order, round and round).
            auto it = g_cands.upper_bound(g_active);
            if (it == g_cands.end()) it = g_cands.begin();
            g_active = it == g_cands.end() ? 0 : it->first;
            g_run = kRun;
            if (g_active && std::find(g_tried.begin(), g_tried.end(), g_active) == g_tried.end()) g_tried.push_back(g_active);
        } else {
            --g_run;
        }
    }
    g_status_cands = static_cast<std::uint32_t>(g_cands.size());
    g_status_active = g_active;
    g_status_locked = g_locked;
    g_log.clear();
    g_final = 0;
    g_bb_frame = 0;
    g_taken_res_last = g_taken_res;
    g_taken_res = 0;
    g_bb_drawn = false;
    g_taken = false;
}

void set_profile(const std::wstring& path) { std::lock_guard<std::mutex> lock(g_mutex); g_profile = path; g_profile_read = false; }
Status status() { return {g_status_cands.load(), g_status_active.load(), g_status_locked.load()}; }
}  // namespace fw::hud_learn
