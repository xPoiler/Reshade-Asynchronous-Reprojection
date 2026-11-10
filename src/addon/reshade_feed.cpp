#include "addon/reshade_feed.hpp"
#include "addon/depth_choice.hpp"
#include "addon/ngx_hooks.hpp"
#include "addon/mv_probe.hpp"
#include "addon/camera_search.hpp"
#include "addon/vk_transport.hpp"
#include <reshade.hpp>
#include <d3d12.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <cctype>
#include <cstdio>
#include <cstring>
#include "addon/diag.hpp"

namespace fw {
namespace {
using namespace reshade::api;

Producer* g_producer = nullptr;

struct State {
    bool scanned = false;
    unsigned waited = 0;
    effect_technique feed{0};
    effect_texture_variable depth{0}, probe{0}, bound{0};
    effect_uniform_variable flip{0}, far_switch{0};
    bool far_on = false;  // XPAR_Far as last set
    char status[200] = "";
} g;

// A texture variable's name as ReShade reports it may carry its namespace; the declared name is its end.
bool named(const char* reported, const char* declared) {
    const std::size_t n = std::strlen(reported), m = std::strlen(declared);
    if (n < m || std::strcmp(reported + n - m, declared) != 0) return false;
    return n == m || !std::isalnum(static_cast<unsigned char>(reported[n - m - 1]));
}

// XPAR.fx's technique and its output.
void scan(effect_runtime* runtime) {
    g.scanned = true;
    g.feed = runtime->find_technique("XPAR.fx", "XPAR_Feed");
    g.depth = {0}; g.probe = {0}; g.bound = {0};
    g.flip = runtime->find_uniform_variable("XPAR.fx", "XPAR_Flip");
    g.far_switch = runtime->find_uniform_variable("XPAR.fx", "XPAR_Far");
    g.far_on = false;  // (a reloaded effect starts with its default)
    runtime->enumerate_texture_variables("XPAR.fx", [&](effect_runtime* r, effect_texture_variable v) {
        char name[160] = "";
        r->get_texture_variable_name(v, name);
        if (named(name, "XPAR_Depth")) g.depth = v;
        if (named(name, "XPAR_Probe")) g.probe = v;         // (depth_choice.hpp)
        if (named(name, "DepthBufferTex")) g.bound = v;     // (ReShade.fxh: what ReShade binds the depth to)
    });
}

void set_status(const char* text) { std::snprintf(g.status, sizeof(g.status), "%s", text); }

}  // namespace

void install_reshade_feed(Producer* producer) { g_producer = producer; mv_probe::set_producer(producer); }

// StencilDump=1 (development, plan_character_coordinates.md option 2): whether the game marks its characters in its depth's stencil. The
// depth an upscaler is handed and the one ReShade gives effects are copies without it (Requiem: format 41 both): the stencil is only in the
// game's own depth-stencil while it draws. So every 60th frame, each time the game stops drawing into a depth-stencil with a stencil plane
// (another bound in its place on that list), up to 8 times a frame, its stencil is copied there - in the depth-write state it was bound in.
namespace {
std::mutex g_stencil_mutex;
std::unordered_map<std::uint64_t, std::uint64_t> g_stencil_bound;  // (list -> the depth-stencil resource bound on it)
std::atomic<std::uint64_t> g_stencil_frame{0};
std::atomic<std::uint32_t> g_stencil_taken{0};
bool stencil_on() {
    static std::atomic<int> on{-1};
    static std::atomic<std::uint64_t> checked{0};
    const std::uint64_t f = g_stencil_frame.load(std::memory_order_relaxed);
    if (on.load() < 0 || f - checked.load() > 60) { on = GetEnvironmentVariableW(L"FW_STENCIL_DUMP", nullptr, 0) ? 1 : 0; checked = f; }
    return on.load() == 1;
}
// (StencilMask=1, development: every frame, the stencil kept for the presenter's character mask - see Producer::stencil_stage)
std::atomic<std::uint32_t> g_stencil_staged{0};
bool stencil_mask_on() {
    static std::atomic<int> on{-1};
    static std::atomic<std::uint64_t> checked{0};
    const std::uint64_t f = g_stencil_frame.load(std::memory_order_relaxed);
    if (on.load() < 0 || f - checked.load() > 60) { on = GetEnvironmentVariableW(L"FW_STENCIL_MASK", nullptr, 0) ? 1 : 0; checked = f; }
    return on.load() == 1;
}
void on_stencil_bind(command_list* cmd, uint32_t, const resource_view*, resource_view dsv) {
    if (!cmd || !g_producer || cmd->get_device()->get_api() != device_api::d3d12) return;
    const bool dump = stencil_on(), mask = stencil_mask_on();
    if (!dump && !mask) return;
    const std::uint64_t frame = g_stencil_frame.load(std::memory_order_relaxed);
    device* dev = cmd->get_device();
    const std::uint64_t now = dsv.handle ? dev->get_resource_from_view(dsv).handle : 0;
    std::uint64_t before = 0;
    {
        std::lock_guard<std::mutex> lock(g_stencil_mutex);
        auto& b = g_stencil_bound[cmd->get_native()];
        before = b; b = now;
    }
    if (!before || before == now) return;
    auto* res = reinterpret_cast<ID3D12Resource*>(before);
    const D3D12_RESOURCE_DESC d = res->GetDesc();
    if (d.Format != DXGI_FORMAT_R32G8X24_TYPELESS && d.Format != DXGI_FORMAT_D32_FLOAT_S8X24_UINT && d.Format != DXGI_FORMAT_R24G8_TYPELESS && d.Format != DXGI_FORMAT_D24_UNORM_S8_UINT) return;
    if (d.Width < 640) return;
    // (at most 4 a frame: the latest before the depth tag is handed over - Requiem's 5 snapshots a frame were alike)
    if (mask && g_stencil_staged.fetch_add(1) < 4)
        g_producer->stage_stencil(res, reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native()));
    if (!dump || frame % 60 != 30) return;
    const std::uint32_t k = g_stencil_taken.fetch_add(1);
    if (k >= 8) return;
    g_producer->dump_stencil(frame, k, res, D3D12_RESOURCE_STATE_DEPTH_WRITE, reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native()));
}
void on_stencil_present(command_queue*, swapchain*, const rect*, const rect*, uint32_t, const rect*) {
    g_stencil_frame.fetch_add(1, std::memory_order_relaxed);
    g_stencil_taken = 0;
    g_stencil_staged = 0;
    if (g_producer && stencil_on()) g_producer->collect_stencil();
}
}  // namespace
// (only with StencilDump=1 or StencilMask=1 - their keys reach the environment as the presenter is started, so this is called again
// then: off, nothing is hooked, not even a callback on the game's every depth-stencil bind)
namespace { bool g_stencil_events = false; }
void register_stencil_dump_events() {
    if (g_stencil_events || (!GetEnvironmentVariableW(L"FW_STENCIL_DUMP", nullptr, 0) && !GetEnvironmentVariableW(L"FW_STENCIL_MASK", nullptr, 0))) return;
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_stencil_bind);
    reshade::register_event<reshade::addon_event::present>(on_stencil_present);
    g_stencil_events = true;
}
void unregister_stencil_dump_events() {
    if (!g_stencil_events) return;
    reshade::unregister_event<reshade::addon_event::present>(on_stencil_present);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_stencil_bind);
    g_stencil_events = false;
}
const char* reshade_feed_status() { return g.status; }

void on_reshade_reloaded_effects(effect_runtime*) { g.scanned = false; }  // (every handle from before is gone)

void on_reshade_finish_effects(effect_runtime* runtime, command_list* cmd_list, resource_view, resource_view) {
    Producer* p = g_producer;
    if (!p || !p->ready() || !runtime || !cmd_list) return;
    device* dev = runtime->get_device();
    if (!dev || (dev->get_api() != device_api::d3d12 && !(dev->get_api() == device_api::d3d11 && p->d3d11()) &&
                 !(dev->get_api() == device_api::vulkan && p->vulkan())))
        return;
    // The game's own depth and motion vectors (an upscaler's, Streamline's) always come first: the feed is
    // what there is when nothing else gives them (a game without DLSS or FSR, or with both switched off, at
    // the start or in the middle of a session), and it is out from the first frame an upscaler gives. It
    // starts after 600 frames without any, so that a game whose upscaler is on does not see it while it
    // loads or in a short pause; what was learned from the one source is dropped for the other (presenter).
    // Not while the game's frame generation is in use either: its frames come from there (the generated images
    // are taken for the game's own frames), and the feed - run per presented image, on frame generation's own
    // present thread - would fill the frame slots with frames of its own, leaving the generated images nowhere
    // to go (Cyberpunk 2077 with FSR frame generation and DLSS not running: nothing shown at all).
    const bool wanted = p->shared()->settings.enabled && !p->game_depth_seen() && !p->generation_active();
    if (!g.scanned) scan(runtime);
    if (GetEnvironmentVariableW(L"FW_STENCIL_DUMP", nullptr, 0)) {
        static unsigned stencil_said = 0;
        if (++stencil_said % 300 == 1) {
            resource_view view{0}, unused_view{0};
            if (g.depth.handle) runtime->get_texture_binding(g.depth, &view, &unused_view);
            const std::string text = "XPAR: StencilDump: api " + std::to_string(int(dev->get_api())) + ", XPAR_Depth " + std::to_string(g.depth.handle != 0) +
                                     ", bound " + std::to_string(view.handle != 0) + ", " + p->stencil_status();
            fw::diag_log(reshade::log::level::info, text.c_str());
        }
    }
    // The camera search also while an upscaler gives depth and motion but not the game's camera (NGX or FSR without Streamline): the camera is
    // estimated there otherwise. Its values go with the frame the present closes (Producer::set_present_matrices). Paused once the game's own
    // camera comes (Streamline), and while frame generation runs (its frames are not the game's).
    const bool upscaler_search = p->shared()->settings.enabled && !wanted && !p->generation_active() && p->upscaler_camera_estimated();
    if (wanted) { track_depth_buffers(); camera_search::resume(); } else release_depth_choice(runtime);
    if (upscaler_search) {
        camera_search::resume();
        Shared* ush = p->shared();
        static MatrixCandidate upscaler_matrices[kMatrixCandidates];
        const std::uint32_t count = camera_search::take(upscaler_matrices, kMatrixCandidates, ush->camera_held, ush->camera_held_state);
        if (count) p->set_present_matrices(upscaler_matrices, count);
    }
    // (an upscaler's data now: nothing of the motion search stays - its scale, its timing, its hooks at work on every target. Found again if the feed comes back)
    if (!wanted) {
        mv_probe::pause();
        if (!upscaler_search) camera_search::pause();
        if (p->shared()->motion_found) { InterlockedExchange(&p->shared()->motion_found, 0); InterlockedIncrement(&p->shared()->motion_restarts); }
    }
    // (why the feed has not started, every 300 frames while it has not: a game that stays without depth says what holds it)
    static unsigned s_said = 0;
    if ((!wanted || g.waited < 600) && ++s_said % 300 == 0) {
        char text[200];
        std::snprintf(text, sizeof(text), "XPAR: ReShade feed waiting: on %d, the game's depth seen %d, frame generation %d, depth buffers bound a frame %.0f, frames of a 3D scene %u of 600, XPAR.fx %d",
                      int(p->shared()->settings.enabled), int(p->game_depth_seen()), int(p->generation_active()), depth_binds_per_frame(), g.waited, int(g.feed.handle != 0));
        fw::diag_log(reshade::log::level::info, text);
    }
    if (!wanted) {
        g.status[0] = 0;
        g.waited = 0;
        // (the technique is hidden and only ever enabled here: off whenever the feed is not running)
        if (g.feed.handle && runtime->get_technique_state(g.feed)) runtime->set_technique_state(g.feed, false);
        return;
    }
    if (!g.feed.handle || !g.depth.handle) {
        set_status("No DLSS or FSR data from this game. XPAR.fx is not loaded: put it in reshade-shaders\\Shaders "
                   "(and turn off 'Load only enabled effects' in ReShade's settings).");
        return;
    }
    // The feed runs with the other effects, as any depth effect does (that is when ReShade has the depth
    // buffer bound and readable); enabled here.
    if (!runtime->get_technique_state(g.feed)) runtime->set_technique_state(g.feed, true);
    resource_view depth_view{0}, unused{0};
    runtime->get_texture_binding(g.depth, &depth_view, &unused);
    if (!depth_view.handle) return;  // (not created yet: from the next frame on)
    const resource depth = dev->get_resource_from_view(depth_view);
    if (!depth.handle || !cmd_list->get_native()) return;
    const resource_desc desc = dev->get_resource_desc(depth);
    const std::uint32_t w = desc.texture.width, h = desc.texture.height;
    // Which of the game's depth buffers ReShade hands over (depth_choice.hpp): while that is being found out,
    // this frame's depth is a candidate's and is not published.
    resource_view probe_view{0}, bound_view{0};
    if (g.probe.handle) runtime->get_texture_binding(g.probe, &probe_view, &unused);
    if (g.bound.handle) runtime->get_texture_binding(g.bound, &bound_view, &unused);
    const bool choosing = choose_depth_buffer(runtime, cmd_list, probe_view, bound_view) == DepthChoice::kChoosing;
    // Without a depth buffer that belongs to the picture (none stands out, or one is still being looked for, or
    // its orientation is not known yet), everything is handed over as far away: the camera's turns are warped
    // exactly all the same (they need no depth), walking and strafing move at the game's frame rate. Better
    // than an unrelated buffer's depth, which moves parts of the picture by distances that are not there.
    // (An older XPAR.fx without the switch: the depth is not published until it is known, as before.)
    const bool usable = depth_usable() && !choosing;
    // The depth handed over at the size of the game's depth buffer, once the one that belongs to the picture is chosen (XPAR.fx's
    // XPAR_FEED_W/H; the effect is rebuilt with it, from the next frames on): it is the size the game renders at before upscaling.
    // Half of a 4K picture before, 1920x1080, for a game rendering at 1280x720 (Cyberpunk, XeSS Ultra Performance): every pass of the
    // presenter's on 2.25 times the pixels the game's depth had - depth and motion 1.3 ms a game frame, 0.11 ms with DLSS's own depth.
    // Never larger than it was before (XPAR.fx's XPAR_FEED_DIV: the picture up to 1440p, half of it above): the depth chosen can be one
    // the size of the picture - Cyberpunk, XeSS, a 3840x2160 one chosen after a scene change - and at 4K every slot and pass of the
    // presenter's went with it: 2 GB of an 8 GB card's budget of 1.3, XPAR aside for good, the game freezing.
    // (The cap also before a buffer is chosen: a size saved in the preset larger than it is brought down at once.)
    {
        uint32_t sw = 0, shh = 0;
        runtime->get_screenshot_width_and_height(&sw, &shh);
        const uint32_t div = shh > 1440 ? 2 : 1;
        const uint32_t cap_w = sw ? sw / div : w, cap_h = shh ? shh / div : h;
        uint32_t fw_ = std::min(w, cap_w), fh_ = std::min(h, cap_h);
        if (usable && bound_view.handle) {
            const resource_desc gd = dev->get_resource_desc(dev->get_resource_from_view(bound_view));
            std::uint32_t rw = gd.texture.width, rh = gd.texture.height;
            // (the render resolution, when the scene's depth chosen is the picture's size - a copy at the upscaled size, Cyberpunk's 3840x2160
            // beside its 1280x720: depth_choice.hpp. Not otherwise: a buffer below the picture's size is the game's own, and smaller ones of the
            // same scene are its reductions - Enshrouded's 640x360 beside its 1280x720, handed over at a quarter before this)
            if (sw && shh && gd.texture.width >= sw && gd.texture.height >= shh) scene_depth_size(rw, rh);
            fw_ = std::min(rw, cap_w); fh_ = std::min(rh, cap_h);
        }
        static uint32_t asked_w = 0, asked_h = 0;
        if (fw_ >= 64 && fh_ >= 64 && (fw_ != w || fh_ != h) && (fw_ != asked_w || fh_ != asked_h)) {
            asked_w = fw_; asked_h = fh_;
            char v[16];
            std::snprintf(v, sizeof(v), "%u", fw_);
            runtime->set_preprocessor_definition_for_effect("XPAR.fx", "XPAR_FEED_W", v);
            std::snprintf(v, sizeof(v), "%u", fh_);
            runtime->set_preprocessor_definition_for_effect("XPAR.fx", "XPAR_FEED_H", v);
            char text[200];
            std::snprintf(text, sizeof(text), "XPAR: ReShade feed: depth handed over at the game's depth buffer's size, %ux%u (was %ux%u, the picture %ux%u)", fw_, fh_, w, h, sw, shh);
            fw::diag_log(reshade::log::level::info, text);
        }
    }
    const bool oriented = usable && depth_orientation_ready(runtime, cmd_list, probe_view, g.flip);
    const bool far_only = !oriented;
    // Nothing goes to the presenter before the game draws a 3D scene - 600 frames of it in a row: its menus at the start of a session are no
    // reason to start. Started there, in a game whose upscaler comes with gameplay, the presenter took its ~900 MB before the game loaded its
    // world: Enshrouded on an 8 GB card at 6-8 fps for a minute after loading. A 3D scene: depth buffers bound, and the choice among them made -
    // one holds a scene that is not one value everywhere (its orientation read), or none stands out (as a game whose depth reads empty: the
    // feed runs on as far away). Not a count of binds: Dark Souls' world binds 11-15 a frame, a menu of Enshrouded's 8, 20 kept Dark Souls
    // waiting for ever. Once running, the feed stays on through menus and pauses.
    if (g.waited < 600) {
        const bool scene = depth_binds_per_frame() > 0.0 && (oriented || (!choosing && !depth_usable()));
        g.waited = scene ? g.waited + 1 : 0;
        if (g.waited == 600) {
            choose_depth_again(runtime);
            char text[200];
            std::snprintf(text, sizeof(text), "XPAR: no DLSS or FSR data, a 3D scene drawn (%.0f depth buffers bound a frame, its depth %s): the ReShade feed starts",
                          depth_binds_per_frame(), oriented ? "read" : "not told apart - handed over as far away");
            fw::diag_log(reshade::log::level::info, text);
        } else {
            set_status(choosing ? "No DLSS or FSR data from this game: waiting for a 3D scene - finding the game's depth buffer..."
                                : "No DLSS or FSR data from this game: waiting for a 3D scene.");
            return;
        }
    }
    if (g.far_switch.handle) {
        // (applies from the next frame's depth: this frame's was drawn with the switch as it was)
        const bool was = g.far_on;
        runtime->set_uniform_value_bool(g.far_switch, far_only);
        g.far_on = far_only;
        if (far_only && !was) return;
    }
    if (far_only && !g.far_switch.handle) {
        set_status(choosing ? "No DLSS or FSR data from this game: finding the game's depth buffer..."
                            : "No DLSS or FSR data from this game: checking which way round its depth is stored...");
        return;
    }

    p->note_feed();
    const std::uint64_t frame = next_estimated_frame();
    Camera cam{};
    cam.valid = 1;
    cam.estimated = 1;
    cam.depth_inverted = 1;  // (XPAR.fx writes near = 1, far = 0)
    // No motion vectors: the presenter estimates them from the picture, in uv; this makes them pixels of the depth.
    cam.mvec_scale[0] = static_cast<float>(w);
    cam.mvec_scale[1] = static_cast<float>(h);
    // (development, MotionSearch=1: the game's own motion, once found - see mv_probe::motion_step. Direct3D 11 and 12; Vulkan not searched, see kVulkanSearch)
    Shared* sh = p->shared();
    const mv_probe::MotionStep search = mv_probe::motion_step(sh->motion_found != 0, sh->motion_candidate, w, sh->motion_ruled_out);
    if (search.lost) { InterlockedExchange(&sh->motion_found, 0); InterlockedIncrement(&sh->motion_restarts); }
    // (the motion and the candidates at no more than 1920 wide, whatever the depth's size: the warp needs no more of it, and at 4K four slots of it took XPAR over
    // an 8 GB card's budget - the background memory let go of, Cyberpunk with XeSS)
    const std::uint32_t small_w = std::min(w, 1920u), small_h = std::max(1u, static_cast<std::uint32_t>(std::uint64_t(h) * small_w / std::max(w, 1u)));
    if (sh->motion_found) {  // (also while it is not drawn: see Renderer::ingest)
        cam.mvec_scale[0] = sh->motion_scale_x;
        cam.mvec_scale[1] = sh->motion_scale_y;
    }
    p->on_constants(frame, cam);
    if (dev->get_api() == device_api::vulkan) {
        // (XPAR.fx's depth, R32F, left readable by shaders as every effect render target is - see copy_probe)
        if (desc.texture.format != format::r32_float) return;
        constexpr std::uint32_t kVkR32Float = 100, kVkColour = 1, kVkShaderReadOnly = 5;  // (VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                                                                                           // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        p->on_tag_vk(frame, kDepth, depth.handle, kVkR32Float, kVkColour, 0, 0, kVkShaderReadOnly, w, h, 0, 0, w, h, reinterpret_cast<void*>(cmd_list->get_native()));
        // (the add-on's copy of the target, as big as the game made it, in copy_source - see mv_probe. The presenter resamples it to the depth's size)
        if (search.vulkan.handle) {
            constexpr std::uint32_t kVkTransferSource = 6;  // (VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            const resource_desc md = dev->get_resource_desc(search.vulkan);
            const VkFormat format = vk_format_of(static_cast<DXGI_FORMAT>(md.texture.format));
            if (format != VK_FORMAT_UNDEFINED) {
                p->on_tag_vk(frame, search.motion.handle ? kMotion : kMotionProbe, search.vulkan.handle, static_cast<std::uint32_t>(format), kVkColour, 0, 0, kVkTransferSource,
                             md.texture.width, md.texture.height, 0, 0, md.texture.width, md.texture.height, reinterpret_cast<void*>(cmd_list->get_native()));
                p->set_motion_meta(frame, search.candidate, search.motion.handle ? (sh->motion_channel_x + 1) | ((sh->motion_channel_y + 1) << 8) : 0u);
            }
        }
    } else if (dev->get_api() == device_api::d3d11) {
        // (the immediate context: ReShade renders its effects there)
        p->on_tag_d3d11(frame, kDepth, reinterpret_cast<ID3D11Resource*>(depth.handle), 0, 0, w, h,
                        reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native()));
        const resource handed = search.motion.handle ? search.motion : search.probe;
        if (handed.handle) {
            if (p->on_motion_d3d11(frame, search.motion.handle ? kMotion : kMotionProbe, reinterpret_cast<ID3D11Resource*>(handed.handle), small_w, small_h,
                                   reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native())))
                p->set_motion_meta(frame, search.candidate, search.motion.handle ? (sh->motion_channel_x + 1) | ((sh->motion_channel_y + 1) << 8) : 0u);
        }
    } else {
        // ReShade leaves an effect's render targets readable by shaders between passes.
        const auto state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        p->on_tag(frame, kDepth, reinterpret_cast<ID3D12Resource*>(depth.handle), state, 0, 0, w, h,
                  reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native()));
        const resource handed = search.motion.handle ? search.motion : search.probe;
        p->want_motion_d3d12(search.motion.handle ? kMotion : kMotionProbe, search.id, small_w, small_h, search.candidate,
                             search.motion.handle ? (sh->motion_channel_x + 1) | ((sh->motion_channel_y + 1) << 8) : 0u);
    }
    // This frame's data is tagged inside its own present, on a command list not submitted yet: the present is told, or it takes the newest frame submitted, the one
    // before - every picture paired with the depth and motion of the frame before it (Enshrouded: colour fitted the next index's motion in 27 of 27 frames).
    {
        static MatrixCandidate matrices[kMatrixCandidates];
        if (const std::uint32_t count = camera_search::take(matrices, kMatrixCandidates, sh ? sh->camera_held : nullptr, sh ? sh->camera_held_state : 0, frame)) p->set_matrices(frame, matrices, count);
    }
    p->on_present_marker(frame);
    char text[200];
    if (far_only)
        std::snprintf(text, sizeof(text), "%s", choosing && depth_usable()
            ? "No DLSS or FSR data from this game: checking the depth buffer (camera turns only meanwhile)."
            : "No DLSS or FSR data and no usable depth buffer: XPAR warps camera turns only (walking and strafing move at the game's frame rate).");
    else if (depth_restart_needed())
        std::snprintf(text, sizeof(text), "No DLSS or FSR data from this game. Its depth reads empty: XPAR switched on ReShade's depth copy - restart the game once.");
    else
    {
        char camera[64];
        const LONG share = sh->camera_share;
        if (share > 0) std::snprintf(camera, sizeof(camera), "Camera found in the game (used on %ld%% of frames).", share);
        else std::snprintf(camera, sizeof(camera), "Camera estimated.");
        std::snprintf(text, sizeof(text), "Motion vectors from: %s. %s Depth from ReShade (%ux%u).",
                      search.motion.handle ? "the game's own render targets, found automatically"
                      : search.probe.handle ? "the picture, estimated by XPAR (looking for the game's own: turn the camera)"
                                            : "the picture, estimated by XPAR", camera, w, h);
    }
    set_status(text);
}

}  // namespace fw
