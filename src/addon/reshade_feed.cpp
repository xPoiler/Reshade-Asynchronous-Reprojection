#include "addon/reshade_feed.hpp"
#include "addon/depth_choice.hpp"
#include "addon/ngx_hooks.hpp"
#include <reshade.hpp>
#include <d3d12.h>
#include <cctype>
#include <cstdio>
#include <cstring>

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

void install_reshade_feed(Producer* producer) { g_producer = producer; }
const char* reshade_feed_status() { return g.status; }

void on_reshade_reloaded_effects(effect_runtime*) { g.scanned = false; }  // (every handle from before is gone)

void on_reshade_finish_effects(effect_runtime* runtime, command_list* cmd_list, resource_view, resource_view) {
    Producer* p = g_producer;
    if (!p || !p->ready() || p->vulkan() || !runtime || !cmd_list) return;
    device* dev = runtime->get_device();
    if (!dev || (dev->get_api() != device_api::d3d12 && !(dev->get_api() == device_api::d3d11 && p->d3d11()))) return;
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
    if (wanted) track_depth_buffers(); else release_depth_choice(runtime);
    if (!wanted || ++g.waited < 600) {
        g.status[0] = 0;
        if (!wanted) g.waited = 0;
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
    const bool oriented = usable && depth_orientation_ready(runtime, cmd_list, probe_view, g.flip);
    const bool far_only = !oriented;
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
    p->on_constants(frame, cam);
    if (dev->get_api() == device_api::d3d11) {
        // (the immediate context: ReShade renders its effects there)
        p->on_tag_d3d11(frame, kDepth, reinterpret_cast<ID3D11Resource*>(depth.handle), 0, 0, w, h,
                        reinterpret_cast<ID3D11DeviceContext*>(cmd_list->get_native()));
    } else {
        // ReShade leaves an effect's render targets readable by shaders between passes.
        const auto state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        p->on_tag(frame, kDepth, reinterpret_cast<ID3D12Resource*>(depth.handle), state, 0, 0, w, h,
                  reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native()));
    }
    char text[200];
    if (far_only)
        std::snprintf(text, sizeof(text), "%s", choosing && depth_usable()
            ? "No DLSS or FSR data from this game: checking the depth buffer (camera turns only meanwhile)."
            : "No DLSS or FSR data and no usable depth buffer: XPAR warps camera turns only (walking and strafing move at the game's frame rate).");
    else if (depth_restart_needed())
        std::snprintf(text, sizeof(text), "No DLSS or FSR data from this game. Its depth reads empty: XPAR switched on ReShade's depth copy - restart the game once.");
    else
        std::snprintf(text, sizeof(text), "No DLSS or FSR data from this game: depth from ReShade (%ux%u), motion and camera estimated by XPAR.", w, h);
    set_status(text);
}

}  // namespace fw
