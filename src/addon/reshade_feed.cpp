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
    if (!dev || dev->get_api() != device_api::d3d12) return;
    // The game's own depth and motion vectors (an upscaler's, Streamline's) always come first: the feed is
    // what there is when nothing else gives them (a game without DLSS or FSR, or with both switched off, at
    // the start or in the middle of a session), and it is out from the first frame an upscaler gives. It
    // starts after 600 frames without any, so that a game whose upscaler is on does not see it while it
    // loads or in a short pause; what was learned from the one source is dropped for the other (presenter).
    const bool wanted = p->shared()->settings.enabled && !p->game_depth_seen();
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
    auto* depth = reinterpret_cast<ID3D12Resource*>(dev->get_resource_from_view(depth_view).handle);
    auto* list = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native());
    if (!depth || !list) return;
    const D3D12_RESOURCE_DESC desc = depth->GetDesc();
    const std::uint32_t w = static_cast<std::uint32_t>(desc.Width), h = desc.Height;
    // Which of the game's depth buffers ReShade hands over (depth_choice.hpp): while that is being found out,
    // this frame's depth is a candidate's and is not published.
    resource_view probe_view{0}, bound_view{0};
    if (g.probe.handle) runtime->get_texture_binding(g.probe, &probe_view, &unused);
    if (g.bound.handle) runtime->get_texture_binding(g.bound, &bound_view, &unused);
    if (choose_depth_buffer(runtime, cmd_list, probe_view, bound_view) == DepthChoice::kChoosing) {
        set_status("No DLSS or FSR data from this game: finding the game's depth buffer...");
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
    // ReShade leaves an effect's render targets readable by shaders between passes.
    const auto state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    p->on_tag(frame, kDepth, depth, state, 0, 0, w, h, list);
    char text[200];
    std::snprintf(text, sizeof(text), "No DLSS or FSR data from this game: depth from ReShade (%ux%u), motion and camera estimated by XPAR.", w, h);
    set_status(text);
}

}  // namespace fw
