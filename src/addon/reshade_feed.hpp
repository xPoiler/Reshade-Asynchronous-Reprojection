#pragma once
#include "addon/producer.hpp"
#include <reshade_api.hpp>

namespace fw {
// Games without DLSS or FSR (D3D12): depth from ReShade's depth buffer, through XPAR.fx (an effect the
// add-on enables itself; its one output is the depth as XPAR wants it). Published as the frame's depth,
// without motion vectors and with a camera to estimate: the presenter estimates the motion from the
// picture (Renderer::estimate_motion) and the camera from that. Only while nothing else gives depth and
// motion vectors: a frame an upscaler gives them (a DLSS or FSR call, Streamline depth) ends the feed
// at once, and it comes back 600 frames after the last one (the upscaler switched off in the game).
void install_reshade_feed(Producer* producer);
// addon_event::reshade_finish_effects / reshade_reloaded_effects.
void on_reshade_finish_effects(reshade::api::effect_runtime* runtime, reshade::api::command_list* cmd_list,
                               reshade::api::resource_view rtv, reshade::api::resource_view rtv_srgb);
void on_reshade_reloaded_effects(reshade::api::effect_runtime* runtime);
// For the panel: what the feed is doing, or what it is missing ("" while the game's own data is used).
const char* reshade_feed_status();
}
