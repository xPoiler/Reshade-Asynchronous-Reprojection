#pragma once
#include "addon/depth_probe_score.hpp"
#include <reshade_api.hpp>
#include <cstdint>
#include <vector>

namespace fw {
// Which of the game's depth buffers is the scene's (games without DLSS or FSR: reshade_feed.hpp).
//
// ReShade's Generic Depth add-on picks the depth buffer with the most draw calls, which in some games is
// not the scene's; its list lets the user tick another, and forgets that at the end of the session. This
// does the ticking: each depth buffer the game draws into is selected in turn (the same override the
// checkbox sets), and the one whose depth belongs to the picture is kept - the one whose depth edges lie
// on edges of the picture, over the largest part of it (XPAR.fx hands over a small probe of both).
// Generic Depth keeps doing the rest (copies before clears, its own settings). Without Generic Depth, or
// when nothing stands out (a menu), its own choice stays and the question is asked again later.

void register_depth_choice_events();
void unregister_depth_choice_events();

// Once per frame from when the feed is wanted (its wait included): the depth buffers the game binds are
// noted from then on, so that they are known when the feed starts.
void track_depth_buffers();
enum class DepthChoice { kUnavailable, kChoosing, kSettled };
// Once per frame while the feed runs (reshade_finish_effects), with XPAR.fx's probe texture and the
// texture ReShade binds the depth to (0: unknown). kChoosing: the depth of this frame is a candidate's,
// not to be used.
DepthChoice choose_depth_buffer(reshade::api::effect_runtime* runtime, reshade::api::command_list* cmd_list,
                                reshade::api::resource_view probe, reshade::api::resource_view bound_depth);
// Which way round the depth is stored (ReShade's RESHADE_DEPTH_INPUT_IS_REVERSED does not have to match the
// game): checked on XPAR.fx's probe and corrected through its XPAR_Flip uniform. False until known; the
// depth of such frames is not to be used.
bool depth_orientation_ready(reshade::api::effect_runtime* runtime, reshade::api::command_list* cmd_list,
                             reshade::api::resource_view probe, reshade::api::effect_uniform_variable flip_variable);
// ReShade's depth copy setting was switched on (the depth read empty): the game has to be restarted once.
bool depth_restart_needed();
// The feed stopped (an upscaler gives depth again, XPAR disabled): Generic Depth's own choice again.
void release_depth_choice(reshade::api::effect_runtime* runtime);
}
