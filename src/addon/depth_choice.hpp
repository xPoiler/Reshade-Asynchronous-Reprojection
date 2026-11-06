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
// Depth buffers the game bound per frame, over the last 60 frames (while track_depth_buffers runs): none in a menu or a
// video (Dark Souls), and a scene's count differs by game - 11-15 in Dark Souls' world - so not a test of a scene by itself.
double depth_binds_per_frame();
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
// Whether the depth ReShade hands over belongs to the picture: a buffer chosen here or ticked by the user (or a
// Generic Depth this cannot steer: its own choice, as before). False while none stands out - nothing in use,
// or none whose depth lies on the picture's edges (Cyber Hook: every one reads empty or unrelated).
bool depth_usable();
// The size the scene's depth is drawn at - the game's render resolution - when the buffer in use was chosen here: the smallest of the buffers
// holding the same scene (a game can keep it at its render resolution and at the picture's too). False otherwise.
bool scene_depth_size(std::uint32_t& w, std::uint32_t& h);
// ReShade's depth copy setting was switched on (the depth read empty): the game has to be restarted once.
bool depth_restart_needed();
// The feed starts (600 frames of a 3D scene): the choice made again, among the buffers of the scene now shown. One made while the game
// loads can be judged on a picture that is not there yet - Enshrouded, fading in from black: its 1280x720 depth probed on the black
// picture (brightness changes 0.00x), a 640x360 one with 1.4% edges probed once it showed and chosen, kept the whole session.
void choose_depth_again(reshade::api::effect_runtime* runtime);
// The feed stopped (an upscaler gives depth again, XPAR disabled): Generic Depth's own choice again.
void release_depth_choice(reshade::api::effect_runtime* runtime);
}
