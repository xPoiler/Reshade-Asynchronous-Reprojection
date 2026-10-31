// Motion buffer probe (development, [FrameWarp] MotionProbe=1): does the game keep its own motion vectors somewhere XPAR could read,
// in a game that hands none to an upscaler? On Ctrl+Shift+V two game frames in a row are recorded to FrameWarp\captures: every
// render target of a two-channel format as it was left after its draws, those of a format that can hold motion among other things, the back buffer of each frame, and a list of all
// the render targets bound. dev/mv_probe_score.py warps one back buffer with each candidate: one that lines it up with the other
// frame is the game's motion. Nothing is done with the game's frames otherwise: without the key the events are not registered.
#pragma once
#include <reshade.hpp>

namespace fw { class Producer; }

namespace fw::mv_probe {
void init();
void shutdown();
// (development, [FrameWarp] MotionSearch=1) The motion search's side in the game, asked once a frame by the ReShade feed: every large render target of
// a format that can hold motion is tried in turn - `probe`, the presenter judges it (presenter/motion_search.hpp) - until `motion` is the
// one found, with nothing assumed about formats, channels or units. 0 both: nothing to hand over this frame.
// lost: the target found is gone or no longer drawn: the search has to start over.
// vulkan: that target as the game finished writing it this frame, in the add-on's own image, in copy_source (0: not written since the last call).
struct MotionStep { reshade::api::resource motion{0}, probe{0}, vulkan{0}; std::uint32_t candidate = 0, id = 0; bool lost = false; };
MotionStep motion_step(bool found, std::uint32_t found_candidate, std::uint32_t min_width, const std::uint64_t* ruled_out);
// The feed is off: nothing is noted or copied until the next motion_step.
void pause();
void set_producer(Producer* producer);
// Vulkan: images of this format and these uses must be made copyable when they are created, or they could not be tried.
bool wants_copyable(reshade::api::format format, reshade::api::resource_usage usage);
}
