#pragma once
// Prevent GPU queueing ("Latency-aware frame cap" internally; on by default): holds the game back right after its present,
// a little below the rate its GPU sustains, so no frame waits in the GPU's queue and the next frame reads its
// input as late as possible (the idea of NVIDIA Reflex, for games without it). FrameWarp keeps the output at the
// display's rate, so a slightly lower game rate costs little.

#include <reshade.hpp>
#include <cstdint>

namespace fw {

struct FrameCapStatus {
    bool active = false;      // capping now (on, frames measured, no frame generation, no Reflex)
    bool reflex = false;      // the game's NVIDIA Reflex low-latency mode is on (the cap stands aside)
    bool gpu_bound = false;   // the cap stands aside: frames queued although it held the game back (the GPU is the limit)
    float cap_fps = 0;        // the cap
    float game_fps = 0;       // the game's presents per second
    float present_to_done_ms = 0;  // the GPU finishes a frame this long after its present (median)
    float start_to_done_ms = 0;    // ...and this long after the frame started (released by the cap)
    float wait_ms = 0;        // held back per frame (mean)
    float queued_pct = 0;     // frames that waited in the GPU's queue (the GPU still on the one before at their present)
};

// After every present of the game's swapchain (ReShade's finish_present). enabled: the option (and reprojection on);
// generation: the game's frame generation is presenting (never capped: its presents are not the game's frames).
// Never capped either while the game's own NVIDIA Reflex is on (it does the same).
void frame_cap_after_present(reshade::api::command_queue* queue, bool enabled, bool generation);
FrameCapStatus frame_cap_status();
// Present ceiling (always on, after every present of the game's swapchain): a game presenting faster than twice the
// refresh rate of the monitor its window is on - an uncapped menu, splash or loading screen, well over a thousand
// presents a second - is held to that rate. FrameWarp's work per present at such rates broke DOOM Eternal on a single
// NVIDIA card (its next texture view refused: out of device memory); frames that fast never reach the screen anyway.
void present_ceiling(void* window);  // (the HWND)
// The game's presents so far, and whether the GPU has done everything submitted up to present n (its fence, signalled after
// every present): for readbacks of work recorded during a frame - done once that frame's present is (n = the count when
// recorded + 1). Without the fence (Vulkan, or not set up): true.
std::uint64_t frame_cap_presents();
bool frame_cap_gpu_done(std::uint64_t present);
void frame_cap_shutdown();

}  // namespace fw
