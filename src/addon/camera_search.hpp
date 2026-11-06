// The game's side of the camera search (presenter/camera_search.hpp, [FrameWarp] CameraSearch=1, Direct3D 11 and 12; Vulkan through the game's mapped memory): every constant buffer the game fills or binds
// is looked through for what can be a camera's matrix, and those of this frame go to the presenter with the frame. Only read: nothing of the game's is changed.
#pragma once
#include "shared/protocol.hpp"

namespace fw { class Producer; }

namespace fw::camera_search {
void init();
void shutdown();
// While the game's camera and motion come from its upscaler (the ReShade feed off): nothing searched, the hooks return at once.
void pause();
void resume();
// The matrices seen since the last call, every value of the shape of a camera, in the order they came (the ReShade feed, once a frame).
std::uint32_t take(MatrixCandidate* out, std::uint32_t capacity, const volatile long* held, long held_state, std::uint64_t frame = 0);  // (held: 8 keys, the places that win; may be null. held_state: Shared::camera_held_state)
}
