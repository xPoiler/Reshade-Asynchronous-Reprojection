// The game's HUD layer, where it draws its whole HUD into a texture of its own and blends it onto the picture at the end
// (Cyberpunk 2077: found with the UI probe). Recognised by what the game does, not by the game: a texture the size of
// the back buffer (not a back buffer), cleared to transparent black every frame, then drawn into as a render target -
// and at the end of the frame (a back buffer bound as the render target, or a copy onto a back buffer from a texture
// the game did not draw into: Assassin's Creed Black Flag Resynced composes picture + HUD layer with a compute pass and
// copies the result) the last such layer the game can read then (pixel or compute shaders) is copied into the frame's
// UI slot, as a UI layer a game tags for frame generation would be. Layers drawn without depth come first; one drawn
// with depth (Black Flag: its 3D markers) only in a frame that has no other. The copy is recorded on that command list only where
// the game changed the layer's state on that same list before (the state then known exactly); otherwise it waits.
// Second pattern - the HUD drawn straight onto the final picture (Unreal Engine: S.T.A.L.K.E.R. 2): the final picture
// is a texture the game copies onto the back buffer (Stalker 2: two, every other frame); once the frame's picture has been
// copied into it, the next time it is bound as a render target the HUD is about to be drawn - a copy of it then, on that
// command list, is the frame's HUD-less picture. Followed by the final picture's own steps, not by presents: Unreal
// records the next frame's command lists before it presents the last one. A final picture the game draws itself (no copy
// into it) is left alone. A frame with a HUD layer hands that over instead (both together would pass for a game's
// HUD-less picture + UI layer).
// Direct3D 12.
#pragma once
#include <cstdint>
#include <functional>

namespace reshade::api { struct swapchain; struct command_list; }

namespace fw::ui_layer {
// take(resource, D3D12 state, native command list, alpha inverted - a layer cleared to opaque black): records the copy into
// the frame's UI slot; take_hudless: the
// same for the HUD-less picture.
// take_vk (Vulkan games): the layer's VkImage, its format (ReShade's / DXGI), width, height, state (ReShade's usage) and
// the VkCommandBuffer. take_before (Direct3D 11, third pattern): the picture (ID3D11Resource), what happened (0 the HUD
// starts: copy it; 1 it reached the screen with no HUD this frame; 2 the copy confirmed: the final draw reads it; 3 the
// copy given up) and the context.
void init(std::function<void(std::uint64_t, std::uint32_t, std::uint64_t, bool)> take,
          std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> take_hudless,
          std::function<void(std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t)> take_vk,
          std::function<void(std::uint64_t, std::uint32_t, std::uint64_t)> take_before);
void shutdown();
// Every present of the game's swapchain: its back buffers and size; one frame ends.
void on_present(reshade::api::swapchain* chain);
}
