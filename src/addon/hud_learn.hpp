// Learning, at runtime, where a game draws its HUD - instead of a rule per way games do it (ui_layer.hpp's patterns).
// Every game has a texture with its finished picture that reaches the screen (the back buffer itself, or a texture
// copied or drawn onto it); the HUD is what is drawn into it after some point of the frame. That point is learned:
// - recorded (any Direct3D version): which texture reaches the back buffer, and the draws into it and into the back
//   buffer, in groups of consecutive draws by one pixel shader - its fingerprint (a hash of its code: the same every
//   frame and every launch) names the group;
// - tried: a candidate is "a copy of the picture just before the first draw of group X"; one at a time, a few frames
//   each, one copy per frame. The add-on compares it with the final picture at the present (where they differ: what
//   was drawn after that point) and hands that over for scoring only (kHudProbe, SlotMeta::hud_candidate);
// - scored by the presenter: the HUD stays in place on the screen while the camera moves, what is still the scene
//   moves with it.
// - used: once every candidate has been tried and the presenter picks one (scored on enough frames, changing a small
//   part of the screen, staying on it clearly more than moving with the scenery: the earliest point after which only
//   the HUD is drawn), it is locked: every frame the same comparison, handed over as the frame's HUD (a coverage, as
//   a game's own HUD layer - only where nothing else gave the HUD). A frame where that group does not draw shows no
//   HUD. The presenter keeps scoring it; one that stops behaving like a HUD is dropped and learning starts again (as
//   with the panel's Relearn).
#pragma once
#include <cstdint>
#include <functional>
#include <string>

namespace reshade::api { struct swapchain; }

namespace fw::hud_learn {
// probe(picture, d3d11, D3D12 state, native list / immediate context, candidate, end): copy the picture now, for candidate.
// end: Direct3D 12, a candidate in the final picture - the final picture as the game draws it onto the back buffer, to
// compare the first copy with (the back buffer is not the same picture: Elden Ring's draw onto it changes all of it).
// use: the learned point's (its comparison is the frame's HUD). none(): the learned point did not draw this frame.
// (state bit 31, Direct3D 12: a copy in the final picture - compared only with its end copy, never with the back buffer)
using Probe = std::function<void(std::uint64_t, bool, std::uint32_t, std::uint64_t, std::uint32_t, bool, bool)>;
void init(Probe probe, std::function<void()> none);
void shutdown();
// Every present of the game's swapchain: its back buffers; one frame ends. pick / fail: the presenter's (see
// PresenterStatus); relearn: the panel's counter.
void on_present(reshade::api::swapchain* chain, bool learning, std::uint32_t pick, std::uint32_t fail, std::uint32_t relearn);
struct Status { std::uint32_t candidates = 0, active = 0, locked = 0; };
// The learned point is saved once it has given the HUD for a minute of frames, and locked from the start on the next
// launch (the panel's Relearn forgets it).
void set_profile(const std::wstring& path);
Status status();
}
