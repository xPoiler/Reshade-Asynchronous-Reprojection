// The fourth HUD pattern's steps on one back buffer, draw by draw (ui_layer.cpp): the picture drawn onto it by an overwriting
// draw, then the HUD alpha-blended straight onto it. At the first blended draw after the picture a HUD-less copy of the back
// buffer is taken - the HUD is what differs from it at the present. An overwriting draw after the copy by what drew the picture
// (the picture drawn again: a post-processing pass) gives the copy up for the frame; by anything else it is an opaque part of
// the HUD (Hi-Fi Rush draws one among its blended ones). Used by the Direct3D 11 path (dev/PLAN_d3d11_fourth_pattern.md), where
// "what drew" is the bound pixel shader - Direct3D 12's own path keeps its pipelines (drawn_back_buffer).
#pragma once
#include <cstdint>

namespace fw::fourth {

struct State {
    std::uint8_t step = 0;  // 0 nothing yet, 1 the picture drawn, 2 the copy taken, 3 given up
    std::uint64_t picture_by = 0;
};
enum class Act { None, Take, GiveUp };

// overwriting: the draw replaces what is there (no blending); blended: alpha-blended. Other blending (additive, unknown)
// changes nothing.
inline Act step(State& s, bool overwriting, bool blended, std::uint64_t by) {
    if (overwriting) {
        if (s.step == 2) {
            if (by == s.picture_by) { s.step = 3; return Act::GiveUp; }
            return Act::None;
        }
        if (s.step != 3) { s.step = 1; s.picture_by = by; }
        return Act::None;
    }
    if (blended && s.step == 1) { s.step = 2; return Act::Take; }
    return Act::None;
}

}  // namespace fw::fourth
