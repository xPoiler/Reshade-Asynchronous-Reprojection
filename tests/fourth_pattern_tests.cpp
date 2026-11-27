// The fourth HUD pattern's steps (src/addon/fourth_pattern.hpp): what each draw onto a back buffer means.
#include "addon/fourth_pattern.hpp"
#include <cstdio>

using fw::fourth::Act;
using fw::fourth::State;
using fw::fourth::step;
static int failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); } } while (0)

int main() {
    const std::uint64_t picture = 11, hud = 22, opaque_hud = 33;
    {  // the picture, then the HUD blended: one copy, at the HUD's first draw
        State s;
        EXPECT(step(s, true, false, picture) == Act::None, "the picture: nothing taken");
        EXPECT(step(s, false, true, hud) == Act::Take, "the HUD's first blended draw: the copy taken");
        EXPECT(step(s, false, true, hud) == Act::None, "the HUD's next draws: no second copy");
    }
    {  // no HUD: the picture alone - nothing taken
        State s;
        step(s, true, false, picture);
        EXPECT(s.step == 1, "the picture drawn, waiting");
    }
    {  // an opaque part of the HUD among the blended ones: the copy kept
        State s;
        step(s, true, false, picture);
        step(s, false, true, hud);
        EXPECT(step(s, true, false, opaque_hud) == Act::None, "an opaque HUD part: the copy kept");
        EXPECT(s.step == 2, "still taken");
    }
    {  // the picture drawn again after the copy (a post-processing pass): given up, and no new copy that frame
        State s;
        step(s, true, false, picture);
        step(s, false, true, hud);
        EXPECT(step(s, true, false, picture) == Act::GiveUp, "the picture's shader again: given up");
        EXPECT(step(s, false, true, hud) == Act::None, "given up for the frame");
        EXPECT(step(s, true, false, picture) == Act::None, "and stays given up");
    }
    {  // a blended draw before any picture (onto last frame's contents): nothing
        State s;
        EXPECT(step(s, false, true, hud) == Act::None, "blended before the picture: nothing");
    }
    {  // other blending (additive, unknown) changes nothing
        State s;
        step(s, true, false, picture);
        EXPECT(step(s, false, false, hud) == Act::None && s.step == 1, "additive: still waiting for the HUD");
    }
    std::printf(failures ? "fourth_pattern_tests: %d FAILED\n" : "fourth_pattern_tests: all passed\n", failures);
    return failures ? 1 : 0;
}
