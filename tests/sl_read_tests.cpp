// Reading a Streamline game's hand-over (src/addon/sl_read.hpp): its constants (a camera, none, or not at this offset) and the
// depth's area.
#include "addon/sl_read.hpp"
#include <cstdio>

using fw::sl::Area;
using fw::sl::Constants;
static int failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); } } while (0)

namespace {
void put(std::uint8_t* raw, int off, float f) { std::memcpy(raw + off, &f, 4); }
// sl::Constants as a game writes them (after the BaseStructure): a camera, or every matrix and camera vector at INVALID_FLOAT.
void constants(std::uint8_t* raw, bool camera, float mv_scale = 1.0f) {
    std::memset(raw, 0, fw::sl::kConstantsBytes);
    for (int off = 0; off < 320; off += 4) put(raw, off, camera ? (off % 20 == 0 ? 1.0f : 0.0f) : FLT_MAX);
    put(raw, 320, 0.25f); put(raw, 324, -0.25f);  // jitter
    put(raw, 328, mv_scale); put(raw, 332, mv_scale);
    if (camera) {
        put(raw, 344, 5.0f);                           // pos
        put(raw, 356 + 4, 1.0f);                       // up 0 1 0
        put(raw, 368, 1.0f);                           // right 1 0 0
        put(raw, 380 + 8, 1.0f);                       // fwd 0 0 1
    } else {
        for (int off = 344; off < 392; off += 4) put(raw, off, FLT_MAX);
    }
    put(raw, 392, 0.1f); put(raw, 396, 1000.0f); put(raw, 400, 1.0f); put(raw, 404, 1.78f);
    raw[412] = 1; raw[413] = 1;
}
}  // namespace

int main() {
    std::uint8_t raw[fw::sl::kConstantsBytes + 32];
    fw::Camera cam{};
    {  // a camera
        constants(raw, true);
        EXPECT(fw::sl::read_constants(raw, cam) == Constants::Camera, "a camera read as one");
        EXPECT(cam.estimated == 0 && cam.depth_inverted == 1 && cam.pos[0] == 5.0f, "its values");
    }
    {  // no camera: estimated, with what the constants still say
        constants(raw, false, 0.5f);
        EXPECT(fw::sl::read_constants(raw, cam) == Constants::NoCamera, "INVALID_FLOAT matrices and vectors: no camera");
        EXPECT(cam.estimated == 1 && cam.mvec_scale[0] == 0.5f && cam.jitter[0] == 0.25f && cam.depth_inverted == 1, "jitter, scale, depth direction kept");
    }
    {  // the wrong base offset: neither a camera nor a "no camera"
        std::uint8_t shifted[fw::sl::kConstantsBytes + 32];
        constants(raw, true);
        std::memset(shifted, 0, sizeof(shifted));
        std::memcpy(shifted + 32, raw, fw::sl::kConstantsBytes);
        EXPECT(fw::sl::read_constants(shifted, cam) == Constants::NotHere, "a camera read 32 bytes early: not here");
        constants(raw, false);
        std::memset(shifted, 0, sizeof(shifted));
        std::memcpy(shifted + 32, raw, fw::sl::kConstantsBytes);
        EXPECT(fw::sl::read_constants(shifted, cam) == Constants::NotHere, "no camera read 32 bytes early: not here");
        EXPECT(fw::sl::read_constants(shifted + 32, cam) == Constants::NoCamera, "...and at the right offset: no camera");
    }
    {  // no camera but a broken vector scale: not taken for constants
        constants(raw, false, 0.0f);
        EXPECT(fw::sl::read_constants(raw, cam) == Constants::NotHere, "zero vector scale: not constants");
    }
    {  // the depth's area
        const Area ext{64, 32, 640, 360};
        Area a = fw::sl::depth_area(1280, 720, &ext, 1280, 720, &ext);
        EXPECT(a.x == 64 && a.y == 32 && a.w == 640 && a.h == 360, "its extent when tagged");
        a = fw::sl::depth_area(1280, 720, nullptr, 640, 360, nullptr);
        EXPECT(a.x == 0 && a.y == 0 && a.w == 640 && a.h == 360, "no extent, bigger than the vectors: their size at the top left");
        a = fw::sl::depth_area(640, 360, nullptr, 640, 360, nullptr);
        EXPECT(a.w == 640 && a.h == 360, "no extent, the vectors' size: the whole");
        a = fw::sl::depth_area(1280, 720, nullptr, 0, 0, nullptr);
        EXPECT(a.w == 1280 && a.h == 720, "no vectors: the whole");
        a = fw::sl::depth_area(640, 360, nullptr, 1280, 720, nullptr);
        EXPECT(a.w == 640 && a.h == 360, "vectors bigger than the depth: the whole depth");
        const Area outside{700, 0, 640, 360};
        a = fw::sl::depth_area(1280, 720, &outside, 1280, 720, nullptr);
        EXPECT(a.x == 0 && a.w == 1280, "an extent outside the texture: not taken");
        const Area mext{0, 0, 960, 540};
        a = fw::sl::depth_area(1280, 720, nullptr, 1280, 720, &mext);
        EXPECT(a.w == 960 && a.h == 540, "no depth extent: the vectors' extent's size");
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
