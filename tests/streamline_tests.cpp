// sl::Constants without a camera, and where the frame lies in tagged textures (addon/streamline_layout.hpp).
#include "addon/streamline_layout.hpp"
#include <cfloat>
#include <cstdio>
#include <cstring>

static int failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

using namespace fw;
using namespace fw::sl_layout;

namespace {

void put(std::uint8_t* raw, std::size_t at, float v) { std::memcpy(raw + at, &v, 4); }

// Constants as a game gives them that sets only what DLSS needs: every matrix and camera vector sl::INVALID_FLOAT.
void constants_without_camera(std::uint8_t* raw) {
    std::memset(raw, 0, kConstantsBytes);
    for (std::size_t at = kViewToClip; at < kJitter; at += 4) put(raw, at, FLT_MAX);
    for (std::size_t at = kPos; at < kNear; at += 4) put(raw, at, FLT_MAX);
    put(raw, kJitter, 0.25f); put(raw, kJitter + 4, -0.125f);
    put(raw, kMvecScale, -1.0f); put(raw, kMvecScale + 4, -1.0f);
    put(raw, kNear, 0.2f); put(raw, kFar, 750.0f); put(raw, kFov, 1.309f); put(raw, kAspect, 1.778f);
    raw[kDepthInverted] = 0;
    raw[kDepthInverted + 1] = 1;  // (cameraMotionIncluded)
}

bool all_zero(const float* v, int n) {
    for (int i = 0; i < n; ++i) if (v[i] != 0.0f) return false;
    return true;
}

void test_unset_values() {
    EXPECT(!is_set(FLT_MAX), "FLT_MAX is unset");
    EXPECT(!is_set(-FLT_MAX), "-FLT_MAX is unset");
    EXPECT(!is_set(std::nanf("")), "NaN is unset");
    EXPECT(is_set(0.0f) && is_set(-1.0f) && is_set(750.0f), "ordinary values are set");
}

void test_estimated_camera() {
    std::uint8_t raw[kConstantsBytes];
    constants_without_camera(raw);
    Camera cam{};
    EXPECT(estimated_camera(raw, cam), "constants with jitter and a motion vector scale give an estimated camera");
    EXPECT(cam.valid == 1 && cam.estimated == 1, "valid %u estimated %u", cam.valid, cam.estimated);
    EXPECT(all_zero(cam.view_to_clip, 16) && all_zero(cam.clip_to_view, 16) && all_zero(cam.clip_to_prev_clip, 16), "unset matrices are zero");
    EXPECT(all_zero(cam.pos, 3) && all_zero(cam.up, 3) && all_zero(cam.right, 3) && all_zero(cam.fwd, 3), "unset camera vectors are zero");
    EXPECT(cam.jitter[0] == 0.25f && cam.jitter[1] == -0.125f, "jitter %f %f", cam.jitter[0], cam.jitter[1]);
    EXPECT(cam.mvec_scale[0] == -1.0f && cam.mvec_scale[1] == -1.0f, "motion vector scale %f %f", cam.mvec_scale[0], cam.mvec_scale[1]);
    EXPECT(cam.near_plane == 0.2f && cam.far_plane == 750.0f && cam.fov == 1.309f, "near %f far %f fov %f", cam.near_plane, cam.far_plane, cam.fov);
    EXPECT(cam.depth_inverted == 0 && cam.reset == 0, "flags: inverted %u reset %u", cam.depth_inverted, cam.reset);

    raw[kDepthInverted] = 1; raw[kReset] = 1;
    put(raw, kFov, FLT_MAX);
    put(raw, kMvecScale + 4, 0.0f);
    EXPECT(estimated_camera(raw, cam), "still a camera");
    EXPECT(cam.depth_inverted == 1 && cam.reset == 1, "flags: inverted %u reset %u", cam.depth_inverted, cam.reset);
    EXPECT(cam.fov == 0.0f, "an unset field of view is zero (%f)", cam.fov);
    EXPECT(cam.mvec_scale[1] == 1.0f, "a motion vector scale of 0 counts as 1 (%f)", cam.mvec_scale[1]);

    constants_without_camera(raw);
    put(raw, kJitter, FLT_MAX);
    EXPECT(!estimated_camera(raw, cam), "no jitter: no camera");
    constants_without_camera(raw);
    put(raw, kMvecScale + 4, FLT_MAX);
    EXPECT(!estimated_camera(raw, cam), "no motion vector scale: no camera");
}

void test_scale_to_pixels() {
    Camera cam{};
    cam.mvec_scale[0] = -1.0f; cam.mvec_scale[1] = -1.0f;
    scale_to_pixels(cam, 2560, 1440);
    EXPECT(cam.mvec_scale[0] == -2560.0f && cam.mvec_scale[1] == -1440.0f, "scale %f %f", cam.mvec_scale[0], cam.mvec_scale[1]);
}

void test_tag_size() {
    EXPECT((tag_size({}, 2560, 1440) == Rect{0, 0, 2560, 1440}), "no extent: the texture");
    EXPECT((tag_size({8, 4, 1280, 720}, 2560, 1440) == Rect{0, 0, 1280, 720}), "an extent: its size");
}

void test_depth_area() {
    // Depth made at the output size, the frame drawn into its top left at the render size, no extent tagged.
    EXPECT((depth_area(3840, 2160, {}, 2560, 1440) == Rect{0, 0, 2560, 1440}), "top left at the render size");
    EXPECT((depth_area(3840, 2160, {}, 3840, 2160) == Rect{0, 0, 3840, 2160}), "rendered at full size: all of it");
    EXPECT((depth_area(2560, 1440, {}, 2560, 1440) == Rect{0, 0, 2560, 1440}), "depth at the render size: all of it");
    EXPECT((depth_area(3840, 2160, {}, 0, 0) == Rect{0, 0, 3840, 2160}), "render size unknown: all of it");
    EXPECT((depth_area(2560, 1440, {}, 3840, 2160) == Rect{0, 0, 2560, 1440}), "render size larger than the depth: all of it");
    EXPECT((depth_area(3840, 2160, {}, 3840, 1440) == Rect{0, 0, 3840, 1440}), "smaller in one direction only");
    // A tagged extent wins, wherever it is.
    EXPECT((depth_area(3840, 2160, {100, 50, 2560, 1440}, 2560, 1440) == Rect{100, 50, 2560, 1440}), "the extent, offset kept");
    EXPECT((depth_area(3840, 2160, {3000, 1000, 2560, 1440}, 0, 0) == Rect{3000, 1000, 840, 1160}), "an extent past the edge is cut to the texture");
    EXPECT((depth_area(3840, 2160, {4000, 0, 100, 100}, 2560, 1440) == Rect{0, 0, 2560, 1440}), "an extent outside the texture is ignored");
}

}  // namespace

int main() {
    test_unset_values();
    test_estimated_camera();
    test_scale_to_pixels();
    test_tag_size();
    test_depth_area();
    if (failures) std::printf("%d failure(s)\n", failures);
    else std::printf("streamline layout: all passed\n");
    return failures ? 1 : 0;
}
