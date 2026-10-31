#include "presenter/motion_search.hpp"
#include <cstdio>
#include <random>
using namespace fw;

static int g_failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++g_failures; std::printf("FAILED line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// A grid moving smoothly, a character's patch that does not follow, and a target's channels around it.
struct Scene {
    std::mt19937 rng{7};
    float noise(float a) { return std::normal_distribution<float>(0, a)(rng); }
    std::vector<MotionSearch::Sample> frame(int t, int channel_x, int channel_y, float scale_x, float scale_y, bool real) {
        std::vector<MotionSearch::Sample> out;
        const float turn = 14.0f * std::sin(t * 0.4f) + (std::sin(t * 0.4f) >= 0 ? 6.0f : -6.0f), rise = 5.0f * std::cos(t * 0.3f);
        for (int y = 0; y < 45; ++y)
            for (int x = 0; x < 80; ++x) {
                const bool character = x > 30 && x < 45 && y > 15;
                const float depth = 1.0f / (1.0f + x * 0.05f);
                const float rx = (character ? 0.0f : turn + (x - 40) * 0.2f * depth) + noise(0.4f), ry = (character ? 0.0f : rise + (y - 22) * 0.2f * depth) + noise(0.4f);
                MotionSearch::Sample s{};
                s.rx = rx; s.ry = ry; s.valid = 1;
                for (int c = 0; c < 4; ++c) s.c[c] = 0.5f + 0.4f * std::sin(x * 0.3f + c) * std::cos(y * 0.2f + t * 0.05f);  // (albedo, normals: no motion)
                if (real) {
                    s.c[channel_x] = (rx + noise(0.3f)) / scale_x;
                    s.c[channel_y] = (ry + noise(0.3f)) / scale_y;
                }
                out.push_back(s);
            }
        return out;
    }
};

static void finds_any_layout() {
    const struct { int cx, cy; float sx, sy; } kCases[] = {{1, 2, 1920, -1080}, {0, 1, 3840, 2160}, {2, 3, 960, 540}, {3, 0, 1.0f, -1.0f}, {1, 0, 1920, 1080}};
    for (const auto& k : kCases) {
        MotionSearch search;
        Scene scene;
        for (int t = 0; t < 30; ++t) {
            search.add(5, scene.frame(t, 0, 1, 1, 1, false));  // (a G-buffer's normals)
            search.add(9, scene.frame(t, k.cx, k.cy, k.sx, k.sy, true));
        }
        const auto f = search.decide();
        EXPECT(f.found && f.candidate == 9, "layout ch%d ch%d: found %d candidate %u", k.cx, k.cy, int(f.found), f.candidate);
        EXPECT(f.channel_x == k.cx && f.channel_y == k.cy, "channels %d %d, expected %d %d", f.channel_x, f.channel_y, k.cx, k.cy);
        EXPECT(std::fabs(f.scale_x / k.sx - 1) < 0.1f && std::fabs(f.scale_y / k.sy - 1) < 0.1f, "scales %g %g, expected %g %g", f.scale_x, f.scale_y, k.sx, k.sy);
    }
}
static void finds_nothing_without_motion() {
    MotionSearch search;
    Scene scene;
    for (int t = 0; t < 60; ++t) { search.add(1, scene.frame(t, 0, 1, 1, 1, false)); search.add(2, scene.frame(t + 100, 0, 1, 1, 1, false)); }
    EXPECT(!search.decide().found, "a target without motion taken for motion");
}
static void one_channel_is_not_enough() {
    MotionSearch search;
    Scene scene;
    for (int t = 0; t < 30; ++t) {
        auto f = scene.frame(t, 0, 1, 1920, 1080, true);
        for (auto& s : f) s.c[1] = 0.25f;  // (x alone)
        search.add(3, f);
    }
    EXPECT(!search.decide().found, "one axis alone taken for motion");
}
static void still_frames_do_not_count() {
    MotionSearch search;
    Scene scene;
    for (int t = 0; t < 30; ++t) {
        auto f = scene.frame(t, 0, 1, 1920, 1080, true);
        for (auto& s : f) { s.rx *= 0.01f; s.ry *= 0.01f; }
        EXPECT(!search.add(3, f), "a still frame counted");
    }
    EXPECT(!search.decide().found && search.frames(3) == 0, "decided on still frames");
}
static void needs_enough_frames() {
    MotionSearch search;
    Scene scene;
    for (int t = 0; t < MotionSearch::kFramesNeeded - 1; ++t) search.add(4, scene.frame(t, 0, 1, 1920, 1080, true));
    EXPECT(!search.decide().found, "decided too early");
}

// A noisy picture holds the true target under the pass mark: it still stands far above the others.
static MotionSearch::Found noisy_search(float noise_x, float noise_y) {
    MotionSearch search;
    Scene scene;
    std::mt19937 rng{3};
    std::normal_distribution<float> noise(0, 1);
    for (int t = 0; t < 120; ++t) {
        auto f = scene.frame(t, 0, 1, 1920, 1080, true);
        for (auto& s : f) { s.rx += noise_x * noise(rng); s.ry += noise_y * noise(rng); }
        search.add(9, f);
        auto g = scene.frame(t, 0, 1, 1, 1, false);
        if (t < 24) { search.add(1, g); search.add(2, g); }
    }
    return search.decide();
}
static void far_above_the_others() {
    const auto r = noisy_search(1.0f, 0.6f);
    EXPECT(r.found && r.candidate == 9 && r.channel_x == 0 && r.channel_y == 1, "the target far above the others not taken: found %d candidate %u quality %g", int(r.found), r.candidate, r.quality);
    const auto n = noisy_search(2.2f, 2.2f);  // (a frame's own fit drowns in this, the frames' medians do not: it is the true target)
    EXPECT(n.found && n.candidate == 9 && n.channel_x == 0 && n.channel_y == 1, "the true target under heavy noise: found %d candidate %u channels %d %d", int(n.found), n.candidate, n.channel_x, n.channel_y);
}

// Plain turns: the whole picture moves alike, a frame's own fit sees only noise. Across frames the target follows them, both ways.
static std::vector<MotionSearch::Sample> plain_turn(Scene& scene, int t, bool real, float drift) {
    std::vector<MotionSearch::Sample> out;
    const float turn = 16.0f * std::sin(t * 0.25f), rise = 9.0f * std::cos(t * 0.17f);
    for (int y = 0; y < 45; ++y)
        for (int x = 0; x < 80; ++x) {
            MotionSearch::Sample s{};
            s.rx = turn + scene.noise(1.0f); s.ry = rise + scene.noise(1.0f); s.valid = 1;
            for (int c = 0; c < 4; ++c) s.c[c] = 0.5f + 0.4f * std::sin(x * 0.3f + c) * std::cos(y * 0.2f + t * 0.05f) + drift * t;
            if (real) { s.c[0] = (turn + scene.noise(0.3f)) / 1920; s.c[1] = (rise + scene.noise(0.3f)) / 1080; }
            out.push_back(s);
        }
    return out;
}
static void plain_turns() {
    MotionSearch search;
    Scene scene;
    for (int t = 0; t < 60; ++t) search.add(9, plain_turn(scene, t, true, 0));
    const auto r = search.decide();
    EXPECT(r.found && r.candidate == 9 && r.channel_x == 0 && r.channel_y == 1, "plain turns not followed: found %d candidate %u channels %d %d", int(r.found), r.candidate, r.channel_x, r.channel_y);
    EXPECT(std::fabs(r.scale_x / 1920 - 1) < 0.1f && std::fabs(r.scale_y / 1080 - 1) < 0.1f, "scales %g %g", r.scale_x, r.scale_y);
}
// A turn one way, and channels that drift along with it: no proof of anything.
static void one_way_is_not_enough() {
    MotionSearch search;
    Scene scene;
    for (int t = 0; t < 60; ++t) {
        auto f = plain_turn(scene, t, false, 0.004f);
        for (auto& s : f) { s.rx += -16.0f * std::sin(t * 0.25f) + 8.0f + t * 0.3f; s.ry += -9.0f * std::cos(t * 0.17f) + 6.0f + t * 0.2f; }
        search.add(4, f);
    }
    EXPECT(!search.decide().found, "a drift taken for motion");
}
static void no_motion_through_plain_turns() {
    MotionSearch search;
    Scene scene;
    for (int t = 0; t < 120; ++t) search.add(4, plain_turn(scene, t, false, 0));
    EXPECT(!search.decide().found, "a target without motion followed the turns");
}

// Motion that reaches XPAR a frame late follows the turn of the frame before: said, and still found (it is all there may be).
static void late_motion_is_told_and_found() {
    MotionSearch search;
    Scene scene;
    std::vector<MotionSearch::Sample> previous = plain_turn(scene, 0, true, 0);
    for (int t = 1; t < 80; ++t) {
        auto f = plain_turn(scene, t, true, 0);
        auto stale = f;
        for (std::size_t i = 0; i < f.size(); ++i) { stale[i].c[0] = previous[i].c[0]; stale[i].c[1] = previous[i].c[1]; }
        search.add(7, stale);
        previous = f;
    }
    EXPECT(search.late(7), "late motion not told");
    MotionSearch on_time;
    Scene again;
    for (int t = 0; t < 80; ++t) on_time.add(9, plain_turn(again, t, true, 0));
    EXPECT(!on_time.late(9), "this frame's motion told as late");
    EXPECT(on_time.decide().found, "this frame's motion not found");
}
// A target tried over a turn one way shows nothing yet: not ruled out.
static void not_ruled_out_before_turns_both_ways() {
    MotionSearch search;
    Scene scene;
    for (int t = 0; t < 30; ++t) {
        auto f = plain_turn(scene, t, false, 0);
        for (auto& s : f) { s.rx = 12.0f + scene.noise(1.0f); s.ry = scene.noise(1.0f); }
        search.add(3, f);
    }
    EXPECT(!search.ruled_out(3), "ruled out on a turn one way");
    MotionSearch both;
    for (int t = 0; t < 120; ++t) both.add(3, plain_turn(scene, t, false, 0));
    EXPECT(both.ruled_out(3), "a target without motion not ruled out after turns both ways");
}
static void whole_units() {
    const struct { float x, y, want_x, want_y; } kCases[] = {{1774, 1073, 1920, 1080}, {1917, 968, 1920, 1080}, {1478, 1078.5f, 1920, 1080}, {955.9f, -540.4f, 960, -540}, {-1082.8f, -1078.7f, -1080, -1080}, {1500, 300, 1500, 300}};
    for (const auto& k : kCases) {
        MotionSearch::Found f;
        f.scale_x = k.x; f.scale_y = k.y;
        MotionSearch::to_units(f, 1920, 1080);
        EXPECT(f.scale_x == k.want_x && f.scale_y == k.want_y, "%g x %g became %g x %g", k.x, k.y, f.scale_x, f.scale_y);
    }
}

int main() {
    whole_units();
    not_ruled_out_before_turns_both_ways();
    late_motion_is_told_and_found();
    plain_turns();
    one_way_is_not_enough();
    no_motion_through_plain_turns();
    far_above_the_others();
    finds_any_layout();
    finds_nothing_without_motion();
    one_channel_is_not_enough();
    still_frames_do_not_count();
    needs_enough_frames();
    if (g_failures) { std::printf("%d failed\n", g_failures); return 1; }
    std::printf("motion search ok\n");
    return 0;
}
