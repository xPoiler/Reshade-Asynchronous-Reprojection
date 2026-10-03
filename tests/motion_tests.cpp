#include "shared/camera_motion.hpp"
#include "shared/camera_check.hpp"
#include "addon/depth_probe_score.hpp"
#include "presenter/pose.hpp"
#include <random>
#include <vector>
#include <cstdio>

using namespace fw;
static int failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static M4 mm(const M4& a, const M4& b) {
    M4 r{};
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) r[i][j] += a[i][k] * b[k][j];
    return r;
}

static void recovers(double yaw, double pitch, V3 t, double near_plane) {
    // E33-like reversed-Z infinite projection with TAA jitter (row-vector).
    const float P[16] = {1.1667f, 0, 0, 0, 0, 2.0741f, 0, 0, 0.00039f, 0.00108f, 0, 1, 0, 0, float(near_plane), 0};
    // Column-form rotation R = Ry(yaw) * Rx(pitch) mapping current view -> previous view.
    const double cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
    const R3 Ry{{{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}}}, Rx{{{1, 0, 0}, {0, cp, -sp}, {0, sp, cp}}};
    R3 R{};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) for (int k = 0; k < 3; ++k) R[i][j] += Ry[i][k] * Rx[k][j];
    M4 Mrow{};  // row-vector: v_prev = v_cur * Mrow  (upper 3x3 = R^T, last row = t)
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Mrow[i][j] = R[j][i];
    for (int j = 0; j < 3; ++j) Mrow[3][j] = t[j];
    Mrow[3][3] = 1;
    const M4 Pm = m4_from(P);
    M4 Pinv; invert(Pm, Pinv);
    const M4 C = mm(mm(Pinv, Mrow), Pm);
    float Cf[16];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) Cf[i * 4 + j] = float(C[i][j]);
    const FrameMotion m = recover_motion(P, Cf);
    EXPECT(m.valid, "valid");
    double rerr = 0, terr = 0;
    for (int i = 0; i < 3; ++i) { for (int j = 0; j < 3; ++j) rerr = std::max(rerr, std::fabs(m.rotation[i][j] - R[i][j])); terr = std::max(terr, std::fabs(m.translation[i] - t[i])); }
    EXPECT(rerr < 1e-4, "rotation error %g (yaw %g pitch %g)", rerr, yaw, pitch);
    EXPECT(terr < 0.05 + 0.002 * std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]), "translation error %g (t %g %g %g)", terr, t[0], t[1], t[2]);
}

// Frame generation: an image generated between two frames is seen from the frame's camera moved part of the way
// back along the frame's own motion (generated_to_target). All the way back it is the previous frame's camera:
// a point where the previous frame saw it lands where this frame sees it; no way back, nothing changes.
static void generated_camera(double yaw, V3 t) {
    const float P[16] = {1.1667f, 0, 0, 0, 0, 2.0741f, 0, 0, 0.00039f, 0.00108f, 0, 1, 0, 0, 10.0f, 0};
    const double cy = std::cos(yaw), sy = std::sin(yaw);
    M4 Mrow{};
    Mrow[0][0] = cy; Mrow[0][2] = -sy; Mrow[1][1] = 1; Mrow[2][0] = sy; Mrow[2][2] = cy;  // (R^T of a yaw)
    for (int j = 0; j < 3; ++j) Mrow[3][j] = t[j];
    Mrow[3][3] = 1;
    const M4 Pm = m4_from(P);
    M4 Pinv; invert(Pm, Pinv);
    const M4 C = mm(mm(Pinv, Mrow), Pm);
    float Cf[16];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) Cf[i * 4 + j] = float(C[i][j]);
    const Mat4 identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const Mat4 back = generated_to_target(Cf, 1.0, identity), none = generated_to_target(Cf, 0.0, identity);
    double worst = 0, worst_none = 0;
    for (double x : {-0.6, 0.0, 0.6})
        for (double y : {-0.5, 0.3})
            for (double z : {10.0 / 100.0, 10.0 / 2000.0}) {  // (reversed depth: near / distance)
                const std::array<double, 4> cur{x, y, z, 1.0};
                std::array<double, 4> prev{};
                for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) prev[j] += cur[k] * C[k][j];
                std::array<double, 4> again{}, same{};
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 4; ++k) { again[j] += prev[k] * back[k * 4 + j]; same[j] += cur[k] * none[k * 4 + j]; }
                for (int j = 0; j < 2; ++j) {
                    worst = std::max(worst, std::fabs(again[j] / again[3] - x * (j == 0) - y * (j == 1)));
                    worst_none = std::max(worst_none, std::fabs(same[j] / same[3] - x * (j == 0) - y * (j == 1)));
                }
            }
    std::printf("generated image camera (yaw %.3f, move %g %g %g): all the way back off by %.5f, no way back %.5f\n", yaw, t[0], t[1], t[2], worst,
                worst_none);
    EXPECT(worst < 2e-3, "all the way back is the previous frame's camera (%g)", worst);
    EXPECT(worst_none < 1e-5, "no way back is the frame's camera (%g)", worst_none);
}

// The same with a game's own matrices (Resident Evil Requiem, RE Engine: right-handed, reversed depth with a far
// plane): all the way back, a point where the previous frame saw it lands where this frame sees it.
static void generated_camera_game() {
    const float P[16] = {1.45260334f, 0, 0, 0, 0, 2.58240604f, 0, 0, 0, 0, 1.56164169e-05f, -1, 0, 0, 0.0626075864f, 0};
    const float C[16] = {1.00079298f, -0.000647655746f, 3.45222304e-08f, -0.00219388981f, 0.000187643425f, 1.00075185f, 5.86583759e-08f,
                         -0.0037277434f, -0.0730272681f, -0.293101788f, 1.00324225f, -0.012904115f, 0.00463666068f, 0.0248720516f,
                         -6.77646312e-08f, 0.999948263f};
    const Mat4 identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const Mat4 g = generated_to_target(C, 1.0, identity), half = generated_to_target(C, 0.5, identity);
    Mat4 half_inv{};
    mat_inverse(half, half_inv);  // (this frame's clip space -> the half-way image's)
    double worst = 0, worst_half = 0;
    for (double d : {0.5, 2.0, 10.0, 100.0})
        for (double x : {-0.5, 0.5}) {
            const double cur[4] = {x, 0.3, 0.0626075864 / d - 1.56164169e-05, 1};
            double prev[4] = {}, again[4] = {};
            for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) prev[j] += cur[k] * C[k * 4 + j];
            for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) again[j] += prev[k] * g[k * 4 + j];
            const double ex = std::fabs(again[0] / again[3] - x), ey = std::fabs(again[1] / again[3] - 0.3);
            worst = std::max({worst, ex, ey});
            // Half way back, the point is seen about half way between where the two frames see it.
            double mid[4] = {};
            for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) mid[j] += cur[k] * half_inv[k * 4 + j];
            worst_half = std::max({worst_half, std::fabs(mid[0] / mid[3] - 0.5 * (x + prev[0] / prev[3])),
                                   std::fabs(mid[1] / mid[3] - 0.5 * (0.3 + prev[1] / prev[3]))});
        }
    std::printf("  game camera: all the way back off by %.5f, half way off the midpoint by %.5f\n", worst, worst_half);
    EXPECT(worst < 2e-3, "game camera: all the way back is the previous frame's camera (%g)", worst);
    EXPECT(worst_half < 2e-3, "game camera: half way back sees points half way (%g)", worst_half);
}

// A zoom between the frames (aiming: the FOV narrows) with a still camera must not read as movement.
static void zoom_is_not_motion(double zoom, V3 t, double near_change = 1.0) {
    const float P[16] = {1.1667f, 0, 0, 0, 0, 2.0741f, 0, 0, 0, 0, 0, 1, 0, 0, 1.0f, 0};
    float Pp[16];
    for (int i = 0; i < 16; ++i) Pp[i] = P[i];
    Pp[0] = float(P[0] / zoom); Pp[5] = float(P[5] / zoom);  // previous frame: wider field of view
    Pp[14] = float(P[14] / near_change);                      // ... and (RE9 when aiming) another near plane
    M4 Mrow{};
    for (int i = 0; i < 4; ++i) Mrow[i][i] = 1;
    for (int j = 0; j < 3; ++j) Mrow[3][j] = t[j];
    M4 Pinv; invert(m4_from(P), Pinv);
    const M4 C = mm(mm(Pinv, Mrow), m4_from(Pp));  // clip_prev = clip_cur * Pinv_cur * M * P_prev
    float Cf[16];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) Cf[i * 4 + j] = float(C[i][j]);
    const FrameMotion with_prev = recover_motion(P, Cf, Pp), without = recover_motion(P, Cf);
    double err = 0, naive = 0;
    for (int i = 0; i < 3; ++i) { err = std::max(err, std::fabs(with_prev.translation[i] - t[i])); naive = std::max(naive, std::fabs(without.translation[i] - t[i])); }
    std::printf("zoom %.2f, move (%g %g %g): translation error %.4f with the previous projection, %.1f without\n", zoom, t[0], t[1], t[2], err, naive);
    EXPECT(with_prev.valid && err < 0.05, "zoom does not read as motion (error %g)", err);
}

// The check of the game's own camera against its motion vectors (camera_check.hpp). One game frame's sums
// for motion vectors g = c / scale + noise over `n` pixels with camera motion of `motion` uv rms.
static void sums(double motion, double scale, double noise, double n, double gc[2], double gg[2], double cc[2]) {
    for (int k = 0; k < 2; ++k) {
        const double c2 = motion * motion * n * (k ? 0.25 : 1.0);  // (mostly a turn: less vertical motion)
        cc[k] = c2;
        gc[k] = c2 / scale;
        gg[k] = c2 / (scale * scale) + noise * noise * n / (scale * scale);
    }
}
static void camera_check_tests() {
    double gc[2], gg[2], cc[2];
    // A game whose camera is right: the motion vectors follow it (a moving character adds a little noise).
    {
        CameraCheck check;
        sums(0.01, 1.0 / 1920, 0.002, 2e6, gc, gg, cc);
        for (int i = 0; i < 400; ++i) check.add(gc, gg, cc, 2e6, false);
        EXPECT(check.verdict == CameraCheck::kUndecided, "a consistent camera is never found unusable (%d)", int(check.verdict));
        EXPECT(check.add(gc, gg, cc, 2e6, true) == CameraCheck::kTrusted, "trusted once the motion vector scale locks");
        sums(0.01, 1.0, 1.0, 2e6, gc, gg, cc);
        for (int i = 0; i < 400; ++i) check.add(gc, gg, cc, 2e6, false);
        EXPECT(check.verdict == CameraCheck::kTrusted, "a trusted camera stays trusted");
    }
    // A camera that stands still, and frames with too few pixels, say nothing - however long.
    {
        CameraCheck check;
        sums(0.0002, 1.0, 1.0, 2e6, gc, gg, cc);
        for (int i = 0; i < 2000; ++i) check.add(gc, gg, cc, 2e6, false);
        sums(0.01, 1.0, 1.0, 500, gc, gg, cc);
        for (int i = 0; i < 2000; ++i) check.add(gc, gg, cc, 500, false);
        EXPECT(check.verdict == CameraCheck::kUndecided && check.frames == 0, "a still camera decides nothing (%d frames)", check.frames);
    }
    // A game whose reprojection matrix is not the camera's motion: the motion vectors have nothing to do with it.
    {
        CameraCheck check;
        sums(0.3, 1.0, 1.0, 2e6, gc, gg, cc);
        gc[0] *= 0.05; gc[1] *= -0.05;
        int frames = 0;
        while (check.add(gc, gg, cc, 2e6, false) == CameraCheck::kUndecided && frames < 1000) ++frames;
        EXPECT(check.verdict == CameraCheck::kUnusable && frames == CameraCheck::kWindow - 1, "an inconsistent camera is found unusable after one window (%d frames)", frames);
    }
    // A messy stretch (a cutscene with large moving things) in a game whose camera is right: one frame in
    // five still fits, so the window starts over instead of deciding.
    {
        CameraCheck check;
        for (int i = 0; i < 600; ++i) {
            sums(0.01, 1.0, i % 5 == 0 ? 0.002 : 0.05, 2e6, gc, gg, cc);
            check.add(gc, gg, cc, 2e6, false);
        }
        EXPECT(check.verdict == CameraCheck::kUndecided, "a messy stretch does not condemn a camera that fits now and then (%d)", int(check.verdict));
    }
}

// Which depth buffer belongs to the picture: a probe of a scene (ground receding into the distance, two
// nearer boxes that are also drawn differently in the picture) against the same picture with the right
// depth, the depth turned round, another view's depth, only part of the scene and an empty buffer.
static void depth_probe_tests() {
    const std::uint32_t w = 256, h = 144;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> noise(-0.03f, 0.03f);
    auto in_box = [](int x, int y, int k) { return k == 0 ? (x > 40 && x < 90 && y > 50 && y < 120) : (x > 150 && x < 200 && y > 30 && y < 80); };
    std::vector<float> right(w * h * 2), turned(w * h * 2), other(w * h * 2), partial(w * h * 2), empty(w * h * 2);
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x) {
            const std::size_t i = (std::size_t(y) * w + x) * 2;
            const bool box = in_box(int(x), int(y), 0) || in_box(int(x), int(y), 1);
            const float ground = y < 40 ? 0.0f : 0.002f + 0.02f * float(y - 40) / float(h - 40);  // sky above
            const float depth = box ? 0.08f : ground;
            const float luma = (box ? 0.25f : y < 40 ? 0.8f : 0.55f) + noise(rng);
            right[i] = depth; right[i + 1] = luma;
            turned[i] = 1.0f - depth; turned[i + 1] = luma;
            // (another view: the same scene mirrored left to right)
            const bool mbox = in_box(int(w - 1 - x), int(y), 0) || in_box(int(w - 1 - x), int(y), 1);
            other[i] = mbox ? 0.08f : ground; other[i + 1] = luma;
            partial[i] = in_box(int(x), int(y), 0) ? 0.08f : 0.0f; partial[i + 1] = luma;
            empty[i] = 0.0f; empty[i + 1] = luma;
        }
    const auto r = fw::score_depth_probe(right.data(), w, h), t = fw::score_depth_probe(turned.data(), w, h), o = fw::score_depth_probe(other.data(), w, h),
               p = fw::score_depth_probe(partial.data(), w, h), e = fw::score_depth_probe(empty.data(), w, h);
    std::printf("depth probe: right %.2f, turned round %.2f, another view %.2f, part of the scene %.2f, empty %.2f\n", r.score, t.score, o.score, p.score, e.score);
    EXPECT(r.score > 2.0, "the right depth stands out (%.2f)", r.score);
    EXPECT(t.score > 0.8 * r.score, "whichever way round it is stored (%.2f against %.2f)", t.score, r.score);
    EXPECT(o.score < 0.5 * r.score && p.score < 0.5 * r.score && e.score == 0.0, "another view's, a partial and an empty one do not (%.2f, %.2f, %.2f)", o.score, p.score, e.score);
    EXPECT(e.flat && !r.flat && !p.flat && e.detail > 0.1, "an empty buffer is told apart (one value everywhere) in a picture with detail (%.2f)", e.detail);
}

int main() {
    depth_probe_tests();
    camera_check_tests();
    generated_camera(0.04, {0, 0, 0});
    generated_camera(-0.02, {12, 0, 30});
    generated_camera_game();
    zoom_is_not_motion(1.06, {0, 0, 0});
    zoom_is_not_motion(0.94, {0, 0, -30});
    zoom_is_not_motion(1.10, {5, 0, 20});
    zoom_is_not_motion(1.06, {0, 0, 0}, 1.1);    // RE9 aim: zoom + near plane change, camera still
    zoom_is_not_motion(0.94, {0, 0, -3}, 0.9);   // ... while walking backwards
    recovers(0, 0, {0, 0, 0}, 1.0);
    recovers(0.02, -0.01, {0, 0, 0}, 1.0);
    recovers(0.03, 0.015, {5.0, -2.0, 1.5}, 1.0);     // orbit-like step (cm)
    recovers(-0.05, 0.02, {-20.0, 3.0, -8.0}, 10.0);  // other near plane
    recovers(0.001, 0.0, {0.0, 0.0, 12.0}, 1.0);      // walking forward
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("motion tests passed\n");
    return 0;
}
