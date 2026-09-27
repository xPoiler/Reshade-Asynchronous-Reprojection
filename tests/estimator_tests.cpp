// CameraEstimator: recovers per-frame rotation and the field of view from motion vectors alone, and
// its synthesized clipToPrevClip maps each pixel to where its motion vector says it came from.
#include "presenter/camera_estimator.hpp"
#include <windows.h>
#include <cstdio>
#include <random>

using namespace fw;
static int failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static const double kW = 1920, kH = 1080, kPi = 3.14159265358979;

// Samples on a gw x gh grid for a camera rotating by `omega` (current -> previous camera coords) with a
// vertical field of view `fov` (`fov_prev` in the previous frame: zooming; 0 = the same); `outliers` of
// them get random motion (moving objects).
static std::vector<MotionSample> make(const CameraEstimator::V3& omega, double fov, double outliers, std::mt19937& rng,
                                      const CameraEstimator::V3& T = {0, 0, 0}, double fov_prev = 0, int gw = 64, int gh = 36) {
    const double f = (kH * 0.5) / std::tan(fov * 0.5), fp = (kH * 0.5) / std::tan((fov_prev > 0 ? fov_prev : fov) * 0.5);
    const auto R = CameraEstimator::rotation(omega);
    std::uniform_real_distribution<double> u(0, 1), noise(-0.1, 0.1), wild(-40, 40);
    std::vector<MotionSample> s;
    for (int gy = 0; gy < gh; ++gy)
        for (int gx = 0; gx < gw; ++gx) {
            MotionSample p{};
            p.x = float((gx + 0.5) * kW / gw); p.y = float((gy + 0.5) * kH / gh);
            const double X = (p.x - kW * 0.5) / f, Y = -(p.y - kH * 0.5) / f;
            // depth = 1 / distance: a mix of far scenery and near geometry (0.5 = two units away)
            p.depth = float(u(rng) < 0.5 ? 0.001 + 0.01 * u(rng) : 0.05 + 0.45 * u(rng));
            const double c0 = R[0][0] * X + R[0][1] * Y + R[0][2] + T[0] * p.depth, c1 = R[1][0] * X + R[1][1] * Y + R[1][2] + T[1] * p.depth,
                         c2 = R[2][0] * X + R[2][1] * Y + R[2][2] + T[2] * p.depth;
            p.mx = float(kW * 0.5 + fp * c0 / c2 - p.x + noise(rng));
            p.my = float(kH * 0.5 - fp * c1 / c2 - p.y + noise(rng));
            if (u(rng) < outliers) { p.mx = float(wild(rng)); p.my = float(wild(rng)); }
            p.valid = 1;
            s.push_back(p);
        }
    return s;
}

int main() {
    std::mt19937 rng(7);
    const double fov = 60 * kPi / 180;
    // Rotation fit with a known focal length.
    for (const CameraEstimator::V3 omega : {CameraEstimator::V3{0, 0.02, 0}, CameraEstimator::V3{0.01, -0.03, 0.002},
                                            CameraEstimator::V3{-0.05, 0.08, 0}}) {
        const auto s = make(omega, fov, 0.0, rng);
        CameraEstimator::V3 fit{};
        CameraEstimator::fit_rotation(s, kW, kH, (kH * 0.5) / std::tan(fov * 0.5), fit);
        double err = 0;
        for (int k = 0; k < 3; ++k) err = std::max(err, std::fabs(fit[k] - omega[k]));
        std::printf("rotation (%.3f %.3f %.3f) fitted (%.4f %.4f %.4f), error %.2e rad\n", omega[0], omega[1], omega[2], fit[0], fit[1], fit[2], err);
        EXPECT(err < 2e-4, "rotation recovered (error %g)", err);
    }
    // Rotation + translation (strafing, walking forward, orbiting) from all samples.
    for (const auto& case_ : {std::make_pair(CameraEstimator::V3{0, 0, 0}, CameraEstimator::V3{0.05, 0, 0}),
                              std::make_pair(CameraEstimator::V3{0, 0.01, 0}, CameraEstimator::V3{0, 0, -0.1}),
                              std::make_pair(CameraEstimator::V3{0.002, 0.02, 0}, CameraEstimator::V3{-0.06, 0.01, 0.02})}) {
        const auto s = make(case_.first, fov, 0.0, rng, case_.second);
        CameraEstimator::V3 o{}, t{};
        CameraEstimator::fit_rotation(s, kW, kH, (kH * 0.5) / std::tan(fov * 0.5), o);
        CameraEstimator::fit_motion(s, kW, kH, (kH * 0.5) / std::tan(fov * 0.5), o, t);
        double eo = 0, et = 0;
        for (int k = 0; k < 3; ++k) { eo = std::max(eo, std::fabs(o[k] - case_.first[k])); et = std::max(et, std::fabs(t[k] - case_.second[k])); }
        std::printf("rotation+translation: T (%.3f %.3f %.3f) fitted (%.4f %.4f %.4f), rotation error %.1e, translation error %.1e\n",
                    case_.second[0], case_.second[1], case_.second[2], t[0], t[1], t[2], eo, et);
        EXPECT(eo < 3e-4 && et < 3e-3, "rotation and translation recovered (%g, %g)", eo, et);
    }
    // Full estimator: learns the field of view from turning frames with 10% outliers, then tracks.
    CameraEstimator est;
    Camera game{};
    game.depth_inverted = 1;
    std::uniform_real_distribution<double> turn(-0.03, 0.03);
    for (int i = 0; i < 80; ++i) est.update(make({turn(rng) * 0.5, turn(rng), 0}, fov, 0.10, rng), kW, kH, game);
    std::printf("learned vertical FOV %.2f deg (true 60), locked %d\n", est.vertical_fov() * 180 / kPi, int(est.fov_locked()));
    EXPECT(std::fabs(est.vertical_fov() - fov) < 1.5 * kPi / 180, "field of view learned (%.2f deg)", est.vertical_fov() * 180 / kPi);
    EXPECT(est.fov_locked(), "field of view locks");
    const CameraEstimator::V3 omega{0.004, 0.025, 0};
    const CameraEstimator::V3 strafe{0.04, 0, 0.02};
    const auto s = make(omega, fov, 0.10, rng, strafe);
    const Camera cam = est.update(s, kW, kH, game);
    const auto got = est.last_omega();
    double err = 0;
    for (int k = 0; k < 3; ++k) err = std::max(err, std::fabs(got[k] - omega[k]));
    EXPECT(err < 5e-4, "rotation with outliers (error %g)", err);
    // clipToPrevClip must map each (inlier) pixel to where its motion vector says it came from.
    double worst = 0;
    for (const auto& p : make(omega, fov, 0.0, rng, strafe)) {
        const double ndc_x = p.x / kW * 2 - 1, ndc_y = 1 - p.y / kH * 2;
        const double clip[4] = {ndc_x, ndc_y, p.depth, 1};
        double prev[4] = {};
        for (int j = 0; j < 4; ++j) for (int i = 0; i < 4; ++i) prev[j] += clip[i] * cam.clip_to_prev_clip[i * 4 + j];
        const double px = (prev[0] / prev[3] * 0.5 + 0.5) * kW, py = (0.5 - prev[1] / prev[3] * 0.5) * kH;
        worst = std::max(worst, std::hypot(px - (p.x + p.mx), py - (p.y + p.my)));
    }
    std::printf("clipToPrevClip vs motion vectors: worst %.3f px\n", worst);
    EXPECT(worst < 1.0, "clipToPrevClip agrees with the motion vectors (worst %.3f px)", worst);
    // The same with standard depth (1 - near / distance, as some games hand FSR): the synthesized camera
    // keeps the game's convention.
    {
        CameraEstimator standard;
        Camera g = game;
        g.depth_inverted = 0;
        g.fov = float(fov);
        standard.update(make({0, 0, 0}, fov, 0.0, rng), kW, kH, g);
        const auto ss = make(omega, fov, 0.0, rng, strafe);
        const Camera sc = standard.update(ss, kW, kH, g);
        double worst_std = 0;
        for (const auto& p : ss) {
            const double clip[4] = {p.x / kW * 2 - 1, 1 - p.y / kH * 2, 1.0 - p.depth, 1};
            double prev[4] = {};
            for (int j = 0; j < 4; ++j) for (int i = 0; i < 4; ++i) prev[j] += clip[i] * sc.clip_to_prev_clip[i * 4 + j];
            const double px = (prev[0] / prev[3] * 0.5 + 0.5) * kW, py = (0.5 - prev[1] / prev[3] * 0.5) * kH;
            worst_std = std::max(worst_std, std::hypot(px - (p.x + p.mx), py - (p.y + p.my)));
        }
        std::printf("standard depth: clipToPrevClip worst %.3f px, depth_inverted %u\n", worst_std, sc.depth_inverted);
        EXPECT(worst_std < 1.0 && sc.depth_inverted == 0, "standard depth convention kept (worst %.3f px)", worst_std);
    }
    // Turning right (+yaw of the view, i.e. the world moves left on screen) turns the basis to the right.
    CameraEstimator yaw;
    const auto before = yaw.update(make({0, 0, 0}, fov, 0, rng), kW, kH, game);
    Camera after{};
    // omega maps current -> previous camera: +yaw means the current view points to the right of the
    // previous one (the image content moved left, motion vectors to the previous frame point right).
    for (int i = 0; i < 10; ++i) after = yaw.update(make({0, 0.02, 0}, fov, 0, rng), kW, kH, game);
    const double dot_right = after.fwd[0] * before.right[0] + after.fwd[1] * before.right[1] + after.fwd[2] * before.right[2];
    std::printf("after turning: forward . old right = %.3f (positive = turned right)\n", dot_right);
    EXPECT(dot_right > 0.1, "basis turns the way the image says");
    // Fast turn (7 degrees in one game frame) while moving, with motion vectors 4% off: the fit is less
    // exact at this speed but must still be used, translation included (not replaced by a still camera).
    {
        const CameraEstimator::V3 fast{0.01, 0.12, 0}, move{0.03, 0, 0.05};
        auto fs = make(fast, fov, 0.05, rng, move);
        std::normal_distribution<double> rel(0, 0.04);
        for (auto& p : fs) { const double k = 1 + rel(rng); p.mx = float(p.mx * k); p.my = float(p.my * k); }
        est.update(fs, kW, kH, game);
        const auto o = est.last_omega(), t = est.last_translation();
        std::printf("fast turn: rotation %.3f %.3f (true %.3f %.3f), translation %.3f %.3f %.3f (true %.3f %.3f %.3f), residual %.2f px\n",
                    o[0], o[1], fast[0], fast[1], t[0], t[1], t[2], move[0], move[1], move[2], est.last_residual());
        EXPECT(std::fabs(o[1] - fast[1]) < 0.005, "fast turn rotation kept (%g)", o[1]);
        EXPECT(std::fabs(t[0] - move[0]) < 0.015 && std::fabs(t[2] - move[2]) < 0.015, "fast turn translation kept");
    }
    // The game's field of view is checked against the picture: taken as vertical, as horizontal (some games
    // hand FSR the horizontal one), or not used when it matches neither.
    {
        const double true_v = 45 * kPi / 180, true_h = 2 * std::atan(std::tan(true_v * 0.5) * kW / kH);
        const struct { double told; int mode; const char* what; } cases[] = {
            {true_v, 1, "vertical"}, {true_h, 2, "horizontal"}, {100 * kPi / 180, 3, "wrong"}};
        for (const auto& c : cases) {
            CameraEstimator e;
            Camera g = game;
            g.fov = float(c.told);
            for (int i = 0; i < 80; ++i) e.update(make({turn(rng) * 0.5, turn(rng), 0}, true_v, 0.10, rng), kW, kH, g);
            e.update(make({0.002, 0.01, 0}, true_v, 0.10, rng), kW, kH, g);  // the decision applies from the next frame
            std::printf("game tells the %s field of view: mode %d, using %.2f deg vertical (true 45)\n", c.what, e.game_fov_mode(),
                        e.vertical_fov() * 180 / kPi);
            EXPECT(e.game_fov_mode() == c.mode, "%s field of view recognised (mode %d)", c.what, e.game_fov_mode());
            EXPECT(std::fabs(e.vertical_fov() - true_v) < 2 * kPi / 180, "%s field of view: right one used (%.2f deg)", c.what,
                   e.vertical_fov() * 180 / kPi);
        }
    }
    // Zooming in while standing still, with the game telling the field of view every frame (FSR): the
    // change of field of view is not movement (no translation, no rotation), and the synthesized
    // clipToPrevClip still lands on the motion vectors.
    {
        CameraEstimator zoom;
        Camera g = game;
        double worst_t = 0, worst_o = 0, worst_px = 0, max_res = 0;
        double prev = 74 * kPi / 180;
        for (int i = 0; i < 30; ++i) {
            const double cur = (74 - 0.3 * i) * kPi / 180;
            g.fov = float(cur);
            const auto zs = make({0, 0, 0}, cur, 0.0, rng, {0, 0, 0}, i ? prev : 0);
            const Camera c = zoom.update(zs, kW, kH, g);
            prev = cur;
            if (i < 2) continue;
            const auto o = zoom.last_omega(), t = zoom.last_translation();
            worst_t = std::max(worst_t, std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]));
            worst_o = std::max(worst_o, std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]));
            max_res = std::max(max_res, zoom.last_residual());
            for (const auto& p : zs) {
                if (p.depth > 0.02f) continue;  // far samples (no translation to resolve)
                const float cx = 2 * p.x / float(kW) - 1, cy = 1 - 2 * p.y / float(kH);
                const float v[4] = {cx, cy, p.depth, 1};
                float r[4] = {};
                for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) r[j] += v[k] * c.clip_to_prev_clip[k * 4 + j];
                const double px = (r[0] / r[3] * 0.5 + 0.5) * kW, py = (0.5 - r[1] / r[3] * 0.5) * kH;
                worst_px = std::max(worst_px, std::hypot(px - (p.x + p.mx), py - (p.y + p.my)));
            }
        }
        std::printf("zoom: worst translation %.4f, rotation %.4f, clipToPrevClip error %.2f px, residual %.2f px\n", worst_t, worst_o, worst_px, max_res);
        EXPECT(max_res > 0 && max_res < 0.5, "zoom frames are fitted, not rejected (%.2f px)", max_res);
        EXPECT(worst_t < 0.003 && worst_o < 0.0005, "zooming is not taken for movement");
        EXPECT(worst_px < 1.0, "clipToPrevClip follows the zoom (%.2f px)", worst_px);
    }
    // CPU cost per game frame (the presenter runs this on every new frame): 80x45 samples (the presenter's
    // grid), 10% outliers, turning and moving.
    {
        std::vector<std::vector<MotionSample>> frames;
        for (int i = 0; i < 50; ++i) frames.push_back(make({turn(rng) * 0.5, turn(rng), 0}, fov, 0.10, rng, {turn(rng), 0, turn(rng)}, 0, 80, 45));
        LARGE_INTEGER f0, a, b; QueryPerformanceFrequency(&f0); QueryPerformanceCounter(&a);
        for (const auto& fr : frames) est.update(fr, kW, kH, game);
        QueryPerformanceCounter(&b);
        const double ms = double(b.QuadPart - a.QuadPart) * 1000.0 / double(f0.QuadPart) / frames.size();
        std::printf("estimator CPU: %.3f ms per game frame\n", ms);
        EXPECT(ms < 1.5, "estimator is cheap enough per game frame (%.3f ms)", ms);
    }
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("estimator tests passed\n");
    return 0;
}
