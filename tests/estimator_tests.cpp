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

// Samples on a 64x36 grid for a camera rotating by `omega` (current -> previous camera coords) with a
// vertical field of view `fov`; `outliers` of them get random motion (moving objects).
static std::vector<MotionSample> make(const CameraEstimator::V3& omega, double fov, double outliers, std::mt19937& rng,
                                      const CameraEstimator::V3& T = {0, 0, 0}) {
    const double f = (kH * 0.5) / std::tan(fov * 0.5);
    const auto R = CameraEstimator::rotation(omega);
    std::uniform_real_distribution<double> u(0, 1), noise(-0.1, 0.1), wild(-40, 40);
    std::vector<MotionSample> s;
    for (int gy = 0; gy < 36; ++gy)
        for (int gx = 0; gx < 64; ++gx) {
            MotionSample p{};
            p.x = float((gx + 0.5) * kW / 64); p.y = float((gy + 0.5) * kH / 36);
            const double X = (p.x - kW * 0.5) / f, Y = -(p.y - kH * 0.5) / f;
            // depth = 1 / distance: a mix of far scenery and near geometry (0.5 = two units away)
            p.depth = float(u(rng) < 0.5 ? 0.001 + 0.01 * u(rng) : 0.05 + 0.45 * u(rng));
            const double c0 = R[0][0] * X + R[0][1] * Y + R[0][2] + T[0] * p.depth, c1 = R[1][0] * X + R[1][1] * Y + R[1][2] + T[1] * p.depth,
                         c2 = R[2][0] * X + R[2][1] * Y + R[2][2] + T[2] * p.depth;
            p.mx = float(kW * 0.5 + f * c0 / c2 - p.x + noise(rng));
            p.my = float(kH * 0.5 - f * c1 / c2 - p.y + noise(rng));
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
    // CPU cost per game frame (the presenter runs this on every new frame): 64x36 samples, 10% outliers,
    // turning and moving.
    {
        std::vector<std::vector<MotionSample>> frames;
        for (int i = 0; i < 50; ++i) frames.push_back(make({turn(rng) * 0.5, turn(rng), 0}, fov, 0.10, rng, {turn(rng), 0, turn(rng)}));
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
