#pragma once
// Camera estimation for games that give DLSS depth and motion vectors but no camera (DLSS without
// Streamline). A camera rotation moves distant pixels in a pattern that depends on the rotation and
// on the focal length, so both can be fitted from a grid of motion-vector samples. The estimator keeps
// an integrated camera orientation and synthesizes the Camera the rest of the presenter expects
// (basis, projection, clipToPrevClip, position). Translation comes from parallax: with the depth
// buffer (reversed-Z, depth = 1 / distance in the synthesized projection's units), a camera move shifts
// near samples more than far ones, so rotation and translation are fitted together.
#include "shared/protocol.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace fw {

struct MotionSample {
    float x, y;    // render pixel position (centre of the sampled pixel)
    float mx, my;  // motion to the previous frame, render pixels (x right, y down)
    float depth;   // raw depth (reversed-Z: smaller = farther)
    float valid;
};

class CameraEstimator {
public:
    using V3 = std::array<double, 3>;
    using M3 = std::array<V3, 3>;  // rows

    // Rotation (current camera -> previous camera, camera coords x right, y up, z forward) that best
    // explains the samples' motion for focal length f (pixels). Returns the mean squared residual (px^2).
    static double fit_rotation(const std::vector<MotionSample>& s, double w, double h, double f, V3& omega) {
        auto residual = [&](const V3& o) {
            const M3 R = rotation(o);
            double sum = 0; int n = 0;
            for (const auto& p : s) {
                if (p.valid < 0.5f) continue;
                const double X = (p.x - w * 0.5) / f, Y = -(p.y - h * 0.5) / f;
                const V3 c = mul(R, V3{X, Y, 1.0});
                if (c[2] <= 1e-6) { sum += 1e6; ++n; continue; }
                const double px = w * 0.5 + f * c[0] / c[2], py = h * 0.5 - f * c[1] / c[2];
                const double ex = px - (p.x + p.mx), ey = py - (p.y + p.my);
                sum += ex * ex + ey * ey; ++n;
            }
            return n ? sum / n : 1e30;
        };
        // Linear start (small rotations), then Gauss-Newton with a numeric Jacobian on the exact model.
        omega = linear_rotation(s, w, h, f);
        double best = residual(omega);
        for (int it = 0; it < 4; ++it) {
            double JtJ[3][3] = {}, Jtr[3] = {};
            const double eps = 1e-5;
            // Build J^T J and J^T r from per-sample residual vectors.
            const M3 R0 = rotation(omega);
            M3 Rd[3];
            for (int k = 0; k < 3; ++k) { V3 o = omega; o[k] += eps; Rd[k] = rotation(o); }
            for (const auto& p : s) {
                if (p.valid < 0.5f) continue;
                const double X = (p.x - w * 0.5) / f, Y = -(p.y - h * 0.5) / f;
                auto project = [&](const M3& R, double& px, double& py) {
                    const V3 c = mul(R, V3{X, Y, 1.0});
                    const double z = std::max(c[2], 1e-6);
                    px = w * 0.5 + f * c[0] / z; py = h * 0.5 - f * c[1] / z;
                };
                double px, py; project(R0, px, py);
                const double r[2] = {px - (p.x + p.mx), py - (p.y + p.my)};
                double J[2][3];
                for (int k = 0; k < 3; ++k) {
                    double qx, qy; project(Rd[k], qx, qy);
                    J[0][k] = (qx - px) / eps; J[1][k] = (qy - py) / eps;
                }
                for (int a = 0; a < 2; ++a)
                    for (int i = 0; i < 3; ++i) {
                        Jtr[i] += J[a][i] * r[a];
                        for (int j = 0; j < 3; ++j) JtJ[i][j] += J[a][i] * J[a][j];
                    }
            }
            V3 step{};
            if (!solve3(JtJ, Jtr, step)) break;
            V3 next{omega[0] - step[0], omega[1] - step[1], omega[2] - step[2]};
            const double e = residual(next);
            if (!(e < best)) break;
            omega = next; best = e;
        }
        return best;
    }

    // Rotation and translation together (current -> previous camera: p_prev = R p_cur + T). For a sample
    // at direction c and depth d (= 1 / distance): p_prev / distance = R c + T d, so the sky (d = 0) only
    // constrains the rotation. Starts from `omega`; returns the mean squared residual (px^2).
    // f: this frame's focal length (px), fp: the previous frame's (0: the same). A zoom between the two
    // (the game told us both fields of view) is then part of the model instead of looking like movement.
    static double fit_motion(const std::vector<MotionSample>& s, double w, double h, double f, V3& omega, V3& T, double fp = 0) {
        if (fp <= 0) fp = f;
        // Gauss-Newton with the exact Jacobian (one projection per sample and iteration). The rotation is
        // updated multiplicatively: R <- exp(delta) R, for which d(R c)/d(delta) = -[R c]x.
        M3 R = rotation(omega);
        auto evaluate = [&](const M3& Rm, const V3& t, double A[6][6], double g[6]) {
            double sum = 0; int n = 0;
            for (const auto& p : s) {
                if (p.valid < 0.5f) continue;
                const double X = (p.x - w * 0.5) / f, Y = -(p.y - h * 0.5) / f, d = p.depth;
                const V3 c = mul(Rm, V3{X, Y, 1.0});
                const double qx = c[0] + t[0] * d, qy = c[1] + t[1] * d, qz = std::max(c[2] + t[2] * d, 1e-6);
                const double px = w * 0.5 + fp * qx / qz, py = h * 0.5 - fp * qy / qz;
                const double r[2] = {px - (p.x + p.mx), py - (p.y + p.my)};
                sum += r[0] * r[0] + r[1] * r[1]; ++n;
                if (!A) continue;
                // d(px, py)/dq
                const double iz = 1.0 / qz;
                const double dq[2][3] = {{fp * iz, 0, -fp * qx * iz * iz}, {0, -fp * iz, fp * qy * iz * iz}};
                // dq/d(delta) = -[c]x  (c = R X), dq/dT = d * I
                const double sk[3][3] = {{0, c[2], -c[1]}, {-c[2], 0, c[0]}, {c[1], -c[0], 0}};
                double J[2][6];
                for (int a = 0; a < 2; ++a) {
                    for (int k = 0; k < 3; ++k) J[a][k] = dq[a][0] * sk[0][k] + dq[a][1] * sk[1][k] + dq[a][2] * sk[2][k];
                    for (int k = 0; k < 3; ++k) J[a][3 + k] = dq[a][k] * d;
                }
                for (int a = 0; a < 2; ++a)
                    for (int i = 0; i < 6; ++i) {
                        g[i] += J[a][i] * r[a];
                        for (int j = i; j < 6; ++j) A[i][j] += J[a][i] * J[a][j];
                    }
            }
            return n ? sum / n : 1e30;
        };
        double best = 1e30;
        for (int it = 0; it < 5; ++it) {
            double A[6][6] = {}, g[6] = {};
            const double e = evaluate(R, T, A, g);
            if (it == 0) best = e;
            for (int i = 0; i < 6; ++i) for (int j = 0; j < i; ++j) A[i][j] = A[j][i];
            for (int i = 0; i < 6; ++i) A[i][i] += 1e-9 + 1e-6 * A[i][i];  // damping: translation is weak without near samples
            double step[6];
            if (!solve6(A, g, step)) break;
            const M3 Rn = mul(rotation(V3{-step[0], -step[1], -step[2]}), R);
            const V3 Tn{T[0] - step[3], T[1] - step[4], T[2] - step[5]};
            const double en = evaluate(Rn, Tn, nullptr, nullptr);
            if (!(en < best)) break;
            const double gain = best - en;
            R = Rn; T = Tn; best = en;
            if (gain < 1e-6 * (best + 1e-9)) break;
        }
        omega = log_rotation(R);
        return best;
    }

    // Rotation vector (axis * angle) of a rotation matrix.
    static V3 log_rotation(const M3& R) {
        const double c = std::clamp((R[0][0] + R[1][1] + R[2][2] - 1.0) * 0.5, -1.0, 1.0);
        const double a = std::acos(c);
        const V3 v{R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]};
        if (a < 1e-9) return V3{v[0] * 0.5, v[1] * 0.5, v[2] * 0.5};
        const double k = a / (2.0 * std::sin(a));
        return V3{v[0] * k, v[1] * k, v[2] * k};
    }

    // One game frame. Updates the orientation and returns the synthesized camera.
    Camera update(const std::vector<MotionSample>& all, double w, double h, const Camera& game) {
        // The game may tell the field of view (FSR asks for the vertical one). It is checked against the one
        // learned from the motion: some games hand over the horizontal one, and some values are simply off.
        // (only plausible values: some games hand FSR nonsense while loading)
        const double kMinFov = 10.0 * 3.14159265358979 / 180.0, kMaxFov = 150.0 * 3.14159265358979 / 180.0;
        game_v_ = game_h_ = 0;
        if (game.fov > kMinFov && game.fov < kMaxFov) { game_v_ = game.fov; game_h_ = 2.0 * std::atan(std::tan(game.fov * 0.5) * h / w); }
        choose_fov();
        // Zooms are followed from frame to frame only with a field of view the game tells.
        const bool follow_zoom = following_game_ && prev_fov_ > 0;
        // Distant pixels carry rotation almost only (translation parallax falls off with distance):
        // use the farther half of the valid samples.
        std::vector<float> depths;
        for (const auto& p : all) if (p.valid > 0.5f) depths.push_back(p.depth);
        std::vector<MotionSample> distant;
        if (depths.size() >= 32) {
            std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
            const float median = depths[depths.size() / 2];
            for (const auto& p : all) if (p.valid > 0.5f && p.depth <= median) distant.push_back(p);
        }
        V3 omega{};
        bool ok = false;
        // The previous frame's focal length (differs only while the game zooms and tells us its field of view).
        const double fprev_fov = prev_fov_ > 0 ? prev_fov_ : fov_;
        if (distant.size() >= 16) {
            if (!learned_locked_) learn_fov(distant, w, h);
            const double f = focal(h), fp = (h * 0.5) / std::tan((follow_zoom ? fprev_fov : fov_) * 0.5);
            omega = linear_rotation(distant, w, h, f, fp);
            V3 no_move{};
            double e = fit_motion(distant, w, h, f, omega, no_move, fp);  // distant: rotation dominates; T is refitted below
            // Reject samples the rotation does not explain (moving objects, near geometry), then refit.
            std::vector<double> errs;
            const M3 R = rotation(omega);
            for (const auto& p : distant) errs.push_back(sample_error(p, R, w, h, f, fp));
            std::vector<double> sorted = errs;
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const double limit = std::max(1.0, 3.0 * sorted[sorted.size() / 2]);
            std::vector<MotionSample> inliers;
            for (std::size_t i = 0; i < distant.size(); ++i) if (errs[i] <= limit) inliers.push_back(distant[i]);
            if (inliers.size() >= 16) { V3 t0{}; e = fit_motion(inliers, w, h, f, omega, t0, fp); }
            ok = std::isfinite(e) && std::sqrt(e) < acceptable(distant);
            last_residual_ = std::sqrt(e);
        }
        // Rotation and translation from all samples (near ones carry the translation), starting from the
        // distant-sample rotation; samples the motion does not explain (moving objects) are dropped.
        V3 T{};
        if (ok) {
            std::vector<MotionSample> valid;
            for (const auto& p : all) if (p.valid > 0.5f) valid.push_back(p);
            const double f = focal(h), fp = (h * 0.5) / std::tan((follow_zoom ? fprev_fov : fov_) * 0.5);
            V3 o = omega, t{};
            double e = fit_motion(valid, w, h, f, o, t, fp);
            std::vector<double> errs;
            const M3 R0 = rotation(o);
            for (const auto& p : valid) {
                const double X = (p.x - w * 0.5) / f, Y = -(p.y - h * 0.5) / f;
                const V3 c = mul(R0, V3{X, Y, 1.0});
                const double qz = std::max(c[2] + t[2] * p.depth, 1e-6);
                const double ex = w * 0.5 + fp * (c[0] + t[0] * p.depth) / qz - (p.x + p.mx);
                const double ey = h * 0.5 - fp * (c[1] + t[1] * p.depth) / qz - (p.y + p.my);
                errs.push_back(std::sqrt(ex * ex + ey * ey));
            }
            std::vector<double> sorted = errs;
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const double limit = std::max(1.0, 3.0 * sorted[sorted.size() / 2]);
            std::vector<MotionSample> inliers;
            for (std::size_t i = 0; i < valid.size(); ++i) if (errs[i] <= limit) inliers.push_back(valid[i]);
            if (inliers.size() >= 32) e = fit_motion(inliers, w, h, f, o, t, fp);
            if (std::isfinite(e) && std::sqrt(e) < acceptable(valid)) { omega = o; T = t; last_residual_ = std::sqrt(e); }
        }
        ++frames_;
        if (!ok) ++rejected_;
        last_rejected_ = !ok;
        if (!ok || game.reset) { omega = V3{0, 0, 0}; T = V3{0, 0, 0}; }
        last_omega_ = omega;
        last_translation_ = T;
        // The camera now sits at T in the previous camera's coordinates: world move = B_prev * T.
        for (int i = 0; i < 3; ++i) position_[i] += basis_[i][0] * T[0] + basis_[i][1] * T[1] + basis_[i][2] * T[2];
        // world = B_prev * c_prev = B_prev * R * c_cur  ->  B_cur = B_prev * R  (B columns: right, up, fwd)
        const M3 R = rotation(omega);
        basis_ = mul(basis_, R);
        orthonormalize(basis_);
        const Camera out = synthesize(w, h, R, T, game, (h * 0.5) / std::tan((follow_zoom ? prev_fov_ : fov_) * 0.5));
        prev_fov_ = fov_;
        if (learned_locked_ && game_v_ > 0 && game_fov_mode_ == 0) decide_game_fov();
        return out;
    }

    // Largest fit error (px, root mean square) still accepted: 4 px, or 6% of the typical motion on fast
    // turns (motion vectors and the model both get less exact with speed; dropping the fit then would
    // show a still camera, or no translation, exactly when it matters most).
    static double acceptable(const std::vector<MotionSample>& s) {
        std::vector<double> m;
        for (const auto& p : s) if (p.valid > 0.5f) m.push_back(std::hypot(p.mx, p.my));
        if (m.empty()) return 4.0;
        std::nth_element(m.begin(), m.begin() + m.size() / 2, m.end());
        return std::max(4.0, 0.06 * m[m.size() / 2]);
    }

    double vertical_fov() const { return fov_; }
    bool fov_locked() const { return fov_locked_; }
    double last_residual() const { return last_residual_; }
    // Frames whose motion no camera move explained (the camera then held still for that frame); read and cleared.
    bool last_rejected() const { return last_rejected_; }
    double take_rejected_fraction() { const double r = frames_ ? double(rejected_) / double(frames_) : 0.0; frames_ = rejected_ = 0; return r; }
    V3 last_omega() const { return last_omega_; }
    V3 last_translation() const { return last_translation_; }
    // How the game's field of view is used: 0 not checked yet (taken as vertical meanwhile), 1 as the vertical
    // one, 2 as the horizontal one, 3 not at all (it does not match the picture; the learned one is used).
    int game_fov_mode() const { return game_fov_mode_; }
    double learned_fov() const { return learned_fov_; }
    void reset() {
        basis_ = identity_basis(); position_ = V3{}; fov_votes_.clear(); fov_locked_ = false; fov_ = kDefaultFov; prev_fov_ = 0;
        learned_fov_ = kDefaultFov; learned_locked_ = false; game_fov_mode_ = 0; game_votes_v_.clear(); game_votes_h_.clear();
    }

    static constexpr double kDefaultFov = 1.2217;  // 70 degrees vertical until learned

    // Rotation matrix for a rotation vector (axis * angle).
    static M3 rotation(const V3& o) {
        const double a = std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
        M3 R{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
        if (a < 1e-12) return R;
        const double x = o[0] / a, y = o[1] / a, z = o[2] / a, c = std::cos(a), s = std::sin(a), t = 1 - c;
        return M3{{{t * x * x + c, t * x * y - s * z, t * x * z + s * y},
                   {t * x * y + s * z, t * y * y + c, t * y * z - s * x},
                   {t * x * z - s * y, t * y * z + s * x, t * z * z + c}}};
    }

private:
    static V3 mul(const M3& R, const V3& v) {
        return V3{R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
                  R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]};
    }
    static M3 mul(const M3& a, const M3& b) {
        M3 r{};
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) for (int k = 0; k < 3; ++k) r[i][j] += a[i][k] * b[k][j];
        return r;
    }
    static bool solve3(double A[3][3], const double b[3], V3& x) {
        const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        if (std::fabs(det) < 1e-18) return false;
        for (int k = 0; k < 3; ++k) {
            double M[3][3];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) M[i][j] = j == k ? b[i] : A[i][j];
            x[k] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) / det;
        }
        return true;
    }
    static bool solve6(double A[6][6], const double b[6], double x[6]) {
        double M[6][7];
        for (int i = 0; i < 6; ++i) { for (int j = 0; j < 6; ++j) M[i][j] = A[i][j]; M[i][6] = b[i]; }
        for (int c = 0; c < 6; ++c) {
            int piv = c;
            for (int r = c + 1; r < 6; ++r) if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
            if (std::fabs(M[piv][c]) < 1e-18) return false;
            for (int j = 0; j < 7; ++j) std::swap(M[c][j], M[piv][j]);
            for (int r = 0; r < 6; ++r) {
                if (r == c) continue;
                const double k = M[r][c] / M[c][c];
                for (int j = c; j < 7; ++j) M[r][j] -= k * M[c][j];
            }
        }
        for (int i = 0; i < 6; ++i) x[i] = M[i][6] / M[i][i];
        return true;
    }
    // Small-rotation flow model: dX = wy(1+X^2) - wx XY - wz Y,  dY = -wx(1+Y^2) + wy XY + wz X (normalized).
    static V3 linear_rotation(const std::vector<MotionSample>& s, double w, double h, double f, double fp = 0) {
        if (fp <= 0) fp = f;
        double A[3][3] = {}, b[3] = {};
        for (const auto& p : s) {
            if (p.valid < 0.5f) continue;
            const double X = (p.x - w * 0.5) / f, Y = -(p.y - h * 0.5) / f;
            // where the pixel was in the previous frame, in that frame's normalized coordinates
            const double dX = (p.x + p.mx - w * 0.5) / fp - X, dY = -(p.y + p.my - h * 0.5) / fp - Y;
            const double rx[3] = {-X * Y, 1 + X * X, -Y}, ry[3] = {-(1 + Y * Y), X * Y, X};
            for (int i = 0; i < 3; ++i) {
                b[i] += rx[i] * dX + ry[i] * dY;
                for (int j = 0; j < 3; ++j) A[i][j] += rx[i] * rx[j] + ry[i] * ry[j];
            }
        }
        V3 o{};
        solve3(A, b, o);
        return o;
    }
    static double sample_error(const MotionSample& p, const M3& R, double w, double h, double f, double fp) {
        const double X = (p.x - w * 0.5) / f, Y = -(p.y - h * 0.5) / f;
        const V3 c = mul(R, V3{X, Y, 1.0});
        const double z = std::max(c[2], 1e-6);
        const double ex = w * 0.5 + fp * c[0] / z - (p.x + p.mx), ey = h * 0.5 - fp * c[1] / z - (p.y + p.my);
        return std::sqrt(ex * ex + ey * ey);
    }
    double focal(double h) const { return (h * 0.5) / std::tan(fov_ * 0.5); }

    // The focal length only shows in how the flow bends towards the screen edges, so it is learned from
    // frames with clear rotation: each such frame votes for the field of view that fits it best
    // (coarse-to-fine search), and the median of the votes locks once enough agree.
    void learn_fov(const std::vector<MotionSample>& s, double w, double h) {
        const double pi = 3.14159265358979;
        auto score = [&](double fov) { V3 o; return fit_rotation(s, w, h, (h * 0.5) / std::tan(fov * 0.5), o); };
        V3 o;
        fit_rotation(s, w, h, focal(h), o);
        const double angle = std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
        if (angle < 0.004) return;  // under ~0.25 degrees this frame: the bend is too small to measure
        double best_fov = fov_, best = 1e30;
        for (double deg = 30; deg <= 120; deg += 5) { const double e = score(deg * pi / 180); if (e < best) { best = e; best_fov = deg * pi / 180; } }
        const double centre = best_fov;
        for (double d = -4; d <= 4; d += 1) {
            const double fov = centre + d * pi / 180;
            if (fov <= 0.3) continue;
            const double e = score(fov);
            if (e < best) { best = e; best_fov = fov; }
        }
        fov_votes_.push_back(best_fov);
        // How far the game's value is from this frame's measurement, read either way (focal length ratio).
        if (game_v_ > 0) {
            game_votes_v_.push_back(std::log(std::tan(best_fov * 0.5) / std::tan(game_v_ * 0.5)));
            game_votes_h_.push_back(std::log(std::tan(best_fov * 0.5) / std::tan(game_h_ * 0.5)));
        }
        std::vector<double> v = fov_votes_;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        learned_fov_ = v[v.size() / 2];
        if (fov_votes_.size() >= 30) {
            int close = 0;
            for (double x : fov_votes_) if (std::fabs(x - learned_fov_) < 3 * pi / 180) ++close;
            if (close * 3 >= int(fov_votes_.size()) * 2) learned_locked_ = true;
        }
    }

    static double median_abs(std::vector<double> v) {
        if (v.empty()) return 1e30;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return std::fabs(v[v.size() / 2]);
    }

    // The learned field of view settles which way the game's value is meant, compared frame by frame (so
    // zooming while learning does not matter): within 12% of the focal length, or not used at all.
    void decide_game_fov() {
        double ev = median_abs(game_votes_v_), eh = median_abs(game_votes_h_);
        if (game_votes_v_.size() < 10) {  // the game told its field of view only after learning finished
            ev = std::fabs(std::log(std::tan(learned_fov_ * 0.5) / std::tan(game_v_ * 0.5)));
            eh = std::fabs(std::log(std::tan(learned_fov_ * 0.5) / std::tan(game_h_ * 0.5)));
        }
        constexpr double kTolerance = 0.12;
        game_fov_mode_ = (ev <= eh && ev < kTolerance) ? 1 : (eh < kTolerance) ? 2 : 3;
    }

    void choose_fov() {
        following_game_ = game_v_ > 0 && game_fov_mode_ != 3;
        if (following_game_) {
            fov_ = game_fov_mode_ == 2 ? game_h_ : game_v_;
            fov_locked_ = game_fov_mode_ != 0 || learned_locked_;
        } else {
            fov_ = learned_fov_;
            fov_locked_ = learned_locked_;
        }
    }

    static M3 identity_basis() {
        // Unreal-like world axes: forward +X, right +Y, up +Z. Columns: right, up, forward.
        return M3{{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}};
    }
    static void orthonormalize(M3& B) {
        V3 f{B[0][2], B[1][2], B[2][2]}, u{B[0][1], B[1][1], B[2][1]};
        auto norm = [](V3& v) { const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); for (auto& x : v) x /= l; };
        norm(f);
        const double d = u[0] * f[0] + u[1] * f[1] + u[2] * f[2];
        for (int i = 0; i < 3; ++i) u[i] -= d * f[i];
        norm(u);
        const V3 r{u[1] * f[2] - u[2] * f[1], u[2] * f[0] - u[0] * f[2], u[0] * f[1] - u[1] * f[0]};  // right = up x forward
        for (int i = 0; i < 3; ++i) { B[i][0] = r[i]; B[i][1] = u[i]; B[i][2] = f[i]; }
    }

    Camera synthesize(double w, double h, const M3& R, const V3& T, const Camera& game, double fprev) const {
        Camera c = game;
        const double f = focal(h);
        const float sx = float(2 * f / w), sy = float(2 * f / h), near_plane = 1.0f;
        // Row-vector infinite projection with view z = forward (clip.w = z), in the game's depth convention:
        // reversed (depth = near / z: clip.z = near) or standard (depth = 1 - near / z: clip.z = z - near).
        const bool reversed = game.depth_inverted != 0;
        const float zz = reversed ? 0.0f : 1.0f, wz = reversed ? near_plane : -near_plane;
        const float P[16] = {sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, zz, 1, 0, 0, wz, 0};
        float Pinv[16] = {1 / sx, 0, 0, 0, 0, 1 / sy, 0, 0, 0, 0, 0, 1 / near_plane, 0, 0, 1, 0};
        if (!reversed) { Pinv[10] = 0; Pinv[11] = -1 / near_plane; Pinv[14] = 1; Pinv[15] = 1 / near_plane; }
        // The previous frame's projection (its own field of view while zooming).
        const float psx = float(2 * fprev / w), psy = float(2 * fprev / h);
        const float Pprev[16] = {psx, 0, 0, 0, 0, psy, 0, 0, 0, 0, zz, 1, 0, 0, wz, 0};
        std::copy(P, P + 16, c.view_to_clip);
        std::copy(Pinv, Pinv + 16, c.clip_to_view);
        // clipToPrevClip = Pinv * Mrow * Pprev, Mrow = row-vector rotation current -> previous view (= R^T).
        double M[4][4] = {}, PM[4][4] = {}, C[4][4] = {};
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) M[i][j] = R[j][i];
        for (int j = 0; j < 3; ++j) M[3][j] = T[j];
        M[3][3] = 1;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) PM[i][j] += double(Pinv[i * 4 + k]) * M[k][j];
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) C[i][j] += PM[i][k] * double(Pprev[k * 4 + j]);
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) c.clip_to_prev_clip[i * 4 + j] = float(C[i][j]);
        for (int i = 0; i < 3; ++i) {
            c.right[i] = float(basis_[i][0]); c.up[i] = float(basis_[i][1]); c.fwd[i] = float(basis_[i][2]);
            c.pos[i] = float(position_[i]);
        }
        c.near_plane = near_plane; c.far_plane = 0;
        c.fov = float(fov_); c.aspect = float(w / h);
        c.depth_inverted = reversed ? 1u : 0u;
        c.valid = 1;
        return c;
    }

    M3 basis_ = identity_basis();
    double fov_ = kDefaultFov, last_residual_ = 0, prev_fov_ = 0;
    bool fov_locked_ = false;
    std::vector<double> fov_votes_;
    double learned_fov_ = kDefaultFov, game_v_ = 0, game_h_ = 0;
    bool learned_locked_ = false, following_game_ = false;
    int game_fov_mode_ = 0;
    std::vector<double> game_votes_v_, game_votes_h_;
    V3 last_omega_{}, last_translation_{}, position_{};
    std::uint64_t frames_ = 0, rejected_ = 0;
    bool last_rejected_ = false;
};

}  // namespace fw
