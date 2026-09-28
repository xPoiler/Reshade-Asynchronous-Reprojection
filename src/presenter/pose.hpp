#pragma once
// Camera pose model for asynchronous reprojection.
//
// Games smooth their camera (E33: first-order lag, tau ~150 ms), so the rendered camera is not a
// direct function of mouse counts. Per axis (yaw, pitch) we model the game camera as
//     theta(t) = theta_N + tau * w_N * (1 - exp(-h/tau)) + g * sum_i dc_i * (1 - exp(-(t - s_i)/tau))
// where theta_N / w_N are the angle and angular velocity of the newest rendered frame, h = t - t_N,
// dc_i are raw mouse counts arriving at s_i (shifted by the input delay d). tau, d are found by grid
// search and g by least squares on the game's own frame history, continuously.
//
// The output shows the camera at "now - latency" (latency = the game's measured sim-to-ingest time
// minus an optional prediction), so each prediction spans about one game frame. When a new frame
// arrives, the residual difference to the previous prediction is blended out instead of snapped.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace fw {

struct Vec3 {
    double x = 0, y = 0, z = 0;
};
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) { const double l = length(a); return l > 1e-12 ? a * (1.0 / l) : a; }
// Rodrigues rotation of v about unit axis k by angle.
inline Vec3 rotate(Vec3 v, Vec3 k, double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.0 - c));
}

struct CameraBasis {
    Vec3 pos, right, up, fwd;
};

using Mat4 = std::array<float, 16>;

// Sign of view-space z for points in front of the camera, from a row-vector projection's w column:
// +1 when clip.w = +z (Unreal: view z = forward), -1 when clip.w = -z (right-handed engines such as
// RE Engine: the camera looks down -z).
inline double view_z_sign(const float* view_to_clip) { return view_to_clip[11] < 0.0f ? -1.0 : 1.0; }

// Row-vector world-to-view matrix (view x=right, y=up, z=forward*z_sign), positions relative to origin.
inline Mat4 view_matrix(const CameraBasis& c, Vec3 origin, double z_sign = 1.0) {
    const Vec3 p = c.pos - origin;
    const Vec3 z = c.fwd * z_sign;
    return {static_cast<float>(c.right.x), static_cast<float>(c.up.x), static_cast<float>(z.x), 0.0f,
            static_cast<float>(c.right.y), static_cast<float>(c.up.y), static_cast<float>(z.y), 0.0f,
            static_cast<float>(c.right.z), static_cast<float>(c.up.z), static_cast<float>(z.z), 0.0f,
            static_cast<float>(-dot(p, c.right)), static_cast<float>(-dot(p, c.up)), static_cast<float>(-dot(p, z)), 1.0f};
}

// Row-vector 4x4 helpers for the own warp engine.
inline Mat4 mat_mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double v = 0;
            for (int k = 0; k < 4; ++k) v += double(a[i * 4 + k]) * double(b[k * 4 + j]);
            r[i * 4 + j] = static_cast<float>(v);
        }
    return r;
}
inline bool mat_inverse(const Mat4& m, Mat4& out) {
    double a[4][8];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j) a[i][j] = j < 4 ? double(m[i * 4 + j]) : (j - 4 == i ? 1.0 : 0.0);
    for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r) if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
        if (std::fabs(a[pivot][c]) < 1e-12) return false;
        for (int j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
        const double d = a[c][c];
        for (int j = 0; j < 8; ++j) a[c][j] /= d;
        for (int r = 0; r < 4; ++r)
            if (r != c) { const double f = a[r][c]; for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j]; }
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out[i * 4 + j] = static_cast<float>(a[i][j + 4]);
    return true;
}
// Displayed camera clip space -> rendered frame clip space for the camera's rotation only (translation
// removed): clip_t * P^-1 * V_t^-1 * V_s * P.
inline Mat4 clip_to_source_rotation(const Mat4& projection, const Mat4& view_target, const Mat4& view_source) {
    Mat4 inv_p{}, inv_vt{};
    Mat4 identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    if (!mat_inverse(projection, inv_p) || !mat_inverse(view_target, inv_vt)) return identity;
    Mat4 rot = mat_mul(inv_vt, view_source);
    rot[12] = rot[13] = rot[14] = 0.0f; rot[3] = rot[7] = rot[11] = 0.0f; rot[15] = 1.0f;
    return mat_mul(mat_mul(inv_p, rot), projection);
}

// Rendered frame clip space (with depth) -> displayed camera clip space, rotation and movement:
// clip_s * P^-1 * V_s^-1 * V_t * P.
inline Mat4 clip_source_to_target(const Mat4& projection, const Mat4& view_source, const Mat4& view_target) {
    Mat4 inv_p{}, inv_vs{};
    Mat4 identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    if (!mat_inverse(projection, inv_p) || !mat_inverse(view_source, inv_vs)) return identity;
    return mat_mul(mat_mul(mat_mul(inv_p, inv_vs), view_target), projection);
}

// Signed yaw about world up from a to b, and elevation of a forward vector.
inline double yaw_between(Vec3 a, Vec3 b, Vec3 up) {
    const Vec3 ah = normalize(a - up * dot(a, up)), bh = normalize(b - up * dot(b, up));
    return std::atan2(dot(up, cross(ah, bh)), dot(ah, bh));
}
inline double elevation(Vec3 fwd, Vec3 up) { return std::asin(std::clamp(dot(fwd, up), -1.0, 1.0)); }

// Applies yaw about world up, then pitch about the (new) right axis, preserving roll.
inline CameraBasis apply_rotation(const CameraBasis& c, Vec3 world_up, double yaw, double pitch, Vec3 pivot) {
    CameraBasis r = c;
    auto spin = [&](Vec3 axis, double angle) {
        r.right = rotate(r.right, axis, angle);
        r.up = rotate(r.up, axis, angle);
        r.fwd = rotate(r.fwd, axis, angle);
        r.pos = pivot + rotate(r.pos - pivot, axis, angle);
    };
    if (yaw != 0.0) spin(world_up, yaw);
    if (pitch != 0.0) {
        const double e0 = elevation(r.fwd, world_up);
        const double limit = 88.0 * 3.14159265358979 / 180.0;
        const double target = std::clamp(e0 + pitch, -limit, limit);
        const Vec3 axis = normalize(r.right);
        // Rotating fwd about axis by +angle changes it by axis x fwd; pick the sign that raises elevation.
        const double sign = dot(cross(axis, r.fwd), world_up) >= 0.0 ? 1.0 : -1.0;
        spin(axis, sign * (target - e0));
    }
    return r;
}

// Raw mouse history (thread-safe: input thread appends, render thread queries).
class MouseHistory {
public:
    void add(double t, long dx, long dy) {
        std::lock_guard lock(mutex_);
        samples_.push_back({t, dx, dy});
        while (samples_.size() > 2 && samples_.front().t < t - 60.0) samples_.pop_front();
    }
    // Sum over events with t0 < t_i <= t1 of dc_i * (1 - exp(-(t_end - t_i)/tau)) (tau 0: plain sum).
    double filtered(int axis, double t0, double t1, double t_end, double tau) const {
        std::lock_guard lock(mutex_);
        auto it = std::upper_bound(samples_.begin(), samples_.end(), t0, [](double v, const Sample& s) { return v < s.t; });
        double total = 0;
        for (; it != samples_.end() && it->t <= t1; ++it) {
            const double dc = axis ? double(it->dy) : double(it->dx);
            total += tau > 0 ? dc * (1.0 - std::exp(-(t_end - it->t) / tau)) : dc;
        }
        return total;
    }
    double counts(int axis, double t0, double t1) const { return filtered(axis, t0, t1, t1, 0.0); }
    void reset() { std::lock_guard lock(mutex_); samples_.clear(); }

private:
    struct Sample { double t; long dx, dy; };
    mutable std::mutex mutex_;
    std::deque<Sample> samples_;
};

// Picks the world axis the camera's right vector is always perpendicular to (no-roll cameras).
class WorldUp {
public:
    void add(const CameraBasis& c) {
        const double comps[3] = {std::fabs(c.right.x), std::fabs(c.right.y), std::fabs(c.right.z)};
        const double ups[3] = {c.up.x, c.up.y, c.up.z};
        for (int i = 0; i < 3; ++i) { sum_[i] += comps[i]; sign_[i] += ups[i]; }
        ++count_;
    }
    Vec3 get() const {
        if (count_ < 8) return {0, 0, 1};  // Unreal default until enough evidence
        int best = 0;
        for (int i = 1; i < 3; ++i) if (sum_[i] < sum_[best]) best = i;
        Vec3 v{};
        const double s = sign_[best] >= 0 ? 1.0 : -1.0;
        (best == 0 ? v.x : best == 1 ? v.y : v.z) = s;
        return v;
    }
private:
    double sum_[3] = {0, 0, 0}, sign_[3] = {0, 0, 0};
    int count_ = 0;
};

// Parameters of one axis of the camera model.
struct AxisParams {
    double gain = 0;   // radians per count (0: mouse unused)
    double tau = 0.1;  // camera smoothing time constant, s (0: none)
    double delay = 0;  // input delay, s
    double quality = 0;  // 1 - rms(model)/rms(hold) on the fit window
    bool fitted = false;
};

// Camera-only motion over h seconds from angular velocity w under first-order smoothing.
inline double velocity_term(double tau, double w, double h) {
    return tau > 0 ? tau * w * (1.0 - std::exp(-h / tau)) : w * h;
}

struct PoseSettings {
    bool use_mouse = true;
    double rotation_extrapolation = 1.0;  // scales the camera-velocity term
    double orbit_distance = 0.0;          // > 0: manual orbit pivot distance; 0: learned
    double translation_extrapolation = 1.0;
    bool blend_positions = false;
    double blend_tau = 0.05;              // seconds over which a new frame's correction of the prediction fades in
    double latency_percentile = 0.9;      // of recent frame delays (0.5 = median); p90 halves how often a late frame forces extrapolation
    double max_horizon = 0.1;             // seconds
    double prediction = 0.0;              // seconds shown ahead of the game's own latency
    double auto_fraction = 1.0;           // > 0: prediction = -auto_fraction * (measured game frame interval)
    bool manual_gain = false;
    double manual_gain_x = 0, manual_gain_y = 0, manual_delay = 0;
};

struct Prediction {
    CameraBasis camera;
    double yaw = 0, pitch = 0;  // applied deltas relative to the source (radians)
    double horizon = 0;          // seconds predicted past the source
};

class PoseModel {
public:
    MouseHistory mouse;

    void configure(const PoseSettings& s) { settings_ = s; }
    void set_mouse_gate(bool open) { gate_ = open; }
    bool mouse_gate() const { return gate_; }
    const AxisParams& params(int axis) const { return params_[axis]; }
    double latency() const { return latency_; }
    // Median of the recent game frame intervals (0 until measured).
    double frame_interval() const { return frame_interval_; }
    // The latency setting actually in use (auto: exactly one game frame behind).
    double effective_prediction() const {
        return settings_.auto_fraction > 0 && frame_interval_ > 0 ? -settings_.auto_fraction * frame_interval_ : settings_.prediction;
    }
    // Orbit pivot distance used for rotations (manual only: on E33 the camera's measured velocity
    // already contains the orbit motion and predicts position better, measured on recorded play sessions).
    double orbit() const { return settings_.orbit_distance; }
    double learned_orbit() const { return orbit_fit_; }
    void seed(const AxisParams& yaw, const AxisParams& pitch) { params_[0] = yaw; params_[1] = pitch; }
    void reset_fit() { params_[0] = params_[1] = AxisParams{}; frames_.clear(); since_fit_ = 0; }
    // The camera source changed (the game's own camera <-> one estimated from motion vectors): their
    // orientations have nothing in common, so the history starts over. The fitted calibration stays.
    void reset_history() {
        frames_.clear(); since_fit_ = 0; up_ = WorldUp{}; have_source_ = false;
        blend_[0] = blend_[1] = 0; blend_pos_ = {}; velocity_ = {};
        orbit_num_ = orbit_den_ = orbit_fit_ = 0; intervals_.clear(); frame_interval_ = 0; latencies_.clear();
    }

    // A new rendered frame: t is the game's simulation (input) time, ingest_t when we received it.
    void add_source(std::uint64_t id, double t, double ingest_t, const CameraBasis& camera, bool reset, double now,
                    std::uint32_t epoch = 0) {
        up_.add(camera);
        const Vec3 world_up = up_.get();
        // Absolute prediction with the previous source, for handoff blending.
        double old_abs[2] = {0, 0};
        const bool had = have_source_;
        Vec3 old_pos{};
        if (had) {
            const Prediction p = predict(now);
            old_abs[0] = src_.theta[0] + p.yaw; old_abs[1] = src_.theta[1] + p.pitch; old_pos = p.camera.pos;
        }

        // Yaw is unwrapped against the previous frame; only consecutive, cut-free frames feed the fit.
        Frame f{id, t, {0, elevation(camera.fwd, world_up)}, gate_, camera.fwd, false, camera.pos, epoch};
        if (!frames_.empty()) {
            const Frame& prev = frames_.back();
            const double dyaw = yaw_between(prev.fwd, camera.fwd, world_up);
            f.theta[0] = prev.theta[0] + dyaw;
            f.consecutive = !reset && epoch == prev.epoch && id == prev.id + 1 && t > prev.t && t - prev.t < 0.25 && std::fabs(dyaw) < 0.3 &&
                            std::fabs(f.theta[1] - prev.theta[1]) < 0.3;
        }
        frames_.push_back(f);
        while (frames_.size() > 1200) frames_.pop_front();

        src_.t = t; src_.basis = camera;
        src_.theta[0] = f.theta[0]; src_.theta[1] = f.theta[1];
        src_.w[0] = src_.w[1] = 0;
        src_.dt = 0;
        if (f.consecutive && frames_.size() >= 2) {
            const Frame& a = frames_[frames_.size() - 2];
            src_.dt = t - a.t;
            // Game frame interval: median of the last 15 consecutive intervals (robust to hitches).
            intervals_.push_back(src_.dt);
            if (intervals_.size() > 15) intervals_.pop_front();
            std::vector<double> sorted(intervals_.begin(), intervals_.end());
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            frame_interval_ = sorted[sorted.size() / 2];
            src_.prev_pos = a.pos;
            // Orbit fit: for a camera orbiting a pivot d ahead, pos_b - pos_a = d * (fwd_a - fwd_b).
            const Vec3 moved = f.pos - a.pos, u = a.fwd - f.fwd;
            constexpr double decay = 0.995;
            orbit_num_ = orbit_num_ * decay + dot(moved, u);
            orbit_den_ = orbit_den_ * decay + dot(u, u);
            if (orbit_den_ > 1e-3) orbit_fit_ = std::clamp(orbit_num_ / orbit_den_, 0.0, 2000.0);
            // Translation the orbit does not explain (walking, camera lag) becomes a velocity.
            const Vec3 residual = (moved - u * orbit()) * (1.0 / src_.dt);
            const double a_v = std::clamp(src_.dt / 0.05, 0.0, 1.0);
            velocity_ = velocity_ + (residual - velocity_) * a_v;
            for (int k = 0; k < 2; ++k) { src_.w[k] = (f.theta[k] - a.theta[k]) / src_.dt; src_.prev_theta[k] = a.theta[k]; }
        }
        have_source_ = true;

        const double observed = ingest_t - t;
        if (observed > 0 && observed < 0.25) {
            // Frames arrive with jitter (they count once the game's GPU finished them). A mean latency
            // leaves the presenter without the frame it needs for ~half the late ones, so it extrapolates;
            // a high percentile of recent delays keeps the displayed camera between real frames.
            latencies_.push_back(observed);
            if (latencies_.size() > 90) latencies_.pop_front();
            std::vector<double> sorted(latencies_.begin(), latencies_.end());
            std::sort(sorted.begin(), sorted.end());
            const double q = std::clamp(settings_.latency_percentile, 0.0, 1.0);
            latency_ = sorted[static_cast<std::size_t>(q * double(sorted.size() - 1))];
        }

        if (++since_fit_ >= 30) { fit(); since_fit_ = 0; }

        // Blend: keep the output continuous across the source change.
        blend_[0] = blend_[1] = 0;
        blend_pos_ = {};
        if (!f.consecutive) velocity_ = {};
        if (had && f.consecutive) {
            const Prediction p = predict(now);
            const double off[2] = {old_abs[0] - (src_.theta[0] + p.yaw), old_abs[1] - (src_.theta[1] + p.pitch)};
            if (std::fabs(off[0]) < 0.15 && std::fabs(off[1]) < 0.15) { blend_[0] = off[0]; blend_[1] = off[1]; }
            const Vec3 dp = old_pos - p.camera.pos;
            if (length(dp) < 100.0 && settings_.blend_positions) blend_pos_ = dp;
        }
        blend_t_ = now;
    }

    // Camera for display at wall time `now`, relative to the newest source.
    Prediction predict(double now) const {
        Prediction out;
        if (!have_source_) return out;
        // prediction > 0 shows the camera ahead of the game's own latency (more extrapolation);
        // prediction < 0 shows it further in the past, down to interpolating between the last two
        // rendered cameras (smoothest, +|prediction| latency).
        const double target = now - (latency_ - effective_prediction());
        const double h = std::min(target - src_.t, settings_.max_horizon);
        double delta[2] = {0, 0};
        if (h < 0) {
            // The displayed time lies in the past: interpolate the game's real camera history (any depth,
            // not only the last two frames), holding the oldest contiguous frame if it is older still.
            const double target_t = src_.t + h;
            double theta[2] = {src_.theta[0], src_.theta[1]};
            Vec3 pos = src_.basis.pos;
            double reached = src_.t;
            for (std::size_t i = frames_.size(); i-- > 1 && frames_.size() - i < 64;) {
                const Frame& b = frames_[i];
                const Frame& a = frames_[i - 1];
                if (!b.consecutive) break;
                if (a.t <= target_t) {
                    const double w = (target_t - a.t) / (b.t - a.t);
                    for (int k = 0; k < 2; ++k) theta[k] = a.theta[k] + (b.theta[k] - a.theta[k]) * w;
                    pos = a.pos + (b.pos - a.pos) * w;
                    reached = target_t;
                    break;
                }
                theta[0] = a.theta[0]; theta[1] = a.theta[1]; pos = a.pos; reached = a.t;
            }
            const double fade = std::exp(-(now - blend_t_) / std::max(settings_.blend_tau, 1e-4));
            for (int k = 0; k < 2; ++k) delta[k] = theta[k] - src_.theta[k] + blend_[k] * fade;
            const Vec3 pivot = src_.basis.pos + src_.basis.fwd * orbit();
            out.camera = apply_rotation(src_.basis, up_.get(), delta[0], delta[1], pivot);
            out.camera.pos = pos + blend_pos_ * fade;
            out.yaw = delta[0]; out.pitch = delta[1]; out.horizon = reached - src_.t;
            return out;
        }
        for (int k = 0; k < 2; ++k) {
            const AxisParams& p = params_[k];
            const double tau = p.tau;
            delta[k] = velocity_term(tau, src_.w[k], h) * settings_.rotation_extrapolation;
            const double g = gain(k);
            if (settings_.use_mouse && gate_ && g != 0.0) {
                const double d = settings_.manual_gain ? settings_.manual_delay : p.delay;
                // Input after the source's cutoff up to the displayed camera time, smoothed like the game.
                const double end = src_.t + h + d;
                delta[k] += g * mouse.filtered(k, src_.t + d, std::min(now, end), end, tau);
            }
            const double fade = std::exp(-(now - blend_t_) / std::max(settings_.blend_tau, 1e-4));
            delta[k] += blend_[k] * fade;
        }
        const Vec3 world_up = up_.get();
        const Vec3 pivot = src_.basis.pos + src_.basis.fwd * orbit();
        out.camera = apply_rotation(src_.basis, world_up, delta[0], delta[1], pivot);
        out.camera.pos = out.camera.pos + velocity_ * (h * settings_.translation_extrapolation) +
                         blend_pos_ * std::exp(-(now - blend_t_) / std::max(settings_.blend_tau, 1e-4));
        out.yaw = delta[0]; out.pitch = delta[1]; out.horizon = h;
        return out;
    }

    double gain(int axis) const {
        if (settings_.manual_gain) return axis ? settings_.manual_gain_y : settings_.manual_gain_x;
        return params_[axis].gain;
    }
    const CameraBasis& source_basis() const { return src_.basis; }
    double source_theta(int axis) const { return src_.theta[axis]; }
    double source_time() const { return src_.t; }
    Vec3 world_up() const { return up_.get(); }

    // Refit tau / delay (grid) and gain (least squares) per axis on the frame history.
    void fit() {
        static constexpr double kTaus[] = {0.0, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2, 0.3};
        static constexpr double kDelays[] = {-0.01, 0.0, 0.01, 0.02, 0.03};
        for (int k = 0; k < 2; ++k) {
            double best_rms = 1e30, best_tau = params_[k].tau, best_delay = params_[k].delay, best_gain = 0, hold_rms = 0;
            bool informative = false;  // enough mouse movement in the history to judge a gain
            for (double tau : kTaus)
                for (double d : kDelays) {
                    double suy = 0, suu = 0, syy = 0, excitation = 0, hold = 0;
                    int n = 0;
                    for (std::size_t i = 2; i < frames_.size(); ++i) {
                        const Frame &a = frames_[i - 2], &b = frames_[i - 1], &c = frames_[i];
                        if (!b.consecutive || !c.consecutive || !a.gate || !b.gate || !c.gate) continue;
                        const double h = c.t - b.t;
                        const double w = (b.theta[k] - a.theta[k]) / (b.t - a.t);
                        const double y = c.theta[k] - b.theta[k] - velocity_term(tau, w, h);
                        const double u = mouse.filtered(k, b.t + d, c.t + d, c.t + d, tau);
                        suy += u * y; suu += u * u; syy += y * y; excitation += std::fabs(u);
                        hold += (c.theta[k] - b.theta[k]) * (c.theta[k] - b.theta[k]);
                        ++n;
                    }
                    if (n < 30) continue;
                    double g = suu > 0 ? suy / suu : 0;
                    if (excitation < 200) g = 0;  // not enough mouse movement to trust a gain
                    else informative = true;
                    const double sse = syy - 2 * g * suy + g * g * suu;
                    const double rms = std::sqrt(std::max(0.0, sse) / n);
                    if (rms < best_rms) { best_rms = rms; best_tau = tau; best_delay = d; best_gain = g; hold_rms = std::sqrt(hold / n); }
                }
            // Too little mouse movement lately (standing still, menus, the ReShade menu open): keep what was
            // learned instead of dropping to no mouse at all until the next movement is learned again.
            if (!informative && params_[k].fitted) continue;
            if (best_rms < 1e29) {
                params_[k].tau = best_tau; params_[k].delay = best_delay; params_[k].gain = best_gain;
                params_[k].quality = hold_rms > 0 ? 1.0 - best_rms / hold_rms : 0;
                params_[k].fitted = true;
            }
        }
    }

private:
    struct Frame {
        std::uint64_t id;
        double t;
        double theta[2];  // unwrapped yaw, pitch (elevation)
        bool gate;
        Vec3 fwd;
        bool consecutive = false;
        Vec3 pos{};
        std::uint32_t epoch = 0;
    };
    struct Source {
        double t = 0;
        CameraBasis basis{};
        double theta[2] = {0, 0}, w[2] = {0, 0};
        double prev_theta[2] = {0, 0}, dt = 0;  // previous rendered camera (for interpolation)
        Vec3 prev_pos{};
    };
    WorldUp up_;
    PoseSettings settings_;
    AxisParams params_[2];
    std::deque<Frame> frames_;
    Source src_;
    bool have_source_ = false;
    bool gate_ = true;
    double latency_ = 0;
    std::deque<double> latencies_;
    double blend_[2] = {0, 0}, blend_t_ = 0;
    Vec3 blend_pos_{}, velocity_{};
    double orbit_num_ = 0, orbit_den_ = 0, orbit_fit_ = 0;
    int since_fit_ = 0;
    std::deque<double> intervals_;
    double frame_interval_ = 0;
};

}  // namespace fw
