#include "presenter/pose.hpp"
#include "presenter/frame_clock.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace fw;

static int failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static const Vec3 kUp{0, 0, 1};  // Unreal: Z up
static CameraBasis camera_at(double yaw, double pitch) {
    CameraBasis c{{}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}};  // UE: fwd +X, right +Y, up +Z
    return apply_rotation(c, kUp, yaw, pitch, c.pos);
}

static void rotation_conventions() {
    const CameraBasis base = camera_at(0, 0);
    const CameraBasis turned = apply_rotation(base, kUp, 0.3, 0.0, base.pos);
    EXPECT(std::fabs(yaw_between(base.fwd, turned.fwd, kUp) - 0.3) < 1e-9, "yaw measured %f", yaw_between(base.fwd, turned.fwd, kUp));
    const CameraBasis raised = apply_rotation(base, kUp, 0.0, 0.2, base.pos);
    EXPECT(std::fabs(elevation(raised.fwd, kUp) - 0.2) < 1e-9, "pitch measured %f", elevation(raised.fwd, kUp));
    EXPECT(std::fabs(dot(raised.right, raised.fwd)) < 1e-9 && std::fabs(dot(raised.up, raised.fwd)) < 1e-9, "basis orthogonal");
    EXPECT(elevation(apply_rotation(base, kUp, 0.0, 3.0, base.pos).fwd, kUp) < 1.54, "pitch clamps below 88 deg");
    const Vec3 pivot = base.pos + base.fwd * 300.0;
    const CameraBasis orbit = apply_rotation(base, kUp, 0.5, 0.0, pivot);
    EXPECT(length(orbit.pos + orbit.fwd * 300.0 - pivot) < 1e-6, "orbit keeps pivot");
    const Mat4 v = view_matrix(turned, {});
    const double z = turned.fwd.x * v[2] + turned.fwd.y * v[6] + turned.fwd.z * v[10];
    EXPECT(std::fabs(z - 1.0) < 1e-6, "forward maps to +z (%f)", z);
    // An unmoved camera gives the identity in float (the own engine then shows the frame as it is), for
    // reversed and standard depth, small and large near planes, cameras far from the world origin.
    double worst = 0;
    for (double znear : {0.01, 0.1, 10.0})
        for (double fov : {0.6, 1.2, 1.9})
            for (int standard = 0; standard < 2; ++standard) {
                const float sy = float(1.0 / std::tan(fov * 0.5)), sx = sy / (16.0f / 9.0f), n = float(znear);
                const Mat4 proj = standard ? Mat4{sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 1, 1, 0, 0, -n, 0}
                                           : Mat4{sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 0, 1, 0, 0, n, 0};
                CameraBasis cam = apply_rotation(base, kUp, 2.1, -0.4, base.pos);
                cam.pos = {123456.0, -98765.0, 4321.0};
                const Mat4 m = clip_source_to_target(proj, view_matrix(cam, cam.pos), view_matrix(cam, cam.pos));
                for (int i = 0; i < 16; ++i) worst = std::max(worst, std::fabs(double(m[i]) - (i % 5 == 0 ? 1.0 : 0.0)));
            }
    EXPECT(worst < 2e-6, "unmoved camera: identity within 2e-6 (worst %.2e)", worst);
}

// Right-handed engines (RE Engine): Y up, the camera looks down view -z and the projection has
// clip.w = -z. A point ahead and to the right must land in front of the camera, on the right.
static void right_handed_projection() {
    const float near_plane = 0.01f;
    const float rh[16] = {1.18f, 0, 0, 0, 0, 2.1f, 0, 0, 0, 0, 0, -1, 0, 0, near_plane, 0};  // row-vector, reversed-Z infinite
    const float ue[16] = {0.98f, 0, 0, 0, 0, 1.73f, 0, 0, 0, 0, 0, 1, 0, 0, 10.0f, 0};
    EXPECT(view_z_sign(rh) == -1.0 && view_z_sign(ue) == 1.0, "z sign from the projection's w column");
    const CameraBasis cam{{3, 1, 2}, {1, 0, 0}, {0, 1, 0}, {0, 0, -1}};  // right x up = -fwd
    const Vec3 point = cam.pos + cam.fwd * 10.0 + cam.right * 0.5;
    auto project = [&](double z_sign, double& x_ndc, double& w) {
        const Mat4 v = view_matrix(cam, {}, z_sign);
        const double p[4] = {point.x, point.y, point.z, 1.0};
        double view[4]{}, clip[4]{};
        for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r) view[c] += p[r] * v[r * 4 + c];
        for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r) clip[c] += view[r] * rh[r * 4 + c];
        w = clip[3]; x_ndc = clip[0] / clip[3];
    };
    double x = 0, w = 0;
    project(view_z_sign(rh), x, w);
    EXPECT(w > 0 && x > 0, "right-handed: point ahead-right projects in front (w %.3f) and right (x %.3f)", w, x);
    project(1.0, x, w);
    EXPECT(w < 0, "the Unreal-style view with a right-handed projection puts the scene behind the camera (w %.3f)", w);
}

// A game whose camera follows target = gain * counts through a first-order lag (like E33),
// rendering at `fps` and consuming input `delay` after each simulation start.
struct Game {
    double gain_x = -0.0023, gain_y = 0.0015, tau = 0.15, delay = 0.01, fps = 30.0, latency = 0.02;
    // Controller: the right stick (sx, sy) turns the camera at stick_gain * |s|^stick_power rad/s; the
    // presenter reads it every 4 ms.
    double stick_gain_x = 0, stick_gain_y = 0, stick_power = 2.0, sx = 0, sy = 0;
    double cam[2] = {0, 0}, target[2] = {0, 0};
    double t = 0, next_frame = 0.0, last_sim = 0;
    std::uint64_t frame = 0;
    std::vector<std::pair<double, long>> pending_x, pending_y;
    struct Out { double t, yaw, pitch; };
    std::vector<Out> truth;  // camera at every 1 ms step
    PoseModel* model = nullptr;
    bool feed_model = true;
    bool menu = false;  // the mouse moves a cursor the game draws itself: the camera stays put
    // Advances 1 ms: mouse event, camera integration, maybe a rendered frame.
    void step(long dx, long dy) {
        t += 0.001;
        if (dx || dy) model->mouse.add(t, dx, dy);
        if (!menu) { target[0] += gain_x * dx; target[1] += gain_y * dy; }
        target[0] += stick_gain_x * StickHistory::curve(sx, stick_power) * 0.001;
        target[1] += stick_gain_y * StickHistory::curve(sy, stick_power) * 0.001;
        if ((stick_gain_x != 0 || stick_gain_y != 0) && (static_cast<long>(std::lround(t * 1000.0)) % 4) == 0)
            model->stick.add(t, std::round(sx * 1000.0) / 1000.0, std::round(sy * 1000.0) / 1000.0);
        // The game applies input with `delay`, smoothing continuously.
        const double a = 1.0 - std::exp(-0.001 / tau);
        cam[0] += (target_at_delay(0) - cam[0]) * a;
        cam[1] += (target_at_delay(1) - cam[1]) * a;
        history_target.push_back({t, target[0], target[1]});
        truth.push_back({t, cam[0], cam[1]});
        if (t >= next_frame) {
            next_frame += 1.0 / fps;
            // Rendered now, reaches the presenter `latency` later (like the real pipeline).
            in_flight.push_back({t, cam[0], cam[1]});
        }
        while (!in_flight.empty() && t >= in_flight.front().t + latency) {
            const auto f = in_flight.front();
            in_flight.erase(in_flight.begin());
            last_sim = f.t;
            if (feed_model) model->add_source(++frame, f.t, t, camera_at(f.yaw, f.pitch), false, t);
        }
    }
    std::vector<Out> in_flight;
    struct T3 { double t, x, y; };
    std::vector<T3> history_target;
    double target_at_delay(int axis) const {
        const double when = t - delay;
        for (auto it = history_target.rbegin(); it != history_target.rend(); ++it)
            if (it->t <= when) return axis ? it->y : it->x;
        return 0;
    }
    double truth_at(double when, int axis) const {
        for (auto it = truth.rbegin(); it != truth.rend(); ++it)
            if (it->t <= when) return axis ? it->pitch : it->yaw;
        return 0;
    }
};

// 125 Hz mouse reports of a few counts (like the recorded session): < 1 rad/s camera motion.
static bool report(double t) { return (static_cast<long>(std::lround(t * 1000.0)) % 8) == 0; }
static long pattern_x(double t) { return report(t) ? long(std::lround(3.0 * std::sin(t * 2.3) + 1.5 * std::sin(t * 7.1) + 1.0 * std::sin(t * 0.7))) : 0; }
static long pattern_y(double t) { return report(t) ? long(std::lround(1.5 * std::cos(t * 1.9) + 1.0 * std::sin(t * 5.3))) : 0; }

static void fit_recovers_smoothed_camera() {
    PoseModel model;
    Game g; g.model = &model;
    for (int i = 0; i < 20000; ++i) g.step(pattern_x(g.t), pattern_y(g.t));
    model.fit();
    const auto& x = model.params(0);
    const auto& y = model.params(1);
    EXPECT(x.fitted && y.fitted, "fitted");
    // tau, gain and delay partly trade off on short windows; what matters is that smoothing is found.
    EXPECT(x.tau >= 0.05, "yaw smoothing detected: tau %.3f (true %.3f)", x.tau, g.tau);
    EXPECT(std::fabs(x.gain / g.gain_x - 1.0) < 0.2, "yaw gain %g (true %g)", x.gain, g.gain_x);
    EXPECT(std::fabs(y.gain / g.gain_y - 1.0) < 0.2, "pitch gain %g (true %g)", y.gain, g.gain_y);
    EXPECT(x.quality > 0.7, "yaw quality %.2f", x.quality);
}

// Displayed camera vs the true camera at the displayed time, and smoothness at 120 Hz.
static void prediction_is_accurate_and_continuous() {
    PoseModel model;
    Game g; g.model = &model;
    for (int i = 0; i < 20000; ++i) g.step(pattern_x(g.t), pattern_y(g.t));  // learn
    double sse = 0, hold_sse = 0, worst_step = 0, worst_true_step = 0;
    int n = 0;
    double prev_shown = 0, prev_true = 0;
    bool have_prev = false;
    for (int i = 0; i < 3000; ++i) {
        g.step(pattern_x(g.t), pattern_y(g.t));
        if (i % 8 != 0) continue;  // 125 Hz output
        const double now = g.t;
        const Prediction p = model.predict(now);
        const double shown = model.source_theta(0) + p.yaw;
        const double truth = g.truth_at(g.last_sim + p.horizon, 0);
        const double hold = model.source_theta(0);
        sse += (shown - truth) * (shown - truth);
        hold_sse += (hold - truth) * (hold - truth);
        if (have_prev) {
            worst_step = std::max(worst_step, std::fabs(shown - prev_shown));
            worst_true_step = std::max(worst_true_step, std::fabs(truth - prev_true));
        }
        prev_shown = shown; prev_true = truth; have_prev = true;
        ++n;
    }
    const double rms = std::sqrt(sse / n), hold_rms = std::sqrt(hold_sse / n);
    std::printf("  tracking rms %.2f mrad vs hold %.2f mrad; worst 8 ms step %.2f mrad (true %.2f)\n", rms * 1000, hold_rms * 1000,
                worst_step * 1000, worst_true_step * 1000);
    EXPECT(rms < hold_rms * 0.3, "prediction must beat holding the frame by a wide margin");
    EXPECT(worst_step < worst_true_step * 2.0 + 0.002, "no snaps at frame handoffs");
}

static void cursor_gate_blocks_mouse() {
    PoseModel model;
    Game g; g.model = &model;
    for (int i = 0; i < 20000; ++i) g.step(pattern_x(g.t), pattern_y(g.t));
    g.feed_model = false;
    for (int i = 0; i < 5; ++i) g.step(0, 0);
    model.mouse.add(g.t + 0.0005, 400, 0);
    // Manual latency 0 (prediction mode): at the auto setting (-1 frame) the camera is interpolated
    // between two known frames and the mouse is intentionally not used.
    PoseSettings ps; ps.rotation_extrapolation = 0.0; ps.max_horizon = 0.2; ps.auto_fraction = 0.0;
    model.configure(ps);
    model.set_mouse_gate(true);
    const double open = model.predict(g.t + 0.05).yaw;
    model.set_mouse_gate(false);
    const double closed = model.predict(g.t + 0.05).yaw;
    EXPECT(std::fabs(open) > 0.05, "open gate uses the mouse (%f)", open);
    EXPECT(std::fabs(closed) < std::fabs(open) * 0.05, "closed gate ignores the mouse (%f)", closed);
}

static void auto_latency_tracks_frame_interval() {
    PoseModel model;
    Game g; g.model = &model;
    g.fps = 40.0;
    for (int i = 0; i < 6000; ++i) g.step(pattern_x(g.t), pattern_y(g.t));
    EXPECT(std::fabs(model.frame_interval() - 0.025) < 0.0015, "frame interval %.4f (40 fps)", model.frame_interval());
    EXPECT(std::fabs(model.effective_prediction() + model.frame_interval()) < 1e-9, "auto = -1 frame");
    for (double fraction : {0.5, 0.25}) {
        PoseSettings ps; ps.auto_fraction = fraction;
        model.configure(ps);
        EXPECT(std::fabs(model.effective_prediction() + fraction * model.frame_interval()) < 1e-9, "auto = -%.2f frame", fraction);
    }
    model.configure(PoseSettings{});
    double worst = 0, prev = 0; bool have = false;
    for (int i = 0; i < 2000; ++i) {
        g.step(pattern_x(g.t), pattern_y(g.t));
        if (i % 8) continue;
        const Prediction p = model.predict(g.t);
        const double shown = model.source_theta(0) + p.yaw;
        EXPECT(p.horizon <= 0.001, "auto mode never extrapolates beyond the newest frame (h %.4f)", p.horizon);
        if (have) worst = std::max(worst, std::fabs(shown - prev));
        prev = shown; have = true;
    }
    EXPECT(worst < 0.02, "auto mode output continuous (worst step %.4f rad)", worst);
}

// A menu with a cursor the game draws itself: the mouse moves, the camera does not. The mouse must stop
// being applied within a fraction of a second, and apply again once the camera follows it.
static void mouse_ignored_in_menus() {
    PoseModel model;
    Game g; g.model = &model;
    for (int i = 0; i < 20000; ++i) g.step(pattern_x(g.t), pattern_y(g.t));  // learn
    const bool before = model.camera_follows_mouse();
    g.menu = true;
    for (int i = 0; i < 1500; ++i) g.step(pattern_x(g.t), pattern_y(g.t));
    const bool in_menu = model.camera_follows_mouse();
    const double yaw_in_menu = std::fabs(model.predict(g.t).yaw);
    g.menu = false;
    for (int i = 0; i < 1500; ++i) g.step(pattern_x(g.t), pattern_y(g.t));
    const bool after = model.camera_follows_mouse();
    std::printf("mouse in a menu: follows before %d, in the menu %d (predicted turn %.4f rad), after %d\n", before, in_menu,
                yaw_in_menu, after);
    EXPECT(before && !in_menu && after, "mouse ignored only while the camera does not follow it");
    EXPECT(yaw_in_menu < 1e-3, "no predicted turn in the menu (%.4f rad)", yaw_in_menu);
}

// A long stretch where the mouse moves slowly and the camera does not (an inventory with a cursor the game draws,
// movements too small for the "camera follows the mouse" check to judge): the learned gain survives it.
static void gain_survives_menus() {
    PoseModel model;
    Game g; g.model = &model;
    for (int i = 0; i < 20000; ++i) g.step(pattern_x(g.t), pattern_y(g.t));  // learn
    const double before_x = model.params(0).gain, before_y = model.params(1).gain;
    g.menu = true;
    for (int i = 0; i < 45000; ++i) {  // 45 s: the whole fit window
        const bool tick = (static_cast<long>(std::lround(g.t * 1000.0)) % 40) == 0;  // 25 small moves a second
        g.step(tick ? (std::sin(g.t) > 0 ? 1 : -1) : 0, tick ? 1 : 0);
    }
    const double after_x = model.params(0).gain, after_y = model.params(1).gain;
    std::printf("gain through a long menu: yaw %.6g -> %.6g, pitch %.6g -> %.6g\n", before_x, after_x, before_y, after_y);
    EXPECT(std::fabs(after_x / before_x - 1.0) < 0.05 && std::fabs(after_y / before_y - 1.0) < 0.05, "the learned gains survive a long menu");
}

static void world_up_detection() {
    WorldUp up;
    for (int i = 0; i < 20; ++i) up.add(camera_at(i * 0.3, 0.1 * std::sin(i)));
    const Vec3 u = up.get();
    EXPECT(u.z == 1.0 && u.x == 0.0 && u.y == 0.0, "world up %f %f %f", u.x, u.y, u.z);
    // An estimated camera whose up has drifted 30 degrees off the coordinate axes (turning around it,
    // looking up and down): followed, not snapped back to z.
    const double tilt = 30.0 * 3.14159265358979 / 180.0;
    const Vec3 true_up{std::sin(tilt), 0, std::cos(tilt)}, a{std::cos(tilt), 0, -std::sin(tilt)}, b{0, 1, 0};  // a, b span the horizontal
    WorldUp drifted;
    for (int i = 0; i < 400; ++i) {
        const double heading = 0.02 * i, pitch = 0.4 * std::sin(0.05 * i);
        CameraBasis c{};
        const Vec3 level = a * std::cos(heading) + b * std::sin(heading);  // horizontal forward
        c.right = a * std::sin(heading) - b * std::cos(heading);           // horizontal, perpendicular to it
        c.fwd = level * std::cos(pitch) + true_up * std::sin(pitch);
        c.up = true_up * std::cos(pitch) - level * std::sin(pitch);
        drifted.add(c);
    }
    const Vec3 d = drifted.get();
    const double err = std::acos(std::min(1.0, dot(d, true_up))) * 180.0 / 3.14159265358979;
    std::printf("world up of a drifted estimated camera: %.3f %.3f %.3f (true %.3f %.3f %.3f), error %.2f deg\n", d.x, d.y, d.z,
                true_up.x, true_up.y, true_up.z, err);
    EXPECT(err < 0.5, "a drifted up axis is followed (error %.2f deg)", err);
}

// FrameClock: a game simulating at a steady 60 Hz whose presents come late now and then (one in five 6 ms
// late) is timed at its steady pace, never after a frame was presented; a skipped frame starts it again.
static void frame_clock_tests() {
    fw::FrameClock clock;
    double worst = 0, later = 0;
    std::uint64_t frame = 100;
    for (int i = 0; i < 300; ++i, ++frame) {
        const double sim = i / 60.0, presented = sim + 0.002 + (i % 5 == 4 ? 0.006 : 0.0);
        const double t = clock.next(frame, presented);
        if (i > 20) worst = std::max(worst, std::fabs((t - sim) - 0.002 - 0.0012));  // (a steady offset is fine: about the mean delay)
        later = std::max(later, t - presented);
    }
    std::printf("frame clock: off the steady pace by at most %.2f ms, after the present by %.2f ms\n", worst * 1000, later * 1000);
    EXPECT(worst < 0.0015, "late presents do not move the frame times (%.2f ms)", worst * 1000);
    EXPECT(later <= 0.0, "never after the present");
    const double gap = clock.next(frame + 5, 10.0);
    EXPECT(gap == 10.0, "a skipped frame starts again from the presented time");
}

// A game played with a controller: the camera turns at a rate the right stick sets, through a squared
// response curve. The model learns gain and curve; predicting with the held stick beats both holding the
// frame and the same model with the controller option off.
static double stick_x(double t) { return 0.9 * std::sin(t * 0.9) + 0.3 * std::sin(t * 3.1); }
static double stick_y(double t) { return 0.5 * std::sin(t * 0.6 + 1.0); }
static void controller_stick() {
    PoseModel model;
    Game g; g.model = &model;
    g.gain_x = g.gain_y = 0;  // no mouse
    g.stick_gain_x = 2.5; g.stick_gain_y = -1.2;
    for (int i = 0; i < 20000; ++i) { g.sx = stick_x(g.t); g.sy = stick_y(g.t); g.step(0, 0); }
    model.fit();
    const auto& x = model.params(0);
    const auto& y = model.params(1);
    std::printf("  stick fit: yaw %.2f rad/s curve %.0f, pitch %.2f rad/s curve %.0f (true %.2f, %.2f, curve %.0f)\n", x.stick_gain,
                x.stick_power, y.stick_gain, y.stick_power, g.stick_gain_x, g.stick_gain_y, g.stick_power);
    EXPECT(std::fabs(x.stick_gain / g.stick_gain_x - 1.0) < 0.2 && x.stick_power == g.stick_power, "yaw stick gain and curve learned");
    EXPECT(std::fabs(y.stick_gain / g.stick_gain_y - 1.0) < 0.25, "pitch stick gain learned (sign included)");
    EXPECT(x.gain == 0, "no mouse gain from stick-only play (%g)", x.gain);
    // Tracking with the latency the presenter uses in games without HUD layers (1/4 frame behind): two
    // identical sessions (the patterns are deterministic), with the controller option on and off.
    auto track = [&](bool use_stick) {
        PoseModel m;
        Game h; h.model = &m;
        h.gain_x = h.gain_y = 0; h.stick_gain_x = g.stick_gain_x; h.stick_gain_y = g.stick_gain_y;
        PoseSettings ps; ps.auto_fraction = 0.25; ps.use_stick = use_stick;
        m.configure(ps);
        double sse = 0, hold_sse = 0;
        int n = 0;
        for (int i = 0; i < 23000; ++i) {
            h.sx = stick_x(h.t); h.sy = stick_y(h.t); h.step(0, 0);
            if (i < 20000 || i % 8) continue;  // learn, then measure at 125 Hz
            const Prediction p = m.predict(h.t);
            const double shown = m.source_theta(0) + p.yaw, truth = h.truth_at(h.last_sim + p.horizon, 0);
            sse += (shown - truth) * (shown - truth);
            hold_sse += (m.source_theta(0) - truth) * (m.source_theta(0) - truth);
            ++n;
        }
        return std::make_pair(std::sqrt(sse / n), std::sqrt(hold_sse / n));
    };
    const auto with_stick = track(true), without = track(false);
    std::printf("  stick tracking rms %.2f mrad, option off %.2f mrad, holding the frame %.2f mrad\n",
                with_stick.first * 1000, without.first * 1000, with_stick.second * 1000);
    EXPECT(with_stick.first < without.first * 0.5, "the held stick predicts the camera clearly better than without it");
}

// The presenter fits on a worker thread: same parameters as the fit done right away, mouse and stick.
static void worker_fit_matches() {
    auto run = [](bool async_fit, bool stick) {
        PoseModel m;
        Game g; g.model = &m;
        if (stick) { g.stick_gain_x = 2.0; g.stick_gain_y = -1.0; }
        PoseSettings ps; ps.async_fit = async_fit;
        m.configure(ps);
        for (int i = 0; i < 12000; ++i) {
            g.sx = stick_x(g.t); g.sy = stick_y(g.t);
            g.step(pattern_x(g.t), pattern_y(g.t));
            if (async_fit) m.finish_fit();  // (each fit is applied before the next frame, as with the fit done right away)
        }
        return std::make_pair(m.params(0), m.params(1));
    };
    for (bool stick : {false, true}) {
        const auto now = run(false, stick), worker = run(true, stick);
        const bool same = now.first.gain == worker.first.gain && now.first.tau == worker.first.tau && now.first.delay == worker.first.delay &&
                          now.second.gain == worker.second.gain && now.first.stick_gain == worker.first.stick_gain &&
                          now.first.stick_power == worker.first.stick_power && now.second.stick_gain == worker.second.stick_gain;
        EXPECT(same, "worker-thread fit gives the same parameters (%s): yaw gain %.6g vs %.6g, stick %.4g vs %.4g", stick ? "mouse + stick" : "mouse",
               now.first.gain, worker.first.gain, now.first.stick_gain, worker.first.stick_gain);
    }
}

int main() {
    frame_clock_tests();
    rotation_conventions();
    right_handed_projection();
    fit_recovers_smoothed_camera();
    prediction_is_accurate_and_continuous();
    cursor_gate_blocks_mouse();
    auto_latency_tracks_frame_interval();
    world_up_detection();
    mouse_ignored_in_menus();
    gain_survives_menus();
    controller_stick();
    worker_fit_matches();
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("pose tests passed\n");
    return 0;
}
