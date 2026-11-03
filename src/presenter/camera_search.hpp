// Finds a game's own camera among the matrices in its constant buffers, by what they do: a matrix of the shape of world to clip (camera_search.cpp in the add-on), with last
// frame's value of the same place, gives the motion the camera made at each point of the picture, from the depth there. Against the motion measured, in pixels: a camera's
// predicts it with nothing to fit, a light's, a bone's or an object's that moves does not. What is measured on Cyberpunk, on every value of 9300 frames (dev/camera_dump*.py): the
// camera lies among some 900 matrices a frame, in several places that take turns, and a place can hold other values in a frame besides. So every value of a place is tried with
// every value of that place in the frame before, and the places count wins: the 8 that win most covered 72% of the frames that move, all of them 73%, with 23 pairs a frame to try.
#pragma once
#include "shared/protocol.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <unordered_map>
#include <vector>

namespace fw {

class CameraSearch {
public:
    using Matrix = std::array<float, 16>;
    struct Point { float x, y, mx, my, depth; };  // pixels; the motion to the last frame; the depth as stored
    static constexpr std::size_t kPlaces = 8;
    static constexpr float kMovingPx = 2.0f;      // the median motion of a frame that counts
    static constexpr float kWin = 0.9f;           // (a stopgap: a value that predicts the frame this well wins it for its place)
    static constexpr int kWinsNeeded = 3;
    static constexpr std::size_t kPairs = 4000;   // (a frame at most: 1000 matrices, each with what its place held)

    // One frame that moved, with the one before it: which places hold a value that predicts it. Slow: on its own thread.
    void judge(const std::vector<MatrixCandidate>& before, const std::vector<MatrixCandidate>& now, const std::vector<Point>& points, double w, double h) {
        std::unordered_map<std::uint32_t, std::vector<const MatrixCandidate*>> last;
        for (const MatrixCandidate& c : before) last[c.key].push_back(&c);
        std::unordered_map<std::uint32_t, float> best[2];
        std::size_t pairs = 0;
        for (const MatrixCandidate& c : now) {
            const auto it = last.find(c.key);
            if (it == last.end() || !(c.flags & 1)) continue;
            for (const MatrixCandidate* l : it->second) {
                if (pairs++ >= kPairs) break;
                for (int turned = 0; turned < 2; ++turned) {
                    const float q = quality(form_of(c, turned != 0), matrix_of(c), matrix_of(*l), points, w, h);
                    if (const auto b = best[turned].find(c.key); b == best[turned].end() || q > b->second) best[turned][c.key] = q;
                }
            }
        }
        // (did the places that win so far predict this frame? If not, the add-on searches whole frames until one does)
        const std::vector<std::uint32_t> held = places();
        const auto& by = best[depth_turned() ? 1 : 0];
        bool hit = false;
        for (const std::uint32_t key : held) if (const auto b = by.find(key); b != by.end() && b->second >= kWin) hit = true;
        held_state_ = held.empty() ? 0 : hit ? 1 : 2;
        float top = -1;
        for (int turned = 0; turned < 2; ++turned)
            for (const auto& [key, q] : best[turned]) { top = std::max(top, q); if (q >= kWin) ++wins_[turned][key]; }
        tops_.push_back(top);
        paired_.push_back(float(best[0].size()));
        tried_.push_back(float(std::min(pairs, kPairs)));
        // (the old fades: places change with the scene)
        if (++judged_ % 200 == 0) for (auto& w2 : wins_) for (auto it = w2.begin(); it != w2.end();) { it->second /= 2; it = it->second ? std::next(it) : w2.erase(it); }
    }
    // The places that win most, the depth as it wins.
    bool depth_turned() const { return total(1) > total(0); }
    std::vector<std::uint32_t> places() const {
        const auto& wins = wins_[depth_turned() ? 1 : 0];
        std::vector<std::pair<int, std::uint32_t>> order;
        for (const auto& [key, n] : wins) if (n >= kWinsNeeded) order.push_back({n, key});
        std::sort(order.rbegin(), order.rend());
        std::vector<std::uint32_t> out;
        for (std::size_t i = 0; i < order.size() && i < kPlaces; ++i) out.push_back(order[i].second);
        return out;
    }
    int judged() const { return judged_; }
    int held_state() const { return held_state_; }  // (Shared::camera_held_state)
    // (diagnostics, since the last call: a frame's best value (median), frames where one reached kWin, places with a value in both frames and pairs tried (medians))
    struct Seen { float top, paired, tried; int winning, frames; };
    Seen seen() {
        const auto mid = [](std::vector<float>& v) { if (v.empty()) return 0.0f; std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
        Seen s{0, 0, 0, int(std::count_if(tops_.begin(), tops_.end(), [](float q) { return q >= kWin; })), int(tops_.size())};
        s.top = mid(tops_); s.paired = mid(paired_); s.tried = mid(tried_);
        tops_.clear(); paired_.clear(); tried_.clear();
        return s;
    }

    static Matrix matrix_of(const MatrixCandidate& c) { Matrix r; std::copy(c.m, c.m + 16, r.begin()); return r; }
    // A world to clip matrix holds its whole camera: forward is the row of clip w, the position the one point that row and those of clip x and y send to 0, right and up
    // those rows without their forward part (what remains is jitter and the centre), the field of view the length of up's. The turn and move from the last one in
    // CameraEstimator's terms: current camera to previous, x right, y up, z forward, the move in near planes - the estimator's unit, as it reads depth as near / distance.
    // The near plane is where depth x distance, a straight line along any view, meets distance 0 (at -near with depth stored the other way round). False: not a camera's.
    struct Motion { std::array<std::array<double, 3>, 3> turn; std::array<double, 3> move; double fov; };
    static bool motion_of(int form, const Matrix& now, const Matrix& last, Motion& out) {
        struct View { std::array<double, 3> right, up, fwd, pos; double sy, plane; };
        const auto view = [form](const Matrix& m, View& v) {
            const Exact e = exact(m, (form & 1) != 0);
            const auto row = [&](int r) { return std::array<double, 3>{e[r * 4], e[r * 4 + 1], e[r * 4 + 2]}; };
            const auto dot = [](const std::array<double, 3>& a, const std::array<double, 3>& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
            const auto unit = [&](std::array<double, 3> a, double& length) { length = std::sqrt(dot(a, a)); for (double& x : a) x /= length; return a; };
            double l;
            v.fwd = unit(row(3), l);
            const auto across = [&](int r) { std::array<double, 3> a = row(r); const double d = dot(a, v.fwd); for (int i = 0; i < 3; ++i) a[i] -= d * v.fwd[i]; return a; };
            double sx;
            v.up = unit(across(1), v.sy);
            std::array<double, 3> right = across(0);
            const double d = dot(right, v.up);
            for (int i = 0; i < 3; ++i) right[i] -= d * v.up[i];
            v.right = unit(right, sx);
            if (!(v.sy > 1e-6 && sx > 1e-6 && l > 0)) return false;
            // (the position: clip x, y and w all 0 there)
            Exact a{};
            for (int c = 0; c < 3; ++c) { a[c] = e[c]; a[4 + c] = e[4 + c]; a[8 + c] = e[12 + c]; }
            a[15] = 1;
            Exact inv;
            if (!inverse(a, inv)) return false;
            const double b[3] = {-e[3], -e[7], -e[15]};
            for (int i = 0; i < 3; ++i) v.pos[i] = inv[i * 4] * b[0] + inv[i * 4 + 1] * b[1] + inv[i * 4 + 2] * b[2];
            const auto depth_times_distance = [&](double z) {
                const std::array<double, 3> p{v.pos[0] + v.fwd[0] * z, v.pos[1] + v.fwd[1] * z, v.pos[2] + v.fwd[2] * z};
                const double cz = e[8] * p[0] + e[9] * p[1] + e[10] * p[2] + e[11], cw = e[12] * p[0] + e[13] * p[1] + e[14] * p[2] + e[15];
                const double depth = (form & 2) ? 1.0 - cz / cw : cz / cw;
                return depth * z;
            };
            v.plane = std::fabs(2.0 * depth_times_distance(1.0) - depth_times_distance(2.0));  // (the depth stored the other way round: 1 - near / distance, the line meets 0 at -near)
            return v.plane > 0 && std::isfinite(v.plane);
        };
        View n, p;
        if (!view(now, n) || !view(last, p)) return false;
        const std::array<std::array<double, 3>, 3> bn{{{n.right[0], n.up[0], n.fwd[0]}, {n.right[1], n.up[1], n.fwd[1]}, {n.right[2], n.up[2], n.fwd[2]}}};
        const std::array<std::array<double, 3>, 3> bp{{{p.right[0], p.up[0], p.fwd[0]}, {p.right[1], p.up[1], p.fwd[1]}, {p.right[2], p.up[2], p.fwd[2]}}};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) { out.turn[i][j] = 0; for (int k = 0; k < 3; ++k) out.turn[i][j] += bp[k][i] * bn[k][j]; }  // (previous basis, transposed, by this one)
            out.move[i] = 0;
            for (int k = 0; k < 3; ++k) out.move[i] += bp[k][i] * (n.pos[k] - p.pos[k]) / n.plane;
        }
        out.fov = 2.0 * std::atan(1.0 / n.sy);
        for (int i = 0; i < 3; ++i) if (!std::isfinite(out.move[i])) return false;
        return std::isfinite(out.fov);
    }
    static int form_of(const MatrixCandidate& c, bool turned) { return ((c.flags & 2) ? 1 : 0) | (turned ? 2 : 0); }
    // The motion this matrix gives over these points, in their place. form: 1 stored in columns, 2 the depth turned. False: no.
    static bool predict(int form, const Matrix& now, const Matrix& last, std::vector<Point>& points, double w, double h) {
        Exact t;
        if (!to_last(form, now, last, t)) return false;
        for (Point& p : points) {
            double mx, my;
            if (!motion(t, form, p, w, h, mx, my)) return false;
            p.mx = float(mx); p.my = float(my);
        }
        return true;
    }
    // 1: the motion predicted is the motion measured, 0: as far off as there is motion at all (the median of each).
    static float quality(int form, const Matrix& now, const Matrix& last, const std::vector<Point>& points, double w, double h) {
        Exact t;
        if (points.empty() || !to_last(form, now, last, t)) return -1e6f;
        std::vector<float> errors, sizes;
        errors.reserve(points.size());
        sizes.reserve(points.size());
        for (const Point& p : points) {
            double mx, my;
            if (!motion(t, form, p, w, h, mx, my)) return -1e6f;
            errors.push_back(float(std::hypot(mx - p.mx, my - p.my)));
            sizes.push_back(std::hypot(p.mx, p.my));
        }
        const float size = median(sizes);
        return size > 0 ? 1.0f - median(errors) / size : 0.0f;
    }
    static float median(std::vector<float>& v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }

private:
    // (in doubles: in a world some thousands from its origin, last frame's matrix by this frame's inverse is nearly the identity, a difference of large numbers)
    using Exact = std::array<double, 16>;
    static Exact exact(const Matrix& m, bool columns) {
        Exact r;
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) r[y * 4 + x] = columns ? m[x * 4 + y] : m[y * 4 + x];
        return r;
    }
    static bool inverse(const Exact& m, Exact& out) {
        double a[4][8];
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) { a[y][x] = m[y * 4 + x]; a[y][x + 4] = x == y; }
        for (int c = 0; c < 4; ++c) {
            int p = c;
            for (int r = c + 1; r < 4; ++r) if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
            if (std::fabs(a[p][c]) < 1e-18) return false;
            if (p != c) for (int x = 0; x < 8; ++x) std::swap(a[p][x], a[c][x]);
            const double d = a[c][c];
            for (int x = 0; x < 8; ++x) a[c][x] /= d;
            for (int r = 0; r < 4; ++r) {
                if (r == c) continue;
                const double f = a[r][c];
                for (int x = 0; x < 8; ++x) a[r][x] -= f * a[c][x];
            }
        }
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) out[y * 4 + x] = a[y][x + 4];
        return true;
    }
    static bool to_last(int form, const Matrix& now, const Matrix& last, Exact& out) {  // (this frame's clip space to the last frame's)
        Exact inverted;
        if (!inverse(exact(now, (form & 1) != 0), inverted)) return false;
        const Exact before = exact(last, (form & 1) != 0);
        out = Exact{};
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) for (int k = 0; k < 4; ++k) out[y * 4 + x] += before[y * 4 + k] * inverted[k * 4 + x];
        return true;
    }
    static bool motion(const Exact& t, int form, const Point& p, double w, double h, double& mx, double& my) {
        const double x = 2.0 * p.x / w - 1.0, y = 1.0 - 2.0 * p.y / h, z = (form & 2) ? 1.0 - p.depth : p.depth;
        const double px = t[0] * x + t[1] * y + t[2] * z + t[3], py = t[4] * x + t[5] * y + t[6] * z + t[7], pw = t[12] * x + t[13] * y + t[14] * z + t[15];
        if (std::fabs(pw) < 1e-9) return false;
        mx = (px / pw - x) * 0.5 * w;
        my = -(py / pw - y) * 0.5 * h;
        return std::isfinite(mx) && std::isfinite(my);
    }
    long long total(int turned) const { long long n = 0; for (const auto& [key, wins] : wins_[turned]) n += wins; return n; }

    std::map<std::uint32_t, int> wins_[2];  // (by the depth's way round)
    int judged_ = 0;
    int held_state_ = 0;
    std::vector<float> tops_, paired_, tried_;
};

}  // namespace fw
