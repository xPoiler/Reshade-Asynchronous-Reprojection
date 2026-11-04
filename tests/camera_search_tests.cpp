#include "presenter/camera_search.hpp"
#include <cstdio>
#include <cstring>
#include <random>
using namespace fw;

static int g_failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++g_failures; std::printf("FAILED line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

using Matrix = CameraSearch::Matrix;
static constexpr double kW = 1920, kH = 1080;

static Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix r{};
    for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) for (int k = 0; k < 4; ++k) r[y * 4 + x] += a[y * 4 + k] * b[k * 4 + x];
    return r;
}
static Matrix flip(const Matrix& m) { Matrix r; for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) r[y * 4 + x] = m[x * 4 + y]; return r; }
// World to clip, a camera at (x, 0, z) turned by yaw, reversed depth (near 0.1, far infinite).
static Matrix view_projection(float yaw, float pitch, float x, float z) {
    const float c = std::cos(yaw), s = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch), f = 1.0f / std::tan(0.5f), a = float(kW / kH);
    const Matrix view = {c, 0, -s, -(c * x - s * z), 0, 1, 0, 0, s, 0, c, -(s * x + c * z), 0, 0, 0, 1};
    const Matrix tilt = {1, 0, 0, 0, 0, cp, -sp, 0, 0, sp, cp, 0, 0, 0, 0, 1};
    const Matrix projection = {f / a, 0, 0, 0, 0, f, 0, 0, 0, 0, 0, 0.1f, 0, 0, 1, 0};
    return multiply(projection, multiply(tilt, view));
}
// (independent of the search's own arithmetic: a point at depth z, back through this camera to the world, forward through the last one, in doubles)
static std::vector<CameraSearch::Point> motion(const Matrix& now, const Matrix& before, std::mt19937& rng, bool depth_turned) {
    std::uniform_real_distribution<float> depths(0.002f, 0.05f);
    std::normal_distribution<float> noise(0, 0.5f);
    std::vector<CameraSearch::Point> out;
    for (int gy = 0; gy < 12; ++gy)
        for (int gx = 0; gx < 21; ++gx) {
            const float px = float((gx + 0.5) * kW / 21), py = float((gy + 0.5) * kH / 12), z = depths(rng);
            double a[4][5];
            for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) a[y][x] = now[y * 4 + x];
            a[0][4] = 2.0 * px / kW - 1; a[1][4] = 1 - 2.0 * py / kH; a[2][4] = z; a[3][4] = 1;
            for (int c = 0; c < 4; ++c) {
                int p = c;
                for (int r = c + 1; r < 4; ++r) if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
                for (int x = 0; x < 5; ++x) std::swap(a[p][x], a[c][x]);
                const double d = a[c][c];
                for (int x = 0; x < 5; ++x) a[c][x] /= d;
                for (int r = 0; r < 4; ++r) if (r != c) { const double f = a[r][c]; for (int x = 0; x < 5; ++x) a[r][x] -= f * a[c][x]; }
            }
            double c[4] = {};
            for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) c[y] += before[y * 4 + x] * a[x][4];
            const double lx = c[0] / c[3], ly = c[1] / c[3];
            CameraSearch::Point p{px, py, float((lx - (2.0 * px / kW - 1)) * 0.5 * kW) + noise(rng), float(-(ly - (1 - 2.0 * py / kH)) * 0.5 * kH) + noise(rng), depth_turned ? 1 - z : z};
            if (gx > 8 && gx < 12 && gy > 5) p.mx = p.my = 0;  // (a character)
            out.push_back(p);
        }
    return out;
}
static MatrixCandidate candidate(std::uint32_t key, const Matrix& m, bool columns) {
    MatrixCandidate c{};
    c.key = key;
    c.flags = columns ? 3 : 1;
    const Matrix stored = columns ? flip(m) : m;
    std::memcpy(c.m, stored.data(), sizeof(c.m));
    return c;
}
// The camera takes turns between two places, among a light's camera, an object's matrix, and another view in the same place. ahead: each frame's values are
// the next frame's camera (DOOM Eternal binds frame N+1's constants before N is shown).
static void run(bool columns, bool depth_turned, float far_away, bool ahead) {
    CameraSearch search;
    std::mt19937 rng(5);
    const auto camera = [&](int t) { return view_projection(0.02f * t + 0.1f * std::sin(t * 0.2f), 0.1f * std::sin(t * 0.13f), far_away + 0.05f * std::sin(t * 0.3f), far_away + 0.03f * t); };
    const auto frame = [&](int t) {
        std::vector<MatrixCandidate> f;
        f.push_back(candidate(5, view_projection(0.4f, 0.1f, 3, 3), columns));                                     // (stands still)
        f.push_back(candidate(6, view_projection(std::sin(t * 1.7f), std::cos(t * 0.9f), t * 0.7f, 2), columns));  // (wanders)
        const std::uint32_t place = t / 4 % 2 ? 77 : 78;
        f.push_back(candidate(place, view_projection(1.0f + t * 0.3f, 0.2f, 1, 1), columns));                      // (another view, first in that place)
        f.push_back(candidate(place, camera(ahead ? t + 1 : t), columns));
        return f;
    };
    for (int t = 2; t < 60; ++t) search.judge(frame(t - 2), frame(t - 1), frame(t), motion(camera(t), camera(t - 1), rng, depth_turned), kW, kH);
    const auto places = search.places();
    EXPECT(!places.empty(), "columns %d turned %d far %g: no place", int(columns), int(depth_turned), far_away);
    for (const std::uint32_t k : places) EXPECT(k == 77 || k == 78, "columns %d turned %d far %g: place %u taken", int(columns), int(depth_turned), far_away, k);
    EXPECT(search.depth_turned() == depth_turned, "columns %d turned %d far %g: depth the wrong way round", int(columns), int(depth_turned), far_away);
    EXPECT(search.ahead() == ahead, "columns %d turned %d far %g ahead %d: the values' frame taken wrong", int(columns), int(depth_turned), far_away, int(ahead));
}
static void nothing_without_a_camera() {
    CameraSearch search;
    std::mt19937 rng(9);
    Matrix before = view_projection(0, 0, 0, 0);
    std::vector<MatrixCandidate> last = {candidate(5, view_projection(0.4f, 0.1f, 3, 3), false)}, last2 = last;
    for (int t = 1; t < 60; ++t) {
        const Matrix now = view_projection(0.03f * t, 0, 0, 0);
        const std::vector<MatrixCandidate> f = {candidate(5, view_projection(0.4f, 0.1f, 3, 3), false)};
        search.judge(last2, last, f, motion(now, before, rng, false), kW, kH);
        last2 = last;
        last = f;
        before = now;
    }
    EXPECT(search.places().empty(), "a matrix that does not move taken for the camera");
}
// What a found matrix predicts is what its camera does.
static void predicts_the_camera() {
    std::mt19937 rng(11);
    const Matrix before = view_projection(0.1f, 0.02f, 0, 0), now = view_projection(0.13f, 0.03f, 0.02f, 0.05f);
    std::vector<CameraSearch::Point> measured = motion(now, before, rng, false), predicted = measured;
    for (auto& p : predicted) p.mx = p.my = 1000;
    EXPECT(CameraSearch::predict(0, now, before, predicted, kW, kH), "no prediction");
    int off = 0;
    for (std::size_t i = 0; i < measured.size(); ++i)
        if ((measured[i].mx != 0 || measured[i].my != 0) && std::hypot(measured[i].mx - predicted[i].mx, measured[i].my - predicted[i].my) > 3) ++off;
    EXPECT(off == 0, "%d points off", off);
}

// A TAA jitter in both matrices (clip x and y moved by a part of a pixel, new each frame): the prediction is the camera's motion alone, as the game's vectors are.
static void jitter_left_out() {
    std::mt19937 rng(13);
    const Matrix before = view_projection(0.1f, 0.02f, 0, 0), now = view_projection(0.13f, 0.03f, 0.02f, 0.05f);
    const auto jittered = [](Matrix m, double jx, double jy) { for (int c = 0; c < 4; ++c) { m[c] += float(jx * 2 / kW * m[12 + c]); m[4 + c] += float(jy * 2 / kH * m[12 + c]); } return m; };
    std::vector<CameraSearch::Point> clean = motion(now, before, rng, false), predicted = clean;
    EXPECT(CameraSearch::predict(0, now, before, clean, kW, kH) && CameraSearch::predict(0, jittered(now, 0.4, -0.3), jittered(before, -0.4, 0.3), predicted, kW, kH), "no prediction");
    int off = 0;  // (against the pair without jitter: 0.8 px apart in x, 0.6 in y with it left in)
    for (std::size_t i = 0; i < clean.size(); ++i)
        if (std::hypot(clean[i].mx - predicted[i].mx, clean[i].my - predicted[i].my) > 0.05) ++off;
    EXPECT(off == 0, "%d points off by the jitter", off);
}

// What a found matrix says of its camera (turn, move, field of view), put through the estimator's own camera model, moves each point as the matrices do.
static void camera_of_the_matrix() {
    std::mt19937 rng(13);
    for (const bool turned : {false, true}) {
        const Matrix before = view_projection(0.1f, 0.02f, 3000, -2000), now = view_projection(0.14f, 0.035f, 3000.02f, -1999.95f);
        CameraSearch::Motion mo{};
        EXPECT(CameraSearch::motion_of(turned ? 2 : 0, now, before, mo), "turned %d: no camera in the matrix", int(turned));
        EXPECT(std::fabs(mo.fov - 1.0) < 1e-4, "turned %d: field of view %f, made with 1", int(turned), mo.fov);
        std::vector<CameraSearch::Point> points = motion(now, before, rng, turned), exact = points;
        CameraSearch::predict(turned ? 2 : 0, now, before, exact, kW, kH);
        const double f = (kH * 0.5) / std::tan(mo.fov * 0.5);
        int off = 0;
        for (const auto& p : exact) {
            const double z = 1.0 / (turned ? 1.0 - p.depth : p.depth);  // (in near planes: depth is near / distance)
            const double c[3] = {(p.x - kW * 0.5) / f * z, -(p.y - kH * 0.5) / f * z, z};
            double q[3];
            for (int i = 0; i < 3; ++i) q[i] = mo.turn[i][0] * c[0] + mo.turn[i][1] * c[1] + mo.turn[i][2] * c[2] + mo.move[i];
            const double mx = kW * 0.5 + f * q[0] / q[2] - p.x, my = kH * 0.5 - f * q[1] / q[2] - p.y;
            if (std::hypot(mx - p.mx, my - p.my) > 0.05) ++off;
        }
        EXPECT(off == 0, "turned %d: %d of %zu points moved otherwise than the matrices move them", int(turned), off, exact.size());
    }
}

int main() {
    predicts_the_camera();
    jitter_left_out();
    camera_of_the_matrix();
    for (int columns = 0; columns < 2; ++columns)
        for (int turned = 0; turned < 2; ++turned)
            for (const float far_away : {0.0f, 5000.0f})
                for (const bool ahead : {false, true}) run(columns != 0, turned != 0, far_away, ahead);
    nothing_without_a_camera();
    if (g_failures) { std::printf("%d failed\n", g_failures); return 1; }
    std::printf("camera search ok\n");
    return 0;
}
