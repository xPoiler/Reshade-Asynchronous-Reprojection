#pragma once
// Is the game's own camera usable? Its frame-to-frame reprojection (with depth) says how every pixel of
// the static scene moves; the game's motion vectors say the same thing independently. In a game whose
// camera data is right the two agree almost exactly whenever the camera moves (one scale maps one to the
// other); in a game whose camera data is not what it claims to be they never do, and everything built on
// it - the warp's parallax, the character/weapon mask, the background memory - is wrong. Then the camera
// is estimated from the motion vectors instead, as in games that send no camera at all.
#include <cmath>

namespace fw {

struct CameraCheck {
    enum Verdict { kUndecided, kTrusted, kUnusable };
    static constexpr int kWindow = 150;            // game frames with real camera motion per decision
    static constexpr double kMinMotion = 1e-6;     // mean squared camera motion (uv^2): about 2 px at 1080p
    static constexpr double kConsistent = 0.5;     // share of the camera motion the motion vectors explain
    int frames = 0, consistent = 0;
    Verdict verdict = kUndecided;

    // How much of the camera-only motion c one scale of the motion vectors g explains (1: all of it), from
    // the least-squares sums of one game frame. False: the frame says nothing (camera still, too few pixels).
    static bool quality(const double gc[2], const double gg[2], const double cc[2], double samples, double& q) {
        if (samples < 1000 || !(cc[0] + cc[1] > 0) || (cc[0] + cc[1]) / samples < kMinMotion) return false;
        double residual = 0;
        for (int k = 0; k < 2; ++k) residual += gg[k] > 0 ? cc[k] - gc[k] * gc[k] / gg[k] : cc[k];
        q = 1.0 - residual / (cc[0] + cc[1]);
        return std::isfinite(q);
    }

    // One game frame's sums; scale_locked: the motion vector scale has been found (several frames fitted
    // almost perfectly), which settles it. Unusable only after a whole window of camera motion in which
    // fewer than one frame in ten was consistent, and never once the camera has been trusted.
    Verdict add(const double gc[2], const double gg[2], const double cc[2], double samples, bool scale_locked) {
        if (verdict != kUndecided) return verdict;
        if (scale_locked) return verdict = kTrusted;
        double q = 0;
        if (!quality(gc, gg, cc, samples, q)) return verdict;
        ++frames;
        if (q >= kConsistent) ++consistent;
        if (frames >= kWindow) {
            if (consistent * 10 < frames) return verdict = kUnusable;
            frames = consistent = 0;
        }
        return verdict;
    }
};

}  // namespace fw
