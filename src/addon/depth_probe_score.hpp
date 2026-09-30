#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace fw {
// Does a depth buffer belong to the picture (depth_choice.hpp)? From a small probe of both: how much of
// it has depth at all, and how much stronger the picture's brightness changes across depth edges than
// elsewhere (1 = no relation: another view's depth, a shadow map). px: depth (near = 1, far = 0; or the
// other way round, which is tried as well) and brightness per probe pixel.
struct DepthProbeScore {
    double coverage = 0, edges = 0, lift = 0, score = 0;
};
inline DepthProbeScore score_depth_probe(const float* px, std::uint32_t w, std::uint32_t h) {
    DepthProbeScore best;
    if (!px || w < 8 || h < 8) return best;
    // (the depth as handed over, and turned round: ReShade's "reversed" setting may be the wrong way for the game)
    for (int turned = 0; turned < 2; ++turned) {
        auto depth = [&](std::uint32_t x, std::uint32_t y) {
            const float d = px[(std::size_t(y) * w + x) * 2];
            if (!std::isfinite(d)) return 0.0f;
            return std::clamp(turned ? 1.0f - d : d, 0.0f, 1.0f);
        };
        auto luma = [&](std::uint32_t x, std::uint32_t y) {
            const float l = px[(std::size_t(y) * w + x) * 2 + 1];
            return std::isfinite(l) ? l : 0.0f;
        };
        double covered = 0, at_edges = 0, elsewhere = 0;
        std::size_t edges = 0, others = 0;
        for (std::uint32_t y = 0; y < h; ++y)
            for (std::uint32_t x = 0; x < w; ++x) {
                const float d = depth(x, y), l = luma(x, y);
                if (d > 1e-6f) covered += 1;
                for (int dir = 0; dir < 2; ++dir) {
                    const std::uint32_t nx = x + (dir == 0), ny = y + (dir == 1);
                    if (nx >= w || ny >= h) continue;
                    const float e = depth(nx, ny);
                    const double change = std::fabs(double(l) - double(luma(nx, ny)));
                    // a depth edge: the distance changes by more than 15% from one probe pixel to the next
                    if (std::fabs(d - e) > 0.15f * std::max(std::max(d, e), 1e-9f)) { at_edges += change; ++edges; }
                    else { elsewhere += change; ++others; }
                }
            }
        DepthProbeScore s;
        s.coverage = covered / (double(w) * h);
        s.edges = double(edges) / double(std::max<std::size_t>(edges + others, 1));
        if (edges >= 40 && others >= 40) s.lift = (at_edges / double(edges)) / std::max(elsewhere / double(others), 1e-6);
        s.score = std::max(s.lift - 1.0, 0.0) * std::min(1.0, s.coverage / 0.5);
        if (s.score > best.score || turned == 0) best = s;
    }
    return best;
}
}  // namespace fw
