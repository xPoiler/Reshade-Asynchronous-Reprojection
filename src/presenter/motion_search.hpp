// Finds a game's own motion vectors among its render targets, by nothing but what they hold: in a game that hands none to an upscaler,
// each candidate's channels are compared with the motion measured from the pictures, on the camera estimation's grid. For each channel
// and each axis, reference = scale * channel + offset is fitted over the frames that move: a motion channel explains the reference, whatever
// its format, its units and its sign; anything else does not. A character does not follow the rest: the fit is taken again on
// what the first explains. Decided over many frames: one frame's fit can be luck, a real channel is right every time, at one scale.
// Within a frame a plain turn shows nothing: the whole picture moves alike, and the offset takes all of it. So each axis is also fitted across frames, on the
// frames' medians: a motion channel follows the turn, both ways, through one scale and one offset.
// A game can keep last frame's motion too, for its temporal passes: that follows the turn as well, a frame behind, and warps everything a frame off. Fitted against
// the turn of the frame before as well, for the log: a target that follows that one better reaches XPAR a frame late. Never a reason to drop it: Enshrouded's
// only motion target was dropped so, and the search found nothing.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace fw {

class MotionSearch {
public:
    struct Sample { float rx, ry, valid; float c[4]; };
    struct Found { bool found = false; std::uint32_t candidate = 0; int channel_x = 0, channel_y = 1; float scale_x = 0, scale_y = 0, quality = 0; };
    static constexpr int kFramesNeeded = 12;      // frames that move, per candidate, before anything is decided
    static constexpr float kMovingPx = 2.0f;      // the median reference motion of a frame that counts
    static constexpr float kQuality = 0.6f;       // (a stopgap, tuned on RE2: real channels 0.71-0.91 a frame, the best other 0.54)
    static constexpr float kNothing = 0.15f;      // (a stopgap too: others scored -0.29 to 0.11 in four games)
    static constexpr float kAbove = 3.0f;         // (how far above the others)
    static constexpr double kUnitsOff = 1.5;      // (a stopgap: how far a measured scale may lie from a whole unit)
    static constexpr float kScaleSpread = 0.15f;  // ...and a stopgap too: how far the scales of a channel's frames may lie apart

    // One frame's grid of one candidate. true: this frame counted.
    bool add(std::uint32_t candidate, const std::vector<Sample>& samples) {
        std::vector<const Sample*> used;
        used.reserve(samples.size());
        for (const Sample& s : samples) if (s.valid > 0.5f && std::isfinite(s.rx) && std::isfinite(s.ry)) used.push_back(&s);
        if (used.size() < 64) return false;
        ++calls_;
        Candidate& k = candidates_[candidate];
        float now[4], turn[2];
        {
            std::vector<float> v(used.size());
            for (int ch = 0; ch < 4; ++ch) {
                for (std::size_t i = 0; i < used.size(); ++i) v[i] = used[i]->c[ch];
                now[ch] = median(v);
            }
            for (int axis = 0; axis < 2; ++axis) {
                for (std::size_t i = 0; i < used.size(); ++i) v[i] = axis ? used[i]->ry : used[i]->rx;
                turn[axis] = median(v);
            }
        }
        const bool follows = k.last_call + 1 == calls_;
        float turned[2];
        std::copy(k.last_turn, k.last_turn + 2, turned);
        std::copy(turn, turn + 2, k.last_turn);
        k.last_call = calls_;
        std::vector<float> size;
        size.reserve(used.size());
        for (const Sample* s : used) size.push_back(std::hypot(s->rx, s->ry));
        std::nth_element(size.begin(), size.begin() + size.size() / 2, size.end());
        if (size[size.size() / 2] < kMovingPx) return false;
        ++k.frames;
        for (int axis = 0; axis < 2; ++axis) {
            k.reference[axis].push_back(turn[axis]);
            if (follows) { k.reference_now[axis].push_back(turn[axis]); k.reference_before[axis].push_back(turned[axis]); }
        }
        for (int ch = 0; ch < 4; ++ch) {
            k.medians[ch].push_back(now[ch]);
            if (follows) k.medians_now[ch].push_back(now[ch]);
        }
        for (int ch = 0; ch < 4; ++ch)
            for (int axis = 0; axis < 2; ++axis) {
                const Fit f = fit(used, ch, axis);
                if (f.ok) k.fits[ch][axis].push_back(f);
            }
        return true;
    }
    // The candidate that explains both axes by two different channels, the best of those.
    Found decide() const {
        Found best;
        for (const auto& [id, k] : candidates_) {
            if (k.frames < kFramesNeeded) continue;
            Channel x[4], y[4];
            for (int ch = 0; ch < 4; ++ch) { x[ch] = channel(k, ch, 0); y[ch] = channel(k, ch, 1); }
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) {
                    if (a == b || !x[a].good || !y[b].good) continue;
                    const float q = std::min(x[a].quality, y[b].quality);
                    if (!best.found || q > best.quality) best = {true, id, a, b, x[a].scale, y[b].scale, q};
                }
        }
        if (best.found) return best;
        // How good a fit can get depends on how good the picture's motion is, not only on the target: Enshrouded's own stayed at 0.58 and 0.62 over 2500 frames, every other
        // target at ~0. So over many frames: far above all the others that were tried, at a steady scale.
        for (const auto& [id, k] : candidates_) {
            if (k.frames < kFramesNeeded * 8) continue;
            float others = 0;
            int tried = 0;
            for (const auto& [other, o] : candidates_) {
                if (other == id || o.frames < kFramesNeeded) continue;
                float q[2];
                best_so_far(other, q);
                others = std::max({others, q[0], q[1]});
                ++tried;
            }
            if (tried < 2) continue;
            Channel x[4], y[4];
            for (int ch = 0; ch < 4; ++ch) { x[ch] = channel(k, ch, 0); y[ch] = channel(k, ch, 1); }
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) {
                    if (a == b || !x[a].steady || !y[b].steady) continue;
                    const float q = std::min(x[a].quality, y[b].quality);
                    if (q >= kAbove * std::max(others, kNothing) && (!best.found || q > best.quality)) best = {true, id, a, b, x[a].scale, y[b].scale, q};
                }
        }
        return best;
    }
    // Follows the frame before better than this one, on the axis that moved most.
    bool late(std::uint32_t candidate) const {
        float now = 0, before = 0;
        timing(candidate, now, before);
        return before > now;
    }
    // For the log: how well its best channel follows this frame's turn, and the frame before's (0 both: no turns both ways yet).
    void timing(std::uint32_t candidate, float& now, float& before) const {
        now = before = 0;
        const auto it = candidates_.find(candidate);
        if (it == candidates_.end()) return;
        const Candidate& k = it->second;
        for (int axis = 0; axis < 2; ++axis)
            for (int ch = 0; ch < 4; ++ch) {
                const Channel a = across_of(k.reference_now[axis], k.medians_now[ch]), b = across_of(k.reference_before[axis], k.medians_now[ch]);
                if (std::max(a.quality, b.quality) > std::max(now, before)) { now = a.quality; before = b.quality; }
            }
    }
    // A fit against a noisy picture leaves the scale a few percent off (Enshrouded: 1774 to 1917 x 968 to 1121 over five runs, its upscaler is told 1920 x 1080). A game
    // stores motion in whole units of its picture, so the nearest of those is the true one: uv, clip, or either in the picture's height (Armored Core: 1083 x 1079).
    // A scale near none of them is kept as measured. (Off by as much as 1478 x 1078 at 0.63 in Enshrouded: the picture's motion falls short in fast turns.)
    static void to_units(Found& f, double w, double h) {
        const double units[4][2] = {{w, h}, {w * 0.5, h * 0.5}, {h, h}, {h * 0.5, h * 0.5}};
        const double limit = std::log(kUnitsOff);
        double best = 2 * limit;
        const double* unit = nullptr;
        for (const auto& u : units) {
            const double x = std::fabs(std::log(std::fabs(f.scale_x) / u[0])), y = std::fabs(std::log(std::fabs(f.scale_y) / u[1]));
            if (x <= limit && y <= limit && x + y < best) { best = x + y; unit = u; }
        }
        if (!unit) return;
        f.scale_x = float(f.scale_x < 0 ? -unit[0] : unit[0]);
        f.scale_y = float(f.scale_y < 0 ? -unit[1] : unit[1]);
    }
    // Ruled out: enough frames, and nothing in it shows any motion. So that the turns go to the others. (Not by the pass mark: Enshrouded's motion scored 0.23
    // and 0.35 over a slow first 12 frames, and 0.82 another time - ruled out, it was never tried again in that session. What is not motion scores ~0)
    bool ruled_out(std::uint32_t candidate) const {
        const auto it = candidates_.find(candidate);
        if (it == candidates_.end() || it->second.frames < kFramesNeeded) return false;
        // (only once the turns went both ways on an axis: over plain turns a frame's own fit sees nothing, whatever the target - Enshrouded's motion at 0.12 and 0.00, ruled out)
        bool turned_both_ways = false;
        for (int axis = 0; axis < 2; ++axis) {
            int forth = 0, back = 0;
            for (const float r : it->second.reference[axis]) { forth += r >= kMovingPx ? 1 : 0; back += r <= -kMovingPx ? 1 : 0; }
            turned_both_ways = turned_both_ways || std::min(forth, back) >= kFramesNeeded;
        }
        if (!turned_both_ways) return false;
        float q[2];
        best_so_far(candidate, q);
        return q[0] < kNothing && q[1] < kNothing;
    }
    // For the log: its best channels so far, whatever the verdict (x, y: their quality; -1 none).
    void best_so_far(std::uint32_t candidate, float out[2]) const {
        out[0] = out[1] = -1;
        const auto it = candidates_.find(candidate);
        if (it == candidates_.end()) return;
        for (int ch = 0; ch < 4; ++ch)
            for (int axis = 0; axis < 2; ++axis) out[axis] = std::max(out[axis], channel(it->second, ch, axis).quality);
    }
    // For the log: of its best channels (x, y), the share of frames near their scale.
    void steadiness(std::uint32_t candidate, float close[2], float roughly[2]) const {
        close[0] = close[1] = roughly[0] = roughly[1] = 0;
        const auto it = candidates_.find(candidate);
        if (it == candidates_.end()) return;
        for (int axis = 0; axis < 2; ++axis) {
            float best = -2;
            for (int ch = 0; ch < 4; ++ch) {
                const Channel c = channel(it->second, ch, axis);
                if (c.quality > best) { best = c.quality; close[axis] = c.close; roughly[axis] = c.roughly; }
            }
        }
    }
    int frames(std::uint32_t candidate) const { const auto it = candidates_.find(candidate); return it == candidates_.end() ? 0 : it->second.frames; }
    void reset() { candidates_.clear(); }

private:
    struct Fit { bool ok = false; float scale = 0, quality = 0; };
    struct Channel { bool good = false, steady = false; float scale = 0, quality = 0, close = 0, roughly = 0; };
    struct Candidate {
        int frames = 0;
        std::vector<Fit> fits[4][2];
        std::vector<float> reference[2], medians[4];
        std::vector<float> reference_now[2], reference_before[2], medians_now[4];  // (frames whose frame before was seen too)
        float last_turn[2] = {};
        std::uint64_t last_call = 0;
    };

    static float median(std::vector<float>& v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }
    // reference = scale * channel + offset by least squares, over what `keep` lets through.
    static bool solve(const std::vector<const Sample*>& s, int ch, int axis, const std::vector<char>* keep, double& scale, double& offset) {
        double n = 0, c = 0, r = 0, cc = 0, cr = 0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (keep && !(*keep)[i]) continue;
            const double x = s[i]->c[ch], y = axis ? s[i]->ry : s[i]->rx;
            n += 1; c += x; r += y; cc += x * x; cr += x * y;
        }
        const double det = n * cc - c * c;
        if (n < 16 || det <= 1e-12 * n * n) return false;
        scale = (n * cr - c * r) / det;
        offset = (r - scale * c) / n;
        return std::isfinite(scale) && std::isfinite(offset);
    }
    static Fit fit(const std::vector<const Sample*>& s, int ch, int axis) {
        Fit f;
        double scale = 0, offset = 0;
        if (!solve(s, ch, axis, nullptr, scale, offset)) return f;
        std::vector<float> errors(s.size());
        auto measure = [&]() {
            for (std::size_t i = 0; i < s.size(); ++i)
                errors[i] = float(std::fabs(scale * s[i]->c[ch] + offset - (axis ? s[i]->ry : s[i]->rx)));
        };
        measure();
        std::vector<float> e = errors;
        const float limit = median(e) * 3.0f + 0.5f;
        std::vector<char> keep(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) keep[i] = errors[i] <= limit;
        if (!solve(s, ch, axis, &keep, scale, offset)) return f;
        measure();
        std::vector<float> spread(s.size());
        std::vector<float> values(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) values[i] = axis ? s[i]->ry : s[i]->rx;
        std::vector<float> sorted = values;
        const float centre = median(sorted);
        for (std::size_t i = 0; i < s.size(); ++i) spread[i] = std::fabs(values[i] - centre);
        const float base = median(spread);
        f.quality = 1.0f - median(errors) / std::max(base, 1e-3f);
        f.scale = float(scale);
        f.ok = true;
        return f;
    }
    // Good: most of the candidate's frames fit, and at one scale.
    static Channel summary(const std::vector<Fit>& fits, int frames) {
        Channel c;
        if (int(fits.size()) * 2 < frames || fits.empty()) return c;
        std::vector<float> scales, qualities;
        for (const Fit& f : fits) { scales.push_back(f.scale); qualities.push_back(f.quality); }
        std::vector<float> sorted = scales;
        c.scale = median(sorted);
        c.quality = median(qualities);
        if (c.scale == 0) return c;
        int close = 0, roughly = 0;  // (roughly: for a target judged against the others - a noisy picture spreads each frame's scale, the median of many stays right)
        for (const float s : scales) {
            close += std::fabs(s - c.scale) <= kScaleSpread * std::fabs(c.scale) ? 1 : 0;
            roughly += std::fabs(s - c.scale) <= 2.0f * kScaleSpread * std::fabs(c.scale) ? 1 : 0;
        }
        c.close = float(close) / float(scales.size());
        c.roughly = float(roughly) / float(scales.size());
        c.steady = roughly * 4 >= int(scales.size()) * 3;
        c.good = c.quality >= kQuality && close * 4 >= int(scales.size()) * 3;
        return c;
    }
    // Across frames: the frames' medians, one scale, one offset. Only over turns both ways: anything that drifts follows a turn one way.
    static Channel across(const Candidate& k, int ch, int axis) { return across_of(k.reference[axis], k.medians[ch]); }
    static Channel across_of(const std::vector<float>& reference, const std::vector<float>& medians) {
        Channel c;
        int forth = 0, back = 0;
        for (const float r : reference) { forth += r >= kMovingPx ? 1 : 0; back += r <= -kMovingPx ? 1 : 0; }
        if (std::min(forth, back) < kFramesNeeded) return c;
        std::vector<Sample> frames(reference.size());
        std::vector<const Sample*> used(reference.size());
        for (std::size_t i = 0; i < frames.size(); ++i) {
            frames[i].rx = reference[i];
            frames[i].c[0] = medians[i];
            used[i] = &frames[i];
        }
        const Fit f = fit(used, 0, 0);
        if (!f.ok || f.scale == 0) return c;
        c.scale = f.scale;
        c.quality = f.quality;
        c.good = c.steady = f.quality >= kQuality;
        c.close = c.roughly = 1;
        return c;
    }
    // The better of the two.
    static Channel channel(const Candidate& k, int ch, int axis) {
        const Channel within = summary(k.fits[ch][axis], k.frames), other = across(k, ch, axis);
        if (within.good != other.good) return within.good ? within : other;
        return other.quality > within.quality ? other : within;
    }
    std::map<std::uint32_t, Candidate> candidates_;
    std::uint64_t calls_ = 0;
};

}  // namespace fw
