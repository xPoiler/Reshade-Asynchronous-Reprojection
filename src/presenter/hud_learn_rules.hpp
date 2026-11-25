// HUD learning, the presenter's side: which learned point to pick, and when to drop or set aside the one in use. Pure CPU
// logic fed one record per probed game frame (Renderer::score_hud_probe's sums, read back), so that every rule can be
// replayed offline from a run's logs\probe.csv (tests/hud_learn_tests.cpp) instead of in game.
//
// The cases these rules have to survive, from Baldur's Gate 3, Earth Defense Force 6 and Metro 2033 Redux (2026-11-24):
// - a title screen where the point gives next to nothing (its text and the cursor, ~0.1% of the screen: 30-50 tiles of 16
//   px at 4K) or a menu drawn with the HUD's shaders over most of it: no verdict either way;
// - a loading screen between them (BG3: 35-42% of the screen, a few frames scored at "stays on the screen" 0.43): a few
//   frames, weighed as one frame each - not as the thousands of tiles they hold;
// - a camera cut (the motion between the two frames is no motion): no verdict;
// - a flip: the point's content replaced whole from one frame to the next (BG3's title screen: two pictures by turns,
//   nothing of either at the same spot or where the scenery came from) - nothing told either way: no verdict;
// - the HUD hidden (cutscene, photo mode): no verdict;
// - a still camera: nothing tells the HUD from the scenery (no tile moved far enough - the probe leaves those out);
// - a repeated picture (the detection's guard: the camera said it moved, the picture did not): no verdict;
// - a panel opened over the HUD (inventory): the share of the screen grows, the usual HUD stays - not a sign of a wrong
//   point;
// - a point named by a pixel shader the game uses at several places (BG3's e7948cbd: its whole HUD, part of its hotbar,
//   the whole picture by spells of seconds): set aside for the session;
// - a point holding only part of the HUD (BG3's e504b18c, c7c69144): not picked, where the detection gives a reference;
// - the reference: only the map the detection learned itself, on frames it learned from - the combined detection's, or in a
//   game without the upscaler's output the plain learned detection's (EDF6: without it the first point scored 30 times,
//   holding 7% of the HUD, was picked). Never a HUD map taken from elsewhere: the upscaler output's unconfirmed guess (40%
//   of BG3's screen at times) or the point's own layer while one is in use;
// - a game whose detection never learns: no reference, the pick as before 2026-11-24;
// - a HUD of a crosshair alone: scored at any size, and picked without judging it by a reference too small to tell
//   anything (small_hud) - never right after the detection's map started over;
// - the panel's Relearn: everything here starts over (Renderer::hud_learn_forget);
// - a probe and its game frame's detection verdict read back in either order (Pairing).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace fw::hudlearn {

constexpr double kPickArea = 0.6;  // (a candidate that changes this share of the screen or more is a menu, not a HUD)
// The point in use is judged only on frames where what moved far enough holds this share of the screen's tiles: a HUD's
// worth (BG3 ~10%, EDF6 ~20%, Metro a few %), not a title screen's text and cursor (~0.1%). A HUD smaller than this is never
// judged - kept, as nothing could tell it from the scenery anyway.
constexpr double kVerdictShare = 0.003;

// One probed game frame: the candidate's copy against the final picture, in 16x16-px tiles (see cs_probe_score).
struct ProbeFrame {
    std::uint32_t cand = 0;
    bool locked = false;  // the point in use (its layer is the HUD now)
    double area = 0;      // share of the screen it differs on
    // Where the camera moved a tile by 1.5 tiles or more: overlap with the previous frame's tiles at the same spot
    // (stays on the screen) and where the scenery came from (moves with it), as sums of min and max (in tiles).
    double same_min = 0, same_max = 0, world_min = 0, world_max = 0;
    bool ref = false;         // the detection learned from this frame and its confirmed map was compared
    double hud_tiles = 0;     // tiles holding confirmed HUD (ref only)
    double covered = 0;       // of them, those the candidate differs on
    double beyond = 0;        // tiles the candidate differs on that hold no confirmed HUD
    double kept = -1;         // locked only: share of the point's usual tiles there this frame (-1: not measured)
    bool repeated = false;    // the detection's guard held this frame back (a repeated picture)
    bool cut = false;         // a camera cut between the two frames
    bool ref_possible = false;  // the detection's confirmed map exists in this game (its combined detection runs)
    bool detect_telling = false;  // the detection ran on this frame and had enough telling pixels to learn from
    double tiles = 0;         // the screen's tiles (for the share a frame must show to weigh in - kVerdictShare)
};

struct Score {
    std::uint32_t frames = 0, scored = 0, ref_frames = 0;
    double area = 0, same_min = 0, same_max = 0, world_min = 0, world_max = 0;
    double hud_tiles = 0, covered = 0, beyond = 0;
};

class Rules {
public:
    // What the point in use did, for the report.
    struct InUse {
        double same = 1, world = 0;  // the running verdict (decayed sums)
        std::uint32_t n = 0;         // scored frames since it came into use (or its last verdict)
    };

    void add(const ProbeFrame& f) {
        ++frames_total_;
        if (f.ref_possible) ref_possible_seen_ = true;
        if (f.ref) ++ref_frames_total_;
        if (f.detect_telling) ++detect_telling_total_;
        if (f.ref) {  // (the size of the confirmed HUD itself, whatever the candidate - see small_hud)
            const double a = 1.0 / std::min(60.0, double(++ref_since_reset_));
            ref_now_ += a * (f.hud_tiles - ref_now_);
            if (ref_since_reset_ >= 30) ref_peak_ = std::max(ref_peak_, ref_now_);
        }
        Score& sc = scores_[f.cand];
        ++sc.frames;
        sc.area += f.area;
        // (the overlap sums on frames that can tell a HUD from the scenery: not a menu over most of the screen, a repeated
        // picture, a camera cut, or a flip. For the pick at any size - its sums weigh each frame by what it holds, so a title
        // screen's text and cursor count for little, and a HUD of a crosshair alone is still picked (until 2026-11-25 it
        // never was: the size floor below applied here too). The point in use is judged only on frames showing a HUD's worth
        // (kVerdictShare), as its verdict weighs each frame as one)
        const bool tells = f.area < kPickArea && !f.repeated && !f.cut && f.same_max > 0 && f.world_max > 0 &&
                           (f.same_min > 0 || f.world_min > 0);
        const bool weighs = tells && (f.tiles <= 0 || f.same_max >= kVerdictShare * f.tiles);
        if (tells) {
            ++sc.scored;
            sc.same_min += f.same_min; sc.same_max += f.same_max;
            sc.world_min += f.world_min; sc.world_max += f.world_max;
        }
        if (f.ref) {
            ++sc.ref_frames;
            sc.hud_tiles += f.hud_tiles; sc.covered += f.covered; sc.beyond += f.beyond;
        }
        if (!f.locked) {
            if (fail_ && f.cand != fail_) fail_ = 0;  // (learning again)
            return;
        }
        if (f.cand != locked_) { locked_ = f.cand; reset_in_use(); }
        check_in_use(f, weighs);
        check_stable(f, weighs);
    }

    // The pick: of the candidates scored on 30 frames or more that change under kPickArea of the screen, stay on the
    // screen (0.9 of their overlap) and clearly more than they move with the scenery (0.2), and hold the HUD (see
    // holds_hud) - the tightest around the confirmed HUD (least of it beyond), or without a reference the largest.
    std::uint32_t pick() const {
        std::uint32_t best = 0;
        bool best_ref = false;
        double best_key = 0;
        for (const auto& [cand, sc] : scores_) {
            if (sc.scored < 30 || !sc.frames || sc.same_max <= 0 || sc.world_max <= 0 || unstable_.count(cand)) continue;
            const double area = sc.area / sc.frames, same = sc.same_min / sc.same_max, world = sc.world_min / sc.world_max;
            if (area >= kPickArea || same < 0.9 || same - world < 0.2) continue;
            bool by_ref = false;
            if (!holds_hud(sc, by_ref)) continue;
            const double key = by_ref ? -tightness(sc) : area;  // (larger is better)
            if (!best || (by_ref && !best_ref) || (by_ref == best_ref && key > best_key)) { best = cand; best_ref = by_ref; best_key = key; }
        }
        return best;
    }

    // A candidate holds the HUD: where the detection has confirmed HUD on 30 of its frames (8 tiles a frame on average,
    // ~2000 px), it differs on 60% of it. Without that evidence it waits - unless the game has no reference at all (no
    // combined detection: the pick as before), or the detection learned on under 300 of a minute of frames it could have
    // learned from (enough telling pixels - a still camera is no such frame: BG3 standing still learned on ~60 frames in
    // 50 s, and the pick must still wait there).
    // A small HUD (a crosshair alone): the detection has learned on 300 frames since its map last started over, and the HUD
    // it confirms never came to 8 tiles a frame (running average over 60 of them) - nothing to judge a candidate by: the
    // pick as without a reference. Never right after the map started over (a Relearn, the upscaler's output coming or going):
    // the peak is that map's, and 300 learned frames must pass first.
    bool holds_hud(const Score& sc, bool& by_ref) const {
        by_ref = false;
        if (!ref_possible_seen_ || small_hud()) return true;
        by_ref = sc.ref_frames >= 30 && sc.hud_tiles >= 8.0 * sc.ref_frames;
        if (by_ref) return sc.covered >= 0.6 * sc.hud_tiles;
        return ref_frames_total_ < 300 && detect_telling_total_ >= 3600;
    }
    bool small_hud() const { return ref_since_reset_ >= 300 && ref_peak_ < 8.0; }
    // The detection's map started over: its size is measured afresh.
    void reference_reset() { ref_since_reset_ = 0; ref_now_ = ref_peak_ = 0; }
    static double tightness(const Score& sc) { return sc.covered + sc.beyond > 0 ? sc.beyond / (sc.covered + sc.beyond) : 1.0; }

    std::uint32_t fail() const { return fail_; }
    std::uint32_t locked() const { return locked_; }
    const InUse& in_use() const { return in_use_; }
    const std::map<std::uint32_t, Score>& scores() const { return scores_; }
    const std::set<std::uint32_t>& unstable() const { return unstable_; }

private:
    void reset_in_use() {
        in_use_ = {};
        bins_ = {}; bins_kept_ = {}; bins_n_ = 0;
    }

    // The point in use, checked: it stays on the screen (0.8 of its overlap) and clearly more than it moves with the
    // scenery (0.1), as running averages over its last ~60 frames that show a HUD's worth (kVerdictShare) - each such frame
    // weighs one, however much it holds, so a few frames of a loading screen do not outweigh the HUD. Not on menu frames
    // (kPickArea of the screen or more), repeated pictures or camera cuts.
    void check_in_use(const ProbeFrame& f, bool weighs) {
        if (!weighs) return;
        const double a = 1.0 / std::min(60.0, double(++in_use_.n));
        in_use_.same += a * (f.same_min / f.same_max - in_use_.same);
        in_use_.world += a * (f.world_min / f.world_max - in_use_.world);
        if (in_use_.n >= 60 && (in_use_.same < 0.8 || in_use_.same - in_use_.world < 0.1)) {
            fail_ = locked_;
            scores_.erase(locked_);
            in_use_ = {};
        }
    }

    // ...and it gives much the same share of the screen from frame to frame: over its frames in use (halved every 300),
    // at least 80% within a factor of 2 either way of the share it gives most (bins of a factor of 2) - on the frames that
    // weigh in (as check_in_use: in play, the camera moving - not a title or loading screen, which can last longer than
    // the play around it). A frame off that range counts against it only where the point's usual tiles are gone too
    // (kept under half): a panel opened over the HUD keeps them. Set aside for the session; the detection gives the HUD
    // meanwhile.
    void check_stable(const ProbeFrame& f, bool weighs) {
        if (!weighs || f.area <= 1.0 / 1024.0) return;
        const std::size_t b = std::size_t(std::clamp(int(std::floor(std::log2(f.area * 1024.0))), 0, int(bins_.size()) - 1));
        ++bins_[b];
        if (f.kept >= 0.5) ++bins_kept_[b];
        if (++bins_n_ < 300) return;
        std::size_t most = 0;
        for (std::size_t k = 1; k < bins_.size(); ++k) if (bins_[k] > bins_[most]) most = k;
        std::uint32_t off_lost = 0;
        for (std::size_t k = 0; k < bins_.size(); ++k)
            if (k + 1 < most || k > most + 1) off_lost += bins_[k] - bins_kept_[k];
        if (off_lost > 0.2 * bins_n_) {
            unstable_.insert(locked_);
            fail_ = locked_;
            scores_.erase(locked_);
            in_use_ = {};
        }
        bins_n_ = 0;
        for (std::size_t k = 0; k < bins_.size(); ++k) { bins_[k] /= 2; bins_kept_[k] /= 2; bins_n_ += bins_[k]; }
    }

    std::map<std::uint32_t, Score> scores_;
    std::set<std::uint32_t> unstable_;
    std::uint32_t locked_ = 0, fail_ = 0;
    InUse in_use_;
    std::array<std::uint32_t, 10> bins_{}, bins_kept_{};
    std::uint32_t bins_n_ = 0;
    std::uint64_t frames_total_ = 0, ref_frames_total_ = 0, detect_telling_total_ = 0;
    bool ref_possible_seen_ = false;
    std::uint64_t ref_since_reset_ = 0;
    double ref_now_ = 0, ref_peak_ = 0;  // (the confirmed HUD's tiles: running average, its peak since the map started over)
};

// A probed frame and the HUD detection's verdict on the same game frame, paired by the game frame's number - read back
// from slots of their own, in either order (split queues with the intake in pieces record the two in different command
// lists: paired by slot, a probe was read with the next frame's verdict). A probe waits for its verdict and goes without one
// (nothing learned, nothing repeated, not telling) once a later frame's verdict is in, or 8 game frames have passed - the
// detection did not run on its frame.
class Pairing {
public:
    struct Wait {
        std::uint64_t seq = 0;
        ProbeFrame f;                    // (hud_tiles, covered, beyond: as measured - kept only if the detection learned)
        std::uint32_t raw[10] = {};      // (the probe's sums, for the diagnostics)
        std::int64_t qpc = 0;
    };
    void probe(const Wait& w) { waiting_.push_back(w); }
    void verdict(std::uint64_t seq, bool learned, bool telling, bool repeated) {
        verdicts_[seq % verdicts_.size()] = {seq, learned, telling, repeated};
        latest_ = std::max(latest_, seq);
    }
    // Each probe whose verdict is in or can no longer come, in order, completed (ref, repeated, detect_telling) to `out`.
    template <class Out> void drain(std::uint64_t intake_seq, Out&& out) {
        std::size_t done = 0;
        for (; done < waiting_.size(); ++done) {
            Wait& w = waiting_[done];
            const Verdict& d = verdicts_[w.seq % verdicts_.size()];
            const bool have = w.seq != 0 && d.seq == w.seq;
            if (!have && latest_ < w.seq && intake_seq < w.seq + 8) break;  // (its verdict may still come)
            ProbeFrame& f = w.f;
            f.repeated = have && d.repeated;
            f.detect_telling = have && d.telling;
            f.ref = f.ref_possible && have && d.learned;
            if (!f.ref) f.hud_tiles = f.covered = f.beyond = 0;
            out(w);
        }
        waiting_.erase(waiting_.begin(), waiting_.begin() + static_cast<std::ptrdiff_t>(done));
    }
    void clear() { waiting_.clear(); }
    std::size_t waiting() const { return waiting_.size(); }

private:
    struct Verdict { std::uint64_t seq = 0; bool learned = false, telling = false, repeated = false; };
    std::array<Verdict, 16> verdicts_{};
    std::uint64_t latest_ = 0;
    std::vector<Wait> waiting_;
};

}  // namespace fw::hudlearn
