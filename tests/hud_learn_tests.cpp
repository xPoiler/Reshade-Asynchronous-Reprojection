// HUD learning's rules (src/presenter/hud_learn_rules.hpp), on the cases they must survive - and, given a run's
// logs\probe.csv, a replay of that run through them: hud_learn_tests <probe.csv> prints each pick, drop and set-aside.
#include "presenter/hud_learn_rules.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using fw::hudlearn::ProbeFrame;
using fw::hudlearn::Rules;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// A frame of a point: share of the screen, tiles that moved far enough (n), the share of them that stays on the screen
// and that moves with the scenery.
ProbeFrame frame(std::uint32_t cand, bool locked, double area, double n, double same, double world) {
    ProbeFrame f;
    f.cand = cand; f.locked = locked; f.area = area;
    f.tiles = 240.0 * 135.0;  // (4K in tiles of 16 px)
    f.same_max = n; f.same_min = n * same; f.world_max = n; f.world_min = n * world;
    return f;
}
ProbeFrame with_ref(ProbeFrame f, double hud, double covered, double beyond) {
    f.ref_possible = f.ref = true; f.hud_tiles = hud; f.covered = covered; f.beyond = beyond;
    return f;
}

constexpr std::uint32_t kHud = 0x4b4b6f00, kOther = 0xe7948cbd, kPart = 0xe504b18c;

void in_use_cases() {
    {   // the title screen: the point gives the cursor alone, which moves - no verdict
        Rules r;
        for (int i = 0; i < 2000; ++i) r.add(frame(kHud, true, 0.001, 1, 0.1, 0.05));
        check(r.fail() == 0, "title screen with a cursor alone: the point kept");
    }
    for (const double n : {16.0, 32.0, 48.0}) {  // the title screen as BG3 logged it: ~0.1% of the screen, 16-48 tiles
        for (const double same : {0.03, 0.1}) {
            Rules r;
            for (int i = 0; i < 2000; ++i) r.add(frame(kHud, true, 0.001, n, same, same * 0.5));
            for (int i = 0; i < 2000; ++i) { auto f = frame(kHud, true, 0.10, 400, 0.97, 0.5); f.kept = 0.97; r.add(f); }
            char what[128];
            std::snprintf(what, sizeof(what), "title screen at %.0f tiles, stays %.2f, then play: the point kept", n, same);
            check(r.fail() == 0 && r.unstable().empty(), what);
        }
    }
    {   // a launch as BG3's: title, a long loading screen with a few frames scored at 35-42% and "stays" 0.43, then play
        Rules r;
        for (int i = 0; i < 1800; ++i) r.add(frame(kHud, true, 0.001, 32, 0.05, 0.02));
        for (int i = 0; i < 1800; ++i) {
            if (i % 300 == 0) r.add(frame(kHud, true, 0.38, 11000, 0.43, 0.16));
            else r.add(frame(kHud, true, 0.38, 0, 0, 0));
        }
        for (int i = 0; i < 3000; ++i) { auto f = frame(kHud, true, 0.10, 400, 0.97, 0.5); f.kept = 0.97; r.add(f); }
        check(r.fail() == 0 && r.unstable().empty(), "title, loading screen, play: the point kept");
    }
    {   // BG3's title screen (probe.csv 2026-11-24, rows 276-351): two pictures by turns, 2.5% and 0.1% of the screen,
        // nothing of either at the same spot or where the scenery came from - then play
        Rules r;
        for (int i = 0; i < 600; ++i) r.add(frame(kHud, true, i % 2 ? 0.001 : 0.025, 200 + i % 700, 0.0, 0.0));
        for (int i = 0; i < 2000; ++i) { auto f = frame(kHud, true, 0.10, 400, 0.97, 0.5); f.kept = 0.97; r.add(f); }
        check(r.fail() == 0 && r.unstable().empty(), "a title screen flipping between two pictures: the point kept");
    }
    {   // a camera cut every few seconds (the motion between the frames is none): no verdict on those
        Rules r;
        for (int i = 0; i < 3000; ++i) {
            auto f = i % 100 < 5 ? frame(kHud, true, 0.10, 3000, 0.1, 0.1) : frame(kHud, true, 0.10, 400, 0.97, 0.5);
            f.cut = i % 100 < 5;
            r.add(f);
        }
        check(r.fail() == 0, "camera cuts: the point kept");
    }
    {   // the HUD in game: stays on the screen, partly moves with the scenery where the hotbar runs along the pan
        Rules r;
        for (int i = 0; i < 3000; ++i) r.add(frame(kHud, true, 0.12, 400, 0.97, 0.55));
        check(r.fail() == 0 && r.unstable().empty(), "the HUD in game: kept");
    }
    {   // a wrong point in use: it moves with the scenery
        Rules r;
        for (int i = 0; i < 200; ++i) r.add(frame(kOther, true, 0.3, 2000, 0.5, 0.5));
        check(r.fail() == kOther, "a point that moves with the scenery: dropped");
    }
    {   // repeated pictures: the camera said it moved, nothing did
        Rules r;
        for (int i = 0; i < 1000; ++i) { auto f = frame(kHud, true, 0.12, 400, 0.98, 0.97); f.repeated = true; r.add(f); }
        check(r.fail() == 0, "repeated pictures: the point kept");
    }
    {   // the HUD hidden (a cutscene): the point gives nothing
        Rules r;
        for (int i = 0; i < 1000; ++i) r.add(frame(kHud, true, 0.0, 0, 0, 0));
        for (int i = 0; i < 1000; ++i) r.add(frame(kHud, true, 0.12, 400, 0.97, 0.5));
        check(r.fail() == 0 && r.unstable().empty(), "the HUD hidden for a while: the point kept");
    }
    {   // a menu drawn with the HUD's shaders over most of the screen
        Rules r;
        for (int i = 0; i < 1000; ++i) r.add(frame(kHud, true, 0.86, 3000, 0.63, 0.6));
        check(r.fail() == 0, "a menu over 60% of the screen: no verdict");
    }
    {   // a still camera: no tile moved far enough, no overlap sums
        Rules r;
        for (int i = 0; i < 1000; ++i) r.add(frame(kHud, true, 0.12, 0, 0, 0));
        check(r.fail() == 0 && r.unstable().empty(), "a still camera: the point kept");
    }
    {   // an inventory opened over the HUD for seconds at a time: the share grows 5x, the usual tiles stay
        Rules r;
        for (int k = 0; k < 6; ++k) {
            for (int i = 0; i < 300; ++i) { auto f = frame(kHud, true, 0.08, 300, 0.97, 0.5); f.kept = 0.97; r.add(f); }
            for (int i = 0; i < 200; ++i) { auto f = frame(kHud, true, 0.42, 1500, 0.98, 0.4); f.kept = 0.95; r.add(f); }
        }
        check(r.fail() == 0 && r.unstable().empty(), "an inventory opened over the HUD again and again: the point kept");
    }
    {   // a point that lands on another place by spells: its whole HUD, then part of the hotbar
        Rules r;
        for (int k = 0; k < 6; ++k) {
            for (int i = 0; i < 300; ++i) { auto f = frame(kOther, true, 0.15, 500, 0.97, 0.5); f.kept = 0.97; r.add(f); }
            for (int i = 0; i < 200; ++i) { auto f = frame(kOther, true, 0.03, 100, 0.97, 0.5); f.kept = 0.2; r.add(f); }
        }
        check(r.unstable().count(kOther) == 1 && r.fail() == kOther, "a point that lands elsewhere by spells: set aside");
    }
    {   // ...and the same without the usual-tiles measure (kept unknown): set aside as before
        Rules r;
        for (int k = 0; k < 6; ++k) {
            for (int i = 0; i < 300; ++i) r.add(frame(kOther, true, 0.15, 500, 0.97, 0.5));
            for (int i = 0; i < 200; ++i) r.add(frame(kOther, true, 0.03, 100, 0.97, 0.5));
        }
        check(r.unstable().count(kOther) == 1, "spells without the usual-tiles measure: set aside");
    }
}

void pick_cases() {
    {   // with the detection's confirmed HUD: the tightest that holds it, never the one holding a part
        Rules r;
        for (int i = 0; i < 100; ++i) {
            r.add(with_ref(frame(kHud, false, 0.10, 400, 0.97, 0.5), 300, 280, 30));     // holds 93%, 10% beyond
            r.add(with_ref(frame(kOther, false, 0.30, 900, 0.97, 0.5), 300, 290, 400));  // holds 97%, 58% beyond
            r.add(with_ref(frame(kPart, false, 0.02, 80, 0.97, 0.5), 300, 30, 2));       // holds 10%
        }
        check(r.pick() == kHud, "with a reference: the tightest point that holds the HUD");
    }
    {   // EDF6 (2026-11-24): a point holding 7% of the HUD is scored 30 times first, the whole HUD's later - never the part
        Rules r;
        for (int i = 0; i < 40; ++i) r.add(with_ref(frame(kPart, false, 0.04, 150, 0.98, 0.6), 300, 21, 60));
        check(r.pick() == 0, "EDF6: the part scored first: not picked");
        for (int i = 0; i < 40; ++i) r.add(with_ref(frame(kHud, false, 0.13, 400, 0.99, 0.6), 300, 297, 120));
        check(r.pick() == kHud, "EDF6: ...the point holding the whole HUD picked once scored");
    }
    {   // a point holding a part, alone: not picked
        Rules r;
        for (int i = 0; i < 100; ++i) r.add(with_ref(frame(kPart, false, 0.02, 80, 0.97, 0.5), 300, 30, 2));
        check(r.pick() == 0, "a point holding a part of the HUD: not picked");
    }
    {   // the detection runs but has confirmed nothing yet: wait
        Rules r;
        for (int i = 0; i < 1000; ++i) { auto f = frame(kHud, false, 0.10, 400, 0.97, 0.5); f.ref_possible = true; f.detect_telling = true; r.add(f); }
        check(r.pick() == 0, "the detection runs, nothing confirmed yet: no pick");
        for (int i = 0; i < 3000; ++i) { auto f = frame(kHud, false, 0.10, 400, 0.97, 0.5); f.ref_possible = true; f.detect_telling = true; r.add(f); }
        check(r.pick() == kHud, "...a minute of telling frames later it still has not learned: the pick as without a reference");
    }
    {   // a still camera for minutes (BG3: ~60 frames learned from in 50 s): the detection never had its chance - wait
        Rules r;
        for (int i = 0; i < 6000; ++i) {
            auto f = frame(kHud, false, 0.10, 400, 0.97, 0.5);
            f.ref_possible = true; f.detect_telling = i % 50 == 0;
            r.add(f);
        }
        check(r.pick() == 0, "minutes of a mostly still camera: no pick yet");
    }
    {   // no detection reference in this game: the pick as before (the largest)
        Rules r;
        for (int i = 0; i < 100; ++i) {
            r.add(frame(kHud, false, 0.10, 400, 0.97, 0.5));
            r.add(frame(kPart, false, 0.02, 80, 0.97, 0.5));
        }
        check(r.pick() == kHud, "no reference: the largest point that stays on the screen");
    }
    {   // a menu-only shader and one that moves with the scenery: never
        Rules r;
        for (int i = 0; i < 100; ++i) {
            r.add(frame(kOther, false, 0.8, 3000, 0.97, 0.5));
            r.add(frame(kPart, false, 0.1, 400, 0.6, 0.55));
        }
        check(r.pick() == 0, "a menu-sized candidate and one moving with the scenery: no pick");
    }
    {   // a HUD of a crosshair alone (a few tiles), no detection: picked (until 2026-11-25 the size floor kept it from ever
        // being scored)
        Rules r;
        for (int i = 0; i < 200; ++i) r.add(frame(kHud, false, 0.0005, 4, 0.98, 0.2));
        check(r.pick() == kHud, "a crosshair-only HUD, no detection: picked");
    }
    {   // ...with the detection learning, its confirmed HUD a crosshair's (4 tiles): picked once it has learned on 300 frames
        Rules r;
        for (int i = 0; i < 250; ++i) r.add(with_ref(frame(kHud, false, 0.0005, 4, 0.98, 0.2), 4, 4, 2));
        check(r.pick() == 0, "a crosshair-only HUD with the detection: waits while it learns");
        for (int i = 0; i < 200; ++i) r.add(with_ref(frame(kHud, false, 0.0005, 4, 0.98, 0.2), 4, 4, 2));
        check(r.pick() == kHud, "...then picked: the confirmed HUD too small to judge by");
    }
    {   // a title screen's cursor alone, moving: never picked, at any size
        Rules r;
        for (int i = 0; i < 500; ++i) r.add(frame(kOther, false, 0.001, 30, 0.1, 0.05));
        check(r.pick() == 0, "a title screen's moving cursor: not picked");
    }
    {   // the map started over (Relearn, the upscaler's output coming or going) in a game with a large HUD: while it learns
        // again its confirmed HUD is small - a point holding part of the HUD is not picked meanwhile, nor after
        Rules r;
        for (int i = 0; i < 400; ++i) r.add(with_ref(frame(kPart, false, 0.02, 80, 0.97, 0.5), 300, 30, 2));
        r.reference_reset();
        for (int i = 0; i < 280; ++i) r.add(with_ref(frame(kPart, false, 0.02, 80, 0.97, 0.5), 2, 2, 60));
        check(r.pick() == 0, "the map started over, still small: a point holding a part not picked");
        for (int i = 0; i < 400; ++i) r.add(with_ref(frame(kPart, false, 0.02, 80, 0.97, 0.5), 300, 30, 2));
        check(r.pick() == 0, "...nor once the map holds the HUD again");
    }
    {   // a large HUD hidden for a long cutscene while the detection learns (its confirmed HUD fades): no small-HUD pick
        Rules r;
        for (int i = 0; i < 300; ++i) r.add(with_ref(frame(kPart, false, 0.02, 80, 0.97, 0.5), 300, 30, 2));
        for (int i = 0; i < 2000; ++i) r.add(with_ref(frame(kPart, false, 0.02, 80, 0.97, 0.5), 1, 1, 60));
        check(r.pick() == 0, "a large HUD hidden for long: a point holding a part still not picked");
    }
    {   // a point set aside is not picked again this session
        Rules r;
        for (int k = 0; k < 6; ++k) {
            for (int i = 0; i < 300; ++i) r.add(frame(kOther, true, 0.15, 500, 0.97, 0.5));
            for (int i = 0; i < 200; ++i) r.add(frame(kOther, true, 0.03, 100, 0.97, 0.5));
        }
        for (int i = 0; i < 100; ++i) r.add(frame(kOther, false, 0.15, 500, 0.97, 0.5));
        check(r.pick() == 0, "a point set aside: not picked again");
    }
}

// A probe and its game frame's detection verdict, read back in either order (split queues with the intake in pieces).
void pairing_cases() {
    using fw::hudlearn::Pairing;
    auto probe = [](std::uint64_t seq) {
        Pairing::Wait w;
        w.seq = seq; w.f.cand = kHud; w.f.ref_possible = true; w.f.hud_tiles = 100; w.f.covered = 90;
        return w;
    };
    {   // the verdict read first, then the probe; and the probe first, then the verdict
        Pairing p;
        std::vector<Pairing::Wait> out;
        auto take = [&](const Pairing::Wait& w) { out.push_back(w); };
        p.verdict(1, true, true, false);
        p.probe(probe(1));
        p.drain(1, take);
        p.probe(probe(2));
        p.drain(2, take);
        const bool waited = out.size() == 1;
        p.verdict(2, false, true, true);
        p.drain(2, take);
        check(waited && out.size() == 2, "pairing: a probe waits for its own frame's verdict");
        check(out.size() == 2 && out[0].f.ref && !out[0].f.repeated && out[0].f.hud_tiles == 100,
              "pairing: frame 1 learned from - compared with the reference");
        check(out.size() == 2 && !out[1].f.ref && out[1].f.repeated && out[1].f.hud_tiles == 0,
              "pairing: frame 2 a repeated picture - its own verdict, not frame 1's");
    }
    {   // the next frame's verdict comes before the probe's own (one slot behind): never taken for it
        Pairing p;
        std::vector<Pairing::Wait> out;
        p.probe(probe(5));
        p.verdict(6, true, true, false);
        p.drain(6, [&](const Pairing::Wait& w) { out.push_back(w); });
        check(out.size() == 1 && !out[0].f.ref && !out[0].f.detect_telling, "pairing: a later frame's verdict is not the probe's");
    }
    {   // no detection on its frame at all: goes without a verdict after 8 frames
        Pairing p;
        std::vector<Pairing::Wait> out;
        p.probe(probe(10));
        p.drain(15, [&](const Pairing::Wait& w) { out.push_back(w); });
        const bool held = out.empty();
        p.drain(18, [&](const Pairing::Wait& w) { out.push_back(w); });
        check(held && out.size() == 1 && !out[0].f.ref, "pairing: without a verdict it waits, then goes after 8 frames");
    }
}

// A run's probe.csv through the rules: each change of pick, drop and set-aside, with its row.
int replay(const char* path) {
    std::ifstream in(path);
    if (!in) { std::printf("cannot open %s\n", path); return 2; }
    std::string line;
    std::getline(in, line);
    Rules r;
    std::uint32_t last_pick = 0, last_fail = 0;
    std::size_t last_unstable = 0, rows = 0;
    while (std::getline(in, line)) {
        std::vector<std::string> c;
        std::stringstream ss(line);
        for (std::string x; std::getline(ss, x, ',');) c.push_back(x);
        if (c.size() < 15) continue;
        ProbeFrame f;
        f.cand = std::uint32_t(std::strtoul(c[1].c_str(), nullptr, 16));
        f.locked = c[2] == "1";
        f.area = std::atof(c[3].c_str());
        f.same_min = std::atof(c[4].c_str()); f.same_max = std::atof(c[5].c_str());
        f.world_min = std::atof(c[6].c_str()); f.world_max = std::atof(c[7].c_str());
        f.ref_possible = c[8] == "1"; f.ref = c[9] == "1";
        if (f.ref) { f.hud_tiles = std::atof(c[10].c_str()); f.covered = std::atof(c[11].c_str()); f.beyond = std::atof(c[12].c_str()); }
        f.kept = std::atof(c[13].c_str());
        f.repeated = c[14] == "1";
        if (c.size() >= 23) { f.tiles = std::atof(c[20].c_str()); f.cut = c[21] == "1"; f.detect_telling = c[22] == "1"; }
        r.add(f);
        ++rows;
        const std::uint32_t p = r.pick();
        if (p != last_pick) { std::printf("row %s: pick %08x\n", c[0].c_str(), p); last_pick = p; }
        if (r.fail() != last_fail) { if (r.fail()) std::printf("row %s: %08x dropped\n", c[0].c_str(), r.fail()); last_fail = r.fail(); }
        if (r.unstable().size() != last_unstable) { std::printf("row %s: %08x set aside\n", c[0].c_str(), f.cand); last_unstable = r.unstable().size(); }
    }
    std::printf("%zu rows\n", rows);
    for (const auto& [cand, sc] : r.scores()) {
        bool by_ref = false;
        const bool holds = r.holds_hud(sc, by_ref);
        std::printf("  %08x: %u frames, area %.3f, same %.2f, world %.2f (%u scored), holds %.0f%% (%u compared, %.0f%% beyond)%s\n", cand,
                    sc.frames, sc.frames ? sc.area / sc.frames : 0.0, sc.same_max > 0 ? sc.same_min / sc.same_max : 0.0,
                    sc.world_max > 0 ? sc.world_min / sc.world_max : 0.0, sc.scored, sc.hud_tiles > 0 ? 100.0 * sc.covered / sc.hud_tiles : 0.0,
                    sc.ref_frames, 100.0 * Rules::tightness(sc), holds ? "" : by_ref ? " - too little" : " - waiting");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) return replay(argv[1]);
    in_use_cases();
    pick_cases();
    pairing_cases();
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
