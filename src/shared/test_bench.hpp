#pragma once
// Development (dev key TestBench=1 -> FW_TESTBENCH): the presenter hands every refresh it draws to the XPAR test game's scorer
// (dev/testgame) - the picture it presented (a copy of its back buffer, in a ring of shared textures) and what it drew it from (the
// game frame, its camera and the camera the warp aimed at, the moments). The scorer renders the real picture for the same moment
// and scores the warp. Nothing of this runs without the key.
//
// Names (the game's process id, as the presenter's --pid): Local\XPAR_Bench_<pid> (this header, a file mapping),
// Local\XPAR_Bench_<pid>_tex<i> (the textures, D3D12 shared NT handles), Local\XPAR_Bench_<pid>_fence (a shared fence: a refresh's
// copy is done once it reaches the refresh's fence_value), Local\XPAR_Bench_<pid>_input (an event: scripted mouse moves written).
// The presenter makes them all; the scorer and the test game open them.
//
// Scripted input: in its scripted runs the test game writes its mouse moves into `mouse` and sets input_owner. The presenter then
// takes its mouse from there only (not from Windows: the cursor never moves, the real mouse is left alone) and keeps showing
// while the game is not in front, so a bench runs in the background.
#include <cstdint>

namespace fw::bench {

constexpr std::uint32_t kMagic = 0x48434E42;  // 'BNCH'
constexpr std::uint32_t kVersion = 4;
constexpr int kMouseRing = 4096;
constexpr int kSlots = 6;
constexpr int kRing = 512;

enum Flags : std::uint32_t {
    kWarped = 1,           // the warp drew this refresh (else the game frame as it is)
    kRepeat = 2,           // nothing new presented: the screen keeps the previous picture (no copy)
    kNotShown = 4,         // the presenter is not showing (the game's own picture is on the screen)
    kEstimated = 8,        // the frame's camera is XPAR's estimate (cameras in the estimate's own space)
    kObjectsBetween = 16,  // what moves on its own drawn at the held-back moment (Renderer::set_objects_between)
    kHeldBetween = 32,     // the held weapon at the held-back moment
    kNoCopy = 64,          // presented, but no slot was free: no picture for this refresh
};

struct Refresh {
    std::uint64_t serial;            // 1, 2, ... (written last)
    std::uint64_t fence_value;       // the copy's fence value (0: no picture)
    std::int32_t slot;               // its texture (-1: none)
    std::uint32_t flags;
    std::int64_t qpc_present_call;   // when the presenter called Present
    std::int64_t qpc_target_vblank;  // the refresh it was drawn for (0: not known)
    std::int64_t qpc_camera;         // the moment the camera was shown for (the camera model's time)
    std::uint64_t source_frame;      // the game frame shown (the add-on's frame id)
    std::int64_t source_qpc_present; // ...when the game presented it (the add-on's time: matches the game's own Present)
    float source_pos[3], source_right[3], source_up[3], source_fwd[3];  // its camera
    float target_pos[3], target_right[3], target_up[3], target_fwd[3];  // the camera the warp aimed at
    float held_back;                 // game frames the movement is held back (PoseModel::position_frames_back)
    float frame_interval;            // the game's frame interval, seconds
    std::uint32_t present_count, present_refresh_count, sync_refresh_count;  // DXGI frame statistics after the Present
    std::uint32_t hud_source;        // where this refresh's HUD came from (PresenterStatus::hud_source: 0 none / not known, 1-7)
    std::int64_t sync_qpc;
    // (version 4) the camera model's view of the shown frame and this refresh - how XPAR times its camera (to-consolidate #13):
    // its time for the frame's camera (PoseModel source time, QPC), its frame-to-intake latency (PoseModel::latency, the time
    // it shows Auto's camera behind), how far ahead of that it showed (now_ahead), the game latency mode 7 aims at, the fit's
    // delay and smoothing per axis (yaw, pitch), and the Auto latency mode
    std::int64_t source_qpc_model;
    float model_latency_ms, display_ahead_ms, game_latency_ms;
    float model_delay_ms[2], model_tau_ms[2];
    std::uint32_t auto_mode;
};

struct Header {
    std::uint32_t magic, version;
    volatile std::int32_t generation;     // textures made again (a new size): the scorer opens them again
    std::uint32_t width, height, format;  // the textures (the presenter's back buffer: DXGI_FORMAT)
    std::uint32_t refresh_hz_x1000;
    volatile std::int32_t slot_busy[kSlots];  // 1: the presenter wrote it, the scorer has not let it go yet (0)
    volatile std::int64_t written;            // refreshes written (ring index = (written - 1) % kRing)
    Refresh ring[kRing];
    // scripted input (the test game writes): the moves, as the presenter's raw mouse would see them
    volatile std::uint32_t input_owner;       // 1: the test game drives the mouse - the presenter takes only these
    std::uint32_t pad2;
    volatile std::int64_t mouse_written;      // moves written (ring index = n % kMouseRing; the entry before the count)
    struct Mouse { std::int64_t qpc; std::int32_t dx, dy; } mouse[kMouseRing];
    // the HUD cycle (the test game's hud_cycle): it counts hud_relearn up when it switches how its HUD is drawn - the presenter
    // passes it on to HUD learning (learn again, as the panel's Relearn); the presenter writes where the HUD of the frame shown
    // last came from (PresenterStatus::hud_source: 0 none / not known yet, 1-7 its sources) - the game waits for it
    volatile std::uint32_t hud_relearn;
    volatile std::uint32_t hud_source;
};

}  // namespace fw::bench
