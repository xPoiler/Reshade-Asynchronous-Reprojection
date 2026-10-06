#pragma once
// Shared-memory contract between the in-game add-on (producer) and FrameWarpPresenter (consumer).
// Everything here is plain data: both sides are x64 MSVC builds of this same header.
#include <windows.h>
#include <cstdint>
#include <string>

namespace fw {

inline std::int64_t qpc_now() { LARGE_INTEGER v; QueryPerformanceCounter(&v); return v.QuadPart; }

constexpr std::uint32_t kMagic = 0x46574152;  // 'FWAR'
constexpr std::uint32_t kVersion = 48;
constexpr int kSlots = 4;

// Streamline buffer kinds we capture. Values are our own; tags are classified by BufferType + format.
// kScene: the upscaler's output (games without Streamline): the scene before post-processing and HUD.
// kGen0..2: the game's frame generation (DLSS / FSR), the images it generated between the previous frame
// and this one, in order (SlotMeta::generated).
enum Tex : int { kBackbuffer = 0, kHudless, kUi, kDepth, kMotion, kScene, kGen0, kGen1, kGen2, kTexCount };
constexpr int kMaxGenerated = 3;
inline const char* tex_name(int t) {
    static const char* n[] = {"backbuffer", "hudless", "ui", "depth", "motion", "scene", "gen0", "gen1", "gen2"};
    return t >= 0 && t < kTexCount ? n[t] : "?";
}

// Slot lifecycle: Free -> Writing (game) -> Ready (game signalled fence) -> Reading (presenter) -> Free.
enum SlotState : LONG { kFree = 0, kWriting = 1, kReady = 2, kReading = 3 };

struct TexInfo {
    std::uint32_t width, height, format;  // DXGI_FORMAT of the shared copy (== source format)
    std::uint32_t generation;             // bumps when the shared texture is recreated
    std::uint32_t ext_x, ext_y, ext_w, ext_h;  // valid region inside the texture
    std::uint32_t valid;                  // written for this slot's frame
    std::uint32_t pad;
};

// Camera for one rendered frame, taken verbatim from sl::Constants (row-major, row-vector, UE view space:
// x right, y up, z forward). Positions/directions are in game world space.
struct Camera {
    float view_to_clip[16];
    float clip_to_view[16];
    float clip_to_prev_clip[16];  // the game's own reprojection to the previous frame
    float pos[3], up[3], right[3], fwd[3];
    float near_plane, far_plane, fov, aspect;
    float jitter[2], mvec_scale[2];
    std::uint32_t depth_inverted, reset, valid, position_epoch;  // epoch changes when pos is discontinuous
    std::uint32_t estimated;  // 1: no camera from the game (DLSS without Streamline): the presenter estimates it
    std::uint32_t pad;
};

struct SlotMeta {
    volatile LONG state;
    std::uint32_t pad0;
    std::uint64_t frame_id;     // Streamline frame token
    std::uint64_t fence_value;  // game-side shared fence value that completes this slot's copies
    std::int64_t qpc_sim_start;  // PCL simulation-start marker for frame_id (0 if unknown)
    std::int64_t qpc_constants;  // when slSetConstants arrived for frame_id
    std::int64_t qpc_present;    // when the game presented frame_id
    Camera camera;
    TexInfo tex[kTexCount];
    // Frame generation: images generated between the previous frame and this one, `per_frame` of them per frame
    // (2x: 1, 4x: 3, 6x: 5), of which `generated` are taken (kGen0.., at most kMaxGenerated, spread evenly).
    // Taken image i is the game's image number gen_index[i] (1 .. per_frame), showing the moment
    // gen_index[i] / (per_frame + 1) of the way from the previous frame to this one. gen_last: the highest
    // image number seen so far (all seen: gen_last == per_frame). 0: none (no frame generation).
    std::uint32_t generated, per_frame, gen_last;
    std::uint32_t gen_index[kMaxGenerated];
};

// Settings are written by the add-on UI and read by the presenter.
struct Settings {
    std::uint32_t enabled;         // presenter shows warped frames
    std::uint32_t show_original;   // A/B: present latest game frame without warp
    std::uint32_t use_mouse;       // raw mouse drives rotation
    std::uint32_t use_ui_tags;     // pass HUD-less + UI to Latewarp (HUD stays still)
    float rotation_extrapolation;  // 0..1, non-mouse (residual) angular velocity extrapolation
    float translation_extrapolation;  // 0..1
    float orbit_distance;          // world units in front of the camera to orbit around (0 = first person)
    float max_horizon_ms;          // clamp for extrapolation
    std::uint32_t reset_calibration;  // incremented by UI
    std::uint32_t manual_gain;     // 1: use manual gains below instead of learned
    float manual_gain_x, manual_gain_y;  // radians per count
    float manual_delay_ms;
    std::uint32_t overlay_debug;   // presenter draws a debug indicator
    std::uint32_t invert_warp;     // debug: flip the applied rotation
    float prediction_ms;           // show the camera this far ahead of the game's own latency
    std::uint32_t auto_prediction; // 0: manual slider; 1, 2, 4: -(1/n of the measured game frame); 3: 1/2 with HUD layers, 1/4 without, 1 with XPAR's own motion vectors
    float present_lead_ms;         // render this long before the next vblank (0: right after the previous one)
    std::uint32_t gpu_priority;    // presenter GPU scheduling class: 0 realtime (default), 1 high, 2 normal
    std::uint32_t moving_objects;  // XPAR engine: objects move at the display rate, between the game's frames (option, off by default)
    std::uint32_t no_warp_mask;    // games without HUD layers: detect the HUD and keep it unwarped
    std::uint32_t show_mask;       // debug: tint the no-warp mask (magenta) and the HUD score still learning (green)
    std::uint32_t hud_from_scene;  // HUD detection, saved per game in ReShade.ini: 0 learned, 1 from the upscaler's output,
                                   // 2 (default) from the upscaler's output where the pixels do not follow the world (combined);
                                   // without an upscaler output the learned detection is used
    std::uint32_t keep_attached;   // keep what moves with the camera (third-person character, first-person weapon) unwarped, every game
    std::uint32_t warp_engine;     // 0: NVIDIA Latewarp, 1: own engine (default; also used when Latewarp is missing)
    std::uint32_t record_diagnostics;  // opt-in: detailed CSV recordings in logs\ (saved per game in ReShade.ini)
    std::uint32_t hud_fill;            // default on (own warp, HUD from the upscaler's output): fill what the HUD covers from the upscaler's output (saved per game)
    std::uint32_t turn_rule;           // default on: what stays nearly still on screen while the camera turns is held (orbit cameras; saved per game)
    std::uint32_t background_memory;   // default on, own warp: uncovered areas shown from the scenery last seen there (saved per game)
    std::uint32_t stretch_width;       // own warp: render px the scenery around the character/weapon stretches over, 0 off, default 1 (saved per game)
    std::uint32_t near_camera_rule;    // character/weapon detection: near-camera pixels moving against the camera count (default on, saved per game)
    std::uint32_t use_controller;      // the controller's right stick drives rotation too (option, on by default, saved per game)
    std::uint32_t present_lead_auto;   // default on: the present lead follows the measured warp time (present_lead_ms unused)
};

// Presenter status, displayed by the add-on UI.
struct PresenterStatus {
    volatile LONG pid;
    std::uint32_t hardware_scheduling;  // Windows hardware-accelerated GPU scheduling on the game's GPU: 0 unknown, 1 off, 2 on
    std::int64_t heartbeat_qpc;
    float output_fps, source_fps, warp_gpu_ms, source_age_ms;
    float gain_x, gain_y, delay_ms, fit_quality_x, fit_quality_y;
    std::uint32_t calibrated_x, calibrated_y;
    std::uint32_t frames_presented, frames_warped, sources_consumed;
    std::uint32_t frame_generation;  // 1: the game presents more images than it renders (frame generation): stepped aside
    float tau_x_ms, tau_y_ms, latency_ms, orbit_cm;
    float frame_interval_ms, effective_prediction_ms;
    float display_hz;  // measured refresh rate of the display the overlay is on (0 = not measured yet)
    float mv_scale_x, mv_scale_y;  // fitted game motion vector -> uv scale (0: not fitted)
    float mv_fit_quality;          // R^2 of the fit on the latest frames
    float moving_fraction;         // share of pixels flagged as moving objects
    std::uint32_t latewarp;        // NVIDIA Latewarp: 0 not known yet, 1 not available (no DLL, or not an NVIDIA GPU), 2 ready
    // Video memory nearly full (within the last 10 s): Windows left the presenter less than a quarter more than
    // it uses, as when the game and XPAR together need more than the GPU has. MB: the adapter's total use (all
    // processes, -1 unknown) and its size; the presenter's use and budget.
    std::uint32_t vram_pressure;
    float vram_adapter_mb, vram_adapter_size_mb, vram_own_mb, vram_budget_mb;
    // Output held below the refresh rate by something outside XPAR (fps, 0: not seen): for 3 s the presents
    // themselves were held back while XPAR had little to do - a frame rate cap of the graphics driver, such as
    // NVIDIA's "Background Application Max Frame Rate" (the presenter's window is never the focused one).
    std::uint32_t outside_cap_fps;
    // Controller (XInput): 1 connected and read; the learned camera turn rate at full right-stick deflection
    // (rad/s, 0: not learned yet) and the response curve (|s|^power) per axis.
    std::uint32_t controller;
    float stick_gain_x, stick_gain_y, stick_power_x, stick_power_y;
    // Latency breakdown (medians of the last 5 s, ms; 0: not measured): how far behind the moment of rendering the
    // displayed camera is (Auto latency's choice: game latency + a fraction of a frame), from the present to the
    // screen (the swapchain's frame statistics), and the age of the displayed camera when it reaches the screen
    // (the two together: what camera rotation costs, mouse included). Game latency: latency_ms.
    float lat_behind_ms, lat_display_ms, lat_age_ms;
    // The camera is estimated (no camera from the game) and explains its frames poorly lately: the camera model does
    // not learn from it meanwhile (share of the last 60 estimated frames unexplained, percent; 0: fine or not estimated).
    std::uint32_t estimate_unreliable_pct;
    // The present lead in use (automatic or the slider's), ms; presents not yet shown right after ours (median of
    // the last 5 s, ours included: 1 = nothing queued ahead of it; 0: not measured).
    float present_lead_ms;
    float presents_queued;
    char message[256];
};

struct HookStats {
    std::uint32_t constants_calls, tag_calls, tag_for_frame_calls, marker_calls;
    std::int32_t constants_base, tag_base;  // detected BaseStructure size (-1 unknown)
    std::uint32_t hooks_installed;          // bitmask
    std::uint32_t pcl_hooked;
    std::uint32_t marker_counts[16];
    // One entry per Streamline BufferType value seen (first 32): format and size of the last tag.
    std::uint32_t tag_format[64], tag_width[64], tag_height[64], tag_count[64];
    std::uint32_t copies_failed, slots_dropped, frames_published, pad;  // pad = detected ResourceTag stride
    char message[256];
    // Broad Streamline activity diagnostics.
    std::uint32_t export_calls[16];      // per entry of kCountedExports (see streamline_hooks.cpp)
    std::int32_t feature_result[12];     // slIsFeatureLoaded result code per probed feature (-1 = not probed)
    std::uint32_t feature_loaded[12];
    std::int32_t pcl_lookup_result, reflex_lookup_result;
    std::uint32_t probe_calls[8];  // game-specific plugin probes (game_probe.cpp)
    std::uint32_t probe_installed;
};

// Game-side event log (lock-free ring) written by the add-on, dumped to CSV by the presenter.
// Frame generation (diagnostics): kEvFrameGen per call into it (extra: the source, see ffx_hooks.cpp /
// ngx_hooks.cpp; FFX API calls carry the description type above the low byte), kEvImage per reshade_present
// (frame: the effect runtime, extra: its back buffer), kEvSwapPresent per present ReShade sees (frame: the
// swapchain, extra: its current back buffer). An image of the game's frame generation not taken: kEvFrameGen
// with extra 0x300 | reason << 16 (kGenSkip...; frame: the frame it was for, 0 when not known yet).
enum EventKind : std::uint32_t { kEvConstants = 1, kEvTag = 2, kEvSimStart = 3, kEvPresentMarker = 4, kEvPresent = 5, kEvPublished = 6,
                                 kEvFrameGen = 7, kEvImage = 8, kEvSwapPresent = 9 };
enum GenSkip : std::uint32_t {
    kGenSkipNotReady = 1,      // the add-on has no device, fence or marker buffer (or no image / list)
    kGenSkipNotWanted = 2,     // reprojection off, or NVIDIA Latewarp as the warp engine
    kGenSkipNoSlot = 3,        // no frame waiting for it (the frame named and the next one)
    kGenSkipNoCamera = 4,      // the frame has no camera yet
    kGenSkipOrder = 5,         // image number out of order or repeated
    kGenSkipNotKept = 6,       // more images per frame than taken (6x)
    kGenSkipNoList2 = 7,       // the command list cannot write the marker (no ID3D12GraphicsCommandList2)
    kGenSkipComputeState = 8,  // a compute list, the image in a graphics-only state
    kGenSkipTexture = 9,       // the shared texture could not be created
    kGenSkipNoOutput = 10,     // the frame generation call had no image (hook side)
    kGenSkipFailed = 11,       // the frame generation call failed (hook side)
};
struct TimelineEvent {
    std::int64_t qpc;
    std::uint32_t kind, tid;
    std::uint64_t frame, extra;
};
constexpr int kTimeline = 4096;

// NGX (DLSS without Streamline) diagnostics, written by the add-on's hooks in the driver's _nvngx.dll.
struct NgxStats {
    std::uint32_t hooks;  // bit 0 CreateFeature, bit 1 EvaluateFeature
    std::uint32_t create_calls, dlss_creates, evaluate_calls, dlss_calls, unknown_handle_calls, resets;
    std::uint32_t feature_calls[16];  // EvaluateFeature calls per NGX feature id
    std::uint32_t create_flags;       // DLSS create flags (HDR, low-res MV, jittered MV, depth inverted, ...)
    std::uint32_t render_w, render_h, out_w, out_h;
    std::uint32_t depth_w, depth_h, depth_format, mv_w, mv_h, mv_format;
    std::uint32_t color_w, color_h, color_format, output_w, output_h, output_format;
    std::uint32_t subrect_w, subrect_h;
    float jitter[2], mv_scale[2];
    std::uint32_t dlss_feature;  // 1 Super Resolution, 13 Ray Reconstruction (0: none seen)
    std::uint32_t identified_by_inputs;  // DLSS handles recognised from their inputs (created before our hooks)
    std::uint32_t frames_published;  // frames published from DLSS calls (camera estimated by the presenter)
    std::uint32_t frames_joined;     // depth and motion vectors taken at the DLSS call for the game's own (Streamline) camera
};

// AMD FidelityFX (FSR 3.1 / FSR 4) upscaler calls, for diagnostics and games without Streamline.
struct FsrStats {
    std::uint32_t hooks;  // bits: 2*i create, 2*i+1 dispatch for amd_fidelityfx_dx12 / _loader_dx12 / _upscaler_dx12; 6/7: FSR 3.0 SDK create/dispatch; 8/9: FSR 2 create/dispatch; 10-12: FFX API configure; 13-18: frame generation entry points (diagnostics, ffx_hooks.cpp kFgEntries)
    std::uint32_t upscale_creates, upscale_dispatches, resets, create_flags;
    std::uint32_t render_w, render_h, out_w, out_h, depth_format, mv_format, depth_state, output_state;
    float jitter[2], mv_scale[2];
    float near_plane, far_plane, fov;  // as FSR is told them (fov: vertical, radians)
    std::uint32_t frames_published, outputs_copied;
};

struct Shared {
    std::uint32_t magic, version;
    std::uint32_t producer_pid, pad0;
    std::uint64_t session;  // unique per producer instance; part of every object name
    LUID adapter;
    std::uint64_t game_hwnd;
    std::uint32_t backbuffer_format, color_space;  // DXGI_FORMAT / DXGI_COLOR_SPACE_TYPE
    std::uint32_t backbuffer_width, backbuffer_height;
    volatile LONG64 latest_ready_frame;
    std::int64_t qpc_frequency;
    SlotMeta slots[kSlots];
    Settings settings;
    PresenterStatus presenter;
    HookStats hooks;
    NgxStats ngx;
    FsrStats fsr;
    volatile LONG presents_without_frame;  // consecutive game presents with no captured frame (menus, loading)
    volatile LONG overlay_open;            // the ReShade menu is open (the game gets no mouse; its frame is shown as it is)
    // The game's own camera checked against its motion vectors (camera_check.hpp), written by the presenter
    // and remembered per game by the add-on: 0 used (nothing against it so far), 1 unusable - the add-on
    // publishes frames from the upscaler's call with a camera the presenter estimates, 2 the estimate was no
    // better - the game's camera stays.
    volatile LONG game_camera_check;
    // Totals since start: images the game presented and frames it rendered (distinct frames with camera
    // data). Frame generation presents two or more images per rendered frame.
    volatile LONG presents_total, frames_total;
    // ...and calls the game made to its frame generation (DLSS FG evaluations, FSR frame generation
    // dispatches): frame generation is on even where ReShade sees only the rendered frames' presents.
    volatile LONG generation_calls;
    volatile LONG64 timeline_count;
    TimelineEvent timeline[kTimeline];
};

inline void push_event(Shared* s, EventKind kind, std::uint64_t frame, std::uint64_t extra = 0) {
    if (!s) return;
    const LONG64 index = InterlockedIncrement64(&s->timeline_count) - 1;
    auto& e = s->timeline[index % kTimeline];
    e.qpc = qpc_now(); e.kind = kind; e.tid = GetCurrentThreadId(); e.frame = frame; e.extra = extra;
}

inline std::wstring map_name(DWORD pid) { return L"Local\\FrameWarp_" + std::to_wstring(pid); }
inline std::wstring object_name(DWORD pid, std::uint64_t session, const wchar_t* what, int slot = -1, int tex = -1,
                                std::uint32_t generation = 0) {
    std::wstring s = L"Local\\FrameWarp_" + std::to_wstring(pid) + L"_" + std::to_wstring(session) + L"_" + what;
    if (slot >= 0) s += L"_s" + std::to_wstring(slot);
    if (tex >= 0) s += L"_t" + std::to_wstring(tex) + L"_g" + std::to_wstring(generation);
    return s;
}


}  // namespace fw
