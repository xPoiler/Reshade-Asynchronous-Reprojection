#pragma once
// Shared-memory contract between the in-game add-on (producer) and FrameWarpPresenter (consumer).
// Everything here is plain data: both sides are x64 MSVC builds of this same header.
#include <windows.h>
#include <cstdint>
#include <string>

namespace fw {

inline std::int64_t qpc_now() { LARGE_INTEGER v; QueryPerformanceCounter(&v); return v.QuadPart; }

constexpr std::uint32_t kMagic = 0x46574152;  // 'FWAR'
constexpr std::uint32_t kVersion = 94;  // (94: Settings::dev_features_off; 93: Settings::whole_pixels; 92: Settings::even_pacing; 90: a test build installed 2026-11-12, never committed)
// (four: the presenter taking one in, one ready, and the add-on writing two - a game that starts its next frame before
// the last one is presented (its tags and constants come first: Cyberpunk, Enshrouded) has two being written. Three
// slots, 2026-10-28, dropped 43-55% of their frames before the presenter saw them - the reprojection fed at half the
// game's rate, uneven. Each is a 4K picture + HUD layer + depth, ~72 MB of video memory)
constexpr int kSlots = 4;

// Streamline buffer kinds we capture. Values are our own; tags are classified by BufferType + format.
// kScene: the upscaler's output (games without Streamline): the scene before post-processing and HUD.
// kGen0..2: the game's frame generation (DLSS / FSR), the images it generated between the previous frame
// and this one, in order (SlotMeta::generated).
// kHudProbe: HUD learning (hud_learn.hpp) - where the final picture differs from a copy of it taken at one candidate point
// of the frame (R8, 1 differs), for scoring only (SlotMeta::hud_candidate).
// kMotionProbe: the motion search (motion_search.hpp) - one of the game's render targets, tried as its motion vectors.
// kStencil: the stencil of the game's depth-stencil (R8, StencilMask=1, development - see Producer::stencil_stage): where the game marks its
// characters, for the character mask (Renderer::stencil_character).
enum Tex : int { kBackbuffer = 0, kHudless, kUi, kDepth, kMotion, kScene, kGen0, kGen1, kGen2, kHudProbe, kMotionProbe, kStencil, kTexCount };
constexpr int kMaxGenerated = 3;
inline const char* tex_name(int t) {
    static const char* n[] = {"backbuffer", "hudless", "ui", "depth", "motion", "scene", "gen0", "gen1", "gen2", "hudprobe", "motionprobe", "stencil"};
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

// The camera search: a matrix from the game's constant buffers, and where it lies (camera_search.hpp).
struct MatrixCandidate { std::uint32_t key; std::uint32_t flags; float m[16]; };  // (flags: 1 a world to clip by its shape, 2 ...in columns,)
constexpr std::uint32_t kMatrixCandidates = 2048;

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
    std::uint32_t hud_candidate;  // HUD learning: the candidate kHudProbe was taken at (the fingerprint of the drawing that follows it)
    std::uint32_t hud_learned;    // ...kUi is the learned point's coverage (1), or empty: that point did not draw this frame (2)
    std::uint32_t motion_candidate;  // the motion search: the candidate kMotionProbe holds (0: none)
    std::uint32_t matrices;          // the camera search: this frame's candidates
    MatrixCandidate matrix[kMatrixCandidates];
    std::uint32_t motion_channels;   // ...kMotion's channels for x and y, each + 1 (x, y << 8; 0: as they are)
};

// Settings are written by the add-on UI and read by the presenter.
struct Settings {
    std::uint32_t enabled;         // presenter shows warped frames
    std::uint32_t show_original;   // A/B: present latest game frame without warp
    std::uint32_t use_mouse;       // raw mouse drives rotation
    std::uint32_t use_ui_tags;     // pass HUD-less + UI to Latewarp (HUD stays still)
    float rotation_extrapolation;  // 0..1, non-mouse (residual) angular velocity extrapolation
    float max_horizon_ms;          // clamp for extrapolation
    std::uint32_t reset_calibration;  // incremented by UI
    std::uint32_t manual_gain;     // 1: use manual gains below instead of learned
    float manual_gain_x, manual_gain_y;  // radians per count
    float manual_delay_ms;
    std::uint32_t overlay_debug;   // presenter draws a debug indicator
    std::uint32_t invert_warp;     // debug: flip the applied rotation
    float prediction_ms;           // show the camera this far ahead of the game's own latency
    std::uint32_t auto_prediction; // 0: manual slider; 1, 2, 4: -(1/n of the measured game frame); 3: 1/2 with HUD layers, 1/4 without, 1 with XPAR's own motion vectors;
                                   // 5: the camera as of now, held back only as far as the uncovered edge needs (edge_limit_pct, edge_unlimited);
                                   // 6: the lowest latency that uncovers nothing the game has not rendered (now .. one game frame back)
    float present_lead_ms;         // render this long before the next vblank (0: right after the previous one)
    std::uint32_t gpu_priority;    // presenter GPU scheduling class: 0 realtime (default), 1 high, 2 normal
    std::uint32_t no_warp_mask;    // games without HUD layers: detect the HUD and keep it unwarped
    std::uint32_t show_mask;       // debug: tint the no-warp mask (magenta) and the HUD score still learning (green)
    std::uint32_t floor_release;   // option, on by default (saved per game): while turns are predicted around a measured orbit pivot, the
                                   // turn rule does not hold the ground around the character (near the pivot's distance)
    std::uint32_t hold_reticles;   // option, on by default (saved per game): a sight's reticle drawn without depth of its own moves
                                   // with the gun (what is on the glass, learned as a layer; see cs_ret_layer)
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
    std::uint32_t edge_fill;           // own warp, what a turn uncovers past the frame's edge: 0 extended edge pixels, 1 soft, 2 dithered (default; saved per game)
    float edge_limit_pct;              // Auto latency "camera as of now": the most of the screen's width (percent) the warp may uncover
                                       // beyond what the regular Auto latency uncovers lately (0: no more than it; saved per game)
    std::uint32_t edge_unlimited;      // ...or no limit at all (saved per game)
    std::uint32_t frame_cap;           // default on: prevent GPU queueing (hold the game just below its GPU rate) while its NVIDIA Reflex is off (add-on; saved per game)
    std::uint32_t game_hud_layers;     // default on: the HUD from the game's own HUD layer / a HUD-less copy of its final picture where the add-on
                                       // finds one (Direct3D 12; ui_layer.hpp); off: HUD detection only (add-on; saved per game)
    std::uint32_t warp_splat;          // default off: the XPAR warp by forward splat (every rendered pixel claims where it lands, the nearest
                                       // winning) instead of the search from each displayed pixel alone (saved per game)
    std::uint32_t hold_characters;     // default on: what stays in place on screen over several frames while the camera moves is
                                       // held, and the character's limbs with it (Renderer::hold_characters; saved per game)
    std::uint32_t hud_learn;           // default on: learn where the game draws its HUD (hud_learn.hpp) and take the HUD from there
    std::uint32_t hud_relearn;         // incremented by the panel: learn again
    std::uint32_t follow_share;        // default on: the character/weapon by how much of the camera's turn each spot follows
                                       // (Renderer::follow_share) instead of held whole (hold_characters, the stretch; saved per game)
    std::uint32_t share_from_input;    // default on: ...measured against the turn the input explains (what the warp applies), not
                                       // the game camera's whole turn (a car's steering, a gun's own animation; saved per game)
    std::uint32_t memory_resolution;   // background memory: 0 the picture's own resolution, 1 half, 2 a quarter (default) - less
                                       // video memory and GPU time, softer where it fills (saved per game)
    std::uint32_t fps_counter;         // the presenter's frame counter in a corner of its picture - the game's frames and the
                                       // refreshes shown a second: 0 off, 1 top left, 2 top right (default), 3 bottom left, 4 bottom
                                       // right (saved per game; the Steam overlay's own counter is not drawn on the presenter)
    std::uint32_t reshade_on_output;   // default on: the game's ReShade effects drawn on XPAR's own picture, every refresh, instead of on the
                                       // game's (output_effects.hpp) - the picture shown then has them whichever way the frame was taken in
                                       // (back buffer, HUD-less picture, frame generation). Read when the presenter starts (saved per game)
    std::uint32_t object_interp;       // default 2, on (0 off; 1 the newest frame shifted, development): what moves on its own - characters, other movers - shown between game
                                       // frames where it was at the moment the camera's position is shown for, as the first-person weapon
                                       // is (Renderer::set_objects_between; XPAR engine, follow share; saved per game)
    float ui_scale;                    // the XPAR panel's text and controls, 1 (default) .. 4 times (add-on only; saved per game)
    std::uint32_t fps_counter_scale;   // the frame counter's size, 1 (default) .. 4 times (saved per game)
    std::uint32_t latency_counter;     // default off: two rows under the frame counter - FT, the game's frame time (ms, 1000 /
                                       // its frames a second; LAT, the whole latency, until 2026-11-18), and XPAR, what the warp
                                       // adds (+) or takes off (-) against the game's own latency (the camera's age on screen
                                       // minus the game's input to finished frame) - see main.cpp, LatencyRows (saved per game)
    std::uint32_t even_pacing;         // default on: while the controller's right stick turns the camera, the turn shown goes with the stick,
                                       // fitted to the game's camera over its last frames, not with when the game's frames came - even at any
                                       // frame rate, also where frames carry uneven game time (PoseSettings::input_paced; saved per game)
    std::uint32_t whole_pixels;        // default on: the warp's picture from the nearest of the game's pixels, never a blend of neighbours -
                                       // a blend softened fine detail by how far between pixels the warp landed, a turn going sharp and soft
                                       // with its speed (Renderer::set_whole_pixels; saved per game)
    std::uint32_t dev_features_off;    // development (the panel's "Feature test", shown with ShowAllSettings=1; never saved, 0 each launch):
                                       // categories of work switched off by what they do in the picture - DevFeature bits, applied over the
                                       // settings above by the presenter (main.cpp, apply_feature_test) and the add-on's HUD capture
};
// (Settings::dev_features_off: each bit one category, by what it does in the final picture)
enum DevFeature : std::uint32_t {
    kDevCarried = 1u,    // the weapon / character carried with the camera moves with it (follow share, attached, hold, sight reticles)
    kDevTether = 2u,     // no tears between near and far (the continuous warp's field; off: the old holding way)
    kDevHud = 4u,        // the HUD kept still (game HUD layers, HUD detection and learning, fill behind the HUD)
    kDevMemory = 8u,     // what a turn uncovers shown from the scenery seen before (background memory)
    kDevBetween = 16u,   // what moves on its own shown between game frames (two frames blended, the weapon's own motion)
    kDevEdges = 32u,     // the screen's edges past the frame dithered (off: the frame's edge pixels extended)
    kDevLimits = 64u,    // not a picture category: each pass only where it can change its result (Renderer::set_pass_limits,
                         // set_tether_skip) - the same picture (the solve's skip: but for a few pixels at the screen's edge);
                         // off: every pass over the whole picture, as before, for the A/B
    // (variants - set: tried instead of the shipped way)
    kDevNearSolve = 128u,  // (no box: the near-steps solve, 8 x 2, 4 x 3, is the shipped one since 2026-11-20)
    kDevBlendNear = 256u,  // the frame before blended in only where something moves on its own within reach (not the scenery's own)
    kDevAdaptSolve = 512u,  // (no box: tried 2026-11-20, came out as the shipped solve) the field solve's steps chosen per area
    // (a category again - set: off) the progressive work of 2026-11-20 - the field solve's blocks by the camera's speed, the
    // learned HUD detection by tile, the solve's untorn blocks skipped, the search pass with a sight-reticle layer; off: each as
    // before, for the A/B in game. (What gives the same picture bit for bit - the sight search near the centre, the pick in
    // pieces, the HUD work packed - has no way back)
    kDevProgressive = 1024u,
};
inline bool dev_off(const Settings& s, std::uint32_t bit) { return (s.dev_features_off & bit) != 0; }
// (the add-on's HUD capture - the game's HUD layers and HUD learning - as the feature test leaves it)
inline bool hud_layers_on(const Settings& s) { return s.game_hud_layers != 0 && !dev_off(s, kDevHud); }
inline bool hud_learn_on(const Settings& s) { return s.hud_learn != 0 && !dev_off(s, kDevHud); }

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
    // The presenter's video memory guard (main.cpp, MemoryGuard): 0 everything in use; close to its budget, let go of -
    // 1 the sight reticle layers, 2 the background memory too, 3 XPAR stepped aside (the game shown as it is).
    std::uint32_t memory_guard;
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
    std::uint32_t orbit_in_use;  // turns are predicted around the measured orbit pivot right now (the game's own camera, a consistent orbit)
    // Where the HUD of the frames taken in comes from (the panel says it): 0 none (not kept still, or not known yet),
    // 1 the game's HUD layers (HUD-less picture + UI layer), 2 the game's HUD-less picture, 3 the game's own HUD layer
    // (found by the add-on), 4 the upscaler's output, 5 the upscaler's output + camera motion check, 6 learned, 7 the HUD point
    // learned (hud_learn.hpp).
    std::uint32_t hud_source;
    // The game's HUD layer (found by the add-on) checked: 0 not yet, 1 drawn over the finished picture - a HUD (a frame
    // without it shows none; the add-on stops copying the upscaler's output), 2 not a HUD (ignored, the HUD is detected).
    std::uint32_t layer_trust;
    char message[256];
    // HUD learning: the presenter's pick of the candidates scored so far (0: none yet), and the learned point that stopped
    // behaving like a HUD (0: none) - the add-on learns again.
    std::uint32_t hud_learn_pick, hud_learn_fail;
    // Where the character mask comes from (the panel says it): 0 estimated from the motion, 1 the game's depth-stencil marks the character
    // (StencilMask=1, development - Renderer::stencil_character)
    std::uint32_t character_source;
    // The game's ReShade effects on the presenter's picture (Settings::reshade_on_output, output_effects.hpp): 0 off (the setting off when
    // the presenter started), 1 not available (reshade_note says why), 2 ready - not drawing (no effects on in the game, its ReShade menu
    // open, the picture not shown, effects still loading), 3 drawing. The last time they were drawn (qpc; the add-on switches the game's
    // own effects off while it is recent) and their GPU time a refresh (ms, median of the last second; 0 not measured).
    std::uint32_t reshade_state;
    float reshade_gpu_ms;
    volatile LONG64 reshade_drawn_qpc;
    char reshade_note[160];
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
    // Prevent GPU queueing (frame_cap.hpp), for the presenter's log: 1 capping; the cap and the game's rate
    // (fps); the GPU done after the present / after the frame's start (medians, ms); held back per frame (ms).
    std::uint32_t cap_active, cap_reflex;  // (cap_reflex: the game's Reflex is on, the cap stands aside)
    std::uint32_t cap_gpu_bound;          // (the cap stands aside: the GPU is the limit, see frame_cap.hpp)
    float cap_fps, cap_game_fps, cap_present_to_done_ms, cap_start_to_done_ms, cap_wait_ms, cap_queued_pct;
    // The game's own HUD layer (ui_layer.hpp), for the panel: the last one found (size, DXGI format; 0: none), how many
    // frames it was taken on and how many of them reached a frame handed over; its coverage pass: 0 not built yet, 1
    // working, 2 failed.
    std::uint32_t ui_layer_w, ui_layer_h, ui_layer_format, ui_layer_taken, ui_layer_handed, ui_layer_pass;
    // ...and the HUD-less copies of its final picture (second pattern): taken, handed over with a frame.
    std::uint32_t hudless_copies, hudless_handed;
    std::uint32_t learn_candidates, learn_active, learn_probes;  // HUD learning: candidates known, the one being tried, copies taken
    std::uint32_t learn_locked, learn_used;  // ...the learned point in use (0: learning), frames it gave the HUD
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
    // Optional masks the game hands DLSS (diagnostics; bits: kNgxMask...), and the bias-current-colour mask's texture.
    std::uint32_t masks, bias_w, bias_h, bias_format;
};
enum NgxMask : std::uint32_t { kNgxMaskTransparency = 1, kNgxMaskParticles = 2, kNgxMaskAnimatedTexture = 4, kNgxMaskBiasCurrentColor = 8,
                               kNgxMaskTransparencyLayer = 16, kNgxMaskTransparencyOpacity = 32, kNgxMaskTransparencyMvecs = 64,
                               kNgxMaskDisocclusion = 128, kNgxMaskResponsivity = 256 };

// AMD FidelityFX (FSR 3.1 / FSR 4) upscaler calls, for diagnostics and games without Streamline.
struct FsrStats {
    std::uint32_t hooks;  // bits: 2*i create, 2*i+1 dispatch for amd_fidelityfx_dx12 / _loader_dx12 / _upscaler_dx12; 6/7: FSR 3.0 SDK create/dispatch; 8/9: FSR 2 create/dispatch; 10-12: FFX API configure; 13-18: frame generation entry points (diagnostics, ffx_hooks.cpp kFgEntries)
    std::uint32_t upscale_creates, upscale_dispatches, resets, create_flags;
    std::uint32_t render_w, render_h, out_w, out_h, depth_format, mv_format, depth_state, output_state;
    float jitter[2], mv_scale[2];
    float near_plane, far_plane, fov;  // as FSR is told them (fov: vertical, radians)
    std::uint32_t frames_published, outputs_copied;
    // Optional masks the game hands FSR (diagnostics): bit 0 reactive, bit 1 transparency & composition; their sizes.
    std::uint32_t masks, reactive_w, reactive_h, composition_w, composition_h;
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
    // 1: the game runs on Vulkan (the add-on, at its present) - the measured orbit pivot is never used there (main.cpp measured_orbit).
    volatile LONG game_vulkan;
    // The motion search's result, written by the presenter (motion_search.hpp): the add-on then hands that target over as the motion.
    volatile LONG motion_found;
    std::uint32_t motion_candidate, motion_channel_x, motion_channel_y;
    float motion_scale_x, motion_scale_y, motion_quality;
    // ...a target that holds vectors only for what moves on its own (motion_partial 1): the raw value of its channels x, y where nothing was drawn, and
    // whether its vectors are the whole motion on the screen (motion_full 1) or the motion on top of the camera's (0).
    std::uint32_t motion_partial, motion_full;
    float motion_clear_x, motion_clear_y;
    // The add-on's turn: the target found is gone (a change of resolution): +1, and the presenter starts over. The presenter's turn: candidates ruled out, a bit each.
    volatile LONG motion_restarts;
    std::uint64_t motion_ruled_out[4];
    volatile LONG camera_share;  // (percent of the last frames the camera found was used on: the status line)
    volatile LONG camera_held[8];  // (the camera search's places that win, by key: the add-on reads them every frame, whatever its budget; 0: none)
    volatile LONG camera_held_state;  // (on the last frame that moved and was judged, the places held: 1 predicted it, 2 did not - the add-on searches whole frames again; 0 not known)
    // The game's ReShade, for the presenter's own (output_effects.hpp), written by the add-on: 1 its effects are on and at least one of the
    // user's techniques is enabled; the preset as the game has it now, written to FrameWarp\ReShadePreset.ini - +1 each time it is written.
    volatile LONG reshade_fx_active;
    volatile LONG reshade_preset_serial;
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
