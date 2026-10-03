#pragma once
// Presenter-side D3D12: own device + high-priority queue on the game's adapter, a DirectComposition
// swapchain on the overlay window, conversion of shared captures into typed private textures, and
// the final blit. Latewarp evaluation is recorded between ingest() and present().
#include "presenter/latewarp12.hpp"
#include "shared/protocol.hpp"
#include <d3d12.h>
#include <dcomp.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <string>
#include <functional>
#include <vector>

namespace fw {
using Microsoft::WRL::ComPtr;

struct IngestedSource {
    bool valid = false;
    bool has_hudless = false, has_ui = false, has_depth = false, has_scene = false;
    std::uint32_t color_w = 0, color_h = 0;
    Rect2 color_rect, depth_rect;
    bool has_motion = false;  // game motion vectors converted into the private motion texture
    bool motion_estimated = false;  // ...or none from the game: XPAR's own, estimated from the picture
    // Frame generation (the game's own): images generated between the previous frame and this one, taken in
    // with it (kPGen0..), of per_frame per frame; taken image i is the game's number gen_index[i], showing the
    // moment gen_index[i] / (per_frame + 1) of the way from the previous frame.
    int generated = 0, per_frame = 0;
    int gen_index[kMaxGenerated] = {};
};

// HUD detection frames: learned from, skipped for too few telling pixels, or held back by the
// whole-frame guard (too many pixels looked like HUD).
struct HudStats {
    int frames = 0, learned = 0, too_little = 0, guarded = 0;
    double share_sum = 0;  // HUD-like share of the telling pixels, over learned frames
    int scene_frames = 0;  // frames whose HUD came from the upscaler's output instead
    double scene_share_sum = 0;  // share of the screen found to be HUD, summed over those frames
    double scene_pass_ms[18] = {};  // GPU time per pass, summed (the last one is the full-resolution final pass)
};

// Least-squares sums relating the game's motion vectors g to the camera-only motion c (uv per frame,
// towards the previous frame) over one game frame; the scale that maps g to uv is sum(g*c) / sum(g*g).
struct MotionFit {
    double gc[2] = {}, gg[2] = {}, cc[2] = {};
    double samples = 0, moving = 0;  // pixels used / pixels flagged as moving objects
    std::uint64_t frame = 0;         // the game frame the sums are from
};

class Renderer {
public:
    ~Renderer();
    bool init(const LUID& adapter, HWND window, std::uint32_t width, std::uint32_t height, DXGI_FORMAT game_format,
              std::uint32_t color_space, std::string& error);
    ID3D12Device* device() const { return device_.Get(); }
    HANDLE waitable() const { return waitable_; }
    const char* queue_priority() const { return priority_name_; }
    bool resize(std::uint32_t width, std::uint32_t height);
    // 1: a frame can only start once the previous one is displayed; 2: one frame may wait in the queue.
    void set_frame_latency(UINT frames) { if (frames != frame_latency_) { swapchain_->SetMaximumFrameLatency(frames); frame_latency_ = frames; } }
    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }

    // Shared objects of one producer session.
    bool open_session(DWORD pid, std::uint64_t session, std::string& error);
    bool session_open(std::uint64_t session) const { return fence_game_ && session_ == session; }
    // Last game fence value the GPU has finished: only frames at or below it are taken, so the
    // presenter never queues behind the game's in-flight GPU work.
    std::uint64_t game_fence_completed() const { return fence_game_ ? fence_game_->GetCompletedValue() : 0; }

    // Frame recording. begin_frame starts a refresh (warp + present) on the realtime queue.
    ID3D12GraphicsCommandList* begin_frame();
    // Split queues (XPAR's own engine, paced): a new game frame is taken in on its own compute queue
    // (normal priority) into one set of the textures the warp reads, while the warp keeps reading the
    // other, shown set; the warp never waits behind the several ms of HUD/mask work of a new frame.
    // Without split queues everything is recorded on the realtime queue as before.
    // set_split switches the mode (waits for the GPU; call with no frame being taken in): true if changed.
    bool set_split(bool on);
    bool split() const { return split_; }
    // Starts recording the taking in of a game frame (the intake queue when split, else as begin_frame);
    // submit_work submits it. mark_intake_complete: the frame's work is all submitted - show_intake then
    // makes it the shown set once the GPU has finished it (true when it did: switch the source then).
    ID3D12GraphicsCommandList* begin_intake();
    void mark_intake_complete() { if (split_) pending_show_ = fence_values_[1]; }
    bool intake_pending() const { return split_ && pending_show_ != 0; }
    bool show_intake();
    // Fence values of the intake work (the realtime queue's fence without split queues).
    std::uint64_t intake_submitted() const { return fence_values_[split_ ? 1 : 0]; }
    bool intake_completed(std::uint64_t value) const { return fences_[split_ ? 1 : 0]->GetCompletedValue() >= value; }
    void wait_for_intake(std::uint64_t value);
    // Development (FW_D3D_DEBUG=1): prints the debug layer's errors since the last call; returns their number.
    int print_debug_messages();
    // Records conversion of slot `slot` into private textures; the queue waits for the game's fence.
    // after_depth (optional) runs once depth and motion vectors are recorded, before the colour work.
    IngestedSource ingest(const Shared& shared, int slot, const std::function<void(const IngestedSource&)>& after_depth = {});
    // Latewarp inputs referring to the private textures of the last ingest.
    LatewarpInputs latewarp_inputs(const IngestedSource& src, bool use_ui_tags);
    // Blits the warped output (or the unwarped private backbuffer) to the swapchain and presents.
    // marker: 0 none, 1 green (warped), 2 red (original).
    void finish_frame(bool warped, int marker);
    // Nothing new to show this refresh (the picture presented last is still the right one): what the refresh
    // recorded so far (a frame taken in on the same queue, for one) is submitted, with no warp, no blit and no
    // present - the composition keeps showing the last picture.
    void skip_frame();
    // Submits the work recorded since begin_frame without presenting (a new game frame taken in between
    // refreshes, so the refresh itself only warps).
    // continuation: the second half of a frame taken in over two submissions (for the GPU usage log).
    void submit_work(bool continuation = false);
    float last_intake_gpu_ms() const { return intake_gpu_ms_; }  // GPU span of the last such submission
    // GPU time spent by the presenter since the last call, for the log: new game frames taken in (split
    // into depth/motion incl. the camera-estimation readback, 4K colour copies, and the rest: HUD
    // detection, masks, motion analysis) and warped refreshes.
    struct GpuUsage {
        std::uint32_t intakes = 0, warps = 0, split = 0, skipped = 0;
        double intake_ms = 0, access_ms = 0, depth_ms = 0, colour_ms = 0, rest_ms = 0, warp_ms = 0;
    };
    GpuUsage take_gpu_usage() { const GpuUsage u = usage_; usage_ = {}; return u; }
    // The work on each game frame after its textures are taken in, pass by pass (GPU timestamps): pass_stamp(p)
    // marks the end of pass p (kPassStart its beginning); each pass's time is from the stamp before it.
    enum IntakePass { kPassStart, kPassAnalyze, kPassObjects, kPassHud, kPassCharacter, kPassMask, kPassMemory, kPassKinds };
    struct PassTimes { double ms[kPassKinds] = {}; std::uint32_t frames = 0; };
    void pass_stamp(IntakePass pass);
    PassTimes take_pass_times() { const PassTimes t = pass_times_; pass_times_ = {}; return t; }
    // Debug: tints the no-warp mask and the HUD score onto the warped output (call after Latewarp).
    void tint_mask();
    // FrameWarp's own warp engine (experimental): writes the warped output like Latewarp would.
    // source_to_target maps the rendered frame's clip space (with depth) to the displayed camera's (row vectors).
    // memory: what the warp uncovers beside held pixels and at the frame's edges comes from the background
    // memory when it holds that spot (see update_memory).
    // objects: moving objects are shown where they are at the displayed moment (see object_motion).
    struct ObjectWarp {
        float frames_back = 0;       // the displayed moment, in game frames before the shown one (0..1)
        float prev_to_target[16];    // the previous frame's clip space -> the displayed camera's
        float view_to_clip[16];      // the shown frame's projection
        bool debug = false;          // development: colour the pixels by what the moving-object pass did
    };
    // generated: show the game's generated image of that index (frame generation, see prepare_generated)
    // instead of the frame; source_to_target is then that image's camera's.
    bool own_warp(const IngestedSource& src, bool use_ui_tags, bool use_mask, const float source_to_target[16], bool depth_inverted,
                  bool memory = false, const ObjectWarp* objects = nullptr, int generated = -1);
    // Frame generation, once per game frame (after the motion analysis): the generated images taken in with
    // the frame made ready for the warp - each one's depth (this frame's, moved back along the motion vectors
    // to the image's moment) and, with the HUD-less picture and UI layer, the UI they show taken back out
    // (the warp puts the frame's UI back over it, as on every frame).
    void prepare_generated(const IngestedSource& src, float scale_x, float scale_y, bool scale_valid, bool use_ui_tags);
    // ...and once frame generation is off: whether its textures are still held, and letting them go (both
    // texture sets and the game's shared copies; waits for the GPU).
    bool has_generated() const;
    void release_generated();
    // Background memory (once per game frame, after the no-warp mask): the scenery around the view as last
    // seen, behind held pixels and just beyond the frame, carried along with the camera (clip_to_prev_clip
    // of this frame) and forgotten after about a second. consecutive: the previous frame taken in was the
    // game's previous frame (otherwise the memory starts over). False: none for this frame.
    bool update_memory(const IngestedSource& src, const float clip_to_prev_clip[16], bool depth_inverted, bool consecutive);
    // Signalled value that completes all realtime-queue work recorded so far.
    std::uint64_t submitted_value() const { return fence_values_[0]; }
    bool completed(std::uint64_t value) const { return fences_[0]->GetCompletedValue() >= value; }
    HRESULT device_removed_reason() const { return device_ ? device_->GetDeviceRemovedReason() : S_OK; }
    // Notes about shared buffers opened during ingest (size/format changes), drained for logging.
    std::vector<std::string> take_notes() { std::vector<std::string> n; n.swap(notes_); return n; }
    void wait_idle();
    void wait_for(std::uint64_t value);
    void flush_and_wait();  // submit what is recorded, wait for it, continue recording the same frame  // until the GPU completed `value` (see submitted_value)
    float last_gpu_ms() const { return gpu_ms_; }
    // Timing of the most recently completed presenter frame, all in CPU QPC ticks.
    struct FrameTiming { std::int64_t submit = 0, gpu_start = 0, gpu_end = 0; bool valid = false; };
    FrameTiming last_timing() const { return timing_; }
    // DXGI frame statistics after the latest present (0 when unavailable).
    struct PresentStats { UINT last_present_count = 0, present_count = 0, present_refresh = 0, sync_refresh = 0; std::int64_t sync_qpc = 0; HRESULT hr = S_OK; };
    PresentStats present_stats() const { return present_stats_; }
    // Once per new game frame with depth and motion vectors: per-pixel object motion (scaled game motion
    // vectors minus the camera-only motion from depth + clipToPrevClip), for the masks, plus the
    // motion-vector scale fit of `frame` (read back a few frames later, see take_motion_fit).
    // near_rule: pixels near the camera that move against the camera model count as attached; turn_rule:
    // so do pixels nearly still on screen while the camera turns (third-person orbit cameras). See cs_analyze.
    void analyze_motion(const IngestedSource& src, const float clip_to_prev_clip[16], float scale_x, float scale_y, bool scale_valid,
                        bool depth_inverted = true, bool near_rule = true, bool turn_rule = false, std::uint64_t frame = 0);
    // Moving objects (option, XPAR engine). set_object_history: each game frame taken in keeps the previous
    // one's picture and depth (what objects uncover, and where they were). object_motion: once per game
    // frame, after analyze_motion: for each pixel the analysis found moving on its own, its straight-line
    // motion in 3D to the previous frame (both frames' depth; the game's clipToPrevClip and projection).
    // False when there is nothing for the warp to move (no previous frame kept, a size change).
    void set_object_history(bool on) { object_history_ = on; if (!on) history_valid_ = objects_built_ = shown_objects_ = false; }
    // held: what is attached to the camera and kept still (character/weapon) moves along its own on-screen motion.
    // picture: also what the picture shows moving differently from the game's motion vectors (shadows, glare,
    // markers), from picture_motion of the same frame.
    bool object_motion(const IngestedSource& src, const float clip_to_prev_clip[16], const float view_to_clip[16], float scale_x, float scale_y,
                       bool held = false, bool picture = false);
    // Moving objects: the motion seen in the picture between the previous game frame and this one (XPAR's own
    // motion estimation, without replacing the game's motion vectors). False: none this frame (no previous picture).
    bool picture_motion();
    // Keep a copy of the previous game frame's colour at each ingest (background for uncovered areas).
    void set_keep_previous_colour(bool on) { keep_previous_ = on; if (!on) previous_valid_ = false; }
    bool take_motion_fit(MotionFit& fit);
    // HUD learning per game frame since the last call (read back a few frames later).
    HudStats take_hud_stats() { const HudStats s = hud_stats_; hud_stats_ = {}; return s; }
    // Once per new game frame (after analyze_motion): the no-warp mask.
    // hud: the HUD (games without HUD layers) - pixels that stay unchanged while the camera moves the
    // scene under them (needs the previous colour, see set_keep_previous_colour), or from the upscaler's
    // output when the source has it; attached: the motion analysis knows which pixels move with the
    // camera (the motion-vector scale is known); keep_attached: keep those unwarped (third-person
    // character, first-person weapon); combined: with the upscaler's output, HUD only where the pixels do
    // not follow the world while the camera moves (needs the previous colour); fill: with the upscaler's
    // output, also predict the scene behind the HUD for the own warp. Returns the R8 mask at output
    // resolution, kept for the frame's outputs.
    ID3D12Resource* build_no_warp_mask(const IngestedSource& src, const float clip_to_prev_clip[16], bool hud, bool attached,
                                       bool keep_attached, bool depth_inverted = true, bool combined = false, bool fill = false,
                                       int stretch = 0);  // render px the scenery around the character/weapon stretches over (0: off)
    // Camera estimation (games without a camera): motion vector (raw) and depth on a grid over the render
    // rect, 4 floats per sample (mv.x, mv.y, depth, valid). Submits the frame's work so far and waits.
    bool sample_motion(const IngestedSource& src, std::uint32_t grid_w, std::uint32_t grid_h, std::vector<float>& out);
    // The same in two steps, without waiting: records the sampling and its readback on the current list;
    // once that work has completed on the GPU, read_motion_samples returns them.
    bool record_motion_samples(const IngestedSource& src, std::uint32_t grid_w, std::uint32_t grid_h);
    bool read_motion_samples(std::vector<float>& out);
    float last_flush_ms() const { return last_flush_ms_; }  // how long sample_motion waited for the GPU
    float last_present_call_ms() const { return last_present_call_ms_; }  // how long the last Present call took (CPU)
    ID3D12Resource* no_warp_mask() const {
        if (split_) return shown_mask_ ? front_[kPMask].texture.Get() : nullptr;
        return mask_ready_ ? private_[kPMask].texture.Get() : nullptr;
    }
    void reset_hud_detection() { reset_hud_ = true; mask_ready_ = false; shown_mask_ = false; mem_last_ = -1; flow_previous_ = false; }

    // Test/diagnostic helper: synchronously reads back the warped output (RGBA16F) or private backbuffer.
    // which: 0 the game's frame, 1 the warped output, 2 the upscaler's output (scene before HUD).
    // (10-12: the picture, depth and motion vectors kept by stash_source; 13: own motion estimation's displacements;
    // 14: the HUD-less picture kept; 15-18: the second kept set's picture, HUD-less picture, depth and motion vectors;
    // 19: the game's HUD-less picture)
    bool read_back(int which, std::vector<std::uint16_t>& pixels, std::uint32_t& w, std::uint32_t& h);
    // Diagnostics: keeps a copy of the current game frame's picture (and HUD-less picture), depth and motion
    // vectors on the GPU (a few milliseconds, nothing is read back), so a capture taken on a later game frame
    // holds consecutive ones. Two sets: 0 is read back as 10-14, 1 as 15-18.
    void stash_source(int set = 0, bool hudless = false);

private:
    struct Private {
        ComPtr<ID3D12Resource> texture;
        std::uint32_t width = 0, height = 0;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    };
    enum PrivateId { kPBackbuffer, kPHudless, kPUi, kPDepth, kPMotion, kPZeroUi, kPOutput, kPObject, kPObjKeys, kPObjMove, kPObjTiles, kPObjOutTiles, kPObjKindA, kPObjKindB, kPGen0, kPGen1, kPGen2, kPGenZ0, kPGenZ1, kPGenZ2, kPPrevColour, kPPrevDepth, kPPrevious, kPHudScore, kPMask, kPScene, kPAttached, kPWorld, kPFill, kPMem0, kPMem1, kPMemZ0, kPMemZ1, kPRampRows, kPRamp, kPWarpMask, kPHudKind, kPStashColor, kPStashDepth, kPStashMotion, kPStashHudless, kPStash2Color, kPStash2Hudless, kPStash2Depth, kPStash2Motion, kPLumaA, kPLumaB, kPFeat, kPFlowA, kPFlowB, kPCount };

    bool create_pipelines(std::string& error);
    // The format a private copy of a game colour image is kept in: the game's own 4-byte format when it
    // holds the values exactly (10-bit, 8-bit, R11G11B10 float), half-float RGBA otherwise or when the GPU
    // cannot load/store that format in shaders. Half the memory traffic of half-float at the same values.
    DXGI_FORMAT colour_format(DXGI_FORMAT source);
    bool ensure_private(PrivateId id, std::uint32_t w, std::uint32_t h, DXGI_FORMAT format);
    // The textures the warp reads, kept twice with split queues (front_: the shown set).
    static bool shown_id(PrivateId id) {
        return id == kPBackbuffer || id == kPHudless || id == kPUi || id == kPDepth || id == kPMask || id == kPFill || id == kPWarpMask ||
               id == kPObjMove || id == kPObjTiles || id == kPPrevColour || id == kPPrevDepth || (id >= kPGen0 && id <= kPGenZ2);
    }
    const Private& shown(PrivateId id) const { return split_ && shown_id(id) ? front_[id] : private_[id]; }
    void swap_shown();  // exchanges the two sets (and the per-texture descriptors)
    void copy_shown_to_output();  // unwarped refresh with split queues (the shown frame as it is)
    ID3D12GraphicsCommandList* begin(int context);
    void execute();  // closes and submits list_ on its context's queue (with the waits it needs)
    void signal();   // signals its context's fence for the frame slot
    void transition(Private& p, D3D12_RESOURCE_STATES to);
    ID3D12Resource* shared_texture(int slot, int kind, std::uint32_t generation);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT index) const;
    void convert(ID3D12Resource* source, DXGI_FORMAT source_format, int slot, int kind, PrivateId target, std::uint32_t w, std::uint32_t h);
    void create_swapchain_views();
    // HUD in this frame from the upscaler's output (into the HUD score texture, 1 = HUD).
    void detect_hud_from_scene(const struct XConstants& base, bool fill);
    void set_x_srv(UINT index, PrivateId id, bool from_shown = false);
    void set_x_uav(UINT index, PrivateId id);
    void x_dispatch(ID3D12PipelineState* pso, const void* constants, UINT srv_table, UINT uav_table, UINT groups_x, UINT groups_y);

    // Two recording contexts: 0 the realtime (direct) queue, 1 the intake (compute) queue with split
    // queues. Frame slots 0-2 belong to context 0 and 3-5 to context 1 (allocators, readbacks, timing).
    static constexpr int kRing = 6;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_, iqueue_;
    ComPtr<ID3D12CommandAllocator> allocators_[kRing];
    ComPtr<ID3D12GraphicsCommandList> list_, lists_[2];
    ComPtr<ID3D12Fence> fences_[2];
    HANDLE fence_event_ = nullptr;
    std::uint64_t fence_values_[2] = {}, frame_values_[kRing] = {};
    int context_ = 0, ring_pos_[2] = {0, 0};
    std::uint64_t timestamp_frequencies_[2] = {1, 1};
    bool split_ = false;
    std::uint64_t pending_show_ = 0;   // intake fence value after which the written set can be shown
    std::uint64_t release_wait_ = 0;   // realtime fence value the intake waits for before rewriting (swap)
    bool built_mask_ = false, shown_mask_ = false, shown_fill_ = false, stretch_built_ = false, shown_stretch_ = false;
    // Own motion estimation (games that give depth but no motion vectors): fills the motion texture from
    // the picture of this game frame and the previous one. Called by ingest, between the depth and the
    // motion samples.
    bool estimate_motion(DXGI_FORMAT colour, bool write_motion = true);  // false: no previous picture
    int flow_luma_ = 0;            // which brightness pyramid holds the previous frame
    bool flow_previous_ = false;   // ...and whether it does (false: no motion this frame)
    bool own_motion_ = false;      // the current frame's motion vectors are XPAR's own (samples: confident ones only)
    int mem_last_ = -1, mem_shown_ = -1, mem_pending_ = -1;  // background memory buffer written last / read by the warp / of the frame to show
    ID3D12CommandQueue* context_queue() const { return context_ ? iqueue_.Get() : queue_.Get(); }
    ComPtr<IDXGIFactory4> factory_;
    ComPtr<IDXGISwapChain3> swapchain_;
    ComPtr<IDCompositionDevice> dcomp_;
    ComPtr<IDCompositionTarget> target_;
    ComPtr<IDCompositionVisual> visual_;
    HANDLE waitable_ = nullptr;
    DXGI_FORMAT swap_format_ = DXGI_FORMAT_R8G8B8A8_UNORM;
    std::uint32_t width_ = 0, height_ = 0, frame_index_ = 0;
    const char* priority_name_ = "normal";

    ComPtr<ID3D12RootSignature> root_, root_x_, root_w_;
    // Passes with the second constant block and 10 SRVs (the XPAR warp and the moving objects).
    void w_dispatch(ID3D12PipelineState* pso, const void* constants, const float w[20], UINT table, UINT groups_x, UINT groups_y);
    // Moving objects: the previous frame's picture and depth (ingest; the new frame's sizes and colour format).
    void keep_history(PrivateId colour, std::uint32_t depth_w, std::uint32_t depth_h, std::uint32_t colour_w, std::uint32_t colour_h,
                      DXGI_FORMAT colour_format);
    ComPtr<ID3D12PipelineState> cs_obj_move_, cs_gen_depth_, cs_gen_unui_, cs_hudless_clear_, cs_hudless_count_, cs_hudless_score_, cs_hudless_fill_, cs_obj_join_init_, cs_obj_join_, cs_obj_join_final_, cs_obj_clear_, cs_obj_clear_tiles_, cs_obj_splat_, cs_obj_fix_, cs_copy4_, cs_copy1_;
    bool object_history_ = false, history_valid_ = false, objects_built_ = false, shown_objects_ = false;
    ComPtr<ID3D12PipelineState> cs_attached_, cs_analyze_, cs_reduce_, cs_hud_, cs_hud_world_, cs_mask_, cs_clear_score_, cs_hud_count_, cs_clear_counts_, cs_sample_, cs_tint_, cs_scene_clear_, cs_scene_accum_, cs_scene_finish_, cs_scene_tiles_, cs_scene_hud_, cs_scene_fill_, cs_scene_grey_, cs_scene_grey_finish_, cs_scene_wash_, cs_scene_wash_finish_, cs_own_warp_, cs_memory_, cs_ramp_rows_, cs_ramp_, cs_flow_luma_, cs_flow_feat_, cs_flow_search_, cs_flow_lk_, cs_flow_median_, cs_flow_motion_, cs_flow_fill_;
    ComPtr<ID3D12Resource> samples_, samples_readback_;
    UINT samples_count_ = 0;
    float last_flush_ms_ = 0;
    float last_present_call_ms_ = 0;
    ComPtr<ID3D12Resource> hud_counts_, hud_readback_, scene_fit_, scene_stamps_readback_;
    ComPtr<ID3D12QueryHeap> scene_stamps_;
    bool hud_pending_[kRing] = {}, hud_scene_pending_[kRing] = {}, hud_from_scene_ = false;
    double hud_scene_pixels_ = 1;
    HudStats hud_stats_;
    bool mask_ready_ = false, reset_hud_ = false;
    bool fill_ready_ = false;  // the scene behind the HUD was predicted for the current source (own warp)
    ComPtr<ID3D12Resource> partials_, sums_, fit_readback_;
    UINT partial_groups_ = 0;
    bool fit_pending_[kRing] = {};
    std::uint64_t fit_frame_[kRing] = {};
    bool fit_ready_ = false;
    bool keep_previous_ = false, previous_valid_ = false, previous_from_hudless_ = false, ingested_ = false, last_had_hudless_ = false;
    MotionFit fit_latest_;
    ComPtr<ID3D12PipelineState> cs_color_, cs_depth_, blit_;
    ComPtr<ID3D12DescriptorHeap> heap_, rtv_heap_;
    UINT descriptor_size_ = 0, rtv_size_ = 0;
    ComPtr<ID3D12QueryHeap> timestamps_;
    ComPtr<ID3D12Resource> readback_;
    std::uint64_t timestamp_frequency_ = 1;
    std::uint64_t calib_gpu_ = 0, calib_cpu_ = 0;  // GetClockCalibration pair
    std::int64_t submit_qpc_[kRing] = {};
    FrameTiming timing_;
    std::uint64_t present_counter_ = 0;
    std::vector<std::string> notes_;
    UINT frame_latency_ = 1;
    PresentStats present_stats_;
    float gpu_ms_ = 0, intake_gpu_ms_ = 0;
    bool intake_slot_[kRing] = {}, continuation_slot_[kRing] = {}, skipped_slot_[kRing] = {};
    // Three stamps inside an intake (shared textures made readable, after depth/motion, after the colour
    // copies), one set per frame in flight.
    UINT samples_pending_bytes_ = 0;  // motion samples recorded, not read yet
    ComPtr<ID3D12QueryHeap> stage_stamps_;
    ComPtr<ID3D12Resource> stage_readback_;
    int stages_marked_ = 0;
    bool stage_valid_[kRing] = {};
    GpuUsage usage_;
    static constexpr int kPassStamps = 12;
    ComPtr<ID3D12QueryHeap> pass_stamps_;
    ComPtr<ID3D12Resource> pass_readback_;
    std::uint8_t pass_ids_[kRing][kPassStamps] = {};
    int pass_count_[kRing] = {};
    PassTimes pass_times_;
    void mark_stage() {
        if (stage_stamps_ && stages_marked_ < 3) list_->EndQuery(stage_stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 3 + stages_marked_++);
    }

    Private private_[kPCount];
    Private front_[kPCount];  // the shown set (split queues; shown_id textures only)
    DWORD pid_ = 0;
    std::uint64_t session_ = 0;
    ComPtr<ID3D12Fence> fence_game_;
    std::uint64_t pending_game_waits_[2] = {};  // game fence value the next submission of each context waits for
    struct SharedTex { ComPtr<ID3D12Resource> resource; std::uint32_t generation = 0; };
    SharedTex shared_[kSlots][kTexCount];
    std::vector<ID3D12Resource*> reading_;  // shared textures transitioned for this frame
    ID3D12Resource* srv_resource_[kSlots * kTexCount] = {};  // resource currently described at each source SRV index
};

}  // namespace fw
