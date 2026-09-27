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
};

// HUD detection frames: learned from, skipped for too few telling pixels, or held back by the
// whole-frame guard (too many pixels looked like HUD).
struct HudStats {
    int frames = 0, learned = 0, too_little = 0, guarded = 0;
    double share_sum = 0;  // HUD-like share of the telling pixels, over learned frames
    int scene_frames = 0;  // frames whose HUD came from the upscaler's output instead
    double scene_share_sum = 0;  // share of the screen found to be HUD, summed over those frames
    double scene_pass_ms[10] = {};  // GPU time per pass, summed: clear, accum, finish, accum, finish, tiles, accum, finish, tiles, final
};

// Least-squares sums relating the game's motion vectors g to the camera-only motion c (uv per frame,
// towards the previous frame) over one game frame; the scale that maps g to uv is sum(g*c) / sum(g*g).
struct MotionFit {
    double gc[2] = {}, gg[2] = {}, cc[2] = {};
    double samples = 0, moving = 0;  // pixels used / pixels flagged as moving objects
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

    // Frame recording.
    ID3D12GraphicsCommandList* begin_frame();
    // Records conversion of slot `slot` into private textures; the queue waits for the game's fence.
    // after_depth (optional) runs once depth and motion vectors are recorded, before the colour work.
    IngestedSource ingest(const Shared& shared, int slot, const std::function<void(const IngestedSource&)>& after_depth = {});
    // Latewarp inputs referring to the private textures of the last ingest.
    LatewarpInputs latewarp_inputs(const IngestedSource& src, bool use_ui_tags);
    // Blits the warped output (or the unwarped private backbuffer) to the swapchain and presents.
    // marker: 0 none, 1 green (warped), 2 red (original).
    void finish_frame(bool warped, int marker);
    // Submits the work recorded since begin_frame without presenting (a new game frame taken in between
    // refreshes, so the refresh itself only warps).
    void submit_work();
    float last_intake_gpu_ms() const { return intake_gpu_ms_; }  // GPU span of the last such submission
    // Debug: tints the no-warp mask and the HUD score onto the warped output (call after Latewarp).
    void tint_mask();
    // Signalled value that completes all work recorded so far.
    std::uint64_t submitted_value() const { return fence_value_; }
    bool completed(std::uint64_t value) const { return fence_->GetCompletedValue() >= value; }
    HRESULT device_removed_reason() const { return device_ ? device_->GetDeviceRemovedReason() : S_OK; }
    // Notes about shared buffers opened during ingest (size/format changes), drained for logging.
    std::vector<std::string> take_notes() { std::vector<std::string> n; n.swap(notes_); return n; }
    void wait_idle();
    float last_gpu_ms() const { return gpu_ms_; }
    // Timing of the most recently completed presenter frame, all in CPU QPC ticks.
    struct FrameTiming { std::int64_t submit = 0, gpu_start = 0, gpu_end = 0; bool valid = false; };
    FrameTiming last_timing() const { return timing_; }
    // DXGI frame statistics after the latest present (0 when unavailable).
    struct PresentStats { UINT last_present_count = 0, present_count = 0, present_refresh = 0, sync_refresh = 0; std::int64_t sync_qpc = 0; HRESULT hr = S_OK; };
    PresentStats present_stats() const { return present_stats_; }
    // Moving-object extrapolation (experimental). Once per new game frame: per-pixel object motion
    // (scaled game motion vectors minus the camera-only motion from depth + clipToPrevClip), plus the
    // motion-vector scale fit (read back a few frames later, see take_motion_fit).
    void analyze_motion(const IngestedSource& src, const float clip_to_prev_clip[16], float scale_x, float scale_y, bool scale_valid,
                        bool depth_inverted = true);
    // Per output frame: moves object pixels `alpha` game frames forward (negative: back) into a copy of
    // the hud-less colour (or the backbuffer). Returns the result, or nullptr when unavailable.
    ID3D12Resource* extrapolate_objects(const IngestedSource& src, bool from_hudless, float alpha, const float clip_to_prev_clip[16]);
    // Keep a copy of the previous game frame's colour at each ingest (background for uncovered areas).
    void set_keep_previous_colour(bool on) { keep_previous_ = on; if (!on) previous_valid_ = false; }
    bool take_motion_fit(MotionFit& fit);
    // HUD learning per game frame since the last call (read back a few frames later).
    HudStats take_hud_stats() { const HudStats s = hud_stats_; hud_stats_ = {}; return s; }
    // Once per new game frame (after analyze_motion): the no-warp mask for games without HUD layers.
    // hud: pixels that stay unchanged while the camera moves the scene under them (needs the previous
    // colour, see set_keep_previous_colour); attached: pixels whose motion vectors ignore the camera
    // (first-person weapon). Returns the R8 mask at output resolution, kept for the frame's outputs.
    ID3D12Resource* build_no_warp_mask(const IngestedSource& src, const float clip_to_prev_clip[16], bool hud, bool attached,
                                       bool depth_inverted = true);
    // Camera estimation (games without a camera): motion vector (raw) and depth on a grid over the render
    // rect, 4 floats per sample (mv.x, mv.y, depth, valid). Submits the frame's work so far and waits.
    bool sample_motion(const IngestedSource& src, std::uint32_t grid_w, std::uint32_t grid_h, std::vector<float>& out);
    float last_flush_ms() const { return last_flush_ms_; }  // how long sample_motion waited for the GPU
    ID3D12Resource* no_warp_mask() const { return mask_ready_ ? private_[kPMask].texture.Get() : nullptr; }
    void reset_hud_detection() { reset_hud_ = true; mask_ready_ = false; }

    // Test/diagnostic helper: synchronously reads back the warped output (RGBA16F) or private backbuffer.
    // which: 0 the game's frame, 1 the warped output, 2 the upscaler's output (scene before HUD).
    bool read_back(int which, std::vector<std::uint16_t>& pixels, std::uint32_t& w, std::uint32_t& h);

private:
    struct Private {
        ComPtr<ID3D12Resource> texture;
        std::uint32_t width = 0, height = 0;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    };
    enum PrivateId { kPBackbuffer, kPHudless, kPUi, kPDepth, kPMotion, kPZeroUi, kPOutput, kPObject, kPDest, kPExtrap, kPPrevious, kPHudScore, kPMask, kPScene, kPCount };

    bool create_pipelines(std::string& error);
    bool ensure_private(PrivateId id, std::uint32_t w, std::uint32_t h, DXGI_FORMAT format);
    void transition(Private& p, D3D12_RESOURCE_STATES to);
    ID3D12Resource* shared_texture(int slot, int kind, std::uint32_t generation);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT index) const;
    void convert(ID3D12Resource* source, DXGI_FORMAT source_format, int slot, int kind, PrivateId target, std::uint32_t w, std::uint32_t h);
    void create_swapchain_views();
    // HUD in this frame from the upscaler's output (into the HUD score texture, 1 = HUD).
    void detect_hud_from_scene(const struct XConstants& base);
    void flush_and_wait();  // submit what is recorded, wait for it, continue recording the same frame
    void set_x_srv(UINT index, PrivateId id);
    void set_x_uav(UINT index, PrivateId id);
    void x_dispatch(ID3D12PipelineState* pso, const void* constants, UINT srv_table, UINT uav_table, UINT groups_x, UINT groups_y);

    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocators_[3];
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_ = nullptr;
    std::uint64_t fence_value_ = 0, frame_values_[3] = {};
    ComPtr<IDXGIFactory4> factory_;
    ComPtr<IDXGISwapChain3> swapchain_;
    ComPtr<IDCompositionDevice> dcomp_;
    ComPtr<IDCompositionTarget> target_;
    ComPtr<IDCompositionVisual> visual_;
    HANDLE waitable_ = nullptr;
    DXGI_FORMAT swap_format_ = DXGI_FORMAT_R8G8B8A8_UNORM;
    std::uint32_t width_ = 0, height_ = 0, frame_index_ = 0;
    const char* priority_name_ = "normal";

    ComPtr<ID3D12RootSignature> root_, root_x_;
    ComPtr<ID3D12PipelineState> cs_analyze_, cs_reduce_, cs_clear_, cs_splat_, cs_gather_, cs_hud_, cs_mask_, cs_clear_score_, cs_hud_count_, cs_clear_counts_, cs_sample_, cs_tint_, cs_scene_clear_, cs_scene_accum_, cs_scene_finish_, cs_scene_tiles_, cs_scene_hud_;
    ComPtr<ID3D12Resource> samples_, samples_readback_;
    UINT samples_count_ = 0;
    float last_flush_ms_ = 0;
    ComPtr<ID3D12Resource> hud_counts_, hud_readback_, scene_fit_, scene_stamps_readback_;
    ComPtr<ID3D12QueryHeap> scene_stamps_;
    bool hud_pending_[3] = {}, hud_scene_pending_[3] = {}, hud_from_scene_ = false;
    double hud_scene_pixels_ = 1;
    HudStats hud_stats_;
    bool mask_ready_ = false, reset_hud_ = false;
    ComPtr<ID3D12Resource> partials_, sums_, fit_readback_;
    UINT partial_groups_ = 0;
    bool fit_pending_[3] = {};
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
    std::int64_t submit_qpc_[3] = {};
    FrameTiming timing_;
    std::uint64_t present_counter_ = 0;
    std::vector<std::string> notes_;
    UINT frame_latency_ = 1;
    PresentStats present_stats_;
    float gpu_ms_ = 0, intake_gpu_ms_ = 0;
    bool intake_slot_[3] = {};

    Private private_[kPCount];
    DWORD pid_ = 0;
    std::uint64_t session_ = 0;
    ComPtr<ID3D12Fence> fence_game_;
    std::uint64_t pending_game_wait_ = 0;
    struct SharedTex { ComPtr<ID3D12Resource> resource; std::uint32_t generation = 0; };
    SharedTex shared_[kSlots][kTexCount];
    std::vector<ID3D12Resource*> reading_;  // shared textures transitioned for this frame
    ID3D12Resource* srv_resource_[32] = {};  // resource currently described at each source SRV index
};

}  // namespace fw
