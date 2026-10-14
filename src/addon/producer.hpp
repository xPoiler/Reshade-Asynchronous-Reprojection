#pragma once
#include <atomic>
// Game-side half of the transport. Runs inside the game process and only ever records
// barriers + copies into the game's own command lists, then signals a shared fence at present.
#include "shared/protocol.hpp"
#include "addon/vk_image.hpp"
#include <d3d11_4.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <memory>
#include <mutex>
#include <vector>

namespace reshade::api { struct device; struct command_queue; }

namespace fw {
// While the producer records a tag's copy onto the game's command list (on_tag): the list as ReShade's own events name
// it (addon.cpp's barrier / copy handlers set it). The game's D3D12 lists reach the hooks as ReShade's wrappers, and
// ReShade names the list it wraps when the game hands it to the GPU (execute_command_list): a frame's list is matched by
// that name (present_slot's newest submitted frame).
extern thread_local bool tl_tagging;
extern thread_local std::uint64_t tl_tag_list;
class VkTransport;

// Every setting at its default (producer.cpp).
void default_settings(Settings& s);

class Producer {
public:
    Producer();
    ~Producer();
    Producer(const Producer&) = delete;
    Producer& operator=(const Producer&) = delete;

    Shared* shared() { return shared_; }
    // The game's camera was found not to match its motion vectors: it is left out (see Shared::game_camera_check).
    bool game_camera_unusable() const { return shared_ && shared_->game_camera_check == 1; }
    bool ready() const { return shared_ != nullptr && device_ != nullptr; }
    // The game itself gave a frame depth and motion vectors (an upscaler call, Streamline tags) within the
    // last two seconds: the ReShade feed (reshade_feed.hpp) stays out. Not for the session: DLSS or FSR
    // switched off in the game's settings leaves the feed as the only source, from then on.
    void note_game_depth() { game_depth_qpc_ = qpc_now(); }
    bool game_depth_seen() const {
        const std::int64_t last = game_depth_qpc_.load();
        return last && shared_ && qpc_now() - last < shared_->qpc_frequency * 2;
    }
    // The ReShade feed published a frame within the last quarter of a second: a Streamline camera that
    // arrives without depth is left out meanwhile (the feed's frames carry an estimated camera). Never while
    // the game gives depth itself: the feed is over then, and its camera counts from that very frame.
    void note_feed() { feed_qpc_ = qpc_now(); }
    bool feed_publishing() const {
        const std::int64_t last = feed_qpc_.load();
        return last && !game_depth_seen() && shared_ && qpc_now() - last < shared_->qpc_frequency / 4;
    }

    // First D3D12 device seen (the game's). Creates the shared fence.
    bool attach(ID3D12Device* device);
    // Vulkan games: a D3D12 device of our own on the game's GPU creates the shared textures, which are
    // imported into the game's Vulkan device (see vk_transport.hpp). detach_vulkan before the device goes.
    bool attach_vulkan(reshade::api::device* device);
    void detach_vulkan();
    bool vulkan() const { return vk_ != nullptr; }
    // Direct3D 11 games: the same, the shared textures and fence opened in the game's D3D11 device (11.4:
    // Windows 10 1703 or later); copies and the fence signal go through its immediate context.
    bool attach_d3d11(reshade::api::device* device);
    void detach_d3d11();
    bool d3d11() const { return d11_ != nullptr; }
    void on_tag_d3d11(std::uint64_t frame, Tex kind, ID3D11Resource* source, std::uint32_t ext_x, std::uint32_t ext_y,
                      std::uint32_t ext_w, std::uint32_t ext_h, ID3D11DeviceContext* context);
    std::uint64_t begin_present_d3d11(ID3D11Resource* backbuffer, ID3D11DeviceContext* context);
    void finish_present_d3d11(ID3D11DeviceContext* context, std::uint64_t token);
    void set_swapchain(HWND hwnd, std::uint32_t width, std::uint32_t height, DXGI_FORMAT format, std::uint32_t color_space);

    void on_constants(std::uint64_t frame, const Camera& camera);
    void on_sim_start(std::uint64_t frame, std::int64_t qpc);
    void on_present_marker(std::uint64_t frame);
    // A command list / buffer handed to the GPU (ExecuteCommandLists, vkQueueSubmit): frames whose data was recorded on
    // it are submitted (see present_slot).
    void on_execute(std::uint64_t list);
    // Highest frame id seen from the game (camera data or markers).
    std::uint64_t highest_frame() const { return highest_frame_.load(); }
    // The frame the game is rendering now (Reflex / PCL render-submit marker; 0 if the game sends none).
    void on_render_submit(std::uint64_t frame) {
        rendering_frame_ = frame;
        note_frame(frame);
        std::lock_guard lock(open_mutex_);
        if (open_frames_.empty() || frame > open_frames_.back()) {
            open_frames_.push_back(frame);
            if (open_frames_.size() > 8) open_frames_.erase(open_frames_.begin());
        }
    }
    std::uint64_t rendering_frame() const { return rendering_frame_.load(); }
    // The frame an upscaler call now belongs to (markers only, 0 without): the oldest frame the game started rendering,
    // not presented yet, with no upscaler data yet. A game rendering ahead (Unreal's render thread: Hi-Fi Rush) has
    // already marked frame N+1 as being rendered when its upscaler runs for frame N - the newest marked frame would
    // pair every presented picture with the previous frame's depth, motion and upscaler output. Without rendering
    // ahead it is the frame being rendered, as before.
    std::uint64_t upscaler_frame() {
        std::lock_guard lock(open_mutex_);
        for (const std::uint64_t f : open_frames_)
            if (f > last_upscaled_) { last_upscaled_ = f; return f; }
        return 0;
    }
    // Frames up to `frame` presented: no longer open.
    void close_frames(std::uint64_t frame) {
        std::lock_guard lock(open_mutex_);
        while (!open_frames_.empty() && open_frames_.front() <= frame) open_frames_.erase(open_frames_.begin());
    }
    // Records a copy of one tagged buffer for `frame` into the game's command list.
    void on_tag(std::uint64_t frame, Tex kind, ID3D12Resource* source, D3D12_RESOURCE_STATES state,
                std::uint32_t ext_x, std::uint32_t ext_y, std::uint32_t ext_w, std::uint32_t ext_h,
                ID3D12GraphicsCommandList* list);
    // The game's HUD layer (see ui_layer.hpp): copied where the game finishes with it, on its `list`, into a texture of
    // our own - then at the present into the presented frame's UI slot (the GPU runs the two in that order, so the frame
    // and its HUD stay together; a UI layer the game tags itself is left as it is).
    void on_ui_layer(ID3D12Resource* source, D3D12_RESOURCE_STATES state, ID3D12GraphicsCommandList* list);
    // The same for a HUD-less copy of the game's final picture (second pattern, ui_layer.hpp) - into the frame's HUD-less
    // slot, unless the frame has a HUD layer or its own HUD-less picture.
    void on_hudless_copy(ID3D12Resource* source, D3D12_RESOURCE_STATES state, ID3D12GraphicsCommandList* list);
    // Vulkan games: the game's HUD layer (VkImage, its format, size, VkImageLayout) copied on the game's command buffer into
    // a texture of our own, then at the present into the presented frame's UI slot (whole, the presenter takes its
    // coverage).
    void on_ui_layer_vk(std::uint64_t image, DXGI_FORMAT format, std::uint32_t width, std::uint32_t height, std::uint32_t layout,
                        void* command_buffer);
    // Direct3D 11 games, third pattern (ui_layer.hpp): the picture just before the game draws its HUD into it - copied;
    // at the present the picture with its HUD compared with that copy, and what differs (the HUD) goes into the frame's UI
    // slot as its coverage.
    // kind (ui_layer.hpp): 0 copy, 1 no HUD this frame (an empty HUD handed over), 2 the copy confirmed (only a confirmed
    // one is handed over), 3 the copy given up.
    void on_hud_before_d3d11(ID3D11Resource* picture, std::uint32_t kind, ID3D11DeviceContext* context);
    // HUD learning (hud_learn.hpp): a copy of the picture at a candidate point (the game's list / immediate context);
    // at the present compared with the final picture, into the frame's probe slot (kHudProbe) with the candidate.
    // end: the final picture as the back buffer's draw reads it - compared with instead of the back buffer.
    // use: the learned point's - its comparison is also the frame's HUD (kUi, where nothing else gave the HUD).
    void on_hud_probe(ID3D12Resource* picture, D3D12_RESOURCE_STATES state, ID3D12GraphicsCommandList* list, std::uint32_t candidate, bool end,
                      bool use);
    void on_hud_probe_d3d11(ID3D11Resource* picture, ID3D11DeviceContext* context, std::uint32_t candidate, bool use);
    // The learned point did not draw this frame: an empty HUD (where nothing else gave the HUD).
    void on_hud_none() { std::lock_guard lock(mutex_); probe_none_ = true; }
    // At the game's present: copies the backbuffer via `list`, then the caller must submit `list`
    // on `queue`; finish_present() signals the fence afterwards.
    std::uint64_t begin_present(ID3D12Resource* backbuffer, ID3D12GraphicsCommandList* list);
    void finish_present(ID3D12CommandQueue* queue, std::uint64_t frame);
    // The same for Vulkan games (Vulkan handles as integers: VkImage, VkFormat, VkImageLayout, VkCommandBuffer).
    // `aspect` is the view's aspect mask; the image's own width/height are copied (ext_* is the valid region).
    void on_tag_vk(std::uint64_t frame, Tex kind, std::uint64_t image, std::uint32_t vk_format, std::uint32_t aspect, std::uint32_t mip,
                   std::uint32_t layer, std::uint32_t layout, std::uint32_t width, std::uint32_t height, std::uint32_t ext_x,
                   std::uint32_t ext_y, std::uint32_t ext_w, std::uint32_t ext_h, void* command_buffer);
    std::uint64_t begin_present_vk(std::uint64_t image, DXGI_FORMAT format, std::uint32_t width, std::uint32_t height, void* command_buffer);
    void finish_present_vk(reshade::api::command_queue* queue, std::uint64_t token);

    void set_message(const char* text);

    // Frame generation (the game's DLSS / FSR frame generation): image number `index` (1 .. per_frame, 0: the
    // next one) of the `per_frame` it generates between the previous frame and `frame`, copied right where it is
    // generated (recorded on `list`, after the generation) - at most kMaxGenerated of them per frame, spread
    // evenly; `real` is the frame's own picture (taken with the first). `frame` 0 or unknown: the frame whose
    // present the game started last. Once its last image is seen, the frame is published as soon as the GPU
    // has run `list` that far (wherever it is submitted).
    // `hudless`: the frame's picture without the HUD, when frame generation has one (FSR's setup) - taken with
    // the frame's own picture, unless the frame has one already (Streamline's).
    void on_generated(std::uint64_t frame, std::uint32_t index, std::uint32_t per_frame, ID3D12Resource* image, D3D12_RESOURCE_STATES image_state,
                      ID3D12Resource* real, D3D12_RESOURCE_STATES real_state, ID3D12GraphicsCommandList* list,
                      ID3D12Resource* hudless = nullptr, D3D12_RESOURCE_STATES hudless_state = D3D12_RESOURCE_STATE_COMMON);
    // The game's frame generation is used (its images taken) with XPAR's own warp engine, the one that shows
    // them; with NVIDIA Latewarp the presenter steps aside while frame generation is on.
    bool generation_wanted() const {
        return shared_ && shared_->settings.enabled && (shared_->settings.warp_engine == 1 || shared_->presenter.latewarp != 2);
    }
    // The game called its frame generation, its images wanted, within the last half second: frames are published
    // from there, and the presents (generated and real images in frame generation's own order) publish nothing.
    bool generation_active() const {
        const std::int64_t last = generated_qpc_.load();
        return last && shared_ && qpc_now() - last < shared_->qpc_frequency / 2;
    }

private:
    std::atomic<std::int64_t> generated_qpc_{0};
    // Frame generation: frames waiting for the GPU to reach their marker (see on_generated).
    struct PendingPublish { std::uint64_t frame; std::uint32_t marker; std::int64_t qpc; };
    std::vector<PendingPublish> pending_publish_;
    std::atomic<int> pending_count_{0};
    static constexpr int kMarkers = 64;
    Microsoft::WRL::ComPtr<ID3D12Resource> markers_;  // readback buffer the marker values are written into
    volatile std::uint32_t* marker_values_ = nullptr;
    std::uint32_t marker_counter_ = 0;
    HANDLE marker_event_ = nullptr;
    bool marker_thread_started_ = false;
    std::atomic<bool> stop_markers_{false};
    bool ensure_markers();
    void poll_markers();
    void release_generated_if_idle();
    void publish(int slot, ID3D12CommandQueue* queue);  // signals the fence on `queue`, marks the slot ready
    std::atomic<std::uint64_t> highest_frame_{0}, rendering_frame_{0};
    // Each slot's last tag's command list, and whether it was handed to the GPU since (under mutex_).
    std::uint64_t slot_list_[kSlots] = {};
    bool slot_submitted_[kSlots] = {};
    std::mutex open_mutex_;                  // (markers come from the game's threads)
    std::vector<std::uint64_t> open_frames_;  // started rendering, not presented, oldest first
    std::uint64_t last_upscaled_ = 0;
    std::atomic<std::int64_t> game_depth_qpc_{0};
    std::atomic<std::int64_t> feed_qpc_{0};
    void note_frame(std::uint64_t f) { std::uint64_t h = highest_frame_.load(); while (f > h && !highest_frame_.compare_exchange_weak(h, f)) {} }
    std::uint64_t last_counted_frame_ = 0;  // frames_total counts each rendered frame once
    struct Texture {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        HANDLE handle = nullptr;
        std::uint32_t generation = 0;
        D3D12_RESOURCE_DESC desc{};
        VkSharedImage vk;  // Vulkan games: the same texture imported into the game's device
        Microsoft::WRL::ComPtr<ID3D11Texture2D> d11;  // D3D11 games: the same texture opened in the game's device
    };
    struct Retired {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource; HANDLE handle; std::uint64_t after_fence; VkSharedImage vk;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> d11;
    };
    bool own_device_for(reshade::api::device* device, const char* api, Microsoft::WRL::ComPtr<ID3D12Device>& own);
    void copy_d3d11(ID3D11DeviceContext* context, ID3D11Resource* source, ID3D11Texture2D* target);

    int slot_for_frame(std::uint64_t frame, bool create);
    bool ensure_texture(int slot, Tex kind, const D3D12_RESOURCE_DESC& source_desc, std::uint32_t vk_depth_bytes = 0);
    int present_slot();  // the slot being presented (marks it, releases older ones); -1: none
    void release_textures();
    void copy_into(ID3D12GraphicsCommandList* list, ID3D12Resource* source, D3D12_RESOURCE_STATES state, ID3D12Resource* target);
    void collect_retired();

    std::mutex mutex_;
    HANDLE mapping_ = nullptr;
    Shared* shared_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12Resource> ui_pending_;  // the HUD layer taken this frame (on_ui_layer)
    // Vulkan: the HUD layer taken (on_ui_layer_vk) and presents since (handed over while at most 1: a game recording ahead
    // takes the next frame's before this present - the GPU runs the copies in order, so this frame's is there)
    Texture ui_vk_;
    std::uint32_t ui_vk_age_ = 2;
    bool ensure_own(Texture& t, const D3D12_RESOURCE_DESC& source);
    // Direct3D 11, third pattern: the copy before the HUD, the picture it was taken from, the comparison's pass and output
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d11_before_, d11_cover_;
    Microsoft::WRL::ComPtr<ID3D11Resource> d11_picture_;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> d11_cover_cs_;
    bool d11_before_taken_ = false, d11_cs_failed_ = false, d11_none_ = false, d11_confirmed_ = false;
    void d3d11_cover(ID3D11DeviceContext* context, int slot);
    bool ensure_d11_cover_cs(ID3D11Device* device);
    // HUD learning: the copy at the candidate point (Direct3D 12 / 11), its candidate, presents since; the comparison.
    Microsoft::WRL::ComPtr<ID3D12Resource> probe_pending_, probe_cov_, probe_end_;
    std::uint32_t probe_age_ = 2, probe_candidate_ = 0, probe_end_age_ = 2;
    bool probe_use_ = false, probe_used_ = false, probe_none_ = false, probe_needs_end_ = false;  // (the copy is the learned point's / its comparison done this frame / none)
    Microsoft::WRL::ComPtr<ID3D12Resource> probe_zero_;  // (an empty coverage: never written)
    void d3d12_learned(ID3D12GraphicsCommandList* list, int slot);
    Microsoft::WRL::ComPtr<ID3D12RootSignature> probe_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> probe_pso_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> probe_heap_;
    UINT probe_ring_ = 0;
    bool probe_failed_ = false;
    bool ensure_probe_pass();
    void d3d12_probe(ID3D12GraphicsCommandList* list, int slot);
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d11_probe_before_, d11_probe_cov_;
    Microsoft::WRL::ComPtr<ID3D11Resource> d11_probe_picture_;
    bool d11_probe_taken_ = false, d11_probe_use_ = false, d11_probe_used_ = false;
    void d3d11_learned(ID3D11DeviceContext* context, int slot);
    void d3d11_probe(ID3D11DeviceContext* context, int slot);
    bool ui_pending_valid_ = false;
    double ui_layer_share_ = 1.0;  // (of the frames with a HUD layer or a HUD-less copy taken: the share with a layer, lately)
    Microsoft::WRL::ComPtr<ID3D12Resource> hudless_pending_;  // the HUD-less copy taken this frame (on_hudless_copy)
    std::uint32_t hudless_age_ = 2;  // presents since the last one (handed over while at most 1)
    std::uint32_t hud_given_ = 0;    // frames in a row with the game's HUD handed over (begin_present)
    std::uint32_t game_hud_frames_ = 0, game_hud_missing_ = 0;
public:
    std::uint32_t hud_given() const { return hud_given_; }
    bool game_gives_hud() const { return game_hud_frames_ >= 60; }  // (HUD learning paused, see begin_present)
private:
    // ...handed over as its coverage only (alpha, 8-bit, one channel - made at the present on ReShade's list): a few MB a
    // frame whatever the game's format, and the frames' textures never change when a menu brings a layer of another one.
    Microsoft::WRL::ComPtr<ID3D12Resource> ui_alpha_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> ui_root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> ui_pso_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> ui_heap_;  // (a ring of SRV + UAV pairs: frames in flight keep theirs)
    UINT ui_heap_step_ = 0, ui_ring_ = 0;
    bool ui_pass_failed_ = false;
    bool ensure_ui_pass();
    Microsoft::WRL::ComPtr<ID3D12Device> own_device_;  // Vulkan games
    std::unique_ptr<VkTransport> vk_;
    Microsoft::WRL::ComPtr<ID3D11Device5> d11_;
    Microsoft::WRL::ComPtr<ID3D11Fence> d11_fence_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fence_handle_ = nullptr;
    std::uint64_t fence_value_ = 0;
    Texture textures_[kSlots][kTexCount];
    std::vector<Retired> retired_;
    std::uint64_t pending_present_frame_ = 0;  // from the PCL present-start marker
    bool have_present_marker_ = false;
    std::uint64_t sim_start_frame_[64]{};
    // Integrated camera position (from clipToPrevClip) and the previous frame's basis.
    bool have_prev_camera_ = false;
    std::uint64_t prev_frame_ = 0;
    std::uint32_t position_epoch_ = 0;
    double world_pos_[3] = {0, 0, 0};
    float prev_right_[3]{}, prev_up_[3]{}, prev_fwd_[3]{};
    float prev_view_to_clip_[16]{};  // previous frame's projection (zoom / near plane changes, see recover_motion)
    std::int64_t sim_start_qpc_[64]{};
};

}  // namespace fw
