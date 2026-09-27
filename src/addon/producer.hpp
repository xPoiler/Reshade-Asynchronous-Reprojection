#pragma once
#include <atomic>
// Game-side half of the transport. Runs inside the game process and only ever records
// barriers + copies into the game's own command lists, then signals a shared fence at present.
#include "shared/protocol.hpp"
#include <d3d12.h>
#include <wrl/client.h>
#include <mutex>
#include <vector>

namespace fw {

class Producer {
public:
    Producer();
    ~Producer();
    Producer(const Producer&) = delete;
    Producer& operator=(const Producer&) = delete;

    Shared* shared() { return shared_; }
    bool ready() const { return shared_ != nullptr && device_ != nullptr; }

    // First D3D12 device seen (the game's). Creates the shared fence.
    bool attach(ID3D12Device* device);
    void set_swapchain(HWND hwnd, std::uint32_t width, std::uint32_t height, DXGI_FORMAT format, std::uint32_t color_space);

    void on_constants(std::uint64_t frame, const Camera& camera);
    void on_sim_start(std::uint64_t frame, std::int64_t qpc);
    void on_present_marker(std::uint64_t frame);
    // Highest frame id seen from the game (camera data or markers).
    std::uint64_t highest_frame() const { return highest_frame_.load(); }
    // The frame the game is rendering now (Reflex / PCL render-submit marker; 0 if the game sends none).
    void on_render_submit(std::uint64_t frame) { rendering_frame_ = frame; note_frame(frame); }
    std::uint64_t rendering_frame() const { return rendering_frame_.load(); }
    // Records a copy of one tagged buffer for `frame` into the game's command list.
    void on_tag(std::uint64_t frame, Tex kind, ID3D12Resource* source, D3D12_RESOURCE_STATES state,
                std::uint32_t ext_x, std::uint32_t ext_y, std::uint32_t ext_w, std::uint32_t ext_h,
                ID3D12GraphicsCommandList* list);
    // At the game's present: copies the backbuffer via `list`, then the caller must submit `list`
    // on `queue`; finish_present() signals the fence afterwards.
    std::uint64_t begin_present(ID3D12Resource* backbuffer, ID3D12GraphicsCommandList* list);
    void finish_present(ID3D12CommandQueue* queue, std::uint64_t frame);

    void set_message(const char* text);

private:
    std::atomic<std::uint64_t> highest_frame_{0}, rendering_frame_{0};
    void note_frame(std::uint64_t f) { std::uint64_t h = highest_frame_.load(); while (f > h && !highest_frame_.compare_exchange_weak(h, f)) {} }
    std::uint64_t last_counted_frame_ = 0;  // frames_total counts each rendered frame once
    struct Texture {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        HANDLE handle = nullptr;
        std::uint32_t generation = 0;
        D3D12_RESOURCE_DESC desc{};
    };
    struct Retired { Microsoft::WRL::ComPtr<ID3D12Resource> resource; HANDLE handle; std::uint64_t after_fence; };

    int slot_for_frame(std::uint64_t frame, bool create);
    bool ensure_texture(int slot, Tex kind, const D3D12_RESOURCE_DESC& source_desc);
    void copy_into(ID3D12GraphicsCommandList* list, ID3D12Resource* source, D3D12_RESOURCE_STATES state, ID3D12Resource* target);
    void collect_retired();

    std::mutex mutex_;
    HANDLE mapping_ = nullptr;
    Shared* shared_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
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
