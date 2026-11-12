#pragma once
// The game's ReShade effects on the presenter's own picture (Settings::reshade_on_output) - the presenter's half; the add-on's is
// src/addon/output_effects.hpp (why, and what it hands over).
//
// The game's ReShade, copied next to the presenter as ReShade64.dll, is loaded here as a library: its graphics and input hooks off
// (RESHADE_DISABLE_*_HOOK), its base path and config the presenter's folder (FrameWarp\ReShade.ini, written by the add-on with the game's
// effect folders and FrameWarp\ReShadePreset.ini as the preset). An effect runtime of its own on the presenter's swap chain
// (ReShadeCreateEffectRuntime, ReShade 6.0 on) is driven from the renderer's last pass:
//  - before the final pass: ReShade's own work for the picture (loading effects, uniforms - update_and_present), with this picture's
//    effects marked drawn first, so it draws none - only its splash, on the back buffer the final pass then covers whole;
//  - after the final pass: the effects on the finished back buffer (render_effects with the back buffer itself: ReShade's own
//    permutation, its size, format and colour space), when wanted - then the present.
// The presenter's exe registers itself as a ReShade add-on for the effects' begin/finish events: when they were last drawn (the add-on
// switches the game's own off only while they are) and their GPU time. XPAR's own effect is never on here (the depth feed is the game's).
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdint>
#include <string>
#include <vector>

namespace reshade::api { struct effect_runtime; }

namespace fw {

class OutputEffects {
public:
    ~OutputEffects();
    // ReShade loaded from dir (ReShade64.dll, its config ReShade.ini - both written by the add-on); false, and why (note()), when not.
    bool load(const std::wstring& dir);
    // The effect runtime on the presenter's swap chain - made again after its buffers or colour space change (detach() before).
    bool attach(ID3D12Device* device, ID3D12CommandQueue* queue, IDXGISwapChain3* swapchain, std::uint32_t color_space);
    void detach();
    bool attached() const { return runtime_ != nullptr; }
    // The renderer, each picture it presents: around its final pass (see above).
    void before_final_pass();
    void after_final_pass(ID3D12Resource* back_buffer, UINT index, DXGI_FORMAT format);
    // The main loop: whether the effects are drawn on the pictures from now on; the preset written again by the add-on.
    void want(bool on) { want_ = on; }
    bool wanted() const { return want_; }
    void reload_preset() { reload_ = true; }
    // Drawn since they were last wanted (the add-on's switch); when last drawn (qpc); GPU time a picture (ms, median of the last 64; 0 not known).
    bool drawing() const { return want_ && drawn_since_want_; }
    std::int64_t drawn_qpc() const { return drawn_qpc_; }
    float gpu_ms() const { return gpu_ms_; }
    const std::string& note() const { return note_; }
    // (the effects' begin/finish events, on the list they are recorded on)
    reshade::api::effect_runtime* runtime() const { return runtime_; }
    void stamp_begin(ID3D12GraphicsCommandList* list);
    void stamp_finish(ID3D12GraphicsCommandList* list);

private:
    void read_times();
    void keep_feed_off();

    HMODULE module_ = nullptr;
    std::string config_utf8_, preset_utf8_;
    reshade::api::effect_runtime* runtime_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    bool want_ = false, reload_ = false, drawn_since_want_ = false, was_wanted_ = false;
    std::int64_t drawn_qpc_ = 0;
    std::string note_;
    // The back buffers' render target views (ReShade's API handles), plain and sRGB, as made for which buffer.
    struct View { ID3D12Resource* buffer = nullptr; std::uint64_t rtv = 0, rtv_srgb = 0; };
    View views_[3];
    // GPU time: a timestamp at the effects' begin and finish, per picture in a ring; read once the picture's fence is passed.
    static constexpr UINT kRing = 4;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> stamps_;
    Microsoft::WRL::ComPtr<ID3D12Resource> readback_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    std::uint64_t fence_value_ = 0, slot_fence_[kRing] = {};
    bool slot_begun_[kRing] = {}, slot_done_[kRing] = {};
    UINT slot_ = 0;
    double tick_ms_ = 0.0;
    std::vector<float> times_;
    float gpu_ms_ = 0.0f;
};

}  // namespace fw
