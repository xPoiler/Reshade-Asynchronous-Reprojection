#include "presenter/output_effects.hpp"
#include <reshade.hpp>
#include <algorithm>
#include <cstdio>

namespace fw {
namespace {
using namespace reshade::api;

OutputEffects* g_self = nullptr;

// (ReShade's private data on a DXGI swap chain: the colour space set on it, which its own swap chain implementation reads - set by its
// SetColorSpace1 hook in a game, by us here: without hooks nothing else sets it, and the effects took an HDR picture for SDR)
constexpr GUID kSwapChainColorSpace = {0x18b57e4, 0x1493, 0x4953, {0xad, 0xf2, 0xde, 0x6d, 0x99, 0xcc, 0x5, 0xe5}};

std::string utf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(size_t(std::max(n, 0)), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

void on_begin_effects(effect_runtime* runtime, command_list* cmd_list, resource_view, resource_view) {
    if (g_self && cmd_list && runtime == g_self->runtime()) g_self->stamp_begin(reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native()));
}
void on_finish_effects(effect_runtime* runtime, command_list* cmd_list, resource_view, resource_view) {
    if (g_self && cmd_list && runtime == g_self->runtime()) g_self->stamp_finish(reinterpret_cast<ID3D12GraphicsCommandList*>(cmd_list->get_native()));
}
}  // namespace

OutputEffects::~OutputEffects() {
    detach();
    if (g_self == this) {
        reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(on_finish_effects);
        reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
        reshade::unregister_addon(GetModuleHandleW(nullptr), module_);
        g_self = nullptr;
    }
    // (ReShade itself stays loaded until the process ends)
}

bool OutputEffects::load(const std::wstring& dir) {
    if (module_) return true;
    const std::wstring dll = dir + L"\\ReShade64.dll", config = dir + L"\\ReShade.ini";
    if (GetFileAttributesW(dll.c_str()) == INVALID_FILE_ATTRIBUTES) { note_ = "the game's ReShade was not copied next to the presenter"; return false; }
    if (GetFileAttributesW(config.c_str()) == INVALID_FILE_ATTRIBUTES) { note_ = "no ReShade.ini next to the presenter"; return false; }
    // A library here, not a hook: nothing of the presenter's own D3D12 or input taken over, and its folder and config the presenter's
    // (a base path the game's environment would point elsewhere is not ours).
    SetEnvironmentVariableW(L"RESHADE_DISABLE_GRAPHICS_HOOK", L"1");
    SetEnvironmentVariableW(L"RESHADE_DISABLE_INPUT_HOOK", L"1");
    SetEnvironmentVariableW(L"RESHADE_BASE_PATH_OVERRIDE", nullptr);
    HMODULE module = LoadLibraryW(dll.c_str());
    if (!module) {
        char text[128];
        std::snprintf(text, sizeof(text), "ReShade did not load next to the presenter (Windows error %lu)", GetLastError());
        note_ = text;
        return false;
    }
    if (!GetProcAddress(module, "ReShadeCreateEffectRuntime")) { note_ = "the game's ReShade has no add-on support"; return false; }
    if (!reshade::register_addon(GetModuleHandleW(nullptr), module)) { note_ = "the game's ReShade refused the presenter as an add-on (older than XPAR's API)"; return false; }
    module_ = module;
    g_self = this;
    reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_begin_effects);
    reshade::register_event<reshade::addon_event::reshade_finish_effects>(on_finish_effects);
    config_utf8_ = utf8(config);
    preset_utf8_ = utf8(dir + L"\\ReShadePreset.ini");
    note_.clear();
    return true;
}

bool OutputEffects::attach(ID3D12Device* device, ID3D12CommandQueue* queue, IDXGISwapChain3* swapchain, std::uint32_t color_space) {
    if (!module_ || !device || !queue || !swapchain) return false;
    if (runtime_) return true;
    const DXGI_COLOR_SPACE_TYPE cs = static_cast<DXGI_COLOR_SPACE_TYPE>(color_space);
    swapchain->SetPrivateData(kSwapChainColorSpace, sizeof(cs), &cs);
    effect_runtime* runtime = nullptr;
    if (!reshade::create_effect_runtime(device_api::d3d12, device, queue, swapchain, config_utf8_.c_str(), &runtime) || !runtime) {
        note_ = "ReShade could not start on the presenter's swap chain";
        return false;
    }
    runtime_ = runtime;
    queue_ = queue;
    if (!stamps_) {
        D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2 * kRing, 0};
        D3D12_HEAP_PROPERTIES rb{};
        rb.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = 16 * kRing; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
        bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        UINT64 frequency = 0;
        if (FAILED(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&stamps_))) ||
            FAILED(device->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_))) || FAILED(queue->GetTimestampFrequency(&frequency)) || !frequency) {
            stamps_.Reset(); readback_.Reset(); fence_.Reset();
        } else {
            tick_ms_ = 1000.0 / double(frequency);
        }
    }
    for (UINT i = 0; i < kRing; ++i) slot_begun_[i] = slot_done_[i] = false, slot_fence_[i] = 0;
    drawn_since_want_ = false;
    note_.clear();
    return true;
}

void OutputEffects::detach() {
    if (!runtime_) return;
    // (the GPU done with what the effects drew - the renderer waits for its own work before it detaches)
    if (fence_ && fence_->GetCompletedValue() < fence_value_) fence_->SetEventOnCompletion(fence_value_, nullptr);
    device* dev = runtime_->get_device();
    for (View& v : views_) {
        if (v.rtv_srgb && v.rtv_srgb != v.rtv) dev->destroy_resource_view(resource_view{v.rtv_srgb});
        if (v.rtv) dev->destroy_resource_view(resource_view{v.rtv});
        v = View{};
    }
    reshade::destroy_effect_runtime(runtime_);
    runtime_ = nullptr;
    drawn_since_want_ = false;
}

void OutputEffects::keep_feed_off() {
    for (const char* effect : {"XPAR.addonfx", "XPAR.fx"}) {
        const effect_technique t = runtime_->find_technique(effect, "XPAR_Feed");
        if (t.handle && runtime_->get_technique_state(t)) runtime_->set_technique_state(t, false);
    }
}

void OutputEffects::before_final_pass() {
    if (!runtime_) return;
    if (reload_) { reload_ = false; runtime_->set_current_preset_path(preset_utf8_.c_str()); }
    keep_feed_off();
    if (want_ && !was_wanted_) drawn_since_want_ = false;
    was_wanted_ = want_;
    // This picture's effects marked drawn (nothing drawn: no target), so ReShade's own pass for it draws none - then its work for the
    // picture, its splash on the back buffer the final pass covers next.
    command_list* list = runtime_->get_command_queue()->get_immediate_command_list();
    runtime_->render_effects(list, resource_view{0}, resource_view{0});
    reshade::update_and_present_effect_runtime(runtime_);
    read_times();
}

void OutputEffects::after_final_pass(ID3D12Resource* back_buffer, UINT index, DXGI_FORMAT dxgi_format) {
    if (!runtime_ || !want_ || !back_buffer || index >= 3) return;
    device* dev = runtime_->get_device();
    View& v = views_[index];
    if (v.buffer != back_buffer) {
        if (v.rtv_srgb && v.rtv_srgb != v.rtv) dev->destroy_resource_view(resource_view{v.rtv_srgb});
        if (v.rtv) dev->destroy_resource_view(resource_view{v.rtv});
        v = View{};
        v.buffer = back_buffer;
        const resource res{reinterpret_cast<std::uint64_t>(back_buffer)};
        const format plain = format_to_default_typed(static_cast<format>(dxgi_format), 0), srgb = format_to_default_typed(static_cast<format>(dxgi_format), 1);
        resource_view a{0}, b{0};
        if (dev->create_resource_view(res, resource_usage::render_target, resource_view_desc(plain), &a)) {
            v.rtv = a.handle;
            v.rtv_srgb = srgb != plain && dev->create_resource_view(res, resource_usage::render_target, resource_view_desc(srgb), &b) ? b.handle : a.handle;
        }
    }
    if (!v.rtv) return;
    command_queue* queue = runtime_->get_command_queue();
    command_list* list = queue->get_immediate_command_list();
    const resource res{reinterpret_cast<std::uint64_t>(back_buffer)};
    list->barrier(res, resource_usage::present, resource_usage::render_target);
    runtime_->render_effects(list, resource_view{v.rtv}, resource_view{v.rtv_srgb});
    list->barrier(res, resource_usage::render_target, resource_usage::present);
    queue->flush_immediate_command_list();
    if (fence_ && slot_begun_[slot_] && !slot_fence_[slot_]) {
        queue_->Signal(fence_.Get(), ++fence_value_);
        slot_fence_[slot_] = fence_value_;
    }
}

void OutputEffects::stamp_begin(ID3D12GraphicsCommandList* list) {
    LARGE_INTEGER q;
    QueryPerformanceCounter(&q);
    drawn_qpc_ = q.QuadPart;
    drawn_since_want_ = true;
    if (!stamps_ || !list) return;
    const UINT next = (slot_ + 1) % kRing;
    if (slot_begun_[next]) return;  // (its picture not read yet: this one goes untimed)
    slot_ = next;
    list->EndQuery(stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot_ * 2);
    slot_begun_[slot_] = true; slot_done_[slot_] = false; slot_fence_[slot_] = 0;
}

void OutputEffects::stamp_finish(ID3D12GraphicsCommandList* list) {
    if (!stamps_ || !list || !slot_begun_[slot_] || slot_done_[slot_]) return;
    list->EndQuery(stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot_ * 2 + 1);
    list->ResolveQueryData(stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot_ * 2, 2, readback_.Get(), UINT64(slot_) * 16);
    slot_done_[slot_] = true;
}

void OutputEffects::read_times() {
    if (!fence_) return;
    const std::uint64_t completed = fence_->GetCompletedValue();
    for (UINT i = 0; i < kRing; ++i) {
        if (!slot_begun_[i] || !slot_fence_[i] || completed < slot_fence_[i]) continue;
        if (slot_done_[i]) {
            const D3D12_RANGE range{SIZE_T(i) * 16, SIZE_T(i) * 16 + 16};
            void* data = nullptr;
            if (SUCCEEDED(readback_->Map(0, &range, &data))) {
                const auto* t = static_cast<const std::uint64_t*>(data) + i * 2;
                if (t[1] > t[0]) times_.push_back(float(double(t[1] - t[0]) * tick_ms_));
                const D3D12_RANGE none{0, 0};
                readback_->Unmap(0, &none);
            }
        }
        slot_begun_[i] = slot_done_[i] = false;
        slot_fence_[i] = 0;
    }
    if (times_.size() >= 64) {
        std::vector<float> sorted = times_;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        gpu_ms_ = sorted[sorted.size() / 2];
        times_.clear();
    }
}

}  // namespace fw
