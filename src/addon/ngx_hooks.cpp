#include "addon/ngx_hooks.hpp"
#include "common/inline_hook.hpp"
#include <d3d11.h>
#include <d3d12.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_params.h>
#include <atomic>
#include <string>

namespace fw {
namespace {

Producer* g_producer = nullptr;
InlineHook g_create_hook, g_evaluate_hook;

using CreateFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using EvaluateFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*,
                                                  PFN_NVSDK_NGX_ProgressCallback);

// Feature handles are opaque: remember which pointers belong to which feature.
struct HandleEntry { std::atomic<const void*> handle{nullptr}; std::atomic<std::uint32_t> feature{0}; };
HandleEntry g_handles[32];

void remember(const void* handle, std::uint32_t feature) {
    for (auto& e : g_handles) {
        const void* expected = nullptr;
        if (e.handle.load() == handle || e.handle.compare_exchange_strong(expected, handle)) { e.feature = feature; return; }
    }
}
std::uint32_t feature_of(const void* handle) {
    for (auto& e : g_handles) if (e.handle.load() == handle) return e.feature.load();
    return 0xFFFFFFFFu;
}

NgxStats* stats() { return g_producer && g_producer->shared() ? &g_producer->shared()->ngx : nullptr; }

void describe(const NVSDK_NGX_Parameter* p, const char* name, std::uint32_t& w, std::uint32_t& h, std::uint32_t& format) {
    ID3D12Resource* r = nullptr;
    if (p->Get(name, &r) != NVSDK_NGX_Result_Success || !r) { w = h = format = 0; return; }
    const auto d = r->GetDesc();
    w = static_cast<std::uint32_t>(d.Width); h = d.Height; format = static_cast<std::uint32_t>(d.Format);
}

NVSDK_NGX_Result NVSDK_CONV hk_create(ID3D12GraphicsCommandList* list, NVSDK_NGX_Feature feature, NVSDK_NGX_Parameter* params,
                                      NVSDK_NGX_Handle** out) {
    const auto result = reinterpret_cast<CreateFn>(g_create_hook.original())(list, feature, params, out);
    if (auto* s = stats()) {
        ++s->create_calls;
        if (NVSDK_NGX_SUCCEED(result) && out && *out) {
            remember(*out, static_cast<std::uint32_t>(feature));
            if (feature == NVSDK_NGX_Feature_SuperSampling && params) {
                unsigned int v = 0; int flags = 0;
                if (params->Get(NVSDK_NGX_Parameter_Width, &v) == NVSDK_NGX_Result_Success) s->render_w = v;
                if (params->Get(NVSDK_NGX_Parameter_Height, &v) == NVSDK_NGX_Result_Success) s->render_h = v;
                if (params->Get(NVSDK_NGX_Parameter_OutWidth, &v) == NVSDK_NGX_Result_Success) s->out_w = v;
                if (params->Get(NVSDK_NGX_Parameter_OutHeight, &v) == NVSDK_NGX_Result_Success) s->out_h = v;
                if (params->Get(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, &flags) == NVSDK_NGX_Result_Success)
                    s->create_flags = static_cast<std::uint32_t>(flags);
                ++s->dlss_creates;
            }
        }
    }
    return result;
}

NVSDK_NGX_Result NVSDK_CONV hk_evaluate(ID3D12GraphicsCommandList* list, const NVSDK_NGX_Handle* handle, const NVSDK_NGX_Parameter* params,
                                        PFN_NVSDK_NGX_ProgressCallback callback) {
    if (auto* s = stats()) {
        ++s->evaluate_calls;
        const std::uint32_t feature = feature_of(handle);
        if (feature < 16) ++s->feature_calls[feature];
        else ++s->unknown_handle_calls;
        if (feature == NVSDK_NGX_Feature_SuperSampling && params) {
            ++s->dlss_calls;
            describe(params, NVSDK_NGX_Parameter_Depth, s->depth_w, s->depth_h, s->depth_format);
            describe(params, NVSDK_NGX_Parameter_MotionVectors, s->mv_w, s->mv_h, s->mv_format);
            describe(params, NVSDK_NGX_Parameter_Color, s->color_w, s->color_h, s->color_format);
            describe(params, NVSDK_NGX_Parameter_Output, s->output_w, s->output_h, s->output_format);
            float f = 0; unsigned int v = 0; int reset = 0;
            if (params->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &f) == NVSDK_NGX_Result_Success) s->jitter[0] = f;
            if (params->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &f) == NVSDK_NGX_Result_Success) s->jitter[1] = f;
            if (params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &f) == NVSDK_NGX_Result_Success) s->mv_scale[0] = f;
            if (params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &f) == NVSDK_NGX_Result_Success) s->mv_scale[1] = f;
            if (params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, &v) == NVSDK_NGX_Result_Success) s->subrect_w = v;
            if (params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, &v) == NVSDK_NGX_Result_Success) s->subrect_h = v;
            if (params->Get(NVSDK_NGX_Parameter_Reset, &reset) == NVSDK_NGX_Result_Success && reset) ++s->resets;
        }
    }
    return reinterpret_cast<EvaluateFn>(g_evaluate_hook.original())(list, handle, params, callback);
}

}  // namespace

void install_ngx_hooks(Producer* producer) {
    g_producer = producer;
    if (g_create_hook.installed() && g_evaluate_hook.installed()) return;
    HMODULE ngx = GetModuleHandleW(L"_nvngx.dll");
    if (!ngx) return;
    struct Entry { InlineHook& hook; const char* name; void* detour; std::uint32_t bit; };
    Entry entries[] = {
        {g_create_hook, "NVSDK_NGX_D3D12_CreateFeature", reinterpret_cast<void*>(&hk_create), 1},
        {g_evaluate_hook, "NVSDK_NGX_D3D12_EvaluateFeature", reinterpret_cast<void*>(&hk_evaluate), 2},
    };
    for (auto& e : entries) {
        if (e.hook.installed()) continue;
        void* target = reinterpret_cast<void*>(GetProcAddress(ngx, e.name));
        if (!target) continue;
        if (e.hook.install(target, e.detour)) {
            if (auto* s = stats()) s->hooks |= e.bit;
        } else if (producer) {
            producer->set_message((std::string(e.name) + ": " + e.hook.error()).c_str());
        }
    }
}

}  // namespace fw
