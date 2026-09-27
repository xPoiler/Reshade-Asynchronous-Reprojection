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

// DLSS Super Resolution, or DLSS Ray Reconstruction (which upscales too, from the same inputs).
bool is_dlss(std::uint32_t feature) { return feature == NVSDK_NGX_Feature_SuperSampling || feature == NVSDK_NGX_Feature_RayReconstruction; }

std::atomic<std::uint64_t> g_frame{0};

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
            if (is_dlss(static_cast<std::uint32_t>(feature)) && params) {
                s->dlss_feature = static_cast<std::uint32_t>(feature);
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
    std::uint64_t published = 0;  // frame whose DLSS output is copied once the evaluation is recorded
    ID3D12Resource* output = nullptr;
    std::uint32_t output_w = 0, output_h = 0;
    if (auto* s = stats()) {
        ++s->evaluate_calls;
        std::uint32_t feature = feature_of(handle);
        if (feature == 0xFFFFFFFFu && params) {
            // Created before our hooks (or CreateFeature could not be hooked): an evaluation that carries
            // both depth and motion vectors is an upscaler; treat it as DLSS from now on.
            ID3D12Resource *depth = nullptr, *motion = nullptr;
            if (params->Get(NVSDK_NGX_Parameter_Depth, &depth) == NVSDK_NGX_Result_Success && depth &&
                params->Get(NVSDK_NGX_Parameter_MotionVectors, &motion) == NVSDK_NGX_Result_Success && motion) {
                feature = NVSDK_NGX_Feature_SuperSampling;
                remember(handle, feature);
                if (!s->dlss_feature) s->dlss_feature = feature;
                ++s->identified_by_inputs;
            }
        }
        if (feature < 16) ++s->feature_calls[feature];
        else ++s->unknown_handle_calls;
        if (is_dlss(feature) && params) {
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
            // Games without a Streamline camera: publish this frame's depth and motion vectors with a camera
            // the presenter estimates from the motion vectors. (Streamline games publish through their own
            // hooks; once slSetConstants has been seen, this stays off.)
            auto* shared = g_producer->shared();
            ID3D12Resource *depth = nullptr, *motion = nullptr;
            // Only after 60 DLSS evaluations without any Streamline camera: a Streamline game sets its camera
            // before its first DLSS call, so it can never reach this (and frame numbers cannot mix).
            if (shared && shared->hooks.constants_calls == 0 && s->dlss_calls > 60 &&
                params->Get(NVSDK_NGX_Parameter_Depth, &depth) == NVSDK_NGX_Result_Success && depth &&
                params->Get(NVSDK_NGX_Parameter_MotionVectors, &motion) == NVSDK_NGX_Result_Success && motion) {
                const std::uint64_t frame = ++g_frame;
                Camera cam{};
                cam.valid = 1;
                cam.estimated = 1;
                cam.reset = reset ? 1u : 0u;
                // Unreal and most modern engines use reversed depth; the create flags say so when we saw them.
                cam.depth_inverted = (s->create_flags == 0 || (s->create_flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted)) ? 1u : 0u;
                cam.jitter[0] = s->jitter[0]; cam.jitter[1] = s->jitter[1];
                cam.mvec_scale[0] = s->mv_scale[0] != 0 ? s->mv_scale[0] : 1.0f;  // motion vector * scale = render pixels
                cam.mvec_scale[1] = s->mv_scale[1] != 0 ? s->mv_scale[1] : 1.0f;
                g_producer->on_constants(frame, cam);
                const std::uint32_t w = s->subrect_w ? s->subrect_w : s->depth_w, h = s->subrect_h ? s->subrect_h : s->depth_h;
                // DLSS inputs are in a shader-resource state when evaluated.
                const auto state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                g_producer->on_tag(frame, kDepth, depth, state, 0, 0, w, h, list);
                g_producer->on_tag(frame, kMotion, motion, state, 0, 0, w, h, list);
                ++s->frames_published;
                if (shared->settings.hud_from_scene && params->Get(NVSDK_NGX_Parameter_Output, &output) == NVSDK_NGX_Result_Success && output) {
                    published = frame; output_w = s->out_w; output_h = s->out_h;
                }
            }
        }
    }
    const NVSDK_NGX_Result result = reinterpret_cast<EvaluateFn>(g_evaluate_hook.original())(list, handle, params, callback);
    // The upscaled scene (before post-processing and HUD): DLSS writes its output as an unordered-access
    // resource, recorded on this list just now.
    if (published && output && result == NVSDK_NGX_Result_Success)
        g_producer->on_tag(published, kScene, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0, output_w, output_h, list);
    return result;
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
