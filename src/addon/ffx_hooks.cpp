#include "addon/ffx_hooks.hpp"
#include "addon/ngx_hooks.hpp"
#include "addon/streamline_hooks.hpp"
#include "common/inline_hook.hpp"
#include <d3d12.h>
#include <atomic>
#include <cstring>
#include <string>

namespace fw {
namespace {

// FidelityFX API structures (ffx_api.h, ffx_api_types.h, ffx_upscale.h of the AMD FidelityFX SDK; the
// layout is part of the API and stable across FSR 3.1 and FSR 4).
using ffxStructType_t = std::uint64_t;
using ffxReturnCode_t = std::uint32_t;
using ffxContext = void*;
struct ffxApiHeader { ffxStructType_t type; ffxApiHeader* pNext; };
struct FfxApiResourceDescription { std::uint32_t type, format, width, height, depth, mipCount, flags, usage; };
struct FfxApiResource { void* resource; FfxApiResourceDescription description; std::uint32_t state; };
struct FfxApiFloatCoords2D { float x, y; };
struct FfxApiDimensions2D { std::uint32_t width, height; };
struct ffxCreateContextDescUpscale {
    ffxApiHeader header;
    std::uint32_t flags;
    FfxApiDimensions2D maxRenderSize, maxUpscaleSize;
    void* fpMessage;
};
struct ffxDispatchDescUpscale {
    ffxApiHeader header;
    void* commandList;
    FfxApiResource color, depth, motionVectors, exposure, reactive, transparencyAndComposition, output;
    FfxApiFloatCoords2D jitterOffset, motionVectorScale;
    FfxApiDimensions2D renderSize, upscaleSize;
    bool enableSharpening;
    float sharpness, frameTimeDelta, preExposure;
    bool reset;
    float cameraNear, cameraFar, cameraFovAngleVertical, viewSpaceToMetersFactor;
    std::uint32_t flags;
};
static_assert(sizeof(FfxApiResource) == 48, "FfxApiResource layout");
static_assert(offsetof(ffxDispatchDescUpscale, output) == 312 && offsetof(ffxDispatchDescUpscale, flags) == 428, "dispatch layout");
constexpr ffxStructType_t kCreateUpscale = 0x00010000u, kDispatchUpscale = 0x00010001u;
constexpr std::uint32_t kDepthInverted = 1u << 3;
// FfxApiResourceState -> D3D12_RESOURCE_STATES.
D3D12_RESOURCE_STATES d3d12_state(std::uint32_t s) {
    switch (s) {
        case 1u << 1: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case 1u << 2: return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case 1u << 3: return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        case (1u << 3) | (1u << 2): return D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        case 1u << 4: return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case 1u << 5: return D3D12_RESOURCE_STATE_COPY_DEST;
        case (1u << 4) | (1u << 2): return D3D12_RESOURCE_STATE_GENERIC_READ;
        case 1u << 8: return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case 1u << 7: return D3D12_RESOURCE_STATE_PRESENT;
        default: return D3D12_RESOURCE_STATE_COMMON;
    }
}

using CreateFn = ffxReturnCode_t (*)(ffxContext*, ffxApiHeader*, const void*);
using DispatchFn = ffxReturnCode_t (*)(ffxContext*, const ffxApiHeader*);

Producer* g_producer = nullptr;
// One pair per DLL that exports the API (the FSR 4 loader forwards to the upscaler DLL, so both may be
// hooked; only the outermost call is looked at).
constexpr const wchar_t* kModules[] = {L"amd_fidelityfx_dx12.dll", L"amd_fidelityfx_loader_dx12.dll", L"amd_fidelityfx_upscaler_dx12.dll"};
InlineHook g_create_hooks[3], g_dispatch_hooks[3];
thread_local int t_depth = 0;

FsrStats* stats() { return g_producer && g_producer->shared() ? &g_producer->shared()->fsr : nullptr; }

// Upscale contexts and their creation flags (depth inverted, ...).
struct ContextEntry { std::atomic<ffxContext> context{nullptr}; std::atomic<std::uint32_t> flags{0}; };
ContextEntry g_contexts[16];
void remember(ffxContext c, std::uint32_t flags) {
    for (auto& e : g_contexts) {
        ffxContext expected = nullptr;
        if (e.context.load() == c || e.context.compare_exchange_strong(expected, c)) { e.flags = flags; return; }
    }
}
bool flags_of(ffxContext c, std::uint32_t& flags) {
    for (auto& e : g_contexts) if (e.context.load() == c) { flags = e.flags.load(); return true; }
    return false;
}

template <typename T>
const T* find_desc(const ffxApiHeader* h, ffxStructType_t type) {
    for (int guard = 0; h && guard < 16; h = h->pNext, ++guard)
        if (h->type == type) return reinterpret_cast<const T*>(h);
    return nullptr;
}

template <int I>
ffxReturnCode_t hk_create(ffxContext* context, ffxApiHeader* desc, const void* mem) {
    ++t_depth;
    const ffxReturnCode_t result = reinterpret_cast<CreateFn>(g_create_hooks[I].original())(context, desc, mem);
    --t_depth;
    if (t_depth == 0 && result == 0 && context && *context) {
        if (const auto* up = find_desc<ffxCreateContextDescUpscale>(desc, kCreateUpscale)) {
            remember(*context, up->flags);
            if (auto* s = stats()) { ++s->upscale_creates; s->create_flags = up->flags; }
        }
    }
    return result;
}

template <int I>
ffxReturnCode_t hk_dispatch(ffxContext* context, const ffxApiHeader* desc) {
    const bool outer = t_depth == 0;
    const ffxDispatchDescUpscale* up = outer ? find_desc<ffxDispatchDescUpscale>(desc, kDispatchUpscale) : nullptr;
    std::uint64_t published = 0;
    auto* list = up ? static_cast<ID3D12GraphicsCommandList*>(up->commandList) : nullptr;
    if (up) {
        if (auto* s = stats()) {
            ++s->upscale_dispatches;
            s->render_w = up->renderSize.width; s->render_h = up->renderSize.height;
            s->out_w = up->upscaleSize.width; s->out_h = up->upscaleSize.height;
            s->depth_format = up->depth.description.format; s->mv_format = up->motionVectors.description.format;
            s->depth_state = up->depth.state; s->output_state = up->output.state;
            s->jitter[0] = up->jitterOffset.x; s->jitter[1] = up->jitterOffset.y;
            s->mv_scale[0] = up->motionVectorScale.x; s->mv_scale[1] = up->motionVectorScale.y;
            s->near_plane = up->cameraNear; s->far_plane = up->cameraFar; s->fov = up->cameraFovAngleVertical;
            if (up->reset) ++s->resets;
            std::uint32_t flags = 0;
            const bool known = context && flags_of(*context, flags);
            if (known) s->create_flags = flags;
            auto* shared = g_producer->shared();
            // No Streamline camera for the last second (never had one, or the game stopped sending it when
            // FSR was selected) and DLSS not publishing: this frame's depth and motion vectors with a camera
            // the presenter estimates (FSR tells it the field of view). Switching back hands over again.
            if (shared && !streamline_camera_recent() && !dlss_publishing() && s->upscale_dispatches > 60 && list &&
                up->depth.resource && up->motionVectors.resource) {
                const std::uint64_t frame = next_estimated_frame();
                Camera cam{};
                cam.valid = 1;
                cam.estimated = 1;
                cam.reset = up->reset ? 1u : 0u;
                cam.depth_inverted = (!known || (flags & kDepthInverted)) ? 1u : 0u;
                cam.jitter[0] = up->jitterOffset.x; cam.jitter[1] = up->jitterOffset.y;
                cam.mvec_scale[0] = up->motionVectorScale.x != 0 ? up->motionVectorScale.x : 1.0f;  // motion vector * scale = render pixels
                cam.mvec_scale[1] = up->motionVectorScale.y != 0 ? up->motionVectorScale.y : 1.0f;
                cam.fov = up->cameraFovAngleVertical;
                cam.near_plane = up->cameraNear; cam.far_plane = up->cameraFar;
                g_producer->on_constants(frame, cam);
                const std::uint32_t w = up->renderSize.width, h = up->renderSize.height;
                g_producer->on_tag(frame, kDepth, static_cast<ID3D12Resource*>(up->depth.resource), d3d12_state(up->depth.state), 0, 0, w, h, list);
                g_producer->on_tag(frame, kMotion, static_cast<ID3D12Resource*>(up->motionVectors.resource),
                                   d3d12_state(up->motionVectors.state), 0, 0, w, h, list);
                ++s->frames_published;
                published = frame;
            }
        }
    }
    ++t_depth;
    const ffxReturnCode_t result = reinterpret_cast<DispatchFn>(g_dispatch_hooks[I].original())(context, desc);
    --t_depth;
    // The upscaled scene (before post-processing and HUD), copied after FSR recorded it on this list.
    if (published && result == 0 && up->output.resource && g_producer->shared()->settings.hud_from_scene) {
        g_producer->on_tag(published, kScene, static_cast<ID3D12Resource*>(up->output.resource), d3d12_state(up->output.state), 0, 0,
                           up->upscaleSize.width, up->upscaleSize.height, list);
        if (auto* s = stats()) ++s->outputs_copied;
    }
    return result;
}

}  // namespace

void install_ffx_hooks(Producer* producer) {
    g_producer = producer;
    void* const creates[3] = {reinterpret_cast<void*>(&hk_create<0>), reinterpret_cast<void*>(&hk_create<1>), reinterpret_cast<void*>(&hk_create<2>)};
    void* const dispatches[3] = {reinterpret_cast<void*>(&hk_dispatch<0>), reinterpret_cast<void*>(&hk_dispatch<1>),
                                 reinterpret_cast<void*>(&hk_dispatch<2>)};
    for (int i = 0; i < 3; ++i) {
        if (g_create_hooks[i].installed() && g_dispatch_hooks[i].installed()) continue;
        HMODULE module = GetModuleHandleW(kModules[i]);
        if (!module) continue;
        const char* names[2] = {"ffxCreateContext", "ffxDispatch"};
        InlineHook* hooks[2] = {&g_create_hooks[i], &g_dispatch_hooks[i]};
        void* detours[2] = {creates[i], dispatches[i]};
        for (int k = 0; k < 2; ++k) {
            if (hooks[k]->installed()) continue;
            void* target = reinterpret_cast<void*>(GetProcAddress(module, names[k]));
            if (!target) continue;
            if (hooks[k]->install(target, detours[k])) {
                if (auto* s = stats()) s->hooks |= 1u << (i * 2 + k);
            } else if (producer) {
                producer->set_message((std::string("FSR ") + names[k] + ": " + hooks[k]->error()).c_str());
            }
        }
    }
}

}  // namespace fw
