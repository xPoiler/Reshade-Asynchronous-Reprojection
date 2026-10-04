#include "addon/ffx_hooks.hpp"
#include "addon/ngx_hooks.hpp"
#include "addon/streamline_hooks.hpp"
#include "common/inline_hook.hpp"
#include <reshade.hpp>
#include <d3d12.h>
#include <wrl/client.h>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <algorithm>
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
// Frame generation (ffx_framegeneration.h): the dispatch that generates the images between the previous frame
// and the one being presented, recorded on commandList.
struct FfxApiRect2D { std::int32_t left, top, width, height; };
struct ffxDispatchDescFrameGeneration {
    ffxApiHeader header;
    void* commandList;
    FfxApiResource presentColor;  // the frame being presented
    FfxApiResource outputs[4];    // the generated images, numGeneratedFrames of them
    std::uint32_t numGeneratedFrames;
    bool reset;
    std::uint32_t backbufferTransferFunction;
    float minMaxLuminance[2];
    FfxApiRect2D generationRect;
    std::uint64_t frameID;
};
static_assert(offsetof(ffxDispatchDescFrameGeneration, numGeneratedFrames) == 264 && offsetof(ffxDispatchDescFrameGeneration, frameID) == 304,
              "frame generation dispatch layout");
constexpr ffxStructType_t kDispatchFrameGeneration = 0x00020003u;
// Frame generation's setup (ffx_framegeneration.h): a game may hand over its picture without the HUD
// (HUDLessColor, may be empty); and the frame generation swapchain's UI layer (ffx_api_framegeneration_dx12.h,
// may be empty). Read for the log only, so far.
struct ffxConfigureDescFrameGeneration {
    ffxApiHeader header;
    void* swapChain;
    void* presentCallback;
    void* presentCallbackUserContext;
    void* frameGenerationCallback;
    void* frameGenerationCallbackUserContext;
    bool frameGenerationEnabled;
    bool allowAsyncWorkloads;
    FfxApiResource HUDLessColor;
    std::uint32_t flags;
};
static_assert(offsetof(ffxConfigureDescFrameGeneration, HUDLessColor) == 64 && offsetof(ffxConfigureDescFrameGeneration, flags) == 112,
              "frame generation configure layout");
struct ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12 {
    ffxApiHeader header;
    FfxApiResource uiResource;
    std::uint32_t flags;
};
static_assert(offsetof(ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12, flags) == 64, "UI resource registration layout");
constexpr ffxStructType_t kConfigureFrameGeneration = 0x00020002u, kConfigureRegisterUi = 0x00030002u;
constexpr std::uint32_t kDepthInverted = 1u << 3;  // the same bit in both APIs' upscaler creation flags

// FidelityFX SDK 1.0 (FSR 3.0; ffx_types.h and ffx_fsr3upscaler.h of tags fsr3-v3.0.3 / fsr3-v3.0.4),
// called through ffx_fsr3upscaler_x64.dll. Its resource states use the same bits as the FFX API.
struct FfxSdkResource { void* resource; FfxApiResourceDescription description; std::uint32_t state; wchar_t name[64]; };
struct FfxSdkUpscalerContextDescription { std::uint32_t flags; FfxApiDimensions2D maxRenderSize, displaySize; };
struct FfxSdkUpscalerDispatchDescription {
    void* commandList;
    FfxSdkResource color, depth, motionVectors, exposure, reactive, transparencyAndComposition, dilatedDepth, dilatedMotionVectors,
        reconstructedPrevNearestDepth, output;
    FfxApiFloatCoords2D jitterOffset, motionVectorScale;
    FfxApiDimensions2D renderSize;
    bool enableSharpening;
    float sharpness, frameTimeDelta, preExposure;
    bool reset;
    float cameraNear, cameraFar, cameraFovAngleVertical, viewSpaceToMetersFactor;
};
static_assert(sizeof(FfxSdkResource) == 176, "FfxResource layout (SDK 1.0)");
static_assert(offsetof(FfxSdkUpscalerDispatchDescription, output) == 1592 &&
              offsetof(FfxSdkUpscalerDispatchDescription, cameraFovAngleVertical) == 1820, "dispatch layout (SDK 1.0)");

// FSR 2 (FidelityFX-FSR2 v2.0.1 - v2.2.1, ffx_fsr2.h / ffx_types.h), called through ffx_fsr2_api_x64.dll. The
// fields read here are the same in every release (2.2 only appends). FfxResource carries a name in 2.1 and
// later; 2.0 release builds leave it out (the colour texture's width tells the two apart).
struct Fsr2ResourceDescription { std::uint32_t type, format, width, height, depth, mipCount, flags; };
struct Fsr2Resource21 { void* resource; wchar_t name[64]; Fsr2ResourceDescription description; std::uint32_t state; bool isDepth; std::uint64_t descriptorData; };
struct Fsr2Resource20 { void* resource; Fsr2ResourceDescription description; std::uint32_t state; bool isDepth; std::uint64_t descriptorData; };
template <typename R> struct Fsr2DispatchDescription {
    void* commandList;
    R color, depth, motionVectors, exposure, reactive, transparencyAndComposition, output;
    FfxApiFloatCoords2D jitterOffset, motionVectorScale;
    FfxApiDimensions2D renderSize;
    bool enableSharpening;
    float sharpness, frameTimeDelta, preExposure;
    bool reset;
    float cameraNear, cameraFar, cameraFovAngleVertical;
};
struct Fsr2ContextDescription { std::uint32_t flags; FfxApiDimensions2D maxRenderSize, displaySize; };
static_assert(sizeof(Fsr2Resource21) == 184 && sizeof(Fsr2Resource20) == 56, "FfxResource layouts (FSR 2)");
static_assert(offsetof(Fsr2DispatchDescription<Fsr2Resource21>, cameraFovAngleVertical) == 1348 &&
              offsetof(Fsr2DispatchDescription<Fsr2Resource20>, cameraFovAngleVertical) == 452, "dispatch layouts (FSR 2)");
// FSR 2's own resource states, as its DX12 backend maps them (ffxGetDX12StateFromResourceState).
D3D12_RESOURCE_STATES fsr2_state(std::uint32_t s) {
    switch (s) {
        case (1u << 2) | (1u << 1): return D3D12_RESOURCE_STATE_GENERIC_READ;
        case 1u << 0: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case 1u << 1: return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case 1u << 2: return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case 1u << 3: return D3D12_RESOURCE_STATE_COPY_DEST;
        default: return D3D12_RESOURCE_STATE_COMMON;
    }
}

// FfxApiResourceState / FfxResourceStates (FSR 3.x) -> D3D12_RESOURCE_STATES.
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
using SdkCreateFn = std::int32_t (*)(void*, const FfxSdkUpscalerContextDescription*);
using SdkDispatchFn = std::int32_t (*)(void*, const FfxSdkUpscalerDispatchDescription*);
using Fsr2CreateFn = std::int32_t (*)(void*, const Fsr2ContextDescription*);
using Fsr2DispatchFn = std::int32_t (*)(void*, const void*);
using ConfigureFn = ffxReturnCode_t (*)(ffxContext*, const ffxApiHeader*);
using PassFn = std::int32_t (*)(void*, void*);  // (frame generation entry points, looked at but not read)

Producer* g_producer = nullptr;
// One pair per DLL that exports the API (the FSR 4 loader forwards to the upscaler DLL, so both may be
// hooked; only the outermost call is looked at).
constexpr const wchar_t* kModules[] = {L"amd_fidelityfx_dx12.dll", L"amd_fidelityfx_loader_dx12.dll", L"amd_fidelityfx_upscaler_dx12.dll"};
InlineHook g_create_hooks[3], g_dispatch_hooks[3], g_configure_hooks[3];
// FSR 3.0 (FidelityFX SDK 1.0): its own upscaler DLL and entry points.
constexpr const wchar_t* kSdkModule = L"ffx_fsr3upscaler_x64.dll";
InlineHook g_sdk_create_hook, g_sdk_dispatch_hook;
// FSR 2: its API DLL.
constexpr const wchar_t* kFsr2Module = L"ffx_fsr2_api_x64.dll";
InlineHook g_fsr2_create_hook, g_fsr2_dispatch_hook;
// Frame generation (diagnostics for now): FSR 3.0's (FidelityFX SDK 1.0) entry points, each call a
// kEvFrameGen event with its source: 3 ffxFsr3DispatchFrameGeneration, 4 ffxFrameInterpolationDispatch,
// 5 ffxFsr3ConfigureFrameGeneration, 6 ffxFsr3ContextDispatchUpscale; and FSR 3.1's frame generation DLL
// called directly (not through the loader, e.g. from its own presenting thread): 8 ffxDispatch, 9 ffxConfigure.
// (1: an FFX API dispatch and 2: an FFX API configure of a frame generation or swapchain description; FFX API
// calls carry the description's type above the low byte.)
struct FgEntry { const wchar_t* module; const char* name; std::uint64_t source; };
constexpr FgEntry kFgEntries[] = {{L"ffx_fsr3_x64.dll", "ffxFsr3DispatchFrameGeneration", 3},
                                  {L"ffx_frameinterpolation_x64.dll", "ffxFrameInterpolationDispatch", 4},
                                  {L"ffx_fsr3_x64.dll", "ffxFsr3ConfigureFrameGeneration", 5},
                                  {L"ffx_fsr3_x64.dll", "ffxFsr3ContextDispatchUpscale", 6},
                                  {L"amd_fidelityfx_framegeneration_dx12.dll", "ffxDispatch", 8},
                                  {L"amd_fidelityfx_framegeneration_dx12.dll", "ffxConfigure", 9}};
constexpr int kFgCount = int(sizeof(kFgEntries) / sizeof(kFgEntries[0]));
InlineHook g_fg_hooks[kFgCount];
bool frame_generation_type(ffxStructType_t type) { return (type >> 16) == 2 || (type >> 16) == 3; }
void note_frame_generation(std::uint64_t extra) {
    if (g_producer && g_producer->shared()) push_event(g_producer->shared(), kEvFrameGen, 0, extra);
}
std::atomic<int> g_fsr2_layout{0};  // 0 not known yet, 21: 2.1 and later, 20: 2.0 release, -1: neither (not used)
thread_local int t_depth = 0;

FsrStats* stats() { return g_producer && g_producer->shared() ? &g_producer->shared()->fsr : nullptr; }

// Upscale contexts and their creation flags (depth inverted, ...).
struct ContextEntry { std::atomic<void*> context{nullptr}; std::atomic<std::uint32_t> flags{0}; };
ContextEntry g_contexts[16];
void remember(void* c, std::uint32_t flags) {
    for (auto& e : g_contexts) {
        void* expected = nullptr;
        if (e.context.load() == c || e.context.compare_exchange_strong(expected, c)) { e.flags = flags; return; }
    }
}
bool flags_of(void* c, std::uint32_t& flags) {
    for (auto& e : g_contexts) if (e.context.load() == c) { flags = e.flags.load(); return true; }
    return false;
}

template <typename T>
const T* find_desc(const ffxApiHeader* h, ffxStructType_t type) {
    for (int guard = 0; h && guard < 16; h = h->pNext, ++guard)
        if (h->type == type) return reinterpret_cast<const T*>(h);
    return nullptr;
}

// One upscale call, whichever API it came through.
struct Upscale {
    void* context;
    ID3D12GraphicsCommandList* list;
    void* depth; std::uint32_t depth_state, depth_format;
    void* motion; std::uint32_t motion_state, motion_format;
    void* output; std::uint32_t output_state;
    D3D12_RESOURCE_STATES depth_d3d, motion_d3d, output_d3d;  // the states above, in D3D12 terms
    FfxApiFloatCoords2D jitter, mv_scale;
    FfxApiDimensions2D render, upscale;
    bool reset;
    float near_plane, far_plane, fov;
};

// Before the upscaler records its work: statistics, and this frame's depth and motion vectors when nothing
// else provides them (DLSS not publishing, Streamline sending no depth for the last second):
//  - with the game's own Streamline camera when it still sends one (some games tag depth and motion
//    vectors only for DLSS): they join the frame being rendered;
//  - otherwise with a camera the presenter estimates (FSR tells it the field of view).
// Switching back hands over again. Returns the frame published, or 0.
std::uint64_t before_upscale(const Upscale& u) {
    auto* s = stats();
    if (!s) return 0;
    ++s->upscale_dispatches;
    g_producer->note_game_depth();
    s->render_w = u.render.width; s->render_h = u.render.height;
    s->out_w = u.upscale.width; s->out_h = u.upscale.height;
    s->depth_format = u.depth_format; s->mv_format = u.motion_format;
    s->depth_state = u.depth_state; s->output_state = u.output_state;
    s->jitter[0] = u.jitter.x; s->jitter[1] = u.jitter.y;
    s->mv_scale[0] = u.mv_scale.x; s->mv_scale[1] = u.mv_scale.y;
    s->near_plane = u.near_plane; s->far_plane = u.far_plane; s->fov = u.fov;
    if (u.reset) ++s->resets;
    std::uint32_t flags = 0;
    const bool known = u.context && flags_of(u.context, flags);
    if (known) s->create_flags = flags;
    auto* shared = g_producer->shared();
    // (a Streamline camera found unusable counts as none: the camera is estimated)
    const bool own_camera = !g_producer->game_camera_unusable();
    // Depth and motion vectors from the game's Streamline tags (some games tag them with FSR too): only the
    // upscaler's output is taken here, for the frame being rendered (for the HUD found from it, as with DLSS).
    if (shared && !dlss_publishing() && own_camera && streamline_depth_recent() && streamline_camera_recent() && u.list && u.output &&
        shared->settings.hud_from_scene && s->upscale_dispatches > 60)
        return g_producer->rendering_frame() ? g_producer->rendering_frame() : streamline_current_frame();
    if (!shared || dlss_publishing() || (own_camera && streamline_depth_recent()) || s->upscale_dispatches <= 60 || !u.list || !u.depth ||
        !u.motion)
        return 0;
    const std::uint32_t w = u.render.width, h = u.render.height;
    if (own_camera && streamline_camera_recent()) {
        // The frame whose render work is being submitted (render-submit marker), else the newest camera's.
        const std::uint64_t frame = g_producer->rendering_frame() ? g_producer->rendering_frame() : streamline_current_frame();
        if (!frame) return 0;
        g_producer->on_tag(frame, kDepth, static_cast<ID3D12Resource*>(u.depth), u.depth_d3d, 0, 0, w, h, u.list);
        g_producer->on_tag(frame, kMotion, static_cast<ID3D12Resource*>(u.motion), u.motion_d3d, 0, 0, w, h, u.list);
        ++s->frames_published;
        return frame;
    }
    const std::uint64_t frame = next_estimated_frame();
    Camera cam{};
    cam.valid = 1;
    cam.estimated = 1;
    cam.reset = u.reset ? 1u : 0u;
    cam.depth_inverted = (!known || (flags & kDepthInverted)) ? 1u : 0u;
    cam.jitter[0] = u.jitter.x; cam.jitter[1] = u.jitter.y;
    cam.mvec_scale[0] = u.mv_scale.x != 0 ? u.mv_scale.x : 1.0f;  // motion vector * scale = render pixels
    cam.mvec_scale[1] = u.mv_scale.y != 0 ? u.mv_scale.y : 1.0f;
    cam.fov = u.fov;
    cam.near_plane = u.near_plane; cam.far_plane = u.far_plane;
    g_producer->on_constants(frame, cam);
    g_producer->on_tag(frame, kDepth, static_cast<ID3D12Resource*>(u.depth), u.depth_d3d, 0, 0, w, h, u.list);
    g_producer->on_tag(frame, kMotion, static_cast<ID3D12Resource*>(u.motion), u.motion_d3d, 0, 0, w, h, u.list);
    ++s->frames_published;
    return frame;
}

// After it recorded its work: the upscaled scene (before post-processing and HUD).
void after_upscale(const Upscale& u, std::uint64_t published) {
    if (!published || !u.output || !g_producer->shared()->settings.hud_from_scene) return;
    g_producer->on_tag(published, kScene, static_cast<ID3D12Resource*>(u.output), u.output_d3d, 0, 0, u.upscale.width,
                       u.upscale.height, u.list);
    if (auto* s = stats()) ++s->outputs_copied;
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

// The HUD-less picture the game registered with frame generation's setup (the latest one), kept alive here: it is
// copied where the generated images are (hk_dispatch), possibly on another thread than the setup's.
std::mutex g_hudless_mutex;
Microsoft::WRL::ComPtr<ID3D12Resource> g_hudless;
std::uint32_t g_hudless_state = 0;

// What the game hands to frame generation's setup: its HUD-less picture (kept for the capture), its UI layer
// (both logged when they change).
void log_frame_generation_setup(const ffxApiHeader* desc) {
    static std::atomic<std::uint64_t> said_hudless{~0ull}, said_ui{~0ull};
    auto key = [](const FfxApiResource& r) {
        return r.resource ? (std::uint64_t(r.description.width) << 40) ^ (std::uint64_t(r.description.height) << 20) ^ r.description.format ^ 1ull : 0ull;
    };
    char text[220];
    if (const auto* fg = find_desc<ffxConfigureDescFrameGeneration>(desc, kConfigureFrameGeneration)) {
        {
            std::lock_guard lock(g_hudless_mutex);
            auto* r = static_cast<ID3D12Resource*>(fg->HUDLessColor.resource);
            if (g_hudless.Get() != r) g_hudless = r;
            g_hudless_state = fg->HUDLessColor.state;
        }
        const std::uint64_t k = key(fg->HUDLessColor);
        if (said_hudless.exchange(k) != k) {
            if (fg->HUDLessColor.resource)
                std::snprintf(text, sizeof(text), "XPAR: FSR frame generation setup: the game hands over its HUD-less picture (%ux%u, format %u, state %u)",
                              fg->HUDLessColor.description.width, fg->HUDLessColor.description.height, fg->HUDLessColor.description.format, fg->HUDLessColor.state);
            else
                std::snprintf(text, sizeof(text), "XPAR: FSR frame generation setup: no HUD-less picture (frame generation %s)", fg->frameGenerationEnabled ? "on" : "off");
            reshade::log::message(reshade::log::level::info, text);
        }
    }
    if (const auto* ui = find_desc<ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceDX12>(desc, kConfigureRegisterUi)) {
        const std::uint64_t k = key(ui->uiResource) ^ (std::uint64_t(ui->flags) << 60);
        if (said_ui.exchange(k) != k) {
            if (ui->uiResource.resource)
                std::snprintf(text, sizeof(text), "XPAR: FSR frame generation setup: the game registers a UI layer (%ux%u, format %u, flags %u)",
                              ui->uiResource.description.width, ui->uiResource.description.height, ui->uiResource.description.format, ui->flags);
            else
                std::snprintf(text, sizeof(text), "XPAR: FSR frame generation setup: UI layer registered empty");
            reshade::log::message(reshade::log::level::info, text);
        }
    }
}

template <int I>
ffxReturnCode_t hk_configure(ffxContext* context, const ffxApiHeader* desc) {
    if (t_depth == 0 && desc && frame_generation_type(desc->type)) note_frame_generation(2 | (desc->type << 8));
    if (t_depth == 0 && desc) log_frame_generation_setup(desc);
    ++t_depth;
    const ffxReturnCode_t result = reinterpret_cast<ConfigureFn>(g_configure_hooks[I].original())(context, desc);
    --t_depth;
    return result;
}

template <int I>
std::int32_t hk_fg(void* a, void* b) {
    if (kFgEntries[I].source < 8) note_frame_generation(kFgEntries[I].source);
    // (FSR 3.0's frame generation is not taken, but counts: XPAR steps aside while it is on)
    if (kFgEntries[I].source == 3 && g_producer && g_producer->shared()) InterlockedIncrement(&g_producer->shared()->generation_calls);
    else if (t_depth == 0 && b) note_frame_generation(kFgEntries[I].source | (static_cast<const ffxApiHeader*>(b)->type << 8));
    if (kFgEntries[I].source == 9 && t_depth == 0 && b) log_frame_generation_setup(static_cast<const ffxApiHeader*>(b));
    ++t_depth;
    const std::int32_t result = reinterpret_cast<PassFn>(g_fg_hooks[I].original())(a, b);
    --t_depth;
    return result;
}

template <int I>
ffxReturnCode_t hk_dispatch(ffxContext* context, const ffxApiHeader* desc) {
    if (t_depth == 0 && desc && frame_generation_type(desc->type)) note_frame_generation(1 | (desc->type << 8));
    const ffxDispatchDescUpscale* up = t_depth == 0 ? find_desc<ffxDispatchDescUpscale>(desc, kDispatchUpscale) : nullptr;
    Upscale u{};
    std::uint64_t published = 0;
    if (up) {
        u = {context ? *context : nullptr, static_cast<ID3D12GraphicsCommandList*>(up->commandList),
             up->depth.resource, up->depth.state, up->depth.description.format,
             up->motionVectors.resource, up->motionVectors.state, up->motionVectors.description.format,
             up->output.resource, up->output.state,
             d3d12_state(up->depth.state), d3d12_state(up->motionVectors.state), d3d12_state(up->output.state),
             up->jitterOffset, up->motionVectorScale, up->renderSize, up->upscaleSize, up->reset,
             up->cameraNear, up->cameraFar, up->cameraFovAngleVertical};
        published = before_upscale(u);
    }
    ++t_depth;
    const ffxReturnCode_t result = reinterpret_cast<DispatchFn>(g_dispatch_hooks[I].original())(context, desc);
    --t_depth;
    if (up && result == 0) after_upscale(u, published);
    // Frame generation (FSR 3.1): its images, taken right after it recorded them (with the frame itself).
    if (t_depth == 0 && desc && desc->type == kDispatchFrameGeneration && g_producer && g_producer->shared()) {
        if (result == 0) InterlockedIncrement(&g_producer->shared()->generation_calls);
        // (diagnostics: why its images are not taken)
        const auto* fg = reinterpret_cast<const ffxDispatchDescFrameGeneration*>(desc);
        const GenSkip reason = result != 0 ? kGenSkipFailed : !g_producer->generation_wanted() ? kGenSkipNotWanted
                               : !fg->commandList || !fg->numGeneratedFrames || !fg->outputs[0].resource ? kGenSkipNoOutput : GenSkip(0);
        if (reason) push_event(g_producer->shared(), kEvFrameGen, result, 0x300 | (std::uint64_t(reason) << 16));
    }
    if (t_depth == 0 && result == 0 && desc && desc->type == kDispatchFrameGeneration && g_producer && g_producer->generation_wanted()) {
        const auto* fg = reinterpret_cast<const ffxDispatchDescFrameGeneration*>(desc);
        auto* list = static_cast<ID3D12GraphicsCommandList*>(fg->commandList);
        const std::uint32_t n = std::min<std::uint32_t>(fg->numGeneratedFrames, 4);
        // (the game's HUD-less picture from the setup, read by frame generation right here)
        Microsoft::WRL::ComPtr<ID3D12Resource> hudless;
        std::uint32_t hudless_state = 0;
        {
            std::lock_guard lock(g_hudless_mutex);
            hudless = g_hudless;
            hudless_state = g_hudless_state;
        }
        for (std::uint32_t i = 0; i < n && list; ++i)
            if (fg->outputs[i].resource)
                g_producer->on_generated(0, i + 1, n, static_cast<ID3D12Resource*>(fg->outputs[i].resource), d3d12_state(fg->outputs[i].state),
                                         static_cast<ID3D12Resource*>(fg->presentColor.resource), d3d12_state(fg->presentColor.state), list,
                                         hudless.Get(), d3d12_state(hudless_state));
    }
    return result;
}

std::int32_t hk_sdk_create(void* context, const FfxSdkUpscalerContextDescription* desc) {
    ++t_depth;
    const std::int32_t result = reinterpret_cast<SdkCreateFn>(g_sdk_create_hook.original())(context, desc);
    --t_depth;
    if (t_depth == 0 && result == 0 && context && desc) {
        remember(context, desc->flags);
        if (auto* s = stats()) { ++s->upscale_creates; s->create_flags = desc->flags; }
    }
    return result;
}

std::int32_t hk_sdk_dispatch(void* context, const FfxSdkUpscalerDispatchDescription* d) {
    Upscale u{};
    std::uint64_t published = 0;
    const bool outer = t_depth == 0 && d;
    if (outer) {
        // No upscale size in SDK 1.0: the output's own size.
        u = {context, static_cast<ID3D12GraphicsCommandList*>(d->commandList),
             d->depth.resource, d->depth.state, d->depth.description.format,
             d->motionVectors.resource, d->motionVectors.state, d->motionVectors.description.format,
             d->output.resource, d->output.state,
             d3d12_state(d->depth.state), d3d12_state(d->motionVectors.state), d3d12_state(d->output.state),
             d->jitterOffset, d->motionVectorScale, d->renderSize, {d->output.description.width, d->output.description.height}, d->reset,
             d->cameraNear, d->cameraFar, d->cameraFovAngleVertical};
        published = before_upscale(u);
    }
    ++t_depth;
    const std::int32_t result = reinterpret_cast<SdkDispatchFn>(g_sdk_dispatch_hook.original())(context, d);
    --t_depth;
    if (outer && result == 0) after_upscale(u, published);
    return result;
}

std::int32_t hk_fsr2_create(void* context, const Fsr2ContextDescription* desc) {
    ++t_depth;
    const std::int32_t result = reinterpret_cast<Fsr2CreateFn>(g_fsr2_create_hook.original())(context, desc);
    --t_depth;
    if (t_depth == 0 && result == 0 && context && desc) {
        remember(context, desc->flags);
        if (auto* s = stats()) { ++s->upscale_creates; s->create_flags = desc->flags; }
    }
    return result;
}

// Which FfxResource layout the game's FSR 2 uses: the colour texture's real width must be where the
// layout puts FfxResource::description.width (the resource pointer comes first in both).
int fsr2_layout(const void* desc) {
    const int known = g_fsr2_layout.load();
    if (known != 0) return known;
    const auto* raw = static_cast<const std::uint8_t*>(desc);
    auto* color = *reinterpret_cast<ID3D12Resource* const*>(raw + offsetof(Fsr2DispatchDescription<Fsr2Resource21>, color));
    if (!color) return 0;
    const std::uint64_t width = color->GetDesc().Width;
    auto width_at = [&](std::size_t offset) { std::uint32_t v; std::memcpy(&v, raw + offset, 4); return v; };
    const std::size_t c21 = offsetof(Fsr2DispatchDescription<Fsr2Resource21>, color) + offsetof(Fsr2Resource21, description) +
                            offsetof(Fsr2ResourceDescription, width);
    const std::size_t c20 = offsetof(Fsr2DispatchDescription<Fsr2Resource20>, color) + offsetof(Fsr2Resource20, description) +
                            offsetof(Fsr2ResourceDescription, width);
    const int layout = width_at(c21) == width ? 21 : width_at(c20) == width ? 20 : -1;
    g_fsr2_layout = layout;
    return layout;
}

template <typename R>
Upscale fsr2_upscale(void* context, const Fsr2DispatchDescription<R>* d) {
    return {context, static_cast<ID3D12GraphicsCommandList*>(d->commandList),
            d->depth.resource, d->depth.state, d->depth.description.format,
            d->motionVectors.resource, d->motionVectors.state, d->motionVectors.description.format,
            d->output.resource, d->output.state,
            fsr2_state(d->depth.state), fsr2_state(d->motionVectors.state), fsr2_state(d->output.state),
            d->jitterOffset, d->motionVectorScale, d->renderSize, {d->output.description.width, d->output.description.height}, d->reset,
            d->cameraNear, d->cameraFar, d->cameraFovAngleVertical};
}

std::int32_t hk_fsr2_dispatch(void* context, const void* desc) {
    Upscale u{};
    std::uint64_t published = 0;
    const int layout = t_depth == 0 && desc ? fsr2_layout(desc) : 0;
    const bool known = layout == 21 || layout == 20;
    if (layout == 21) u = fsr2_upscale(context, static_cast<const Fsr2DispatchDescription<Fsr2Resource21>*>(desc));
    else if (layout == 20) u = fsr2_upscale(context, static_cast<const Fsr2DispatchDescription<Fsr2Resource20>*>(desc));
    if (known) published = before_upscale(u);
    ++t_depth;
    const std::int32_t result = reinterpret_cast<Fsr2DispatchFn>(g_fsr2_dispatch_hook.original())(context, desc);
    --t_depth;
    if (known && result == 0) after_upscale(u, published);
    return result;
}

void install(HMODULE module, const char* name, InlineHook& hook, void* detour, std::uint32_t bit, Producer* producer) {
    if (hook.installed()) return;
    void* target = reinterpret_cast<void*>(GetProcAddress(module, name));
    if (!target) return;
    if (hook.install(target, detour)) {
        if (auto* s = stats()) s->hooks |= bit;
    } else if (producer) {
        producer->set_message((std::string("FSR ") + name + ": " + hook.error()).c_str());
    }
}

}  // namespace

void install_ffx_hooks(Producer* producer) {
    g_producer = producer;
    void* const creates[3] = {reinterpret_cast<void*>(&hk_create<0>), reinterpret_cast<void*>(&hk_create<1>), reinterpret_cast<void*>(&hk_create<2>)};
    void* const dispatches[3] = {reinterpret_cast<void*>(&hk_dispatch<0>), reinterpret_cast<void*>(&hk_dispatch<1>),
                                 reinterpret_cast<void*>(&hk_dispatch<2>)};
    void* const configures[3] = {reinterpret_cast<void*>(&hk_configure<0>), reinterpret_cast<void*>(&hk_configure<1>),
                                 reinterpret_cast<void*>(&hk_configure<2>)};
    for (int i = 0; i < 3; ++i) {
        if (g_create_hooks[i].installed() && g_dispatch_hooks[i].installed() && g_configure_hooks[i].installed()) continue;
        HMODULE module = GetModuleHandleW(kModules[i]);
        if (!module) continue;
        install(module, "ffxCreateContext", g_create_hooks[i], creates[i], 1u << (i * 2), producer);
        install(module, "ffxDispatch", g_dispatch_hooks[i], dispatches[i], 1u << (i * 2 + 1), producer);
        install(module, "ffxConfigure", g_configure_hooks[i], configures[i], 1u << (10 + i), producer);
    }
    void* const fg[kFgCount] = {reinterpret_cast<void*>(&hk_fg<0>), reinterpret_cast<void*>(&hk_fg<1>), reinterpret_cast<void*>(&hk_fg<2>),
                                reinterpret_cast<void*>(&hk_fg<3>), reinterpret_cast<void*>(&hk_fg<4>), reinterpret_cast<void*>(&hk_fg<5>)};
    for (int i = 0; i < kFgCount; ++i)
        if (HMODULE module = g_fg_hooks[i].installed() ? nullptr : GetModuleHandleW(kFgEntries[i].module))
            install(module, kFgEntries[i].name, g_fg_hooks[i], fg[i], 1u << (13 + i), producer);
    if (!g_sdk_create_hook.installed() || !g_sdk_dispatch_hook.installed()) {
        if (HMODULE module = GetModuleHandleW(kSdkModule)) {
            install(module, "ffxFsr3UpscalerContextCreate", g_sdk_create_hook, reinterpret_cast<void*>(&hk_sdk_create), 1u << 6, producer);
            install(module, "ffxFsr3UpscalerContextDispatch", g_sdk_dispatch_hook, reinterpret_cast<void*>(&hk_sdk_dispatch), 1u << 7, producer);
        }
    }
    if (!g_fsr2_create_hook.installed() || !g_fsr2_dispatch_hook.installed()) {
        if (HMODULE module = GetModuleHandleW(kFsr2Module)) {
            install(module, "ffxFsr2ContextCreate", g_fsr2_create_hook, reinterpret_cast<void*>(&hk_fsr2_create), 1u << 8, producer);
            install(module, "ffxFsr2ContextDispatch", g_fsr2_dispatch_hook, reinterpret_cast<void*>(&hk_fsr2_dispatch), 1u << 9, producer);
        }
    }
}

}  // namespace fw
