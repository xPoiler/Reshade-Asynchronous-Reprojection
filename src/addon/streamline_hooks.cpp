#include "addon/streamline_hooks.hpp"
#include "addon/sl_read.hpp"
#include "addon/ui_probe.hpp"
#include "common/inline_hook.hpp"
#include <array>
#include <atomic>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

// Streamline 2.x ABI notes (sl.h / sl_consts.h), all structs derive from BaseStructure
// { BaseStructure* next; StructType type (16 bytes); size_t version; } = 32 bytes:
//   Result slSetConstants(const Constants&, const FrameToken&, const ViewportHandle&)
//   Result slSetTag(const ViewportHandle&, const ResourceTag*, uint32_t count, CommandBuffer*)
//   Result slSetTagForFrame(const FrameToken&, const ViewportHandle&, const ResourceTag*, uint32_t, CommandBuffer*)
//   Result slGetFeatureFunction(Feature, const char*, void*&)
//   Result slPCLSetMarker(PCLMarker, const FrameToken&)       (kFeaturePCL = 4)
// FrameToken is an interface whose first virtual is `operator uint32_t() const`.
// The base size is detected at runtime (32 normally, 0 in stripped builds) and reported in the UI.

namespace fw {
namespace {

using SetConstantsFn = int (*)(const void*, const void*, const void*);
using SetTagFn = int (*)(const void*, const void*, std::uint32_t, void*);
using SetTagForFrameFn = int (*)(const void*, const void*, const void*, std::uint32_t, void*);
using GetFeatureFunctionFn = int (*)(std::uint32_t, const char*, void*&);
using MarkerFn = int (*)(std::uint32_t, const void*);
using EvaluateFeatureFn = int (*)(std::uint32_t, const void*, const void**, std::uint32_t, void*);

Producer* g_producer = nullptr;
InlineHook g_constants_hook, g_tag_hook, g_tag_frame_hook, g_marker_hook, g_evaluate_hook;
// The upscaler's output tagged for a frame: DLSS writes it during slEvaluateFeature, so it is copied
// right after that call, not when tagged.
struct PendingOutput { std::uint64_t frame = 0; ID3D12Resource* resource = nullptr; std::uint32_t state = 0, ext[4] = {}; };
std::mutex g_output_mutex;
PendingOutput g_output;
std::atomic<std::uint64_t> g_current_frame{0};
std::atomic<std::int64_t> g_last_constants_qpc{0}, g_last_depth_qpc{0};
std::atomic<int> g_constants_base{-1}, g_tag_base{-1};
std::atomic<std::uint32_t> g_pcl_attempts{0};
// Constants without a camera (matrices and vectors at sl::INVALID_FLOAT): the frame's camera is estimated by the presenter from the
// motion vectors. What the constants still say waits here for the frame's tags: the vector scale is turned into pixels of the frame's
// area, known from the motion vectors' tag.
std::mutex g_estimate_mutex;
struct PendingEstimate { std::uint64_t frame = 0; Camera cam{}; bool set = false; } g_estimate;
std::atomic<std::int64_t> g_last_no_camera_qpc{0};

enum Marker : std::uint32_t { kSimulationStart = 0, kSimulationEnd = 1, kRenderSubmitStart = 2, kRenderSubmitEnd = 3,
                              kPresentStart = 4, kPresentEnd = 5 };
enum BufferType : std::uint32_t { kSlDepth = 0, kSlMotionVectors = 1, kSlHudless = 2, kSlScalingOutputColor = 4, kSlUiColorAlpha = 23 };

HookStats* stats() { return g_producer && g_producer->shared() ? &g_producer->shared()->hooks : nullptr; }

bool read_frame_token(const void* token, std::uint32_t* out) {
    __try {
        using Fn = std::uint32_t (*)(const void*);
        const Fn fn = (*reinterpret_cast<Fn* const*>(token))[0];
        *out = fn(token);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool safe_read(const void* p, void* out, std::size_t n) {
    __try { std::memcpy(out, p, n); return true; } __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

// sl::Constants at base offset b (see sl::read_constants).
sl::Constants read_constants(const std::uint8_t* c, int b, Camera& cam) {
    std::uint8_t raw[sl::kConstantsBytes];
    if (!safe_read(c + b, raw, sizeof(raw))) return sl::Constants::NotHere;
    return sl::read_constants(raw, cam);
}

void try_hook_pcl();

// Generic counting detours for the remaining exports. All of them take <= 5 integer/pointer
// arguments; forwarding six register/stack slots is transparent for them.
using GenericFn = std::uint64_t (*)(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
InlineHook g_generic_hooks[16];
template <int I>
std::uint64_t hk_generic(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d, std::uint64_t e, std::uint64_t f) {
    if (auto* s = stats()) ++s->export_calls[I];
    return reinterpret_cast<GenericFn>(g_generic_hooks[I].original())(a, b, c, d, e, f);
}
template <int... I>
std::array<void*, sizeof...(I)> generic_table(std::integer_sequence<int, I...>) {
    return {reinterpret_cast<void*>(&hk_generic<I>)...};
}

int hk_set_constants(const void* values, const void* frame, const void* viewport) {
    std::uint32_t id = 0;
    if (g_producer && read_frame_token(frame, &id)) {
        g_current_frame = id;
        g_last_constants_qpc = qpc_now();
        if (auto* s = stats()) ++s->constants_calls;
        Camera cam{};
        int base = g_constants_base;
        sl::Constants got = sl::Constants::NotHere;
        if (base >= 0) got = read_constants(static_cast<const std::uint8_t*>(values), base, cam);
        else {
            // (the base offset: where the constants read as a camera, or as a game's "no camera" - not where they read as neither)
            for (int candidate : {32, 0})
                if ((got = read_constants(static_cast<const std::uint8_t*>(values), candidate, cam)) != sl::Constants::NotHere) {
                    g_constants_base = base = candidate;
                    if (auto* s = stats()) s->constants_base = candidate;
                    break;
                }
        }
        // (a camera found unusable is left out: the upscaler's hooks publish the frames, camera estimated)
        if (got == sl::Constants::Camera && !g_producer->game_camera_unusable() && !g_producer->feed_publishing()) g_producer->on_constants(id, cam);
        // (no camera: published with the frame's tags - handle_tags - as an estimated one)
        if (got == sl::Constants::NoCamera) {
            g_last_no_camera_qpc = qpc_now();
            std::lock_guard lock(g_estimate_mutex);
            g_estimate = {id, cam, true};
        }
        if (!g_marker_hook.installed() && g_pcl_attempts < 200 && (g_pcl_attempts++ % 20) == 0) try_hook_pcl();
    }
    return reinterpret_cast<SetConstantsFn>(g_constants_hook.original())(values, frame, viewport);
}

bool is_depth_format(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_UNORM:
            return true;
        default: return false;
    }
}
bool is_rgba8(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            return true;
        default: return false;
    }
}

// Validates one ResourceTag at base b: Resource::type must be eTex2d (0) and Resource::native a texture - an ID3D12Resource
// (Direct3D 12 games) or an ID3D11Texture2D (Direct3D 11 games). UE's helper constructor leaves Resource::width/height at 0, so
// those only have to match when set. Returns 1 a D3D12 texture, 2 a D3D11 one, 0 no texture (null resource), -1 contradicts layout.
int tag_valid(const std::uint8_t* tag, int b) {
    const std::uint8_t* resource = nullptr;
    if (!safe_read(tag + b, &resource, 8)) return -1;
    if (!resource) return 0;
    std::uint8_t type = 0xFF;
    void* native = nullptr; std::uint32_t wh[2]{};
    if (!safe_read(resource + b, &type, 1) || !safe_read(resource + b + 8, &native, 8) || !safe_read(resource + b + 36, wh, 8)) return -1;
    if (!native) return 0;
    if (type != 0) return -1;
    ID3D12Resource* r = nullptr;
    ID3D11Texture2D* t = nullptr;
    __try {
        if (FAILED(static_cast<IUnknown*>(native)->QueryInterface(IID_PPV_ARGS(&r)))) {
            r = nullptr;
            if (FAILED(static_cast<IUnknown*>(native)->QueryInterface(IID_PPV_ARGS(&t)))) return -1;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) { return -1; }
    if (t) {
        D3D11_TEXTURE2D_DESC d{};
        t->GetDesc(&d);
        t->Release();
        if (wh[0] && (d.Width != wh[0] || d.Height != wh[1])) return -1;
        return 2;
    }
    const auto desc = r->GetDesc();
    r->Release();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return -1;
    if (wh[0] && (desc.Width != wh[0] || desc.Height != wh[1])) return -1;
    return 1;
}

// A Direct3D 11 game's command buffer: its immediate context (AddRef'd), or null - a deferred context is left alone (its copies would
// run whenever the game executes it, as hk_d3d11_evaluate does with DLSS), as is anything that is no D3D11 context.
ID3D11DeviceContext* immediate_context(void* command_buffer) {
    ID3D11DeviceContext* c = nullptr;
    __try {
        if (FAILED(static_cast<IUnknown*>(command_buffer)->QueryInterface(IID_PPV_ARGS(&c)))) return nullptr;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    if (c->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) { c->Release(); return nullptr; }
    return c;
}

// What XPAR takes a tag as (both APIs): kTexCount when not taken. Only the buffer types whose numbers are certain; UI colour was
// identified from the logged list first. The upscaler's output (kSlScalingOutputColor) is handled apart (handle_tags).
Tex tag_kind(std::uint32_t type, DXGI_FORMAT format, std::uint64_t width) {
    if (type == kSlDepth && is_depth_format(format)) return kDepth;
    if (type == kSlMotionVectors) return kMotion;
    if (type == kSlHudless && is_rgba8(format)) return kHudless;
    // E33 (UE StreamlineCore): UI colour + alpha from the UI hint extraction pass, full-res BGRA8.
    if (type == kSlUiColorAlpha && is_rgba8(format) && g_producer->shared() && width == g_producer->shared()->backbuffer_width) return kUi;
    return kTexCount;
}

// A layout is accepted when at least one tag validates and none contradicts it.
// sl::ResourceTag grew extra fields in newer Streamline builds, so its size is not assumed: every
// element begins with BaseStructure { next, StructType GUID, version }, and the stride is the
// distance to the next occurrence of the first element's GUID.
std::size_t tag_stride(const std::uint8_t* tags, std::uint32_t count, int b) {
    if (b != 32 || count < 2) return static_cast<std::size_t>(b) + 24;
    std::uint64_t guid[2]{};
    if (!safe_read(tags + 8, guid, 16) || (!guid[0] && !guid[1])) return 56;
    for (std::size_t s = 56; s <= 256; s += 8) {
        std::uint64_t other[2]{};
        if (safe_read(tags + s + 8, other, 16) && other[0] == guid[0] && other[1] == guid[1]) return s;
    }
    return 0;
}

void handle_tags(std::uint64_t frame, const void* tags_ptr, std::uint32_t count, void* command_buffer) {
    if (!g_producer || !tags_ptr || !count || !command_buffer || g_producer->game_camera_unusable()) return;
    auto* tags = static_cast<const std::uint8_t*>(tags_ptr);
    // Base size follows sl::Constants (both derive from BaseStructure); default to Streamline 2.x.
    const int b = g_constants_base >= 0 ? static_cast<int>(g_constants_base) : 32;
    const std::size_t stride = tag_stride(tags, count, b);
    if (auto* s = stats()) { s->tag_base = b; s->pad = static_cast<std::uint32_t>(stride); }
    if (!stride) {
        std::uint64_t q[12]{};
        safe_read(tags, q, sizeof(q));
        char text[256];
        std::snprintf(text, sizeof(text), "tag stride unknown (count %u): %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx %llx",
                      count, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11]);
        g_producer->set_message(text);
        return;
    }
    // First every tag read, then handed over: the depth's area may need the motion vectors' size (sl::depth_area).
    struct Entry { Tex kind; std::uint32_t type, state; void* native; bool d3d11, has_extent; sl::Area extent; std::uint32_t w, h; };
    Entry entries[32];
    int n = 0;
    std::uint32_t motion_w = 0, motion_h = 0;
    sl::Area motion_extent;
    bool motion_has_extent = false;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint8_t* tag = tags + i * stride;
        const std::uint8_t* resource = nullptr; std::uint32_t type = 0; const std::uint32_t* extent = nullptr;
        if (!safe_read(tag + b, &resource, 8) || !safe_read(tag + b + 8, &type, 4) || !safe_read(tag + b + 16, &extent, 8)) continue;
        auto* s = stats();
        if (s && type < 64) ++s->tag_count[type];
        const int api = tag_valid(tag, b);
        if (api != 1 && api != 2) continue;  // no texture (extent-only tag) or not a texture
        void* native = nullptr; std::uint32_t state = 0;
        if (!safe_read(resource + b + 8, &native, 8) || !safe_read(resource + b + 32, &state, 4) || !native) continue;
        std::uint32_t ext[4]{};  // top, left, width, height
        const bool has_extent = extent && safe_read(extent, ext, sizeof(ext)) && ext[2] && ext[3];
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN; std::uint32_t w = 0, h = 0;
        if (api == 1) {
            const auto desc = static_cast<ID3D12Resource*>(native)->GetDesc();
            format = desc.Format; w = static_cast<std::uint32_t>(desc.Width); h = desc.Height;
        } else {
            D3D11_TEXTURE2D_DESC d{};
            static_cast<ID3D11Texture2D*>(static_cast<ID3D11Resource*>(native))->GetDesc(&d);
            format = d.Format; w = d.Width; h = d.Height;
        }
        if (s && type < 64) { s->tag_format[type] = format; s->tag_width[type] = w; s->tag_height[type] = h; }
        // The upscaler's output (the scene before post-processing and HUD): only when the HUD is to be found from it (opt-in) and the
        // game has no HUD layers; kept until DLSS has written it. Direct3D 12 only (copied after slEvaluateFeature on its list).
        if (type == kSlScalingOutputColor && api == 1 && g_producer->shared() && g_producer->shared()->settings.hud_from_scene &&
            w == g_producer->shared()->backbuffer_width && s && s->tag_count[kSlHudless] == 0 && s->tag_count[kSlUiColorAlpha] == 0) {
            std::lock_guard lock(g_output_mutex);
            g_output = {frame, static_cast<ID3D12Resource*>(native), state, {ext[0], ext[1], ext[2], ext[3]}};
            continue;
        }
        const Tex kind = tag_kind(type, format, w);
        if (kind == kTexCount || n >= 32) continue;
        const sl::Area area{ext[1], ext[0], ext[2], ext[3]};
        if (kind == kMotion) { motion_w = w; motion_h = h; motion_extent = area; motion_has_extent = has_extent; }
        entries[n++] = {kind, type, state, native, api == 2, has_extent, area, w, h};
    }
    ID3D11DeviceContext* context = nullptr;
    bool context_read = false;
    bool depth_sent = false, motion_sent = false;
    for (int i = 0; i < n; ++i) {
        Entry& e = entries[i];
        sl::Area a = e.has_extent ? e.extent : sl::Area{};
        if (e.kind == kDepth) {
            a = sl::depth_area(e.w, e.h, e.has_extent ? &e.extent : nullptr, motion_w, motion_h, motion_has_extent ? &motion_extent : nullptr);
        }
        if (!e.d3d11) {
            // D3D12 state value: Streamline passes the native D3D12_RESOURCE_STATES of the tagged resource.
            g_producer->on_tag(frame, e.kind, static_cast<ID3D12Resource*>(e.native), static_cast<D3D12_RESOURCE_STATES>(e.state), a.x, a.y, a.w, a.h,
                               static_cast<ID3D12GraphicsCommandList*>(command_buffer));
        } else {
            if (!context_read) { context = immediate_context(command_buffer); context_read = true; }
            if (!context) {
                if (auto* s = stats()) ++s->copies_failed;  // (a deferred context: see immediate_context)
                continue;
            }
            g_producer->on_tag_d3d11(frame, e.kind, static_cast<ID3D11Resource*>(e.native), a.x, a.y, a.w, a.h, context);
        }
        // (the game's depth seen - the ReShade feed stays out - only once handed over: not for a deferred context's tags)
        if (e.kind == kDepth) { g_last_depth_qpc = qpc_now(); g_producer->note_game_depth(); }
        depth_sent |= e.kind == kDepth;
        motion_sent |= e.kind == kMotion;
    }
    if (context) context->Release();
    // Constants without a camera for this frame: its camera estimated by the presenter, the vectors' scale in pixels of their area
    // (sl::Constants::mvecScale makes them uv; "vector * scale = render pixels" for an estimated camera, as from NGX).
    if (depth_sent && motion_sent) {
        Camera cam{};
        bool publish = false;
        {
            std::lock_guard lock(g_estimate_mutex);
            if (g_estimate.set && g_estimate.frame == frame) { cam = g_estimate.cam; publish = true; g_estimate.set = false; }
        }
        if (publish && !g_producer->feed_publishing()) {
            const std::uint32_t aw = motion_has_extent ? motion_extent.w : motion_w, ah = motion_has_extent ? motion_extent.h : motion_h;
            cam.valid = 1;
            cam.mvec_scale[0] *= static_cast<float>(aw);
            cam.mvec_scale[1] *= static_cast<float>(ah);
            g_producer->on_constants(frame, cam);
        }
    }
}

int hk_set_tag(const void* viewport, const void* tags, std::uint32_t count, void* cmd) {
    if (auto* s = stats()) ++s->tag_calls;
    handle_tags(g_current_frame, tags, count, cmd);
    return reinterpret_cast<SetTagFn>(g_tag_hook.original())(viewport, tags, count, cmd);
}

int hk_set_tag_for_frame(const void* frame, const void* viewport, const void* tags, std::uint32_t count, void* cmd) {
    if (auto* s = stats()) ++s->tag_for_frame_calls;
    std::uint32_t id = 0;
    if (read_frame_token(frame, &id)) handle_tags(id, tags, count, cmd);
    return reinterpret_cast<SetTagForFrameFn>(g_tag_frame_hook.original())(frame, viewport, tags, count, cmd);
}

int hk_evaluate(std::uint32_t feature, const void* frame, const void** inputs, std::uint32_t count, void* cmd) {
    if (auto* s = stats()) ++s->export_calls[3];
    fw::ui_probe::mark(reinterpret_cast<std::uint64_t>(cmd), ("upscaler: Streamline evaluate, feature " + std::to_string(feature)).c_str());
    const int result = reinterpret_cast<EvaluateFeatureFn>(g_evaluate_hook.original())(feature, frame, inputs, count, cmd);
    std::uint32_t id = 0;
    if (g_producer && cmd && read_frame_token(frame, &id)) {
        PendingOutput out;
        {
            std::lock_guard lock(g_output_mutex);
            if (g_output.resource && g_output.frame == id) { out = g_output; g_output = {}; }
        }
        // Recorded after the evaluation on the same command list: the copy sees what DLSS wrote.
        if (out.resource)
            g_producer->on_tag(id, kScene, out.resource, static_cast<D3D12_RESOURCE_STATES>(out.state), out.ext[1], out.ext[0], out.ext[2],
                               out.ext[3], static_cast<ID3D12GraphicsCommandList*>(cmd));
    }
    return result;
}

int hk_marker(std::uint32_t marker, const void* frame) {
    std::uint32_t id = 0;
    if (g_producer && read_frame_token(frame, &id)) {
        if (auto* s = stats()) { ++s->marker_calls; if (marker < 16) ++s->marker_counts[marker]; }
        if (marker == kSimulationStart) g_producer->on_sim_start(id, qpc_now());
        else if (marker == kPresentStart) g_producer->on_present_marker(id);
        else if (marker == kRenderSubmitStart) g_producer->on_render_submit(id);
    }
    return reinterpret_cast<MarkerFn>(g_marker_hook.original())(marker, frame);
}

void try_hook_pcl() {
    HMODULE sl = GetModuleHandleW(L"sl.interposer.dll");
    if (!sl) return;
    auto get = reinterpret_cast<GetFeatureFunctionFn>(GetProcAddress(sl, "slGetFeatureFunction"));
    if (!get) return;
    void* fn = nullptr;
    const int pcl = get(4, "slPCLSetMarker", fn);
    if (auto* s = stats()) s->pcl_lookup_result = pcl;
    if (pcl != 0 || !fn) {
        fn = nullptr;
        const int reflex = get(3, "slReflexSetMarker", fn);
        if (auto* s = stats()) s->reflex_lookup_result = reflex;
        if (reflex != 0 || !fn) return;
    }
    if (g_marker_hook.install(fn, reinterpret_cast<void*>(&hk_marker))) {
        if (auto* s = stats()) { s->pcl_hooked = 1; s->hooks_installed |= 8; }
    } else if (g_producer) {
        g_producer->set_message(("PCL marker hook: " + g_marker_hook.error()).c_str());
    }
}

}  // namespace

const char* const kCountedExports[] = {"slInit", "slShutdown", "slGetNewFrameToken", "slEvaluateFeature", "slIsFeatureSupported",
                                       "slSetFeatureLoaded", "slAllocateResources", "slFreeResources", "slGetFeatureFunction",
                                       "slSetD3DDevice", "slGetFeatureRequirements", "slGetFeatureVersion"};
const int kCountedExportCount = 12;
const std::uint32_t kProbedFeatures[] = {0, 1, 2, 3, 4, 5, 6, 7, 1000, 1001, 1002, 1003};
const int kProbedFeatureCount = 12;

bool streamline_loaded() { return GetModuleHandleW(L"sl.interposer.dll") != nullptr; }
bool streamline_camera_recent() {
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    const std::int64_t last = g_last_constants_qpc.load();
    return last && qpc_now() - last < f.QuadPart;
}
bool streamline_depth_recent() {
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    const std::int64_t last = g_last_depth_qpc.load();
    return last && qpc_now() - last < f.QuadPart;
}
std::uint64_t streamline_current_frame() { return g_current_frame.load(); }
bool streamline_no_camera_recent() {
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    const std::int64_t last = g_last_no_camera_qpc.load();
    return last && qpc_now() - last < f.QuadPart;
}

void probe_streamline_features() {
    HMODULE sl = GetModuleHandleW(L"sl.interposer.dll");
    auto* s = stats();
    if (!sl || !s) return;
    using IsLoadedFn = int (*)(std::uint32_t, bool&);
    auto fn = reinterpret_cast<IsLoadedFn>(GetProcAddress(sl, "slIsFeatureLoaded"));
    if (!fn) return;
    for (int i = 0; i < kProbedFeatureCount; ++i) {
        bool loaded = false;
        s->feature_result[i] = fn(kProbedFeatures[i], loaded);
        s->feature_loaded[i] = loaded ? 1 : 0;
    }
    if (!g_marker_hook.installed()) try_hook_pcl();
}

void install_streamline_hooks(Producer* producer) {
    g_producer = producer;
    HMODULE sl = GetModuleHandleW(L"sl.interposer.dll");
    if (!sl) return;
    struct Entry { InlineHook& hook; const char* name; void* detour; std::uint32_t bit; };
    Entry entries[] = {
        {g_constants_hook, "slSetConstants", reinterpret_cast<void*>(&hk_set_constants), 1},
        {g_tag_hook, "slSetTag", reinterpret_cast<void*>(&hk_set_tag), 2},
        {g_tag_frame_hook, "slSetTagForFrame", reinterpret_cast<void*>(&hk_set_tag_for_frame), 4},
        {g_evaluate_hook, "slEvaluateFeature", reinterpret_cast<void*>(&hk_evaluate), 16},
    };
    static const auto generic = generic_table(std::make_integer_sequence<int, 12>{});
    for (int i = 0; i < kCountedExportCount; ++i) {
        if (g_generic_hooks[i].installed() || i == 3) continue;  // slEvaluateFeature has its own detour (counts too)
        if (void* target = reinterpret_cast<void*>(GetProcAddress(sl, kCountedExports[i])))
            g_generic_hooks[i].install(target, generic[i]);
    }
    for (auto& e : entries) {
        if (e.hook.installed()) continue;
        void* target = reinterpret_cast<void*>(GetProcAddress(sl, e.name));
        if (!target) continue;
        if (e.hook.install(target, e.detour)) {
            if (auto* s = stats()) s->hooks_installed |= e.bit;
        } else if (producer) {
            producer->set_message((std::string(e.name) + ": " + e.hook.error()).c_str());
        }
    }
}

}  // namespace fw
