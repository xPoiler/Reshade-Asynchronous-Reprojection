#include "addon/streamline_hooks.hpp"
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

bool is_unit_basis(const float* up, const float* right, const float* fwd) {
    auto len = [](const float* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    for (const float* v : {up, right, fwd})
        if (!(std::fabs(len(v) - 1.0f) < 0.02f)) return false;
    return std::fabs(dot(up, right)) < 0.02f && std::fabs(dot(up, fwd)) < 0.02f && std::fabs(dot(right, fwd)) < 0.02f;
}

// Converts sl::Constants at base offset b. Returns false when the basis does not look like a camera.
bool read_constants(const std::uint8_t* c, int b, Camera& cam) {
    std::uint8_t raw[424];
    if (!safe_read(c + b, raw, sizeof(raw))) return false;
    std::memcpy(cam.view_to_clip, raw + 0, 64);
    std::memcpy(cam.clip_to_view, raw + 64, 64);
    std::memcpy(cam.clip_to_prev_clip, raw + 192, 64);
    std::memcpy(cam.jitter, raw + 320, 8);
    std::memcpy(cam.mvec_scale, raw + 328, 8);
    std::memcpy(cam.pos, raw + 344, 12);
    std::memcpy(cam.up, raw + 356, 12);
    std::memcpy(cam.right, raw + 368, 12);
    std::memcpy(cam.fwd, raw + 380, 12);
    std::memcpy(&cam.near_plane, raw + 392, 4);
    std::memcpy(&cam.far_plane, raw + 396, 4);
    std::memcpy(&cam.fov, raw + 400, 4);
    std::memcpy(&cam.aspect, raw + 404, 4);
    cam.depth_inverted = raw[412] == 1;
    cam.reset = raw[415] == 1;
    return is_unit_basis(cam.up, cam.right, cam.fwd);
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
        bool ok = false;
        if (base >= 0) ok = read_constants(static_cast<const std::uint8_t*>(values), base, cam);
        else {
            for (int candidate : {32, 0})
                if (read_constants(static_cast<const std::uint8_t*>(values), candidate, cam)) {
                    g_constants_base = base = candidate; ok = true;
                    if (auto* s = stats()) s->constants_base = candidate;
                    break;
                }
        }
        // (a camera found unusable is left out: the upscaler's hooks publish the frames, camera estimated)
        if (ok && !g_producer->game_camera_unusable() && !g_producer->feed_publishing()) g_producer->on_constants(id, cam);
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

// Validates one ResourceTag at base b: Resource::type must be eTex2d (0) and Resource::native an
// ID3D12Resource texture. UE's helper constructor leaves Resource::width/height at 0, so those
// only have to match when set. Returns 1 valid, 0 no texture (null resource), -1 contradicts layout.
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
    __try {
        if (FAILED(static_cast<IUnknown*>(native)->QueryInterface(IID_PPV_ARGS(&r)))) return -1;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return -1; }
    const auto desc = r->GetDesc();
    r->Release();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return -1;
    if (wh[0] && (desc.Width != wh[0] || desc.Height != wh[1])) return -1;
    return 1;
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
    auto* list = static_cast<ID3D12GraphicsCommandList*>(command_buffer);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint8_t* tag = tags + i * stride;
        const std::uint8_t* resource = nullptr; std::uint32_t type = 0; const std::uint32_t* extent = nullptr;
        if (!safe_read(tag + b, &resource, 8) || !safe_read(tag + b + 8, &type, 4) || !safe_read(tag + b + 16, &extent, 8)) continue;
        auto* s = stats();
        if (s && type < 64) ++s->tag_count[type];
        if (tag_valid(tag, b) != 1) continue;  // no texture (extent-only tag) or not a D3D12 texture
        void* native = nullptr; std::uint32_t state = 0;
        if (!safe_read(resource + b + 8, &native, 8) || !safe_read(resource + b + 32, &state, 4) || !native) continue;
        std::uint32_t ext[4]{};  // top, left, width, height
        if (extent) safe_read(extent, ext, sizeof(ext));
        auto* d3d = static_cast<ID3D12Resource*>(native);
        const auto desc = d3d->GetDesc();
        if (s && type < 64) {
            s->tag_format[type] = desc.Format; s->tag_width[type] = static_cast<std::uint32_t>(desc.Width);
            s->tag_height[type] = desc.Height;
        }
        // Only the buffer types whose numbers are certain; UI colour is identified from the logged list first.
        Tex kind = kTexCount;
        if (type == kSlDepth && is_depth_format(desc.Format)) { kind = kDepth; g_last_depth_qpc = qpc_now(); g_producer->note_game_depth(); }
        else if (type == kSlMotionVectors) kind = kMotion;
        else if (type == kSlHudless && is_rgba8(desc.Format)) kind = kHudless;
        // E33 (UE StreamlineCore): UI colour + alpha from the UI hint extraction pass, full-res BGRA8.
        else if (type == kSlUiColorAlpha && is_rgba8(desc.Format) && g_producer->shared() &&
                 desc.Width == g_producer->shared()->backbuffer_width) kind = kUi;
        // The upscaler's output (the scene before post-processing and HUD): only when the HUD is to be found
        // from it (opt-in) and the game has no HUD layers; kept until DLSS has written it.
        if (type == kSlScalingOutputColor && g_producer->shared() && g_producer->shared()->settings.hud_from_scene &&
            desc.Width == g_producer->shared()->backbuffer_width &&
            s && s->tag_count[kSlHudless] == 0 && s->tag_count[kSlUiColorAlpha] == 0) {
            std::lock_guard lock(g_output_mutex);
            g_output = {frame, d3d, state, {ext[0], ext[1], ext[2], ext[3]}};
            continue;
        }
        if (kind == kTexCount) continue;
        // D3D12 state value: Streamline passes the native D3D12_RESOURCE_STATES of the tagged resource.
        g_producer->on_tag(frame, kind, d3d, static_cast<D3D12_RESOURCE_STATES>(state), ext[1], ext[0], ext[2], ext[3], list);
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
