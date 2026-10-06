#include "frame_cap.hpp"

#include <d3d11_4.h>
#include <d3d12.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace fw {
namespace {

using Microsoft::WRL::ComPtr;
using namespace reshade::api;

std::int64_t qpc() { LARGE_INTEGER q; QueryPerformanceCounter(&q); return q.QuadPart; }
double qpc_frequency() { static const double f = [] { LARGE_INTEGER q; QueryPerformanceFrequency(&q); return double(q.QuadPart); }(); return f; }

// NVIDIA Reflex: whether the game turned its low-latency mode on for its device (NVAPI, the driver's own state,
// whether the game sets it directly or through Streamline). No NVAPI (other GPUs): never on.
// NV_GET_SLEEP_STATUS_PARAMS_V1: the version word, then the low-latency flag (byte 4), then fields and reserved
// bytes whose total size differs between NVAPI releases - the driver checks the size in the version word. The
// size it accepts is found once (sizes tried until it stops answering "incompatible struct version", -9).
struct SleepStatusParams {
    std::uint32_t version;
    std::uint8_t low_latency_mode;
    std::uint8_t rest[507];
};
using QueryInterfaceFn = void* (*)(std::uint32_t);
using InitializeFn = int (*)();
using GetSleepStatusFn = int (*)(IUnknown*, SleepStatusParams*);

GetSleepStatusFn sleep_status_fn() {
    static GetSleepStatusFn fn = [] () -> GetSleepStatusFn {
        HMODULE nvapi = LoadLibraryW(L"nvapi64.dll");
        if (!nvapi) return nullptr;
        const auto query = reinterpret_cast<QueryInterfaceFn>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
        if (!query) return nullptr;
        const auto initialize = reinterpret_cast<InitializeFn>(query(0x0150E828));    // NvAPI_Initialize
        const auto status = reinterpret_cast<GetSleepStatusFn>(query(0xAEF96CA1));    // NvAPI_D3D_GetSleepStatus
        if (!initialize || !status || initialize() != 0) return nullptr;
        return status;
    }();
    return fn;
}

// The last result of the Reflex query, logged when it changes (the query's result code and the mode).
int g_reflex_logged = -2;

void log_reflex(int result, bool on) {
    const int key = result * 2 + (on ? 1 : 0);
    if (key == g_reflex_logged) return;
    g_reflex_logged = key;
    char text[160];
    std::snprintf(text, sizeof(text), "XPAR frame cap: the game's NVIDIA Reflex is %s (NVAPI sleep status: %d)", on ? "on - the cap stands aside" : "off", result);
    reshade::log::message(reshade::log::level::info, text);
}

bool reflex_on_logged(IUnknown* device) {
    const GetSleepStatusFn fn = sleep_status_fn();
    if (!fn || !device) { log_reflex(-1000, false); return false; }
    static std::uint32_t size = 0;  // the accepted size, once found
    int result = -9;
    SleepStatusParams p{};
    if (size) {
        p.version = size | (1u << 16);
        result = fn(device, &p);
    } else {
        for (std::uint32_t tried = 8; tried <= sizeof(SleepStatusParams) && result == -9; ++tried) {
            p = {};
            p.version = tried | (1u << 16);
            result = fn(device, &p);
            if (result != -9) size = tried;
        }
        if (size) {
            char text[96];
            std::snprintf(text, sizeof(text), "XPAR frame cap: NVAPI sleep status takes %u bytes", size);
            reshade::log::message(reshade::log::level::info, text);
        }
    }
    const bool on = result == 0 && p.low_latency_mode != 0;
    log_reflex(result, on);
    return on;
}

// Per present: when it was presented, when the frame after it was released (the cap's wait over), and when the
// GPU finished everything submitted up to it (a fence signalled right after the present). Values only grow, across
// fences too, so an entry is never mistaken for one of an earlier fence.
constexpr int kRing = 512;
struct Entry {
    std::atomic<std::uint64_t> value{0};
    std::atomic<std::int64_t> present{0}, release{0}, done{0};
};
Entry g_ring[kRing];

// One per fence, shared with its thread, which waits for the values in order and notes when each completed. A
// waiter replaced (another queue presents) or left at unload is told to stop and ends on its own: never joined
// (unloading may run under the loader lock), and the add-on stays loaded (pinned), so the thread's code does too.
struct Waiter {
    ComPtr<ID3D12Fence> fence12;
    ComPtr<ID3D11Fence> fence11;
    HANDLE wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HANDLE completion = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    std::atomic<std::uint64_t> signalled{0};
    std::atomic<bool> stop{false};
    std::uint64_t first = 0;  // the first value signalled on this fence
    ~Waiter() { CloseHandle(wake); CloseHandle(completion); }
};

void wait_loop(std::shared_ptr<Waiter> w) {
    std::uint64_t waited = w->first - 1;
    while (!w->stop.load()) {
        if (w->signalled.load() <= waited) { WaitForSingleObject(w->wake, 100); continue; }
        const std::uint64_t v = waited + 1;
        const HRESULT hr = w->fence12 ? w->fence12->SetEventOnCompletion(v, w->completion) : w->fence11->SetEventOnCompletion(v, w->completion);
        if (FAILED(hr)) { waited = v; continue; }
        if (WaitForSingleObject(w->completion, 100) != WAIT_OBJECT_0) continue;  // (checks stop, then waits again)
        const std::int64_t t = qpc();
        Entry& e = g_ring[v % kRing];
        if (e.value.load() == v) e.done = t;
        waited = v;
    }
}

struct Cap {
    // The game's present queue (D3D12) or immediate context (D3D11, 11.4 fences) and its device; another queue
    // presenting 30 times in a row takes over (a game that recreated its device).
    void* native = nullptr;
    IUnknown* device = nullptr;
    bool failed = false;
    void* candidate = nullptr;
    int candidate_presents = 0;
    std::shared_ptr<Waiter> waiter;
    ComPtr<ID3D12CommandQueue> queue12;
    ComPtr<ID3D11DeviceContext4> context11;
    std::uint64_t next = 0;
    HANDLE timer = nullptr;
    bool reflex = false;
    std::int64_t reflex_checked = 0;
    std::int64_t reflex_off_since = 0;  // (0: Reflex seen on, or not checked yet)

    // The cap: the time between two frame starts (s, 0: not set). Started 5% above the uncapped interval; every
    // 60 frames it backs off 3% when frames wait in the GPU's queue, and otherwise moves faster (see control).
    double interval = 0;
    std::int64_t last_release = 0;
    int frames = 0;
    double wait_ms_sum = 0;

    std::mutex status_mutex;
    FrameCapStatus status;
} g;

void release_queue() {
    if (g.waiter) { g.waiter->stop = true; SetEvent(g.waiter->wake); g.waiter.reset(); }
    g.queue12.Reset(); g.context11.Reset();
    g.native = nullptr; g.device = nullptr; g.failed = false;
    g.interval = 0; g.last_release = 0; g.frames = 0; g.wait_ms_sum = 0;
    g.reflex = false; g.reflex_checked = 0; g.reflex_off_since = 0;
    std::lock_guard<std::mutex> lock(g.status_mutex);
    g.status = {};
}

bool ensure_fence(command_queue* queue) {
    void* native = reinterpret_cast<void*>(queue->get_native());
    if (g.native == native) { g.candidate_presents = 0; return !g.failed; }
    if (g.native) {
        if (native != g.candidate) { g.candidate = native; g.candidate_presents = 0; }
        if (++g.candidate_presents < 30) return false;
        release_queue();
    }
    g.candidate = nullptr; g.candidate_presents = 0;
    g.native = native;
    g.device = reinterpret_cast<IUnknown*>(queue->get_device()->get_native());
    auto w = std::make_shared<Waiter>();
    const device_api api = queue->get_device()->get_api();
    if (api == device_api::d3d12) {
        g.queue12 = reinterpret_cast<ID3D12CommandQueue*>(native);
        ComPtr<ID3D12Device> device;
        if (FAILED(g.queue12->GetDevice(IID_PPV_ARGS(&device))) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&w->fence12)))) g.failed = true;
    } else if (api == device_api::d3d11) {
        auto* context = reinterpret_cast<ID3D11DeviceContext*>(native);
        ComPtr<ID3D11Device> device;
        context->GetDevice(&device);
        ComPtr<ID3D11Device5> device5;
        if (!device || FAILED(device.As(&device5)) || FAILED(context->QueryInterface(IID_PPV_ARGS(&g.context11))) ||
            FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&w->fence11))))
            g.failed = true;
    } else {
        g.failed = true;  // (Vulkan: not yet)
    }
    if (g.failed) { g.queue12.Reset(); g.context11.Reset(); return false; }
    if (!g.timer) g.timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    w->first = g.next + 1;
    g.waiter = w;
    std::thread(wait_loop, w).detach();
    return true;
}

void sleep_until(std::int64_t target) {
    const double f = qpc_frequency();
    for (;;) {
        const std::int64_t left = target - qpc();
        if (left <= 0) return;
        if (left > static_cast<std::int64_t>(1e-3 * f) && g.timer) {
            LARGE_INTEGER due;
            due.QuadPart = -static_cast<LONGLONG>(double(left - static_cast<std::int64_t>(5e-4 * f)) / f * 1e7);
            if (SetWaitableTimer(g.timer, &due, 0, nullptr, nullptr, FALSE)) { WaitForSingleObject(g.timer, 100); continue; }
        }
        YieldProcessor();
    }
}

double median(std::vector<double>& v) {
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
}

// Once 60 frames: the window's numbers (frames old enough for their GPU work to be done), then the cap.
void control(bool capping) {
    const double ms = 1000.0 / qpc_frequency();
    std::vector<double> after_present, after_start, intervals;
    int queued = 0, pairs = 0;
    const std::uint64_t newest = g.next;
    const std::uint64_t oldest = std::max<std::uint64_t>(g.waiter ? g.waiter->first + 1 : 1, newest > 120 ? newest - 120 : 1);
    for (std::uint64_t v = oldest; v + 2 <= newest; ++v) {
        const Entry& e = g_ring[v % kRing];
        const Entry& before = g_ring[(v - 1) % kRing];
        if (e.value.load() != v) continue;
        const std::int64_t done = e.done.load(), present = e.present.load();
        if (before.value.load() == v - 1) {
            const std::int64_t start = before.release.load();
            if (start && e.release.load() > start) intervals.push_back(double(e.release.load() - start) * ms);
            if (start && done > start) after_start.push_back(double(done - start) * ms);
        }
        if (done > present) after_present.push_back(double(done - present) * ms);
        // Queued: the GPU still on this frame when the game had already presented the next one (that one waited).
        const Entry& after = g_ring[(v + 1) % kRing];
        if (done && after.value.load() == v + 1 && after.present.load()) { ++pairs; queued += done > after.present.load(); }
    }
    if (after_present.size() < 30 || intervals.size() < 30) return;
    const double l = median(after_present), frame = median(intervals);
    const double e = after_start.empty() ? 0.0 : median(after_start);
    const double queued_share = pairs ? double(queued) / pairs : 0.0;
    if (capping) {
        if (g.interval <= 0) {
            g.interval = frame * 1.05e-3;  // (5% under the uncapped rate: the queue it had drains within a second)
        } else if (queued_share > 0.25) {
            g.interval *= 1.03;  // frames wait in the GPU's queue: slower
        } else if (queued_share < 0.05) {
            // No queue. A frame done well before the next one is due (its start to its GPU work done, under 3/4 of
            // the cap): the GPU idles part of every frame - much faster at once (loading screens leave the cap low).
            // Otherwise creep 0.5% faster, until a queue forms.
            g.interval = e > 0 && e < 0.75 * frame ? std::max(g.interval * 0.85, e * 1.1e-3) : g.interval * 0.995;
        }
        g.interval = std::clamp(g.interval, 1.0 / 500.0, 0.05);  // (never below 20 fps)
    } else {
        g.interval = 0;
    }
    std::lock_guard<std::mutex> lock(g.status_mutex);
    g.status.active = capping && g.interval > 0;
    g.status.cap_fps = g.interval > 0 ? float(1.0 / g.interval) : 0.0f;
    g.status.game_fps = frame > 0 ? float(1000.0 / frame) : 0.0f;
    g.status.present_to_done_ms = float(l);
    g.status.start_to_done_ms = float(e);
    g.status.wait_ms = float(g.wait_ms_sum / std::max(g.frames, 1));
    g.status.queued_pct = float(queued_share * 100.0);
}

}  // namespace

void frame_cap_after_present(command_queue* queue, bool enabled, bool generation) {
    if (!queue || !ensure_fence(queue)) return;
    const std::int64_t now = qpc();
    const std::uint64_t v = ++g.next;
    Entry& e = g_ring[v % kRing];
    e.value = 0;
    e.present = now; e.release = 0; e.done = 0;
    e.value = v;
    if (g.queue12) {
        g.queue12->Signal(g.waiter->fence12.Get(), v);
    } else {
        g.context11->Signal(g.waiter->fence11.Get(), v);
        g.context11->Flush();
    }
    g.waiter->signalled = v;
    SetEvent(g.waiter->wake);

    // (the game's Reflex, checked twice a second: it may be switched in the game's menu)
    if (now - g.reflex_checked > static_cast<std::int64_t>(0.5 * qpc_frequency())) {
        g.reflex = reflex_on_logged(g.device);
        g.reflex_checked = now;
        if (g.reflex) g.reflex_off_since = 0;
        else if (!g.reflex_off_since) g.reflex_off_since = now;
        std::lock_guard<std::mutex> lock(g.status_mutex);
        g.status.reflex = g.reflex;
        if (g.reflex || !enabled || generation) g.status.active = false;
    }
    // (Reflex off for 5 s first: a game switches it on a few seconds after its first frames)
    const bool reflex_off = g.reflex_off_since && now - g.reflex_off_since > static_cast<std::int64_t>(5.0 * qpc_frequency());
    const bool capping = enabled && !generation && reflex_off;
    std::int64_t release = now;
    if (capping && g.interval > 0 && g.last_release) {
        const std::int64_t target = g.last_release + static_cast<std::int64_t>(g.interval * qpc_frequency());
        // (more than a whole interval late: no wait, and no debt carried into the next frames)
        if (target > now) { sleep_until(target); release = qpc(); }
    }
    e.release = release;
    g.wait_ms_sum += double(release - now) * 1000.0 / qpc_frequency();
    g.last_release = release;
    if (++g.frames >= 60) {
        control(capping);
        g.frames = 0;
        g.wait_ms_sum = 0;
    }
}

FrameCapStatus frame_cap_status() {
    std::lock_guard<std::mutex> lock(g.status_mutex);
    return g.status;
}

void frame_cap_shutdown() {
    // (ReShade unloads the add-on whenever the game's device goes away, and loads it again for the next: everything
    // starts over then; the add-on itself stays loaded, so this state does too)
    release_queue();
}

}  // namespace fw
