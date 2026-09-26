// FrameWarpPresenter: presents the game's latest frame at display rate, reprojected with NVIDIA
// Latewarp to the camera predicted from raw mouse input and the game's own camera history.
#include "presenter/pose.hpp"
#include "presenter/camera_estimator.hpp"
#include "presenter/renderer.hpp"
#include <shellapi.h>
#include <winternl.h>  // NTSTATUS for d3dkmthk.h
#include <d3dkmthk.h>
#include <dxgi1_4.h>
#include <pdh.h>
#include <wrl/client.h>
#include <atomic>
#include <cstdarg>
#include <cstring>
#include <share.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace fw {
namespace {

double g_qpc_to_seconds = 0;
std::int64_t g_qpc_frequency = 10000000;
char g_gpu_priority[32] = "normal";
double seconds(std::int64_t qpc) { return double(qpc) * g_qpc_to_seconds; }
double now_seconds() { return seconds(qpc_now()); }

struct App {
    DWORD pid = 0;
    HANDLE game_process = nullptr;
    HANDLE mapping = nullptr;
    Shared* shared = nullptr;
    HWND overlay = nullptr;
    HWND game = nullptr;
    std::atomic<bool> running{true};
    std::atomic<bool> visible{false};     // game window in front and reprojection enabled
    std::atomic<bool> has_frames{false};  // presenter has something valid to show
    std::atomic<std::uint32_t> client_w{0}, client_h{0};
    PoseModel model;
    std::filesystem::path base_dir, profile_path;
    FILE* log = nullptr;
    // Analysis logs (all times are raw QPC ticks, same clock in both processes).
    FILE* csv_events = nullptr;   // game-side timeline
    FILE* csv_mouse = nullptr;    // raw input
    FILE* csv_sources = nullptr;  // ingested frames + camera
    FILE* csv_outputs = nullptr;  // presented frames + applied warp
    std::int64_t events_read = 0;
    bool mark_was_down = false;
};
App g_app;

void logf(const char* format, ...) {
    if (!g_app.log) return;
    va_list args; va_start(args, format);
    std::fprintf(g_app.log, "[%.3f] ", now_seconds());
    std::vfprintf(g_app.log, format, args);
    std::fputc('\n', g_app.log);
    std::fflush(g_app.log);
    va_end(args);
}

void set_status(const char* format, ...) {
    if (!g_app.shared) return;
    va_list args; va_start(args, format);
    std::vsnprintf(g_app.shared->presenter.message, sizeof(g_app.shared->presenter.message), format, args);
    va_end(args);
}


// Video memory, logged when it moves: the adapter's total dedicated usage (all processes, via the
// "GPU Adapter Memory" counters) and the presenter's own usage and budget.
struct VramMonitor {
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER counter = nullptr;
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
    wchar_t luid_tag[64]{};
    double last_log = 0, logged_total = -1;
    void init(const LUID& luid) {
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
        std::swprintf(luid_tag, 64, L"luid_0x%08lX_0x%08lX", static_cast<unsigned long>(luid.HighPart), luid.LowPart);
        if (PdhOpenQueryW(nullptr, 0, &query) == ERROR_SUCCESS &&
            PdhAddEnglishCounterW(query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &counter) == ERROR_SUCCESS)
            PdhCollectQueryData(query);
        else counter = nullptr;
    }
    double adapter_total_mb() {
        if (!counter || PdhCollectQueryData(query) != ERROR_SUCCESS) return -1;
        DWORD size = 0, count = 0;
        PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &size, &count, nullptr);
        if (!size) return -1;
        std::vector<unsigned char> buffer(size);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &size, &count, items) != ERROR_SUCCESS) return -1;
        double total = 0;
        for (DWORD i = 0; i < count; ++i)
            if (_wcsnicmp(items[i].szName, luid_tag, wcslen(luid_tag)) == 0) total += items[i].FmtValue.doubleValue;
        return total / (1024.0 * 1024.0);
    }
    void poll(double now, const char* reason = nullptr) {
        if (!reason && now - last_log < 2.0) return;
        const double total = adapter_total_mb();
        DXGI_QUERY_VIDEO_MEMORY_INFO own{};
        if (adapter) adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &own);
        if (!reason && std::abs(total - logged_total) < 100 && now - last_log < 30.0) return;
        logf("vram%s%s: adapter %.0f MB used (all processes) | presenter %.0f MB, budget %.0f MB", reason ? " at " : "", reason ? reason : "",
             total, own.CurrentUsage / 1048576.0, own.Budget / 1048576.0);
        logged_total = total; last_log = now;
    }
};

// Scale that maps the game's motion vectors to uv (per axis), measured against the camera-only motion of
// every pixel. Only frames that fit almost perfectly count (cuts, camera animations and large moving
// areas do not), and the scale locks once several agree. Motion vectors in render pixels (Unreal) are
// recognised and then follow the render resolution, so DLSS preset changes need no relearning.
struct MotionVectorScale {
    double quality = 0, moving_fraction = 0;
    bool valid = false, pixel_units = false;
    double locked[2] = {};     // uv per unit (generic) or +-1 per render pixel (pixel units)
    double candidate[2] = {};
    bool candidate_pixels = false;
    int agree = 0;
    // Returns true when the locked value changed.
    bool add(const MotionFit& f, double render_w, double render_h) {
        if (f.samples < 1000) return false;
        moving_fraction = f.moving / f.samples;
        if ((f.cc[0] + f.cc[1]) / f.samples < 1e-8) return false;  // camera nearly still: no information
        double scale[2], q = 1;
        for (int k = 0; k < 2; ++k) {
            if (f.gg[k] <= 0 || f.cc[k] <= 0) return false;
            scale[k] = f.gc[k] / f.gg[k];
            const double residual = scale[k] * scale[k] * f.gg[k] - 2 * scale[k] * f.gc[k] + f.cc[k];
            q = std::min(q, 1.0 - residual / f.cc[k]);
        }
        quality = q;
        if (q < 0.97) return false;
        const double px = scale[0] * render_w, py = scale[1] * render_h;
        const bool pixels = std::fabs(std::fabs(px) - 1) < 0.03 && std::fabs(std::fabs(py) - 1) < 0.03;
        double value[2] = {pixels ? (px > 0 ? 1.0 : -1.0) : scale[0], pixels ? (py > 0 ? 1.0 : -1.0) : scale[1]};
        // Other common conventions: motion vectors already in uv (scale 1, RE Engine) or in NDC (0.5).
        if (!pixels)
            for (const double unit : {1.0, 0.5})
                if (std::fabs(std::fabs(scale[0]) / unit - 1) < 0.05 && std::fabs(std::fabs(scale[1]) / unit - 1) < 0.05) {
                    value[0] = scale[0] > 0 ? unit : -unit;
                    value[1] = scale[1] > 0 ? unit : -unit;
                    break;
                }
        const bool same = candidate[0] != 0 && pixels == candidate_pixels &&
                          std::fabs(value[0] / candidate[0] - 1) < 0.05 && std::fabs(value[1] / candidate[1] - 1) < 0.05;
        agree = same ? agree + 1 : 1;
        if (!same) { candidate[0] = value[0]; candidate[1] = value[1]; candidate_pixels = pixels; }
        const bool differs = !valid || pixels != pixel_units || std::fabs(value[0] / locked[0] - 1) > 0.05 ||
                             std::fabs(value[1] / locked[1] - 1) > 0.05;
        if (differs && agree >= (valid ? 10 : 3)) {
            locked[0] = candidate[0]; locked[1] = candidate[1]; pixel_units = candidate_pixels; valid = true;
            return true;
        }
        return false;
    }
    double scale(int axis, double render_size) const { return pixel_units ? locked[axis] / render_size : locked[axis]; }
};

CameraBasis to_basis(const Camera& c) {
    auto v = [](const float* f) { return Vec3{f[0], f[1], f[2]}; };
    return {v(c.pos), normalize(v(c.right)), normalize(v(c.up)), normalize(v(c.fwd))};
}

void load_profile() {
    std::ifstream in(g_app.profile_path);
    AxisParams p[2];
    int version = 0;
    if (in >> version && version == 2 &&
        in >> p[0].gain >> p[0].tau >> p[0].delay >> p[1].gain >> p[1].tau >> p[1].delay) {
        p[0].fitted = p[1].fitted = true;
        g_app.model.seed(p[0], p[1]);
        logf("profile loaded: yaw gain %.4g tau %.0f ms, pitch gain %.4g tau %.0f ms", p[0].gain, p[0].tau * 1000, p[1].gain, p[1].tau * 1000);
    }
}
void save_profile() {
    const auto &x = g_app.model.params(0), &y = g_app.model.params(1);
    if (!x.fitted && !y.fitted) return;
    std::filesystem::create_directories(g_app.profile_path.parent_path());
    std::ofstream out(g_app.profile_path);
    out << 2 << ' ' << x.gain << ' ' << x.tau << ' ' << x.delay << ' ' << y.gain << ' ' << y.tau << ' ' << y.delay << '\n';
}

// Predicts vblanks from DXGI frame statistics of our own swapchain (the display it is shown on).
struct VblankClock {
    std::int64_t ref_qpc = 0;
    UINT ref_count = 0;
    double period = 0;  // QPC ticks per refresh
    void update(const Renderer::PresentStats& st) {
        if (st.hr != S_OK || !st.sync_qpc || !st.sync_refresh) return;
        if (!ref_qpc || st.sync_refresh < ref_count) { ref_qpc = st.sync_qpc; ref_count = st.sync_refresh; return; }
        const UINT refreshes = st.sync_refresh - ref_count;
        if (refreshes >= 60) {  // measure over >= half a second, then move the reference forward
            const double p = double(st.sync_qpc - ref_qpc) / double(refreshes);
            // Stale statistics (e.g. while the window was hidden) can pair an old timestamp with a new
            // refresh count: ignore measurements that disagree with the estimate by more than 2%.
            if (p > 0 && p < 0.1 * g_qpc_frequency && (period <= 0 || std::fabs(p / period - 1.0) < 0.02))
                period = period > 0 ? period + (p - period) * 0.2 : p;
            ref_qpc = st.sync_qpc; ref_count = st.sync_refresh;
        } else if (period <= 0 && refreshes >= 2) {
            period = double(st.sync_qpc - ref_qpc) / double(refreshes);
        }
    }
    bool valid() const { return ref_qpc && period > 0; }
    std::int64_t next_after(std::int64_t t) const {
        const double k = std::ceil(double(t - ref_qpc) / period);
        return ref_qpc + static_cast<std::int64_t>(k * period);
    }
};

// Sleeps until an absolute QPC time with a high-resolution waitable timer (+ short spin).
void sleep_until(std::int64_t target) {
    static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    const std::int64_t spin = g_qpc_frequency / 2000;  // last 0.5 ms spun for precision
    const std::int64_t now = qpc_now();
    if (target - now > spin && timer) {
        LARGE_INTEGER due;
        due.QuadPart = -static_cast<LONGLONG>(double(target - now - spin) * 1e7 / double(g_qpc_frequency));
        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, 100);
    }
    while (qpc_now() < target) YieldProcessor();
}

// GPU scheduler priority class for the whole process, from the UI (applied live).
std::uint32_t g_applied_priority = 0xFFFFFFFF;
void apply_gpu_priority(std::uint32_t setting) {
    if (setting == g_applied_priority) return;
    g_applied_priority = setting;
    using SetClassFn = LONG(APIENTRY*)(HANDLE, int);
    auto fn = reinterpret_cast<SetClassFn>(GetProcAddress(GetModuleHandleW(L"gdi32.dll"), "D3DKMTSetProcessSchedulingPriorityClass"));
    const int wanted = setting == 1 ? 4 : setting == 2 ? 2 : 5;  // D3DKMT class: 2 normal, 4 high, 5 realtime
    const char* result = "unavailable";
    if (fn) {
        if (fn(GetCurrentProcess(), wanted) == 0) result = wanted == 5 ? "realtime" : wanted == 4 ? "high" : "normal";
        else if (fn(GetCurrentProcess(), 4) == 0) result = "high (fallback)";
        else result = "normal (refused)";
    }
    logf("GPU scheduling priority class: %s", result);
    std::snprintf(g_gpu_priority, sizeof(g_gpu_priority), "%s", result);
}

// Hardware-accelerated GPU scheduling actually active on this adapter (not just the setting, which only
// applies after a reboot): 2 on, 1 off, 0 unknown. Without it the realtime priority cannot preempt the
// game's GPU work well and output falls below the refresh rate.
std::uint32_t hardware_scheduling(const LUID& adapter) {
    HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
    auto open = reinterpret_cast<PFND3DKMT_OPENADAPTERFROMLUID>(GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid"));
    auto query = reinterpret_cast<PFND3DKMT_QUERYADAPTERINFO>(GetProcAddress(gdi, "D3DKMTQueryAdapterInfo"));
    auto close = reinterpret_cast<PFND3DKMT_CLOSEADAPTER>(GetProcAddress(gdi, "D3DKMTCloseAdapter"));
    if (!open || !query || !close) return 0;
    D3DKMT_OPENADAPTERFROMLUID opened{};
    opened.AdapterLuid = adapter;
    if (open(&opened) != 0) return 0;
    D3DKMT_WDDM_2_7_CAPS caps{};
    D3DKMT_QUERYADAPTERINFO info{};
    info.hAdapter = opened.hAdapter;
    info.Type = KMTQAITYPE_WDDM_2_7_CAPS;
    info.pPrivateDriverData = &caps;
    info.PrivateDriverDataSize = sizeof(caps);
    const LONG result = query(&info);
    D3DKMT_CLOSEADAPTER closing{};
    closing.hAdapter = opened.hAdapter;
    close(&closing);
    if (result != 0) return 0;
    return caps.HwSchEnabled ? 2u : 1u;
}

void render_thread() {
    auto& sh = *g_app.shared;
    Renderer renderer;
    std::string error;
    const std::uint32_t w = sh.backbuffer_width, h = sh.backbuffer_height;
    if (!renderer.init(sh.adapter, g_app.overlay, w, h, static_cast<DXGI_FORMAT>(sh.backbuffer_format), sh.color_space, error)) {
        set_status("Renderer init failed: %s", error.c_str()); logf("renderer init failed: %s", error.c_str());
        g_app.running = false; PostMessageW(g_app.overlay, WM_CLOSE, 0, 0); return;
    }
    logf("renderer ready %ux%u, queue priority %s", w, h, renderer.queue_priority());
    VramMonitor vram;
    vram.init(sh.adapter);
    vram.poll(now_seconds(), "start");
    Latewarp12 latewarp;
    if (!latewarp.initialize(renderer.device(), g_app.base_dir, g_app.base_dir / L"logs"))
        logf("latewarp unavailable: %s", latewarp.status().c_str());
    else
        logf("latewarp ready");

    struct Held { int slot; std::uint64_t release_value; };
    std::vector<Held> held;
    IngestedSource source;
    Camera source_camera{};
    CameraBasis source_basis{};
    double source_time = 0;
    std::uint64_t source_frame = 0;
    bool first_eval = false;
    std::uint32_t last_reset = sh.settings.reset_calibration;
    double stat_start = now_seconds(), last_source_present = 0, last_profile_save = stat_start;
    std::uint32_t stat_frames = 0, stat_sources = 0;
    double source_interval = 0;
    VblankClock vblank;
    std::int64_t wake_qpc = 0;
    std::int64_t last_target_vblank = 0;
    bool pacing_paused = false;
    bool device_loss_logged = false;
    MotionVectorScale mv_scale;
    double last_mv_log = 0;
    bool game_has_hud_layers = false, mask_logged = false;
    CameraEstimator estimator;
    bool estimator_logged = false, fov_logged = false;
    std::vector<float> raw_samples;
    std::vector<MotionSample> motion_samples;

    while (g_app.running) {
        WaitForSingleObjectEx(renderer.waitable(), 100, FALSE);
        if (!g_app.running) break;
        const Settings settings = sh.settings;
        apply_gpu_priority(settings.gpu_priority);
        // DWM composes shortly after each vblank; a frame that is not finished by then waits a
        // whole refresh. Render `lead` ms before the next vblank so the GPU has room even when the
        // game delays our work.
        vblank.update(renderer.present_stats());
        // DWM composes ~3 ms after each vblank and shows the result at the following vblank; with a
        // frame latency of 1 the swapchain only frees us at that display vblank, leaving ~2.5 ms. With a
        // lead we allow one queued frame and wake `lead` ms before each composition instead, one frame
        // per refresh (a new target is always a later vblank than the previous one).
        const bool paced = settings.present_lead_ms > 0 && vblank.valid();
        renderer.set_frame_latency(paced ? 2 : 1);
        if (paced) {
            const double f = double(g_qpc_frequency);
            const std::int64_t lead = static_cast<std::int64_t>(settings.present_lead_ms * 1e-3 * f);
            const std::int64_t compose = static_cast<std::int64_t>(2.5e-3 * f);  // composition deadline after a vblank
            const std::int64_t now_q = qpc_now();
            std::int64_t v = vblank.next_after(now_q - compose + lead);  // first vblank whose deadline - lead is ahead
            if (last_target_vblank && v <= last_target_vblank + static_cast<std::int64_t>(vblank.period / 2))
                v = last_target_vblank + static_cast<std::int64_t>(vblank.period);
            last_target_vblank = v;
            sleep_until(v + compose - lead);
        } else {
            last_target_vblank = 0;
        }
        wake_qpc = qpc_now();

        if (!renderer.session_open(sh.session)) {
            held.clear(); source = {}; source_frame = 0;
            if (!renderer.open_session(sh.producer_pid, sh.session, error)) {
                set_status("Waiting for game: %s", error.c_str());
                Sleep(100); continue;
            }
            logf("session %llu opened", static_cast<unsigned long long>(sh.session));
        }
        if (sh.backbuffer_width != renderer.width() || sh.backbuffer_height != renderer.height())
            renderer.resize(sh.backbuffer_width, sh.backbuffer_height);
        {
            // The overlay window should cover exactly the game's frame; a mismatch (display scaling,
            // windowed modes) would crop or offset the picture, so it is logged when it changes.
            static std::uint32_t logged_w = 0, logged_h = 0;
            const std::uint32_t cw = g_app.client_w, ch = g_app.client_h;
            if (cw && ch && (cw != logged_w || ch != logged_h)) {
                logf("overlay window %ux%u, game frame %ux%u%s", cw, ch, sh.backbuffer_width, sh.backbuffer_height,
                     cw == sh.backbuffer_width && ch == sh.backbuffer_height ? "" : " (MISMATCH)");
                logged_w = cw; logged_h = ch;
            }
        }

        PoseSettings ps;
        ps.use_mouse = settings.use_mouse != 0;
        ps.rotation_extrapolation = settings.rotation_extrapolation;
        ps.prediction = settings.prediction_ms / 1000.0;
        // Auto: half a game frame, or a quarter when the game sends no HUD layers - the HUD/weapon mask
        // misses some pixels (semi-transparent HUD), and a shorter warp moves them less.
        const std::uint32_t fraction = settings.auto_prediction == 3 ? (game_has_hud_layers ? 2u : 4u) : settings.auto_prediction;
        ps.auto_fraction = (fraction == 1 || fraction == 2 || fraction == 4) ? 1.0 / fraction : 0.0;
        ps.orbit_distance = settings.orbit_distance;
        ps.max_horizon = settings.max_horizon_ms / 1000.0;
        ps.manual_gain = settings.manual_gain != 0;
        ps.manual_gain_x = settings.manual_gain_x; ps.manual_gain_y = settings.manual_gain_y;
        ps.manual_delay = settings.manual_delay_ms / 1000.0;
        g_app.model.configure(ps);
        CURSORINFO cursor{sizeof(cursor)};
        const bool cursor_visible = GetCursorInfo(&cursor) && (cursor.flags & CURSOR_SHOWING);
        g_app.model.set_mouse_gate(!cursor_visible);
        if (settings.reset_calibration != last_reset) {
            last_reset = settings.reset_calibration;
            g_app.model.reset_fit();
            std::error_code ec; std::filesystem::remove(g_app.profile_path, ec);
            logf("calibration reset");
        }

        // Return slots whose conversion has finished on the GPU.
        for (auto it = held.begin(); it != held.end();) {
            if (renderer.completed(it->release_value)) {
                InterlockedCompareExchange(&sh.slots[it->slot].state, kFree, kReading);
                it = held.erase(it);
            } else ++it;
        }

        if (!g_app.visible) {  // game not in front / disabled: don't compete with it for the GPU
            source_frame = sh.latest_ready_frame;
            pacing_paused = true;
            Sleep(10);
            continue;
        }
        if (pacing_paused) {
            // Back from a pause (alt-tab, hidden overlay): the frame statistics and our vblank schedule are
            // stale. Start over exactly like switching the present lead off and on again.
            pacing_paused = false;
            vblank = VblankClock{};
            last_target_vblank = 0;
            renderer.set_frame_latency(1);
            logf("pacing re-synchronised after a pause");
        }

        // Newest published frame whose GPU work is already complete.
        const std::uint64_t game_done = renderer.game_fence_completed();
        int newest = -1;
        for (int i = 0; i < kSlots; ++i) {
            const auto& m = sh.slots[i];
            if (m.state == kReady && m.frame_id > source_frame && m.camera.valid && m.fence_value <= game_done &&
                (newest < 0 || m.frame_id > sh.slots[newest].frame_id)) newest = i;
        }

        auto* list = renderer.begin_frame();
        if (newest >= 0 && InterlockedCompareExchange(&sh.slots[newest].state, kReading, kReady) == kReady) {
            const SlotMeta& m = sh.slots[newest];
            // The previous frame's colour feeds the HUD detector (and the shelved object interpolation).
            renderer.set_keep_previous_colour(settings.extrapolate_objects != 0 || (settings.no_warp_mask != 0 && !game_has_hud_layers));
            IngestedSource s = renderer.ingest(sh, newest);
            held.push_back({newest, renderer.submitted_value() + 1});
            if (s.valid) {
                // Games without a camera (DLSS without Streamline): estimate it from the motion vectors.
                Camera cam = m.camera;
                if (cam.estimated && s.has_depth && s.has_motion && s.depth_rect.w && s.depth_rect.h) {
                    constexpr std::uint32_t kGridW = 64, kGridH = 36;
                    if (renderer.sample_motion(s, kGridW, kGridH, raw_samples)) {
                        const double rw = s.depth_rect.w, rh = s.depth_rect.h;
                        motion_samples.clear();
                        for (std::uint32_t gy = 0; gy < kGridH; ++gy)
                            for (std::uint32_t gx = 0; gx < kGridW; ++gx) {
                                const float* v = &raw_samples[(gy * kGridW + gx) * 4];
                                MotionSample p{};
                                p.x = float((std::floor((gx + 0.5) * rw / kGridW)) + 0.5);
                                p.y = float((std::floor((gy + 0.5) * rh / kGridH)) + 0.5);
                                p.mx = v[0] * m.camera.mvec_scale[0]; p.my = v[1] * m.camera.mvec_scale[1];
                                p.depth = v[2];
                                p.valid = (v[3] > 0.5f && std::isfinite(p.mx) && std::isfinite(p.my) && std::fabs(p.mx) < 1e4f) ? 1.0f : 0.0f;
                                motion_samples.push_back(p);
                            }
                        cam = estimator.update(motion_samples, rw, rh, m.camera);
                        if (!estimator_logged) { logf("no camera from the game: estimating it from DLSS motion vectors"); estimator_logged = true; }
                        if (estimator.fov_locked() && !fov_logged) {
                            logf("field of view learned: %.1f deg vertical", estimator.vertical_fov() * 180.0 / 3.14159265358979);
                            fov_logged = true;
                        }
                    }
                }
                source = s;
                source_camera = cam;
                source_basis = to_basis(cam);
                source_time = seconds(m.qpc_sim_start ? m.qpc_sim_start : m.qpc_constants);
                source_frame = m.frame_id;
                g_app.model.add_source(m.frame_id, source_time, seconds(qpc_now()), source_basis, cam.reset != 0, now_seconds(),
                                       cam.position_epoch);
                if (g_app.csv_sources) {
                    const auto& c = cam;
                    std::fprintf(g_app.csv_sources,
                                 "%llu,%lld,%lld,%lld,%lld,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.5f,%.5f,%u,%d,%d,%d",
                                 static_cast<unsigned long long>(m.frame_id), m.qpc_sim_start, m.qpc_constants, m.qpc_present, qpc_now(),
                                 c.pos[0], c.pos[1], c.pos[2], c.fwd[0], c.fwd[1], c.fwd[2], c.up[0], c.up[1], c.up[2],
                                 c.right[0], c.right[1], c.right[2], c.fov, c.aspect, c.reset, int(g_app.model.mouse_gate()),
                                 int(s.has_depth), int(s.has_hudless));
                    // Projection and the game's frame-to-frame reprojection (same line, appended).
                    for (float v : c.view_to_clip) std::fprintf(g_app.csv_sources, ",%.7g", v);
                    for (float v : c.clip_to_prev_clip) std::fprintf(g_app.csv_sources, ",%.7g", v);
                    std::fputc('\n', g_app.csv_sources);
                }
                game_has_hud_layers = s.has_hudless && s.has_ui && settings.use_ui_tags;
                const bool mask = settings.no_warp_mask && !game_has_hud_layers && s.has_depth;
                if ((settings.extrapolate_objects || mask) && s.has_motion)
                    renderer.analyze_motion(s, cam.clip_to_prev_clip, float(mv_scale.scale(0, s.depth_rect.w)),
                                            float(mv_scale.scale(1, s.depth_rect.h)), mv_scale.valid, cam.depth_inverted != 0);
                if (mask) {
                    renderer.build_no_warp_mask(s, cam.clip_to_prev_clip, true, s.has_motion && mv_scale.valid, cam.depth_inverted != 0);
                    if (!mask_logged) { logf("no HUD layers from the game: detecting HUD and first-person weapon for the no-warp mask"); mask_logged = true; }
                }
                first_eval = true;
                const double present_t = seconds(m.qpc_present);
                if (last_source_present > 0) source_interval = present_t - last_source_present;
                last_source_present = present_t;
                ++stat_sources;
            }
        }

        const double now = now_seconds();
        bool warped = false;
        Prediction applied{};
        if (source.valid && settings.enabled && !settings.show_original && latewarp.ready() && source.has_depth) {
            Prediction p = g_app.model.predict(now);
            if (settings.invert_warp) p.camera = apply_rotation(source_basis, g_app.model.world_up(), -p.yaw, -p.pitch, source_basis.pos);
            applied = p;
            const Vec3 origin = source_basis.pos;
            Mat4 projection;
            std::memcpy(projection.data(), source_camera.view_to_clip, sizeof(projection));
            auto inputs = renderer.latewarp_inputs(source, settings.use_ui_tags != 0);
            inputs.depth_inverted = source_camera.depth_inverted != 0;
            if (settings.no_warp_mask && !game_has_hud_layers) {
                inputs.no_warp_mask = renderer.no_warp_mask();
                inputs.mask_rect = source.color_rect;
            }
            if (settings.extrapolate_objects && source.has_motion && mv_scale.valid) {
                const double interval = g_app.model.frame_interval();
                // Interpolation only: between the object's exact previous and current positions (from the
                // motion vectors). When the camera is shown ahead of the newest frame, objects hold there.
                const float alpha = interval > 0 ? float(std::clamp(p.horizon / interval, -1.0, 0.0)) : 0.0f;
                const bool split = inputs.hudless != inputs.backbuffer;
                if (ID3D12Resource* moved = alpha < 0 ? renderer.extrapolate_objects(source, split, alpha, source_camera.clip_to_prev_clip) : nullptr) {
                    inputs.hudless = moved;
                    if (!split) inputs.backbuffer = moved;
                }
            }
            const double z_sign = view_z_sign(source_camera.view_to_clip);
            warped = latewarp.evaluate(list, inputs, first_eval, view_matrix(p.camera, origin, z_sign),
                                       view_matrix(source_basis, origin, z_sign), projection);
            if (warped) first_eval = false;
        }
        g_app.has_frames = source.valid;
        renderer.finish_frame(warped, settings.overlay_debug ? (warped ? 1 : 2) : 0);
        // Ctrl+Shift+M marks "it looks bad now" in the log.
        const bool mark_down = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) && (GetAsyncKeyState('M') & 0x8000);
        const int mark = mark_down && !g_app.mark_was_down ? 1 : 0;
        g_app.mark_was_down = mark_down;
        if (mark) logf("MARK (user flagged a bad moment)");
        if (g_app.csv_outputs) {
            const auto tm = renderer.last_timing();
            const auto pstats = renderer.present_stats();
            std::fprintf(g_app.csv_outputs, "%lld,%llu,%d,%.6f,%.6f,%.4f,%.3f,%d,%lld,%lld,%lld,%u,%u,%u,%u,%lld,%ld,%lld\n", qpc_now(),
                         static_cast<unsigned long long>(source_frame), int(warped), applied.yaw, applied.pitch, applied.horizon * 1000.0,
                         renderer.last_gpu_ms(), mark, tm.submit, tm.gpu_start, tm.gpu_end, pstats.last_present_count, pstats.present_count,
                         pstats.present_refresh, pstats.sync_refresh, pstats.sync_qpc, static_cast<long>(pstats.hr), wake_qpc);
        }
        // Dump the game-side timeline.
        if (g_app.csv_events) {
            const std::int64_t count = sh.timeline_count;
            if (count - g_app.events_read > kTimeline) g_app.events_read = count - kTimeline;
            for (; g_app.events_read < count; ++g_app.events_read) {
                const auto& e = sh.timeline[g_app.events_read % kTimeline];
                std::fprintf(g_app.csv_events, "%lld,%u,%u,%llu,%llu\n", e.qpc, e.kind, e.tid,
                             static_cast<unsigned long long>(e.frame), static_cast<unsigned long long>(e.extra));
            }
        }
        ++stat_frames;

        {
            const auto notes = renderer.take_notes();
            for (const auto& note : notes) logf("%s", note.c_str());
            MotionFit fit;
            if (renderer.take_motion_fit(fit) && mv_scale.add(fit, source.depth_rect.w, source.depth_rect.h))
                logf("motion vectors locked: %s, scale %.4g x %.4g (sign %+.0f %+.0f), fit quality %.3f",
                     mv_scale.pixel_units ? "render pixels" : std::fabs(std::fabs(mv_scale.locked[0]) - 1) < 1e-9 ? "uv" :
                     std::fabs(std::fabs(mv_scale.locked[0]) - 0.5) < 1e-9 ? "ndc" : "custom units", mv_scale.scale(0, source.depth_rect.w),
                     mv_scale.scale(1, source.depth_rect.h), mv_scale.locked[0] < 0 ? -1.0 : 1.0, mv_scale.locked[1] < 0 ? -1.0 : 1.0,
                     mv_scale.quality);
            vram.poll(now, notes.empty() ? nullptr : "texture change");
        }
        if (now - stat_start >= 0.5) {
            const HRESULT removed = renderer.device_removed_reason();
            if (removed != S_OK && !device_loss_logged) {
                logf("GPU DEVICE LOST: reason 0x%08lX - presenter exits; use 'Start presenter' in the ReShade panel",
                     static_cast<unsigned long>(removed));
                device_loss_logged = true;
                // Nothing on a lost device ever completes: release the game's frames and leave without
                // touching the GPU again (device/NGX teardown can block on a lost device).
                for (const auto& entry : held) InterlockedCompareExchange(&sh.slots[entry.slot].state, kFree, kReading);
                sh.presenter.pid = 0;
                if (g_app.log) std::fflush(g_app.log);
                for (FILE* csv : {g_app.csv_events, g_app.csv_mouse, g_app.csv_sources, g_app.csv_outputs}) if (csv) std::fflush(csv);
                ExitProcess(0);
            }
            auto& st = sh.presenter;
            st.output_fps = float(stat_frames / (now - stat_start));
            st.source_fps = source_interval > 0 ? float(1.0 / source_interval) : 0.0f;
            st.warp_gpu_ms = renderer.last_gpu_ms();
            st.source_age_ms = source.valid ? float((now - source_time) * 1000.0) : 0.0f;
            const auto &px = g_app.model.params(0), &py = g_app.model.params(1);
            st.gain_x = float(g_app.model.gain(0)); st.gain_y = float(g_app.model.gain(1));
            st.delay_ms = float(px.delay * 1000.0);
            st.tau_x_ms = float(px.tau * 1000.0); st.tau_y_ms = float(py.tau * 1000.0);
            st.latency_ms = float(g_app.model.latency() * 1000.0);
            st.orbit_cm = float(g_app.model.learned_orbit());
            st.frame_interval_ms = float(g_app.model.frame_interval() * 1000.0);
            st.effective_prediction_ms = float(g_app.model.effective_prediction() * 1000.0);
            st.mv_scale_x = mv_scale.valid ? float(mv_scale.scale(0, source.depth_rect.w)) : 0.0f;
            st.mv_scale_y = mv_scale.valid ? float(mv_scale.scale(1, source.depth_rect.h)) : 0.0f;
            st.mv_fit_quality = float(mv_scale.quality);
            st.moving_fraction = float(mv_scale.moving_fraction);
            if (settings.extrapolate_objects && mv_scale.valid && now - last_mv_log > 10.0) {
                logf("objects: motion vectors %s, last fit quality %.3f, moving pixels %.1f%%",
                     mv_scale.pixel_units ? "in render pixels" : "in custom units", mv_scale.quality, mv_scale.moving_fraction * 100.0);
                last_mv_log = now;
            }
            {
                LARGE_INTEGER qf; QueryPerformanceFrequency(&qf);
                st.display_hz = vblank.period > 0 ? float(double(qf.QuadPart) / vblank.period) : 0.0f;
            }
            st.fit_quality_x = float(px.quality); st.fit_quality_y = float(py.quality);
            st.calibrated_x = px.fitted; st.calibrated_y = py.fitted;
            st.frames_presented += stat_frames; st.sources_consumed += stat_sources;
            st.heartbeat_qpc = qpc_now();
            if (!source.valid) set_status("Waiting for frames (need Streamline tags + camera constants)");
            else if (!latewarp.ready()) set_status("Latewarp unavailable: %s", latewarp.status().c_str());
            else if (!source.has_depth) set_status("No depth tag from the game: showing unwarped frames");
            else set_status("%s | queue %s, process %s | HUD split %s | mouse %s | %s", warped ? "warping" : "passthrough", renderer.queue_priority(), g_gpu_priority,
                            (source.has_hudless && source.has_ui) ? "yes" : "no",
                            g_app.model.mouse_gate() ? "camera" : "cursor visible (ignored)", latewarp.status().c_str());
            stat_frames = stat_sources = 0; stat_start = now;
            if (now - last_profile_save > 10.0) { save_profile(); last_profile_save = now; }
        }
    }
    renderer.wait_idle();
    for (const auto& entry : held) InterlockedCompareExchange(&sh.slots[entry.slot].state, kFree, kReading);
    save_profile();
    latewarp.shutdown();
}

void follow_game_window() {
    HWND game = g_app.game;
    const bool alive = IsWindow(game) != FALSE;
    RECT rc{};
    POINT origin{0, 0};
    bool show = alive && !IsIconic(game) && g_app.shared->settings.enabled && GetClientRect(game, &rc) && ClientToScreen(game, &origin);
    const HWND fg = GetForegroundWindow();
    show = show && (fg == game || fg == g_app.overlay);
    g_app.visible = show;
    // Never cover the game until we actually have frames to show (avoids a black overlay).
    if (show && g_app.has_frames) {
        const int w = rc.right - rc.left, h = rc.bottom - rc.top;
        SetWindowPos(g_app.overlay, HWND_TOPMOST, origin.x, origin.y, w, h, SWP_NOACTIVATE | SWP_SHOWWINDOW);
        g_app.client_w = static_cast<std::uint32_t>(w); g_app.client_h = static_cast<std::uint32_t>(h);
    } else if (IsWindowVisible(g_app.overlay)) {
        ShowWindow(g_app.overlay, SW_HIDE);
    }
}

LRESULT CALLBACK window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_INPUT: {
            RAWINPUT raw{};
            UINT size = sizeof(raw);
            if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lp), RID_INPUT, &raw, &size, sizeof(RAWINPUTHEADER)) != UINT(-1) &&
                raw.header.dwType == RIM_TYPEMOUSE && !(raw.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE) &&
                (raw.data.mouse.lLastX || raw.data.mouse.lLastY))
            {
                const std::int64_t q = qpc_now();
                g_app.model.mouse.add(seconds(q), raw.data.mouse.lLastX, raw.data.mouse.lLastY);
                if (g_app.csv_mouse) std::fprintf(g_app.csv_mouse, "%lld,%ld,%ld\n", q, raw.data.mouse.lLastX, raw.data.mouse.lLastY);
            }
            break;  // DefWindowProc must still run for WM_INPUT cleanup
        }
        case WM_TIMER:
            if (!g_app.running || (g_app.game_process && WaitForSingleObject(g_app.game_process, 0) == WAIT_OBJECT_0) ||
                g_app.shared->magic != kMagic) {
                g_app.running = false;
                DestroyWindow(hwnd);
                return 0;
            }
            follow_game_window();
            return 0;
        case WM_MOUSEACTIVATE: return MA_NOACTIVATE;
        case WM_NCHITTEST: return HTTRANSPARENT;
        case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

bool open_shared(DWORD pid) {
    for (int attempt = 0; attempt < 300; ++attempt) {
        g_app.mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, map_name(pid).c_str());
        if (g_app.mapping) break;
        Sleep(100);
    }
    if (!g_app.mapping) return false;
    g_app.shared = static_cast<Shared*>(MapViewOfFile(g_app.mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Shared)));
    if (!g_app.shared || g_app.shared->magic != kMagic || g_app.shared->version != kVersion) return false;
    for (int attempt = 0; attempt < 600 && (!g_app.shared->game_hwnd || !g_app.shared->backbuffer_width); ++attempt) Sleep(100);
    return g_app.shared->game_hwnd && g_app.shared->backbuffer_width;
}

}  // namespace
}  // namespace fw

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    using namespace fw;
    LARGE_INTEGER f; QueryPerformanceFrequency(&f); g_qpc_to_seconds = 1.0 / double(f.QuadPart); g_qpc_frequency = f.QuadPart;
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (int i = 1; i + 1 < argc; ++i)
        if (std::wstring(argv[i]) == L"--pid") g_app.pid = static_cast<DWORD>(std::wcstoul(argv[i + 1], nullptr, 10));
    wchar_t exe[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    g_app.base_dir = std::filesystem::path(exe).parent_path();
    std::filesystem::create_directories(g_app.base_dir / L"logs");
    // Keep the previous run's logs: after a crash the user restarts the presenter, and that run is the one we need.
    {
        std::error_code ec;
        const auto previous = g_app.base_dir / L"logs" / L"previous";
        std::filesystem::remove_all(previous, ec);
        std::filesystem::create_directories(previous, ec);
        for (const auto& entry : std::filesystem::directory_iterator(g_app.base_dir / L"logs", ec))
            if (entry.is_regular_file(ec)) std::filesystem::rename(entry.path(), previous / entry.path().filename(), ec);
    }
    g_app.log = _wfsopen((g_app.base_dir / L"logs" / L"presenter.log").c_str(), L"w", _SH_DENYNO);
    auto open_csv = [](const wchar_t* name, const char* header) {
        FILE* f = _wfsopen((g_app.base_dir / L"logs" / name).c_str(), L"w", _SH_DENYNO);
        if (f) { std::setvbuf(f, nullptr, _IOFBF, 1 << 16); std::fprintf(f, "%s\n", header); }
        return f;
    };
    g_app.csv_events = open_csv(L"events.csv", "qpc,kind,tid,frame,extra");
    g_app.csv_mouse = open_csv(L"mouse.csv", "qpc,dx,dy");
    g_app.csv_sources = open_csv(L"sources.csv",
        "frame,qpc_sim,qpc_constants,qpc_present,qpc_ingest,px,py,pz,fx,fy,fz,ux,uy,uz,rx,ry,rz,fov,aspect,reset,mouse_gate,has_depth,has_hudless,"
        "p00,p01,p02,p03,p10,p11,p12,p13,p20,p21,p22,p23,p30,p31,p32,p33,"
        "c00,c01,c02,c03,c10,c11,c12,c13,c20,c21,c22,c23,c30,c31,c32,c33");
    g_app.csv_outputs = open_csv(L"outputs.csv",
        "qpc,source_frame,warped,yaw,pitch,horizon_ms,gpu_ms,mark,t_submit,t_gpu_start,t_gpu_end,last_present,present_count,present_refresh,sync_refresh,sync_qpc,stats_hr,t_wake");
    {
        LARGE_INTEGER fq; QueryPerformanceFrequency(&fq);
        logf("FrameWarp presenter " FW_VERSION ", qpc frequency %lld", fq.QuadPart);
    }
    if (!g_app.pid) { logf("usage: FrameWarpPresenter --pid <game pid>"); return 1; }

    // One presenter per game.
    const std::wstring mutex_name = L"Local\\FrameWarpPresenter_" + std::to_wstring(g_app.pid);
    HANDLE single = CreateMutexW(nullptr, TRUE, mutex_name.c_str());
    if (GetLastError() == ERROR_ALREADY_EXISTS) { logf("presenter already running"); return 0; }

    g_app.game_process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, g_app.pid);
    wchar_t game_exe[MAX_PATH]{}; DWORD len = MAX_PATH;
    if (g_app.game_process && QueryFullProcessImageNameW(g_app.game_process, 0, game_exe, &len))
        g_app.profile_path = g_app.base_dir / L"profiles" / (std::filesystem::path(game_exe).stem().wstring() + L".txt");
    else
        g_app.profile_path = g_app.base_dir / L"profiles" / L"default.txt";
    if (!open_shared(g_app.pid)) { logf("shared memory from the game add-on not found"); return 1; }
    g_app.shared->presenter.pid = static_cast<LONG>(GetCurrentProcessId());
    g_app.game = reinterpret_cast<HWND>(g_app.shared->game_hwnd);
    load_profile();
    logf("attached to pid %lu, backbuffer %ux%u fmt %u cs %u", g_app.pid, g_app.shared->backbuffer_width,
         g_app.shared->backbuffer_height, g_app.shared->backbuffer_format, g_app.shared->color_space);

    // Render thread priority matters as much as GPU priority for hitting every vblank.
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    apply_gpu_priority(g_app.shared->settings.gpu_priority);
    {
        const std::uint32_t hags = hardware_scheduling(g_app.shared->adapter);
        g_app.shared->presenter.hardware_scheduling = hags;
        logf("hardware-accelerated GPU scheduling: %s", hags == 2 ? "on" : hags == 1 ? "OFF (output may not reach the refresh rate)" : "unknown");
    }

    // Real pixels everywhere. Without this, on a display with Windows scaling (laptops: 125-150%) the game
    // window's size reads scaled (2560x1440 for a 4K panel at 150%) and the overlay showed only the
    // top-left part of the frame, enlarged.
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    WNDCLASSW wc{};
    wc.lpfnWndProc = window_proc; wc.hInstance = instance; wc.lpszClassName = L"FrameWarpOverlay";
    wc.hCursor = nullptr;
    RegisterClassW(&wc);
    // Click-through, never activated, no redirection surface (content comes from DirectComposition).
    g_app.overlay = CreateWindowExW(WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW |
                                        WS_EX_NOREDIRECTIONBITMAP,
                                    wc.lpszClassName, L"FrameWarp", WS_POPUP, 0, 0, 16, 16, nullptr, nullptr, instance, nullptr);
    SetLayeredWindowAttributes(g_app.overlay, 0, 255, LWA_ALPHA);
    RAWINPUTDEVICE rid{0x01, 0x02, RIDEV_INPUTSINK, g_app.overlay};  // generic desktop / mouse
    if (!RegisterRawInputDevices(&rid, 1, sizeof(rid))) logf("raw input registration failed");
    SetTimer(g_app.overlay, 1, 100, nullptr);
    follow_game_window();

    std::thread renderer(render_thread);
    SetThreadPriority(renderer.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    g_app.running = false;
    renderer.join();
    if (g_app.shared) g_app.shared->presenter.pid = 0;
    logf("exit");
    for (FILE* csv : {g_app.csv_events, g_app.csv_mouse, g_app.csv_sources, g_app.csv_outputs}) if (csv) std::fclose(csv);
    if (g_app.log) std::fclose(g_app.log);
    if (single) CloseHandle(single);
    return 0;
}
