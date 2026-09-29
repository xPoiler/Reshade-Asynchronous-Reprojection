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
#include <condition_variable>
#include <cstdarg>
#include <mutex>
#include <deque>
#include <cstring>
#include <share.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
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
    std::filesystem::path data_dir;  // logs, calibration, captures: base_dir, or %LOCALAPPDATA%\FrameWarp\<game folder> when that is read-only
    FILE* log = nullptr;
    // Analysis logs (all times are raw QPC ticks, same clock in both processes).
    FILE* csv_events = nullptr;   // game-side timeline
    FILE* csv_mouse = nullptr;    // raw input
    FILE* csv_motion = nullptr;   // per game frame: agreement of the motion vectors with depth + camera
    FILE* csv_sources = nullptr;  // ingested frames + camera
    FILE* csv_outputs = nullptr;  // presented frames + applied warp
    std::int64_t events_read = 0;
    bool mark_was_down = false, dump_was_down = false;
};
App g_app;

// Files (the log and the recordings) are written by a background thread. A disk write can take several
// ms (a slow or busy drive, antivirus), and the refresh loop must never wait for one: written in place, a
// full 64 KB recording buffer went to disk on the refresh thread about once a second and cost a refresh
// each time. Text is handed over here and written out a few times a second.
class FileWriter {
public:
    void write(FILE* f, const char* text, std::size_t n) {
        if (!f || !n) return;
        std::lock_guard lock(mutex_);
        for (auto& p : pending_) if (p.first == f) { p.second.append(text, n); return; }
        pending_.emplace_back(f, std::string(text, n));
    }
    void vprintf(FILE* f, const char* format, va_list args) {
        char buffer[1024];
        const int n = std::vsnprintf(buffer, sizeof(buffer), format, args);
        if (n > 0) write(f, buffer, std::min<std::size_t>(std::size_t(n), sizeof(buffer) - 1));
    }
    // A whole small file to (re)write (the camera profile); the newest text for a path wins.
    void replace(const std::filesystem::path& path, std::string text) {
        std::lock_guard lock(mutex_);
        for (auto& r : replace_) if (r.first == path) { r.second = std::move(text); return; }
        replace_.emplace_back(path, std::move(text));
    }
    // Writes everything handed over so far, on the calling thread (exit, device loss).
    void flush_now() {
        std::lock_guard io(io_mutex_);
        std::vector<std::pair<FILE*, std::string>> batch;
        std::vector<std::pair<std::filesystem::path, std::string>> files;
        { std::lock_guard lock(mutex_); batch.swap(pending_); files.swap(replace_); }
        for (auto& [f, text] : batch) { std::fwrite(text.data(), 1, text.size(), f); std::fflush(f); }
        for (auto& [path, text] : files) {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            std::ofstream out(path, std::ios::binary);
            out << text;
        }
    }
    void start() {
        if (thread_.joinable()) return;
        thread_ = std::thread([this] {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            for (;;) {
                {
                    std::unique_lock lock(stop_mutex_);
                    if (wake_.wait_for(lock, std::chrono::milliseconds(250), [this] { return stop_; })) return;
                }
                flush_now();
            }
        });
    }
    void stop() {
        if (!thread_.joinable()) return;
        { std::lock_guard lock(stop_mutex_); stop_ = true; }
        wake_.notify_one();
        thread_.join();
        flush_now();
    }
private:
    std::mutex mutex_, io_mutex_, stop_mutex_;
    std::condition_variable wake_;
    bool stop_ = false;
    std::vector<std::pair<FILE*, std::string>> pending_;
    std::vector<std::pair<std::filesystem::path, std::string>> replace_;
    std::thread thread_;
};
FileWriter g_files;
void wr(FILE* f, const char* format, ...) {
    if (!f) return;
    va_list args; va_start(args, format);
    g_files.vprintf(f, format, args);
    va_end(args);
}

// Detailed CSV recordings (opt-in): opened the first time they are switched on, written only while on,
// closed at exit (the window thread writes mouse input too).
std::atomic<bool> g_recording{false};
FILE* rec(FILE* f) { return g_recording.load(std::memory_order_acquire) ? f : nullptr; }
void open_recordings() {
    static bool opened = false;
    if (!opened) {
        opened = true;
        auto open_csv = [](const wchar_t* name, const char* header) {
            FILE* f = _wfsopen((g_app.data_dir / L"logs" / name).c_str(), L"w", _SH_DENYNO);
            if (f) { std::setvbuf(f, nullptr, _IOFBF, 1 << 16); wr(f, "%s\n", header); }
            return f;
        };
        g_app.csv_events = open_csv(L"events.csv", "qpc,kind,tid,frame,extra");
        g_app.csv_mouse = open_csv(L"mouse.csv", "qpc,dx,dy");
        g_app.csv_motion = open_csv(L"motion.csv", "qpc,frame,samples,moving_fraction,agree_x,agree_y,render_w,render_h");
        g_app.csv_sources = open_csv(L"sources.csv",
            "frame,qpc_sim,qpc_constants,qpc_present,qpc_ingest,px,py,pz,fx,fy,fz,ux,uy,uz,rx,ry,rz,fov,aspect,reset,mouse_gate,has_depth,has_hudless,"
            "p00,p01,p02,p03,p10,p11,p12,p13,p20,p21,p22,p23,p30,p31,p32,p33,"
            "c00,c01,c02,c03,c10,c11,c12,c13,c20,c21,c22,c23,c30,c31,c32,c33");
        g_app.csv_outputs = open_csv(L"outputs.csv",
            "qpc,source_frame,warped,yaw,pitch,horizon_ms,gpu_ms,mark,t_submit,t_gpu_start,t_gpu_end,last_present,present_count,present_refresh,"
            "sync_refresh,sync_qpc,stats_hr,t_wake,cam_x,cam_y,cam_z,cam_fx,cam_fy,cam_fz");
    }
    g_recording.store(true, std::memory_order_release);
}

void logf(const char* format, ...) {
    if (!g_app.log) return;
    char line[1100];
    int n = std::snprintf(line, 32, "[%.3f] ", now_seconds());
    va_list args; va_start(args, format);
    const int m = std::vsnprintf(line + n, sizeof(line) - n - 1, format, args);
    va_end(args);
    if (m > 0) n += std::min(m, int(sizeof(line)) - n - 2);
    line[n++] = '\n';
    g_files.write(g_app.log, line, std::size_t(n));
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
        // Compared in uv units: the same scale can come out as "render pixels" in one window and as a
        // slightly different custom value in the next; only a real change (> 5%) relocks, or an exact
        // render-pixel match replacing a custom value.
        const double uv_new[2] = {pixels ? value[0] / render_w : value[0], pixels ? value[1] / render_h : value[1]};
        const double uv_old[2] = {pixel_units ? locked[0] / render_w : locked[0], pixel_units ? locked[1] / render_h : locked[1]};
        const bool differs = !valid || (pixels && !pixel_units) || std::fabs(uv_new[0] / uv_old[0] - 1) > 0.05 ||
                             std::fabs(uv_new[1] / uv_old[1] - 1) > 0.05;
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
    std::ostringstream out;
    out << 2 << ' ' << x.gain << ' ' << x.tau << ' ' << x.delay << ' ' << y.gain << ' ' << y.tau << ' ' << y.delay << '\n';
    g_files.replace(g_app.profile_path, out.str());  // (written by the file thread: see FileWriter)
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
    if (!latewarp.initialize(renderer.device(), g_app.base_dir, g_app.data_dir / L"logs"))
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
    // With split queues a frame taken in becomes the source the warp uses only once the GPU has finished
    // it (Renderer::show_intake): until then the warp keeps the previous one, textures and camera together.
    struct Incoming { IngestedSource s; Camera cam{}; CameraBasis basis{}; bool masked = false; } incoming;
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
    std::deque<std::uint64_t> analyzed_frames;  // game frames whose motion fit is on its way back
    bool have_source_kind = false, source_was_estimated = false;
    double last_mv_log = 0, last_usage_log = 0;
    bool game_has_hud_layers = false, mask_logged = false, source_masked = false;
    CameraEstimator estimator;
    bool estimator_logged = false, fov_logged = false;
    bool fsr_logged = false;
    int logged_fov_mode = 0;
    double estimator_ms_sum = 0, flush_ms_sum = 0;
    int estimator_runs = 0, mask_builds = 0;
    std::vector<float> raw_samples;
    double intake_wait_ms = 0;  // how long the refresh loop had to wait for motion samples (camera estimation)
    double intake_cpu_ms = 0;   // CPU time of the first step of the frame being taken in
    // A game frame taken in over two steps (see take_in_start / take_in_finish below).
    struct Intake { int slot = -1; IngestedSource s; bool sampled = false; };
    Intake intake;
    std::uint64_t intake_fence = 0;  // submission carrying the motion samples of the frame in `intake`
    std::vector<MotionSample> motion_samples;
    double intake_ms = 4.0;  // how long taking in a new game frame takes (GPU span, smoothed)
    // Frame generation detection: presented images vs rendered frames per half-second window.
    bool frame_generation = false;
    int fg_votes = 0;
    LONG fg_presents = 0, fg_frames = 0;
    std::int64_t fg_window_start = 0;
    std::uint64_t waiting_frame = 0;  // newest game frame seen while waiting for a refresh, and since when
    std::int64_t waiting_since = 0;
    std::uint64_t intakes = 0;

    while (g_app.running) {
        WaitForSingleObjectEx(renderer.waitable(), 100, FALSE);
        if (!g_app.running) break;
        const Settings settings = sh.settings;
        apply_gpu_priority(settings.gpu_priority);
        // Newest published frame whose GPU work is already complete (-1: none).
        auto find_newest = [&]() {
            const std::uint64_t game_done = renderer.game_fence_completed();
            int newest = -1;
            for (int i = 0; i < kSlots; ++i) {
                const auto& m = sh.slots[i];
                if (m.state == kReady && m.frame_id > source_frame && m.camera.valid && m.fence_value <= game_done &&
                    (newest < 0 || m.frame_id > sh.slots[newest].frame_id)) newest = i;
            }
            if (newest >= 0 || !source_frame) return newest;
            // Frame numbers that come from the game (Streamline, FSR) can start over, e.g. when it returns to
            // its main menu and recreates the upscaler: a frame rendered after the current one but numbered
            // lower. Follow the game's new count (otherwise the view stays frozen until something resets it).
            for (int i = 0; i < kSlots; ++i) {
                const auto& m = sh.slots[i];
                const std::int64_t q = m.qpc_sim_start ? m.qpc_sim_start : m.qpc_constants;
                if (m.state == kReady && m.frame_id < source_frame && m.camera.valid && m.fence_value <= game_done &&
                    seconds(q) > source_time && (newest < 0 || m.frame_id > sh.slots[newest].frame_id)) newest = i;
            }
            if (newest >= 0) {
                logf("the game's frame numbers started over (%llu after %llu): following them",
                     static_cast<unsigned long long>(sh.slots[newest].frame_id), static_cast<unsigned long long>(source_frame));
                source_frame = 0;
            }
            return newest;
        };
        // Taking in a new game frame, in two steps. take_in_start converts it and, for games without a
        // camera, records the motion samples the camera is estimated from; take_in_finish reads those,
        // estimates the camera, builds the no-warp mask and makes the frame the source. Between refreshes
        // the two steps are separate submissions, so the CPU never waits for the GPU in between.
        auto take_in_start = [&](int newest) {
            if (InterlockedCompareExchange(&sh.slots[newest].state, kReading, kReady) != kReady) return false;
            const SlotMeta& m = sh.slots[newest];
            // The previous frame's colour feeds the learned HUD detector (and the shelved object
            // interpolation); not needed while the HUD comes from the upscaler's output.
            const bool hud_from_output = settings.hud_from_scene == 1 && m.tex[kScene].valid;  // (combined: 2 needs it)
            renderer.set_keep_previous_colour(settings.extrapolate_objects != 0 ||
                                              (settings.no_warp_mask != 0 && !game_has_hud_layers && !hud_from_output));
            intake = {};
            intake.slot = newest;
            const bool wants_samples = m.camera.estimated != 0;
            intake.s = renderer.ingest(sh, newest, [&](const IngestedSource& early) {
                if (wants_samples && early.has_depth && early.has_motion) intake.sampled = renderer.record_motion_samples(early, 80, 45);
            });
            held.push_back({newest, renderer.intake_submitted() + 1});
            return true;
        };
        auto commit_incoming = [&]() {
            source = incoming.s;
            source_camera = incoming.cam;
            source_basis = incoming.basis;
            source_masked = incoming.masked;
            first_eval = true;
        };
        auto take_in_finish = [&]() {  // true: a valid frame was taken in
            const int newest = intake.slot;
            intake.slot = -1;
            if (newest < 0) return false;
            const SlotMeta& m = sh.slots[newest];
            const IngestedSource s = intake.s;
            const bool sampled = intake.sampled && renderer.read_motion_samples(raw_samples);
            if (s.valid) {
                // Games without a camera (DLSS without Streamline): estimate it from the motion vectors.
                Camera cam = m.camera;
                // A switch between the game's own camera and one estimated from motion vectors (e.g. the
                // player changed DLSS <-> FSR in a game that sends Streamline data only with DLSS): nothing
                // learned from one carries over to the other except the input calibration.
                if (have_source_kind && (cam.estimated != 0) != source_was_estimated) {
                    g_app.model.reset_history();
                    estimator.reset();
                    mv_scale = MotionVectorScale{};
                    renderer.reset_hud_detection();
                    logf("camera source switched to %s", cam.estimated ? "the upscaler's motion vectors (estimated)" : "the game's own camera data");
                }
                have_source_kind = true; source_was_estimated = cam.estimated != 0;
                if (cam.estimated && s.has_depth && s.has_motion && s.depth_rect.w && s.depth_rect.h) {
                    constexpr std::uint32_t kGridW = 80, kGridH = 45;
                    if (sampled) {
                        const double rw = s.depth_rect.w, rh = s.depth_rect.h;
                        motion_samples.clear();
                        for (std::uint32_t gy = 0; gy < kGridH; ++gy)
                            for (std::uint32_t gx = 0; gx < kGridW; ++gx) {
                                const float* v = &raw_samples[(gy * kGridW + gx) * 4];
                                MotionSample p{};
                                p.x = float((std::floor((gx + 0.5) * rw / kGridW)) + 0.5);
                                p.y = float((std::floor((gy + 0.5) * rh / kGridH)) + 0.5);
                                p.mx = v[0] * m.camera.mvec_scale[0]; p.my = v[1] * m.camera.mvec_scale[1];
                                // near / distance (0 = sky) whichever way round the game stores depth
                                p.depth = m.camera.depth_inverted ? v[2] : 1.0f - v[2];
                                p.valid = (v[3] > 0.5f && std::isfinite(p.mx) && std::isfinite(p.my) && std::fabs(p.mx) < 1e4f) ? 1.0f : 0.0f;
                                motion_samples.push_back(p);
                            }
                        const double t0 = now_seconds();
                        cam = estimator.update(motion_samples, rw, rh, m.camera);
                        estimator_ms_sum += (now_seconds() - t0) * 1000.0; ++estimator_runs;
                        flush_ms_sum += renderer.last_flush_ms() + intake_wait_ms;
                        intake_wait_ms = 0;
                        if (estimator_runs >= 300) {
                            logf("camera estimation: %.2f ms CPU + %.2f ms waiting for the samples per game frame, residual %.2f px, field of view %.1f deg%s, "
                                 "%.0f%% of frames unexplained, depth %s",
                                 estimator_ms_sum / estimator_runs, flush_ms_sum / estimator_runs, estimator.last_residual(),
                                 estimator.vertical_fov() * 180.0 / 3.14159265358979, estimator.fov_locked() ? "" : " (learning)",
                                 estimator.take_rejected_fraction() * 100.0, m.camera.depth_inverted ? "reversed" : "standard");
                            estimator_ms_sum = 0; flush_ms_sum = 0; estimator_runs = 0;
                        }
                        if (!estimator_logged) { logf("no camera from the game: estimating it from the upscaler's motion vectors"); estimator_logged = true; }
                        // The upscaler's motion vector settings, once (FSR games).
                        if (!fsr_logged && g_app.shared->fsr.upscale_dispatches) {
                            const FsrStats& fs = g_app.shared->fsr;
                            logf("FSR: render %ux%u -> %ux%u, motion vector scale %g x %g, create flags 0x%x, near %g far %g, field of view %.1f deg",
                                 fs.render_w, fs.render_h, fs.out_w, fs.out_h, fs.mv_scale[0], fs.mv_scale[1], fs.create_flags, fs.near_plane,
                                 fs.far_plane, fs.fov * 180.0 / 3.14159265358979);
                            fsr_logged = true;
                        }
                        if (estimator.game_fov_mode() != logged_fov_mode) {
                            logged_fov_mode = estimator.game_fov_mode();
                            const double deg = 180.0 / 3.14159265358979;
                            if (logged_fov_mode == 1) logf("the game's field of view matches the picture: %.1f deg vertical", m.camera.fov * deg);
                            else if (logged_fov_mode == 2)
                                logf("the game's field of view (%.1f deg) is the horizontal one: %.1f deg vertical", m.camera.fov * deg, 2.0 * std::atan(std::tan(m.camera.fov * 0.5) * rh / rw) * deg);
                            else if (logged_fov_mode == 3)
                                logf("the game's field of view (%.1f deg) does not match the picture: using the learned %.1f deg vertical", m.camera.fov * deg,
                                     estimator.learned_fov() * deg);
                        }
                        if (estimator.fov_locked() && !fov_logged) {
                            logf("field of view learned: %.1f deg vertical", estimator.vertical_fov() * 180.0 / 3.14159265358979);
                            fov_logged = true;
                        }
                    }
                }
                incoming.s = s;
                incoming.cam = cam;
                incoming.basis = to_basis(cam);
                source_time = seconds(m.qpc_sim_start ? m.qpc_sim_start : m.qpc_constants);
                source_frame = m.frame_id;
                g_app.model.add_source(m.frame_id, source_time, seconds(qpc_now()), incoming.basis, cam.reset != 0, now_seconds(),
                                       cam.position_epoch);
                if (rec(g_app.csv_sources)) {
                    const auto& c = cam;
                    wr(g_app.csv_sources,
                                 "%llu,%lld,%lld,%lld,%lld,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.5f,%.5f,%u,%d,%d,%d",
                                 static_cast<unsigned long long>(m.frame_id), m.qpc_sim_start, m.qpc_constants, m.qpc_present, qpc_now(),
                                 c.pos[0], c.pos[1], c.pos[2], c.fwd[0], c.fwd[1], c.fwd[2], c.up[0], c.up[1], c.up[2],
                                 c.right[0], c.right[1], c.right[2], c.fov, c.aspect, c.reset, int(g_app.model.mouse_gate()),
                                 int(s.has_depth), int(s.has_hudless));
                    // Projection and the game's frame-to-frame reprojection (same line, appended).
                    for (float v : c.view_to_clip) wr(g_app.csv_sources, ",%.7g", v);
                    for (float v : c.clip_to_prev_clip) wr(g_app.csv_sources, ",%.7g", v);
                    wr(g_app.csv_sources, "\n");
                }
                game_has_hud_layers = s.has_hudless && s.has_ui && settings.use_ui_tags;
                // Two independent parts of the no-warp mask: the HUD (only games without HUD layers) and
                // what moves with the camera (every game: third-person character, first-person weapon).
                const bool hud_mask = settings.no_warp_mask && !game_has_hud_layers && s.has_depth;
                const bool attached_mask = settings.keep_attached && s.has_depth && s.has_motion;
                const bool mask = hud_mask || attached_mask;
                incoming.masked = mask;
                if ((settings.extrapolate_objects || mask) && s.has_motion)
                    analyzed_frames.push_back(m.frame_id);
                    if (analyzed_frames.size() > 16) analyzed_frames.pop_front();
                    renderer.analyze_motion(s, cam.clip_to_prev_clip, float(mv_scale.scale(0, s.depth_rect.w)),
                                            float(mv_scale.scale(1, s.depth_rect.h)), mv_scale.valid, cam.depth_inverted != 0,
                                            settings.near_camera_rule != 0);
                if (mask) {
                    // HUD from the upscaler's output only when chosen (opt-in: its colour model cannot be
                    // proven for every game); otherwise the learned HUD map, which works everywhere.
                    IngestedSource for_mask = s;
                    if (!settings.hud_from_scene) for_mask.has_scene = false;
                    renderer.build_no_warp_mask(for_mask, cam.clip_to_prev_clip, hud_mask, s.has_motion && mv_scale.valid, attached_mask,
                                                cam.depth_inverted != 0, settings.hud_from_scene == 2,
                                                settings.hud_fill != 0 && (settings.warp_engine == 1 || !latewarp.ready()));
                    if (++mask_builds >= 300) {
                        mask_builds = 0;
                        const HudStats hs = renderer.take_hud_stats();
                        if (hs.scene_frames) {
                            double fits = 0;
                            for (int i = 0; i + 1 < 18; ++i) fits += hs.scene_pass_ms[i];
                            logf("HUD from the upscaler's output: %d frames, %.1f%% of the screen on average; GPU ms per frame: fits %.2f, final %.2f",
                                 hs.scene_frames, 100.0 * hs.scene_share_sum / hs.scene_frames, fits / hs.scene_frames,
                                 hs.scene_pass_ms[17] / hs.scene_frames);
                        }
                        if (hs.frames)
                            logf("HUD detection: %d frames, learned from %d (HUD-like share %.0f%%), %d with too little detail, %d held back by the guard",
                                 hs.frames, hs.learned, hs.learned ? 100.0 * hs.share_sum / hs.learned : 0.0, hs.too_little, hs.guarded);
                    }
                    if (!mask_logged) { logf("no-warp mask: %s%s", hud_mask ? "HUD (the game has no HUD layers)" : "", attached_mask ? (hud_mask ? " + character/weapon" : "character/weapon") : ""); mask_logged = true; }
                }
                const double present_t = seconds(m.qpc_present);
                if (last_source_present > 0) source_interval = present_t - last_source_present;
                last_source_present = present_t;
                ++stat_sources;
                if (!renderer.split()) commit_incoming();
            }
            return s.valid;
        };
        // Both steps on the current command list (the refresh itself, without pacing): the samples, if
        // any, are waited for in between.
        auto take_in = [&](int newest) {
            if (!take_in_start(newest)) return false;
            if (intake.sampled) {
                const double w0 = now_seconds();
                renderer.flush_and_wait();
                intake_wait_ms += (now_seconds() - w0) * 1000.0;
            }
            take_in_finish();
            return true;
        };
        // DWM composes shortly after each vblank; a frame that is not finished by then waits a
        // whole refresh. Render `lead` ms before the next vblank so the GPU has room even when the
        // game delays our work.
        vblank.update(renderer.present_stats());
        // DWM composes ~3 ms after each vblank and shows the result at the following vblank; with a
        // frame latency of 1 the swapchain only frees us at that display vblank, leaving ~2.5 ms. With a
        // lead we allow one queued frame and wake `lead` ms before each composition instead, one frame
        // per refresh (a new target is always a later vblank than the previous one).
        const bool paced = settings.present_lead_ms > 0 && vblank.valid();
        // Split queues with XPAR's own engine when paced (Latewarp and the unpaced path record everything on
        // the realtime queue). Switched only while no frame is being taken in.
        {
            const bool want_split = paced && (settings.warp_engine == 1 || !latewarp.ready());
            if (want_split != renderer.split() && intake.slot < 0) {
                if (renderer.intake_pending() && renderer.show_intake()) commit_incoming();
                if (!renderer.intake_pending() && renderer.set_split(want_split)) {
                    // (the GPU is idle now: every slot still held can go back to the game)
                    for (const auto& entry : held) InterlockedCompareExchange(&sh.slots[entry.slot].state, kFree, kReading);
                    held.clear();
                    logf("split queues %s", renderer.split() ? "on: game frames are taken in on their own GPU queue" : "off");
                }
            }
        }
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
            // New game frames are taken in while waiting for the refresh, each as its own GPU submission, so
            // the refresh itself only warps (taking a frame in costs several ms of GPU at 4K). One that
            // arrives too close to the refresh waits until just after it.
            const std::int64_t tick = v + compose - lead;
            const std::int64_t budget = static_cast<std::int64_t>((intake_ms + 1.0) * 1e-3 * f);
            const bool can_take_in = g_app.visible && !frame_generation && renderer.session_open(sh.session) &&
                                     sh.backbuffer_width == renderer.width() && sh.backbuffer_height == renderer.height();
            for (;;) {
                const std::int64_t t = qpc_now();
                // A frame taken in on the intake queue whose GPU work has finished: shown from now on.
                if (renderer.intake_pending() && renderer.show_intake()) commit_incoming();
                // A frame whose motion samples are on their way: finished as soon as the GPU has them, and
                // before the refresh at the latest (with split queues after it, if not back by then: its
                // work cannot delay the warp there).
                if (intake.slot >= 0) {
                    if (renderer.intake_completed(intake_fence) || (t >= tick && !renderer.split())) {
                        const double w0 = now_seconds();
                        renderer.wait_for_intake(intake_fence);
                        const double w1 = now_seconds();
                        intake_wait_ms += (w1 - w0) * 1000.0;
                        renderer.begin_intake();
                        const bool valid = take_in_finish();
                        renderer.submit_work(true);
                        if (valid) renderer.mark_intake_complete();
                        const double sample_ms = intake_cpu_ms + (w1 - w0) * 1000.0 + (now_seconds() - w1) * 1000.0;
                        intake_ms = std::clamp(0.8 * intake_ms + 0.2 * std::max<double>(renderer.last_intake_gpu_ms(), sample_ms), 2.0, 12.0);
                        if (++intakes % 600 == 0) logf("game frames taken in between refreshes: %.1f ms each (smoothed)", intake_ms);
                        continue;
                    }
                    if (t >= tick) break;
                    sleep_until(std::min(tick, t + static_cast<std::int64_t>(2.5e-4 * double(g_qpc_frequency))));
                    continue;
                }
                if (t >= tick) break;
                // With split queues a new frame starts only once the previous one is shown (the intake
                // writes the other set), and needs only CPU time before the refresh (its GPU work runs on
                // its own queue).
                const int newest = can_take_in && !renderer.intake_pending() ? find_newest() : -1;
                if (newest >= 0 && sh.slots[newest].frame_id != waiting_frame) { waiting_frame = sh.slots[newest].frame_id; waiting_since = t; }
                // ...unless it has already waited a whole refresh (taking in would never fit otherwise).
                const bool overdue = newest >= 0 && t - waiting_since > static_cast<std::int64_t>(vblank.period);
                const std::int64_t room = renderer.split() ? static_cast<std::int64_t>(2e-3 * f) : budget;
                if (newest >= 0 && (t + room < tick || overdue)) {
                    renderer.begin_intake();
                    const double t0 = now_seconds();
                    const bool taken = take_in_start(newest);
                    if (taken && intake.sampled) {  // the rest once the samples are back (above)
                        renderer.submit_work();
                        intake_fence = renderer.intake_submitted();
                        intake_cpu_ms = (now_seconds() - t0) * 1000.0;
                        continue;
                    }
                    const bool valid = taken && take_in_finish();
                    renderer.submit_work();
                    if (valid) renderer.mark_intake_complete();
                    if (taken) {
                        intake_ms = std::clamp(0.8 * intake_ms + 0.2 * std::max<double>(renderer.last_intake_gpu_ms(), (now_seconds() - t0) * 1000.0), 2.0, 12.0);
                        if (++intakes % 600 == 0)
                            logf("game frames taken in between refreshes: %.1f ms each (smoothed)%s", intake_ms, overdue ? ", this one late" : "");
                    }
                    continue;
                }
                sleep_until(std::min(tick, t + static_cast<std::int64_t>(1e-3 * f)));  // look for new frames every ms
            }
        } else {
            last_target_vblank = 0;
        }
        wake_qpc = qpc_now();
        if (renderer.intake_pending() && renderer.show_intake()) commit_incoming();

        if (!renderer.session_open(sh.session)) {
            held.clear(); source = {}; source_frame = 0; intake.slot = -1; incoming = {};
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
            if (renderer.intake_completed(it->release_value)) {
                InterlockedCompareExchange(&sh.slots[it->slot].state, kFree, kReading);
                it = held.erase(it);
            } else ++it;
        }

        // Frame generation (DLSS, FSR, XeSS - any kind) presents two or more images per rendered frame. It
        // does the same job as FrameWarp and the two cannot be combined (generated images would be warped
        // as if they were rendered ones): step aside and say so, and come back when it is turned off.
        if (qpc_now() - fg_window_start >= g_qpc_frequency / 2) {
            const LONG presents = sh.presents_total, frames = sh.frames_total;
            const LONG dp = presents - fg_presents, df = frames - fg_frames;
            fg_presents = presents; fg_frames = frames; fg_window_start = qpc_now();
            if (df >= 5) {  // the game is rendering (menus and loading screens present without it)
                const double ratio = double(dp) / double(df);
                if (ratio >= 1.6) fg_votes = std::min(fg_votes + 1, 2);
                else if (ratio <= 1.25) fg_votes = std::max(fg_votes - 1, -2);
                if (!frame_generation && fg_votes >= 2) {
                    frame_generation = true;
                    logf("frame generation detected (%.1f presented images per rendered frame): stepping aside until it is off", ratio);
                } else if (frame_generation && fg_votes <= -2) {
                    frame_generation = false;
                    logf("frame generation off: reprojecting again");
                }
            }
            sh.presenter.frame_generation = frame_generation ? 1u : 0u;
        }
        if (!g_app.visible || frame_generation) {  // game not in front / disabled / frame generation: don't compete with it for the GPU
            if (frame_generation) g_app.has_frames = false;
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

        // Without pacing, the frame is taken in right here, in the refresh (paced loops take frames in
        // between refreshes, above).
        auto* list = renderer.begin_frame();
        if (!paced && !renderer.split()) {
            const int newest = find_newest();
            if (newest >= 0) take_in(newest);
        }

        const double now = now_seconds();
        bool warped = false;
        bool unmoved = false;  // own engine: the displayed camera is the frame's own, the frame is shown as it is
        Prediction applied{};
        // FrameWarp's own warp engine when chosen, or when Latewarp is not available.
        const bool own_engine = settings.warp_engine == 1 || !latewarp.ready();
        if (source.valid && settings.enabled && !settings.show_original && !sh.overlay_open && source.has_depth) {
            Prediction p = g_app.model.predict(now);
            if (settings.invert_warp) p.camera = apply_rotation(source_basis, g_app.model.world_up(), -p.yaw, -p.pitch, source_basis.pos);
            applied = p;
            const Vec3 origin = source_basis.pos;
            Mat4 projection;
            std::memcpy(projection.data(), source_camera.view_to_clip, sizeof(projection));
            auto inputs = renderer.latewarp_inputs(source, settings.use_ui_tags != 0);
            inputs.depth_inverted = source_camera.depth_inverted != 0;
            if (source_masked) {
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
            if (own_engine) {
                const Mat4 m = clip_source_to_target(projection, view_matrix(source_basis, origin, z_sign), view_matrix(p.camera, origin, z_sign));
                // Nothing moved since the frame (well under a hundredth of a pixel anywhere): every pixel stays
                // where it is, so there is nothing to warp. (The mask's debug tint and the object extrapolation
                // need the warp pass.)
                unmoved = !settings.show_mask && !settings.extrapolate_objects;
                for (int i = 0; i < 16 && unmoved; ++i) unmoved = std::fabs(m.data()[i] - (i % 5 == 0 ? 1.0f : 0.0f)) < 2e-6f;
                warped = unmoved || renderer.own_warp(source, settings.use_ui_tags != 0, inputs.no_warp_mask != nullptr, m.data(), inputs.depth_inverted);
            } else {
                warped = latewarp.evaluate(list, inputs, first_eval, view_matrix(p.camera, origin, z_sign),
                                           view_matrix(source_basis, origin, z_sign), projection);
            }
            if (warped) first_eval = false;
            if (warped && settings.show_mask && inputs.no_warp_mask) renderer.tint_mask();
        }
        // Menus and loading screens often skip the game's usual frame (no DLSS call, no tags): when several
        // presents in a row bring no frame, step aside and let the game's own picture show.
        g_app.has_frames = source.valid && sh.presents_without_frame < 3;
        renderer.finish_frame(warped && !unmoved, settings.overlay_debug ? (warped ? 1 : 2) : 0);
        // Ctrl+Shift+M marks "it looks bad now" in the log.
        const bool mark_down = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) && (GetAsyncKeyState('M') & 0x8000);
        const int mark = mark_down && !g_app.mark_was_down ? 1 : 0;
        g_app.mark_was_down = mark_down;
        if (mark) logf("MARK (user flagged a bad moment)");
        // Ctrl+Shift+D (development): saves the newest game frame, the warped output, the upscaler's output,
        // depth, motion vectors, the motion analysis and the masks (half-float RGBA, 8-byte header: width,
        // height) to captures\.
        const bool dump_down = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) && (GetAsyncKeyState('D') & 0x8000);
        if (dump_down && !g_app.dump_was_down && source.valid) {
            static int captures = 0;
            const auto dir = g_app.data_dir / L"captures";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            const int n = ++captures;
            for (const auto& [which, name] : {std::pair{0, L"frame"}, std::pair{1, L"output"}, std::pair{2, L"scene"}, std::pair{3, L"depth"},
                                              std::pair{4, L"motion"}, std::pair{5, L"object"}, std::pair{6, L"mask"}, std::pair{7, L"hudscore"},
                                              std::pair{8, L"world"}, std::pair{9, L"fill"}}) {
                std::vector<std::uint16_t> px;
                std::uint32_t cw = 0, ch = 0;
                if (which == 2 && !source.has_scene) { logf("capture %d: no upscaler output this frame", n); continue; }
                if (!renderer.read_back(which, px, cw, ch)) continue;
                const auto path = dir / (L"capture_" + std::to_wstring(n) + L"_" + name + L".f16");
                FILE* f = nullptr;
                if (_wfopen_s(&f, path.c_str(), L"wb") == 0 && f) {
                    const std::uint32_t header[2] = {cw, ch};
                    std::fwrite(header, sizeof(header), 1, f);
                    std::fwrite(px.data(), 2, px.size(), f);
                    std::fclose(f);
                }
            }
            logf("capture %d saved", n);
        }
        g_app.dump_was_down = dump_down;
        if (rec(g_app.csv_outputs)) {
            const auto tm = renderer.last_timing();
            const auto pstats = renderer.present_stats();
            // (the displayed camera too: position and forward, in the game's world)
            const auto& dc = applied.camera;
            wr(g_app.csv_outputs, "%lld,%llu,%d,%.6f,%.6f,%.4f,%.3f,%d,%lld,%lld,%lld,%u,%u,%u,%u,%lld,%ld,%lld,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f\n",
                         qpc_now(), static_cast<unsigned long long>(source_frame), int(warped), applied.yaw, applied.pitch,
                         applied.horizon * 1000.0, renderer.last_gpu_ms(), mark, tm.submit, tm.gpu_start, tm.gpu_end, pstats.last_present_count,
                         pstats.present_count, pstats.present_refresh, pstats.sync_refresh, pstats.sync_qpc, static_cast<long>(pstats.hr), wake_qpc,
                         dc.pos.x, dc.pos.y, dc.pos.z, dc.fwd.x, dc.fwd.y, dc.fwd.z);
        }
        // Dump the game-side timeline.
        if ((settings.record_diagnostics != 0) != g_recording.load()) {
            if (settings.record_diagnostics) { open_recordings(); logf("detailed diagnostics: recording to the logs folder"); }
            else { g_recording = false; logf("detailed diagnostics: off"); }
        }
        if (rec(g_app.csv_events)) {
            const std::int64_t count = sh.timeline_count;
            if (count - g_app.events_read > kTimeline) g_app.events_read = count - kTimeline;
            for (; g_app.events_read < count; ++g_app.events_read) {
                const auto& e = sh.timeline[g_app.events_read % kTimeline];
                wr(g_app.csv_events, "%lld,%u,%u,%llu,%llu\n", e.qpc, e.kind, e.tid,
                             static_cast<unsigned long long>(e.frame), static_cast<unsigned long long>(e.extra));
            }
        }
        ++stat_frames;

        {
            const auto notes = renderer.take_notes();
            for (const auto& note : notes) logf("%s", note.c_str());
            MotionFit fit;
            const bool have_fit = renderer.take_motion_fit(fit);
            if (have_fit && rec(g_app.csv_motion)) {
                // How well the game's motion vectors match the camera motion computed from depth (1: exactly, for
                // the static scene), with the locked scale; a frame whose depth does not belong to its picture
                // drops here. (Fits come back in order, a few frames later.)
                const std::uint64_t f = analyzed_frames.empty() ? 0 : analyzed_frames.front();
                if (!analyzed_frames.empty()) analyzed_frames.pop_front();
                double agree[2] = {-1, -1};
                for (int k = 0; k < 2 && mv_scale.valid; ++k) {
                    const double sc = mv_scale.scale(k, k ? source.depth_rect.h : source.depth_rect.w);
                    if (fit.cc[k] > 0) agree[k] = 1.0 - (sc * sc * fit.gg[k] - 2 * sc * fit.gc[k] + fit.cc[k]) / fit.cc[k];
                }
                wr(g_app.csv_motion, "%lld,%llu,%.0f,%.4f,%.4f,%.4f,%u,%u\n", qpc_now(), static_cast<unsigned long long>(f), fit.samples,
                             fit.samples > 0 ? fit.moving / fit.samples : 0.0, agree[0], agree[1], source.depth_rect.w, source.depth_rect.h);
            }
            if (have_fit && mv_scale.add(fit, source.depth_rect.w, source.depth_rect.h))
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
                g_files.flush_now();
                ExitProcess(0);
            }
            auto& st = sh.presenter;
            st.latewarp = latewarp.ready() ? 2u : 1u;
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
            // What the presenter costs the GPU (shared with the game), every 10 s.
            if (now - last_usage_log >= 10.0) {
                const Renderer::GpuUsage u = renderer.take_gpu_usage();
                const double span = last_usage_log > 0 ? now - last_usage_log : 0.0;
                last_usage_log = now;
                if (span > 0 && (u.intakes || u.warps)) {
                    const double n = std::max<double>(1, u.split);
                    logf("presenter GPU: %.0f%% of the time | game frames %.1f/s, %.2f ms each (reaching the game's textures %.2f, depth+motion %.2f, "
                         "colour copies %.2f, HUD/masks/motion %.2f) | refreshes %.1f/s, warp %.2f ms each",
                         (u.intake_ms + u.warp_ms) / (span * 10.0), u.intakes / span, u.intakes ? u.intake_ms / u.intakes : 0.0,
                         u.access_ms / n, u.depth_ms / n, u.colour_ms / n, u.rest_ms / n, u.warps / span, u.warps ? u.warp_ms / u.warps : 0.0);
                }
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
                            !g_app.model.mouse_gate() ? "cursor visible (ignored)" : g_app.model.camera_follows_mouse() ? "camera" : "camera not following (ignored)", latewarp.status().c_str());
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
                // (not while the ReShade menu has the mouse: the game's camera does not move then)
                if (!g_app.shared || !g_app.shared->overlay_open) g_app.model.mouse.add(seconds(q), raw.data.mouse.lLastX, raw.data.mouse.lLastY);
                if (rec(g_app.csv_mouse)) wr(g_app.csv_mouse, "%lld,%ld,%ld\n", q, raw.data.mouse.lLastX, raw.data.mouse.lLastY);
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
    // Logs, calibration and captures go next to the presenter; some game folders only let administrators
    // write (the game then runs as the user), so fall back to the user's local app data.
    g_app.data_dir = g_app.base_dir;
    {
        std::error_code ec;
        std::filesystem::create_directories(g_app.base_dir / L"logs", ec);
        const auto probe = g_app.base_dir / L"logs" / L".write-test";
        FILE* written = _wfsopen(probe.c_str(), L"w", _SH_DENYNO);
        if (written) { std::fclose(written); std::filesystem::remove(probe, ec); }
        else {
            wchar_t local[MAX_PATH]{};
            if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH))
                g_app.data_dir = std::filesystem::path(local) / L"FrameWarp" / g_app.base_dir.parent_path().filename();
            std::filesystem::create_directories(g_app.data_dir / L"logs", ec);
        }
    }
    // Keep the previous run's logs: after a crash the user restarts the presenter, and that run is the one we need.
    {
        std::error_code ec;
        const auto previous = g_app.data_dir / L"logs" / L"previous";
        std::filesystem::remove_all(previous, ec);
        std::filesystem::create_directories(previous, ec);
        for (const auto& entry : std::filesystem::directory_iterator(g_app.data_dir / L"logs", ec))
            if (entry.is_regular_file(ec)) std::filesystem::rename(entry.path(), previous / entry.path().filename(), ec);
    }
    g_app.log = _wfsopen((g_app.data_dir / L"logs" / L"presenter.log").c_str(), L"w", _SH_DENYNO);
    g_files.start();
    {
        LARGE_INTEGER fq; QueryPerformanceFrequency(&fq);
        logf("FrameWarp presenter " FW_VERSION ", qpc frequency %lld", fq.QuadPart);
        if (g_app.data_dir != g_app.base_dir) logf("the FrameWarp folder is read-only: logs and calibration are kept in %ls", g_app.data_dir.c_str());
    }
    if (!g_app.pid) { logf("usage: FrameWarpPresenter --pid <game pid>"); return 1; }

    // One presenter per game.
    const std::wstring mutex_name = L"Local\\FrameWarpPresenter_" + std::to_wstring(g_app.pid);
    HANDLE single = CreateMutexW(nullptr, TRUE, mutex_name.c_str());
    if (GetLastError() == ERROR_ALREADY_EXISTS) { logf("presenter already running"); g_files.stop(); return 0; }

    g_app.game_process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, g_app.pid);
    wchar_t game_exe[MAX_PATH]{}; DWORD len = MAX_PATH;
    if (g_app.game_process && QueryFullProcessImageNameW(g_app.game_process, 0, game_exe, &len))
        g_app.profile_path = g_app.data_dir / L"profiles" / (std::filesystem::path(game_exe).stem().wstring() + L".txt");
    else
        g_app.profile_path = g_app.data_dir / L"profiles" / L"default.txt";
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
    g_files.stop();
    for (FILE* csv : {g_app.csv_events, g_app.csv_mouse, g_app.csv_sources, g_app.csv_outputs, g_app.csv_motion}) if (csv) std::fclose(csv);
    if (g_app.log) std::fclose(g_app.log);
    if (single) CloseHandle(single);
    return 0;
}
