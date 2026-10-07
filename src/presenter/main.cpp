// FrameWarpPresenter: presents the game's latest frame at display rate, reprojected with NVIDIA
// Latewarp to the camera predicted from raw mouse input and the game's own camera history.
#include "presenter/pose.hpp"
#include "presenter/camera_estimator.hpp"
#include "presenter/frame_clock.hpp"
#include "presenter/renderer.hpp"
#include "shared/camera_check.hpp"
#include <shellapi.h>
#include <winternl.h>  // NTSTATUS for d3dkmthk.h
#include <d3dkmthk.h>
#include <dxgi1_4.h>
#include <pdh.h>
#include <Xinput.h>
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
    // The refresh rate of the display the game is on, from its current mode (0: not known); read by the window
    // thread (see follow_game_window), used by the render thread's vblank clock.
    std::atomic<double> display_mode_hz{0.0};
    HMONITOR display_monitor = nullptr;
    double display_checked = 0;
    PoseModel model;
    std::filesystem::path base_dir, profile_path;
    std::filesystem::path data_dir;  // logs, calibration, captures: base_dir, or %LOCALAPPDATA%\FrameWarp\<game folder> when that is read-only
    FILE* log = nullptr;
    // Analysis logs (all times are raw QPC ticks, same clock in both processes).
    FILE* csv_events = nullptr;   // game-side timeline
    FILE* csv_mouse = nullptr;    // raw input
    FILE* csv_stick = nullptr;    // controller right stick (on change)
    FILE* csv_hooks = nullptr;    // the add-on's call counters (Streamline, NGX, FSR, frame generation), cumulative, once a second
    FILE* csv_upscaler = nullptr; // what the game hands its upscaler (DLSS / FSR inputs: sizes, formats, motion vector scale, jitter), once a second
    std::atomic<bool> controller{false};  // an XInput controller is connected and read
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
        g_app.csv_stick = open_csv(L"stick.csv", "qpc,x,y");
        g_app.csv_upscaler = open_csv(L"upscaler.csv",
            "qpc,ngx_feature,ngx_create_flags,ngx_render_w,ngx_render_h,ngx_out_w,ngx_out_h,ngx_subrect_w,ngx_subrect_h,ngx_depth_w,ngx_depth_h,ngx_depth_format,"
            "ngx_mv_w,ngx_mv_h,ngx_mv_format,ngx_mv_scale_x,ngx_mv_scale_y,ngx_jitter_x,ngx_jitter_y,"
            "fsr_create_flags,fsr_render_w,fsr_render_h,fsr_out_w,fsr_out_h,fsr_mv_scale_x,fsr_mv_scale_y,fsr_jitter_x,fsr_jitter_y,fsr_near,fsr_far,fsr_fov");
        g_app.csv_hooks = open_csv(L"hooks.csv",
            "qpc,sl_constants,sl_tags,sl_tags_for_frame,sl_markers,sl_evaluate,sl_new_frame_token,sl_set_feature_loaded,sl_allocate,sl_free,"
            "sl_get_feature_function,ngx_creates,ngx_dlss_creates,ngx_evaluates,ngx_dlss_evaluates,ngx_feature1,ngx_feature11,ngx_feature13,"
            "ngx_published,ngx_joined,fsr_creates,fsr_dispatches,fsr_published,generation_calls,presents,frames,frames_published,slots_dropped");
        g_app.csv_motion = open_csv(L"motion.csv", "qpc,frame,samples,moving_fraction,agree_x,agree_y,render_w,render_h,camera_fit");
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

// Detailed diagnostics: the add-on's call counters once a second (hooks.csv) - which calls stop when the frames
// stop; also while paused.
void record_hooks(const Shared& sh, double now) {
    static double last = 0;
    // (what the game hands its upscaler, also in presenter.log whenever it changes - recording or not)
    {
        const auto& n = sh.ngx;
        static char logged[320] = "";
        char line[320];
        if (n.dlss_feature) {
            std::snprintf(line, sizeof(line), "DLSS inputs: feature %u, flags 0x%x, render %ux%u -> %ux%u, subrect %ux%u, depth %ux%u fmt %u, motion vectors %ux%u fmt %u, "
                          "scale %.6g x %.6g", n.dlss_feature, n.create_flags, n.render_w, n.render_h, n.out_w, n.out_h, n.subrect_w, n.subrect_h, n.depth_w,
                          n.depth_h, n.depth_format, n.mv_w, n.mv_h, n.mv_format, n.mv_scale[0], n.mv_scale[1]);
            if (std::strcmp(line, logged) != 0) { logf("%s", line); std::snprintf(logged, sizeof(logged), "%s", line); }
        }
    }
    if (now - last < 1.0 || (!rec(g_app.csv_hooks) && !rec(g_app.csv_upscaler))) return;
    last = now;
    if (rec(g_app.csv_upscaler)) {
        const auto &n = sh.ngx; const auto &f = sh.fsr;
        wr(g_app.csv_upscaler, "%lld,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%.6g,%.6g,%.4f,%.4f,%u,%u,%u,%u,%u,%.6g,%.6g,%.4f,%.4f,%.6g,%.6g,%.6g\n",
           static_cast<long long>(qpc_now()), n.dlss_feature, n.create_flags, n.render_w, n.render_h, n.out_w, n.out_h, n.subrect_w, n.subrect_h,
           n.depth_w, n.depth_h, n.depth_format, n.mv_w, n.mv_h, n.mv_format, n.mv_scale[0], n.mv_scale[1], n.jitter[0], n.jitter[1],
           f.create_flags, f.render_w, f.render_h, f.out_w, f.out_h, f.mv_scale[0], f.mv_scale[1], f.jitter[0], f.jitter[1], f.near_plane, f.far_plane, f.fov);
    }
    if (!rec(g_app.csv_hooks)) return;
    const auto &h = sh.hooks; const auto &n = sh.ngx; const auto &f = sh.fsr;
    wr(g_app.csv_hooks, "%lld,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%ld,%ld,%ld,%u,%u\n",
       static_cast<long long>(qpc_now()), h.constants_calls, h.tag_calls, h.tag_for_frame_calls, h.marker_calls, h.export_calls[3],
       h.export_calls[2], h.export_calls[5], h.export_calls[6], h.export_calls[7], h.export_calls[8], n.create_calls, n.dlss_creates,
       n.evaluate_calls, n.dlss_calls, n.feature_calls[1], n.feature_calls[11], n.feature_calls[13], n.frames_published, n.frames_joined,
       f.upscale_creates, f.upscale_dispatches, f.frames_published, static_cast<long>(sh.generation_calls), static_cast<long>(sh.presents_total),
       static_cast<long>(sh.frames_total), h.frames_published, h.slots_dropped);
}

// Detailed diagnostics switched on/off, and the game-side timeline dumped while they are on (also while
// FrameWarp steps aside for the game's frame generation).
void record_timeline(Shared& sh, const Settings& settings) {
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
    double last_log = 0, logged_total = -1, last_poll = 0;
    // (the latest reading, every 2 s; pressure_until: the warning holds 10 s past the last tight reading)
    double total_mb = -1, size_mb = 0, own_mb = 0, budget_mb = 0, pressure_until = -1;
    void init(const LUID& luid) {
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
        DXGI_ADAPTER_DESC1 desc{};
        if (adapter && SUCCEEDED(adapter->GetDesc1(&desc))) size_mb = double(desc.DedicatedVideoMemory) / 1048576.0;
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
        if (!reason && now - last_poll < 2.0) return;
        last_poll = now;
        const double total = adapter_total_mb();
        DXGI_QUERY_VIDEO_MEMORY_INFO own{};
        if (adapter) adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &own);
        total_mb = total; own_mb = own.CurrentUsage / 1048576.0; budget_mb = own.Budget / 1048576.0;
        // Windows shrinks a process's budget when the GPU's memory runs out: less than a quarter of headroom
        // left means the game and XPAR together no longer fit (they start to stutter and drop frames). So does
        // the card itself being 95% full, whatever XPAR's own budget says (DOOM Eternal at 4K on an 8 GB card:
        // 7.7 of 8 GB in use, XPAR's budget briefly tight and then roomier again - the warning came and went
        // within 10 s while both kept stuttering, at 10 game fps).
        const bool own_tight = own.Budget && double(own.Budget) < double(own.CurrentUsage) * 1.25;
        const bool card_full = size_mb > 0 && total > 0 && total >= 0.95 * size_mb;
        if (own_tight || card_full) {
            if (pressure_until < now) logf("vram nearly full: presenter %.0f MB of a %.0f MB budget, adapter %.0f of %.0f MB in use", own_mb, budget_mb, total,
                                           size_mb);
            pressure_until = now + 30.0;
        }
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
    if (in >> version && version >= 2 && version <= 4 &&
        in >> p[0].gain >> p[0].tau >> p[0].delay >> p[1].gain >> p[1].tau >> p[1].delay) {
        if (version >= 3 && !(in >> p[0].stick_gain >> p[0].stick_power >> p[1].stick_gain >> p[1].stick_power))
            p[0].stick_gain = p[1].stick_gain = 0, p[0].stick_power = p[1].stick_power = 1;
        // (the fit's quality: older profiles have none and count as unproven - any good fit replaces them)
        if (version >= 4 && !(in >> p[0].quality >> p[1].quality)) p[0].quality = p[1].quality = 0;
        p[0].fitted = p[1].fitted = true;
        g_app.model.seed(p[0], p[1]);
        logf("profile loaded: yaw gain %.4g tau %.0f ms, pitch gain %.4g tau %.0f ms", p[0].gain, p[0].tau * 1000, p[1].gain, p[1].tau * 1000);
    }
}
void save_profile() {
    const auto &x = g_app.model.params(0), &y = g_app.model.params(1);
    if (!x.fitted && !y.fitted) return;
    // (a model that explains the camera poorly is not saved: the last good one stays - see kMinFitQuality)
    if (x.quality < kMinFitQuality || y.quality < kMinFitQuality) return;
    std::ostringstream out;
    out << 4 << ' ' << x.gain << ' ' << x.tau << ' ' << x.delay << ' ' << y.gain << ' ' << y.tau << ' ' << y.delay << ' '
        << x.stick_gain << ' ' << x.stick_power << ' ' << y.stick_gain << ' ' << y.stick_power << ' ' << x.quality << ' ' << y.quality << '\n';
    g_files.replace(g_app.profile_path, out.str());  // (written by the file thread: see FileWriter)
}

// Predicts vblanks from DXGI frame statistics of our own swapchain (the display it is shown on).
struct VblankClock {
    std::int64_t ref_qpc = 0;
    UINT ref_count = 0;
    double period = 0;  // QPC ticks per refresh
    // The display's refresh rate from its current mode (set_display_hz): the period is that, exactly, and our
    // presents only tell when the refreshes happen. (Measured from the presents alone, a first estimate taken
    // from stale statistics could lock the period to a wrong rate until the next pause.)
    double display_period = 0;
    void set_display_hz(double hz) {
        const double p = hz > 1.0 ? double(g_qpc_frequency) / hz : 0.0;
        if (p > 0 && (display_period <= 0 || std::fabs(p / display_period - 1.0) > 1e-4)) period = p;
        display_period = p;
    }
    void update(const Renderer::PresentStats& st) {
        if (st.hr != S_OK || !st.sync_qpc || !st.sync_refresh) return;
        if (display_period > 0) {
            period = display_period;
            ref_qpc = st.sync_qpc; ref_count = st.sync_refresh;  // (a real refresh time: the phase)
            return;
        }
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
// One timer per thread: a timer shared between threads is re-armed by each caller, waking the others early or
// late (the render thread's refresh schedule must not depend on any other thread's waits). precise: the last
// 0.5 ms spun (the render thread's pacing); other threads just wait.
void sleep_until(std::int64_t target, bool precise = true) {
    static thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    const std::int64_t spin = precise ? g_qpc_frequency / 2000 : 0;
    const std::int64_t now = qpc_now();
    if (target - now > spin && timer) {
        LARGE_INTEGER due;
        due.QuadPart = -static_cast<LONGLONG>(double(target - now - spin) * 1e7 / double(g_qpc_frequency));
        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, 100);
    }
    if (precise) while (qpc_now() < target) YieldProcessor();
}

// Controller: the right stick of the first connected XInput pad, every 4 ms, into the camera model (the stick
// sets a turn rate, learned from the game's own camera as the mouse gain is - also while the option is off,
// so that switching it on applies at once). Its own thread: XInput calls can take a while.
void controller_thread() {
    int pad = -1;
    double next_scan = 0, last_x = 2, last_y = 2;
    while (g_app.running) {
        const double now = now_seconds();
        if (pad < 0 && now >= next_scan) {
            next_scan = now + 2.0;  // (looking for a pad on empty slots is slow: every 2 s)
            for (DWORD i = 0; i < XUSER_MAX_COUNT && pad < 0; ++i) {
                XINPUT_STATE state{};
                if (XInputGetState(i, &state) == ERROR_SUCCESS) pad = static_cast<int>(i);
            }
            if (pad >= 0) { logf("controller connected (XInput pad %d)", pad); g_app.controller = true; }
        }
        if (pad >= 0) {
            XINPUT_STATE state{};
            if (XInputGetState(static_cast<DWORD>(pad), &state) != ERROR_SUCCESS) {
                logf("controller disconnected");
                pad = -1; g_app.controller = false;
                g_app.model.stick.add(now, 0, 0);
            } else {
                // Radial dead zone (XInput's recommended size), rescaled so that deflection starts at 0 past it.
                double x = std::clamp(state.Gamepad.sThumbRX / 32767.0, -1.0, 1.0), y = std::clamp(state.Gamepad.sThumbRY / 32767.0, -1.0, 1.0);
                const double magnitude = std::hypot(x, y), dead = XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE / 32767.0;
                if (magnitude <= dead) x = y = 0;
                else { const double scale = std::min(1.0, (magnitude - dead) / (1.0 - dead)) / magnitude; x *= scale; y *= scale; }
                x = std::round(x * 1000.0) / 1000.0; y = std::round(y * 1000.0) / 1000.0;
                // (not while the ReShade menu is open: the game's camera does not move then)
                if (g_app.shared && g_app.shared->overlay_open) x = y = 0;
                g_app.model.stick.add(now, x, y);
                if ((x != last_x || y != last_y) && rec(g_app.csv_stick)) wr(g_app.csv_stick, "%lld,%.3f,%.3f\n", static_cast<long long>(qpc_now()), x, y);
                last_x = x; last_y = y;
            }
        }
        sleep_until(qpc_now() + g_qpc_frequency / 250, false);
    }
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
    FrameClock frame_clock;
    std::uint64_t source_frame = 0;
    std::uint64_t memory_frame = 0;  // game frame the background memory was last updated with
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
    bool dump_requested = false;
    int dump_stage = 0;              // Ctrl+Shift+D: 1, 2 = frames kept so far; written on the game frame after dump_after
    std::uint64_t dump_after = 0;
    struct DumpFrame { std::uint64_t frame = 0; double time = 0; Camera cam{}; IngestedSource s; double mv[2] = {}; bool mv_valid = false; };
    DumpFrame dump_frames[3];        // (oldest first)
    std::deque<std::pair<std::uint64_t, double>> frame_times;  // game frame -> its time (source_time), recent ones
    std::uint64_t shown_frame = 0;   // the game frame the warp shows (with split queues: after the one taken in last)
    double shown_since = 0;          // ...since when (moving objects: their own clock)
    double last_generated = -1e9;    // when a frame last came with generated images
    bool was_enabled = false;        // ("Enable reprojection", for the frame generation message)
    double last_pause_log = 0;
    // The picture presented last, when it was a frame shown as it is (0: anything else): the same again is
    // not drawn (see skip_frame).
    std::uint64_t as_is_frame = 0;
    std::uint32_t as_is_w = 0, as_is_h = 0;
    bool presented_last = true, paced_last = false;
    double enabled_since = 0;
    bool generation_logged = false;
    CameraCheck camera_check;        // the game's own camera against its motion vectors
    int poor_estimate_windows = 0;   // ...and, once it was found unusable, whether the estimate does any better
    bool have_source_kind = false, source_was_estimated = false, source_motion_estimated = false;
    double last_mv_log = 0, last_usage_log = 0;
    bool game_has_hud_layers = false, mask_logged = false, source_masked = false, hudless_hud_logged = false;
    CameraEstimator estimator;
    // Whether the estimated camera can be learned from: the frame itself was explained, and so were most of the last 60.
    std::deque<bool> estimate_rejects;
    bool estimate_trusted = true, estimate_was_trusted = true;
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
    LONG fg_presents = 0, fg_frames = 0, fg_calls = 0;
    std::int64_t fg_window_start = 0;
    std::uint64_t waiting_frame = 0;  // newest game frame seen while waiting for a refresh, and since when
    std::int64_t waiting_since = 0;
    std::uint64_t intakes = 0;
    // Where the refresh loop is held (per status window, and per 5 s for the log): the swapchain wait (the GPU
    // still busy with earlier refreshes) and the Present call (a frame rate cap of the graphics driver holds
    // the presents there). Output well below the refresh rate with the presents held: a cap from outside.
    double swap_wait_ms = 0, present_call_ms = 0, log_swap_wait_ms = 0, log_present_call_ms = 0, log_present_call_max = 0;
    // Latency breakdown: per present, the displayed camera's time and the present's time (by present count), matched
    // with the swapchain's statistics (the refresh that showed it); medians over 5 s.
    struct ShownPresent { UINT count = 0; double present_t = 0, camera_t = 0; bool done = true; std::int64_t target = 0; };
    ShownPresent shown_presents[64];
    std::vector<double> lat_behind, lat_display, lat_age, lat_queued;
    // Automatic present lead: as late as the warp still makes its refresh. Per warped frame, the time from the
    // planned wake-up to its GPU work finished (ready, a window of them), and the refreshes from its target vblank
    // to the screen: the usual count (the most common of a window) or more (late: missed its refresh).
    // target = p99 of ready + margin; the margin grows with late frames and shrinks slowly while there are none.
    // The lead in use glides towards the target a little each refresh (a jump would move the camera's time by
    // as much in one refresh: a visible hitch).
    struct AutoLead {
        double lead_ms = 3.0, target_ms = 3.0, margin_ms = 0.75;
        std::vector<double> ready;
        int refreshes[8] = {}, matched = 0, late = 0, usual = -1;
        int late_logged = 0, matched_logged = 0, prev_r = -1;
        UINT prev_count = 0, stat_count = 0, stat_refresh = 0;
    } auto_lead;
    struct PlannedFrame { std::int64_t submit = 0, tick = 0; };
    PlannedFrame planned[16];
    int planned_at = 0;
    std::int64_t frame_tick = 0, frame_vblank = 0, timing_seen = 0;
    // Auto latency "camera as of now": the time ahead of Auto's camera in use (s).
    double now_ahead = 0;
    std::uint32_t log_presents = 0, capped_windows = 0;
    double log_since = now_seconds();

    while (g_app.running) {
        // (the swapchain frees a buffer only after a present: after a refresh with nothing new to show, none
        // to wait for - without pacing, a short sleep instead)
        if (presented_last) {
            const double w0 = now_seconds();
            WaitForSingleObjectEx(renderer.waitable(), 100, FALSE);
            swap_wait_ms += (now_seconds() - w0) * 1000.0;
        }
        else if (!paced_last) sleep_until(qpc_now() + g_qpc_frequency / 1000);
        if (!g_app.running) break;
        const Settings settings = sh.settings;
        apply_gpu_priority(settings.gpu_priority);
        // Moving objects at the display rate: an option of the XPAR engine.
        // XPAR's own warp engine (chosen, or Latewarp not available): the one that moves objects and shows the
        // game's generated images.
        auto own_engine_on = [&]() { return settings.warp_engine == 1 || !latewarp.ready(); };
        // Moving objects without frame generation (option): XPAR moves them itself, one game frame late.
        auto moving_objects_on = [&]() { return settings.moving_objects != 0 && own_engine_on(); };
        // The game's frame generation is on and its images are not coming with the frames (NVIDIA Latewarp
        // chosen, or a kind of frame generation XPAR does not take): step aside.
        auto generation_aside = [&]() {
            if (!frame_generation) return false;
            if (!own_engine_on()) return true;
            const auto latest = static_cast<std::uint64_t>(sh.latest_ready_frame);
            for (const auto& m : sh.slots)
                if (m.frame_id == latest && m.state != kFree && m.generated) last_generated = std::max(last_generated, now_seconds());
            return now_seconds() - last_generated >= 0.5;
        };
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
            // The previous frame's colour feeds the learned HUD detector; not needed while the HUD comes from
            // the upscaler's output.
            const bool hud_from_output = settings.hud_from_scene == 1 && m.tex[kScene].valid;  // (combined: 2 needs it)
            renderer.set_keep_previous_colour(settings.no_warp_mask != 0 && !game_has_hud_layers && !hud_from_output);
            // Moving objects (option, XPAR engine): every frame keeps the previous one's picture and depth.
            renderer.set_object_history(moving_objects_on());
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
            if (shown_frame != source_frame) shown_since = now_seconds();
            shown_frame = source_frame;
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
                // The same between an upscaler's motion vectors and XPAR's own, estimated from the picture (a
                // game whose upscaler starts after its intro): a field of view, a motion vector scale or a HUD
                // learned from the one must not be kept for the other.
                if (have_source_kind && ((cam.estimated != 0) != source_was_estimated || s.motion_estimated != source_motion_estimated)) {
                    g_app.model.reset_history();
                    estimator.reset();
                    mv_scale = MotionVectorScale{};
                    camera_check = CameraCheck{};
                    renderer.reset_hud_detection();
                    logf("camera source switched to %s", !cam.estimated ? "the game's own camera data" :
                         s.motion_estimated ? "XPAR's own motion vectors (estimated from the picture)" : "the upscaler's motion vectors (estimated)");
                }
                have_source_kind = true; source_was_estimated = cam.estimated != 0; source_motion_estimated = s.motion_estimated;
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
                        cam = estimator.update(motion_samples, rw, rh, m.camera, s.motion_estimated);
                        estimator_ms_sum += (now_seconds() - t0) * 1000.0; ++estimator_runs;
                        {
                            estimate_rejects.push_back(estimator.last_rejected());
                            if (estimate_rejects.size() > 60) estimate_rejects.pop_front();
                            int rejected = 0;
                            for (const bool r : estimate_rejects) rejected += r;
                            const double share = double(rejected) / double(estimate_rejects.size());
                            const bool reliable = estimate_rejects.size() < 30 || share <= 0.3;
                            estimate_trusted = reliable && !estimator.last_rejected();
                            sh.presenter.estimate_unreliable_pct = reliable ? 0u : static_cast<std::uint32_t>(share * 100.0 + 0.5);
                            if (reliable != estimate_was_trusted) {
                                logf(reliable ? "camera estimate reliable again: the camera model learns from it"
                                              : "camera estimate unreliable (%.0f%% of the last frames unexplained): the camera model does not learn from it meanwhile",
                                     share * 100.0);
                                estimate_was_trusted = reliable;
                            }
                        }
                        flush_ms_sum += renderer.last_flush_ms() + intake_wait_ms;
                        intake_wait_ms = 0;
                        if (estimator_runs >= 300) {
                            const double by_consensus = estimator.consensus_fraction(), counted = estimator.valid_fraction();
                            const double unexplained = estimator.take_rejected_fraction();
                            char own[96] = "";
                            if (s.motion_estimated)
                                std::snprintf(own, sizeof(own), "; own motion vectors: %.0f%% of samples confident, %.0f%% of frames fitted by consensus",
                                              counted * 100.0, by_consensus * 100.0);
                            logf("camera estimation: %.2f ms CPU + %.2f ms waiting for the samples per game frame, residual %.2f px, field of view %.1f deg%s, "
                                 "%.0f%% of frames unexplained, depth %s%s",
                                 estimator_ms_sum / estimator_runs, flush_ms_sum / estimator_runs, estimator.last_residual(),
                                 estimator.vertical_fov() * 180.0 / 3.14159265358979, estimator.fov_locked() ? "" : " (learning)",
                                 unexplained * 100.0, m.camera.depth_inverted ? "reversed" : "standard", own);
                            estimator_ms_sum = 0; flush_ms_sum = 0; estimator_runs = 0;
                            // The game's camera was found unusable: the estimate has to do better. When it explains
                            // less than half of the frames for two such stretches in a row, the game's camera stays
                            // (and is not checked again).
                            // (judged on an upscaler's motion vectors only: XPAR's own, estimated from the picture,
                            // are harder to explain and say nothing about the game's camera)
                            if (sh.game_camera_check == 1 && !s.motion_estimated) {
                                poor_estimate_windows = unexplained > 0.6 ? poor_estimate_windows + 1 : 0;
                                if (poor_estimate_windows >= 2) {
                                    InterlockedExchange(&sh.game_camera_check, 2);
                                    logf("the estimated camera explains the motion vectors no better (%.0f%% of frames unexplained): back to the game's own camera",
                                         unexplained * 100.0);
                                }
                            }
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
                // XPAR's own motion vectors (estimated from the picture) are in uv by construction: nothing to
                // learn, and their noise must not teach a scale.
                if (s.motion_estimated && !(mv_scale.valid && !mv_scale.pixel_units && mv_scale.locked[0] == 1.0 && mv_scale.locked[1] == 1.0)) {
                    mv_scale = MotionVectorScale{};
                    mv_scale.valid = true;
                    mv_scale.locked[0] = mv_scale.locked[1] = 1.0;
                    logf("no motion vectors from the game: estimating them from the picture");
                }
                incoming.s = s;
                incoming.cam = cam;
                incoming.basis = to_basis(cam);
                // Frames that come without a simulation time from XPAR's own motion path (the ReShade path):
                // timed at the game's steady pace instead of when they were presented (see FrameClock).
                const double presented = seconds(m.qpc_sim_start ? m.qpc_sim_start : m.qpc_constants);
                if (!m.qpc_sim_start && s.motion_estimated) source_time = frame_clock.next(m.frame_id, presented);
                else { source_time = presented; frame_clock.reset(); }
                source_frame = m.frame_id;
                frame_times.emplace_back(m.frame_id, source_time);
                if (frame_times.size() > 8) frame_times.pop_front();
                g_app.model.set_camera_trusted(!cam.estimated || estimate_trusted);
                if (!cam.estimated) sh.presenter.estimate_unreliable_pct = 0;
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
                // Every frame with depth and motion vectors is analysed, whatever is kept still: besides the masks
                // (and the shelved object interpolation) the analysis gives the per-frame fit that locks the motion
                // vector scale and checks the game's camera (camera_check.hpp), which every game needs. (It does
                // nothing without depth or motion vectors.)
                renderer.pass_stamp(Renderer::kPassStart);
                // Floor release (debug test): only while turns are predicted around a measured orbit pivot - its depth
                // value through the frame's projection (row vector: clip = (0, 0, z, 1) * view_to_clip).
                float floor_pivot_depth = 0.0f;
                if (settings.floor_release && settings.orbit_mode == 0 && source_was_estimated && !source_motion_estimated && g_app.model.orbit() > 0) {
                    const float* p = cam.view_to_clip;
                    const double z = g_app.model.orbit() * view_z_sign(p);
                    const double cz = z * p[10] + p[14], cw = z * p[11] + p[15];
                    if (std::fabs(cw) > 1e-9 && cz / cw > 0) floor_pivot_depth = float(cz / cw);
                }
                renderer.analyze_motion(s, cam.clip_to_prev_clip, float(mv_scale.scale(0, s.depth_rect.w)),
                                        float(mv_scale.scale(1, s.depth_rect.h)), mv_scale.valid, cam.depth_inverted != 0,
                                        settings.near_camera_rule != 0, settings.turn_rule != 0, m.frame_id, floor_pivot_depth);
                // Moving objects (option): what moves on its own, as a straight line back to the previous frame.
                // Only with the game's own motion vectors (an upscaler's): those estimated from the picture are
                // too rough at the edges of things.
                // (with the motion seen in the picture, for what the game's motion vectors miss: shadows, glare)
                // Frame generation (the game's own): objects move in its generated images instead - made
                // ready for the warp here.
                renderer.pass_stamp(Renderer::kPassAnalyze);
                if (own_engine_on() && s.generated > 0) {
                    renderer.prepare_generated(s, float(mv_scale.scale(0, s.depth_rect.w)), float(mv_scale.scale(1, s.depth_rect.h)), mv_scale.valid,
                                               settings.use_ui_tags != 0);
                    last_generated = now_seconds();
                    if (!generation_logged) {
                        logf("frame generation: showing the game's generated images (%d per frame), each moved to the displayed camera", s.per_frame);
                        generation_logged = true;
                    }
                } else if (moving_objects_on() && s.has_motion && !s.motion_estimated && mv_scale.valid) {
                    const bool picture = renderer.picture_motion();
                    renderer.object_motion(s, cam.clip_to_prev_clip, cam.view_to_clip, float(mv_scale.scale(0, s.depth_rect.w)),
                                           float(mv_scale.scale(1, s.depth_rect.h)), attached_mask, picture);
                }
                renderer.pass_stamp(Renderer::kPassObjects);
                if (mask) {
                    // HUD from the upscaler's output only when chosen (opt-in: its colour model cannot be
                    // proven for every game); otherwise the learned HUD map, which works everywhere.
                    IngestedSource for_mask = s;
                    if (!settings.hud_from_scene) for_mask.has_scene = false;
                    renderer.build_no_warp_mask(for_mask, cam.clip_to_prev_clip, hud_mask, s.has_motion && mv_scale.valid, attached_mask,
                                                cam.depth_inverted != 0, settings.hud_from_scene == 2,
                                                settings.hud_fill != 0 && (settings.warp_engine == 1 || !latewarp.ready()),
                                                settings.warp_engine == 1 || !latewarp.ready() ? static_cast<int>(settings.stretch_width) : 0);
                    // Background memory (option, XPAR engine): the scenery last seen behind held pixels and just
                    // beyond the frame, for what the warp uncovers there.
                    if (settings.background_memory && (settings.warp_engine == 1 || !latewarp.ready())) {
                        renderer.update_memory(s, cam.clip_to_prev_clip, cam.depth_inverted != 0, m.frame_id == memory_frame + 1);
                        memory_frame = m.frame_id;
                        renderer.pass_stamp(Renderer::kPassMemory);
                    }
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
                    if (hud_mask && s.has_hudless && !s.has_ui && !hudless_hud_logged) {
                        logf("HUD: found from the game's HUD-less picture (sent without a UI layer)");
                        hudless_hud_logged = true;
                    }
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
        vblank.set_display_hz(g_app.display_mode_hz.load());
        vblank.update(renderer.present_stats());
        // DWM composes ~3 ms after each vblank and shows the result at the following vblank; with a
        // frame latency of 1 the swapchain only frees us at that display vblank, leaving ~2.5 ms. With a
        // lead we allow one queued frame and wake `lead` ms before each composition instead, one frame
        // per refresh (a new target is always a later vblank than the previous one).
        if (settings.present_lead_auto && vblank.valid()) {
            const double d = auto_lead.target_ms - auto_lead.lead_ms;
            auto_lead.lead_ms += std::clamp(d, -0.01, 0.03);  // per refresh: at most 1.2 ms/s down, 3.6 ms/s up
        }
        const double lead_ms = settings.present_lead_auto ? auto_lead.lead_ms : double(settings.present_lead_ms);
        const bool paced = lead_ms > 0 && vblank.valid();
        frame_tick = frame_vblank = 0;
        paced_last = paced;
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
            const std::int64_t lead = static_cast<std::int64_t>(lead_ms * 1e-3 * f);
            const std::int64_t compose = static_cast<std::int64_t>(2.5e-3 * f);  // composition deadline after a vblank
            const std::int64_t now_q = qpc_now();
            // (Pacing to the screen's own refreshes, VRR or not: XPAR's window is always composed by Windows, so a VRR
            // screen runs at its maximum with it - a pace below that, Blur Busters' VRR rule, only repeated frames;
            // tried 2026-10-07, the TV stayed at 119 Hz with NVIDIA's windowed G-SYNC on, for the presenter as well.)
            std::int64_t v = vblank.next_after(now_q - compose + lead);  // first vblank whose deadline - lead is ahead
            if (last_target_vblank && v <= last_target_vblank + static_cast<std::int64_t>(vblank.period / 2))
                v = last_target_vblank + static_cast<std::int64_t>(vblank.period);
            last_target_vblank = v;
            // New game frames are taken in while waiting for the refresh, each as its own GPU submission, so
            // the refresh itself only warps (taking a frame in costs several ms of GPU at 4K). One that
            // arrives too close to the refresh waits until just after it.
            const std::int64_t tick = v + compose - lead;
            frame_tick = tick; frame_vblank = v;
            const std::int64_t budget = static_cast<std::int64_t>((intake_ms + 1.0) * 1e-3 * f);
            const bool can_take_in = g_app.visible && !generation_aside() && renderer.session_open(sh.session) &&
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
        // (HDR switched on or off in the game, or its colour space set after its swap chain was made)
        renderer.set_output(static_cast<DXGI_FORMAT>(sh.backbuffer_format), sh.color_space);
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
        ps.use_stick = settings.use_controller != 0;
        ps.async_fit = true;  // (the fit runs on its own thread: tens of ms with a controller)
        ps.rotation_extrapolation = settings.rotation_extrapolation;
        ps.prediction = settings.prediction_ms / 1000.0;
        // Auto: half a game frame, or a quarter when the game sends no HUD layers - the HUD/weapon mask
        // misses some pixels (semi-transparent HUD), and a shorter warp moves them less. A whole game frame
        // when XPAR estimates the motion from the picture itself (no DLSS or FSR): its camera cannot be
        // carried ahead of the newest frame as well (RE2: raw mouse input, ~100 mouse events a second to go
        // by), and showing the moment between the last two frames instead held the view on the camera
        // (slow wander in a replay: 11 px with a quarter, 3 px with a whole frame).
        // (camera as of now, 5: starts from Auto's choice, see the prediction below)
        const std::uint32_t fraction = settings.auto_prediction == 3 || settings.auto_prediction == 5 || settings.auto_prediction == 6
                                           ? (source_motion_estimated ? 1u : game_has_hud_layers ? 2u : 4u)
                                                                     : settings.auto_prediction;
        ps.auto_fraction = (fraction == 1 || fraction == 2 || fraction == 4) ? 1.0 / fraction : 0.0;
        ps.orbit_distance = settings.orbit_mode == 2 ? settings.orbit_distance : 0.0;
        // (measured orbit: only for a camera estimated from the game's own motion vectors - an upscaler's. Estimated
        // from the picture (no DLSS or FSR) the pivot wanders: RE2, spread 12-35% against Black Flag's 4%, switched
        // on and off, and turning on the spot looked steadier there)
        ps.use_measured_orbit = settings.orbit_mode == 0 && source_was_estimated && !source_motion_estimated;
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

        // Frame generation (DLSS, FSR, XeSS - any kind): two or more images presented per rendered frame, or the
        // game calling its DLSS / FSR frame generation (where ReShade sees only the rendered frames' presents).
        // Without its images coming with the frames, the two cannot be combined (generated images would be
        // warped as if they were rendered ones): step aside and say so, and come back when it is turned off.
        if (qpc_now() - fg_window_start >= g_qpc_frequency / 2) {
            const LONG presents = sh.presents_total, frames = sh.frames_total, calls = sh.generation_calls;
            const LONG dp = presents - fg_presents, df = frames - fg_frames, dc = calls - fg_calls;
            fg_presents = presents; fg_frames = frames; fg_calls = calls; fg_window_start = qpc_now();
            if (df >= 5) {  // the game is rendering (menus and loading screens present without it)
                const double ratio = double(dp) / double(df), called = double(dc) / double(df);
                if (ratio >= 1.6 || called >= 0.5) fg_votes = std::min(fg_votes + 1, 2);
                else if (ratio <= 1.25 && called < 0.1) fg_votes = std::max(fg_votes - 1, -2);
                if (!frame_generation && fg_votes >= 2) {
                    frame_generation = true;
                    logf("frame generation detected (%.1f presented images, %.1f frame generation calls per rendered frame)%s", ratio, called,
                         own_engine_on() ? "" : ": stepping aside until it is off (XPAR's own warp engine uses it)");
                } else if (frame_generation && fg_votes <= -2) {
                    frame_generation = false;
                    logf("frame generation off");
                }
            }
        }
        // With XPAR's own engine, the game's frame generation is used (its images come with the frames).
        const bool aside = generation_aside();
        // (no message while reprojection is off - the add-on takes frame generation's images only while it is
        // on - or for 2 s after it is turned back on, while they start coming again)
        if (settings.enabled && !was_enabled) enabled_since = now_seconds();
        was_enabled = settings.enabled != 0;
        const bool settling = !settings.enabled || now_seconds() - enabled_since < 2.0;
        sh.presenter.frame_generation = aside ? (settling ? 0u : 1u) : frame_generation ? 2u : 0u;
        // Frame generation off for a while: the textures its images were taken into go (video memory).
        if (renderer.has_generated() && !source.generated && !incoming.s.generated && now_seconds() - last_generated > 3.0) {
            renderer.release_generated();
            logf("frame generation: its textures released (no generated images for 3 s)");
        }
        if (!g_app.visible || aside) {  // game not in front / disabled / frame generation: don't compete with it for the GPU
            if (aside) g_app.has_frames = false;
            record_timeline(sh, settings);
            record_hooks(sh, now_seconds());
            // (while paused, what keeps it so - every 5 s)
            if (now_seconds() - last_pause_log > 5.0) {
                last_pause_log = now_seconds();
                const auto latest = static_cast<std::uint64_t>(sh.latest_ready_frame);
                int latest_generated = -1;
                for (const auto& m : sh.slots) if (m.frame_id == latest && m.state != kFree) latest_generated = int(m.generated);
                logf("paused: visible %d, stepped aside %d, reprojection %s, frame generation %d (last generated image %.1f s ago), newest frame %llu "
                     "with %d generated images, ReShade menu %s", int(g_app.visible), int(aside), settings.enabled ? "on" : "off", int(frame_generation),
                     now_seconds() - last_generated, static_cast<unsigned long long>(latest), latest_generated, sh.overlay_open ? "open" : "closed");
            }
            source_frame = sh.latest_ready_frame;
            pacing_paused = true;
            as_is_frame = 0;  // (the window may have been hidden: drawn again when it comes back)
            presented_last = true;
            Sleep(10);
            continue;
        }
        if (pacing_paused) {
            // Back from a pause (alt-tab, hidden overlay): the frame statistics and our vblank schedule are
            // stale. Start over exactly like switching the present lead off and on again.
            pacing_paused = false;
            vblank = VblankClock{};
            vblank.set_display_hz(g_app.display_mode_hz.load());
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
            // (paced: the camera as of the planned wake-up, a fixed time before the refresh it is meant for - not
            // the moment the loop got there, which varies by a fraction of a millisecond and would jitter the motion)
            const double t_camera = frame_tick && seconds(frame_tick) <= now ? seconds(frame_tick) : now;
            Prediction p = g_app.model.predict(t_camera);
            // Camera as of now (Auto latency 5): Auto's camera moved forward to this moment, the mouse up to now
            // through the learned model - as far as the uncovered edge allows: no more of the screen than
            // edge_limit_pct of its width beyond what Auto's own camera uncovers (bisection on the time; the
            // edge grows with it during a turn).
            // The limit is Auto's own turn away from the game's frame right now, per axis (left/right, up/down): the
            // warp may not uncover more at either edge - nor warp the scenery behind a held weapon further - than
            // Auto does, plus the extra allowed. (Against Auto's largest turn of the last 100 ms, quick back-and-forth
            // kept the limit high: "now" stayed turned that far all the time, and the weapon showed artifacts.) The
            // time ahead follows the allowed one gradually (at most 0.6 ms back / 0.25 ms forward per refresh):
            // jumping to it every refresh made the camera tremble.
            if (settings.auto_prediction == 5) {
                const double ahead = std::max(0.0, g_app.model.latency() - g_app.model.effective_prediction());
                const double sx = std::fabs(source_camera.view_to_clip[0]), sy = std::fabs(source_camera.view_to_clip[5]);
                auto edge_x = [&](const Prediction& q) { return 0.5 * std::tan(std::min(std::fabs(q.yaw), 1.4)) * sx; };
                auto edge_y = [&](const Prediction& q) { return 0.5 * std::tan(std::min(std::fabs(q.pitch), 1.4)) * sy; };
                const double extra = std::max(0.0f, settings.edge_limit_pct) / 100.0;
                const double limit_x = edge_x(p) + extra, limit_y = edge_y(p) + extra;
                auto within = [&](const Prediction& q) { return edge_x(q) <= limit_x && edge_y(q) <= limit_y; };
                double allowed = ahead;
                if (!settings.edge_unlimited && !within(g_app.model.predict(t_camera + ahead))) {
                    double lo = 0, hi = ahead;
                    for (int it = 0; it < 7; ++it) {
                        const double mid = 0.5 * (lo + hi);
                        if (within(g_app.model.predict(t_camera + mid))) lo = mid; else hi = mid;
                    }
                    allowed = lo;
                }
                now_ahead = std::clamp(now_ahead + std::clamp(allowed - now_ahead, -0.6e-3, 0.25e-3), 0.0, ahead);
                if (now_ahead > 0) p = g_app.model.predict(t_camera + now_ahead);
            } else if (settings.auto_prediction == 6) {
                // Lowest latency without edge fill: the latest camera time whose view stays between the game's last two
                // frames' cameras (per axis) - everything the warp reveals then was rendered by one of them (the older
                // one through background memory). From now back to one game frame before the newest. The time follows
                // gradually: back faster (1.5 ms per refresh, a flick's edge), forward slowly (0.25 ms).
                const double ahead = std::max(0.0, g_app.model.latency() - g_app.model.effective_prediction());
                const double frame = g_app.model.frame_interval();
                double previous[2] = {0, 0};
                g_app.model.previous_source_delta(previous);
                constexpr double kTolerance = 1e-3;  // radians (~0.06 degrees)
                auto covered = [&](const Prediction& q) {
                    const double d[2] = {q.yaw, q.pitch};
                    for (int k = 0; k < 2; ++k)
                        if (d[k] < std::min(0.0, previous[k]) - kTolerance || d[k] > std::max(0.0, previous[k]) + kTolerance) return false;
                    return true;
                };
                // (predict(t) shows the camera as of t - ahead: the oldest allowed is one frame before the newest frame)
                const double oldest = std::min(0.0, g_app.model.source_time() - (frame > 0 ? frame : 0.0) + ahead - t_camera);
                double allowed = ahead;
                if (!covered(g_app.model.predict(t_camera + ahead))) {
                    double lo = oldest, hi = ahead;
                    for (int it = 0; it < 8; ++it) {
                        const double mid = 0.5 * (lo + hi);
                        if (covered(g_app.model.predict(t_camera + mid))) lo = mid; else hi = mid;
                    }
                    allowed = lo;
                }
                now_ahead = std::clamp(now_ahead + std::clamp(allowed - now_ahead, -1.5e-3, 0.25e-3), oldest, ahead);
                if (now_ahead != 0) p = g_app.model.predict(t_camera + now_ahead);
            } else {
                now_ahead = 0;
            }
            g_app.model.set_display_ahead(now_ahead);
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
            const double z_sign = view_z_sign(source_camera.view_to_clip);
            if (own_engine) {
                Mat4 m = clip_source_to_target(projection, view_matrix(source_basis, origin, z_sign), view_matrix(p.camera, origin, z_sign));
                // Frame generation (the game's own): over each frame interval, the images it generated between the
                // previous frame and this one, then the frame itself - objects one game frame late, moved by the
                // game. Each image is seen from the camera of its moment: the frame's own motion from the previous
                // frame (clipToPrevClip), taken part of the way back.
                int generated = -1;
                const double gen_interval = g_app.model.frame_interval();
                if (own_engine_on() && source.generated > 0 && gen_interval > 0) {
                    // (the image numbered for this part of the interval, or the nearest one taken: 6x keeps 3 of 5)
                    const int images = source.per_frame + 1;
                    const int wanted = int(std::floor(std::max(0.0, now - shown_since) / gen_interval * images)) + 1;
                    if (wanted < images) {
                        for (int i = 0; i < source.generated; ++i)
                            if (generated < 0 || std::abs(source.gen_index[i] - wanted) < std::abs(source.gen_index[generated] - wanted)) generated = i;
                        m = generated_to_target(source_camera.clip_to_prev_clip, 1.0 - double(source.gen_index[generated]) / images, m);
                    }
                }
                // Moving objects (option), on their own clock, one game frame behind (frame generation): when a
                // frame is first shown they are where the previous one had them, and over the next frame
                // interval they move to where this one has them - always between two real frames, never guessed
                // ahead. The camera is not delayed: it is the warp's, at the displayed moment.
                Renderer::ObjectWarp objects;
                const double interval = g_app.model.frame_interval();
                objects.frames_back = interval > 0 ? float(std::clamp(1.0 - (now - shown_since) / interval, 0.0, 1.0)) : 0.0f;
                Mat4 prev_clip{}, back{};
                std::memcpy(prev_clip.data(), source_camera.clip_to_prev_clip, sizeof(float) * 16);
                const bool with_objects = moving_objects_on() && source.generated == 0 && objects.frames_back > 0 && mat_inverse(prev_clip, back);
                if (with_objects) {
                    const Mat4 to_target = mat_mul(back, m);
                    std::memcpy(objects.prev_to_target, to_target.data(), sizeof(objects.prev_to_target));
                    std::memcpy(objects.view_to_clip, source_camera.view_to_clip, sizeof(objects.view_to_clip));
                }
                // Nothing moved since the frame (well under a hundredth of a pixel anywhere): every pixel stays
                // where it is, so there is nothing to warp. (The mask's debug tint and moving objects need the
                // warp pass.)
                unmoved = !settings.show_mask && !with_objects && generated < 0;
                for (int i = 0; i < 16 && unmoved; ++i) unmoved = std::fabs(m.data()[i] - (i % 5 == 0 ? 1.0f : 0.0f)) < 2e-6f;
                warped = unmoved || renderer.own_warp(source, settings.use_ui_tags != 0, inputs.no_warp_mask != nullptr, m.data(), inputs.depth_inverted,
                                                     settings.background_memory != 0, with_objects ? &objects : nullptr, generated,
                                                     settings.edge_fill != 0);
            } else {
                warped = latewarp.evaluate(list, inputs, first_eval, view_matrix(p.camera, origin, z_sign),
                                           view_matrix(source_basis, origin, z_sign), projection);
            }
            if (warped) first_eval = false;
            if (warped && settings.show_mask && inputs.no_warp_mask) renderer.tint_mask();
        }
        // Menus and loading screens often skip the game's usual frame (no DLSS call, no tags): when several
        // presents in a row bring no frame, step aside and let the game's own picture show.
        // (the ReShade menu is drawn on what the game presents, after the generated images XPAR shows were taken:
        // with them, the game's own picture is shown while it is open)
        g_app.has_frames = source.valid && sh.presents_without_frame < 3 && !(sh.overlay_open && source.generated > 0 && own_engine_on());
        // The same picture as the one presented last - this frame shown as it is again (the camera has not
        // moved, or nothing is warped), nothing drawn on it that changes: nothing to draw or present. With a still
        // camera the GPU does nothing at all between game frames.
        const bool as_is = !(warped && !unmoved) && !settings.overlay_debug && !settings.show_mask && g_app.has_frames;
        const bool same_picture = as_is && shown_frame && as_is_frame == shown_frame && as_is_w == renderer.width() && as_is_h == renderer.height();
        if (same_picture) renderer.skip_frame();
        else renderer.finish_frame(warped && !unmoved, settings.overlay_debug ? (warped ? 1 : 2) : 0, paced_last);
        presented_last = !same_picture;
        as_is_frame = as_is ? shown_frame : 0;
        as_is_w = renderer.width(); as_is_h = renderer.height();
        // Ctrl+Shift+M marks "it looks bad now" in the log. Both keys (this and Ctrl+Shift+D below) only with
        // "Record detailed diagnostics" ticked: pressed by accident, a capture freezes the game for a second.
        const bool keys_on = settings.record_diagnostics != 0;
        const bool mark_down = keys_on && (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) && (GetAsyncKeyState('M') & 0x8000);
        const int mark = mark_down && !g_app.mark_was_down ? 1 : 0;
        g_app.mark_was_down = mark_down;
        if (mark) logf("MARK (user flagged a bad moment)");
        // Ctrl+Shift+D (development): saves the newest game frame, the warped output, the upscaler's output,
        // depth, motion vectors, the motion analysis and the masks (half-float RGBA, 8-byte header: width,
        // height) to captures\.
        const bool dump_down = keys_on && (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) && (GetAsyncKeyState('D') & 0x8000);
        // The key keeps the current game frame's picture, depth and motion vectors on the GPU (quick), and the
        // next game frame's too; the capture itself is written on the one after, together with the two kept
        // ("prev2_", "prev_"): three consecutive game frames, and their cameras (_cameras.txt). (Reading
        // everything back and writing it stalls the presenter for about a second, so the frame after a capture
        // would be dozens of frames later.)
        auto dump_frame = [&]() {
            DumpFrame d;
            d.frame = shown_frame; d.cam = source_camera; d.s = source; d.mv_valid = mv_scale.valid;
            for (const auto& [f, t] : frame_times) if (f == shown_frame) d.time = t;
            for (int k = 0; k < 2; ++k) d.mv[k] = mv_scale.scale(k, k ? source.depth_rect.h : source.depth_rect.w);
            return d;
        };
        // (only while no newer frame is being taken in: with split queues its motion vectors would already be
        // in place of the shown frame's)
        const bool dump_now = source.valid && !renderer.intake_pending() && intake.slot < 0;
        if (dump_down && !g_app.dump_was_down && !dump_stage) dump_requested = true;
        if (dump_requested && dump_now) {
            dump_requested = false;
            renderer.stash_source(1, source.has_hudless);
            dump_frames[0] = dump_frame();
            dump_stage = 1; dump_after = shown_frame;
        } else if (dump_stage == 1 && dump_now && shown_frame != dump_after) {
            renderer.stash_source(0, source.has_hudless);
            dump_frames[1] = dump_frame();
            dump_stage = 2; dump_after = shown_frame;
        } else if (dump_stage == 2 && dump_now && shown_frame != dump_after) {
            dump_stage = 0;
            dump_frames[2] = dump_frame();
            static int captures = -1;
            const auto dir = g_app.data_dir / L"captures";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (captures < 0) {  // (continue after the captures already there: a new session used to overwrite them)
                captures = 0;
                for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
                    const std::wstring name = entry.path().filename().wstring();
                    if (name.rfind(L"capture_", 0) == 0) captures = std::max(captures, _wtoi(name.c_str() + 8));
                }
            }
            const int n = ++captures;
            for (const auto& [which, name] : {std::pair{0, L"frame"}, std::pair{1, L"output"}, std::pair{2, L"scene"}, std::pair{3, L"depth"},
                                              std::pair{4, L"motion"}, std::pair{5, L"object"}, std::pair{6, L"mask"}, std::pair{7, L"hudscore"},
                                              std::pair{8, L"world"}, std::pair{9, L"fill"}, std::pair{10, L"prev_frame"},
                                              std::pair{11, L"prev_depth"}, std::pair{12, L"prev_motion"}, std::pair{14, L"prev_hudless"},
                                              std::pair{15, L"prev2_frame"}, std::pair{16, L"prev2_hudless"}, std::pair{17, L"prev2_depth"},
                                              std::pair{18, L"prev2_motion"}, std::pair{19, L"hudless"}, std::pair{20, L"gen0"},
                                              std::pair{21, L"gen1"}, std::pair{22, L"gen2"}, std::pair{23, L"gen0_depth"},
                                              std::pair{24, L"gen1_depth"}, std::pair{25, L"gen2_depth"}, std::pair{26, L"ui"}}) {
                std::vector<std::uint16_t> px;
                std::uint32_t cw = 0, ch = 0;
                if (which == 2 && !source.has_scene) { logf("capture %d: no upscaler output this frame", n); continue; }
                if (which == 19 && !source.has_hudless) continue;
                if (which >= 20 && which <= 25 && (which - 20) % 3 >= source.generated) continue;  // (frame generation's images)
                if (which == 26 && !source.has_ui) continue;
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
            // Per frame (prev2, prev, current): the frame, its time, the rectangles of the picture and the depth, the
            // motion vector scale (to uv), depth_inverted, then the projection and the game's reprojection to the
            // previous frame (row vector: prev = clip * M).
            FILE* f = nullptr;
            if (_wfopen_s(&f, (dir / (L"capture_" + std::to_wstring(n) + L"_cameras.txt")).c_str(), L"w") == 0 && f) {
                static const char* const kNames[] = {"prev2", "prev", "current"};
                for (int i = 0; i < 3; ++i) {
                    const DumpFrame& d = dump_frames[i];
                    std::fprintf(f, "%s frame %llu time %.6f color_rect %u %u %u %u depth_rect %u %u %u %u mv_scale %.9g %.9g %d depth_inverted %u estimated %u hudless %d generated %d per_frame %d\n",
                                 kNames[i], static_cast<unsigned long long>(d.frame), d.time, d.s.color_rect.x, d.s.color_rect.y, d.s.color_rect.w,
                                 d.s.color_rect.h, d.s.depth_rect.x, d.s.depth_rect.y, d.s.depth_rect.w, d.s.depth_rect.h, d.mv[0], d.mv[1],
                                 int(d.mv_valid), d.cam.depth_inverted, d.cam.estimated, int(d.s.has_hudless), d.s.generated, d.s.per_frame);
                    std::fprintf(f, "%s view_to_clip", kNames[i]);
                    for (float v : d.cam.view_to_clip) std::fprintf(f, " %.9g", v);
                    std::fprintf(f, "\n%s clip_to_prev_clip", kNames[i]);
                    for (float v : d.cam.clip_to_prev_clip) std::fprintf(f, " %.9g", v);
                    std::fprintf(f, "\n");
                }
                std::fclose(f);
            }
            logf("capture %d saved: game frames %llu (prev2_), %llu (prev_), %llu", n, static_cast<unsigned long long>(dump_frames[0].frame),
                 static_cast<unsigned long long>(dump_frames[1].frame), static_cast<unsigned long long>(shown_frame));
        }
        g_app.dump_was_down = dump_down;
        if (rec(g_app.csv_outputs)) {
            const auto tm = renderer.last_timing();
            const auto pstats = renderer.present_stats();
            // (the displayed camera too: position and forward, in the game's world)
            const auto& dc = applied.camera;
            wr(g_app.csv_outputs, "%lld,%llu,%d,%.6f,%.6f,%.4f,%.3f,%d,%lld,%lld,%lld,%u,%u,%u,%u,%lld,%ld,%lld,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f\n",
                         qpc_now(), static_cast<unsigned long long>(source_frame), same_picture ? 2 : int(warped), applied.yaw, applied.pitch,
                         applied.horizon * 1000.0, renderer.last_gpu_ms(), mark, tm.submit, tm.gpu_start, tm.gpu_end, pstats.last_present_count,
                         pstats.present_count, pstats.present_refresh, pstats.sync_refresh, pstats.sync_qpc, static_cast<long>(pstats.hr), wake_qpc,
                         dc.pos.x, dc.pos.y, dc.pos.z, dc.fwd.x, dc.fwd.y, dc.fwd.z);
        }
        record_timeline(sh, settings);
        // (a refresh: a present, or with pacing a refresh that keeps the same picture - not the 1 ms polls of the
        // unpaced loop while nothing changes)
        if (!same_picture || paced_last) ++stat_frames;
        present_call_ms += renderer.last_present_call_ms();
        {
            const auto ps = renderer.present_stats();
            if (!same_picture && warped && ps.last_present_count) {
                const double camera_t = g_app.model.source_time() + applied.horizon;
                lat_behind.push_back((now - camera_t) * 1000.0);
                shown_presents[ps.last_present_count % 64] = {ps.last_present_count, now_seconds(), camera_t, false, frame_vblank};
                if (ps.hr == S_OK && ps.present_count && ps.last_present_count >= ps.present_count)
                    lat_queued.push_back(double(ps.last_present_count - ps.present_count));
            }
            if (!same_picture && frame_tick) { planned[planned_at] = {renderer.last_submit_qpc(), frame_tick}; planned_at = (planned_at + 1) % 16; }
            // GPU timing comes back a few frames later: matched with its frame's planned wake-up by submit time.
            const auto tm = renderer.last_timing();
            if (tm.valid && tm.submit != timing_seen) {
                timing_seen = tm.submit;
                for (const auto& pf : planned)
                    if (pf.submit == tm.submit && tm.gpu_end > pf.tick) {
                        auto_lead.ready.push_back(double(tm.gpu_end - pf.tick) * 1000.0 / double(g_qpc_frequency));
                        break;
                    }
            }
            // Presents replaced before they were shown (more presents than refreshes showing them between two
            // statistics): with the newest frame shown at each refresh, a frame that missed its refresh.
            if (ps.hr == S_OK && ps.present_count && ps.present_count != auto_lead.stat_count) {
                if (auto_lead.stat_count && ps.present_count > auto_lead.stat_count && ps.present_refresh > auto_lead.stat_refresh) {
                    const UINT presents = ps.present_count - auto_lead.stat_count, refreshes = ps.present_refresh - auto_lead.stat_refresh;
                    if (presents > refreshes && presents - refreshes < 8) auto_lead.late += static_cast<int>(presents - refreshes);
                }
                auto_lead.stat_count = ps.present_count; auto_lead.stat_refresh = ps.present_refresh;
            }
            if (ps.hr == S_OK && ps.sync_qpc && ps.present_count) {
                auto& e = shown_presents[ps.present_count % 64];
                if (e.count == ps.present_count && !e.done) {
                    e.done = true;
                    const double screen_t = seconds(ps.sync_qpc);
                    if (screen_t >= e.present_t && screen_t - e.present_t < 0.25) {
                        lat_display.push_back((screen_t - e.present_t) * 1000.0);
                        lat_age.push_back((screen_t - e.camera_t) * 1000.0);
                    }
                    if (e.target && vblank.period > 0) {
                        const long long r = std::llround(double(ps.sync_qpc - e.target) / vblank.period);
                        if (r >= 0 && r < 8) {
                            ++auto_lead.refreshes[r]; ++auto_lead.matched;
                            // Late (a missed refresh): shown a refresh later than the present just before it.
                            if (auto_lead.prev_count + 1 == ps.present_count && r > auto_lead.prev_r) ++auto_lead.late;
                            auto_lead.prev_count = ps.present_count; auto_lead.prev_r = int(r);
                        }
                    }
                }
            }
            // Once a second of matched presents: the usual refresh count, and the lead from the warp's measured time.
            if (auto_lead.matched >= 120) {
                int mode = 0;
                for (int k = 1; k < 8; ++k) if (auto_lead.refreshes[k] > auto_lead.refreshes[mode]) mode = k;
                if (auto_lead.usual >= 0)
                    auto_lead.margin_ms = auto_lead.late ? std::min(auto_lead.margin_ms + (auto_lead.late > 2 ? 0.5 : 0.25), 4.0)
                                                         : std::max(auto_lead.margin_ms - 0.05, 0.5);
                // (the usual count only from a window that mostly agrees: one of mostly late frames would hide them)
                if (auto_lead.usual < 0 || auto_lead.refreshes[mode] * 10 >= auto_lead.matched * 9) auto_lead.usual = mode;
                if (auto_lead.ready.size() >= 60) {
                    auto& v = auto_lead.ready;
                    const std::size_t k = std::min(v.size() - 1, v.size() * 99 / 100);
                    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
                    auto_lead.target_ms = std::clamp(v[k] + auto_lead.margin_ms, 1.0, 8.0);
                    v.clear();
                }
                auto_lead.late_logged += auto_lead.late; auto_lead.matched_logged += auto_lead.matched;
                std::fill(std::begin(auto_lead.refreshes), std::end(auto_lead.refreshes), 0);
                auto_lead.matched = auto_lead.late = 0;
            }
        }
        log_present_call_max = std::max(log_present_call_max, double(renderer.last_present_call_ms()));

        {
            const auto notes = renderer.take_notes();
            for (const auto& note : notes) logf("%s", note.c_str());
            MotionFit fit;
            const bool have_fit = renderer.take_motion_fit(fit);
            if (have_fit && rec(g_app.csv_motion)) {
                // How well the game's motion vectors match the camera motion computed from depth (1: exactly, for
                // the static scene), with the locked scale; a frame whose depth does not belong to its picture
                // drops here. (Fits come back a few frames later, each with its own frame.)
                const std::uint64_t f = fit.frame;
                double agree[2] = {-1, -1};
                for (int k = 0; k < 2 && mv_scale.valid; ++k) {
                    const double sc = mv_scale.scale(k, k ? source.depth_rect.h : source.depth_rect.w);
                    if (fit.cc[k] > 0) agree[k] = 1.0 - (sc * sc * fit.gg[k] - 2 * sc * fit.gc[k] + fit.cc[k]) / fit.cc[k];
                }
                // camera_fit: how much of the camera's motion the motion vectors explain at their best scale
                // (needs no locked scale; -1: the camera stood still), see camera_check.hpp.
                double camera_fit = -1;
                if (!CameraCheck::quality(fit.gc, fit.gg, fit.cc, fit.samples, camera_fit)) camera_fit = -1;
                wr(g_app.csv_motion, "%lld,%llu,%.0f,%.4f,%.4f,%.4f,%u,%u,%.4f\n", qpc_now(), static_cast<unsigned long long>(f), fit.samples,
                             fit.samples > 0 ? fit.moving / fit.samples : 0.0, agree[0], agree[1], source.depth_rect.w, source.depth_rect.h,
                             camera_fit);
            }
            if (have_fit && !source.motion_estimated && mv_scale.add(fit, source.depth_rect.w, source.depth_rect.h))
                logf("motion vectors locked: %s, scale %.4g x %.4g (sign %+.0f %+.0f), fit quality %.3f",
                     mv_scale.pixel_units ? "render pixels" : std::fabs(std::fabs(mv_scale.locked[0]) - 1) < 1e-9 ? "uv" :
                     std::fabs(std::fabs(mv_scale.locked[0]) - 0.5) < 1e-9 ? "ndc" : "custom units", mv_scale.scale(0, source.depth_rect.w),
                     mv_scale.scale(1, source.depth_rect.h), mv_scale.locked[0] < 0 ? -1.0 : 1.0, mv_scale.locked[1] < 0 ? -1.0 : 1.0,
                     mv_scale.quality);
            // The game's own camera against its motion vectors (camera_check.hpp): when they never agree
            // while the camera moves, the camera is estimated from the motion vectors from here on.
            if (have_fit && have_source_kind && !source_was_estimated && g_app.shared->game_camera_check == 0 &&
                camera_check.verdict == CameraCheck::kUndecided) {
                const auto verdict = camera_check.add(fit.gc, fit.gg, fit.cc, fit.samples, mv_scale.valid);
                if (verdict == CameraCheck::kTrusted)
                    logf("the game's camera matches its motion vectors");
                else if (verdict == CameraCheck::kUnusable) {
                    InterlockedExchange(&g_app.shared->game_camera_check, 1);
                    poor_estimate_windows = 0;
                    logf("the game's camera does not match its motion vectors (%d of %d frames with camera motion): "
                         "estimating the camera from the motion vectors instead", camera_check.consistent, camera_check.frames);
                }
            }
            vram.poll(now, notes.empty() ? nullptr : "texture change");
            sh.presenter.vram_pressure = vram.pressure_until >= now ? 1u : 0u;
            sh.presenter.vram_adapter_mb = float(vram.total_mb); sh.presenter.vram_adapter_size_mb = float(vram.size_mb);
            sh.presenter.vram_own_mb = float(vram.own_mb); sh.presenter.vram_budget_mb = float(vram.budget_mb);
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
            st.orbit_cm = float(g_app.model.consistent_orbit());
            st.orbit_in_use = settings.orbit_mode == 0 && source_was_estimated && !source_motion_estimated && g_app.model.orbit() > 0;
            st.frame_interval_ms = float(g_app.model.frame_interval() * 1000.0);
            st.effective_prediction_ms = float(g_app.model.effective_prediction() * 1000.0);
            st.mv_scale_x = mv_scale.valid ? float(mv_scale.scale(0, source.depth_rect.w)) : 0.0f;
            st.mv_scale_y = mv_scale.valid ? float(mv_scale.scale(1, source.depth_rect.h)) : 0.0f;
            st.mv_fit_quality = float(mv_scale.quality);
            st.moving_fraction = float(mv_scale.moving_fraction);
            if (moving_objects_on() && mv_scale.valid && now - last_mv_log > 10.0) {
                logf("moving objects: motion vectors %s, last fit quality %.3f, moving pixels %.1f%%",
                     mv_scale.pixel_units ? "in render pixels" : "in custom units", mv_scale.quality, mv_scale.moving_fraction * 100.0);
                last_mv_log = now;
            }
            // What the presenter costs the GPU (shared with the game), every 10 s.
            if (now - last_usage_log >= 10.0) {
                const Renderer::GpuUsage u = renderer.take_gpu_usage();
                const double span = last_usage_log > 0 ? now - last_usage_log : 0.0;
                last_usage_log = now;
                // (the HUD/masks/motion part of it, pass by pass)
                const Renderer::PassTimes pt = renderer.take_pass_times();
                if (pt.frames) {
                    const double k = 1.0 / pt.frames;
                    logf("game frame GPU by pass (ms, %u frames): motion analysis %.2f, moving objects / generated images %.2f, HUD %.2f, "
                         "character/weapon + stretch %.2f, mask %.2f, background memory %.2f", pt.frames, pt.ms[Renderer::kPassAnalyze] * k,
                         pt.ms[Renderer::kPassObjects] * k, pt.ms[Renderer::kPassHud] * k, pt.ms[Renderer::kPassCharacter] * k,
                         pt.ms[Renderer::kPassMask] * k, pt.ms[Renderer::kPassMemory] * k);
                }
                if (span > 0 && (u.intakes || u.warps)) {
                    const double n = std::max<double>(1, u.split);
                    logf("presenter GPU: %.0f%% of the time | game frames %.1f/s, %.2f ms each (reaching the game's textures %.2f, depth+motion %.2f, "
                         "colour copies %.2f, HUD/masks/motion %.2f) | refreshes %.1f/s drawn (%.2f ms each), %.1f/s with nothing new (not drawn)",
                         (u.intake_ms + u.warp_ms) / (span * 10.0), u.intakes / span, u.intakes ? u.intake_ms / u.intakes : 0.0,
                         u.access_ms / n, u.depth_ms / n, u.colour_ms / n, u.rest_ms / n, u.warps / span, u.warps ? u.warp_ms / u.warps : 0.0,
                         u.skipped / span);
                }
            }
            {
                LARGE_INTEGER qf; QueryPerformanceFrequency(&qf);
                st.display_hz = vblank.period > 0 ? float(double(qf.QuadPart) / vblank.period) : 0.0f;
            }
            st.fit_quality_x = float(px.quality); st.fit_quality_y = float(py.quality);
            {
                // (the camera model, when it changes notably: gain by a fifth, quality by 0.1)
                static double logged[4] = {0, 0, -1, -1};
                auto changed = [](double a, double b) { return (a == 0) != (b == 0) || (a != 0 && std::fabs(b / a - 1.0) > 0.2); };
                if (changed(logged[0], px.gain) || changed(logged[1], py.gain) || std::fabs(logged[2] - px.quality) > 0.1 ||
                    std::fabs(logged[3] - py.quality) > 0.1) {
                    logf("camera model: yaw gain %.4g (quality %.2f, smoothing %.0f ms, delay %.0f ms), pitch gain %.4g (quality %.2f, smoothing %.0f ms, delay %.0f ms)",
                         px.gain, px.quality, px.tau * 1000.0, px.delay * 1000.0, py.gain, py.quality, py.tau * 1000.0, py.delay * 1000.0);
                    logged[0] = px.gain; logged[1] = py.gain; logged[2] = px.quality; logged[3] = py.quality;
                }
            }
            st.controller = g_app.controller ? 1u : 0u;
            st.stick_gain_x = float(px.stick_gain); st.stick_gain_y = float(py.stick_gain);
            st.stick_power_x = float(px.stick_power); st.stick_power_y = float(py.stick_power);
            st.calibrated_x = px.fitted; st.calibrated_y = py.fitted;
            st.frames_presented += stat_frames; st.sources_consumed += stat_sources;
            st.heartbeat_qpc = qpc_now();
            if (!source.valid) set_status("Waiting for frames (need Streamline tags + camera constants)");
            else if (!latewarp.ready()) set_status("Latewarp unavailable: %s", latewarp.status().c_str());
            else if (!source.has_depth) set_status("No depth tag from the game: showing unwarped frames");
            else set_status("%s | queue %s, process %s | HUD split %s | mouse %s | %s", warped ? "warping" : "passthrough", renderer.queue_priority(), g_gpu_priority,
                            (source.has_hudless && source.has_ui) ? "yes" : "no",
                            !g_app.model.mouse_gate() ? "cursor visible (ignored)" : g_app.model.camera_follows_mouse() ? "camera" : "camera not following (ignored)", latewarp.status().c_str());
            {
                // Held back from outside: under 80% of the refresh rate, and the presents themselves took more
                // than half of each output frame's time - for 3 s in a row (6 windows); cleared the same way.
                const double span_ms = (now - stat_start) * 1000.0;
                const bool capped = st.display_hz > 1.0f && stat_frames >= 2 && st.output_fps < 0.8f * st.display_hz &&
                                    present_call_ms > 0.5 * span_ms;
                capped_windows = capped ? std::min(capped_windows + 1, 12u) : (capped_windows > 6 ? 6u : capped_windows > 0 ? capped_windows - 1 : 0u);
                const std::uint32_t before = st.outside_cap_fps;
                st.outside_cap_fps = capped_windows >= 6 ? std::max(1u, std::uint32_t(st.output_fps + 0.5f)) : 0u;
                if (!before && st.outside_cap_fps)
                    logf("output held at %u fps from outside XPAR (the presents themselves are held: a frame rate cap of the graphics driver?)", st.outside_cap_fps);
                else if (before && !st.outside_cap_fps)
                    logf("output no longer held from outside XPAR");
                log_swap_wait_ms += swap_wait_ms; log_present_call_ms += present_call_ms; log_presents += stat_frames;
                swap_wait_ms = present_call_ms = 0;
                if (now - log_since >= 5.0) {
                    auto median = [](std::vector<double>& v) {
                        if (v.empty()) return 0.0;
                        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
                        return v[v.size() / 2];
                    };
                    st.lat_behind_ms = float(median(lat_behind)); st.lat_display_ms = float(median(lat_display));
                    st.lat_age_ms = float(median(lat_age));
                    st.presents_queued = lat_queued.empty() ? 0.0f : float(median(lat_queued));
                    st.present_lead_ms = paced_last ? float(settings.present_lead_auto ? auto_lead.lead_ms : settings.present_lead_ms) : 0.0f;
                    if (!lat_behind.empty())
                        logf("latency: game %.1f ms (input to frame, p90) | displayed camera %.1f ms behind when rendered | present to screen %.1f ms "
                             "| camera age on screen %.1f ms (medians; %zu presents matched) | present lead %.2f ms (%s, margin %.2f) | "
                             "presents queued %.1f | late %d of %d (usual: shown %d refreshes after the target vblank)",
                             g_app.model.latency() * 1000.0, st.lat_behind_ms, st.lat_display_ms, st.lat_age_ms, lat_age.size(), st.present_lead_ms,
                             settings.present_lead_auto ? "automatic" : "manual", auto_lead.margin_ms, st.presents_queued, auto_lead.late_logged,
                             auto_lead.matched_logged, auto_lead.usual);
                    {
                        const auto& hk = sh.hooks;
                        if (sh.settings.frame_cap && hk.cap_reflex) logf("prevent GPU queueing: standing aside, the game's NVIDIA Reflex is on");
                        else if (hk.cap_active)
                            logf("prevent GPU queueing: game held to %.1f fps (now %.1f fps) | GPU done %.1f ms after the game's present, %.1f ms after its frame start | held %.1f ms per frame | queued %.0f%%",
                                 hk.cap_fps, hk.cap_game_fps, hk.cap_present_to_done_ms, hk.cap_start_to_done_ms, hk.cap_wait_ms, hk.cap_queued_pct);
                    }
                    lat_behind.clear(); lat_display.clear(); lat_age.clear(); lat_queued.clear();
                    auto_lead.late_logged = auto_lead.matched_logged = 0;
                    const double span = now - log_since;
                    logf("refresh loop: %.1f presents/s | per present: Present call %.2f ms (max %.1f), swapchain wait %.2f ms | display %.1f Hz",
                         log_presents / span, log_presents ? log_present_call_ms / log_presents : 0.0, log_present_call_max,
                         log_presents ? log_swap_wait_ms / log_presents : 0.0, st.display_hz);
                    log_swap_wait_ms = log_present_call_ms = log_present_call_max = 0; log_presents = 0; log_since = now;
                }
            }
            record_hooks(sh, now);
            stat_frames = stat_sources = 0; stat_start = now;
            if (now - last_profile_save > 10.0) { save_profile(); last_profile_save = now; }
        }
    }
    renderer.wait_idle();
    for (const auto& entry : held) InterlockedCompareExchange(&sh.slots[entry.slot].state, kFree, kReading);
    save_profile();
    latewarp.shutdown();
}

// The refresh rate of the display a window is on, as its current mode has it (exact: 119.88 Hz is 120000/1001),
// from the display configuration; the whole-hertz value of the display settings if that is not available.
double display_refresh_hz(HMONITOR monitor) {
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    if (!monitor || !GetMonitorInfoW(monitor, &mi)) return 0;
    UINT32 path_count = 0, mode_count = 0;
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) == ERROR_SUCCESS) {
        std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
        std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
        if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) == ERROR_SUCCESS) {
            for (UINT32 i = 0; i < path_count; ++i) {
                DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
                source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
                source.header.size = sizeof(source);
                source.header.adapterId = paths[i].sourceInfo.adapterId;
                source.header.id = paths[i].sourceInfo.id;
                if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS || wcscmp(source.viewGdiDeviceName, mi.szDevice) != 0) continue;
                const auto& r = paths[i].targetInfo.refreshRate;
                if (r.Numerator && r.Denominator) return double(r.Numerator) / double(r.Denominator);
            }
        }
    }
    DEVMODEW dm{};
    dm.dmSize = sizeof(dm);
    if (EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency > 1) return double(dm.dmDisplayFrequency);
    return 0;
}

void follow_game_window() {
    // The display's refresh rate: read again when the game is on another display, and every second (a mode change).
    {
        const HMONITOR monitor = MonitorFromWindow(g_app.game, MONITOR_DEFAULTTONEAREST);
        const double now = now_seconds();
        if (monitor != g_app.display_monitor || now - g_app.display_checked >= 1.0) {
            const double hz = display_refresh_hz(monitor);
            const double before = g_app.display_mode_hz.load();
            if (std::fabs(hz - before) > 1e-3) logf("display refresh rate: %.3f Hz (its current mode)%s", hz, hz > 0 ? "" : " - not known: measured from the presents");
            g_app.display_mode_hz = hz;
            g_app.display_monitor = monitor;
            g_app.display_checked = now;
        }
    }
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
    std::thread controller(controller_thread);
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    g_app.running = false;
    renderer.join();
    controller.join();
    if (g_app.shared) g_app.shared->presenter.pid = 0;
    logf("exit");
    g_files.stop();
    for (FILE* csv : {g_app.csv_events, g_app.csv_mouse, g_app.csv_sources, g_app.csv_outputs, g_app.csv_motion, g_app.csv_stick, g_app.csv_hooks, g_app.csv_upscaler}) if (csv) std::fclose(csv);
    if (g_app.log) std::fclose(g_app.log);
    if (single) CloseHandle(single);
    return 0;
}
