// FrameWarp ReShade add-on: captures Streamline camera/depth/HUD-less/UI data and the backbuffer
// into shared slots, launches FrameWarpPresenter, and exposes settings/status in the ReShade overlay.
#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>
#include "addon/producer.hpp"
#include "addon/game_probe.hpp"
#include "addon/streamline_hooks.hpp"
#include "addon/ngx_hooks.hpp"
#include "addon/ffx_hooks.hpp"
#include <d3d12.h>
#include <cstdio>
#include <memory>
#include <string>

using namespace reshade::api;

namespace {
std::unique_ptr<fw::Producer> g_producer;
HMODULE g_module = nullptr;
bool g_presenter_launched = false;
HANDLE g_presenter_process = nullptr;
std::wstring g_presenter_path;

std::uint32_t to_dxgi_color_space(color_space cs) {
    switch (cs) {
        case color_space::scrgb: return 1;       // DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709
        case color_space::hdr10_pq: return 12;   // DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
        case color_space::hdr10_hlg: return 18;  // DXGI_COLOR_SPACE_YCBCR_FULL_GHLG_TOPLEFT_P2020
        default: return 0;                       // DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
    }
}

void launch_presenter() {
    if (g_presenter_launched) return;
    g_presenter_launched = true;
    // Next to the add-on first. Some games (e.g. RE9's launcher) copy the DLLs from the game folder into a
    // staging folder and load them from there without subfolders, so also look next to the game's exe.
    wchar_t exe[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring exe_dir = exe;
    exe_dir = exe_dir.substr(0, exe_dir.find_last_of(L"\\/"));
    const std::wstring candidates[] = {g_presenter_path, exe_dir + L"\\FrameWarp\\FrameWarpPresenter.exe"};
    const std::wstring args = L" --pid " + std::to_wstring(GetCurrentProcessId());
    DWORD error = ERROR_FILE_NOT_FOUND;
    for (const auto& path : candidates) {
        if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) continue;
        std::wstring cmd = L"\"" + path + L"\"" + args;
        const std::wstring dir = path.substr(0, path.find_last_of(L"\\/"));
        STARTUPINFOW si{sizeof(si)};
        PROCESS_INFORMATION pi{};
        if (CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, dir.c_str(), &si, &pi)) {
            CloseHandle(pi.hThread);
            g_presenter_process = pi.hProcess;
            return;
        }
        error = GetLastError();
    }
    char text[512];
    std::snprintf(text, sizeof(text), "Failed to start the presenter (Windows error %lu). Looked in: %ls ; %ls", error,
                  candidates[0].c_str(), candidates[1].c_str());
    g_producer->set_message(text);
}

bool presenter_running() {
    if (g_presenter_process && WaitForSingleObject(g_presenter_process, 0) == WAIT_TIMEOUT) return true;
    // Started some other way (e.g. by hand): adopt the presenter that registered in shared memory.
    const LONG pid = g_producer && g_producer->shared() ? g_producer->shared()->presenter.pid : 0;
    if (!pid) return false;
    if (HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(pid))) {
        if (WaitForSingleObject(h, 0) == WAIT_TIMEOUT) {
            if (g_presenter_process) CloseHandle(g_presenter_process);
            g_presenter_process = h;
            return true;
        }
        CloseHandle(h);
    }
    return false;
}

void on_init_swapchain(swapchain* sc, bool) {
    device* dev = sc->get_device();
    if (!g_producer) return;
    if (dev->get_api() == device_api::d3d12) {
        if (!g_producer->attach(reinterpret_cast<ID3D12Device*>(dev->get_native()))) return;
    } else if (dev->get_api() == device_api::vulkan) {
        if (!g_producer->attach_vulkan(dev)) return;
    } else {
        return;
    }
    const auto desc = dev->get_resource_desc(sc->get_current_back_buffer());
    g_producer->set_swapchain(static_cast<HWND>(sc->get_hwnd()), desc.texture.width, desc.texture.height,
                              static_cast<DXGI_FORMAT>(desc.texture.format), to_dxgi_color_space(sc->get_color_space()));
    fw::install_ngx_hooks(g_producer.get());
    if (dev->get_api() == device_api::d3d12) {  // (Streamline and FSR: D3D12 only for now)
        fw::install_streamline_hooks(g_producer.get());
        fw::install_ffx_hooks(g_producer.get());
    }
    // The settings remembered per game (ReShade.ini, [FrameWarp]).
    int from_scene = 2, record = 0, near_rule = 1;
    if (g_producer->shared() && reshade::get_config_value(nullptr, "FrameWarp", "HudFromDlssOutput", from_scene))
        g_producer->shared()->settings.hud_from_scene = from_scene == 2 ? 2 : (from_scene != 0 ? 1 : 0);
    if (g_producer->shared() && reshade::get_config_value(nullptr, "FrameWarp", "NearCameraRule", near_rule))
        g_producer->shared()->settings.near_camera_rule = near_rule != 0;
    if (g_producer->shared() && reshade::get_config_value(nullptr, "FrameWarp", "RecordDiagnostics", record))
        g_producer->shared()->settings.record_diagnostics = record != 0;
}

// The ReShade menu takes the mouse and is drawn on the final frame only (not in a game's HUD layers): while
// it is open the presenter shows the game's frame as it is and ignores the mouse.
bool on_open_overlay(effect_runtime*, bool open, reshade::api::input_source) {
    if (g_producer && g_producer->shared()) InterlockedExchange(&g_producer->shared()->overlay_open, open ? 1 : 0);
    return false;  // never block the menu
}

// Vulkan games: what the add-on copies must allow copies (transfer source). Games rarely give that to depth
// buffers and motion vectors, which they only sample; add it where DLSS takes its inputs from.
bool on_create_resource(device* dev, resource_desc& desc, subresource_data*, resource_usage) {
    if (dev->get_api() != device_api::vulkan || desc.type != resource_type::texture_2d || desc.texture.samples != 1) return false;
    if ((desc.usage & resource_usage::copy_source) != 0) return false;
    const bool depth = (desc.usage & resource_usage::depth_stencil) != 0;
    const bool two_channel = desc.texture.format == format::r16g16_float || desc.texture.format == format::r32g32_float;
    const bool storage = (desc.usage & resource_usage::unordered_access) != 0;  // upscaler outputs
    if (!depth && !two_channel && !storage) return false;
    desc.usage |= resource_usage::copy_source;
    return true;
}

// Vulkan: the images imported into the game's device go before the device does.
void on_destroy_device(device* dev) {
    if (g_producer && dev->get_api() == device_api::vulkan) g_producer->detach_vulkan();
}

// Vulkan games: copies into the ReShade queue's immediate command buffer, then a timeline signal.
void present_vulkan(effect_runtime* runtime, command_queue* queue) {
    fw::install_ngx_hooks(g_producer.get());
    const resource bb = runtime->get_current_back_buffer();
    const auto desc = queue->get_device()->get_resource_desc(bb);
    command_list* cl = queue->get_immediate_command_list();
    const auto token = g_producer->begin_present_vk(bb.handle, static_cast<DXGI_FORMAT>(desc.texture.format), desc.texture.width,
                                                    desc.texture.height, reinterpret_cast<void*>(cl->get_native()));
    queue->flush_immediate_command_list();
    g_producer->finish_present_vk(queue, token);
    if (token && g_producer->shared()->settings.enabled && !presenter_running()) launch_presenter();
}

// reshade_present runs after ReShade has drawn its effects and menu, so the captured frame (which
// the presenter shows on top of the game) still contains the ReShade UI.
void on_reshade_present(effect_runtime* runtime) {
    command_queue* queue = runtime->get_command_queue();
    if (!g_producer || !g_producer->ready() || !queue) return;
    if (queue->get_device()->get_api() == device_api::vulkan) {
        if (g_producer->vulkan()) present_vulkan(runtime, queue);
        return;
    }
    if (queue->get_device()->get_api() != device_api::d3d12) return;
    fw::install_streamline_hooks(g_producer.get());  // no-op once everything is hooked
    fw::install_ngx_hooks(g_producer.get());
    fw::install_ffx_hooks(g_producer.get());
    static unsigned probe_counter = 0;
    if ((probe_counter++ % 120) == 0) fw::probe_streamline_features();
    auto* bb = reinterpret_cast<ID3D12Resource*>(runtime->get_current_back_buffer().handle);
    command_list* cl = queue->get_immediate_command_list();
    const auto token = g_producer->begin_present(bb, reinterpret_cast<ID3D12GraphicsCommandList*>(cl->get_native()));
    queue->flush_immediate_command_list();
    g_producer->finish_present(reinterpret_cast<ID3D12CommandQueue*>(queue->get_native()), token);
    if (token && g_producer->shared()->settings.enabled && !presenter_running()) launch_presenter();
}

const char* format_name(std::uint32_t f) {
    switch (f) {
        case 0: return "-";
        case DXGI_FORMAT_R32G8X24_TYPELESS: return "R32G8X24_TYPELESS";
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return "D32S8";
        case DXGI_FORMAT_R32_TYPELESS: return "R32_TYPELESS";
        case DXGI_FORMAT_D32_FLOAT: return "D32";
        case DXGI_FORMAT_R24G8_TYPELESS: return "R24G8_TYPELESS";
        case DXGI_FORMAT_R16G16_FLOAT: return "RG16F";
        case DXGI_FORMAT_R32G32_FLOAT: return "RG32F";
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "RGBA16F";
        case DXGI_FORMAT_R10G10B10A2_UNORM: return "RGB10A2";
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return "RGB10A2_TYPELESS";
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "RGBA8";
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return "RGBA8_TYPELESS";
        case DXGI_FORMAT_B8G8R8A8_UNORM: return "BGRA8";
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return "BGRA8_TYPELESS";
        case DXGI_FORMAT_R11G11B10_FLOAT: return "R11G11B10F";
        default: return nullptr;
    }
}

void draw_overlay(effect_runtime*) {
    if (!g_producer || !g_producer->shared()) { ImGui::TextUnformatted("Shared memory unavailable."); return; }
    auto& sh = *g_producer->shared();
    auto& s = sh.settings;
    auto& h = sh.hooks;
    auto& p = sh.presenter;
    ImGui::TextDisabled("XPAR (xPoiler's Asynchronous Reprojection) " FW_VERSION);

    bool enabled = s.enabled != 0;
    // Label follows the refresh rate the presenter measures on the overlay's display.
    char enable_label[64] = "Enable reprojection###enable";
    if (p.pid && p.display_hz > 1.0f)
        std::snprintf(enable_label, sizeof(enable_label), "Enable reprojection (%.0f Hz)###enable", p.display_hz);
    if (ImGui::Checkbox(enable_label, &enabled)) s.enabled = enabled;
    ImGui::SameLine();
    bool original = s.show_original != 0;
    if (ImGui::Checkbox("Show original (A/B)", &original)) s.show_original = original;

    const bool alive = presenter_running();
    ImGui::Text("Presenter: %s", alive ? "running" : "not running");
    if (alive && p.hardware_scheduling == 1)
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                           "Hardware-accelerated GPU scheduling is OFF: output may not reach the refresh rate.\n"
                           "Turn it on in Windows Settings > System > Display > Graphics, then restart the PC.");
    if (alive && p.frame_generation)
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                           "Frame generation is on in the game. XPAR is paused: turn frame generation off -\n"
                           "XPAR already fills your display's refresh rate, and the two cannot be combined.");
    if (!alive && ImGui::Button("Start presenter")) { g_presenter_launched = false; launch_presenter(); }
    if (alive) {
        ImGui::Text("Output %.1f fps | game %.1f fps | warp GPU %.2f ms | source age %.1f ms",
                    p.output_fps, p.source_fps, p.warp_gpu_ms, p.source_age_ms);
        ImGui::TextWrapped("%s", p.message);
    }

    if (ImGui::CollapsingHeader("Camera motion", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool mouse = s.use_mouse != 0;
        if (ImGui::Checkbox("Raw mouse drives rotation", &mouse)) s.use_mouse = mouse;
        ImGui::SliderFloat("Rotation extrapolation", &s.rotation_extrapolation, 0.0f, 1.0f);
        static const char* const kAutoModes[] = {"Off (manual slider)", "Auto (1/2, or 1/4 without HUD layers)", "1 game frame", "1/2 game frame",
                                                 "1/4 game frame"};
        static const std::uint32_t kAutoValues[] = {0, 3, 1, 2, 4};
        int auto_index = 0;
        for (int i = 0; i < 5; ++i) if (kAutoValues[i] == s.auto_prediction) auto_index = i;
        if (ImGui::Combo("Auto latency", &auto_index, kAutoModes, 5)) s.auto_prediction = kAutoValues[auto_index];
        const bool auto_latency = s.auto_prediction != 0;
        if (auto_latency) {
            ImGui::SameLine();
            ImGui::TextDisabled("%.1f ms (game frame %.1f ms)", p.effective_prediction_ms, p.frame_interval_ms);
            ImGui::BeginDisabled();
        }
        ImGui::SliderFloat("Latency <-> smoothness (ms)", &s.prediction_ms, -40.0f, 30.0f, "%.0f");
        if (auto_latency) ImGui::EndDisabled();
        ImGui::TextDisabled("  negative: smoother, adds that much delay | positive: predicts further ahead");
        ImGui::SliderFloat("Orbit distance (0 = off)", &s.orbit_distance, 0.0f, 1000.0f, "%.0f");
        ImGui::SliderFloat("Max extrapolation (ms)", &s.max_horizon_ms, 0.0f, 200.0f, "%.0f");
        bool strip = s.overlay_debug != 0;
        if (ImGui::Checkbox("Debug strip (top-left, shows every presented frame)", &strip)) s.overlay_debug = strip;
        ImGui::SliderFloat("Present lead (ms)", &s.present_lead_ms, 0.0f, 7.0f, "%.1f");
        ImGui::TextDisabled("  render this long before the next refresh; 0 = right after the previous one");
        static const char* const kPriorities[] = {"Realtime (default)", "High", "Normal"};
        int priority = s.gpu_priority <= 2 ? static_cast<int>(s.gpu_priority) : 0;
        if (ImGui::Combo("Presenter GPU priority", &priority, kPriorities, 3)) s.gpu_priority = static_cast<std::uint32_t>(priority);
        static const char* const kEngines[] = {"NVIDIA Latewarp", "XPAR (default)"};
        // NVIDIA Latewarp is optional: only selectable once the presenter reports it ready.
        const bool latewarp = sh.presenter.latewarp == 2;
        int engine = latewarp && s.warp_engine != 1 ? 0 : 1;
        ImGui::BeginDisabled(!latewarp);
        if (ImGui::Combo("Warp engine", &engine, kEngines, 2)) s.warp_engine = static_cast<std::uint32_t>(engine);
        ImGui::EndDisabled();
        if (!latewarp)
            ImGui::TextDisabled("  NVIDIA Latewarp is not installed (optional: nvngx_latewarp.dll, NVIDIA GPUs only)");
        bool invert = s.invert_warp != 0;
        if (ImGui::Checkbox("Invert warp (debug)", &invert)) s.invert_warp = invert;
        bool record = s.record_diagnostics != 0;
        if (ImGui::Checkbox("Record detailed diagnostics (for troubleshooting)", &record)) {
            s.record_diagnostics = record;
            reshade::set_config_value(nullptr, "FrameWarp", "RecordDiagnostics", record ? "1" : "0");
        }
        ImGui::TextDisabled("  frame-by-frame recordings in the FrameWarp\\logs folder; remembered for this game");
        bool ui = s.use_ui_tags != 0;
        if (ImGui::Checkbox("Keep HUD still (HUD-less + UI tags)", &ui)) s.use_ui_tags = ui;
        // What is kept unwarped: the HUD (detected in games without HUD layers) and what moves with the
        // camera (third-person character, first-person weapon).
        static const char* const kKeepStill[] = {"HUD + character/weapon (default)", "HUD only", "Character/weapon only", "Off"};
        int keep = s.no_warp_mask ? (s.keep_attached ? 0 : 1) : (s.keep_attached ? 2 : 3);
        if (ImGui::Combo("Keep still", &keep, kKeepStill, 4)) {
            s.no_warp_mask = keep == 0 || keep == 1;
            s.keep_attached = keep == 0 || keep == 2;
        }
        const bool mask = s.no_warp_mask != 0, attached = s.keep_attached != 0;
        if (attached) {
            ImGui::TextDisabled("  what moves with the camera is not warped and moves at the game's frame rate%s",
                                p.mv_scale_x == 0.0f ? "; starts after a few seconds of turning the camera" : "");
            bool near_rule = s.near_camera_rule != 0;
            if (ImGui::Checkbox("Keep near-camera motion still (weapon animations, hands)", &near_rule)) {
                s.near_camera_rule = near_rule;
                reshade::set_config_value(nullptr, "FrameWarp", "NearCameraRule", near_rule ? "1" : "0");
            }
            ImGui::TextDisabled("  turn off if the floor near the camera is kept still while strafing");
        }
        if (mask) {
            static const char* const kHudFind[] = {"Learned from camera motion", "From the upscaler output (DLSS or FSR)",
                                                   "Upscaler output + camera motion check (default)"};
            int find = s.hud_from_scene > 2 ? 2 : int(s.hud_from_scene);
            if (ImGui::Combo("Find the HUD", &find, kHudFind, 3)) {
                s.hud_from_scene = find;
                static const char* const kValues[] = {"0", "1", "2"};
                reshade::set_config_value(nullptr, "FrameWarp", "HudFromDlssOutput", kValues[find]);
            }
            if (find == 0)
                ImGui::TextDisabled("  learns what stays put while the camera moves; the only choice without an upscaler output");
            else if (find == 1)
                ImGui::TextDisabled("  sharper and instant, but in some games it may keep parts of the scenery still; check with the mask view");
            else if (find == 2)
                ImGui::TextDisabled("  from the upscaler output, minus what is seen moving with the world when the camera turns; check with the mask view");
        }
        if (mask || attached) {
            bool show = s.show_mask != 0;
            if (ImGui::Checkbox("Show the mask (debug: magenta = kept still, green = HUD being learned)", &show)) s.show_mask = show;
        }
        ImGui::Separator();
        ImGui::Text("Camera model  yaw: %s gain %.3g mrad/count, smoothing %.0f ms, quality %.2f",
                    p.calibrated_x ? "fitted" : "learning", p.gain_x * 1000.0f, p.tau_x_ms, p.fit_quality_x);
        ImGui::Text("            pitch: %s gain %.3g mrad/count, smoothing %.0f ms, quality %.2f",
                    p.calibrated_y ? "fitted" : "learning", p.gain_y * 1000.0f, p.tau_y_ms, p.fit_quality_y);
        ImGui::Text("Input delay %.0f ms | game latency %.1f ms | measured orbit %.0f cm", p.delay_ms, p.latency_ms, p.orbit_cm);
        if (ImGui::Button("Reset camera model")) ++s.reset_calibration;
        bool manual = s.manual_gain != 0;
        if (ImGui::Checkbox("Manual mouse gain", &manual)) s.manual_gain = manual;
        if (manual) {
            float gx = s.manual_gain_x * 1000.0f, gy = s.manual_gain_y * 1000.0f;
            if (ImGui::SliderFloat("Yaw mrad/count", &gx, -5.0f, 5.0f, "%.4f")) s.manual_gain_x = gx / 1000.0f;
            if (ImGui::SliderFloat("Pitch mrad/count", &gy, -5.0f, 5.0f, "%.4f")) s.manual_gain_y = gy / 1000.0f;
            ImGui::SliderFloat("Input delay (ms)", &s.manual_delay_ms, -20.0f, 60.0f, "%.1f");
        }
    }

    if (ImGui::CollapsingHeader("FSR diagnostics (AMD FidelityFX 2 / 3.0 / 3.1 / 4)")) {
        const auto& f = sh.fsr;
        ImGui::Text("Hooks: amd_fidelityfx_dx12 %s, loader %s, upscaler %s", (f.hooks & 3) ? "yes" : "no", (f.hooks & 12) ? "yes" : "no",
                    (f.hooks & 48) ? "yes" : "no");
        ImGui::Text("Hooks: FSR 3.0 (ffx_fsr3upscaler_x64) %s, FSR 2 (ffx_fsr2_api_x64) %s", (f.hooks & 192) ? "yes" : "no",
                    (f.hooks & 768) ? "yes" : "no");
        ImGui::Text("Calls: upscale contexts %u, upscale dispatches %u, resets %u", f.upscale_creates, f.upscale_dispatches, f.resets);
        ImGui::Text("Render %ux%u -> output %ux%u, context flags 0x%X%s", f.render_w, f.render_h, f.out_w, f.out_h, f.create_flags,
                    (f.create_flags & 8) ? " (depth inverted)" : "");
        ImGui::Text("Depth fmt %u state 0x%X | motion fmt %u | output state 0x%X", f.depth_format, f.depth_state, f.mv_format, f.output_state);
        ImGui::Text("Camera: near %.3g far %.3g, vertical FOV %.1f deg | jitter %.3f, %.3f | MV scale %g, %g", f.near_plane, f.far_plane,
                    f.fov * 57.29578f, f.jitter[0], f.jitter[1], f.mv_scale[0], f.mv_scale[1]);
        ImGui::Text("Frames published from FSR (camera estimated): %u | outputs copied for HUD detection: %u", f.frames_published, f.outputs_copied);
    }
    if (ImGui::CollapsingHeader("NGX diagnostics (DLSS without Streamline)")) {
        const auto& n = sh.ngx;
        ImGui::Text("Hooks: D3D12 CreateFeature %s, EvaluateFeature %s | Vulkan CreateFeature %s, EvaluateFeature %s",
                    (n.hooks & 1) ? "yes" : "no", (n.hooks & 2) ? "yes" : "no", (n.hooks & 12) ? "yes" : "no", (n.hooks & 16) ? "yes" : "no");
        ImGui::Text("Calls: create %u (DLSS %u), evaluate %u (DLSS %u, unknown handle %u), resets %u", n.create_calls, n.dlss_creates,
                    n.evaluate_calls, n.dlss_calls, n.unknown_handle_calls, n.resets);
        for (int i = 0; i < 16; ++i)
            if (n.feature_calls[i]) ImGui::Text("  feature %2d: %u evaluations", i, n.feature_calls[i]);
        if (n.identified_by_inputs) ImGui::TextDisabled("  DLSS recognised from its inputs (it was created before the hooks)");
        if (n.frames_published) ImGui::Text("  frames published from DLSS (camera estimated): %u", n.frames_published);
        ImGui::Text("DLSS (%s) create: render %ux%u -> output %ux%u, flags 0x%X (%s%s%s%s)",
                    n.dlss_feature == 13 ? "Ray Reconstruction" : n.dlss_feature == 1 ? "Super Resolution" : "none yet", n.render_w, n.render_h, n.out_w, n.out_h,
                    n.create_flags, (n.create_flags & 1) ? "HDR " : "", (n.create_flags & 2) ? "MV-low-res " : "",
                    (n.create_flags & 4) ? "MV-jittered " : "", (n.create_flags & 8) ? "depth-inverted" : "");
        ImGui::Text("  depth %ux%u fmt %u | motion %ux%u fmt %u", n.depth_w, n.depth_h, n.depth_format, n.mv_w, n.mv_h, n.mv_format);
        ImGui::Text("  colour %ux%u fmt %u | output %ux%u fmt %u", n.color_w, n.color_h, n.color_format, n.output_w, n.output_h,
                    n.output_format);
        ImGui::Text("  render subrect %ux%u | jitter %.3f, %.3f | MV scale %.4g, %.4g", n.subrect_w, n.subrect_h, n.jitter[0], n.jitter[1],
                    n.mv_scale[0], n.mv_scale[1]);
    }
    if (ImGui::CollapsingHeader("Streamline diagnostics")) {
        ImGui::Text("Hooks: constants %s, tag %s, tagForFrame %s, PCL %s",
                    (h.hooks_installed & 1) ? "yes" : "NO", (h.hooks_installed & 2) ? "yes" : "NO",
                    (h.hooks_installed & 4) ? "yes" : "NO", (h.hooks_installed & 8) ? "yes" : "no");
        ImGui::Text("Calls: constants %u, tag %u, tagForFrame %u, markers %u", h.constants_calls, h.tag_calls,
                    h.tag_for_frame_calls, h.marker_calls);
        ImGui::Text("Layout base: constants %d, tags %d (stride %u) | published %u, dropped %u, copy failures %u",
                    h.constants_base, h.tag_base, h.pad, h.frames_published, h.slots_dropped, h.copies_failed);
        ImGui::Text("Markers: sim %u/%u submit %u/%u present %u/%u", h.marker_counts[0], h.marker_counts[1],
                    h.marker_counts[2], h.marker_counts[3], h.marker_counts[4], h.marker_counts[5]);
        for (int t = 0; t < 64; ++t) {
            if (!h.tag_count[t]) continue;
            const char* f = format_name(h.tag_format[t]);
            if (f) ImGui::Text("  tag type %2d: %ux%u %s (%u)", t, h.tag_width[t], h.tag_height[t], f, h.tag_count[t]);
            else ImGui::Text("  tag type %2d: %ux%u fmt %u (%u)", t, h.tag_width[t], h.tag_height[t], h.tag_format[t], h.tag_count[t]);
        }
        std::string exports = "Other exports:";
        for (int i = 0; i < fw::kCountedExportCount; ++i)
            exports += " " + std::string(fw::kCountedExports[i] + 2) + "=" + std::to_string(h.export_calls[i]);
        ImGui::TextWrapped("%s", exports.c_str());
        std::string features = "slIsFeatureLoaded (id:result/loaded):";
        for (int i = 0; i < fw::kProbedFeatureCount; ++i)
            features += " " + std::to_string(fw::kProbedFeatures[i]) + ":" + std::to_string(h.feature_result[i]) + "/" + std::to_string(h.feature_loaded[i]);
        ImGui::TextWrapped("%s", features.c_str());
        ImGui::Text("Marker lookup: PCL %d, Reflex %d", h.pcl_lookup_result, h.reflex_lookup_result);
        ImGui::TextUnformatted("UE plugin probes (calls):");
        for (int i = 0; i < fw::kProbeCount; ++i)
            ImGui::Text("  %-28s %s %u", fw::kProbes[i].name, (h.probe_installed & (1u << i)) ? "hooked" : "not hooked", h.probe_calls[i]);
        if (h.message[0]) ImGui::TextWrapped("Last message: %s", h.message);
        if (!fw::streamline_loaded()) ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "sl.interposer.dll is not loaded in this game.");
        else if (!h.tag_calls && !h.tag_for_frame_calls)
            ImGui::TextWrapped("No Streamline tags yet. Add r.Streamline.ForceTagging=1 under [ConsoleVariables] in Engine.ini.");
    }
}

void register_callbacks() {
    reshade::register_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
    reshade::register_event<reshade::addon_event::reshade_present>(on_reshade_present);
    reshade::register_event<reshade::addon_event::create_resource>(on_create_resource);
    reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
    reshade::register_event<reshade::addon_event::reshade_open_overlay>(on_open_overlay);
    reshade::register_overlay("XPAR", draw_overlay);
}
void unregister_callbacks() {
    reshade::unregister_overlay("XPAR", draw_overlay);
    reshade::unregister_event<reshade::addon_event::reshade_open_overlay>(on_open_overlay);
    reshade::unregister_event<reshade::addon_event::destroy_device>(on_destroy_device);
    reshade::unregister_event<reshade::addon_event::create_resource>(on_create_resource);
    reshade::unregister_event<reshade::addon_event::reshade_present>(on_reshade_present);
    reshade::unregister_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
}
}  // namespace

extern "C" {
__declspec(dllexport) const char* NAME = "XPAR";
__declspec(dllexport) const char* DESCRIPTION = "xPoiler's Asynchronous Reprojection: camera reprojection at the display's refresh rate, fed by DLSS or FSR data.";

__declspec(dllexport) bool AddonInit(HMODULE addon_module, HMODULE reshade_module) noexcept {
    if (!reshade::register_addon(addon_module, reshade_module)) {
        // ReShade does not always log why it refused: note what we can see in
        // %LOCALAPPDATA%\FrameWarp\addon-init.log so a failure to load can be diagnosed.
        wchar_t local[MAX_PATH]{};
        if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH)) {
            const std::wstring dir = std::wstring(local) + L"\\FrameWarp";
            CreateDirectoryW(dir.c_str(), nullptr);
            FILE* f = nullptr;
            if (_wfopen_s(&f, (dir + L"\\addon-init.log").c_str(), L"a") == 0 && f) {
                wchar_t exe[MAX_PATH]{}, rs[MAX_PATH]{};
                GetModuleFileNameW(nullptr, exe, MAX_PATH);
                if (reshade_module) GetModuleFileNameW(reshade_module, rs, MAX_PATH);
                const auto reg = reshade_module ? GetProcAddress(reshade_module, "ReShadeRegisterAddon") : nullptr;
                using ImGuiTableFn = const void* (*)(std::uint32_t);
                const auto imgui = reshade_module ? reinterpret_cast<ImGuiTableFn>(GetProcAddress(reshade_module, "ReShadeGetImGuiFunctionTable")) : nullptr;
                std::fprintf(f, "FrameWarp " FW_VERSION " could not register with ReShade in %ls\n", exe);
                std::fprintf(f, "  ReShade module %p (%ls): ReShadeRegisterAddon %p, ImGui table for version %d: %p\n",
                             static_cast<void*>(reshade_module), rs, reinterpret_cast<void*>(reg), IMGUI_VERSION_NUM,
                             imgui ? imgui(IMGUI_VERSION_NUM) : nullptr);
                std::fclose(f);
            }
        }
        return false;
    }
    g_module = addon_module;
    // Inline hooks jump into this module; keep it loaded for the life of the process even when
    // ReShade unloads add-ons between device recreations.
    HMODULE pinned = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                       reinterpret_cast<LPCWSTR>(&AddonInit), &pinned);
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(addon_module, path, MAX_PATH);
    std::wstring dir = path;
    dir = dir.substr(0, dir.find_last_of(L"\\/"));
    g_presenter_path = dir + L"\\FrameWarp\\FrameWarpPresenter.exe";
    if (!g_producer) g_producer = std::make_unique<fw::Producer>();
    fw::install_streamline_hooks(g_producer.get());
    fw::install_ngx_hooks(g_producer.get());
    fw::install_ffx_hooks(g_producer.get());
    fw::install_game_probes(g_producer->shared() ? &g_producer->shared()->hooks : nullptr);
    register_callbacks();
    return true;
}

__declspec(dllexport) void AddonUninit(HMODULE addon_module, HMODULE reshade_module) noexcept {
    unregister_callbacks();
    // The producer (and hooks) stay alive: the module is pinned and hooks may still fire.
    reshade::unregister_addon(addon_module, reshade_module);
}
}
