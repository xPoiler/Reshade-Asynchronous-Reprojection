#include "addon/output_effects.hpp"
#include "addon/reshade_feed.hpp"
#include "addon/diag.hpp"
#include <reshade.hpp>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace fw::output_effects {
namespace {
using namespace reshade::api;

struct State {
    effect_runtime* runtime = nullptr;  // the game's (its swap chain's window is the game's)
    bool user_on = true;                // ReShade's effects as the user has them (its toggle key) - not ours
    bool off_by_us = false;             // switched off by us: the presenter draws them
    bool loaded = false;                // its effects have finished loading once (exported only then: nothing partial)
    bool dirty = true;                  // the preset changed since it was last written for the presenter
    bool active = false;
    bool old_feed = false;              // the depth feed runs from the old XPAR.fx (feed_needs_user_effects)
    bool counted = false, counted_user_on = true;  // (active counted again for the current change / toggle)
    std::int64_t last_export = 0;
    unsigned counter = 0;
    std::string preset_path;  // FrameWarp\ReShadePreset.ini (UTF-8)
} g;

std::string utf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(size_t(std::max(n, 0)), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}
std::wstring wide(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(std::max(n, 0)), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), w.data(), n);
    return w;
}

// A config value of the game's ReShade, its list entries as ReShade hands them (separated by '\0').
std::vector<std::string> config_list(effect_runtime* runtime, const char* section, const char* key) {
    std::vector<std::string> out;
    size_t size = 0;
    if (!reshade::get_config_value(runtime, section, key, nullptr, &size) || size == 0) return out;
    std::string buffer(size + 1, '\0');
    size = buffer.size();
    if (!reshade::get_config_value(runtime, section, key, buffer.data(), &size)) return out;
    for (size_t at = 0; at < size;) {
        const size_t end = buffer.find('\0', at);
        const std::string entry = buffer.substr(at, (end == std::string::npos ? size : end) - at);
        if (!entry.empty()) out.push_back(entry);
        if (end == std::string::npos) break;
        at = end + 1;
    }
    return out;
}

// Relative folders of the game's ReShade are relative to its base path; the presenter's ReShade has another one.
std::string absolute(const std::string& base, const std::string& path) {
    const bool rooted = (path.size() > 1 && path[1] == ':') || (path.size() > 1 && (path[0] == '\\' || path[0] == '/') && (path[1] == '\\' || path[1] == '/'));
    if (rooted || base.empty()) return path;
    std::string p = path;
    if (p.rfind(".\\", 0) == 0 || p.rfind("./", 0) == 0) p = p.substr(2);
    return base + (base.back() == '\\' || base.back() == '/' ? "" : "\\") + p;
}
std::string joined(const std::vector<std::string>& list, const std::string& base) {
    std::string out;
    for (const std::string& e : list) out += (out.empty() ? "" : ",") + absolute(base, e);
    return out;
}

bool from_xpar(const char* effect_name) { return std::strncmp(effect_name, "XPAR.", 5) == 0; }

bool any_user_technique_on(effect_runtime* runtime) {
    struct Found { bool on = false; } found;
    runtime->enumerate_techniques(nullptr, [](effect_runtime* r, effect_technique t, void* user) {
        auto* f = static_cast<Found*>(user);
        if (f->on || !r->get_technique_state(t)) return;
        char effect[128] = "";
        size_t size = sizeof(effect);
        r->get_technique_effect_name(t, effect, &size);
        if (!from_xpar(effect)) f->on = true;
    }, &found);
    return found.on;
}

// Changes the user makes to the preset (the ReShade menu, its keys): written for the presenter at the next present. Not the
// values ReShade sets itself every frame (a uniform with a "source": timers, frame counts), not XPAR's own effect.
bool on_set_uniform(effect_runtime* runtime, effect_uniform_variable variable, const void*, size_t) {
    if (runtime != g.runtime || g.dirty) return false;
    char effect[128] = "";
    size_t size = sizeof(effect);
    runtime->get_uniform_variable_effect_name(variable, effect, &size);
    if (from_xpar(effect)) return false;
    char source[32] = "";
    size = sizeof(source);
    if (runtime->get_annotation_string_from_uniform_variable(variable, "source", source, &size)) return false;
    g.dirty = true;
    return false;
}
bool on_set_technique(effect_runtime* runtime, effect_technique technique, bool) {
    if (runtime != g.runtime) return false;
    char effect[128] = "";
    size_t size = sizeof(effect);
    runtime->get_technique_effect_name(technique, effect, &size);
    if (!from_xpar(effect)) g.dirty = true;
    return false;
}
bool on_reorder(effect_runtime* runtime, size_t, effect_technique*) { if (runtime == g.runtime) g.dirty = true; return false; }
void on_preset_path(effect_runtime* runtime, const char*) { if (runtime == g.runtime) g.dirty = true; }
// (also before the first present told us which runtime is the game's: effects can finish loading first)
void on_reloaded(effect_runtime* runtime) { if (!g.runtime || runtime == g.runtime) { g.loaded = true; g.dirty = true; } }
// The user's own toggle (ReShade's effects key): while we hold the effects off, it toggles what the user wants (the presenter
// follows it), the runtime's own switch stays off.
bool on_effects_state(effect_runtime* runtime, bool enabled) {
    if (runtime != g.runtime) return false;
    if (g.off_by_us) { g.user_on = !g.user_on; return true; }
    g.user_on = enabled;
    return false;
}
void on_destroy_runtime(effect_runtime* runtime) {
    if (runtime != g.runtime) return;
    g.runtime = nullptr; g.off_by_us = false; g.loaded = false; g.dirty = true;
}
}  // namespace

void register_events() {
    reshade::register_event<reshade::addon_event::reshade_set_uniform_value>(on_set_uniform);
    reshade::register_event<reshade::addon_event::reshade_set_technique_state>(on_set_technique);
    reshade::register_event<reshade::addon_event::reshade_reorder_techniques>(on_reorder);
    reshade::register_event<reshade::addon_event::reshade_set_current_preset_path>(on_preset_path);
    reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(on_reloaded);
    reshade::register_event<reshade::addon_event::reshade_set_effects_state>(on_effects_state);
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_runtime);
}
void unregister_events() {
    // (the game's effects back as the user has them, if we held them off)
    if (g.runtime && g.off_by_us) g.runtime->set_effects_state(g.user_on);
    g.off_by_us = false;
    reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_runtime);
    reshade::unregister_event<reshade::addon_event::reshade_set_effects_state>(on_effects_state);
    reshade::unregister_event<reshade::addon_event::reshade_reloaded_effects>(on_reloaded);
    reshade::unregister_event<reshade::addon_event::reshade_set_current_preset_path>(on_preset_path);
    reshade::unregister_event<reshade::addon_event::reshade_reorder_techniques>(on_reorder);
    reshade::unregister_event<reshade::addon_event::reshade_set_technique_state>(on_set_technique);
    reshade::unregister_event<reshade::addon_event::reshade_set_uniform_value>(on_set_uniform);
}

bool game_effects_off() { return g.off_by_us; }
bool old_feed_effect() { return g.old_feed; }

void prepare(Shared* sh, const std::wstring& dir) {
    if (!sh || !sh->settings.reshade_on_output || dir.empty()) return;
    g.preset_path = utf8(dir + L"\\ReShadePreset.ini");
    g.dirty = true;
    // The game's ReShade (proxy DLL or Vulkan layer - the same binary under any name), copied next to the presenter under a name that
    // makes it a library there: its base path and config are the presenter's folder, nothing of the game's is loaded or written.
    // (Copied when it differs; not while a presenter still has the old one loaded - that one stays.)
    wchar_t src[MAX_PATH]{};
    if (HMODULE module = reshade::internal::get_reshade_module_handle()) GetModuleFileNameW(module, src, MAX_PATH);
    const std::wstring dll = dir + L"\\ReShade64.dll";
    WIN32_FILE_ATTRIBUTE_DATA a{}, b{};
    if (src[0] && GetFileAttributesExW(src, GetFileExInfoStandard, &a)) {
        const bool same = GetFileAttributesExW(dll.c_str(), GetFileExInfoStandard, &b) && a.nFileSizeLow == b.nFileSizeLow &&
                          a.nFileSizeHigh == b.nFileSizeHigh && CompareFileTime(&a.ftLastWriteTime, &b.ftLastWriteTime) == 0;
        if (!same && !CopyFileW(src, dll.c_str(), FALSE)) {
            char text[160];
            std::snprintf(text, sizeof(text), "XPAR: ReShade for the presenter's picture: could not copy the game's ReShade next to it (Windows error %lu)", GetLastError());
            diag_log(reshade::log::level::warning, text);
        }
    }
    // Its config: the game's effect and texture folders and definitions (absolute: its base path is another), the preset written for it,
    // and nothing it would show or do on its own (no menu, splash, statistics, saving, depth add-on, preset transitions).
    char base_buffer[1024] = "";
    size_t base_size = sizeof(base_buffer);
    reshade::get_reshade_base_path(base_buffer, &base_size);
    const std::string base = base_buffer;
    effect_runtime* rt = g.runtime;
    std::string ini;
    ini += "[INSTALL]\nLogging=" + std::string(sh->settings.record_diagnostics ? "1" : "0") + "\n\n";
    ini += "[GENERAL]\n";
    ini += "EffectSearchPaths=" + joined(config_list(rt, "GENERAL", "EffectSearchPaths"), base) + "\n";
    ini += "TextureSearchPaths=" + joined(config_list(rt, "GENERAL", "TextureSearchPaths"), base) + "\n";
    {
        std::string defs;
        for (const std::string& d : config_list(rt, "GENERAL", "PreprocessorDefinitions")) defs += (defs.empty() ? "" : ",") + d;
        ini += "PreprocessorDefinitions=" + defs + "\n";
    }
    if (const auto cache = config_list(rt, "GENERAL", "IntermediateCachePath"); !cache.empty()) ini += "IntermediateCachePath=" + absolute(base, cache[0]) + "\n";
    ini += "PresetPath=" + g.preset_path + "\n";
    ini += "PresetTransitionDuration=0\nPerformanceMode=0\nSkipLoadingDisabledEffects=1\nNoReloadOnInit=0\n\n";
    ini += "[OVERLAY]\nAutoSavePreset=0\nTutorialProgress=4\nShowClock=0\nShowFPS=0\nShowFrameTime=0\nShowPresetName=0\n\n";
    ini += "[ADDON]\nDisabledAddons=Generic Depth,Effect Runtime Sync\n";
    std::ofstream(dir + L"\\ReShade.ini", std::ios::binary | std::ios::trunc) << ini;
    // The preset to start from: the game's own file (its values for the effects the game has not loaded too), then written over with
    // what the game has now once its effects are loaded.
    std::string current;
    if (rt) {
        char path[1024] = "";
        size_t size = sizeof(path);
        rt->get_current_preset_path(path, &size);
        current = path;
    } else if (const auto p = config_list(nullptr, "GENERAL", "PresetPath"); !p.empty()) {
        current = absolute(base, p[0]);
    }
    if (!current.empty()) CopyFileW(wide(current).c_str(), wide(g.preset_path).c_str(), FALSE);
    char text[300];
    std::snprintf(text, sizeof(text), "XPAR: ReShade for the presenter's picture prepared: %s, preset %s", src[0] ? "the game's ReShade copied" : "the game's ReShade NOT FOUND",
                  current.empty() ? "(none)" : current.c_str());
    diag_log(reshade::log::level::info, text);
}

void after_present(effect_runtime* runtime, Shared* sh, bool presenter_alive, bool in_process) {
    if (!runtime || !sh || reinterpret_cast<std::uint64_t>(runtime->get_hwnd()) != sh->game_hwnd) return;
    if (g.runtime != runtime) {
        if (g.runtime && g.off_by_us) g.runtime->set_effects_state(g.user_on);  // (a swap chain made again: the old one's back as it was)
        g.runtime = runtime;
        g.user_on = runtime->get_effects_state();
        g.off_by_us = false; g.dirty = true;
    }
    const bool setting = sh->settings.reshade_on_output != 0;
    // (any of the user's techniques on: every 30 pictures, or at once after a change. Not reported while the old XPAR.fx runs the depth
    // feed: the game's effects cannot be switched off then, and drawn by the presenter too they would be drawn twice)
    if ((g.dirty && !g.counted) || ++g.counter % 30 == 0 || g.user_on != g.counted_user_on) {
        g.counted = g.dirty; g.counted_user_on = g.user_on;
        g.active = g.user_on && any_user_technique_on(runtime);
        g.old_feed = feed_needs_user_effects(runtime);
        InterlockedExchange(&sh->reshade_fx_active, g.active && !g.old_feed ? 1 : 0);
    }
    if (!g.dirty) g.counted = false;
    // The preset for the presenter, at most 5 times a second while it changes (a slider dragged).
    const std::int64_t now = qpc_now();
    if (setting && g.loaded && g.dirty && !g.preset_path.empty() && now - g.last_export > sh->qpc_frequency / 5) {
        runtime->export_current_preset(g.preset_path.c_str());
        g.dirty = false;
        g.last_export = now;
        InterlockedIncrement(&sh->reshade_preset_serial);
    }
    // Off while the presenter draws them (recently: it stops when its picture is not shown), never while the ReShade menu is open (with
    // effects off it hides their list - the game draws them then, the presenter does not), never when XPAR's own effect is the old
    // XPAR.fx with its feed running (an ordinary effect: switched off with the rest).
    // (2 s: the presenter presents nothing new while neither the camera nor the game's picture changes; hidden, it says so at once)
    const bool drawing = sh->presenter.reshade_state == 3 && now - sh->presenter.reshade_drawn_qpc < sh->qpc_frequency * 2;
    const bool off = setting && presenter_alive && !in_process && drawing && g.active && !sh->overlay_open && !feed_needs_user_effects(runtime);
    if (off != g.off_by_us) {
        g.off_by_us = off;
        runtime->set_effects_state(off ? false : g.user_on);
        diag_log(reshade::log::level::info, off ? "XPAR: the game's ReShade effects switched off - the presenter draws them on its picture"
                                                : "XPAR: the game's ReShade effects back on");
    }
}

}  // namespace fw::output_effects
