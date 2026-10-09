#include "addon/ui_probe.hpp"
#include <reshade.hpp>
#include <windows.h>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fw::ui_probe {
namespace {
using namespace reshade::api;

enum class Kind : std::uint8_t { Target, Draw, Dispatch, Copy, Clear, Mark };
struct Entry {
    Kind kind = Kind::Target;
    std::uint64_t res = 0, src = 0;   // the target (copies: the destination) and a copy's source
    std::uint32_t w = 0, h = 0, fmt = 0, count = 0, targets = 0;
    bool depth = false;
    float color[4] = {};  // (clears)
    std::string text;
};

// 0 idle, 1 armed (recording starts at the next present), 2 recording (until the present after).
std::atomic<int> g_state{0};
std::mutex g_mutex;
std::unordered_map<std::uint64_t, std::vector<Entry>> g_lists;  // by native command list
std::vector<std::uint64_t> g_order;                             // native command lists as submitted
std::vector<std::uint64_t> g_recorded;                          // ...as first recorded into
std::vector<std::uint64_t> g_back_buffers;
std::size_t g_entries = 0;
constexpr std::size_t kMaxEntries = 400000;
bool g_key_down = false;

bool recording() { return g_state.load(std::memory_order_relaxed) == 2; }
std::uint64_t list_key(command_list* c) { return c ? c->get_native() : 0; }

std::vector<Entry>* list_of(std::uint64_t key) {  // (under the lock)
    if (!key || g_entries >= kMaxEntries) return nullptr;
    auto [it, fresh] = g_lists.try_emplace(key);
    if (fresh) g_recorded.push_back(key);
    return &it->second;
}
void add(command_list* cmd, Entry e) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (auto* l = list_of(list_key(cmd))) { l->push_back(std::move(e)); ++g_entries; }
}
void count(command_list* cmd, Kind kind, std::uint32_t n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto* l = list_of(list_key(cmd));
    if (!l) return;
    if (!l->empty() && l->back().kind == kind) { l->back().count += n; return; }
    Entry e; e.kind = kind; e.count = n;
    l->push_back(e);
    ++g_entries;
}
void describe(device* dev, resource r, Entry& e) {
    e.res = r.handle;
    if (!dev || !r.handle) return;
    const resource_desc d = dev->get_resource_desc(r);
    e.w = d.texture.width; e.h = d.texture.height; e.fmt = static_cast<std::uint32_t>(d.texture.format);
}
void target(command_list* cmd, uint32_t n, resource_view first, bool depth) {
    if (!recording() || !cmd) return;
    Entry e; e.kind = Kind::Target; e.targets = n; e.depth = depth;
    device* dev = cmd->get_device();
    if (n && first.handle && dev) describe(dev, dev->get_resource_from_view(first), e);
    add(cmd, std::move(e));
}
void on_bind(command_list* cmd, uint32_t n, const resource_view* rtvs, resource_view dsv) { target(cmd, n, n && rtvs ? rtvs[0] : resource_view{0}, dsv.handle != 0); }
bool on_begin_pass(command_list* cmd, uint32_t n, const render_pass_render_target_desc* rts, const render_pass_depth_stencil_desc* ds, render_pass_flags) {
    target(cmd, n, n && rts ? rts[0].view : resource_view{0}, ds && ds->view.handle);
    return false;
}
bool on_draw(command_list* cmd, uint32_t, uint32_t, uint32_t, uint32_t) { if (recording()) count(cmd, Kind::Draw, 1); return false; }
bool on_draw_indexed(command_list* cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t) { if (recording()) count(cmd, Kind::Draw, 1); return false; }
bool on_dispatch(command_list* cmd, uint32_t, uint32_t, uint32_t) { if (recording()) count(cmd, Kind::Dispatch, 1); return false; }
bool on_indirect(command_list* cmd, indirect_command type, resource, uint64_t, uint32_t n, uint32_t) {
    if (recording()) count(cmd, type == indirect_command::dispatch ? Kind::Dispatch : Kind::Draw, n);
    return false;
}
void copy(command_list* cmd, resource src, resource dst) {
    if (!recording() || !cmd) return;
    Entry e; e.kind = Kind::Copy; e.src = src.handle;
    describe(cmd->get_device(), dst, e);
    add(cmd, std::move(e));
}
bool on_copy_resource(command_list* cmd, resource src, resource dst) { copy(cmd, src, dst); return false; }
bool on_copy_region(command_list* cmd, resource src, uint32_t, const subresource_box*, resource dst, uint32_t, const subresource_box*, filter_mode) {
    copy(cmd, src, dst);
    return false;
}
bool on_resolve(command_list* cmd, resource src, uint32_t, const subresource_box*, resource dst, uint32_t, uint32_t, uint32_t, uint32_t, format) {
    copy(cmd, src, dst);
    return false;
}
bool on_clear(command_list* cmd, resource_view rtv, const float* color, uint32_t, const rect*) {
    if (!recording() || !cmd) return false;
    Entry e; e.kind = Kind::Clear;
    if (color) for (int i = 0; i < 4; ++i) e.color[i] = color[i];
    if (device* dev = cmd->get_device(); dev && rtv.handle) describe(dev, dev->get_resource_from_view(rtv), e);
    add(cmd, std::move(e));
    return false;
}
void on_execute(command_queue*, command_list* cmd) {
    if (!recording() || !cmd) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_order.push_back(list_key(cmd));
}

void write_file() {
    std::unordered_map<std::uint64_t, std::vector<Entry>> lists;
    std::vector<std::uint64_t> order, recorded, back_buffers;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        lists.swap(g_lists); order.swap(g_order); recorded.swap(g_recorded); back_buffers = g_back_buffers;
        g_entries = 0;
    }
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir = exe;
    dir = dir.substr(0, dir.find_last_of(L"\\/")) + L"\\FrameWarp\\logs\\";
    std::wstring path;
    for (int i = 1; i < 1000; ++i) {
        path = dir + L"ui_probe_" + std::to_wstring(i) + L".txt";
        if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) break;
    }
    FILE* f = nullptr;
    if (_wfopen_s(&f, path.c_str(), L"w") || !f) return;
    // Short names for the resources, in order of appearance.
    std::unordered_map<std::uint64_t, int> names;
    auto name = [&](std::uint64_t r) {
        if (!r) return std::string("-");
        for (std::size_t i = 0; i < back_buffers.size(); ++i) if (back_buffers[i] == r) return "BACKBUFFER" + std::to_string(i);
        auto [it, fresh] = names.try_emplace(r, int(names.size()) + 1);
        return "T" + std::to_string(it->second);
    };
    std::fprintf(f, "UI probe: one game frame - %zu command lists recorded into, %zu submissions\n", recorded.size(), order.size());
    std::fprintf(f, "(per list: render targets bound [first target WxH format, +n more, depth or not], then what was drawn into them)\n\n");
    auto dump = [&](std::uint64_t key, const char* how, std::size_t index) {
        const auto it = lists.find(key);
        if (it == lists.end() || it->second.empty()) return;
        std::fprintf(f, "list %zu (%s, %016llx): %zu entries\n", index, how, static_cast<unsigned long long>(key), it->second.size());
        for (const Entry& e : it->second) {
            switch (e.kind) {
                case Kind::Target:
                    if (!e.targets) std::fprintf(f, "  targets: none%s\n", e.depth ? " + depth" : "");
                    else std::fprintf(f, "  targets: %s %ux%u fmt %u%s%s\n", name(e.res).c_str(), e.w, e.h, e.fmt,
                                      e.targets > 1 ? (" +" + std::to_string(e.targets - 1)).c_str() : "", e.depth ? " + depth" : "");
                    break;
                case Kind::Draw: std::fprintf(f, "    draws %u\n", e.count); break;
                case Kind::Dispatch: std::fprintf(f, "    dispatches %u\n", e.count); break;
                case Kind::Copy: std::fprintf(f, "  copy %s -> %s %ux%u fmt %u\n", name(e.src).c_str(), name(e.res).c_str(), e.w, e.h, e.fmt); break;
                case Kind::Clear:
                    std::fprintf(f, "  clear %s %ux%u fmt %u to (%g, %g, %g, %g)\n", name(e.res).c_str(), e.w, e.h, e.fmt, e.color[0], e.color[1],
                                 e.color[2], e.color[3]);
                    break;
                case Kind::Mark: std::fprintf(f, "  ** %s\n", e.text.c_str()); break;
            }
        }
    };
    std::unordered_map<std::uint64_t, bool> done;
    std::size_t index = 0;
    for (const std::uint64_t key : order) {  // (as submitted; a list submitted twice is shown once)
        if (done[key]) continue;
        done[key] = true;
        dump(key, "submitted", ++index);
    }
    for (const std::uint64_t key : recorded)  // (recorded but not submitted in this frame - or Direct3D 11's immediate context)
        if (!done[key]) dump(key, "recorded, not submitted in the frame", ++index);
    std::fclose(f);
    reshade::log::message(reshade::log::level::info, "XPAR: UI probe written (FrameWarp\\logs\\ui_probe_N.txt)");
}
}  // namespace

void init() {
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::register_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::register_event<reshade::addon_event::draw>(on_draw);
    reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::register_event<reshade::addon_event::dispatch>(on_dispatch);
    reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(on_indirect);
    reshade::register_event<reshade::addon_event::copy_resource>(on_copy_resource);
    reshade::register_event<reshade::addon_event::copy_texture_region>(on_copy_region);
    reshade::register_event<reshade::addon_event::resolve_texture_region>(on_resolve);
    reshade::register_event<reshade::addon_event::clear_render_target_view>(on_clear);
    reshade::register_event<reshade::addon_event::execute_command_list>(on_execute);
}
void shutdown() {
    reshade::unregister_event<reshade::addon_event::execute_command_list>(on_execute);
    reshade::unregister_event<reshade::addon_event::clear_render_target_view>(on_clear);
    reshade::unregister_event<reshade::addon_event::resolve_texture_region>(on_resolve);
    reshade::unregister_event<reshade::addon_event::copy_texture_region>(on_copy_region);
    reshade::unregister_event<reshade::addon_event::copy_resource>(on_copy_resource);
    reshade::unregister_event<reshade::addon_event::draw_or_dispatch_indirect>(on_indirect);
    reshade::unregister_event<reshade::addon_event::dispatch>(on_dispatch);
    reshade::unregister_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::unregister_event<reshade::addon_event::draw>(on_draw);
    reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_begin_pass);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
}

void on_present(swapchain* chain, bool keys_on) {
    const bool down = keys_on && (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) && (GetAsyncKeyState('D') & 0x8000);
    if (down && !g_key_down && g_state.load() == 0) g_state = 1;
    g_key_down = down;
    const int state = g_state.load();
    if (state == 2) {
        g_state = 0;
        write_file();
    } else if (state == 1 && chain) {
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_back_buffers.clear();
            for (uint32_t i = 0; i < chain->get_back_buffer_count(); ++i) g_back_buffers.push_back(chain->get_back_buffer(i).handle);
            g_lists.clear(); g_order.clear(); g_recorded.clear(); g_entries = 0;
        }
        g_state = 2;
    }
}

void mark(std::uint64_t native_cmd_list, const char* what) {
    if (!recording() || !native_cmd_list) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (auto* l = list_of(native_cmd_list)) {
        Entry e; e.kind = Kind::Mark; e.text = what;
        l->push_back(std::move(e));
        ++g_entries;
    }
}
}  // namespace fw::ui_probe
