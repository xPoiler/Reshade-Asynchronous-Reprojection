#include "addon/mv_probe.hpp"
#include "addon/producer.hpp"
#include <reshade.hpp>
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fw::mv_probe {
namespace {
using namespace reshade::api;

struct Grab {
    resource staging{0};
    std::uint32_t w = 0, h = 0, fmt = 0;
    int frame = 0, order = 0;
    std::uint64_t source = 0;
    bool back_buffer = false;
};
struct Seen { std::uint64_t res; std::uint32_t w, h, fmt, draws; int frame; };
struct Bound { std::vector<resource> targets; std::uint32_t draws = 0; std::size_t seen = 0; };

bool g_on = false, g_search = false;
std::atomic<bool> g_paused{false};
// The motion search's candidates: id = position + 1 for as long as the game lives. A candidate is what a target is, not which object: its format, its size, and
// its place among the targets of that shape in the frame - `res` is whichever object that is now. RE Engine on Direct3D 12 makes its targets anew every
// few frames: by the object, Requiem's motion was found and lost 38 times in a session, and the ids ran past 2000.
struct Candidate { std::uint64_t res = 0; std::uint32_t w = 0, h = 0; int age = 0; std::uint32_t format = 0, nth = 0; };
// (age: motion_step calls since it was last seen. The game's lists and the present run on their own threads: a frame's write can land just after the call of its frame,
// two writes in one call's time and none in the next - each such frame taken for a menu, no motion: Enshrouded's turn went 3.7, 0, 3.9, 0, the camera shook wildly)
constexpr int kSeenWithin = 8;  // (a stopgap: calls)
// What was found in an earlier run ([FrameWarp] MotionTarget, format:width in % of the depth's): those are tried first, and judged as any other. A hint, so
// that the next start does not go through every target: a game changed by an update simply falls through to the search.
std::uint32_t g_saved_format = 0, g_saved_width = 0;
std::uint32_t g_saved_id = 0;  // (the candidate already written down)
std::mutex g_search_mutex;
std::vector<Candidate> g_candidates;
Producer* g_producer = nullptr;
std::atomic<std::uint32_t> g_wanted{0};  // (Direct3D 12: the candidate to copy when the game finishes writing it)
std::vector<std::pair<std::uint64_t, std::uint32_t>> g_this_frame;  // (the targets seen since the last motion_step, and their candidates; 0: not one)
std::uint32_t g_depth_width = 0;  // (a change of it is a change of resolution: every target is another now)
// Vulkan: the wanted target's copy, in copy_source between copies. The game's own command buffer takes the copy, as on Direct3D 12: nothing else can be done there.
resource g_vk_held{0};
resource_desc g_vk_desc{};
std::uint32_t g_vk_taken = 0;
std::vector<std::pair<resource, int>> g_vk_old;  // (held images replaced, and the frames left until the GPU is done with them)
std::size_t g_trying = 0;
int g_held = 0;
constexpr std::size_t kMaxCandidates = 64;
int g_unseen = 0;
constexpr int kUnseenFrames = 36000;  // (a stopgap: 10 minutes at 60 fps)
constexpr int kHeldFrames = 30;  // (each candidate's turn, in game frames: a new format costs a new texture, so not every frame)
constexpr int kHintFrames = 1800;  // (a stopgap: a minute or less, the found target needs 96 frames that move)
int g_hint_frames = 0;
bool holds_motion(format f) {
    switch (f) {
        case format::r16g16b16a16_snorm: case format::r16g16b16a16_float: case format::r16g16b16a16_unorm: case format::r16g16b16a16_typeless:
        case format::r16g16_float: case format::r16g16_snorm: case format::r16g16_unorm: case format::r16g16_typeless:
        case format::r32g32_float: case format::r32g32_typeless: case format::r32g32b32a32_float: case format::r32g32b32a32_typeless:
        case format::r10g10b10a2_unorm: case format::r10g10b10a2_typeless: case format::r8g8_snorm: case format::r8g8_unorm: case format::r8g8_typeless:
            return true;
        default: return false;
    }
}
// The candidate this target is, this frame (0: none). Under g_search_mutex.
std::uint32_t note_target(device* dev, resource r) {
    for (const auto& seen : g_this_frame) if (seen.first == r.handle) return seen.second;
    const resource_desc d = dev->get_resource_desc(r);
    const std::uint32_t f = static_cast<std::uint32_t>(d.texture.format);
    std::uint32_t id = 0;
    if (d.type == resource_type::texture_2d && holds_motion(d.texture.format) && d.texture.samples == 1) {
        std::uint32_t nth = 0;
        for (const auto& seen : g_this_frame) {
            if (!seen.second) continue;
            const Candidate& c = g_candidates[seen.second - 1];
            nth += c.format == f && c.w == d.texture.width && c.h == d.texture.height ? 1 : 0;
        }
        for (std::size_t i = 0; i < g_candidates.size() && !id; ++i) {
            Candidate& c = g_candidates[i];
            if (c.format == f && c.w == d.texture.width && c.h == d.texture.height && c.nth == nth) { c.res = r.handle; c.age = 0; id = std::uint32_t(i) + 1; }
        }
        if (!id && g_candidates.size() < kMaxCandidates) {
            g_candidates.push_back({r.handle, d.texture.width, d.texture.height, 0, f, nth});
            id = std::uint32_t(g_candidates.size());
        }
    }
    g_this_frame.push_back({r.handle, id});
    return id;
}
bool g_key_down = false;
// 0 idle, 1 armed (starts at the next present), 2 recording the first frame, 3 the second.
std::atomic<int> g_state{0};
std::mutex g_mutex;
std::unordered_map<std::uint64_t, Bound> g_bound;  // by native command list
std::vector<Grab> g_grabs;
std::vector<Seen> g_seen;
int g_order = 0;
constexpr std::size_t kMaxGrabs = 48;
std::vector<resource> g_wide;  // (this frame's targets of a format that can hold motion among other things: as the frame left them)

int recording() {
    const int s = g_state.load(std::memory_order_relaxed);
    return s == 2 ? 1 : s == 3 ? 2 : 0;
}
bool candidate(format f) {
    switch (f) {
        case format::r16g16_float: case format::r16g16_snorm: case format::r16g16_unorm: case format::r16g16_typeless:
        case format::r32g32_float: case format::r32g32_typeless: case format::r8g8_snorm: case format::r8g8_unorm: case format::r8g8_typeless:
            return true;
        default: return false;
    }
}
bool grab(command_list* cmd, device* dev, resource src, const resource_desc& d, resource_usage state, bool back_buffer) {  // (under the lock)
    if (g_grabs.size() >= kMaxGrabs) return false;
    resource staging{0};
    const resource_desc copy(d.texture.width, d.texture.height, 1, 1, d.texture.format, 1, memory_heap::gpu_to_cpu, resource_usage::copy_dest);
    if (!dev->create_resource(copy, nullptr, resource_usage::copy_dest, &staging)) return false;
    const resource_usage from[1] = {state}, to[1] = {resource_usage::copy_source};
    cmd->barrier(1, &src, from, to);
    cmd->copy_resource(src, staging);
    cmd->barrier(1, &src, to, from);
    Grab g;
    g.staging = staging; g.w = d.texture.width; g.h = d.texture.height; g.fmt = static_cast<std::uint32_t>(d.texture.format);
    g.frame = recording(); g.order = ++g_order; g.source = src.handle; g.back_buffer = back_buffer;
    g_grabs.push_back(g);
    return true;
}
// (a G-buffer's motion shares a target: RE2 has no two-channel target at all, its G-buffer's four are bound some 200 times a frame -
// so one copy each at the frame's end)
bool wide_candidate(format f) { return f == format::r16g16b16a16_snorm || f == format::r16g16b16a16_float || f == format::r10g10b10a2_unorm; }
// What was bound on this list had its draws: as it was left.
void finish(command_list* cmd) {  // (under the lock)
    const auto it = g_bound.find(cmd->get_native());
    if (it == g_bound.end()) return;
    Bound& b = it->second;
    if (b.draws) {
        device* dev = cmd->get_device();
        for (const resource r : b.targets) {
            const resource_desc d = dev->get_resource_desc(r);
            if (candidate(d.texture.format) && d.texture.samples == 1) grab(cmd, dev, r, d, resource_usage::render_target, false);
        }
        if (b.seen) g_seen[b.seen - 1].draws = b.draws;
    }
    g_bound.erase(it);
}
void census_pass(device* dev, resource r);
void on_bind(command_list* cmd, uint32_t n, const resource_view* rtvs, resource_view) {
    if (g_search && !g_paused.load(std::memory_order_relaxed) && cmd && rtvs) {
        device* dev = cmd->get_device();
        std::lock_guard<std::mutex> lock(g_search_mutex);
        for (uint32_t i = 0; i < n; ++i) {
            if (!rtvs[i].handle) continue;
            const resource r = dev->get_resource_from_view(rtvs[i]);
            if (!r.handle) continue;
            census_pass(dev, r);  // (Direct3D: every target bound to draw into, for the log)
            note_target(dev, r);
        }
    }
    if (!recording() || !cmd) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    finish(cmd);
    device* dev = cmd->get_device();
    Bound b;
    for (uint32_t i = 0; rtvs && i < n; ++i) {
        if (!rtvs[i].handle) continue;
        const resource r = dev->get_resource_from_view(rtvs[i]);
        if (!r.handle) continue;
        const resource_desc d = dev->get_resource_desc(r);
        b.targets.push_back(r);
        if (wide_candidate(d.texture.format) && d.texture.samples == 1 && std::find_if(g_wide.begin(), g_wide.end(), [&](resource w) { return w.handle == r.handle; }) == g_wide.end())
            g_wide.push_back(r);
        g_seen.push_back({r.handle, d.texture.width, d.texture.height, static_cast<std::uint32_t>(d.texture.format), 0, recording()});
        if (i == 0) b.seen = g_seen.size();
    }
    g_bound[cmd->get_native()] = std::move(b);
}
void drew(command_list* cmd) {
    if (!recording() || !cmd) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_bound.find(cmd->get_native());
    if (it != g_bound.end()) ++it->second.draws;
}
bool on_draw(command_list* cmd, uint32_t, uint32_t, uint32_t, uint32_t) { drew(cmd); return false; }
bool on_draw_indexed(command_list* cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t) { drew(cmd); return false; }

void write_files(device* dev) {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir = exe;
    dir = dir.substr(0, dir.find_last_of(L"\\/")) + L"\\FrameWarp\\captures\\";
    int set = 1;
    for (; set < 1000; ++set)
        if (GetFileAttributesW((dir + L"mv_" + std::to_wstring(set) + L"_targets.txt").c_str()) == INVALID_FILE_ATTRIBUTES) break;
    const std::wstring base = dir + L"mv_" + std::to_wstring(set) + L"_";
    if (FILE* f = nullptr; !_wfopen_s(&f, (base + L"targets.txt").c_str(), L"w") && f) {
        for (const Seen& s : g_seen)
            std::fprintf(f, "frame %d res %016llx %ux%u fmt %u draws %u\n", s.frame, static_cast<unsigned long long>(s.res), s.w, s.h, s.fmt, s.draws);
        std::fclose(f);
    }
    for (const Grab& g : g_grabs) {
        subresource_data data{};
        if (dev->map_texture_region(g.staging, 0, nullptr, map_access::read_only, &data) && data.data) {
            const std::wstring name = base + std::to_wstring(g.frame) + (g.back_buffer ? L"_back_" : L"_t") + std::to_wstring(g.order) + L".bin";
            if (FILE* f = nullptr; !_wfopen_s(&f, name.c_str(), L"wb") && f) {
                const std::uint32_t head[4] = {g.w, g.h, g.fmt, data.row_pitch};
                std::fwrite(head, sizeof(head), 1, f);
                const std::uint64_t source = g.source;
                std::fwrite(&source, sizeof(source), 1, f);
                std::fwrite(data.data, data.row_pitch, g.h, f);
                std::fclose(f);
            }
            dev->unmap_texture_region(g.staging, 0);
        }
        dev->destroy_resource(g.staging);
    }
    reshade::log::message(reshade::log::level::info, "XPAR: motion buffer probe written (FrameWarp\\captures\\mv_N_*)");
}
void on_present(command_queue* queue, swapchain* chain, const rect*, const rect*, uint32_t, const rect*) {
    const bool key = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) && (GetAsyncKeyState('V') & 0x8000);
    if (key && !g_key_down && g_state.load() == 0) g_state = 1;
    g_key_down = key;
    const int s = g_state.load();
    if (s == 0) return;
    if (s == 1) { g_state = 2; return; }
    if (!queue || !chain) { g_state = 0; return; }
    std::lock_guard<std::mutex> lock(g_mutex);
    command_list* cmd = queue->get_immediate_command_list();
    device* dev = queue->get_device();
    const resource back = chain->get_current_back_buffer();
    for (const resource r : g_wide) if (cmd) grab(cmd, dev, r, dev->get_resource_desc(r), resource_usage::render_target, false);
    g_wide.clear();
    if (cmd && back.handle) grab(cmd, dev, back, dev->get_resource_desc(back), resource_usage::present, true);
    queue->flush_immediate_command_list();
    if (s == 2) { g_state = 3; return; }
    g_state = 0;
    queue->wait_idle();
    write_files(dev);
    g_grabs.clear(); g_seen.clear(); g_bound.clear(); g_order = 0;
}
// The motion search's census (what the game draws into, and on Vulkan how - a target that never shows as a barrier from writing to reading is never tried):
// by format and size, the images seen in render passes, in barriers, and whether they can be copied.
struct Census { std::uint32_t passes = 0, written = 0, read_after = 0, copyable = 0; };
std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, Census> g_census;
int g_census_frames = 0;
void count_census(const resource_desc& d, std::uint32_t Census::* what) {
    if (d.type != resource_type::texture_2d || d.texture.width < 256) return;
    Census& c = g_census[{static_cast<std::uint32_t>(d.texture.format), d.texture.width, d.texture.height}];
    ++(c.*what);
    if ((d.usage & resource_usage::copy_source) != 0) ++c.copyable;
}
void census_pass(device* dev, resource r) { count_census(dev->get_resource_desc(r), &Census::passes); }
void take_vulkan(command_list* cmd, device* dev, resource r, resource_usage state, std::uint32_t id);
// Vulkan: a target drawn in a render pass is done at the end of that pass. The game's barrier out of drawing can come as late as the next frame: Enshrouded's motion
// reached XPAR a frame behind its picture and depth (colour fitted the next frame's motion in 26 of 34 frames, with DLSS its own in all), and all that warped from it
// shook. End of the pass (the event comes before the pass has ended, nothing can be copied there): the first thing on that command buffer after it.
std::unordered_map<std::uint64_t, std::pair<resource, std::uint32_t>> g_in_pass, g_pass_done;
std::atomic<bool> g_taken_from_pass{false};
void after_pass(command_list* cmd, device* dev) {
    std::pair<resource, std::uint32_t> done{};
    {
        std::lock_guard<std::mutex> lock(g_search_mutex);
        const auto it = g_pass_done.find(cmd->get_native());
        if (it == g_pass_done.end()) return;
        done = it->second;
        g_pass_done.erase(it);
    }
    if (done.second != g_wanted.load(std::memory_order_relaxed)) return;
    take_vulkan(cmd, dev, done.first, resource_usage::render_target, done.second);  // (the layout of a target in a pass; one that ends the pass in another is read the same on NVIDIA)
    g_taken_from_pass = true;
}
bool on_census_pass(command_list* cmd, uint32_t n, const render_pass_render_target_desc* rts, const render_pass_depth_stencil_desc*, render_pass_flags) {
    if (!g_search || g_paused.load(std::memory_order_relaxed) || !cmd || !rts) return false;
    device* dev = cmd->get_device();
    if (dev->get_api() != device_api::vulkan) return false;
    after_pass(cmd, dev);
    std::lock_guard<std::mutex> lock(g_search_mutex);
    for (uint32_t i = 0; i < n; ++i) {
        if (!rts[i].view.handle) continue;
        const resource r = dev->get_resource_from_view(rts[i].view);
        if (!r.handle) continue;
        count_census(dev->get_resource_desc(r), &Census::passes);
        const std::uint32_t id = note_target(dev, r);
        if (id && id == g_wanted.load(std::memory_order_relaxed)) g_in_pass[cmd->get_native()] = {r, id};
    }
    return false;
}
bool on_end_pass(command_list* cmd) {
    if (!g_search || !cmd) return false;
    std::lock_guard<std::mutex> lock(g_search_mutex);
    const auto it = g_in_pass.find(cmd->get_native());
    if (it != g_in_pass.end()) { g_pass_done[cmd->get_native()] = it->second; g_in_pass.erase(it); }
    return false;
}
void on_close_list(command_list* cmd) {
    if (!g_search || !cmd || g_paused.load(std::memory_order_relaxed)) return;
    device* dev = cmd->get_device();
    if (dev->get_api() == device_api::vulkan) after_pass(cmd, dev);
}
// Vulkan: the copy, in the command buffer the game's barrier was recorded on.
void take_vulkan(command_list* cmd, device* dev, resource r, resource_usage state, std::uint32_t id) {
    const resource_desc d = dev->get_resource_desc(r);
    if ((d.usage & resource_usage::copy_source) == 0) return;  // (made before the add-on was there to ask for it)
    std::lock_guard<std::mutex> lock(g_search_mutex);
    if (!g_vk_held.handle || g_vk_desc.texture.width != d.texture.width || g_vk_desc.texture.height != d.texture.height || g_vk_desc.texture.format != d.texture.format) {
        if (g_vk_held.handle) g_vk_old.push_back({g_vk_held, 16});
        g_vk_held = {0};
        g_vk_taken = 0;
        const resource_desc held(d.texture.width, d.texture.height, 1, 1, d.texture.format, 1, memory_heap::gpu_only, resource_usage::copy_source | resource_usage::copy_dest);
        if (!dev->create_resource(held, nullptr, resource_usage::copy_source, &g_vk_held)) { g_vk_held = {0}; return; }
        g_vk_desc = held;
    }
    const resource both[2] = {r, g_vk_held};
    const resource_usage from[2] = {state, resource_usage::copy_source}, to[2] = {resource_usage::copy_source, resource_usage::copy_dest};
    cmd->barrier(2, both, from, to);
    cmd->copy_resource(r, g_vk_held);
    cmd->barrier(2, both, to, from);
    g_vk_taken = id;
}
// Direct3D 12: render targets and what compute shaders write alike show here - the game's own barrier from writing to reading says it is done with it
// this frame, and the state it is in. (Binds alone would miss compute, and at the frame's end the state is not known.)
void on_barrier(command_list* cmd, uint32_t n, const resource* resources, const resource_usage* before, const resource_usage* after) {
    if (!g_search || g_paused.load(std::memory_order_relaxed) || !cmd || !resources || !before || !after) return;
    device* dev = cmd->get_device();
    const bool vulkan = dev->get_api() == device_api::vulkan;
    if (!vulkan && dev->get_api() != device_api::d3d12) return;
    if (vulkan) after_pass(cmd, dev);
    constexpr auto kWriting = resource_usage::render_target | resource_usage::unordered_access;
    constexpr auto kReading = resource_usage::shader_resource_pixel | resource_usage::shader_resource_non_pixel;
    if (vulkan) {
        std::lock_guard<std::mutex> lock(g_search_mutex);
        for (uint32_t i = 0; i < n; ++i) {
            if (!resources[i].handle) continue;
            const resource_desc d = dev->get_resource_desc(resources[i]);
            if (static_cast<uint32_t>(before[i] & kWriting) != 0) count_census(d, &Census::written);
            if (static_cast<uint32_t>(after[i] & kReading) != 0) count_census(d, &Census::read_after);
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        // (done writing, whatever comes next: Enshrouded's motion goes from drawing to reading in two steps, and a barrier from writing straight to reading
        // never saw it - 37 1280x720 in the census, drawn and read every frame, not one try)
        if (!resources[i].handle || static_cast<uint32_t>(before[i] & kWriting) == 0 || static_cast<uint32_t>(after[i] & kWriting) != 0 ||
            after[i] == resource_usage::undefined || (!vulkan && static_cast<uint32_t>(after[i] & kReading) == 0)) continue;
        const resource r = resources[i];
        std::uint32_t id = 0;
        {
            std::lock_guard<std::mutex> lock(g_search_mutex);
            id = note_target(dev, r);
        }
        if (vulkan) {
            // (a target compute writes has no pass: its barrier it is)
            if (id && id == g_wanted.load(std::memory_order_relaxed) && !g_taken_from_pass.load(std::memory_order_relaxed)) take_vulkan(cmd, dev, r, after[i], id);
        } else if (g_producer && id && id == g_wanted.load(std::memory_order_relaxed))
            g_producer->on_motion_written_d3d12(reinterpret_cast<ID3D12Resource*>(r.handle), static_cast<D3D12_RESOURCE_STATES>(after[i]),
                                                reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native()), id);
    }
}
// (the object gone: the candidate waits for the next one)
void on_destroy_resource(device*, resource r) {
    std::lock_guard<std::mutex> lock(g_search_mutex);
    for (Candidate& c : g_candidates) if (c.res == r.handle) c.res = 0;
    for (auto& seen : g_this_frame) if (seen.first == r.handle) seen.first = 0;
}
}  // namespace

MotionStep motion_step(bool found, std::uint32_t found_candidate, std::uint32_t min_width, const std::uint64_t* ruled_out) {
    MotionStep out;
    if (!g_search) return out;
    std::lock_guard<std::mutex> lock(g_search_mutex);
    g_paused = false;
    const bool resized = g_depth_width && g_depth_width != min_width;
    g_depth_width = min_width;
    if (found) {
        const bool known = found_candidate && found_candidate <= g_candidates.size();
        if (known && !resized && g_candidates[found_candidate - 1].age <= kSeenWithin && g_candidates[found_candidate - 1].res) {
            const Candidate& c = g_candidates[found_candidate - 1];
            out.motion = resource{c.res};
            out.id = found_candidate;
            g_unseen = 0;
            if (found_candidate != g_saved_id) {
                g_saved_id = found_candidate;
                g_saved_format = c.format;
                g_saved_width = min_width ? c.w * 100 / min_width : 0;
                char text[32];
                std::snprintf(text, sizeof(text), "%u:%u", g_saved_format, g_saved_width);
                reshade::set_config_value(nullptr, "FrameWarp", "MotionTarget", static_cast<const char*>(text));
            }
        }
        // (another resolution, or not drawn for a very long while - never for a menu or a pause, the game comes back to it)
        // (or its format drawn at another size: dynamic resolution, Enshrouded's motion from 1432x780 to 1920x1080 in a session)
        else {
            bool resized_target = false;
            if (known)
                for (const Candidate& c : g_candidates)
                    resized_target = resized_target || (c.format == g_candidates[found_candidate - 1].format && c.age <= kSeenWithin && c.w != g_candidates[found_candidate - 1].w);
            if (!known || resized || (resized_target && ++g_unseen > kSeenWithin * 8) || ++g_unseen > kUnseenFrames) { out.lost = true; g_unseen = 0; g_held = 0; g_hint_frames = 0; }
        }
    } else if (!g_candidates.empty()) {
        // (the next one this frame drew, of a size that can be motion - down to a quarter of the depth's width: with an upscaler the game renders a third of the picture, and motion comes at that size, XeSS Ultra Performance at 4K - not ruled out)
        const auto out_of_it = [&](std::size_t i) { return i + 1 < 256 && (ruled_out[(i + 1) / 64] >> ((i + 1) % 64) & 1) != 0; };
        if (++g_held > kHeldFrames || out_of_it(g_trying % g_candidates.size())) { g_held = 0; ++g_trying; }
        const auto saved = [&](const Candidate& c) { return g_saved_format && c.format == g_saved_format; };  // (not by its size: that changes with the render scale)
        // (first for a while, not for good: a saved format can be another target's in this session, and judging a target against the others needs the others tried)
        const bool hinted = g_saved_format && g_hint_frames < kHintFrames;
        g_hint_frames += hinted ? 1 : 0;
        for (int pass = hinted ? 0 : 1; pass < 2 && !out.probe.handle; ++pass)
            for (std::size_t k = 0; k < g_candidates.size(); ++k) {
                const std::size_t i = (g_trying + k) % g_candidates.size();
                const Candidate& c = g_candidates[i];
                if (c.res && c.age <= kSeenWithin && c.w * 4 >= min_width && !out_of_it(i) && (pass || saved(c))) { out.probe = resource{c.res}; out.candidate = out.id = std::uint32_t(i) + 1; g_trying = i; break; }
            }
    }
    if (++g_census_frames == 1800 && !g_census.empty()) {
        std::string text = "XPAR: motion search census over 1800 frames (format WxH: render passes, written, read, copyable):";
        char item[96];
        for (const auto& [k, c] : g_census) {
            if (text.size() > 1800) break;
            if (!c.passes && !c.written) continue;  // (what is drawn: the list ran out before the formats that matter)
            std::snprintf(item, sizeof(item), " %u %ux%u: %u %u %u %u;", std::get<0>(k), std::get<1>(k), std::get<2>(k), c.passes, c.written, c.read_after, c.copyable);
            text += item;
        }
        reshade::log::message(reshade::log::level::info, text.c_str());
        // (which candidate is which, for the presenter's "motion search:" lines)
        text = "XPAR: motion search candidates (id format WxH, frames since drawn):";
        for (std::size_t i = 0; i < g_candidates.size() && text.size() < 1800; ++i) {
            const Candidate& c = g_candidates[i];
            if (!c.res || c.age > kHeldFrames) continue;
            std::snprintf(item, sizeof(item), " %zu %u %ux%u %d;", i + 1, unsigned(c.format), c.w, c.h, int(c.age));
            text += item;
        }
        reshade::log::message(reshade::log::level::info, text.c_str());
        g_census.clear();
        g_census_frames = 0;
    }
    for (Candidate& c : g_candidates) ++c.age;
    g_this_frame.clear();
    // (whatever call its write came in: copies run in order, so the held image is the newest finished at the present)
    if (g_vk_taken && g_vk_taken == out.id) out.vulkan = g_vk_held;
    g_wanted.store(out.id, std::memory_order_relaxed);
    return out;
}
void pause() {
    if (!g_search || g_paused.exchange(true)) return;
    std::lock_guard<std::mutex> lock(g_search_mutex);
    g_wanted.store(0, std::memory_order_relaxed);
    g_this_frame.clear();
    g_held = 0;
    g_in_pass.clear();
    g_pass_done.clear();
}
void set_producer(Producer* producer) { g_producer = producer; }
bool wants_copyable(format f, resource_usage usage) {
    return g_search && holds_motion(f) && (usage & (resource_usage::render_target | resource_usage::unordered_access)) != 0;
}

void init() {
    char value[8] = {};
    size_t size = sizeof(value);
    g_on = reshade::get_config_value(nullptr, "FrameWarp", "MotionProbe", value, &size) && value[0] == '1';
    value[0] = 0; size = sizeof(value);
    g_search = reshade::get_config_value(nullptr, "FrameWarp", "MotionSearch", value, &size) && value[0] == '1';
    char target[32] = {};
    size = sizeof(target);
    if (g_search && reshade::get_config_value(nullptr, "FrameWarp", "MotionTarget", target, &size) && std::sscanf(target, "%u:%u", &g_saved_format, &g_saved_width) != 2)
        g_saved_format = g_saved_width = 0;
    if (!g_on && !g_search) return;
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy_resource);
    if (g_search) reshade::register_event<reshade::addon_event::barrier>(on_barrier);
    if (g_search) reshade::register_event<reshade::addon_event::begin_render_pass>(on_census_pass);
    if (g_search) reshade::register_event<reshade::addon_event::end_render_pass>(on_end_pass);
    if (g_search) reshade::register_event<reshade::addon_event::close_command_list>(on_close_list);
    if (!g_on) return;
    reshade::register_event<reshade::addon_event::draw>(on_draw);
    reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::register_event<reshade::addon_event::present>(on_present);
}
void shutdown() {
    if (!g_on && !g_search) return;
    if (g_search) reshade::unregister_event<reshade::addon_event::close_command_list>(on_close_list);
    if (g_search) reshade::unregister_event<reshade::addon_event::end_render_pass>(on_end_pass);
    if (g_search) reshade::unregister_event<reshade::addon_event::begin_render_pass>(on_census_pass);
    if (g_search) reshade::unregister_event<reshade::addon_event::barrier>(on_barrier);
    reshade::unregister_event<reshade::addon_event::destroy_resource>(on_destroy_resource);
    reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind);
    if (!g_on) return;
    reshade::unregister_event<reshade::addon_event::present>(on_present);
    reshade::unregister_event<reshade::addon_event::draw_indexed>(on_draw_indexed);
    reshade::unregister_event<reshade::addon_event::draw>(on_draw);
}
}  // namespace fw::mv_probe
