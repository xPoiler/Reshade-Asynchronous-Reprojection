#include "addon/camera_search.hpp"
#include <reshade.hpp>
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace reshade::api;

namespace fw::camera_search {
namespace {
constexpr std::uint64_t kBudgetBytes = 192 * 1024;  // (read a frame: an upload heap is slow to read from. At 512 KB Cyberpunk's streets cost 10 to 29 ms over all threads, 42 fps to 30)
constexpr std::uint64_t kRangeBytes = 65536;        // (of each buffer: Direct3D's largest constant buffer view)
constexpr std::uint64_t kUnsizedBytes = 4096;       // (a stopgap: Direct3D 12's root constant buffers carry an address and no size; a whole 64 KB of each and the budget lasts three of them)
constexpr uint32_t kPageViews = 1024;

bool g_on = false;
std::mutex g_mutex;  // (the frame's matrices and the mapped buffers: never waited for on the game's render threads, see bind)
// Where a matrix lies is the slot it reaches the shaders through and its offset in what is bound there, never the memory: that moves on through a ring every frame. A range is read
// once a frame, however often it is bound, and each slot it is bound to gets its matrices under its own key: keys given by whichever bind came first follow the game's threads.
struct Found { std::uint32_t offset, flags; std::uint64_t value; float m[16]; };
struct Read { std::size_t first, count; };
std::vector<Found> g_found;                                        // (the matrices of what was read this frame)
std::unordered_map<std::uint64_t, Read> g_read;                    // (ranges read this frame)
std::unordered_set<std::uint64_t> g_labelled;                      // (a range under a slot, a value under a key: once)
struct Mapped { std::uint8_t* data = nullptr; std::uint64_t size = 0; std::uint32_t heap = 0; };
std::unordered_map<std::uint64_t, Mapped> g_mapped;                // (Direct3D 12: upload buffers, mapped for reading; data 0: unreadable, or gone)
std::unordered_map<std::uint64_t, std::pair<std::uint8_t*, std::uint64_t>> g_writing;  // (Direct3D 11: the buffer being filled)
std::uint64_t g_bytes = 0, g_ranges = 0;
std::atomic<bool> g_budget_spent{false};
std::uint32_t g_rotate = 0;
std::vector<MatrixCandidate> g_raw;
// Places found: held like a found motion buffer. The presenter sends back the keys that win; a key is mix(slot, offset), and mix can be undone, so wherever a slot is bound
// the offset each key would have there is known, and those 64 bytes are read every frame - past the budget, past 4 KB of a buffer without a size, first in what is sent.
// While places are held the rest is searched only when they stop predicting the motion (Shared::camera_held_state): the camera can take turns between places (AC6,
// Cyberpunk), and a whole frame searched cost Cyberpunk 13 ms in the hooks against 2.3 held alone - on two frames of every eight that was two thirds of it. Searched then on
// every frame until the places predict it again: a pair of frames is what the search judges. Not known (no frame judged yet): two frames running of every eight.
constexpr std::uint32_t kSearchEvery = 8, kSearchFrames = 2;  // (stopgaps)
std::atomic<bool> g_searching{true};
// A frame searched is read from where the last pair of frames searched filled up, round to the start once one fits: what fills the budget and the 2048 values sent is
// what is bound first, and where the camera comes in a frame changes with the scene - Cyberpunk found it at once in one place and never in another, its best value 0.2.
// Counted in reads (a range under a slot read anew, or constants pushed or written), two frames to a window: a pair of frames is what the search judges.
std::uint32_t g_read_index = 0, g_window = 0, g_full_at = 0, g_reads_last = 0;
// (under the lock) Whether this read is the search's: within the window, and noting where the frame filled up.
bool in_window() {
    const std::uint32_t index = g_read_index++;
    if (index < g_window) return false;
    if (!g_full_at && (g_budget_spent || g_raw.size() >= kMatrixCandidates)) g_full_at = index;
    return true;
}
constexpr std::uint32_t kHeld = 8;
constexpr std::uint32_t kUnmix = 0x359c449bu;  // (16777619's inverse, modulo 2^32)
std::uint32_t g_held[kHeld];
std::atomic<std::uint32_t> g_held_count{0};
std::vector<MatrixCandidate> g_held_found;
// The slots the places held were found in: between searches only binds to those take the lock (AC6 binds 2000-3500 tables a frame, a lock for each view of each).
constexpr std::uint32_t kHeldSlots = 16;
std::atomic<std::uint32_t> g_held_slot[kHeldSlots];
std::atomic<std::uint32_t> g_held_slots{0};
bool wanted(std::uint32_t slot) {
    if (g_searching.load(std::memory_order_relaxed)) return true;
    const std::uint32_t n = g_held_slots.load(std::memory_order_acquire);
    for (std::uint32_t i = 0; i < n; ++i) if (g_held_slot[i].load(std::memory_order_relaxed) == slot) return true;
    return false;
}
int g_said = 0;

// Direct3D 12: which constant buffer view lies where in the descriptor heaps. Games copy descriptors by the hundred thousand a frame from many threads: plain arrays, by
// place, no lock (a view being written while it is read gives a range of no use, not a fault: ranges are checked against their buffer). A hash table behind one lock took
// Cyberpunk from 45 to 15 fps.
struct Page { buffer_range views[kPageViews]; };
struct Heap { std::atomic<std::uint64_t> handle{0}; std::atomic<Page*> pages[1024] = {}; };
Heap g_heaps[32];

// What the hooks cost, for the log.
std::atomic<std::uint64_t> g_ticks{0}, g_copied{0}, g_written{0}, g_bound{0}, g_skipped{0};
std::atomic<std::uint32_t> g_frame{0};
std::atomic<std::uint64_t> g_ticks_searching{0}, g_search_frames{0};  // (the hooks' time on frames searched whole, and those frames)
// Where a game keeps its constants, for the log: what is bound and written by kind (0 sampler, 2 shader resource, 3 unordered access, 6 constant buffer...), the memory of the constant
// buffers bound directly (1 default, 2 upload, 3 readback, 4 custom), views found in the tables bound.
std::atomic<std::uint64_t> g_pushed_kind[16], g_written_kind[16], g_pushed_heap[8], g_table_views{0}, g_table_empty{0}, g_unsized{0}, g_large{0}, g_spent_frames{0};
LARGE_INTEGER g_frequency{};
// (counted on each thread and added to the shared counts every 256 calls: a count every render thread writes at each bind keeps its memory moving between the cores)
struct Pending { std::atomic<std::uint64_t>* to[24]; std::uint64_t n[24]; int used = 0; unsigned calls = 0; };
thread_local Pending t_pending;
void tally(std::atomic<std::uint64_t>& count, std::uint64_t n) {
    Pending& p = t_pending;
    for (int i = 0; i < p.used; ++i) if (p.to[i] == &count) { p.n[i] += n; return; }
    if (p.used < 24) { p.to[p.used] = &count; p.n[p.used++] = n; }
    else count.fetch_add(n, std::memory_order_relaxed);
}
void settle() {
    Pending& p = t_pending;
    if (++p.calls % 256) return;
    for (int i = 0; i < p.used; ++i) if (p.n[i]) { p.to[i]->fetch_add(p.n[i], std::memory_order_relaxed); p.n[i] = 0; }
}
// (one call in eight timed, counted eight times: the clock read twice at every bind is a cost of its own)
struct Timed {
    bool timing, searching;
    LARGE_INTEGER start{};
    Timed() : timing(t_pending.calls % 8 == 0), searching(g_searching.load(std::memory_order_relaxed)) { if (timing) QueryPerformanceCounter(&start); }
    ~Timed() {
        if (timing) {
            LARGE_INTEGER end;
            QueryPerformanceCounter(&end);
            const std::uint64_t t = 8 * std::uint64_t(end.QuadPart - start.QuadPart);
            tally(g_ticks, t);
            if (searching) tally(g_ticks_searching, t);
        }
        settle();
    }
};

Heap* heap_of(std::uint64_t handle, bool create) {
    for (Heap& h : g_heaps) {
        std::uint64_t now = h.handle.load(std::memory_order_acquire);
        if (now == handle) return &h;
        if (now == 0) {
            if (!create) return nullptr;
            if (h.handle.compare_exchange_strong(now, handle) || now == handle) return &h;
        }
    }
    return nullptr;
}
buffer_range* view_at(Heap* heap, uint32_t place, bool create) {
    const uint32_t page = place / kPageViews;
    if (page >= 1024) return nullptr;
    Page* p = heap->pages[page].load(std::memory_order_acquire);
    if (!p) {
        if (!create) return nullptr;
        Page* fresh = new Page();
        if (heap->pages[page].compare_exchange_strong(p, fresh)) p = fresh;
        else delete fresh;
    }
    return &p->views[place % kPageViews];
}
bool place_of(device* dev, descriptor_table table, uint32_t binding, uint32_t array_offset, descriptor_heap& heap, uint32_t& offset) {
    heap = {0}; offset = 0;
    dev->get_descriptor_heap_offset(table, binding, array_offset, &heap, &offset);
    return heap.handle != 0;
}

std::uint32_t mix(std::uint32_t h, std::uint32_t v) { return (h ^ v) * 16777619u; }

// What can be a camera's: finite, not a plain world matrix (its last row or column 0 0 0 1), or close to the identity as a reprojection is.
bool can_be_camera(const float* m) {
    double sum = 0;
    for (int i = 0; i < 16; ++i) {
        if (!std::isfinite(m[i]) || std::fabs(m[i]) > 1e7f) return false;
        sum += std::fabs(m[i]);
    }
    if (sum < 1e-3) return false;
    const auto perspective = [](float a, float b, float c, float d) { const float s = std::fabs(a) + std::fabs(b) + std::fabs(c); return s > 1e-5f && s < 1e4f && std::fabs(d) < 1e7f; };
    if (perspective(m[12], m[13], m[14], m[15]) || perspective(m[3], m[7], m[11], m[15])) return true;
    return std::fabs(m[0] - 1) < 0.5f && std::fabs(m[5] - 1) < 0.5f && std::fabs(m[10] - 1) < 0.5f && std::fabs(m[15] - 1) < 0.5f;
}
bool unit_length(float x, float y, float z) { return std::fabs(std::sqrt(x * x + y * y + z * z) - 1.0f) < 0.01f; }
std::uint64_t value_of(const float* m) {
    std::uint64_t value = 14695981039346656037ull;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(m);
    for (std::size_t i = 0; i < 64; ++i) value = (value ^ bytes[i]) * 1099511628211ull;
    return value;
}
// (under the lock)
void scan(const std::uint8_t* data, std::uint64_t size) {
    size = std::min(size, kRangeBytes);
    if (size < 64) return;
    if (g_bytes + size > kBudgetBytes) { g_budget_spent = true; return; }
    g_bytes += size;
    ++g_ranges;
    static thread_local float local[kRangeBytes / 4];
    std::memcpy(local, data, size);  // (at once: written-combined memory is read in one go, and the game writes on)
    for (std::uint64_t offset = 0; offset + 64 <= size; offset += 16) {
        const float* m = local + offset / 4;
        if (!can_be_camera(m)) continue;
        // Every value of the shape, however many a place holds in a frame, in the order they came. Clip w is the distance along the view, so one row of world to clip - the last,
        // or the last column when stored the other way round - is the camera's forward: length 1. (Keeping the first of a place, those that stay, 64 of them: Cyberpunk's
        // camera comes around the 900th, and not always first in its place: with those it was in what the presenter got on 12% of the frames that move, with none 75%.)
        const bool rows = unit_length(m[12], m[13], m[14]);
        if (g_found.size() >= kMatrixCandidates * 4 || (!rows && !unit_length(m[3], m[7], m[11]))) continue;
        Found f{std::uint32_t(offset), rows ? 1u : 3u, value_of(m), {}};
        std::memcpy(f.m, m, sizeof(f.m));
        g_found.push_back(f);
    }
}
// (under the lock) What a range holds, under a slot.
void label(const Read& read, std::uint32_t slot) {
    for (std::size_t i = read.first; i < read.first + read.count && g_raw.size() < kMatrixCandidates; ++i) {
        const Found& f = g_found[i];
        const std::uint32_t key = mix(slot, f.offset);
        if (!g_labelled.insert(f.value ^ (std::uint64_t(key) * 0x9E3779B97F4A7C15ull)).second) continue;
        MatrixCandidate c{};
        c.key = key;
        c.flags = f.flags;
        std::memcpy(c.m, f.m, sizeof(c.m));
        g_raw.push_back(c);
    }
}
// (under the lock) The places held, wherever this slot would have them in what is bound to it.
void read_held(const std::uint8_t* data, std::uint64_t size, std::uint32_t slot) {
    const std::uint32_t count = g_held_count.load(std::memory_order_relaxed);
    for (std::uint32_t i = 0; i < count && g_held_found.size() < 512; ++i) {
        const std::uint32_t offset = (g_held[i] * kUnmix) ^ slot;
        if ((offset & 15) || offset + 64 > size) continue;
        float m[16];
        std::memcpy(m, data + offset, sizeof(m));
        const bool rows = unit_length(m[12], m[13], m[14]);
        if (!can_be_camera(m) || (!rows && !unit_length(m[3], m[7], m[11]))) continue;
        if (!g_labelled.insert(value_of(m) ^ (std::uint64_t(g_held[i]) * 0x9E3779B97F4A7C15ull)).second) continue;
        MatrixCandidate c{};
        c.key = g_held[i];
        c.flags = rows ? 1 : 3;
        std::memcpy(c.m, m, sizeof(c.m));
        g_held_found.push_back(c);
        const std::uint32_t n = g_held_slots.load(std::memory_order_relaxed);
        bool known = false;
        for (std::uint32_t k = 0; k < n; ++k) known = known || g_held_slot[k].load(std::memory_order_relaxed) == slot;
        if (!known && n < kHeldSlots) { g_held_slot[n].store(slot, std::memory_order_relaxed); g_held_slots.store(n + 1, std::memory_order_release); }
    }
}
// (under the lock) What was bound or written just now, seen once.
void read_now(const std::uint8_t* data, std::uint64_t size, std::uint32_t slot) {
    read_held(data, std::min(size, kRangeBytes), slot);
    if (!g_searching.load(std::memory_order_relaxed) || !in_window()) return;
    Read read{g_found.size(), 0};
    scan(data, size);
    read.count = g_found.size() - read.first;
    label(read, slot);
}
std::uint32_t layout_of(pipeline_layout layout) { return std::uint32_t(layout.handle ^ (layout.handle >> 32)); }

void on_push_constants(command_list* cmd, shader_stage, pipeline_layout layout, uint32_t param, uint32_t first, uint32_t count, const void* values) {
    if (!cmd || !values || count < 16) return;
    if (cmd->get_device()->get_api() != device_api::d3d12) return;
    const std::uint32_t slot = mix(mix(mix(1, layout_of(layout)), param), first);
    if (!wanted(slot)) return;
    std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
    if (lock) read_now(static_cast<const std::uint8_t*>(values), std::uint64_t(count) * 4, slot);
}
// (under the lock) An upload buffer's memory, mapped once.
Mapped mapped(device* dev, resource buffer) {
    auto it = g_mapped.find(buffer.handle);
    if (it != g_mapped.end()) return it->second;
    Mapped m;
    void* data = nullptr;
    const resource_desc desc = dev->get_resource_desc(buffer);
    m.heap = std::uint32_t(desc.heap);
    // (gpu_upload: the video memory the CPU can reach with Resizable BAR, where Requiem keeps its 700 constant buffers a frame. custom: maps if its pages can be reached, or does not)
    if (desc.type == resource_type::buffer && (desc.heap == memory_heap::upload || desc.heap == memory_heap::scratch || desc.heap == memory_heap::gpu_upload || desc.heap == memory_heap::custom) &&
        dev->map_buffer_region(buffer, 0, UINT64_MAX, map_access::read_only, &data) && data) {
        m.data = static_cast<std::uint8_t*>(data);
        m.size = desc.buffer.size;
    }
    g_mapped.emplace(buffer.handle, m);
    return m;
}
void read_range(const buffer_range& r, std::uint32_t slot) {  // (under the lock)
    if (!r.buffer.handle) return;
    const std::uint64_t id = r.buffer.handle ^ (r.offset * 0x9E3779B97F4A7C15ull);
    if (!g_labelled.insert(id ^ (std::uint64_t(slot) * 0xC2B2AE3D27D4EB4Full)).second) return;  // (a range under a slot: once a frame)
    const auto it = g_mapped.find(r.buffer.handle);  // (a buffer no view was seen being written for is none of ours: it may be gone by now)
    if (it == g_mapped.end() || !it->second.data || r.offset >= it->second.size) return;
    const std::uint8_t* data = it->second.data + r.offset;
    const std::uint64_t there = std::min(it->second.size - r.offset, kRangeBytes);
    read_held(data, r.size == UINT64_MAX ? there : std::min(r.size, there), slot);
    if (!g_searching.load(std::memory_order_relaxed) || !in_window()) return;
    auto seen = g_read.find(id);
    if (seen == g_read.end()) {
        Read read{g_found.size(), 0};
        scan(data, std::min(r.size == UINT64_MAX ? kUnsizedBytes : r.size, there));
        read.count = g_found.size() - read.first;
        seen = g_read.emplace(id, read).first;
    }
    label(seen->second, slot);
}
void on_push_descriptors(command_list* cmd, shader_stage, pipeline_layout layout, uint32_t param, const descriptor_table_update& update) {
    if (cmd) g_pushed_kind[std::uint32_t(update.type) & 15].fetch_add(update.count, std::memory_order_relaxed);
    if (!cmd || update.type != descriptor_type::constant_buffer || !update.descriptors) return;
    device* dev = cmd->get_device();
    if (dev->get_api() != device_api::d3d12 || (g_budget_spent.load(std::memory_order_relaxed) && !g_held_count.load(std::memory_order_relaxed))) return;
    const auto* ranges = static_cast<const buffer_range*>(update.descriptors);
    const std::uint32_t base = mix(mix(2, layout_of(layout)), param);
    bool any = false;
    for (uint32_t i = 0; i < update.count && !any; ++i) any = wanted(mix(base, update.binding + i));
    if (!any) return;
    std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
    if (!lock) return;
    for (uint32_t i = 0; i < update.count; ++i) {
        if (ranges[i].buffer.handle) g_pushed_heap[mapped(dev, ranges[i].buffer).heap & 7].fetch_add(1, std::memory_order_relaxed);  // (bound now: it exists)
        if (ranges[i].size == UINT64_MAX) g_unsized.fetch_add(1, std::memory_order_relaxed);
        else if (ranges[i].size >= kRangeBytes) g_large.fetch_add(1, std::memory_order_relaxed);
        if (wanted(mix(base, update.binding + i))) read_range(ranges[i], mix(base, update.binding + i));
    }
}
// Direct3D 12: most constants are bound through tables. Which view lies where is noted as the game writes and copies descriptors; what a table holds is read as it is bound.
bool on_update_tables(device* dev, uint32_t count, const descriptor_table_update* updates) {
    if (dev->get_api() != device_api::d3d12) return false;
    Timed timed;
    for (uint32_t u = 0; u < count; ++u) {
        const descriptor_table_update& update = updates[u];
        descriptor_heap heap;
        uint32_t offset;
        if (!update.count || !place_of(dev, update.table, update.binding, update.array_offset, heap, offset)) continue;
        tally(g_written, update.count);
        tally(g_written_kind[std::uint32_t(update.type) & 15], update.count);
        const bool views = update.type == descriptor_type::constant_buffer;
        Heap* h = heap_of(heap.handle, views);
        if (!h) continue;
        const auto* ranges = static_cast<const buffer_range*>(update.descriptors);
        static thread_local std::uint64_t known = 0;  // (a ring of constants is one buffer: the lock once)
        for (uint32_t i = 0; i < update.count; ++i) {
            buffer_range* to = view_at(h, offset + i, views);
            if (!to) continue;
            if (!views) { to->buffer = {0}; continue; }
            if (ranges[i].buffer.handle && ranges[i].buffer.handle != known) {
                std::lock_guard<std::mutex> lock(g_mutex);
                mapped(dev, ranges[i].buffer);
                known = ranges[i].buffer.handle;
            }
            *to = ranges[i];
        }
    }
    return false;
}
bool on_copy_tables(device* dev, uint32_t count, const descriptor_table_copy* copies) {
    if (dev->get_api() != device_api::d3d12) return false;
    Timed timed;
    for (uint32_t c = 0; c < count; ++c) {
        const descriptor_table_copy& copy = copies[c];
        descriptor_heap from_heap, to_heap;
        uint32_t from, to;
        if (!copy.count || !place_of(dev, copy.source_table, copy.source_binding, copy.source_array_offset, from_heap, from) ||
            !place_of(dev, copy.dest_table, copy.dest_binding, copy.dest_array_offset, to_heap, to)) continue;
        tally(g_copied, copy.count);
        Heap* source = heap_of(from_heap.handle, false);
        Heap* dest = heap_of(to_heap.handle, source != nullptr);
        if (!dest) continue;
        for (uint32_t i = 0; i < copy.count;) {
            // (a run within one page of each)
            const uint32_t a = from + i, b = to + i, run = std::min({copy.count - i, kPageViews - a % kPageViews, kPageViews - b % kPageViews});
            const buffer_range* s = source ? view_at(source, a, false) : nullptr;
            buffer_range* d = view_at(dest, b, s != nullptr);
            if (d) {
                if (s) std::memcpy(d, s, run * sizeof(buffer_range));
                else std::memset(d, 0, run * sizeof(buffer_range));
            }
            i += run;
        }
    }
    return false;
}
void on_bind_tables(command_list* cmd, shader_stage, pipeline_layout layout, uint32_t first, uint32_t count, const descriptor_table* tables, uint32_t, const uint32_t*) {
    if (!cmd || !tables || (g_budget_spent.load(std::memory_order_relaxed) && !g_held_count.load(std::memory_order_relaxed))) return;
    device* dev = cmd->get_device();
    if (dev->get_api() != device_api::d3d12) return;
    Timed timed;
    tally(g_bound, count);
    for (uint32_t t = 0; t < count; ++t) {
        // (how long a table is, the layout knows: the first few places are enough for a view's constants)
        const std::uint32_t base = mix(mix(5, layout_of(layout)), first + t);
        bool any = false;
        for (uint32_t i = 0; i < 8 && !any; ++i) any = wanted(mix(base, i));
        descriptor_heap heap;
        uint32_t offset;
        if (!any || !tables[t].handle || !place_of(dev, tables[t], 0, 0, heap, offset)) continue;
        Heap* h = heap_of(heap.handle, false);
        if (!h) continue;
        for (uint32_t i = 0; i < 8; ++i) {
            if (!wanted(mix(base, i))) continue;
            const buffer_range* v = view_at(h, offset + i, false);
            if (!v || !v->buffer.handle) { tally(g_table_empty, 1); continue; }
            tally(g_table_views, 1);
            const buffer_range r = *v;
            // (bound again just before on this thread, the same range in the same slot - draw after draw: read once, without the lock)
            static thread_local std::uint64_t last = 0;
            static thread_local std::uint32_t last_frame = 0;
            const std::uint64_t seen = (r.buffer.handle ^ (r.offset * 0x9E3779B97F4A7C15ull)) ^ (std::uint64_t(mix(base, i)) * 0xC2B2AE3D27D4EB4Full);
            const std::uint32_t frame = g_frame.load(std::memory_order_relaxed);
            if (seen == last && frame == last_frame) continue;
            last = seen; last_frame = frame;
            // (never waited for: a render thread that finds it taken goes on, the camera's constants are bound a hundred times a frame)
            std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
            if (!lock) { tally(g_skipped, 1); return; }
            read_range(r, mix(base, i));
        }
    }
}
// Direct3D 11: constants are written through a map: read as the game finishes.
void on_map(device* dev, resource res, uint64_t, uint64_t size, map_access access, void** data) {
    if (dev->get_api() != device_api::d3d11 || !data || !*data || access == map_access::read_only) return;
    const resource_desc desc = dev->get_resource_desc(res);
    if (desc.type != resource_type::buffer || (desc.usage & resource_usage::constant_buffer) == 0) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_writing[res.handle] = {static_cast<std::uint8_t*>(*data), size == UINT64_MAX ? desc.buffer.size : size};
}
void on_unmap(device* dev, resource res) {
    if (dev->get_api() != device_api::d3d11) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_writing.find(res.handle);
    if (it == g_writing.end()) return;
    const std::uint32_t slot = mix(3, std::uint32_t(std::min(it->second.second, kRangeBytes)));  // (no slot known here: by the buffer's size)
    if (wanted(slot)) read_now(it->second.first, it->second.second, slot);
    g_writing.erase(it);
}
bool on_update_buffer(device* dev, const void* data, resource dest, uint64_t, uint64_t size) {
    if (dev->get_api() != device_api::d3d11 || !data) return false;
    const resource_desc desc = dev->get_resource_desc(dest);
    if (desc.type != resource_type::buffer || (desc.usage & resource_usage::constant_buffer) == 0) return false;
    const std::uint32_t slot = mix(4, std::uint32_t(std::min(size, kRangeBytes)));
    if (!wanted(slot)) return false;
    std::lock_guard<std::mutex> lock(g_mutex);
    read_now(static_cast<const std::uint8_t*>(data), size, slot);
    return false;
}
void on_destroy(device* dev, resource res) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_mapped.find(res.handle);
    if (it == g_mapped.end()) return;
    if (it->second.data) dev->unmap_buffer_region(res);
    it->second = Mapped{};  // (kept: views to it are still listed)
}
}  // namespace

std::uint32_t take(MatrixCandidate* out, std::uint32_t capacity, const volatile long* held, long held_state) {
    if (!g_on) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    // (the places held first: the rest is cut at the capacity)
    const std::uint32_t h = std::uint32_t(std::min<std::size_t>(g_held_found.size(), capacity));
    std::copy(g_held_found.begin(), g_held_found.begin() + h, out);
    const std::uint32_t n = h + std::uint32_t(std::min<std::size_t>(g_raw.size(), capacity - h));
    std::copy(g_raw.begin(), g_raw.begin() + (n - h), out + h);
    if (g_budget_spent) { g_spent_frames.fetch_add(1, std::memory_order_relaxed); for (std::uint32_t i = 0; i < n; ++i) out[i].flags |= 8; }  // (for the dump: this frame was cut short)
    if (g_said < 60 && ++g_rotate % 300 == 150) {
        ++g_said;
        char text[520];
        const double ms = 1000.0 * double(g_ticks.load()) / double(g_frequency.QuadPart) / 300.0;
        std::snprintf(text, sizeof(text), "XPAR: camera search: this frame %llu KB in %llu ranges, %zu matrices, %zu under their slots, %zu from %u places held; a frame: %.2f ms in the hooks (all threads), %llu descriptors written, %llu copied, %llu tables bound, %llu skipped",
                      static_cast<unsigned long long>(g_bytes / 1024), static_cast<unsigned long long>(g_ranges), g_found.size(), g_raw.size(), g_held_found.size(), g_held_count.load(), ms, static_cast<unsigned long long>(g_written.load() / 300),
                      static_cast<unsigned long long>(g_copied.load() / 300), static_cast<unsigned long long>(g_bound.load() / 300), static_cast<unsigned long long>(g_skipped.load() / 300));
        reshade::log::message(reshade::log::level::info, text);
        {
            const double all = double(g_ticks.load()), searching = double(g_ticks_searching.exchange(0)), tick_ms = 1000.0 / double(g_frequency.QuadPart);
            const std::uint64_t searched = g_search_frames.exchange(0);
            std::snprintf(text, sizeof(text), "XPAR: camera search: frames searched whole %llu of 300, the hooks %.2f ms on each of them, %.2f ms on each of the others; searched from read %u of %u",
                          static_cast<unsigned long long>(searched), searched ? searching * tick_ms / double(searched) : 0.0, searched < 300 ? (all - searching) * tick_ms / double(300 - searched) : 0.0, g_window, g_reads_last);
            reshade::log::message(reshade::log::level::info, text);
        }
        const auto list = [](std::atomic<std::uint64_t>* values, int count) {
            std::string out;
            for (int i = 0; i < count; ++i) if (const std::uint64_t n = values[i].exchange(0) / 300) { char one[32]; std::snprintf(one, sizeof(one), " %d:%llu", i, static_cast<unsigned long long>(n)); out += one; }
            return out;
        };
        std::size_t readable = 0;
        for (const auto& [handle, m] : g_mapped) readable += m.data ? 1 : 0;
        const std::string pushed = list(g_pushed_kind, 16), written = list(g_written_kind, 16), heaps = list(g_pushed_heap, 8);
        std::snprintf(text, sizeof(text), "XPAR: camera search: a frame: bound directly, by kind%s; written, by kind%s; constant buffers bound directly, by memory%s; in tables bound, %llu views, %llu places without one; buffers known %zu, readable %zu; bound without a size %llu, of 64 KB or more %llu; budget spent on %llu of 300 frames",
                      pushed.c_str(), written.c_str(), heaps.c_str(), static_cast<unsigned long long>(g_table_views.exchange(0) / 300), static_cast<unsigned long long>(g_table_empty.exchange(0) / 300), g_mapped.size(), readable,
                      static_cast<unsigned long long>(g_unsized.exchange(0) / 300), static_cast<unsigned long long>(g_large.exchange(0) / 300), static_cast<unsigned long long>(g_spent_frames.exchange(0)));
        reshade::log::message(reshade::log::level::info, text);
        g_ticks = 0; g_written = 0; g_copied = 0; g_bound = 0; g_skipped = 0;
    }
    g_raw.clear();
    const bool held_read = !g_held_found.empty();
    g_held_found.clear();
    std::uint32_t count = 0, before[kHeld];
    const std::uint32_t had = g_held_count.load();
    std::copy(g_held, g_held + kHeld, before);
    if (held)
        for (std::uint32_t i = 0; i < kHeld; ++i)
            if (const std::uint32_t key = std::uint32_t(held[i])) g_held[count++] = key;
    g_held_count = count;
    g_frame.fetch_add(1, std::memory_order_relaxed);
    std::uint32_t now[kHeld];
    std::copy(g_held, g_held + count, now);
    std::sort(now, now + count);
    std::sort(before, before + had);
    if (count != had || !std::equal(now, now + count, before)) g_held_slots = 0;  // (other places, not the same in another order: their slots are found again)
    static std::uint32_t frames = 0;
    static std::uint32_t searched = 0;
    if (g_searching && ++searched % 2 == 0) g_window = g_full_at;  // (0 when the frame fitted: round to the start)
    g_reads_last = g_read_index;
    g_read_index = g_full_at = 0;
    // (nothing read from the places held - their slots are found again only by a search, and with nothing sent the presenter judges nothing: its last word stays)
    g_searching = !count || held_state == 2 || !held_read || (held_state == 0 && ++frames % kSearchEvery < kSearchFrames);
    if (g_searching) g_search_frames.fetch_add(1, std::memory_order_relaxed);
    g_found.clear();
    g_read.clear();
    g_labelled.clear();
    g_bytes = g_ranges = 0;
    g_budget_spent = false;
    return n;
}

void init() {
    char value[8] = {};
    size_t size = sizeof(value);
    g_on = reshade::get_config_value(nullptr, "FrameWarp", "CameraSearch", value, &size) && value[0] == '1';
    if (!g_on) return;
    QueryPerformanceFrequency(&g_frequency);
    reshade::register_event<reshade::addon_event::push_constants>(on_push_constants);
    reshade::register_event<reshade::addon_event::push_descriptors>(on_push_descriptors);
    reshade::register_event<reshade::addon_event::update_descriptor_tables>(on_update_tables);
    reshade::register_event<reshade::addon_event::copy_descriptor_tables>(on_copy_tables);
    reshade::register_event<reshade::addon_event::bind_descriptor_tables>(on_bind_tables);
    reshade::register_event<reshade::addon_event::map_buffer_region>(on_map);
    reshade::register_event<reshade::addon_event::unmap_buffer_region>(on_unmap);
    reshade::register_event<reshade::addon_event::update_buffer_region>(on_update_buffer);
    reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy);
}
void shutdown() {
    if (!g_on) return;
    reshade::unregister_event<reshade::addon_event::destroy_resource>(on_destroy);
    reshade::unregister_event<reshade::addon_event::update_buffer_region>(on_update_buffer);
    reshade::unregister_event<reshade::addon_event::unmap_buffer_region>(on_unmap);
    reshade::unregister_event<reshade::addon_event::map_buffer_region>(on_map);
    reshade::unregister_event<reshade::addon_event::bind_descriptor_tables>(on_bind_tables);
    reshade::unregister_event<reshade::addon_event::copy_descriptor_tables>(on_copy_tables);
    reshade::unregister_event<reshade::addon_event::update_descriptor_tables>(on_update_tables);
    reshade::unregister_event<reshade::addon_event::push_descriptors>(on_push_descriptors);
    reshade::unregister_event<reshade::addon_event::push_constants>(on_push_constants);
}
}  // namespace fw::camera_search
