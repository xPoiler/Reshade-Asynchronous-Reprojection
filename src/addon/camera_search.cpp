#include "addon/camera_search.hpp"
#include "common/inline_hook.hpp"
#include <reshade.hpp>
#include <windows.h>
#include <vulkan/vulkan.h>
#include <d3d12.h>
#include <smmintrin.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "addon/diag.hpp"

using namespace reshade::api;

namespace fw::camera_search {
namespace {
constexpr std::uint64_t kBudgetBytes = 192 * 1024;  // (read a frame: an upload heap is slow to read from. At 512 KB Cyberpunk's streets cost 10 to 29 ms over all threads, 42 fps to 30)
constexpr std::uint64_t kRangeBytes = 65536;        // (of each buffer: Direct3D's largest constant buffer view)
constexpr std::uint64_t kUnsizedBytes = 4096;       // (a stopgap: Direct3D 12's root constant buffers carry an address and no size; a whole 64 KB of each and the budget lasts three of them)
constexpr uint32_t kPageViews = 1024;

bool g_on = false;
// (paused - pause(): the game's data comes from its upscaler's hooks, the ReShade feed is off and nothing takes what this would find.
// The hooks then return at once: they run on every descriptor copy and bind of the game's. Map, unmap and destroy keep their
// bookkeeping safe: see pause)
std::atomic<bool> g_paused{false};
// What the game wrote for the GPU to read is write-combined memory, on every API: plain loads from it go to the bus one by one. Read with streaming loads, a
// cache line at a time (SSE4.1; from ordinary memory they are ordinary loads). Measured before (2026-11-04, hooks' time on frames searched whole): reads 3.6-5.8 ms
// a frame in DOOM Eternal and 3.7-8.2 in Enshrouded for at most 192 KB; Dark Souls' copies of its constant buffers 0.7-1.3 ms on every frame.
void wc_copy(void* to, const void* from, std::size_t size) {
    auto* d = static_cast<std::uint8_t*>(to);
    auto* src = static_cast<const std::uint8_t*>(from);
    const std::size_t head = std::min<std::size_t>(size, (16 - (reinterpret_cast<std::uintptr_t>(src) & 15)) & 15);
    std::memcpy(d, src, head);
    d += head; src += head; size -= head;
    for (; size >= 64; size -= 64, src += 64, d += 64) {
        const __m128i a = _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<std::uint8_t*>(src)));
        const __m128i b = _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<std::uint8_t*>(src + 16)));
        const __m128i c = _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<std::uint8_t*>(src + 32)));
        const __m128i e = _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<std::uint8_t*>(src + 48)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d), a);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 16), b);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 32), c);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 48), e);
    }
    for (; size >= 16; size -= 16, src += 16, d += 16) _mm_storeu_si128(reinterpret_cast<__m128i*>(d), _mm_stream_load_si128(reinterpret_cast<__m128i*>(const_cast<std::uint8_t*>(src))));
    std::memcpy(d, src, size);
}
std::mutex g_mutex;  // (the frame's matrices and the mapped buffers: never waited for on the game's render threads, see bind)
// Where a matrix lies is the slot it reaches the shaders through and its offset in what is bound there, never the memory: that moves on through a ring every frame. A range is read
// once a frame, however often it is bound, and each slot it is bound to gets its matrices under its own key: keys given by whichever bind came first follow the game's threads.
struct Found { std::uint32_t offset, flags; std::uint64_t value; float m[16]; };
struct Read { std::size_t first, count; };
std::vector<Found> g_found;                                        // (the matrices of what was read this frame)
std::unordered_map<std::uint64_t, Read> g_read;                    // (ranges read this frame)
// (a range under a slot, a value under a key: once a frame. Open addressing, emptied by a new generation: a hash set's node for each of DOOM Eternal's 40000 tables bound a
// frame while searching was most of its 7-9 ms in the binds)
struct Once {
    struct Entry { std::uint64_t key; std::uint32_t generation; };
    std::vector<Entry> table;
    explicit Once(std::size_t size = std::size_t(1) << 14) : table(size, Entry{0, 0}) {}
    std::uint32_t generation = 1;
    std::size_t used = 0;
    static std::uint64_t spread(std::uint64_t k) { k ^= k >> 33; k *= 0xFF51AFD7ED558CCDull; k ^= k >> 33; return k; }
    bool place(std::uint64_t key) {
        const std::size_t mask = table.size() - 1;
        for (std::size_t i = spread(key) & mask;; i = (i + 1) & mask) {
            Entry& e = table[i];
            if (e.generation != generation) { e = {key, generation}; ++used; return true; }
            if (e.key == key) return false;
        }
    }
    struct Inserted { bool second; };
    Inserted insert(std::uint64_t key) {
        if ((used + 1) * 2 > table.size()) {
            std::vector<Entry> old;
            old.swap(table);
            table.assign(old.size() * 2, Entry{0, 0});
            used = 0;
            for (const Entry& e : old) if (e.generation == generation) place(e.key);
        }
        return {place(key)};
    }
    void clear() { if (++generation == 0) { std::fill(table.begin(), table.end(), Entry{0, 0}); generation = 1; } used = 0; }
};
Once g_labelled;
// (copy: Direct3D 11's, what was last written to it, its version counted - a buffer is written again and again in a frame, each time for the draws after)
// (game: Vulkan's, read through the game's own pointer - never unmapped by us)
struct Mapped { std::uint8_t* data = nullptr; std::uint64_t size = 0; std::uint32_t heap = 0, version = 0; bool copy = false, game = false; };
std::unordered_map<std::uint64_t, Mapped> g_mapped;                // (Direct3D 12: upload buffers, mapped for reading; data 0: unreadable, or gone)
std::unordered_map<std::uint64_t, std::pair<std::uint8_t*, std::uint64_t>> g_writing;  // (Direct3D 11: the buffer being filled)
std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> g_copies;                 // (Direct3D 11: what each constant buffer was last filled with)
std::uint64_t g_copied_bytes = 0;                                  // (Direct3D 11, for the log)
// (Direct3D 11, between searches: only the buffers bound to the slots of the places held are copied as the game writes them - Dark Souls copied all 2000 a frame, 0.5-1.6
// ms, to read a few. Any other's copy is dropped at once, never read stale: if the camera moves on to it, the places held miss and the search starts again)
std::unordered_set<std::uint64_t> g_held_buffers;
// (Direct3D 11) Each constant buffer: the slot it was last bound to, and whether it was bound since its last write. Most games bind a
// buffer after filling it, and it is read there; a game that binds its buffers once and refills them in place (Metro 2033 Redux: 14
// buffers, about 1400 writes a frame, no binds at all) is read at the write instead - a write with no bind since the one before. Its own
// lock: binds are counted before anything else is decided, and between searches most take nothing more.
struct D3D11Buffer { std::uint32_t key = 0; bool keyed = false, bound = false, written = false; };
std::mutex g_d3d11_mutex;
std::unordered_map<std::uint64_t, D3D11Buffer> g_d3d11;
std::uint64_t g_bytes = 0, g_ranges = 0;
std::atomic<bool> g_budget_spent{false};
std::uint32_t g_rotate = 0;
std::vector<MatrixCandidate> g_raw;
// (CameraDump=2, development - plan_character_coordinates.md, Phase 0: every frame searched, and the object matrices too - an object's
// place in the world, rotation and uniform scale with a move, which no camera's shape passes - flags 4, 2 when stored in columns. Only
// for the dump: they go after the camera's values, and the presenter's camera search never sees them)
int g_dump = 0;
std::vector<MatrixCandidate> g_raw_objects;
std::size_t g_found_objects = 0;
// (CameraDump=2, round 2: no budget and no window - the whole frame read, up to kDumpRangeBytes of each range; Direct3D 12's buffer views
// too - root ones, and those in tables - where objects' and bones' transforms are kept; and every value found written by the add-on itself,
// logs/camera_values.bin, by frame: the 2048 the presenter is sent cannot hold them. Started once a camera place is held - the game is being played)
constexpr std::uint64_t kDumpRangeBytes = 1u << 20;
constexpr std::size_t kDumpValues = 1u << 16;
constexpr std::uint64_t kDumpFileBytes = 3ull << 30;
FILE* g_values_file = nullptr;
std::uint64_t g_values_bytes = 0;
bool g_values_started = false;
std::uint64_t g_dump_srv_views = 0, g_dump_root_views = 0, g_dump_root_read = 0;
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
// (a bit for each held slot: most binds are of none - DOOM Eternal binds 40000 tables a frame, 8 places each looked up through the whole list took about a ms. And past the
// budget a frame searched reads nothing more but the places held: the rest was looked up for nothing)
std::atomic<std::uint64_t> g_held_mask{0};
std::uint64_t bit_of(std::uint32_t slot) { return 1ull << ((slot * 0x9E3779B1u) >> 26); }
bool wanted(std::uint32_t slot) {
    if (g_dump == 2) return true;
    if (g_searching.load(std::memory_order_relaxed) && !g_budget_spent.load(std::memory_order_relaxed)) return true;
    if (!(g_held_mask.load(std::memory_order_relaxed) & bit_of(slot))) return false;
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
std::atomic<std::uint64_t> g_ticks_searching{0}, g_search_frames{0}, g_search_frames_parts{0};  // (the hooks' time on frames searched whole, and those frames)
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
// (diagnostics, 2026-11-04: where the hooks' time goes, by part - [0] all frames, [1] frames searched whole. Binds, pushes, descriptor writes, copies, Direct3D 11
// writes are the hooks; places (ReShade's descriptor position lookups), reads (copying what is bound and looking through it) and held reads are within them)
// (parts timed for each of tens of thousands of items a frame - position lookups, held reads, copied entries - measured mostly the clock: 42000 copies' entries took
// 1.3 ms, about what reading the clock costs. Timed now by the hook call only, and the clock's own cost taken off)
enum Part { kBind, kPush, kWrite, kCopy, kD3D11, kPlace, kScan, kParts };
const char* const kPartNames[kParts] = {"binds", "pushes", "descriptor writes", "copies", "Direct3D 11 writes", "-", "reads"};
std::int64_t g_clock_cost = 0;  // (ticks between two readings of the clock, the least of many: taken off each one timed)
std::atomic<std::uint64_t> g_part_ticks[2][kParts], g_part_calls[kParts];
thread_local bool t_timing = false;  // (this call is one of those timed: so are the parts within it)
// (one call in eight timed, counted eight times: the clock read twice at every bind is a cost of its own)
struct Timed {
    bool timing, searching;
    int part;
    LARGE_INTEGER start{};
    explicit Timed(int p) : timing(fw::diag() && t_pending.calls % 8 == 0), searching(g_searching.load(std::memory_order_relaxed)), part(p) {
        tally(g_part_calls[p], 1);
        if (timing) { t_timing = true; QueryPerformanceCounter(&start); }
    }
    ~Timed() {
        if (timing) {
            LARGE_INTEGER end;
            QueryPerformanceCounter(&end);
            const std::uint64_t t = 8 * std::uint64_t(std::max<std::int64_t>(end.QuadPart - start.QuadPart - g_clock_cost, 0));
            tally(g_ticks, t);
            tally(g_part_ticks[0][part], t);
            if (searching) { tally(g_ticks_searching, t); tally(g_part_ticks[1][part], t); }
            t_timing = false;
        }
        settle();
    }
};
// (a part within a hook: timed when the hook is, counted as it is)
struct Within {
    bool timing;
    int part;
    LARGE_INTEGER start{};
    explicit Within(int p) : timing(t_timing), part(p) { if (timing) QueryPerformanceCounter(&start); }
    ~Within() {
        if (!timing) return;
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        const std::uint64_t t = 8 * std::uint64_t(std::max<std::int64_t>(end.QuadPart - start.QuadPart - g_clock_cost, 0));
        tally(g_part_ticks[0][part], t);
        if (g_searching.load(std::memory_order_relaxed)) tally(g_part_ticks[1][part], t);
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
bool holds_ranges(descriptor_type t) {  // (descriptors that are buffer_range: constants and storage buffers, with dynamic offsets or not)
    return t == descriptor_type::constant_buffer || t == descriptor_type::shader_storage_buffer || t == descriptor_type::constant_buffer_with_dynamic_offset || t == descriptor_type::shader_storage_buffer_with_dynamic_offset;
}
// Direct3D 12: where a table lies, worked out by us - ReShade's lookup took Cyberpunk 1.6-3.2 ms a frame over its 15000-22000 lookups. ReShade's tables are not the
// game's descriptor pointers but its own (0xf000000041d84804: the heap's tag, then 967232 descriptors of 32 bytes - logged 2026-11-04): each heap's base is learnt from
// ReShade's first answer in it (one for the CPU's tables, one for the GPU's), its step and size from the heap, and a base is trusted once 64 more answers matched; one that
// differs is never trusted again, and ReShade is asked for it as before.
struct Base { std::uint64_t value = 0; std::atomic<int> state{0}, matched{0}; };  // (state: 0 checking, 1 trusted, -1 not)
struct KnownHeap { std::uint64_t handle = 0, count = 0, step = 0; bool usable = false; Base bases[2]; std::atomic<int> based{0}; };
KnownHeap g_known[64];
std::atomic<int> g_known_count{0};
std::mutex g_known_mutex;
bool heap_size(std::uint64_t handle, ID3D12Device* device, KnownHeap& k) {  // (SEH: a handle that is not a heap's must not take the game down)
    __try {
        const D3D12_DESCRIPTOR_HEAP_DESC desc = reinterpret_cast<ID3D12DescriptorHeap*>(handle)->GetDesc();
        k.count = desc.NumDescriptors;
        k.step = device->GetDescriptorHandleIncrementSize(desc.Type);
        return k.step != 0 && k.count != 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}
// (the base of k holding h, -1 none)
int base_of(const KnownHeap& k, std::uint64_t h) {
    const int n = k.based.load(std::memory_order_acquire);
    for (int b = 0; b < n; ++b)
        if (h >= k.bases[b].value && h < k.bases[b].value + k.count * k.step) return b;
    return -1;
}
bool place_of(device* dev, descriptor_table table, uint32_t binding, uint32_t array_offset, descriptor_heap& heap, uint32_t& offset) {
    heap = {0}; offset = 0;
    const bool d3d12 = dev->get_api() == device_api::d3d12;
    if (d3d12) {
        // (one in 1024 asked of ReShade still: a heap released and another made where it was must not keep our answer)
        static thread_local unsigned recheck = 0;
        const bool check = ++recheck % 1024 == 0;
        const int n = g_known_count.load(std::memory_order_acquire);
        // (the heap last found on this thread first: Cyberpunk copies descriptors one at a time, 42000 a frame, nearly all from one heap to another)
        static thread_local int last = 0;
        for (int j = 0; j < n; ++j) {
            const int i = (last + j) % n;
            KnownHeap& k = g_known[i];
            const int b = k.usable ? base_of(k, table.handle) : -1;
            if (b < 0 || k.bases[b].state.load(std::memory_order_relaxed) != 1) continue;
            offset = std::uint32_t((table.handle - k.bases[b].value) / k.step) + binding + array_offset;
            last = i;
            if (!check) { heap = {k.handle}; return true; }
            descriptor_heap theirs{0}; uint32_t at = 0;
            dev->get_descriptor_heap_offset(table, binding, array_offset, &theirs, &at);
            if (theirs.handle != k.handle || at != offset) k.bases[b].state = -1;
            heap = theirs; offset = at;
            return heap.handle != 0;
        }
    }
    dev->get_descriptor_heap_offset(table, binding, array_offset, &heap, &offset);
    if (!d3d12 || !heap.handle) return heap.handle != 0;
    // (learning: this heap's bases, checked against ReShade's answers)
    int i = 0;
    for (const int n = g_known_count.load(std::memory_order_acquire); i < n && g_known[i].handle != heap.handle; ++i) {}
    if (i == g_known_count.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_known_mutex);
        const int now = g_known_count.load(std::memory_order_relaxed);
        for (i = 0; i < now && g_known[i].handle != heap.handle; ++i) {}
        if (i == now) {
            if (now >= 64) return true;
            KnownHeap& k = g_known[now];
            k.handle = heap.handle;
            k.usable = heap_size(heap.handle, reinterpret_cast<ID3D12Device*>(dev->get_native()), k);
            g_known_count.store(now + 1, std::memory_order_release);
        }
    }
    KnownHeap& k = g_known[i];
    if (!k.usable) return true;
    const std::uint32_t into = offset - std::min(offset, binding + array_offset);
    int b = base_of(k, table.handle);
    if (b < 0) {
        std::lock_guard<std::mutex> lock(g_known_mutex);
        b = base_of(k, table.handle);
        const int n = k.based.load(std::memory_order_relaxed);
        if (b < 0 && n < 2 && offset >= binding + array_offset && table.handle >= std::uint64_t(into) * k.step) {
            k.bases[n].value = table.handle - std::uint64_t(into) * k.step;
            k.based.store(n + 1, std::memory_order_release);
        }
        return true;
    }
    Base& base = k.bases[b];
    if (base.state.load(std::memory_order_relaxed) == 0) {
        const std::uint32_t mine = std::uint32_t((table.handle - base.value) / k.step) + binding + array_offset;
        if (mine != offset) {
            if (base.state.exchange(-1) == 0) {
                char text[400];
                std::snprintf(text, sizeof(text), "XPAR: camera search: Direct3D 12 heap %llx, a base left to ReShade: table %llx binding %u + %u at %u by ReShade, %u by us; base %llx, %llu descriptors of %llu bytes",
                              static_cast<unsigned long long>(k.handle), static_cast<unsigned long long>(table.handle), binding, array_offset, offset, mine, static_cast<unsigned long long>(base.value),
                              static_cast<unsigned long long>(k.count), static_cast<unsigned long long>(k.step));
                fw::diag_log(reshade::log::level::warning, text);
            }
        } else if (base.matched.fetch_add(1, std::memory_order_relaxed) + 1 >= 64) { int checking = 0; base.state.compare_exchange_strong(checking, 1); }
    }
    return true;
}

std::uint32_t mix(std::uint32_t h, std::uint32_t v) { return (h ^ v) * 16777619u; }

// Vulkan: what a buffer holds, through the game's own pointer. ReShade knows no game buffer's memory (every one 'unknown' in DOOM Eternal and Enshrouded, 2026-11-04), has no
// event for vkMapMemory, and memory mapped twice is invalid in Vulkan. So the device's vkBindBufferMemory(2), vkMapMemory(2), vkUnmapMemory(2) and vkFreeMemory are hooked where
// the game's calls go in (what vkGetDeviceProcAddr hands out), and a buffer bound to memory the game keeps mapped is read where it is bound, as Direct3D 12's upload buffers.
namespace vk {
struct Memory { std::uint8_t* data = nullptr; VkDeviceSize offset = 0, size = 0; std::vector<std::uint64_t> buffers; };
struct Buffer { std::uint64_t memory = 0; VkDeviceSize offset = 0, size = 0; };
std::unordered_map<std::uint64_t, Memory> g_memory;  // (under g_mutex, as all of these)
std::unordered_map<std::uint64_t, Buffer> g_buffers;
VkDevice g_device = VK_NULL_HANDLE;
device* g_rs = nullptr;
std::atomic<std::uint64_t> g_maps{0}, g_unmaps{0}, g_frees{0}, g_binds{0}, g_dynamic{0};
std::string g_hooked;  // (for the log: which were hooked, or why not)
// (under the lock) A buffer's pointer, from its memory's map: or none.
void refresh(std::uint64_t buffer) {
    const auto b = g_buffers.find(buffer);
    if (b == g_buffers.end()) return;
    Mapped& m = g_mapped[buffer];
    m.game = true;
    m.data = nullptr; m.size = 0;
    const auto mem = g_memory.find(b->second.memory);
    if (mem == g_memory.end() || !mem->second.data || b->second.offset < mem->second.offset) return;
    const VkDeviceSize into = b->second.offset - mem->second.offset;
    if (mem->second.size != VK_WHOLE_SIZE && into >= mem->second.size) return;
    const VkDeviceSize there = mem->second.size == VK_WHOLE_SIZE ? b->second.size : mem->second.size - into;
    m.size = b->second.size ? std::min(b->second.size, there) : there;
    if (m.size) m.data = mem->second.data + into;
    ++m.version;
}
void bound(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
    const std::uint64_t handle = reinterpret_cast<std::uint64_t>(buffer);
    const std::uint64_t size = g_rs ? g_rs->get_resource_desc(resource{handle}).buffer.size : 0;  // (before the lock: ReShade's own)
    std::lock_guard<std::mutex> lock(g_mutex);
    g_binds.fetch_add(1, std::memory_order_relaxed);
    g_buffers[handle] = Buffer{reinterpret_cast<std::uint64_t>(memory), offset, size};
    g_memory[reinterpret_cast<std::uint64_t>(memory)].buffers.push_back(handle);
    refresh(handle);
}
void mapped_now(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, void* data) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_maps.fetch_add(1, std::memory_order_relaxed);
    Memory& m = g_memory[reinterpret_cast<std::uint64_t>(memory)];
    m.data = static_cast<std::uint8_t*>(data); m.offset = offset; m.size = size;
    for (const std::uint64_t b : m.buffers) if (g_buffers.count(b) && g_buffers[b].memory == reinterpret_cast<std::uint64_t>(memory)) refresh(b);
}
void unmapped(VkDeviceMemory memory, bool freed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    (freed ? g_frees : g_unmaps).fetch_add(1, std::memory_order_relaxed);
    const auto it = g_memory.find(reinterpret_cast<std::uint64_t>(memory));
    if (it == g_memory.end()) return;
    it->second.data = nullptr;
    const std::vector<std::uint64_t> buffers = std::move(it->second.buffers);
    if (freed) g_memory.erase(it); else it->second.buffers = buffers;
    for (const std::uint64_t b : buffers) {
        const auto at = g_buffers.find(b);
        if (at == g_buffers.end() || at->second.memory != reinterpret_cast<std::uint64_t>(memory)) continue;
        refresh(b);
        if (freed) g_buffers.erase(at);
    }
}
InlineHook g_bind_hook, g_bind2_hook, g_map_hook, g_map2_hook, g_unmap_hook, g_unmap2_hook, g_free_hook;
VkResult VKAPI_CALL bind_memory(VkDevice dev, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
    const VkResult r = reinterpret_cast<PFN_vkBindBufferMemory>(g_bind_hook.original())(dev, buffer, memory, offset);
    if (r == VK_SUCCESS && dev == g_device) bound(buffer, memory, offset);
    return r;
}
VkResult VKAPI_CALL bind_memory2(VkDevice dev, uint32_t count, const VkBindBufferMemoryInfo* infos) {
    const VkResult r = reinterpret_cast<PFN_vkBindBufferMemory2>(g_bind2_hook.original())(dev, count, infos);
    if (r == VK_SUCCESS && dev == g_device && infos) for (uint32_t i = 0; i < count; ++i) bound(infos[i].buffer, infos[i].memory, infos[i].memoryOffset);
    return r;
}
VkResult VKAPI_CALL map_memory(VkDevice dev, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** data) {
    const VkResult r = reinterpret_cast<PFN_vkMapMemory>(g_map_hook.original())(dev, memory, offset, size, flags, data);
    if (r == VK_SUCCESS && dev == g_device && data && *data) mapped_now(memory, offset, size, *data);
    return r;
}
VkResult VKAPI_CALL map_memory2(VkDevice dev, const VkMemoryMapInfoKHR* info, void** data) {
    const VkResult r = reinterpret_cast<PFN_vkMapMemory2KHR>(g_map2_hook.original())(dev, info, data);
    if (r == VK_SUCCESS && dev == g_device && info && data && *data) mapped_now(info->memory, info->offset, info->size, *data);
    return r;
}
void VKAPI_CALL unmap_memory(VkDevice dev, VkDeviceMemory memory) {
    if (dev == g_device) unmapped(memory, false);  // (before: the pointer goes with it)
    reinterpret_cast<PFN_vkUnmapMemory>(g_unmap_hook.original())(dev, memory);
}
VkResult VKAPI_CALL unmap_memory2(VkDevice dev, const VkMemoryUnmapInfoKHR* info) {
    if (dev == g_device && info) unmapped(info->memory, false);
    return reinterpret_cast<PFN_vkUnmapMemory2KHR>(g_unmap2_hook.original())(dev, info);
}
void VKAPI_CALL free_memory(VkDevice dev, VkDeviceMemory memory, const VkAllocationCallbacks* allocator) {
    if (dev == g_device && memory) unmapped(memory, true);
    reinterpret_cast<PFN_vkFreeMemory>(g_free_hook.original())(dev, memory, allocator);
}
// The hooks go where the game's calls go in. Inside vkCreateDevice (init_device) the loader does not know the device yet - its vkGetDeviceProcAddr gave nothing for any
// name in Enshrouded (2026-11-04) - so ReShade's own layer is asked first (everything the game calls passes through it, or through what it hands on to), and if that gives
// nothing either, both are asked again at the first swap chain: memory mapped before then is missed (logged by the counts).
std::vector<std::uint8_t*> g_targets;  // (two names leading to the same code: hooked once - a second hook would land in the first one's detour)
void try_hooks(PFN_vkGetDeviceProcAddr get, VkDevice native, const char* route) {
    if (!get) { g_hooked += std::string(" ") + route + ": no vkGetDeviceProcAddr;"; return; }
    g_hooked += std::string(" ") + route + ":";
    const auto hook = [&](InlineHook& h, const char* name, void* detour) {
        if (h.installed()) return true;
        void* f = reinterpret_cast<void*>(get(native, name));
        if (!f) { g_hooked += std::string(" ") + name + "=none"; return false; }
        std::uint8_t* code = follow_jumps(static_cast<std::uint8_t*>(f));
        if (std::find(g_targets.begin(), g_targets.end(), code) != g_targets.end()) { g_hooked += std::string(" ") + name + "=shared"; return false; }
        if (!h.install(f, detour)) { g_hooked += std::string(" ") + name + "=" + h.error(); return false; }
        g_targets.push_back(code);
        g_hooked += std::string(" ") + name + "=ok";
        return true;
    };
    // (order: free and unmap first - a pointer must never outlive its memory in our tables)
    hook(g_free_hook, "vkFreeMemory", reinterpret_cast<void*>(&free_memory));
    hook(g_unmap_hook, "vkUnmapMemory", reinterpret_cast<void*>(&unmap_memory));
    hook(g_unmap2_hook, "vkUnmapMemory2KHR", reinterpret_cast<void*>(&unmap_memory2));
    if (g_free_hook.installed() && g_unmap_hook.installed()) {
        hook(g_map_hook, "vkMapMemory", reinterpret_cast<void*>(&map_memory));
        hook(g_map2_hook, "vkMapMemory2KHR", reinterpret_cast<void*>(&map_memory2));
        hook(g_bind_hook, "vkBindBufferMemory", reinterpret_cast<void*>(&bind_memory));
        hook(g_bind2_hook, "vkBindBufferMemory2", reinterpret_cast<void*>(&bind_memory2));
    } else g_hooked += " (no free/unmap: nothing read)";
    g_hooked += ";";
}
void try_both(const char* when) {
    if (!g_device || (g_map_hook.installed() && g_bind_hook.installed())) return;
    std::lock_guard<std::mutex> lock(g_mutex);  // (g_hooked is read by the log)
    const HMODULE reshade = reshade::internal::get_reshade_module_handle();
    const HMODULE loader = GetModuleHandleW(L"vulkan-1.dll");
    g_hooked += std::string(" [") + when + "]";
    try_hooks(reshade ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(reshade, "vkGetDeviceProcAddr")) : nullptr, g_device, "layer");
    if (!(g_map_hook.installed() && g_bind_hook.installed()))
        try_hooks(loader ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(loader, "vkGetDeviceProcAddr")) : nullptr, g_device, "loader");
}
void on_init_device(device* dev) {
    if (dev->get_api() != device_api::vulkan || g_device) return;
    g_rs = dev;
    g_device = reinterpret_cast<VkDevice>(dev->get_native());
    try_both("device");
}
void on_init_swapchain(swapchain* sc, bool) {
    if (sc && sc->get_device() == g_rs) try_both("swap chain");
}
}  // namespace vk

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
std::uint64_t value_of(const float* m);
std::uint32_t object_form(const float* m);
std::uint32_t mix(std::uint32_t a, std::uint32_t b);
// (CameraDump=2, under the lock) A range's values for the add-on's file: the camera's shape (flags 1 / 3) and objects' (4 / 6), whatever the
// search's budget and window - the search itself goes on as without the dump, and what the presenter is sent stays the same (round 2's
// first build read the frame whole for the search too: the 2048 sent were the frame's first five buffers, the camera never in them, and the
// camera shook)
std::vector<MatrixCandidate> g_dump_found;
std::unordered_set<std::uint64_t> g_dump_seen;
std::uint64_t g_dump_bytes = 0, g_dump_ranges = 0;
void dump_read(const std::uint8_t* data, std::uint64_t size, std::uint32_t slot) {
    size = std::min(size, kDumpRangeBytes);
    if (size < 64 || g_dump_found.size() >= kDumpValues) return;
    g_dump_bytes += size; ++g_dump_ranges;
    static thread_local std::vector<float> local;
    if (local.size() < kDumpRangeBytes / 4) local.resize(kDumpRangeBytes / 4);
    wc_copy(local.data(), data, size);
    for (std::uint64_t offset = 0; offset + 64 <= size && g_dump_found.size() < kDumpValues; offset += 16) {
        const float* m = local.data() + offset / 4;
        std::uint32_t flags = 0;
        if (can_be_camera(m)) {
            if (unit_length(m[12], m[13], m[14])) flags = 1;
            else if (unit_length(m[3], m[7], m[11])) flags = 3;
        } else {
            flags = object_form(m);
        }
        if (!flags) continue;
        // (each place in memory once a frame: a ring of constants bound at many places gave its values again under each - round 2's first
        // run filled the 65536 from 7 bindings)
        if (!g_dump_seen.insert(reinterpret_cast<std::uintptr_t>(data) + offset).second) continue;
        const std::uint32_t key = mix(slot, std::uint32_t(offset));
        MatrixCandidate c{};
        c.key = key; c.flags = flags;
        std::memcpy(c.m, m, sizeof(c.m));
        g_dump_found.push_back(c);
    }
}
// An object's matrix (CameraDump=2): the three rows of a 3x4 - or the three columns of a 4x4 stored in columns, its move in the last
// one - at right angles and of one length, finite, not the identity left in place. 4: the move in m[3], m[7], m[11]; 6: in m[12..14].
// (The tolerance only sorts what is dumped: the dump is read offline)
std::uint32_t object_form(const float* m) {
    for (int i = 0; i < 16; ++i) if (!std::isfinite(m[i]) || std::fabs(m[i]) > 1e6f) return 0;
    const float* r[3] = {m, m + 4, m + 8};
    const auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const float s0 = dot(r[0], r[0]), s1 = dot(r[1], r[1]), s2 = dot(r[2], r[2]);
    if (s0 < 1e-4f || s0 > 1e4f || std::fabs(s1 - s0) > 0.02f * s0 || std::fabs(s2 - s0) > 0.02f * s0) return 0;
    if (std::fabs(dot(r[0], r[1])) > 0.02f * s0 || std::fabs(dot(r[0], r[2])) > 0.02f * s0 || std::fabs(dot(r[1], r[2])) > 0.02f * s0) return 0;
    const bool columns = m[3] == 0.0f && m[7] == 0.0f && m[11] == 0.0f && m[15] == 1.0f;
    const float tx = columns ? m[12] : m[3], ty = columns ? m[13] : m[7], tz = columns ? m[14] : m[11];
    if (tx == 0.0f && ty == 0.0f && tz == 0.0f && m[0] == 1.0f && m[5] == 1.0f && m[10] == 1.0f) return 0;
    return columns ? 6u : 4u;
}
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
    // (where the frame filled up, noted here: once the budget is spent the hooks read nothing more, and the window never moved on - Dark Souls read from 0 on every frame,
    // its camera past the budget in some places, never found there)
    if (g_bytes + size > kBudgetBytes) { g_budget_spent = true; if (!g_full_at && g_read_index) g_full_at = g_read_index - 1; return; }
    Within within(kScan);
    g_bytes += size;
    ++g_ranges;
    static thread_local float local[kRangeBytes / 4];
    wc_copy(local, data, size);  // (at once: written-combined memory is read in one go, and the game writes on)
    for (std::uint64_t offset = 0; offset + 64 <= size; offset += 16) {
        const float* m = local + offset / 4;
        if (!can_be_camera(m)) {
            if (g_dump == 2 && g_found_objects < kMatrixCandidates * 2)
                if (const std::uint32_t form = object_form(m)) {
                    Found f{std::uint32_t(offset), form, value_of(m), {}};
                    std::memcpy(f.m, m, sizeof(f.m));
                    g_found.push_back(f);
                    ++g_found_objects;
                }
            continue;
        }
        // Every value of the shape, however many a place holds in a frame, in the order they came. Clip w is the distance along the view, so one row of world to clip - the last,
        // or the last column when stored the other way round - is the camera's forward: length 1. (Keeping the first of a place, those that stay, 64 of them: Cyberpunk's
        // camera comes around the 900th, and not always first in its place: with those it was in what the presenter got on 12% of the frames that move, with none 75%.)
        const bool rows = unit_length(m[12], m[13], m[14]);
        if (g_found.size() - g_found_objects >= kMatrixCandidates * 4 || (!rows && !unit_length(m[3], m[7], m[11]))) continue;
        Found f{std::uint32_t(offset), rows ? 1u : 3u, value_of(m), {}};
        std::memcpy(f.m, m, sizeof(f.m));
        g_found.push_back(f);
    }
}
// (under the lock) What a range holds, under a slot.
void label(const Read& read, std::uint32_t slot) {
    for (std::size_t i = read.first; i < read.first + read.count; ++i) {
        const Found& f = g_found[i];
        std::vector<MatrixCandidate>& to = (f.flags & 4) ? g_raw_objects : g_raw;
        if (to.size() >= kMatrixCandidates) continue;
        const std::uint32_t key = mix(slot, f.offset);
        if (!g_labelled.insert(f.value ^ (std::uint64_t(key) * 0x9E3779B97F4A7C15ull)).second) continue;
        MatrixCandidate c{};
        c.key = key;
        c.flags = f.flags;
        std::memcpy(c.m, f.m, sizeof(c.m));
        to.push_back(c);
    }
}
// (under the lock) The places held, wherever this slot would have them in what is bound to it.
void read_held(const std::uint8_t* data, std::uint64_t size, std::uint32_t slot) {
    const std::uint32_t count = g_held_count.load(std::memory_order_relaxed);
    if (!count) return;
    for (std::uint32_t i = 0; i < count && g_held_found.size() < 512; ++i) {
        const std::uint32_t offset = (g_held[i] * kUnmix) ^ slot;
        if ((offset & 15) || offset + 64 > size) continue;
        float m[16];
        wc_copy(m, data + offset, sizeof(m));
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
        if (!known && n < kHeldSlots) { g_held_slot[n].store(slot, std::memory_order_relaxed); g_held_mask.fetch_or(bit_of(slot), std::memory_order_relaxed); g_held_slots.store(n + 1, std::memory_order_release); }
    }
}
// (under the lock) What was bound or written just now, seen once.
void read_now(const std::uint8_t* data, std::uint64_t size, std::uint32_t slot) {
    read_held(data, std::min(size, kRangeBytes), slot);
    if (g_dump == 2) dump_read(data, size, slot);
    if (!g_searching.load(std::memory_order_relaxed) || !in_window()) return;
    Read read{g_found.size(), 0};
    scan(data, size);
    read.count = g_found.size() - read.first;
    label(read, slot);
}
std::uint32_t layout_of(pipeline_layout layout) { return std::uint32_t(layout.handle ^ (layout.handle >> 32)); }

void on_push_constants(command_list* cmd, shader_stage, pipeline_layout layout, uint32_t param, uint32_t first, uint32_t count, const void* values) {
    if (g_paused.load(std::memory_order_relaxed) || !cmd || !values || count < 16) return;
    if (cmd->get_device()->get_api() != device_api::d3d12 && cmd->get_device()->get_api() != device_api::vulkan) return;
    const std::uint32_t slot = mix(mix(mix(1, layout_of(layout)), param), first);
    if (!wanted(slot)) return;
    std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
    if (lock) read_now(static_cast<const std::uint8_t*>(values), std::uint64_t(count) * 4, slot);
}
// (under the lock) An upload buffer's memory, mapped once.
Mapped mapped(device* dev, resource buffer) {
    auto it = g_mapped.find(buffer.handle);
    if (it != g_mapped.end()) return it->second;
    if (dev->get_api() == device_api::vulkan) return Mapped{};  // (Vulkan: noted by the hooks, see vk)
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
    // (a buffer no view was seen being written for is none of ours: it may be gone by now. The last few found, kept on each thread - a few ring buffers hold most
    // views, and a lookup in a map of DOOM Eternal's 60000 buffers for each of 300000 views a frame searched was a cost of its own. Entries are never erased: a pointer stays good)
    struct Cached { std::uint64_t handle; std::pair<const std::uint64_t, Mapped>* entry; };
    static thread_local Cached cache[4] = {};
    static thread_local unsigned next = 0;
    std::pair<const std::uint64_t, Mapped>* entry = nullptr;
    for (const Cached& c : cache) if (c.handle == r.buffer.handle) { entry = c.entry; break; }
    if (!entry) {
        const auto found = g_mapped.find(r.buffer.handle);
        if (found == g_mapped.end()) return;
        entry = &*found;
        cache[next++ % 4] = {r.buffer.handle, entry};
    }
    const auto it = entry;
    if (!it->second.data || r.offset >= it->second.size) return;
    const std::uint64_t id = r.buffer.handle ^ (r.offset * 0x9E3779B97F4A7C15ull) ^ (std::uint64_t(it->second.version) * 0xD6E8FEB86659FD93ull);
    if (!g_labelled.insert(id ^ (std::uint64_t(slot) * 0xC2B2AE3D27D4EB4Full)).second) return;  // (a range under a slot: once a frame)
    const std::uint8_t* data = it->second.data + r.offset;
    const std::uint64_t there = std::min(it->second.size - r.offset, kRangeBytes);
    read_held(data, r.size == UINT64_MAX ? there : std::min(r.size, there), slot);
    if (g_dump == 2) {
        const std::uint64_t whole = std::min(it->second.size - r.offset, kDumpRangeBytes);
        dump_read(data, r.size == UINT64_MAX ? std::min(whole, kRangeBytes) : std::min(r.size, whole), slot);  // (no size: a constant buffer's most, 64 KB)
    }
    if (!g_searching.load(std::memory_order_relaxed) || !in_window()) return;
    auto seen = g_read.find(id);
    if (seen == g_read.end()) {
        Read read{g_found.size(), 0};
        scan(data, std::min(r.size == UINT64_MAX ? (it->second.copy ? there : kUnsizedBytes) : r.size, there));  // (a copy's size is what was written)
        read.count = g_found.size() - read.first;
        seen = g_read.emplace(id, read).first;
    }
    label(seen->second, slot);
}
// (CameraDump=2) Direct3D 12's root buffer views (SetGraphicsRootShaderResourceView): where engines keep objects' and bones' transforms.
// Which form ReShade hands them in is not documented for root views: a view first (its description says), else a range whose buffer is checked
// to be a resource before anything is asked of it.
bool is_resource(std::uint64_t handle) {
    __try { return handle && reinterpret_cast<ID3D12Resource*>(handle)->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}
void dump_root_views(device* dev, pipeline_layout layout, uint32_t param, const descriptor_table_update& update) {
    std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
    if (!lock) return;
    g_dump_root_views += update.count;
    const std::uint32_t base = mix(mix(9, layout_of(layout)), param);
    for (uint32_t i = 0; i < update.count && i < 1; ++i) {  // (a root view is one)
        buffer_range r{};
        const resource_view view = static_cast<const resource_view*>(update.descriptors)[i];
        const resource_view_desc vd = view.handle ? dev->get_resource_view_desc(view) : resource_view_desc{};
        if (vd.type == resource_view_type::buffer) {
            r.buffer = dev->get_resource_from_view(view);
            r.offset = vd.buffer.offset;
            r.size = vd.format == format::unknown ? std::uint64_t(vd.buffer.structured.count) * vd.buffer.structured.stride : vd.buffer.size;
        } else {
            r = *static_cast<const buffer_range*>(update.descriptors);
            if (!is_resource(r.buffer.handle)) continue;
        }
        const Mapped m = mapped(dev, r.buffer);
        if (!m.data || r.offset >= m.size) continue;
        ++g_dump_root_read;
        const std::uint64_t whole = std::min(m.size - r.offset, kDumpRangeBytes);
        dump_read(m.data + r.offset, r.size == UINT64_MAX || r.size == 0 ? whole : std::min(r.size, whole), mix(base, i));
    }
}
void on_push_descriptors(command_list* cmd, shader_stage stage, pipeline_layout layout, uint32_t param, const descriptor_table_update& update) {
    if (g_paused.load(std::memory_order_relaxed)) return;
    if (cmd) g_pushed_kind[std::uint32_t(update.type) & 15].fetch_add(update.count, std::memory_order_relaxed);
    if (cmd && update.descriptors && g_dump == 2 && update.type == descriptor_type::buffer_shader_resource_view && cmd->get_device()->get_api() == device_api::d3d12)
        dump_root_views(cmd->get_device(), layout, param, update);
    if (!cmd || !holds_ranges(update.type) || !update.descriptors) return;
    device* dev = cmd->get_device();
    // (Direct3D 11: every constant buffer is bound this way, *SetConstantBuffers, by stage; what it holds is the copy kept at its last write)
    const bool d3d11 = dev->get_api() == device_api::d3d11, vulkan = dev->get_api() == device_api::vulkan;
    if (update.type != descriptor_type::constant_buffer && !vulkan) return;
    if (d3d11) {
        const auto* bound = static_cast<const buffer_range*>(update.descriptors);
        const std::uint32_t key_base = mix(mix(mix(6, layout_of(layout)), param), std::uint32_t(stage));
        std::lock_guard<std::mutex> lock(g_d3d11_mutex);
        for (uint32_t i = 0; i < update.count; ++i)
            if (bound[i].buffer.handle) { D3D11Buffer& d = g_d3d11[bound[i].buffer.handle]; d.key = mix(key_base, update.binding + i); d.keyed = true; d.bound = true; }
    }
    if ((!d3d11 && !vulkan && dev->get_api() != device_api::d3d12) || (g_dump != 2 && g_budget_spent.load(std::memory_order_relaxed) && !g_held_count.load(std::memory_order_relaxed))) return;
    Timed timed(kPush);
    const auto* ranges = static_cast<const buffer_range*>(update.descriptors);
    const std::uint32_t base = d3d11 ? mix(mix(mix(6, layout_of(layout)), param), std::uint32_t(stage)) : mix(mix(2, layout_of(layout)), param);
    bool any = false;
    for (uint32_t i = 0; i < update.count && !any; ++i) any = wanted(mix(base, update.binding + i));
    if (!any) return;
    std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
    if (!lock) return;
    for (uint32_t i = 0; i < update.count; ++i) {
        if (ranges[i].buffer.handle && !d3d11 && !vulkan) g_pushed_heap[mapped(dev, ranges[i].buffer).heap & 7].fetch_add(1, std::memory_order_relaxed);  // (bound now: it exists)
        if (ranges[i].size == UINT64_MAX) g_unsized.fetch_add(1, std::memory_order_relaxed);
        else if (ranges[i].size >= kRangeBytes) g_large.fetch_add(1, std::memory_order_relaxed);
        if (!wanted(mix(base, update.binding + i))) continue;
        if (d3d11 && ranges[i].buffer.handle && !g_searching.load(std::memory_order_relaxed)) g_held_buffers.insert(ranges[i].buffer.handle);
        read_range(ranges[i], mix(base, update.binding + i));
    }
}
// Direct3D 12: most constants are bound through tables. Which view lies where is noted as the game writes and copies descriptors; what a table holds is read as it is bound.
bool on_update_tables(device* dev, uint32_t count, const descriptor_table_update* updates) {
    if (g_paused.load(std::memory_order_relaxed)) return false;
    const bool vulkan = dev->get_api() == device_api::vulkan;
    if (dev->get_api() != device_api::d3d12 && !vulkan) return false;
    Timed timed(kWrite);
    for (uint32_t u = 0; u < count; ++u) {
        const descriptor_table_update& update = updates[u];
        descriptor_heap heap;
        uint32_t offset;
        if (!update.count || !place_of(dev, update.table, update.binding, update.array_offset, heap, offset)) continue;
        tally(g_written, update.count);
        tally(g_written_kind[std::uint32_t(update.type) & 15], update.count);
        // (Vulkan: storage buffers too - Enshrouded has 3 uniform buffers and writes ten thousand storage buffer descriptors a frame)
        // (CameraDump=2: Direct3D 12's buffer views - structured, raw - too, as ranges)
        const bool srvs = g_dump == 2 && !vulkan && (update.type == descriptor_type::shader_resource_view || update.type == descriptor_type::buffer_shader_resource_view);
        const bool views = update.type == descriptor_type::constant_buffer || (vulkan && holds_ranges(update.type)) || srvs;
        Heap* h = heap_of(heap.handle, views);
        if (!h) continue;
        const auto* ranges = static_cast<const buffer_range*>(update.descriptors);
        static thread_local std::uint64_t known = 0;  // (a ring of constants is one buffer: the lock once)
        for (uint32_t i = 0; i < update.count; ++i) {
            buffer_range* to = view_at(h, offset + i, views);
            if (!to) continue;
            if (!views) { to->buffer = {0}; continue; }
            buffer_range range{};
            if (srvs) {
                const resource_view view = static_cast<const resource_view*>(update.descriptors)[i];
                if (!view.handle) { to->buffer = {0}; continue; }
                const resource_view_desc vd = dev->get_resource_view_desc(view);
                if (vd.type != resource_view_type::buffer) { to->buffer = {0}; continue; }
                range.buffer = dev->get_resource_from_view(view);
                range.offset = vd.buffer.offset;
                range.size = vd.format == format::unknown ? std::uint64_t(vd.buffer.structured.count) * vd.buffer.structured.stride : vd.buffer.size;
                ++g_dump_srv_views;
            } else {
                range = ranges[i];
            }
            if (!vulkan && range.buffer.handle && range.buffer.handle != known) {
                std::lock_guard<std::mutex> lock(g_mutex);
                mapped(dev, range.buffer);
                known = range.buffer.handle;
            }
            *to = range;
        }
    }
    return false;
}
bool on_copy_tables(device* dev, uint32_t count, const descriptor_table_copy* copies) {
    if (g_paused.load(std::memory_order_relaxed)) return false;
    if (dev->get_api() != device_api::d3d12 && dev->get_api() != device_api::vulkan) return false;
    Timed timed(kCopy);
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
void on_bind_tables(command_list* cmd, shader_stage, pipeline_layout layout, uint32_t first, uint32_t count, const descriptor_table* tables, uint32_t dynamic_count, const uint32_t*) {
    if (g_paused.load(std::memory_order_relaxed) || !cmd || !tables || (g_dump != 2 && g_budget_spent.load(std::memory_order_relaxed) && !g_held_count.load(std::memory_order_relaxed))) return;
    device* dev = cmd->get_device();
    if (dev->get_api() != device_api::d3d12 && dev->get_api() != device_api::vulkan) return;
    // (Vulkan's dynamic offsets are not added: none in either game measured - counted, for the one that has them)
    if (dynamic_count) vk::g_dynamic.fetch_add(dynamic_count, std::memory_order_relaxed);
    Timed timed(kBind);
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
            // (bound before in this frame on this thread, the same range in the same slot - draw after draw: read once, without the lock. Every range this thread has
            // seen this frame, not only the last: with the last alone Cyberpunk's threads met at the lock 1700-2200 times a frame between searches, binds 2.3-3.5 ms)
            static thread_local Once seen_here(std::size_t(1) << 10);
            static thread_local std::uint32_t seen_frame = 0;
            const std::uint64_t seen = (r.buffer.handle ^ (r.offset * 0x9E3779B97F4A7C15ull)) ^ (std::uint64_t(mix(base, i)) * 0xC2B2AE3D27D4EB4Full);
            const std::uint32_t frame = g_frame.load(std::memory_order_relaxed);
            if (frame != seen_frame) { seen_here.clear(); seen_frame = frame; }
            if (!seen_here.insert(seen).second) continue;
            // (never waited for: a render thread that finds it taken goes on, the camera's constants are bound a hundred times a frame)
            std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
            if (!lock) { tally(g_skipped, 1); return; }
            read_range(r, mix(base, i));
        }
    }
}
// Direct3D 11: constants are written through a map or an update: a copy kept as the game finishes, read where it is bound (the key needs the slot: by the buffer's size
// alone RE2's camera was off 15-131 times the motion).
// (under the lock) What a constant buffer was filled with, from offset on, kept for its binds.
void keep(resource res, const std::uint8_t* data, std::uint64_t offset, std::uint64_t size) {
    if (offset >= kRangeBytes) return;
    // (refilled with no bind since the last write: read here, under the slot it was last bound to - or the buffer's own, never bound)
    bool read_here = false;
    std::uint32_t key = 0;
    {
        std::lock_guard<std::mutex> lock(g_d3d11_mutex);
        D3D11Buffer& d = g_d3d11[res.handle];
        read_here = d.written && !d.bound;
        d.written = true; d.bound = false;
        key = d.keyed ? d.key : mix(mix(8, std::uint32_t(res.handle)), std::uint32_t(res.handle >> 32));
    }
    read_here = read_here && wanted(key);
    if (read_here && !g_searching.load(std::memory_order_relaxed)) g_held_buffers.insert(res.handle);
    if (!g_searching.load(std::memory_order_relaxed) && g_held_count.load(std::memory_order_relaxed) && !g_held_buffers.count(res.handle)) {
        const auto it = g_mapped.find(res.handle);
        if (it != g_mapped.end() && it->second.data) { it->second.data = nullptr; ++it->second.version; }
        return;
    }
    size = std::min(size, kRangeBytes - offset);
    std::vector<std::uint8_t>& copy = g_copies[res.handle];
    if (copy.size() < offset + size) copy.resize(offset + size);
    wc_copy(copy.data() + offset, data, size);
    g_copied_bytes += size;
    Mapped& m = g_mapped[res.handle];
    m.data = copy.data();
    m.size = copy.size();
    m.copy = true;
    ++m.version;
    if (read_here) read_now(copy.data(), copy.size(), key);
}
void on_map(device* dev, resource res, uint64_t, uint64_t size, map_access access, void** data) {
    if (g_paused.load(std::memory_order_relaxed) || dev->get_api() != device_api::d3d11 || !data || !*data || access == map_access::read_only) return;
    const resource_desc desc = dev->get_resource_desc(res);
    if (desc.type != resource_type::buffer || (desc.usage & resource_usage::constant_buffer) == 0) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_writing[res.handle] = {static_cast<std::uint8_t*>(*data), size == UINT64_MAX ? desc.buffer.size : size};
}
void on_unmap(device* dev, resource res) {
    if (g_paused.load(std::memory_order_relaxed) || dev->get_api() != device_api::d3d11) return;
    Timed timed(kD3D11);
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_writing.find(res.handle);
    if (it == g_writing.end()) return;
    keep(res, it->second.first, 0, it->second.second);
    g_writing.erase(it);
}
bool on_update_buffer(device* dev, const void* data, resource dest, uint64_t offset, uint64_t size) {
    if (g_paused.load(std::memory_order_relaxed) || dev->get_api() != device_api::d3d11 || !data) return false;
    const resource_desc desc = dev->get_resource_desc(dest);
    if (desc.type != resource_type::buffer || (desc.usage & resource_usage::constant_buffer) == 0) return false;
    Timed timed(kD3D11);
    std::lock_guard<std::mutex> lock(g_mutex);
    keep(dest, static_cast<const std::uint8_t*>(data), offset, size == UINT64_MAX ? desc.buffer.size - offset : size);
    return false;
}
void on_destroy(device* dev, resource res) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_mapped.find(res.handle);
    if (it == g_mapped.end()) return;
    if (it->second.data && !it->second.copy && !it->second.game) dev->unmap_buffer_region(res);
    if (it->second.game) vk::g_buffers.erase(res.handle);  // (its memory's list keeps the handle: refresh checks the memory still matches)
    it->second = Mapped{};  // (kept: views to it are still listed)
    g_copies.erase(res.handle);
    std::lock_guard<std::mutex> d3d11_lock(g_d3d11_mutex);
    g_d3d11.erase(res.handle);
}
}  // namespace

std::uint32_t take(MatrixCandidate* out, std::uint32_t capacity, const volatile long* held, long held_state, std::uint64_t frame) {
    if (!g_on) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    // (the places held first: the rest is cut at the capacity)
    const std::uint32_t h = std::uint32_t(std::min<std::size_t>(g_held_found.size(), capacity));
    std::copy(g_held_found.begin(), g_held_found.begin() + h, out);
    // (CameraDump=2: every value found this frame to logs/camera_values.bin - 'XPAV', count, frame, the values - once a place is held)
    if (g_dump == 2 && frame) {
        if (!g_values_started && g_held_count.load(std::memory_order_relaxed)) {
            g_values_started = true;
            // (beside the add-on, else beside the game's exe; the path and the error said - the first try could not open it, unexplained)
            std::string said;
            for (int where = 0; where < 2 && !g_values_file; ++where) {
                HMODULE self = nullptr;
                wchar_t path[MAX_PATH]{};
                if (where == 0 && !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&take), &self)) continue;
                if (!GetModuleFileNameW(self, path, MAX_PATH)) continue;
                std::wstring file = path;
                file = file.substr(0, file.find_last_of(L"\\/")) + L"\\FrameWarp\\logs\\camera_values.bin";
                g_values_file = _wfsopen(file.c_str(), L"wb", _SH_DENYNO);
                const int error = errno;
                char one[700];
                std::snprintf(one, sizeof(one), " %ls (%s)", file.c_str(), g_values_file ? "opened" : std::strerror(error));
                said += one;
            }
            fw::diag_log(reshade::log::level::info, (std::string("XPAR: camera search: CameraDump=2, every value found to") + said).c_str());
        }
        if (g_values_file) {
            const std::uint32_t head[2] = {0x56415058u, std::uint32_t(g_dump_found.size())};
            std::fwrite(head, sizeof(head), 1, g_values_file);
            std::fwrite(&frame, sizeof(frame), 1, g_values_file);
            std::fwrite(g_dump_found.data(), sizeof(MatrixCandidate), g_dump_found.size(), g_values_file);
            g_values_bytes += 16 + std::uint64_t(head[1]) * sizeof(MatrixCandidate);
            if (g_values_bytes > kDumpFileBytes) {
                std::fclose(g_values_file);
                g_values_file = nullptr;
                fw::diag_log(reshade::log::level::info, "XPAR: camera search: logs/camera_values.bin written (3 GB)");
            }
        }
        static std::uint32_t dump_frames = 0;
        if (++dump_frames % 300 == 0) {
            char text[300];
            std::size_t objects = 0;
            for (const MatrixCandidate& c : g_dump_found) objects += (c.flags & 4) ? 1 : 0;
            std::snprintf(text, sizeof(text), "XPAR: camera search, CameraDump=2: this frame %llu KB in %llu ranges, %zu values, %zu of them object matrices; buffer views so far: %llu in tables, %llu root (%llu read)",
                          static_cast<unsigned long long>(g_dump_bytes / 1024), static_cast<unsigned long long>(g_dump_ranges), g_dump_found.size(), objects,
                          static_cast<unsigned long long>(g_dump_srv_views), static_cast<unsigned long long>(g_dump_root_views), static_cast<unsigned long long>(g_dump_root_read));
            fw::diag_log(reshade::log::level::info, text);
        }
    }
    // (CameraDump=2: the objects' after the camera's values, in at most half the room left)
    const std::size_t room = capacity - h, objects_least = std::min(g_raw_objects.size(), room / 2);
    const std::uint32_t n = h + std::uint32_t(std::min(g_raw.size(), room - objects_least));
    std::copy(g_raw.begin(), g_raw.begin() + (n - h), out + h);
    const std::uint32_t objects = std::uint32_t(std::min(g_raw_objects.size(), std::size_t(capacity - n)));
    std::copy(g_raw_objects.begin(), g_raw_objects.begin() + objects, out + n);
    if (g_budget_spent) { g_spent_frames.fetch_add(1, std::memory_order_relaxed); for (std::uint32_t i = 0; i < n + objects; ++i) out[i].flags |= 8; }  // (for the dump: this frame was cut short)
    if (fw::diag() && g_said < 60 && ++g_rotate % 300 == 150) {  // (the hooks' times are taken only while it is written - Timed)
        ++g_said;
        char text[1400];
        const double ms = 1000.0 * double(g_ticks.load()) / double(g_frequency.QuadPart) / 300.0;
        std::snprintf(text, sizeof(text), "XPAR: camera search: this frame %llu KB in %llu ranges, %zu matrices, %zu under their slots, %zu from %u places held; a frame: %.2f ms in the hooks (all threads), %llu descriptors written, %llu copied, %llu tables bound, %llu skipped",
                      static_cast<unsigned long long>(g_bytes / 1024), static_cast<unsigned long long>(g_ranges), g_found.size(), g_raw.size(), g_held_found.size(), g_held_count.load(), ms, static_cast<unsigned long long>(g_written.load() / 300),
                      static_cast<unsigned long long>(g_copied.load() / 300), static_cast<unsigned long long>(g_bound.load() / 300), static_cast<unsigned long long>(g_skipped.load() / 300));
        fw::diag_log(reshade::log::level::info, text);
        {
            const double all = double(g_ticks.load()), searching = double(g_ticks_searching.exchange(0)), tick_ms = 1000.0 / double(g_frequency.QuadPart);
            const std::uint64_t searched = g_search_frames.exchange(0);
            std::snprintf(text, sizeof(text), "XPAR: camera search: frames searched whole %llu of 300, the hooks %.2f ms on each of them, %.2f ms on each of the others; searched from read %u of %u; constants kept (Direct3D 11) %llu KB a frame",
                          static_cast<unsigned long long>(searched), searched ? searching * tick_ms / double(searched) : 0.0, searched < 300 ? (all - searching) * tick_ms / double(300 - searched) : 0.0, g_window, g_reads_last,
                          static_cast<unsigned long long>(g_copied_bytes / 300 / 1024));
            g_copied_bytes = 0;
            fw::diag_log(reshade::log::level::info, text);
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
        fw::diag_log(reshade::log::level::info, text);
        if (vk::g_device) {
            std::size_t mapped_now = 0, buffers_readable = 0;
            for (const auto& [handle, m] : vk::g_memory) mapped_now += m.data ? 1 : 0;
            for (const auto& [handle, b] : vk::g_buffers) { const auto at = g_mapped.find(handle); buffers_readable += at != g_mapped.end() && at->second.data ? 1 : 0; }
            std::snprintf(text, sizeof(text), "XPAR: camera search (Vulkan): hooks%s; memory mapped now %zu of %zu known, buffers bound %zu, readable %zu; since the last line: maps %llu, unmaps %llu, frees %llu, binds %llu, dynamic offsets %llu",
                          vk::g_hooked.c_str(), mapped_now, vk::g_memory.size(), vk::g_buffers.size(), buffers_readable, static_cast<unsigned long long>(vk::g_maps.exchange(0)),
                          static_cast<unsigned long long>(vk::g_unmaps.exchange(0)), static_cast<unsigned long long>(vk::g_frees.exchange(0)), static_cast<unsigned long long>(vk::g_binds.exchange(0)),
                          static_cast<unsigned long long>(vk::g_dynamic.exchange(0)));
            fw::diag_log(reshade::log::level::info, text);
        }
        {
            std::string parts;
            const double tick_ms = 1000.0 / double(g_frequency.QuadPart);
            const std::uint64_t searched = std::max<std::uint64_t>(g_search_frames_parts.exchange(0), 1);
            for (int p = 0; p < kParts; ++p) {
                if (p == kPlace) continue;
                char one[160];
                const std::uint64_t calls = g_part_calls[p].exchange(0);
                std::snprintf(one, sizeof(one), "%s %s %.2f ms (searched whole %.2f)%s", p ? ";" : "", kPartNames[p], double(g_part_ticks[0][p].exchange(0)) * tick_ms / 300.0,
                              double(g_part_ticks[1][p].exchange(0)) * tick_ms / double(searched), "");
                parts += one;
                if (p < kPlace) { std::snprintf(one, sizeof(one), " %llu calls", static_cast<unsigned long long>(calls / 300)); parts += one; }
            }
            int trusted = 0, refused = 0, bases = 0;
            for (int i = 0; i < g_known_count.load(); ++i)
                for (int k = 0; k < g_known[i].based.load(); ++k) { const int st = g_known[i].bases[k].state.load(); ++bases; trusted += st == 1; refused += st == -1; }
            std::snprintf(text, sizeof(text), "XPAR: camera search: where the hooks' time goes, a frame (all threads):%s; Direct3D 12 table bases placed by us %d, left to ReShade %d (of %d, in %d heaps)", parts.c_str(), trusted, refused, bases, g_known_count.load());
            fw::diag_log(reshade::log::level::info, text);
        }
        g_ticks = 0; g_written = 0; g_copied = 0; g_bound = 0; g_skipped = 0;
    }
    g_raw.clear();
    g_raw_objects.clear();
    g_found_objects = 0;
    g_dump_found.clear(); g_dump_seen.clear(); g_dump_bytes = g_dump_ranges = 0;
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
    if (count != had || !std::equal(now, now + count, before)) { g_held_slots = 0; g_held_mask = 0; g_held_buffers.clear(); }  // (other places, not the same in another order: their slots are found again)
    static std::uint32_t frames = 0;
    static std::uint32_t searched = 0;
    if (g_searching && ++searched % 2 == 0) g_window = g_full_at;  // (0 when the frame fitted: round to the start)
    g_reads_last = g_read_index;
    g_read_index = g_full_at = 0;
    // (nothing read from the places held - their slots are found again only by a search, and with nothing sent the presenter judges nothing: its last word stays)
    g_searching = g_dump == 2 || !count || held_state == 2 || !held_read || (held_state == 0 && ++frames % kSearchEvery < kSearchFrames);
    if (g_searching) { g_search_frames.fetch_add(1, std::memory_order_relaxed); g_search_frames_parts.fetch_add(1, std::memory_order_relaxed); }
    g_found.clear();
    g_read.clear();
    g_labelled.clear();
    g_bytes = g_ranges = 0;
    g_budget_spent = false;
    return n + objects;
}

void pause() {
    if (!g_on || g_paused.exchange(true)) return;
    // (what a map began is forgotten: an unmap after resume() of a map made while paused finds nothing to copy, never a pointer
    // from before - its buffer may be mapped elsewhere by then)
    std::lock_guard<std::mutex> lock(g_mutex);
    g_writing.clear();
}
void resume() { g_paused.store(false, std::memory_order_relaxed); }
void init() {
    char value[8] = {};
    size_t size = sizeof(value);
    // (on by default since 2.2; CameraSearch=0 turns it off)
    g_on = !(reshade::get_config_value(nullptr, "FrameWarp", "CameraSearch", value, &size) && value[0] == '0');
    if (!g_on) return;
    size = sizeof(value);
    value[0] = 0;
    if (reshade::get_config_value(nullptr, "FrameWarp", "CameraDump", value, &size)) g_dump = value[0] - '0';
    QueryPerformanceFrequency(&g_frequency);
    {
        LARGE_INTEGER a, b;
        std::int64_t least = INT64_MAX;
        for (int i = 0; i < 1000; ++i) { QueryPerformanceCounter(&a); QueryPerformanceCounter(&b); least = std::min<std::int64_t>(least, b.QuadPart - a.QuadPart); }
        g_clock_cost = least;
    }
    reshade::register_event<reshade::addon_event::init_device>(vk::on_init_device);
    reshade::register_event<reshade::addon_event::init_swapchain>(vk::on_init_swapchain);
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
    reshade::unregister_event<reshade::addon_event::init_swapchain>(vk::on_init_swapchain);
    reshade::unregister_event<reshade::addon_event::init_device>(vk::on_init_device);
}
}  // namespace fw::camera_search
