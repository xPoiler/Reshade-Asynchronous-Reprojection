#include "addon/producer.hpp"
#include "addon/vk_transport.hpp"
#include <reshade_api.hpp>
#include "shared/camera_motion.hpp"
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

namespace fw {
static DXGI_FORMAT typed_format(DXGI_FORMAT f);
thread_local bool tl_tagging = false;
thread_local std::uint64_t tl_tag_list = 0;
using Microsoft::WRL::ComPtr;

namespace {
// Typed depth formats cannot be created without ALLOW_DEPTH_STENCIL; copy into their typeless family.
DXGI_FORMAT copy_format(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_TYPELESS;
        case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24G8_TYPELESS;
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32G8X24_TYPELESS;
        case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_TYPELESS;
        default: return f;
    }
}
}  // namespace

// Every setting at its default (the add-on starting, the panel's "Reset all settings to defaults"; the remembered
// ones read from ReShade.ini afterwards).
void default_settings(Settings& s) {
    s = Settings{};
    s.enabled = 1; s.use_mouse = 1; s.use_ui_tags = 1;
    s.warp_engine = 1;  // XPAR's own engine (NVIDIA Latewarp is the alternative)
    s.rotation_extrapolation = 1.0f; s.translation_extrapolation = 1.0f;
    s.orbit_distance = 0.0f; s.max_horizon_ms = 100.0f;
    s.prediction_ms = -16.0f;  // manual value, used when auto is off
    s.present_lead_ms = 6.0f;  // manual value, used when the automatic lead is off
    s.present_lead_auto = 1;
    s.frame_cap = 1;  // prevent GPU queueing, standing aside while the game's Reflex is on
    s.orbit_mode = 1;  // off (2026-10-22, user): the measured pivot wanders when the camera does more than orbit - Enshrouded's and Dawnwalker's character juddered
    s.floor_release = 1;  // the ground around the character moves with a measured orbit (acts only while one is in use)
    s.hold_reticles = 1;  // sight reticles drawn without depth move with the gun (acts only inside a sight's window)
    s.game_hud_layers = 1;  // the HUD from the game's own HUD layer / final picture where found
    s.warp_splat = 1;  // the XPAR warp by forward splat
    s.hold_characters = 1;  // whole characters held
    s.hud_learn = 1;  // learn where the game draws its HUD
    s.wait_for_model = 0;  // (off, no option since 2026-10-22, user: more problems than it solved - see Settings::wait_for_model)
    s.follow_share = 1;  // the character/weapon by the share of the turn it follows
    s.share_from_input = 1;  // ...of the turn the input explains
    s.edge_limit_pct = 0.0f;  // camera as of now: no more of the edge uncovered than Auto's
    s.edge_fill = 2;  // dithered (the frame as it is past the edge, a dithered band inside it - user, 2026-10-21); soft and extended edge pixels are the options
    s.no_warp_mask = 1;   // HUD detection in games without HUD layers
    s.keep_attached = 1;  // character / first-person weapon detection, every game
    s.near_camera_rule = 1;
    s.hud_fill = 1;  // fill behind the HUD from the upscaler's output (XPAR engine)
    s.turn_rule = 1;
    s.stretch_width = 1;
    s.background_memory = 1;  // what the warp uncovers comes from the scenery last seen there (XPAR engine)  // hold what stays nearly still on screen while the camera turns (orbit cameras)
    s.memory_resolution = 1;  // (the background memory at half the picture's resolution: full fills seamlessly, ~4x the video memory)
    s.turns_only = 0;  // (the camera's turn and its movement both reprojected)
    s.hud_from_scene = 2;  // HUD from the upscaler's output + camera-motion check (learned HUD without an upscaler output)
    s.auto_prediction = 6;     // default: the lowest latency without edge fill (now .. one game frame back); 3: Auto - 1/2 game frame, 1/4 in games
                               // without HUD layers (shorter warps hide mask misses), 1 without DLSS/FSR (XPAR's own motion); 1: full, 2: half, 4: quarter
    s.manual_gain_x = s.manual_gain_y = 0.0f; s.manual_delay_ms = 0.0f;
    s.use_controller = 1;  // the right stick drives rotation too
}

Producer::Producer() {
    const DWORD pid = GetCurrentProcessId();
    marker_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    mapping_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(Shared), map_name(pid).c_str());
    if (!mapping_) return;
    shared_ = static_cast<Shared*>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Shared)));
    if (!shared_) return;
    std::memset(shared_, 0, sizeof(Shared));
    shared_->magic = kMagic;
    shared_->version = kVersion;
    shared_->producer_pid = pid;
    shared_->session = static_cast<std::uint64_t>(qpc_now());
    LARGE_INTEGER f; QueryPerformanceFrequency(&f); shared_->qpc_frequency = f.QuadPart;
    shared_->hooks.constants_base = shared_->hooks.tag_base = -1;
    for (auto& r : shared_->hooks.feature_result) r = -1;
    shared_->hooks.pcl_lookup_result = shared_->hooks.reflex_lookup_result = -1;
    default_settings(shared_->settings);
}

Producer::~Producer() {
    stop_markers_ = true;  // (the marker thread ends by itself; at process exit it is gone already)
    SetEvent(marker_event_);
    detach_vulkan();
    for (auto& slot : textures_)
        for (auto& t : slot) if (t.handle) CloseHandle(t.handle);
    if (ui_vk_.handle) CloseHandle(ui_vk_.handle);
    if (hudless_vk_.handle) CloseHandle(hudless_vk_.handle);
    for (auto& r : retired_) if (r.handle) CloseHandle(r.handle);
    if (fence_handle_) CloseHandle(fence_handle_);
    if (shared_) { shared_->magic = 0; UnmapViewOfFile(shared_); }
    if (mapping_) CloseHandle(mapping_);
}

void Producer::set_message(const char* text) {
    if (!shared_) return;
    std::snprintf(shared_->hooks.message, sizeof(shared_->hooks.message), "%s", text);
}

bool Producer::attach(ID3D12Device* device) {
    std::lock_guard lock(mutex_);
    if (!shared_ || !device) return false;
    if (device_.Get() == device) return true;
    if (device_) {
        // The game recreated its device: drop everything from the old one and start a new session
        // so the presenter reopens all shared objects.
        release_textures();
        if (fence_handle_) CloseHandle(fence_handle_);
        fence_handle_ = nullptr; fence_.Reset(); device_.Reset(); fence_value_ = 0;
        for (auto& m : shared_->slots) { m.state = kFree; m.frame_id = 0; }
        shared_->latest_ready_frame = 0;
        shared_->session = static_cast<std::uint64_t>(qpc_now());
    }
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_)))) return false;
    const auto name = object_name(shared_->producer_pid, shared_->session, L"fence");
    if (FAILED(device->CreateSharedHandle(fence_.Get(), nullptr, GENERIC_ALL, name.c_str(), &fence_handle_))) { fence_.Reset(); return false; }
    device_ = device;
    shared_->adapter = device->GetAdapterLuid();
    return true;
}

void Producer::set_swapchain(HWND hwnd, std::uint32_t width, std::uint32_t height, DXGI_FORMAT format, std::uint32_t color_space) {
    if (!shared_) return;
    shared_->game_hwnd = reinterpret_cast<std::uint64_t>(hwnd);
    shared_->backbuffer_width = width; shared_->backbuffer_height = height;
    shared_->backbuffer_format = format; shared_->color_space = color_space;
}

int Producer::slot_for_frame(std::uint64_t frame, bool create) {
    for (int i = 0; i < kSlots; ++i)
        if (shared_->slots[i].state == kWriting && shared_->slots[i].frame_id == frame) return i;
    if (!create) return -1;
    auto claim = [&](int i, LONG from) {
        if (InterlockedCompareExchange(&shared_->slots[i].state, kWriting, from) != from) return false;
        auto& m = shared_->slots[i];
        m.frame_id = frame; m.fence_value = 0; m.qpc_sim_start = 0; m.qpc_constants = 0; m.qpc_present = 0;
        slot_list_[i] = 0; slot_submitted_[i] = false; ui_tagged_[i] = false;
        m.camera.valid = 0;
        for (auto& t : m.tex) t.valid = 0;
        m.generated = m.per_frame = m.gen_last = 0;
        m.hud_candidate = 0; m.hud_learned = 0;
        return true;
    };
    for (int i = 0; i < kSlots; ++i) if (shared_->slots[i].state == kFree && claim(i, kFree)) return i;
    // Reuse the oldest abandoned (never presented) or unread ready frame; never one being read.
    int oldest = -1;
    for (int i = 0; i < kSlots; ++i) {
        const auto st = shared_->slots[i].state;
        if (st != kWriting && st != kReady) continue;
        if (st == kReady && shared_->slots[i].frame_id == static_cast<std::uint64_t>(shared_->latest_ready_frame)) continue;
        if (oldest < 0 || shared_->slots[i].frame_id < shared_->slots[oldest].frame_id) oldest = i;
    }
    if (oldest >= 0 && shared_->slots[oldest].frame_id < frame) {
        // Writing slots belong to us alone; Ready ones may race with the presenter's claim.
        const LONG st = shared_->slots[oldest].state;
        if (st == kWriting) InterlockedExchange(&shared_->slots[oldest].state, kFree);
        if (claim(oldest, st == kWriting ? kFree : kReady)) {
            ++shared_->hooks.slots_dropped;
            return oldest;
        }
    }
    ++shared_->hooks.slots_dropped;
    return -1;
}

void Producer::collect_retired() {
    if (!fence_) return;
    const auto done = fence_->GetCompletedValue();
    for (auto it = retired_.begin(); it != retired_.end();) {
        if (it->after_fence <= done) {
            if (vk_) vk_->destroy(it->vk);
            if (it->handle) CloseHandle(it->handle);
            it = retired_.erase(it);
        } else ++it;
    }
}

void Producer::release_textures() {
    for (auto& slot : textures_)
        for (auto& t : slot) {
            if (vk_) vk_->destroy(t.vk);
            if (t.handle) CloseHandle(t.handle);
            t = Texture{};
        }
    if (vk_) vk_->destroy(ui_vk_.vk);
    if (ui_vk_.handle) CloseHandle(ui_vk_.handle);
    ui_vk_ = Texture{};
    ui_vk_age_ = 2;
    if (vk_) vk_->destroy(hudless_vk_.vk);
    if (hudless_vk_.handle) CloseHandle(hudless_vk_.handle);
    hudless_vk_ = Texture{};
    hudless_vk_fresh_ = false;
    d11_before_.Reset(); d11_cover_.Reset(); d11_picture_.Reset(); d11_cover_cs_.Reset();
    d11_before_taken_ = d11_cs_failed_ = d11_none_ = d11_confirmed_ = false;
    for (auto& r : retired_) {
        if (vk_) vk_->destroy(r.vk);
        if (r.handle) CloseHandle(r.handle);
    }
    retired_.clear();
}

bool Producer::ensure_texture(int slot, Tex kind, const D3D12_RESOURCE_DESC& source, std::uint32_t vk_depth_bytes, bool shader_written) {
    auto& t = textures_[slot][kind];
    const DXGI_FORMAT format = copy_format(source.Format);
    if (t.resource && t.desc.Width == source.Width && t.desc.Height == source.Height && t.desc.Format == format &&
        (!vk_ || (t.vk.image && t.vk.depth_bytes == vk_depth_bytes)) && (!d11_ || t.d11)) return true;
    if (t.resource) retired_.push_back({t.resource, t.handle, fence_value_ + 1, t.vk, t.d11});
    const std::uint32_t generation = t.generation + 1;
    t = Texture{};
    t.generation = generation;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = source.Width; desc.Height = source.Height; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = format; desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    // D3D11 opens a texture shared by D3D12 only if it could be a render target (both vendors' drivers
    // refuse the rest; R32F needs simultaneous access as well): what D3D11 games copy is colour or R32F.
    if (d11_) desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    // (Direct3D 12: the HUD layer's slot written by the add-on's pass itself - see begin_present - one 4K copy less a frame.
    // Refused, the plain shared texture below: the pass writes its own texture and copies it, as before. Only for that
    // pass: a slot the other HUD steps copy into came out black in the presenter with the flag - the pipeline test's)
    if (shader_written && !d11_ && !vk_) desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    HRESULT hr = device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t.resource));
    if (FAILED(hr)) {  // depth families may refuse simultaneous access; retry as a plain shared texture
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        hr = device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t.resource));
    }
    if (FAILED(hr)) {
        char text[160]; std::snprintf(text, sizeof(text), "create %s %ux%u fmt %d failed 0x%08lX", tex_name(kind),
                                      static_cast<unsigned>(source.Width), source.Height, format, static_cast<unsigned long>(hr));
        set_message(text); t.resource.Reset(); return false;
    }
    const auto name = object_name(shared_->producer_pid, shared_->session, L"tex", slot, kind, t.generation);
    if (FAILED(device_->CreateSharedHandle(t.resource.Get(), nullptr, GENERIC_ALL, name.c_str(), &t.handle))) {
        set_message("CreateSharedHandle failed"); t.resource.Reset(); return false;
    }
    t.desc = desc;
    if (vk_) {
        std::string error;
        if (!vk_->import(t.handle, vk_format_of(format), static_cast<std::uint32_t>(desc.Width), desc.Height, vk_depth_bytes, t.vk, error)) {
            set_message(error.c_str());
            CloseHandle(t.handle); t.handle = nullptr; t.resource.Reset();
            return false;
        }
    }
    if (d11_ && FAILED(d11_->OpenSharedResource1(t.handle, IID_PPV_ARGS(&t.d11)))) {
        char text[160]; std::snprintf(text, sizeof(text), "D3D11: opening %s %ux%u fmt %d failed", tex_name(kind),
                                      static_cast<unsigned>(desc.Width), desc.Height, format);
        set_message(text);
        CloseHandle(t.handle); t.handle = nullptr; t.resource.Reset();
        return false;
    }
    return true;
}

void Producer::copy_into(ID3D12GraphicsCommandList* list, ID3D12Resource* source, D3D12_RESOURCE_STATES state, ID3D12Resource* target) {
    D3D12_RESOURCE_BARRIER b[2]{};
    UINT count = 0;
    if (state != D3D12_RESOURCE_STATE_COPY_SOURCE) {
        b[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[count].Transition = {source, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state, D3D12_RESOURCE_STATE_COPY_SOURCE};
        ++count;
    }
    b[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b[count].Transition = {target, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST};
    ++count;
    list->ResourceBarrier(count, b);
    D3D12_TEXTURE_COPY_LOCATION dst{target, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION src{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    dst.SubresourceIndex = 0; src.SubresourceIndex = 0;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    for (UINT i = 0; i < count; ++i) std::swap(b[i].Transition.StateBefore, b[i].Transition.StateAfter);
    list->ResourceBarrier(count, b);
}

int Producer::time_begin(ID3D12GraphicsCommandList* list, int tag) {
    if (!device_ || !list || ts_failed_) return -1;
    if (!ts_heap_) {
        D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, kTsPairs * 2, 0};
        D3D12_HEAP_PROPERTIES rb{}; rb.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = kTsPairs * 16; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
        bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        void* mapped = nullptr;
        if (FAILED(device_->CreateQueryHeap(&qd, IID_PPV_ARGS(&ts_heap_))) ||
            FAILED(device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&ts_readback_))) ||
            FAILED(ts_readback_->Map(0, nullptr, &mapped))) {
            ts_heap_.Reset(); ts_readback_.Reset(); ts_failed_ = true;
            return -1;
        }
        ts_mapped_ = static_cast<std::uint64_t*>(mapped);
        for (auto& g : ts_tag_) g = -1;
    }
    const unsigned pair = ts_next_++ % kTsPairs;
    if (ts_tag_[pair] >= 0) {
        const std::uint64_t a = ts_mapped_[pair * 2], b = ts_mapped_[pair * 2 + 1];
        if (b > a && b - a < (1ull << 32)) ts_ticks_[ts_tag_[pair]] += b - a;
    }
    ts_tag_[pair] = tag;
    list->EndQuery(ts_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, pair * 2);
    return int(pair);
}

void Producer::time_end(ID3D12GraphicsCommandList* list, int pair) {
    if (pair < 0 || !ts_heap_) return;
    list->EndQuery(ts_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, UINT(pair) * 2 + 1);
    list->ResolveQueryData(ts_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, UINT(pair) * 2, 2, ts_readback_.Get(), UINT64(pair) * 16);
}

void Producer::time_report(ID3D12CommandQueue* queue) {
    if (!ts_heap_ || ++ts_presents_ < 600) return;
    UINT64 frequency = 0;
    if (FAILED(queue->GetTimestampFrequency(&frequency)) || !frequency) return;
    static const char* const kNames[kTCount] = {"picture", "depth", "motion vectors", "other tags", "HUD layer taken", "HUD layer pass",
                                                "HUD layer to slot", "HUD-less taken", "HUD-less to slot", "HUD learning"};
    char text[512];
    int n = std::snprintf(text, sizeof(text), "XPAR: the add-on's GPU work in the game's frames, ms per present (%u):", ts_presents_);
    double total = 0;
    for (int k = 0; k < kTCount; ++k) {
        const double ms = double(ts_ticks_[k]) * 1000.0 / double(frequency) / double(ts_presents_);
        total += ms;
        if (ms > 0.0005 && n < int(sizeof(text)) - 40) n += std::snprintf(text + n, sizeof(text) - n, " %s %.3f,", kNames[k], ms);
        ts_ticks_[k] = 0;
    }
    if (n < int(sizeof(text)) - 20) std::snprintf(text + n, sizeof(text) - n, " total %.3f", total);
    if (log_line) log_line(text);
    ts_presents_ = 0;
}

void Producer::on_constants(std::uint64_t frame, const Camera& camera) {
    std::lock_guard lock(mutex_);
    note_frame(frame);
    if (!ready()) return;
    // Streamline's cameraPos is camera-relative (always 0 in UE), but clipToPrevClip carries the real
    // frame-to-frame motion. Integrate it into an absolute position on every frame (even frames that
    // never reach the presenter) so translation - e.g. a third-person orbit - is known.
    Camera cam = camera;
    if (!camera.estimated) {
        // The previous frame's zoom and near plane (they change while aiming in some games); the jitter
        // terms stay the current frame's, so games whose projection only jitters behave exactly as before.
        float previous_projection[16];
        std::memcpy(previous_projection, have_prev_camera_ ? prev_view_to_clip_ : camera.view_to_clip, sizeof(previous_projection));
        previous_projection[8] = camera.view_to_clip[8];
        previous_projection[9] = camera.view_to_clip[9];
        const FrameMotion motion = recover_motion(camera.view_to_clip, camera.clip_to_prev_clip, previous_projection);
        double t[3] = {motion.translation[0], motion.translation[1], motion.translation[2]};
        const double step = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
        if (have_prev_camera_ && frame == prev_frame_ + 1 && !camera.reset && motion.valid && step < 1000.0) {
            // t is the current camera position in the previous frame's view space: x right, y up, z forward
            // (Unreal) or backward (right-handed engines such as RE Engine, where clip.w = -z).
            const double z_sign = camera.view_to_clip[11] < 0.0f ? -1.0 : 1.0;
            for (int i = 0; i < 3; ++i)
                world_pos_[i] += t[0] * prev_right_[i] + t[1] * prev_up_[i] + z_sign * t[2] * prev_fwd_[i];
        } else {
            ++position_epoch_;  // discontinuity (cut, reset, dropped frame): the presenter restarts its history
        }
        for (int i = 0; i < 3; ++i) {
            prev_right_[i] = camera.right[i]; prev_up_[i] = camera.up[i]; prev_fwd_[i] = camera.fwd[i];
            cam.pos[i] = static_cast<float>(world_pos_[i]);
        }
        std::memcpy(prev_view_to_clip_, camera.view_to_clip, sizeof(prev_view_to_clip_));
        cam.position_epoch = position_epoch_;
        prev_frame_ = frame;
        have_prev_camera_ = true;
    }
    if (frame != last_counted_frame_) { last_counted_frame_ = frame; InterlockedIncrement(&shared_->frames_total); }
    const int slot = slot_for_frame(frame, true);
    if (slot < 0) return;
    auto& m = shared_->slots[slot];
    m.camera = cam;
    m.camera.valid = 1;
    m.qpc_constants = qpc_now();
    push_event(shared_, kEvConstants, frame, static_cast<std::uint64_t>(slot));
    for (int i = 0; i < 64; ++i)
        if (sim_start_frame_[i] == frame) m.qpc_sim_start = sim_start_qpc_[i];
}

void Producer::on_sim_start(std::uint64_t frame, std::int64_t qpc) {
    std::lock_guard lock(mutex_);
    note_frame(frame);
    sim_start_frame_[frame % 64] = frame;
    sim_start_qpc_[frame % 64] = qpc;
    push_event(shared_, kEvSimStart, frame);
}

void Producer::on_execute(std::uint64_t list) {
    std::lock_guard lock(mutex_);
    if (!shared_ || !list) return;
    for (int i = 0; i < kSlots; ++i)
        if (slot_list_[i] == list && shared_->slots[i].state == kWriting) slot_submitted_[i] = true;
}

void Producer::on_present_marker(std::uint64_t frame) {
    std::lock_guard lock(mutex_);
    pending_present_frame_ = frame;
    have_present_marker_ = true;
    close_frames(frame);
    push_event(shared_, kEvPresentMarker, frame);
}

void Producer::on_tag(std::uint64_t frame, Tex kind, ID3D12Resource* source, D3D12_RESOURCE_STATES state,
                      std::uint32_t ext_x, std::uint32_t ext_y, std::uint32_t ext_w, std::uint32_t ext_h,
                      ID3D12GraphicsCommandList* list) {
    std::lock_guard lock(mutex_);
    if (!ready() || !source || !list) return;
    const auto desc = source->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1) return;
    // (the upscaler's output is only for detecting the HUD: not while the game hands its HUD over - a HUD layer or a
    // HUD-less picture with every one of the last 60 frames; see begin_present)
    if (kind == kScene && hud_given_ >= 60) return;
    const int slot = slot_for_frame(frame, true);
    if (slot < 0) return;
    slot_submitted_[slot] = false;
    collect_retired();
    if (!ensure_texture(slot, kind, desc)) { ++shared_->hooks.copies_failed; return; }
    auto& t = textures_[slot][kind];
    // (the list as ReShade names it when the game hands it to the GPU - see tl_tag_list)
    tl_tagging = true; tl_tag_list = 0;
    const int timed = time_begin(list, kind == kDepth ? kTDepth : kind == kMotion ? kTMotion : kTOtherTag);
    copy_into(list, source, state, t.resource.Get());
    time_end(list, timed);
    tl_tagging = false;
    slot_list_[slot] = tl_tag_list ? tl_tag_list : reinterpret_cast<std::uint64_t>(list);
    auto& info = shared_->slots[slot].tex[kind];
    if (kind == kUi) ui_tagged_[slot] = true;
    info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height;
    info.format = t.desc.Format; info.generation = t.generation;
    if (!ext_w || !ext_h) { ext_x = ext_y = 0; ext_w = info.width; ext_h = info.height; }
    info.ext_x = ext_x; info.ext_y = ext_y; info.ext_w = ext_w; info.ext_h = ext_h;
    info.valid = 1;
    // extra: the buffer kind (low byte) and which game resource it came from (a game may alternate several).
    push_event(shared_, kEvTag, frame, static_cast<std::uint64_t>(kind) | (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(source)) << 8));
}

void Producer::release_generated_if_idle() {
    const std::int64_t last = generated_qpc_.load();
    if (!last || qpc_now() - last < shared_->qpc_frequency * 3) return;
    for (int slot = 0; slot < kSlots; ++slot)
        for (int i = 0; i < kMaxGenerated; ++i) {
            auto& t = textures_[slot][kGen0 + i];
            if (!t.resource) continue;
            retired_.push_back({t.resource, t.handle, fence_value_ + 1, t.vk, t.d11});
            const std::uint32_t generation = t.generation;  // (a new one gets a new name: the old may still be open)
            t = Texture{};
            t.generation = generation;
            if (shared_->slots[slot].state == kWriting) shared_->slots[slot].tex[kGen0 + i].valid = 0;
        }
    generated_qpc_ = 0;
}

int Producer::present_slot() {
    release_generated_if_idle();
    InterlockedIncrement(&shared_->presents_total);
    // Frame generation's own presents (generated and real images, from its own thread): the frames are
    // published where their images are generated (on_generated).
    if (generation_active()) {
        InterlockedExchange(&shared_->presents_without_frame, 0);
        push_event(shared_, kEvPresent, pending_present_frame_, 0);
        return -1;
    }
    // Which frame is being presented: the PCL present-start marker when the game sends it; otherwise the newest frame
    // whose data the GPU was given before this present (the command list its tags were recorded on submitted) - a game
    // that records the next frame ahead (DOOM Eternal) has not submitted that one yet, a game that does not (Hi-Fi
    // Rush) has submitted this one; the oldest frame still being written paired Hi-Fi Rush's every picture with the
    // frame before, for good, once a frame was left behind. Without that, the oldest frame still being written.
    int slot = -1;
    std::uint64_t via = 0;  // 1: present marker, 2: oldest-frame fallback, 3: newest submitted
    if (have_present_marker_) { slot = slot_for_frame(pending_present_frame_, false); if (slot >= 0) via = 1; }
    if (slot < 0) {
        for (int i = 0; i < kSlots; ++i)
            if (shared_->slots[i].state == kWriting && shared_->slots[i].camera.valid && slot_submitted_[i] &&
                qpc_now() - shared_->slots[i].qpc_constants < shared_->qpc_frequency &&
                (slot < 0 || shared_->slots[i].frame_id > shared_->slots[slot].frame_id)) slot = i;
        if (slot >= 0) via = 3;
    }
    if (slot < 0) {
        via = 2;
        for (int i = 0; i < kSlots; ++i)
            // Only frames from the last second: a slot left behind when the game stopped sending data for it
            // (e.g. switching DLSS -> FSR) must not pass for the current frame forever.
            if (shared_->slots[i].state == kWriting && shared_->slots[i].camera.valid &&
                qpc_now() - shared_->slots[i].qpc_constants < shared_->qpc_frequency &&
                (slot < 0 || shared_->slots[i].frame_id < shared_->slots[slot].frame_id)) slot = i;
    }
    have_present_marker_ = false;
    push_event(shared_, kEvPresent, slot >= 0 ? shared_->slots[slot].frame_id : pending_present_frame_, slot >= 0 ? via : 0);
    if (slot < 0) { InterlockedIncrement(&shared_->presents_without_frame); return -1; }
    InterlockedExchange(&shared_->presents_without_frame, 0);
    auto& m = shared_->slots[slot];
    close_frames(m.frame_id);  // (without present markers: the frame presented closes here)
    // Frames older than the one being presented will never be presented: release them.
    for (int i = 0; i < kSlots; ++i)
        if (i != slot && shared_->slots[i].state == kWriting && shared_->slots[i].frame_id < m.frame_id)
            InterlockedExchange(&shared_->slots[i].state, kFree);
    collect_retired();
    return slot;
}

void Producer::on_ui_layer(ID3D12Resource* source, D3D12_RESOURCE_STATES state, ID3D12GraphicsCommandList* list, bool inverted) {
    std::lock_guard lock(mutex_);
    if (!ready() || !source || !list || !device_) return;
    const auto desc = source->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1) return;
    // (an sRGB layer copied as its family's typeless format: read as plain UNORM, its bytes as the game stored them - read
    // through its sRGB format it came out decoded to linear and was stored in 8 bits so, the darks in steps: banding in
    // RE Requiem's menus and dark corners where the scenery under its layer moved, 2026-10-22)
    DXGI_FORMAT format = copy_format(desc.Format);
    if (format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    else if (format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) format = DXGI_FORMAT_B8G8R8A8_TYPELESS;
    if (!ui_pending_ || ui_pending_->GetDesc().Width != desc.Width || ui_pending_->GetDesc().Height != desc.Height ||
        ui_pending_->GetDesc().Format != format) {
        // (the old one may still be in use on the GPU - the last frame's copies: kept until the fence is past it, as the
        // shared textures are; released at once, the game hung and crashed when its pause menu brought a layer of
        // another format)
        if (ui_pending_) retired_.push_back({ui_pending_, nullptr, fence_value_ + 1, {}, nullptr});
        ui_pending_.Reset();
        ui_pending_valid_ = false;
        collect_retired();
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = desc.Width; d.Height = desc.Height; d.DepthOrArraySize = 1; d.MipLevels = 1;
        d.Format = format; d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ui_pending_)))) {
            set_message("HUD layer: create failed");
            return;
        }
    }
    const int timed = time_begin(list, kTUiTake);
    copy_into(list, source, state, ui_pending_.Get());
    time_end(list, timed);
    ui_pending_valid_ = true;
    ui_pending_inverted_ = inverted;
    shared_->hooks.ui_layer_w = static_cast<std::uint32_t>(desc.Width); shared_->hooks.ui_layer_h = desc.Height;
    shared_->hooks.ui_layer_format = desc.Format;
    ++shared_->hooks.ui_layer_taken;
}

void Producer::on_hudless_copy(ID3D12Resource* source, D3D12_RESOURCE_STATES state, ID3D12GraphicsCommandList* list) {
    std::lock_guard lock(mutex_);
    if (!source) { hudless_age_ = 2; return; }  // (the copy taken this frame given up: not handed over)
    if (!ready() || !list || !device_) return;
    const auto desc = source->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1) return;
    const DXGI_FORMAT format = copy_format(desc.Format);
    if (!hudless_pending_ || hudless_pending_->GetDesc().Width != desc.Width || hudless_pending_->GetDesc().Height != desc.Height ||
        hudless_pending_->GetDesc().Format != format) {
        if (hudless_pending_) retired_.push_back({hudless_pending_, nullptr, fence_value_ + 1, {}, nullptr});
        hudless_pending_.Reset();
        hudless_age_ = 2;
        collect_retired();
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = desc.Width; d.Height = desc.Height; d.DepthOrArraySize = 1; d.MipLevels = 1;
        d.Format = format; d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&hudless_pending_)))) {
            set_message("HUD-less copy: create failed");
            return;
        }
    }
    const int timed = time_begin(list, kTHudlessTake);
    copy_into(list, source, state, hudless_pending_.Get());
    time_end(list, timed);
    hudless_age_ = 0;
    ++shared_->hooks.hudless_copies;
}

// HUD learning, Direct3D 12: the copy at the candidate point (see hud_learn.hpp).
void Producer::on_hud_probe(ID3D12Resource* picture, D3D12_RESOURCE_STATES state, ID3D12GraphicsCommandList* list, std::uint32_t candidate,
                            bool end, bool use) {
    std::lock_guard lock(mutex_);
    if (!ready() || !picture || !list || !device_ || !candidate) return;
    const bool needs_end = (state & 0x80000000u) != 0;  // (see hud_learn.hpp)
    state = static_cast<D3D12_RESOURCE_STATES>(state & ~0x80000000u);
    const auto desc = picture->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1) return;
    const DXGI_FORMAT format = copy_format(desc.Format);
    auto& target = end ? probe_end_ : probe_pending_;
    if (!target || target->GetDesc().Width != desc.Width || target->GetDesc().Height != desc.Height || target->GetDesc().Format != format) {
        if (target) retired_.push_back({target, nullptr, fence_value_ + 1, {}, nullptr});
        target.Reset();
        (end ? probe_end_age_ : probe_age_) = 2;
        collect_retired();
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = desc.Width; d.Height = desc.Height; d.DepthOrArraySize = 1; d.MipLevels = 1;
        d.Format = format; d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target)))) return;
    }
    const int timed = time_begin(list, kTProbe);
    copy_into(list, picture, state, target.Get());
    time_end(list, timed);
    if (end) { probe_end_age_ = 0; return; }
    probe_age_ = 0;
    probe_candidate_ = candidate;
    probe_use_ = use;
    probe_needs_end_ = needs_end;
    ++shared_->hooks.learn_probes;
}
// ...at the present: where the frame's final picture (its back buffer copy) differs from it, into the probe slot.
static const char kProbeShader[] = R"(
Texture2D<float4> before_t : register(t0);
Texture2D<float4> now_t : register(t1);
RWTexture2D<unorm float> cover : register(u0);
[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    cover.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    const float3 a = now_t.Load(int3(id.xy, 0)).rgb, b = before_t.Load(int3(id.xy, 0)).rgb;
    cover[id.xy] = any(abs(a - b) > 0.004 + 0.03 * max(abs(a), abs(b))) ? 1.0 : 0.0;
}
)";
constexpr UINT kProbeRing = 8;
bool Producer::ensure_probe_pass() {
    if (probe_pso_) return true;
    if (probe_failed_ || !device_) return false;
    probe_failed_ = true;
    ComPtr<ID3DBlob> code, errors, blob;
    if (FAILED(D3DCompile(kProbeShader, sizeof(kProbeShader) - 1, "hud_probe", nullptr, nullptr, "main", "cs_5_0", 0, 0, &code, &errors))) return false;
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 2};
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {2, ranges};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rd{1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors)) ||
        FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&probe_root_))))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = probe_root_.Get();
    pd.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3 * kProbeRing, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(device_->CreateComputePipelineState(&pd, IID_PPV_ARGS(&probe_pso_))) || FAILED(device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&probe_heap_)))) {
        probe_pso_.Reset();
        return false;
    }
    probe_failed_ = false;
    return true;
}
void Producer::d3d12_probe(ID3D12GraphicsCommandList* list, int slot) {
    if (!probe_pending_ || probe_age_ > 1 || !probe_candidate_) return;
    probe_age_ = 2;  // (handed over once)
    auto& bbt = textures_[slot][kBackbuffer];
    const auto pd = probe_pending_->GetDesc();
    // (the final picture as the back buffer's draw read it when there is one - a candidate in the final picture - else
    // the back buffer)
    const bool end = probe_end_ && probe_end_age_ <= 1 && probe_end_->GetDesc().Width == pd.Width && probe_end_->GetDesc().Height == pd.Height;
    probe_end_age_ = 2;
    if (probe_needs_end_ && !end) return;  // (a copy in the final picture without the end copy: not compared with the back buffer)
    ID3D12Resource* const now = end ? probe_end_.Get() : bbt.resource.Get();
    const DXGI_FORMAT now_format = end ? probe_end_->GetDesc().Format : bbt.desc.Format;
    if (!now || (!end && (bbt.desc.Width != pd.Width || bbt.desc.Height != pd.Height)) || !ensure_probe_pass()) return;
    if (!probe_cov_ || probe_cov_->GetDesc().Width != pd.Width || probe_cov_->GetDesc().Height != pd.Height) {
        if (probe_cov_) retired_.push_back({probe_cov_, nullptr, fence_value_ + 1, {}, nullptr});
        probe_cov_.Reset();
        D3D12_RESOURCE_DESC ad{};
        ad.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        ad.Width = pd.Width; ad.Height = pd.Height; ad.DepthOrArraySize = 1; ad.MipLevels = 1;
        ad.Format = DXGI_FORMAT_R8_UNORM; ad.SampleDesc.Count = 1; ad.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &ad, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&probe_cov_)))) return;
    }
    if (!ensure_texture(slot, kHudProbe, probe_cov_->GetDesc())) return;
    const UINT step = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT base = (probe_ring_++ % kProbeRing) * 3;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = probe_heap_->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += SIZE_T(base) * step;
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Texture2D.MipLevels = 1;
    sv.Format = typed_format(pd.Format);
    device_->CreateShaderResourceView(probe_pending_.Get(), &sv, cpu);
    cpu.ptr += step;
    sv.Format = typed_format(now_format);
    device_->CreateShaderResourceView(now, &sv, cpu);
    cpu.ptr += step;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
    uv.Format = DXGI_FORMAT_R8_UNORM;
    uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device_->CreateUnorderedAccessView(probe_cov_.Get(), nullptr, &uv, cpu);
    D3D12_RESOURCE_BARRIER b[3]{};
    for (auto& x : b) x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b[0].Transition = {probe_pending_.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    b[1].Transition = {now, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    b[2].Transition = {probe_cov_.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    list->ResourceBarrier(3, b);
    ID3D12DescriptorHeap* heaps[] = {probe_heap_.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(probe_root_.Get());
    list->SetPipelineState(probe_pso_.Get());
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = probe_heap_->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += UINT64(base) * step;
    list->SetComputeRootDescriptorTable(0, gpu);
    list->Dispatch(UINT((pd.Width + 7) / 8), (pd.Height + 7) / 8, 1);
    for (auto& x : b) std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
    list->ResourceBarrier(3, b);
    auto& t = textures_[slot][kHudProbe];
    copy_into(list, probe_cov_.Get(), D3D12_RESOURCE_STATE_COMMON, t.resource.Get());
    auto& m = shared_->slots[slot];
    auto& info = m.tex[kHudProbe];
    info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
    info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
    info.valid = 1;
    m.hud_candidate = probe_candidate_;
    probe_used_ = probe_use_;
}
// The learned point (see hud_learn.hpp): its comparison this frame as the frame's HUD - or, if it did not draw, an empty
// one - where nothing else gave the HUD (the game's HUD layers / HUD-less picture, ui_layer's patterns come first).
void Producer::d3d12_learned(ID3D12GraphicsCommandList* list, int slot) {
    auto& m = shared_->slots[slot];
    const bool used = probe_used_, none = probe_none_ && !probe_used_;
    probe_used_ = probe_none_ = false;
    if ((!used && !none) || m.tex[kUi].valid || m.tex[kHudless].valid) return;
    const auto& bb = m.tex[kBackbuffer];
    if (!bb.valid) return;
    ID3D12Resource* source = nullptr;
    if (used && probe_cov_ && probe_cov_->GetDesc().Width == bb.width && probe_cov_->GetDesc().Height == bb.height) {
        source = probe_cov_.Get();
    } else if (none) {
        if (!probe_zero_ || probe_zero_->GetDesc().Width != bb.width || probe_zero_->GetDesc().Height != bb.height) {
            if (probe_zero_) retired_.push_back({probe_zero_, nullptr, fence_value_ + 1, {}, nullptr});
            probe_zero_.Reset();
            D3D12_RESOURCE_DESC zd{};
            zd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            zd.Width = bb.width; zd.Height = bb.height; zd.DepthOrArraySize = 1; zd.MipLevels = 1;
            zd.Format = DXGI_FORMAT_R8_UNORM; zd.SampleDesc.Count = 1;
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;  // (created zeroed)
            if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &zd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&probe_zero_)))) return;
        }
        source = probe_zero_.Get();
    }
    if (!source || !ensure_texture(slot, kUi, source->GetDesc())) return;
    auto& t = textures_[slot][kUi];
    copy_into(list, source, D3D12_RESOURCE_STATE_COMMON, t.resource.Get());
    auto& info = m.tex[kUi];
    info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
    info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
    info.valid = 1;
    m.hud_learned = used ? 1 : 2;
    ++shared_->hooks.learn_used;
}

bool Producer::ensure_own(Texture& t, const D3D12_RESOURCE_DESC& source) {
    const DXGI_FORMAT format = copy_format(source.Format);
    if (t.resource && t.desc.Width == source.Width && t.desc.Height == source.Height && t.desc.Format == format && (!vk_ || t.vk.image)) return true;
    if (t.resource) retired_.push_back({t.resource, t.handle, fence_value_ + 1, t.vk, t.d11});
    t = Texture{};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = source.Width; desc.Height = source.Height; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = format; desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t.resource))) ||
        FAILED(device_->CreateSharedHandle(t.resource.Get(), nullptr, GENERIC_ALL, nullptr, &t.handle))) {
        set_message("HUD layer (Vulkan): create failed"); t = Texture{}; return false;
    }
    t.desc = desc;
    if (vk_) {
        std::string error;
        if (!vk_->import(t.handle, vk_format_of(format), static_cast<std::uint32_t>(desc.Width), desc.Height, 0, t.vk, error)) {
            set_message(error.c_str());
            CloseHandle(t.handle); t = Texture{};
            return false;
        }
    }
    return true;
}

void Producer::on_ui_layer_vk(std::uint64_t image, DXGI_FORMAT format, std::uint32_t width, std::uint32_t height, std::uint32_t layout,
                              void* command_buffer) {
    std::lock_guard lock(mutex_);
    if (!ready() || !vk_ || !image || !command_buffer || !width || !height || vk_format_of(copy_format(format)) == VK_FORMAT_UNDEFINED) return;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height; desc.Format = format;
    collect_retired();
    if (!ensure_own(ui_vk_, desc)) return;
    vk_->copy(static_cast<VkCommandBuffer>(command_buffer), reinterpret_cast<VkImage>(image), VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
              static_cast<VkImageLayout>(layout), ui_vk_.vk, width, height);
    ui_vk_age_ = 0;
    shared_->hooks.ui_layer_w = width; shared_->hooks.ui_layer_h = height; shared_->hooks.ui_layer_format = format;
    ++shared_->hooks.ui_layer_taken;
}

void Producer::on_hudless_vk(std::uint64_t image, DXGI_FORMAT format, std::uint32_t width, std::uint32_t height, std::uint32_t layout,
                             void* command_buffer) {
    std::lock_guard lock(mutex_);
    // (a swapchain made format-mutable is reported typeless: the plain format, as the back buffer's own copy - begin_present_vk)
    switch (format) {
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: format = DXGI_FORMAT_B8G8R8A8_UNORM; break;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: format = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: format = DXGI_FORMAT_R10G10B10A2_UNORM; break;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: format = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
        default: break;
    }
    if (!ready() || !vk_ || !image || !command_buffer || !width || !height || vk_format_of(format) == VK_FORMAT_UNDEFINED) return;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height; desc.Format = format;
    collect_retired();
    if (!ensure_own(hudless_vk_, desc)) return;
    vk_->copy(static_cast<VkCommandBuffer>(command_buffer), reinterpret_cast<VkImage>(image), VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
              static_cast<VkImageLayout>(layout), hudless_vk_.vk, width, height);
    hudless_vk_fresh_ = true;
}

// Direct3D 11, third pattern: the comparison of the picture with its HUD and the copy before it (see on_hud_before_d3d11).
static const char kD11CoverShader[] = R"(
Texture2D<float4> now_t : register(t0);
Texture2D<float4> before_t : register(t1);
RWTexture2D<unorm float> cover : register(u0);
[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    cover.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    const float3 a = now_t.Load(int3(id.xy, 0)).rgb, b = before_t.Load(int3(id.xy, 0)).rgb;
    // (what the HUD changed: more than a step of an 8-bit picture, or 3% of an HDR one)
    cover[id.xy] = any(abs(a - b) > 0.004 + 0.03 * max(abs(a), abs(b))) ? 1.0 : 0.0;
}
)";
void Producer::on_hud_before_d3d11(ID3D11Resource* picture, std::uint32_t kind, ID3D11DeviceContext* context) {
    std::lock_guard lock(mutex_);
    if (!ready() || !d11_ || !context) return;
    if (kind == 1) {  // (no HUD this frame; a copy taken this frame comes first)
        if (!d11_before_taken_) d11_none_ = true;
        return;
    }
    if (kind == 2) {
        if (d11_before_taken_) d11_confirmed_ = true;
        return;
    }
    if (kind == 3) {
        d11_before_taken_ = d11_confirmed_ = false;
        d11_picture_.Reset();
        return;
    }
    if (!picture) return;
    ComPtr<ID3D11Texture2D> texture;
    D3D11_TEXTURE2D_DESC pd{};
    if (FAILED(picture->QueryInterface(IID_PPV_ARGS(&texture)))) return;
    texture->GetDesc(&pd);
    if (pd.SampleDesc.Count != 1) return;
    D3D11_TEXTURE2D_DESC bd{};
    if (d11_before_) d11_before_->GetDesc(&bd);
    if (!d11_before_ || bd.Width != pd.Width || bd.Height != pd.Height || bd.Format != pd.Format) {
        ComPtr<ID3D11Device> device;
        context->GetDevice(&device);
        D3D11_TEXTURE2D_DESC d = pd;
        d.MipLevels = 1; d.ArraySize = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_SHADER_RESOURCE; d.CPUAccessFlags = 0; d.MiscFlags = 0;
        d11_before_.Reset();
        if (!device || FAILED(device->CreateTexture2D(&d, nullptr, &d11_before_))) { set_message("HUD (Direct3D 11): copy failed"); return; }
    }
    if (pd.MipLevels == 1 && pd.ArraySize == 1) context->CopyResource(d11_before_.Get(), picture);
    else context->CopySubresourceRegion(d11_before_.Get(), 0, 0, 0, 0, picture, 0, nullptr);
    d11_picture_ = picture;
    d11_before_taken_ = true;
    d11_none_ = d11_confirmed_ = false;
    shared_->hooks.ui_layer_w = pd.Width; shared_->hooks.ui_layer_h = pd.Height; shared_->hooks.ui_layer_format = pd.Format;
    ++shared_->hooks.ui_layer_taken;
}
static DXGI_FORMAT d11_srv_format(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        default: return f;
    }
}
bool Producer::ensure_d11_cover_cs(ID3D11Device* device) {
    if (!d11_cover_cs_ && !d11_cs_failed_ && device) {
        ComPtr<ID3DBlob> code, errors;
        d11_cs_failed_ = true;
        if (SUCCEEDED(D3DCompile(kD11CoverShader, sizeof(kD11CoverShader) - 1, "hud_cover", nullptr, nullptr, "main", "cs_5_0", 0, 0, &code, &errors)) &&
            SUCCEEDED(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &d11_cover_cs_)))
            d11_cs_failed_ = false;
        else set_message("HUD (Direct3D 11): shader failed");
    }
    return d11_cover_cs_ != nullptr;
}

// HUD learning, Direct3D 11: the copy at the candidate point (see hud_learn.hpp).
void Producer::on_hud_probe_d3d11(ID3D11Resource* picture, ID3D11DeviceContext* context, std::uint32_t candidate, bool use) {
    std::lock_guard lock(mutex_);
    if (!ready() || !d11_ || !picture || !context || !candidate) return;
    ComPtr<ID3D11Texture2D> texture;
    D3D11_TEXTURE2D_DESC pd{};
    if (FAILED(picture->QueryInterface(IID_PPV_ARGS(&texture)))) return;
    texture->GetDesc(&pd);
    if (pd.SampleDesc.Count != 1) return;
    D3D11_TEXTURE2D_DESC bd{};
    if (d11_probe_before_) d11_probe_before_->GetDesc(&bd);
    if (!d11_probe_before_ || bd.Width != pd.Width || bd.Height != pd.Height || bd.Format != pd.Format) {
        ComPtr<ID3D11Device> device;
        context->GetDevice(&device);
        D3D11_TEXTURE2D_DESC d = pd;
        d.MipLevels = 1; d.ArraySize = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_SHADER_RESOURCE; d.CPUAccessFlags = 0; d.MiscFlags = 0;
        d11_probe_before_.Reset();
        if (!device || FAILED(device->CreateTexture2D(&d, nullptr, &d11_probe_before_))) return;
    }
    if (pd.MipLevels == 1 && pd.ArraySize == 1) context->CopyResource(d11_probe_before_.Get(), picture);
    else context->CopySubresourceRegion(d11_probe_before_.Get(), 0, 0, 0, 0, picture, 0, nullptr);
    d11_probe_picture_ = picture;
    d11_probe_taken_ = true;
    d11_probe_use_ = use;
    probe_candidate_ = candidate;
    ++shared_->hooks.learn_probes;
}
// ...at the present: where the picture now differs from that copy, into the frame's probe slot.
void Producer::d3d11_probe(ID3D11DeviceContext* context, int slot) {
    if (!d11_probe_taken_) return;
    d11_probe_taken_ = false;
    ComPtr<ID3D11Resource> picture = std::move(d11_probe_picture_);
    ComPtr<ID3D11Device> device;
    context->GetDevice(&device);
    if (!picture || !d11_probe_before_ || !device || !ensure_d11_cover_cs(device.Get())) return;
    D3D11_TEXTURE2D_DESC bd{};
    d11_probe_before_->GetDesc(&bd);
    D3D11_TEXTURE2D_DESC cd{};
    if (d11_probe_cov_) d11_probe_cov_->GetDesc(&cd);
    if (!d11_probe_cov_ || cd.Width != bd.Width || cd.Height != bd.Height) {
        D3D11_TEXTURE2D_DESC d{};
        d.Width = bd.Width; d.Height = bd.Height; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_R8_UNORM;
        d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        d11_probe_cov_.Reset();
        if (FAILED(device->CreateTexture2D(&d, nullptr, &d11_probe_cov_))) return;
    }
    ComPtr<ID3D11ShaderResourceView> now_view, before_view;
    ComPtr<ID3D11UnorderedAccessView> cover_view;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = d11_srv_format(bd.Format); sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D; sv.Texture2D.MipLevels = 1;
    if (FAILED(device->CreateShaderResourceView(picture.Get(), &sv, &now_view)) ||
        FAILED(device->CreateShaderResourceView(d11_probe_before_.Get(), &sv, &before_view)) ||
        FAILED(device->CreateUnorderedAccessView(d11_probe_cov_.Get(), nullptr, &cover_view)))
        return;
    ComPtr<ID3D11ComputeShader> old_cs;
    ID3D11ClassInstance* raw_instances[256] = {};
    UINT instance_count = 256;
    context->CSGetShader(&old_cs, raw_instances, &instance_count);
    ID3D11ShaderResourceView* old_srv[2] = {};
    ID3D11UnorderedAccessView* old_uav[1] = {};
    context->CSGetShaderResources(0, 2, old_srv);
    context->CSGetUnorderedAccessViews(0, 1, old_uav);
    ID3D11ShaderResourceView* srvs[2] = {now_view.Get(), before_view.Get()};
    ID3D11UnorderedAccessView* uavs[1] = {cover_view.Get()};
    context->CSSetShader(d11_cover_cs_.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 2, srvs);
    context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    context->Dispatch((bd.Width + 7) / 8, (bd.Height + 7) / 8, 1);
    ID3D11ShaderResourceView* none_srv[2] = {};
    context->CSSetShaderResources(0, 2, none_srv);
    context->CSSetShader(old_cs.Get(), raw_instances, instance_count);
    context->CSSetShaderResources(0, 2, old_srv);
    context->CSSetUnorderedAccessViews(0, 1, old_uav, nullptr);
    for (UINT i = 0; i < instance_count; ++i) if (raw_instances[i]) raw_instances[i]->Release();
    for (auto* v : old_srv) if (v) v->Release();
    for (auto* v : old_uav) if (v) v->Release();
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = bd.Width; desc.Height = bd.Height; desc.Format = DXGI_FORMAT_R8_UNORM;
    if (!ensure_texture(slot, kHudProbe, desc)) return;
    auto& t = textures_[slot][kHudProbe];
    context->CopyResource(t.d11.Get(), d11_probe_cov_.Get());
    auto& m = shared_->slots[slot];
    auto& info = m.tex[kHudProbe];
    info.width = bd.Width; info.height = bd.Height; info.format = t.desc.Format;
    info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = bd.Width; info.ext_h = bd.Height;
    info.valid = 1;
    m.hud_candidate = probe_candidate_;
    d11_probe_used_ = d11_probe_use_;
}
// Direct3D 11: as d3d12_learned.
void Producer::d3d11_learned(ID3D11DeviceContext* context, int slot) {
    auto& m = shared_->slots[slot];
    const bool used = d11_probe_used_, none = probe_none_ && !d11_probe_used_;
    d11_probe_used_ = probe_none_ = false;
    if ((!used && !none) || m.tex[kUi].valid || m.tex[kHudless].valid) return;
    const auto& bb = m.tex[kBackbuffer];
    if (!bb.valid) return;
    ComPtr<ID3D11Device> device;
    context->GetDevice(&device);
    if (!device) return;
    D3D11_TEXTURE2D_DESC cd{};
    if (d11_probe_cov_) d11_probe_cov_->GetDesc(&cd);
    if (!d11_probe_cov_ || cd.Width != bb.width || cd.Height != bb.height) {
        if (used) return;
        D3D11_TEXTURE2D_DESC d{};
        d.Width = bb.width; d.Height = bb.height; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_R8_UNORM;
        d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        d11_probe_cov_.Reset();
        if (FAILED(device->CreateTexture2D(&d, nullptr, &d11_probe_cov_))) return;
    }
    if (none) {  // (the comparison's texture is free this frame: cleared)
        ComPtr<ID3D11UnorderedAccessView> view;
        if (FAILED(device->CreateUnorderedAccessView(d11_probe_cov_.Get(), nullptr, &view))) return;
        const float zero[4] = {};
        context->ClearUnorderedAccessViewFloat(view.Get(), zero);
    }
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = bb.width; desc.Height = bb.height; desc.Format = DXGI_FORMAT_R8_UNORM;
    if (!ensure_texture(slot, kUi, desc)) return;
    auto& t = textures_[slot][kUi];
    context->CopyResource(t.d11.Get(), d11_probe_cov_.Get());
    auto& info = m.tex[kUi];
    info.width = bb.width; info.height = bb.height; info.format = t.desc.Format;
    info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = bb.width; info.ext_h = bb.height;
    info.valid = 1;
    m.hud_learned = used ? 1 : 2;
    ++shared_->hooks.learn_used;
}

void Producer::d3d11_cover(ID3D11DeviceContext* context, int slot) {
    // (under the lock; the game's compute stage is restored after)
    const bool none = d11_none_ && !d11_before_taken_;
    const bool confirmed = d11_confirmed_;
    d11_none_ = d11_confirmed_ = false;
    if (!none && (!d11_before_taken_ || !confirmed || !d11_picture_ || !d11_before_)) {  // (unconfirmed: not handed over)
        d11_before_taken_ = false;
        d11_picture_.Reset();
        return;
    }
    d11_before_taken_ = false;
    auto& m = shared_->slots[slot];
    if (m.tex[kUi].valid) { d11_picture_.Reset(); return; }  // (a UI layer the game tags itself)
    ComPtr<ID3D11Device> device;
    context->GetDevice(&device);
    if (!device) return;
    if (none) {  // (an empty HUD: the coverage cleared, the picture's size - the back buffer's)
        const auto& bb = m.tex[kBackbuffer];
        if (!bb.valid || !bb.width || !bb.height) return;
        D3D11_TEXTURE2D_DESC cd{};
        if (d11_cover_) d11_cover_->GetDesc(&cd);
        if (!d11_cover_ || cd.Width != bb.width || cd.Height != bb.height) {
            D3D11_TEXTURE2D_DESC d{};
            d.Width = bb.width; d.Height = bb.height; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_R8_UNORM;
            d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
            d11_cover_.Reset();
            if (FAILED(device->CreateTexture2D(&d, nullptr, &d11_cover_))) return;
        }
        ComPtr<ID3D11UnorderedAccessView> cover_view;
        if (FAILED(device->CreateUnorderedAccessView(d11_cover_.Get(), nullptr, &cover_view))) return;
        const float zero[4] = {};
        context->ClearUnorderedAccessViewFloat(cover_view.Get(), zero);
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = bb.width; desc.Height = bb.height; desc.Format = DXGI_FORMAT_R8_UNORM;
        if (!ensure_texture(slot, kUi, desc)) return;
        auto& t = textures_[slot][kUi];
        context->CopyResource(t.d11.Get(), d11_cover_.Get());
        auto& info = m.tex[kUi];
        info.width = bb.width; info.height = bb.height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = bb.width; info.ext_h = bb.height;
        info.valid = 1;
        ++shared_->hooks.ui_layer_handed;
        return;
    }
    if (!ensure_d11_cover_cs(device.Get())) return;
    D3D11_TEXTURE2D_DESC bd{};
    d11_before_->GetDesc(&bd);
    D3D11_TEXTURE2D_DESC cd{};
    if (d11_cover_) d11_cover_->GetDesc(&cd);
    if (!d11_cover_ || cd.Width != bd.Width || cd.Height != bd.Height) {
        D3D11_TEXTURE2D_DESC d{};
        d.Width = bd.Width; d.Height = bd.Height; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_R8_UNORM;
        d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        d11_cover_.Reset();
        if (FAILED(device->CreateTexture2D(&d, nullptr, &d11_cover_))) return;
    }
    ComPtr<ID3D11ShaderResourceView> now_view, before_view;
    ComPtr<ID3D11UnorderedAccessView> cover_view;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = d11_srv_format(bd.Format); sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D; sv.Texture2D.MipLevels = 1;
    if (FAILED(device->CreateShaderResourceView(d11_picture_.Get(), &sv, &now_view)) ||
        FAILED(device->CreateShaderResourceView(d11_before_.Get(), &sv, &before_view)) ||
        FAILED(device->CreateUnorderedAccessView(d11_cover_.Get(), nullptr, &cover_view))) {
        d11_picture_.Reset();
        return;
    }
    // (the game's compute stage as it was: shader, the two views, the one output)
    ComPtr<ID3D11ComputeShader> old_cs;
    ComPtr<ID3D11ClassInstance> instances[256];
    UINT instance_count = 256;
    ID3D11ClassInstance* raw_instances[256] = {};
    context->CSGetShader(&old_cs, raw_instances, &instance_count);
    for (UINT i = 0; i < instance_count; ++i) instances[i].Attach(raw_instances[i]);
    ID3D11ShaderResourceView* old_srv[2] = {};
    ID3D11UnorderedAccessView* old_uav[1] = {};
    context->CSGetShaderResources(0, 2, old_srv);
    context->CSGetUnorderedAccessViews(0, 1, old_uav);
    ID3D11ShaderResourceView* srvs[2] = {now_view.Get(), before_view.Get()};
    ID3D11UnorderedAccessView* uavs[1] = {cover_view.Get()};
    context->CSSetShader(d11_cover_cs_.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 2, srvs);
    context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    context->Dispatch((bd.Width + 7) / 8, (bd.Height + 7) / 8, 1);
    ID3D11ShaderResourceView* none_srv[2] = {};
    context->CSSetShaderResources(0, 2, none_srv);
    context->CSSetShader(old_cs.Get(), raw_instances, instance_count);
    context->CSSetShaderResources(0, 2, old_srv);
    context->CSSetUnorderedAccessViews(0, 1, old_uav, nullptr);
    for (auto* v : old_srv) if (v) v->Release();
    for (auto* v : old_uav) if (v) v->Release();
    d11_picture_.Reset();
    // ...into the frame's UI slot
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = bd.Width; desc.Height = bd.Height; desc.Format = DXGI_FORMAT_R8_UNORM;
    if (!ensure_texture(slot, kUi, desc)) return;
    auto& t = textures_[slot][kUi];
    context->CopyResource(t.d11.Get(), d11_cover_.Get());
    auto& info = m.tex[kUi];
    info.width = bd.Width; info.height = bd.Height; info.format = t.desc.Format;
    info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = bd.Width; info.ext_h = bd.Height;
    info.valid = 1;
    ++shared_->hooks.ui_layer_handed;
}

// The HUD layer pass (see begin_present): its colour and coverage - premultiplied RGB as the game blended it in, and alpha;
// 1 - alpha for a layer cleared to opaque black (INVERTED: its alpha is how much of the picture shows through, Armored Core
// VI - its colour is the HUD's all the same). With the colour the presenter takes the HUD back out of the frame exactly
// (cs_layer_unblend): a faint white line or a glow is no longer told from a dark panel by its coverage alone. 8 bits a
// channel, half floats for a layer that is floating point (HDR: its colour above 1).
static const char kUiAlphaShader[] = R"(
Texture2D<float4> layer : register(t0);
RWTexture2D<float4> coverage : register(u0);
[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {
    uint w, h;
    coverage.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    const float4 c = layer.Load(int3(id.xy, 0));
#ifdef INVERTED
    coverage[id.xy] = float4(c.rgb, 1.0 - c.a);
#else
    coverage[id.xy] = c;
#endif
}
)";
constexpr UINT kUiRing = 8;
bool Producer::ensure_ui_pass() {
    if (ui_pso_) return true;
    if (ui_pass_failed_ || !device_) return false;
    ui_pass_failed_ = true;  // (until it succeeds: tried once)
    ComPtr<ID3DBlob> code, code_inverted, errors;
    const D3D_SHADER_MACRO inverted[] = {{"INVERTED", "1"}, {nullptr, nullptr}};
    if (FAILED(D3DCompile(kUiAlphaShader, sizeof(kUiAlphaShader) - 1, "ui_alpha", nullptr, nullptr, "main", "cs_5_0", 0, 0, &code, &errors)) ||
        FAILED(D3DCompile(kUiAlphaShader, sizeof(kUiAlphaShader) - 1, "ui_alpha", inverted, nullptr, "main", "cs_5_0", 0, 0, &code_inverted, &errors))) {
        set_message("HUD layer: shader failed");
        shared_->hooks.ui_layer_pass = 2;
        return false;
    }
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 1};
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {2, ranges};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rd{1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob;
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors)) ||
        FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&ui_root_)))) {
        set_message("HUD layer: root signature failed");
        shared_->hooks.ui_layer_pass = 2;
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = ui_root_.Get();
    pd.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2 * kUiRing, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    D3D12_COMPUTE_PIPELINE_STATE_DESC pi = pd;
    pi.CS = {code_inverted->GetBufferPointer(), code_inverted->GetBufferSize()};
    if (FAILED(device_->CreateComputePipelineState(&pd, IID_PPV_ARGS(&ui_pso_))) || FAILED(device_->CreateComputePipelineState(&pi, IID_PPV_ARGS(&ui_pso_inverted_))) ||
        FAILED(device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&ui_heap_)))) {
        ui_pso_.Reset();
        set_message("HUD layer: pipeline failed");
        shared_->hooks.ui_layer_pass = 2;
        return false;
    }
    ui_heap_step_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    ui_pass_failed_ = false;
    shared_->hooks.ui_layer_pass = 1;
    return true;
}
// A shader can only read a typed format.
static DXGI_FORMAT typed_format(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default: return f;
    }
}

std::uint64_t Producer::begin_present(ID3D12Resource* backbuffer, ID3D12GraphicsCommandList* list) {
    std::lock_guard lock(mutex_);
    if (!ready() || !backbuffer || !list) return 0;
    const int slot = present_slot();
    if (slot < 0) return 0;
    auto& m = shared_->slots[slot];
    const auto desc = backbuffer->GetDesc();
    if (ensure_texture(slot, kBackbuffer, desc)) {
        auto& t = textures_[slot][kBackbuffer];
        const int timed = time_begin(list, kTBackbuffer);
        copy_into(list, backbuffer, D3D12_RESOURCE_STATE_PRESENT, t.resource.Get());
        time_end(list, timed);
        auto& info = m.tex[kBackbuffer];
        info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
        info.valid = 1;
        d3d12_probe(list, slot);  // (HUD learning: the copy at this frame's candidate point, against this picture)
    }
    if (probe_age_ < 2) ++probe_age_;
    if (probe_end_age_ < 2) ++probe_end_age_;
    // A HUD layer and a HUD-less copy both taken this frame: the layer, unless it comes and goes - there on under 95% of the
    // frames lately, while the HUD-less copy is there - and the presenter has not checked it is the HUD (that needs the
    // upscaler's output). Hi-Fi Rush draws its HUD onto the back buffer (the copy before it gives all of it), and in some
    // places also blends a layer without its HUD (on 6-70% of the frames): the HUD warped on those, juddering on fast turns.
    const bool hudless_now = hudless_pending_ && hudless_age_ <= 1;
    if (ui_pending_valid_ || hudless_now) ui_layer_share_ += ((ui_pending_valid_ ? 1.0 : 0.0) - ui_layer_share_) / 300.0;
    const bool layer_not_kept = ui_pending_valid_ && hudless_now && shared_->presenter.layer_trust != 1 && ui_layer_share_ < 0.95;
    // (the HUD layer taken this frame, see on_ui_layer: its coverage into the frame's UI slot - on this list, ReShade's,
    // submitted right after: the descriptor heap bound here disturbs nothing of the game's)
    if (ui_pending_valid_ && !layer_not_kept && !m.tex[kUi].valid && ensure_ui_pass()) {
        const auto pd = ui_pending_->GetDesc();
        // (the layer handed over with its colour - see kUiAlphaShader)
        const DXGI_FORMAT tf = typed_format(pd.Format);
        const DXGI_FORMAT ui_format = tf == DXGI_FORMAT_R16G16B16A16_FLOAT || tf == DXGI_FORMAT_R32G32B32A32_FLOAT || tf == DXGI_FORMAT_R11G11B10_FLOAT
                                          ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
        if (!ui_alpha_ || ui_alpha_->GetDesc().Width != pd.Width || ui_alpha_->GetDesc().Height != pd.Height || ui_alpha_->GetDesc().Format != ui_format) {
            if (ui_alpha_) retired_.push_back({ui_alpha_, nullptr, fence_value_ + 1, {}, nullptr});
            ui_alpha_.Reset();
            D3D12_RESOURCE_DESC ad{};
            ad.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            ad.Width = pd.Width; ad.Height = pd.Height; ad.DepthOrArraySize = 1; ad.MipLevels = 1;
            ad.Format = ui_format; ad.SampleDesc.Count = 1; ad.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &ad, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ui_alpha_))))
                ui_alpha_.Reset();
        }
        if (ui_alpha_ && ensure_texture(slot, kUi, ui_alpha_->GetDesc(), 0, true)) {
            auto& t = textures_[slot][kUi];
            // (straight into the slot where it takes shader writes - see ensure_texture)
            const bool direct = (t.desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;
            ID3D12Resource* written = direct ? t.resource.Get() : ui_alpha_.Get();
            const UINT base = (ui_ring_++ % kUiRing) * 2;
            D3D12_CPU_DESCRIPTOR_HANDLE cpu = ui_heap_->GetCPUDescriptorHandleForHeapStart();
            cpu.ptr += SIZE_T(base) * ui_heap_step_;
            D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
            sv.Format = typed_format(pd.Format);
            sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            sv.Texture2D.MipLevels = 1;
            device_->CreateShaderResourceView(ui_pending_.Get(), &sv, cpu);
            cpu.ptr += ui_heap_step_;
            D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
            uv.Format = ui_format;
            uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            device_->CreateUnorderedAccessView(written, nullptr, &uv, cpu);
            D3D12_RESOURCE_BARRIER b[2]{};
            b[0].Type = b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b[0].Transition = {ui_pending_.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
            b[1].Transition = {written, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
            list->ResourceBarrier(2, b);
            ID3D12DescriptorHeap* heaps[] = {ui_heap_.Get()};
            list->SetDescriptorHeaps(1, heaps);
            list->SetComputeRootSignature(ui_root_.Get());
            list->SetPipelineState(ui_pending_inverted_ ? ui_pso_inverted_.Get() : ui_pso_.Get());
            D3D12_GPU_DESCRIPTOR_HANDLE gpu = ui_heap_->GetGPUDescriptorHandleForHeapStart();
            gpu.ptr += UINT64(base) * ui_heap_step_;
            list->SetComputeRootDescriptorTable(0, gpu);
            const int timed_pass = time_begin(list, kTUiPass);
            list->Dispatch(UINT((pd.Width + 7) / 8), (pd.Height + 7) / 8, 1);
            for (auto& x : b) std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
            list->ResourceBarrier(2, b);
            time_end(list, timed_pass);
            if (!direct) {
                const int timed_slot = time_begin(list, kTUiSlot);
                copy_into(list, ui_alpha_.Get(), D3D12_RESOURCE_STATE_COMMON, t.resource.Get());
                time_end(list, timed_slot);
            }
            auto& info = m.tex[kUi];
            info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
            info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
            info.valid = 1;
            ++shared_->hooks.ui_layer_handed;
        }
    }
    ui_pending_valid_ = false;
    // (the HUD-less copy taken for this frame, see on_hudless_copy: the latest one, its copy run on the GPU before this
    // list; also when it was recorded just after the last present - a game recording ahead - but not older. Not with a
    // HUD layer - the pair would pass for a game's HUD-less picture + UI layer - nor over the game's own)
    if (m.tex[kUi].valid) hudless_age_ = 2;
    if (hudless_pending_ && hudless_age_ <= 1 && !m.tex[kHudless].valid && ensure_texture(slot, kHudless, hudless_pending_->GetDesc())) {
        auto& t = textures_[slot][kHudless];
        const int timed = time_begin(list, kTHudlessSlot);
        copy_into(list, hudless_pending_.Get(), D3D12_RESOURCE_STATE_COMMON, t.resource.Get());
        time_end(list, timed);
        auto& info = m.tex[kHudless];
        info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
        info.valid = 1;
        ++shared_->hooks.hudless_handed;
    }
    if (hudless_age_ < 2) ++hudless_age_;
    // (HUD learning paused while the game gives its HUD - any HUD layer or HUD-less picture, the ones the add-on finds too:
    // after 60 frames with one, until 600 frames without - a loading screen does not start it again)
    if (m.tex[kUi].valid || m.tex[kHudless].valid) { game_hud_frames_ = std::min(game_hud_frames_ + 1, 60u); game_hud_missing_ = 0; }
    else if (++game_hud_missing_ >= 600) game_hud_frames_ = 0;
    d3d12_learned(list, slot);  // (HUD learning: the learned point, where nothing else gave the HUD)
    // Frames in a row the game handed its HUD over (a HUD-less picture, a UI layer it tags, or the HUD layer the add-on takes
    // once the presenter has checked it - until then the check needs the upscaler's output; and then a frame without it is
    // one where the game hides its HUD): past 60 the upscaler's output is no longer copied (on_tag); at 300 its slot
    // textures are released (64 MB each at 4K).
    // (by where it came from: the add-on's own layers have their colour too since 2.1 - told apart by format, they counted
    // as tagged and the upscaler's output stopped after 60 frames, before a wrong layer could be rejected)
    const bool tagged_ui = m.tex[kUi].valid && ui_tagged_[slot];
    hud_given_ = m.tex[kHudless].valid || tagged_ui || shared_->presenter.layer_trust == 1 ? hud_given_ + 1 : 0;
    if (hud_given_ == 300)
        for (int s = 0; s < kSlots; ++s) {
            auto& t = textures_[s][kScene];
            if (!t.resource) continue;
            retired_.push_back({t.resource, t.handle, fence_value_ + 1, t.vk, t.d11});
            const std::uint32_t generation = t.generation;  // (a new one gets a new name: the old may still be open)
            t = Texture{};
            t.generation = generation;
        }
    m.qpc_present = qpc_now();
    return m.frame_id + 1;  // 0 means "nothing to finish"
}

void Producer::finish_present(ID3D12CommandQueue* queue, std::uint64_t token) {
    std::lock_guard lock(mutex_);
    if (!token || !queue || !fence_) return;
    const std::uint64_t frame = token - 1;
    const int slot = slot_for_frame(frame, false);
    if (slot < 0) return;
    publish(slot, queue);
    time_report(queue);
}

void Producer::publish(int slot, ID3D12CommandQueue* queue) {
    if (FAILED(queue->Signal(fence_.Get(), ++fence_value_))) return;
    auto& m = shared_->slots[slot];
    m.fence_value = fence_value_;
    InterlockedExchange(&m.state, kReady);
    InterlockedExchange64(&shared_->latest_ready_frame, static_cast<LONG64>(m.frame_id));
    ++shared_->hooks.frames_published;
    push_event(shared_, kEvPublished, m.frame_id, fence_value_);
}

// Frame generation: the command list a frame's generated images are copied on is the game's frame generation's
// own, submitted wherever it submits it. At its end the frame's marker value is written into a small buffer the
// CPU reads (WriteBufferImmediate, written once everything before it on the list has completed); a thread
// watches for it and publishes the frame (the shared fence signalled from the CPU) as soon as it is there.
bool Producer::ensure_markers() {
    if (markers_) return true;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = kMarkers * 4; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&markers_)))) return false;
    void* mapped = nullptr;
    if (FAILED(markers_->Map(0, nullptr, &mapped))) { markers_.Reset(); return false; }
    marker_values_ = static_cast<volatile std::uint32_t*>(mapped);
    for (int i = 0; i < kMarkers; ++i) marker_values_[i] = 0;
    if (!marker_thread_started_) {
        marker_thread_started_ = true;
        std::thread([this]() {
            HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            while (!stop_markers_.load()) {
                if (pending_count_.load() == 0) { WaitForSingleObject(marker_event_, 100); continue; }
                poll_markers();
                LARGE_INTEGER due; due.QuadPart = -2500;  // 0.25 ms
                if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, 5);
                else Sleep(1);
            }
            if (timer) CloseHandle(timer);
        }).detach();
    }
    return true;
}

void Producer::poll_markers() {
    if (pending_count_.load() == 0) return;
    std::lock_guard lock(mutex_);
    if (!fence_ || !marker_values_) return;
    for (auto it = pending_publish_.begin(); it != pending_publish_.end();) {
        if (marker_values_[it->marker % kMarkers] == it->marker) {
            const int slot = slot_for_frame(it->frame, false);
            if (slot >= 0 && SUCCEEDED(fence_->Signal(++fence_value_))) {
                auto& m = shared_->slots[slot];
                m.fence_value = fence_value_;
                InterlockedExchange(&m.state, kReady);
                InterlockedExchange64(&shared_->latest_ready_frame, static_cast<LONG64>(m.frame_id));
                ++shared_->hooks.frames_published;
                push_event(shared_, kEvPublished, m.frame_id, fence_value_);
            }
            it = pending_publish_.erase(it);
        } else if (qpc_now() - it->qpc > shared_->qpc_frequency / 4) {
            push_event(shared_, kEvFrameGen, it->frame, 0x200);  // (diagnostics: never completed)
            it = pending_publish_.erase(it);
        } else ++it;
    }
    pending_count_ = static_cast<int>(pending_publish_.size());
}

void Producer::on_generated(std::uint64_t frame, std::uint32_t index, std::uint32_t per_frame, ID3D12Resource* image, D3D12_RESOURCE_STATES image_state,
                            ID3D12Resource* real, D3D12_RESOURCE_STATES real_state, ID3D12GraphicsCommandList* list,
                            ID3D12Resource* hudless, D3D12_RESOURCE_STATES hudless_state) {
    std::lock_guard lock(mutex_);
    // (diagnostics: why an image is not taken)
    auto skip = [&](GenSkip reason) { push_event(shared_, kEvFrameGen, frame, 0x300 | (static_cast<std::uint64_t>(reason) << 16)); };
    if (!shared_) return;
    if (!generation_wanted()) return skip(kGenSkipNotWanted);
    // The game is calling its frame generation and its images are wanted: the presents stop publishing frames
    // right away - not only once an image has been taken. (FSR in Cyberpunk 2077 generates a frame's images
    // after that frame's present: with the presents publishing, the frame was already gone each time, no
    // image could ever be taken, and the presents never stopped.)
    generated_qpc_ = qpc_now();
    if (!ready() || !image || !list || !fence_ || !ensure_markers()) return skip(kGenSkipNotReady);
    if (!frame || slot_for_frame(frame, false) < 0) {
        // The frame whose present the game started last - or the next one: frame generation for a frame may
        // also run just before the game's present marker for it (FSR), so when the frame the marker names has
        // all its images already (or is published), these are the next frame's.
        frame = pending_present_frame_;
        const int named = slot_for_frame(frame, false);
        if (named < 0 || (shared_->slots[named].gen_last && shared_->slots[named].gen_last >= shared_->slots[named].per_frame)) ++frame;
    }
    const int slot = slot_for_frame(frame, false);
    if (slot < 0) return skip(kGenSkipNoSlot);
    auto& m = shared_->slots[slot];
    if (!m.camera.valid) return skip(kGenSkipNoCamera);
    per_frame = std::max<std::uint32_t>(per_frame, 1);
    if (!index) index = m.gen_last + 1;
    if (index > per_frame || index <= m.gen_last) return skip(kGenSkipOrder);
    m.per_frame = per_frame;
    m.gen_last = index;
    const bool last = index == per_frame;
    // More than kMaxGenerated per frame (6x: 5): the first, the last and evenly in between (1, 3, 5).
    bool wanted = per_frame <= static_cast<std::uint32_t>(kMaxGenerated);
    constexpr std::uint32_t gaps = kMaxGenerated - 1;
    for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(kMaxGenerated) && !wanted; ++k)
        wanted = index == 1 + (2 * k * (per_frame - 1) + gaps) / (2 * gaps);
    const bool take = wanted && m.generated < static_cast<std::uint32_t>(kMaxGenerated);
    if (!take && !(last && m.generated)) return skip(kGenSkipNotKept);
    ComPtr<ID3D12GraphicsCommandList2> list2;
    if (FAILED(list->QueryInterface(IID_PPV_ARGS(&list2)))) return skip(kGenSkipNoList2);
    // (a compute list - frame generation on async compute - cannot leave graphics-only states)
    const auto graphics_only = D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_DEPTH_WRITE |
                               D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_INDEX_BUFFER;
    const bool compute = list->GetType() == D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (compute && (image_state & graphics_only)) { ++shared_->hooks.copies_failed; return skip(kGenSkipComputeState); }
    if (compute && (real_state & graphics_only)) real = nullptr;
    if (compute && (hudless_state & graphics_only)) hudless = nullptr;
    collect_retired();
    auto set_info = [&](Tex k) {
        const auto& t = textures_[slot][k];
        auto& info = m.tex[k];
        info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
        info.valid = 1;
    };
    if (take) {
        const Tex kind = static_cast<Tex>(kGen0 + m.generated);
        const auto desc = image->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 || !ensure_texture(slot, kind, desc)) {
            ++shared_->hooks.copies_failed; return skip(kGenSkipTexture);
        }
        copy_into(list, image, image_state, textures_[slot][kind].resource.Get());
        set_info(kind);
    }
    if (take && m.generated == 0 && real) {
        const auto rd = real->GetDesc();
        if (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && rd.SampleDesc.Count == 1 && ensure_texture(slot, kBackbuffer, rd)) {
            copy_into(list, real, real_state, textures_[slot][kBackbuffer].resource.Get());
            set_info(kBackbuffer);
            m.qpc_present = qpc_now();
        }
        if (hudless && !m.tex[kHudless].valid) {
            const auto hd = hudless->GetDesc();
            if (hd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && hd.SampleDesc.Count == 1 && hd.Width == rd.Width && hd.Height == rd.Height &&
                ensure_texture(slot, kHudless, hd)) {
                const int timed = time_begin(list, kTHudlessSlot);
                copy_into(list, hudless, hudless_state, textures_[slot][kHudless].resource.Get());
                time_end(list, timed);
                set_info(kHudless);
            }
        }
    }
    if (take) m.gen_index[m.generated++] = index;
    generated_qpc_ = qpc_now();
    // (diagnostics: 0x100 | image number << 16 | per frame << 24 | taken << 32)
    push_event(shared_, kEvFrameGen, frame, 0x100 | (static_cast<std::uint64_t>(index) << 16) | (static_cast<std::uint64_t>(per_frame) << 24) |
                                                (static_cast<std::uint64_t>(take) << 32));
    if (last && m.generated && m.tex[kBackbuffer].valid) {
        const std::uint32_t marker = ++marker_counter_ ? marker_counter_ : ++marker_counter_;  // (never 0: the buffer starts at 0)
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER write{markers_->GetGPUVirtualAddress() + (marker % kMarkers) * 4, marker};
        const D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
        list2->WriteBufferImmediate(1, &write, &mode);
        pending_publish_.push_back({frame, marker, qpc_now()});
        pending_count_ = static_cast<int>(pending_publish_.size());
        SetEvent(marker_event_);
    }
}

bool Producer::attach_vulkan(reshade::api::device* device) {
    if (!shared_ || !device) return false;
    const auto vk_device = reinterpret_cast<VkDevice>(device->get_native());
    if (vk_ && vk_->device() == vk_device) return true;
    detach_vulkan();
    // A D3D12 device of our own on the game's GPU creates what the presenter opens.
    LUID luid{};
    if (!device->get_property(reshade::api::device_properties::adapter_luid, &luid)) { set_message("Vulkan: adapter LUID unknown"); return false; }
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device> own;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) || FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&own)))) {
        set_message("Vulkan: no D3D12 device on the game's GPU"); return false;
    }
    if (!attach(own.Get())) return false;
    auto vk = std::make_unique<VkTransport>();
    std::string error;
    if (!vk->init(device, error)) { set_message(error.c_str()); return false; }
    std::lock_guard lock(mutex_);
    own_device_ = own;
    vk->set_on_complete([this](std::uint64_t value) { if (fence_) fence_->Signal(value); });
    vk_ = std::move(vk);
    return true;
}

void Producer::detach_vulkan() {
    std::unique_ptr<VkTransport> vk;
    {
        std::lock_guard lock(mutex_);
        if (!vk_) return;
        release_textures();  // (their Vulkan images belong to the device going away)
        for (auto& m : shared_->slots) { m.state = kFree; m.frame_id = 0; }
        vk = std::move(vk_);
    }
    vk.reset();  // stops the completion thread (outside the lock: it signals through fence_)
}

void Producer::on_tag_vk(std::uint64_t frame, Tex kind, std::uint64_t image, std::uint32_t vk_format, std::uint32_t aspect, std::uint32_t mip,
                         std::uint32_t layer, std::uint32_t layout, std::uint32_t width, std::uint32_t height, std::uint32_t ext_x,
                         std::uint32_t ext_y, std::uint32_t ext_w, std::uint32_t ext_h, void* command_buffer) {
    std::lock_guard lock(mutex_);
    if (!ready() || !vk_ || !image || !command_buffer || !width || !height) return;
    if (kind == kScene && hud_given_ >= 60) return;  // (as on_tag: not while the game hands its HUD over)
    std::uint32_t depth_bytes = 0;
    const DXGI_FORMAT format = dxgi_format_of(static_cast<VkFormat>(vk_format), &depth_bytes);
    if (format == DXGI_FORMAT_UNKNOWN) {
        char text[96]; std::snprintf(text, sizeof(text), "Vulkan: %s format %u not supported", tex_name(kind), vk_format);
        set_message(text); return;
    }
    const int slot = slot_for_frame(frame, true);
    if (slot < 0) return;
    slot_list_[slot] = reinterpret_cast<std::uint64_t>(command_buffer); slot_submitted_[slot] = false;
    collect_retired();
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height; desc.Format = format;
    if (!ensure_texture(slot, kind, desc, depth_bytes)) { ++shared_->hooks.copies_failed; return; }
    auto& t = textures_[slot][kind];
    VkImageAspectFlags barrier_aspect = aspect ? aspect : VK_IMAGE_ASPECT_COLOR_BIT;
    if (depth_bytes) barrier_aspect = VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil(static_cast<VkFormat>(vk_format)) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    vk_->copy(static_cast<VkCommandBuffer>(command_buffer), reinterpret_cast<VkImage>(image), barrier_aspect, mip, layer,
              static_cast<VkImageLayout>(layout), t.vk, width, height);
    auto& info = shared_->slots[slot].tex[kind];
    info.width = width; info.height = height;
    info.format = t.desc.Format; info.generation = t.generation;
    if (!ext_w || !ext_h) { ext_x = ext_y = 0; ext_w = width; ext_h = height; }
    info.ext_x = ext_x; info.ext_y = ext_y; info.ext_w = ext_w; info.ext_h = ext_h;
    info.valid = 1;
    push_event(shared_, kEvTag, frame, static_cast<std::uint64_t>(kind) | (image << 8));
}

std::uint64_t Producer::begin_present_vk(std::uint64_t image, DXGI_FORMAT format, std::uint32_t width, std::uint32_t height, void* command_buffer) {
    std::lock_guard lock(mutex_);
    if (!ready() || !vk_ || !image || !command_buffer) return 0;
    const int slot = present_slot();
    if (slot < 0) return 0;
    auto& m = shared_->slots[slot];
    // Vulkan swapchains made format-mutable (as ReShade does) are reported typeless: share the plain format.
    switch (format) {
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: format = DXGI_FORMAT_B8G8R8A8_UNORM; break;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: format = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: format = DXGI_FORMAT_R10G10B10A2_UNORM; break;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: format = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
        default: break;
    }
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height; desc.Format = format;
    if (vk_format_of(format) == VK_FORMAT_UNDEFINED) {
        char text[96]; std::snprintf(text, sizeof(text), "Vulkan: backbuffer format %d not supported", static_cast<int>(format));
        set_message(text);
    } else if (ensure_texture(slot, kBackbuffer, desc)) {
        auto& t = textures_[slot][kBackbuffer];
        vk_->copy(static_cast<VkCommandBuffer>(command_buffer), reinterpret_cast<VkImage>(image), VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, t.vk, width, height);
        auto& info = m.tex[kBackbuffer];
        info.width = width; info.height = height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = width; info.ext_h = height;
        info.valid = 1;
    }
    // (the HUD layer taken, see on_ui_layer_vk: into the frame's UI slot, unless the game tags one itself)
    if (ui_vk_.resource && ui_vk_age_ <= 1 && !m.tex[kUi].valid && ensure_texture(slot, kUi, ui_vk_.desc)) {
        auto& t = textures_[slot][kUi];
        vk_->copy(static_cast<VkCommandBuffer>(command_buffer), reinterpret_cast<VkImage>(ui_vk_.vk.image), VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                  VK_IMAGE_LAYOUT_GENERAL, t.vk, static_cast<std::uint32_t>(ui_vk_.desc.Width), ui_vk_.desc.Height);
        auto& info = m.tex[kUi];
        info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
        info.valid = 1;
        ++shared_->hooks.ui_layer_handed;
    }
    if (ui_vk_age_ < 2) ++ui_vk_age_;
    // (the HUD-less picture taken this frame, see on_hudless_vk: into the frame's HUD-less slot, unless it has a HUD layer)
    if (hudless_vk_fresh_ && !m.tex[kUi].valid && !m.tex[kHudless].valid && ensure_texture(slot, kHudless, hudless_vk_.desc)) {
        auto& t = textures_[slot][kHudless];
        vk_->copy(static_cast<VkCommandBuffer>(command_buffer), reinterpret_cast<VkImage>(hudless_vk_.vk.image), VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                  VK_IMAGE_LAYOUT_GENERAL, t.vk, static_cast<std::uint32_t>(hudless_vk_.desc.Width), hudless_vk_.desc.Height);
        auto& info = m.tex[kHudless];
        info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
        info.valid = 1;
    }
    hudless_vk_fresh_ = false;
    // (frames in a row with the HUD handed over, as in begin_present: the upscaler's output no longer copied past 60, its
    // slot textures released at 300. The UI slot here only ever holds the layer the add-on took - its check against the
    // upscaler's output needs that until the presenter trusts it: DOOM Eternal)
    hud_given_ = m.tex[kHudless].valid || shared_->presenter.layer_trust == 1 ? hud_given_ + 1 : 0;
    if (hud_given_ == 300)
        for (int s = 0; s < kSlots; ++s) {
            auto& t = textures_[s][kScene];
            if (!t.resource) continue;
            retired_.push_back({t.resource, t.handle, fence_value_ + 1, t.vk, t.d11});
            const std::uint32_t generation = t.generation;
            t = Texture{};
            t.generation = generation;
        }
    m.qpc_present = qpc_now();
    return m.frame_id + 1;  // 0 means "nothing to finish"
}

bool Producer::own_device_for(reshade::api::device* device, const char* api, ComPtr<ID3D12Device>& own) {
    LUID luid{};
    char text[96];
    if (!device->get_property(reshade::api::device_properties::adapter_luid, &luid)) {
        std::snprintf(text, sizeof(text), "%s: adapter LUID unknown", api); set_message(text); return false;
    }
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) || FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&own)))) {
        std::snprintf(text, sizeof(text), "%s: no D3D12 device on the game's GPU", api); set_message(text); return false;
    }
    return true;
}

bool Producer::attach_d3d11(reshade::api::device* device) {
    if (!shared_ || !device) return false;
    ComPtr<ID3D11Device5> game;
    if (FAILED(reinterpret_cast<ID3D11Device*>(device->get_native())->QueryInterface(IID_PPV_ARGS(&game)))) {
        set_message("D3D11: the game's device has no shared fences (Direct3D 11.4, Windows 10 1703 or later)");
        return false;
    }
    if (d11_ && d11_.Get() == game.Get()) return true;
    detach_d3d11();
    // A D3D12 device of our own on the game's GPU creates what the presenter opens; the game's D3D11 device
    // opens the same textures and fence.
    ComPtr<ID3D12Device> own;
    if (!own_device_for(device, "D3D11", own) || !attach(own.Get())) return false;
    std::lock_guard lock(mutex_);
    ComPtr<ID3D11Fence> fence;
    if (FAILED(game->OpenSharedFence(fence_handle_, IID_PPV_ARGS(&fence)))) { set_message("D3D11: opening the shared fence failed"); return false; }
    own_device_ = own;
    d11_ = game;
    d11_fence_ = fence;
    return true;
}

void Producer::detach_d3d11() {
    std::lock_guard lock(mutex_);
    if (!d11_) return;
    release_textures();  // (their D3D11 views belong to the device going away)
    for (auto& m : shared_->slots) { m.state = kFree; m.frame_id = 0; }
    d11_fence_.Reset();
    d11_.Reset();
}

void Producer::copy_d3d11(ID3D11DeviceContext* context, ID3D11Resource* source, ID3D11Texture2D* target) {
    ComPtr<ID3D11Texture2D> texture;
    D3D11_TEXTURE2D_DESC desc{};
    if (SUCCEEDED(source->QueryInterface(IID_PPV_ARGS(&texture)))) texture->GetDesc(&desc);
    if (desc.SampleDesc.Count > 1) {
        // (a multisampled swapchain, as older games have: resolved, which needs a typed format)
        DXGI_FORMAT typed = desc.Format;
        switch (typed) {
            case DXGI_FORMAT_R8G8B8A8_TYPELESS: typed = DXGI_FORMAT_R8G8B8A8_UNORM; break;
            case DXGI_FORMAT_B8G8R8A8_TYPELESS: typed = DXGI_FORMAT_B8G8R8A8_UNORM; break;
            case DXGI_FORMAT_R10G10B10A2_TYPELESS: typed = DXGI_FORMAT_R10G10B10A2_UNORM; break;
            case DXGI_FORMAT_R16G16B16A16_TYPELESS: typed = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
            default: break;
        }
        context->ResolveSubresource(target, 0, source, 0, typed);
    } else {
        context->CopySubresourceRegion(target, 0, 0, 0, 0, source, 0, nullptr);
    }
}

void Producer::on_tag_d3d11(std::uint64_t frame, Tex kind, ID3D11Resource* source, std::uint32_t ext_x, std::uint32_t ext_y,
                            std::uint32_t ext_w, std::uint32_t ext_h, ID3D11DeviceContext* context) {
    std::lock_guard lock(mutex_);
    if (!ready() || !d11_ || !source || !context) return;
    ComPtr<ID3D11Texture2D> texture;
    if (FAILED(source->QueryInterface(IID_PPV_ARGS(&texture)))) return;
    D3D11_TEXTURE2D_DESC sd{};
    texture->GetDesc(&sd);
    if (sd.SampleDesc.Count != 1) return;
    const int slot = slot_for_frame(frame, true);
    if (slot < 0) return;
    slot_list_[slot] = 0; slot_submitted_[slot] = true;  // (the immediate context: on the GPU's way at once)
    collect_retired();
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = sd.Width; desc.Height = sd.Height; desc.Format = sd.Format;
    if (!ensure_texture(slot, kind, desc)) { ++shared_->hooks.copies_failed; return; }
    auto& t = textures_[slot][kind];
    copy_d3d11(context, source, t.d11.Get());
    auto& info = shared_->slots[slot].tex[kind];
    info.width = sd.Width; info.height = sd.Height;
    info.format = t.desc.Format; info.generation = t.generation;
    if (!ext_w || !ext_h) { ext_x = ext_y = 0; ext_w = info.width; ext_h = info.height; }
    info.ext_x = ext_x; info.ext_y = ext_y; info.ext_w = ext_w; info.ext_h = ext_h;
    info.valid = 1;
    push_event(shared_, kEvTag, frame, static_cast<std::uint64_t>(kind) | (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(source)) << 8));
}

std::uint64_t Producer::begin_present_d3d11(ID3D11Resource* backbuffer, ID3D11DeviceContext* context) {
    std::lock_guard lock(mutex_);
    if (!ready() || !d11_ || !backbuffer || !context) return 0;
    const int slot = present_slot();
    if (slot < 0) return 0;
    auto& m = shared_->slots[slot];
    ComPtr<ID3D11Texture2D> texture;
    D3D11_TEXTURE2D_DESC sd{};
    if (SUCCEEDED(backbuffer->QueryInterface(IID_PPV_ARGS(&texture)))) texture->GetDesc(&sd);
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = sd.Width; desc.Height = sd.Height; desc.Format = sd.Format;
    // (swapchains made format-mutable are typeless: share the plain format)
    switch (desc.Format) {
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; break;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM; break;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
        default: break;
    }
    if (sd.Width && sd.Height && ensure_texture(slot, kBackbuffer, desc)) {
        auto& t = textures_[slot][kBackbuffer];
        copy_d3d11(context, backbuffer, t.d11.Get());
        auto& info = m.tex[kBackbuffer];
        info.width = sd.Width; info.height = sd.Height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = sd.Width; info.ext_h = sd.Height;
        info.valid = 1;
    }
    d3d11_cover(context, slot);  // (third pattern: the HUD, if a copy before it was taken this frame)
    d3d11_probe(context, slot);  // (HUD learning: the copy at this frame's candidate point)
    d3d11_learned(context, slot);  // (...the learned point, where nothing else gave the HUD)
    m.qpc_present = qpc_now();
    return m.frame_id + 1;  // 0 means "nothing to finish"
}

void Producer::finish_present_d3d11(ID3D11DeviceContext* context, std::uint64_t token) {
    std::lock_guard lock(mutex_);
    if (!token || !context || !d11_fence_) return;
    const std::uint64_t frame = token - 1;
    const int slot = slot_for_frame(frame, false);
    if (slot < 0) return;
    ComPtr<ID3D11DeviceContext4> context4;
    if (FAILED(context->QueryInterface(IID_PPV_ARGS(&context4))) || FAILED(context4->Signal(d11_fence_.Get(), ++fence_value_))) return;
    auto& m = shared_->slots[slot];
    m.fence_value = fence_value_;
    InterlockedExchange(&m.state, kReady);
    InterlockedExchange64(&shared_->latest_ready_frame, static_cast<LONG64>(frame));
    ++shared_->hooks.frames_published;
    push_event(shared_, kEvPublished, frame, fence_value_);
}

void Producer::finish_present_vk(reshade::api::command_queue* queue, std::uint64_t token) {
    std::lock_guard lock(mutex_);
    if (!token || !queue || !fence_ || !vk_) return;
    const std::uint64_t frame = token - 1;
    const int slot = slot_for_frame(frame, false);
    if (slot < 0) return;
    // The GPU signals the timeline semaphore after the copies; the completion thread then signals the
    // shared D3D12 fence to the same value, which the presenter checks before taking the frame.
    vk_->signal(queue, ++fence_value_);
    auto& m = shared_->slots[slot];
    m.fence_value = fence_value_;
    InterlockedExchange(&m.state, kReady);
    InterlockedExchange64(&shared_->latest_ready_frame, static_cast<LONG64>(frame));
    ++shared_->hooks.frames_published;
    push_event(shared_, kEvPublished, frame, fence_value_);
}

}  // namespace fw
