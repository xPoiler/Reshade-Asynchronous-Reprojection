#include "addon/producer.hpp"
#include "shared/camera_motion.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace fw {
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

Producer::Producer() {
    const DWORD pid = GetCurrentProcessId();
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
    auto& s = shared_->settings;
    s.enabled = 1; s.use_mouse = 1; s.use_ui_tags = 1;
    s.rotation_extrapolation = 1.0f; s.translation_extrapolation = 1.0f;
    s.orbit_distance = 0.0f; s.max_horizon_ms = 100.0f;
    s.prediction_ms = -16.0f;  // manual value, used when auto is off
    s.present_lead_ms = 6.0f;
    s.no_warp_mask = 1;   // HUD detection in games without HUD layers
    s.keep_attached = 1;  // character / first-person weapon detection, every game
    s.auto_prediction = 3;     // default: 1/2 game frame, 1/4 in games without HUD layers (shorter warps hide mask misses); 1: full, 2: half, 4: quarter
    s.manual_gain_x = s.manual_gain_y = 0.0f; s.manual_delay_ms = 0.0f;
}

Producer::~Producer() {
    for (auto& slot : textures_)
        for (auto& t : slot) if (t.handle) CloseHandle(t.handle);
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
        for (auto& slot : textures_)
            for (auto& t : slot) { if (t.handle) CloseHandle(t.handle); t = Texture{}; }
        for (auto& r : retired_) if (r.handle) CloseHandle(r.handle);
        retired_.clear();
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
        m.camera.valid = 0;
        for (auto& t : m.tex) t.valid = 0;
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
        if (it->after_fence <= done) { if (it->handle) CloseHandle(it->handle); it = retired_.erase(it); }
        else ++it;
    }
}

bool Producer::ensure_texture(int slot, Tex kind, const D3D12_RESOURCE_DESC& source) {
    auto& t = textures_[slot][kind];
    const DXGI_FORMAT format = copy_format(source.Format);
    if (t.resource && t.desc.Width == source.Width && t.desc.Height == source.Height && t.desc.Format == format) return true;
    if (t.resource) retired_.push_back({t.resource, t.handle, fence_value_ + 1});
    const std::uint32_t generation = t.generation + 1;
    t = Texture{};
    t.generation = generation;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = source.Width; desc.Height = source.Height; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = format; desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
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

void Producer::on_present_marker(std::uint64_t frame) {
    std::lock_guard lock(mutex_);
    pending_present_frame_ = frame;
    have_present_marker_ = true;
    push_event(shared_, kEvPresentMarker, frame);
}

void Producer::on_tag(std::uint64_t frame, Tex kind, ID3D12Resource* source, D3D12_RESOURCE_STATES state,
                      std::uint32_t ext_x, std::uint32_t ext_y, std::uint32_t ext_w, std::uint32_t ext_h,
                      ID3D12GraphicsCommandList* list) {
    std::lock_guard lock(mutex_);
    if (!ready() || !source || !list) return;
    const auto desc = source->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1) return;
    const int slot = slot_for_frame(frame, true);
    if (slot < 0) return;
    collect_retired();
    if (!ensure_texture(slot, kind, desc)) { ++shared_->hooks.copies_failed; return; }
    auto& t = textures_[slot][kind];
    copy_into(list, source, state, t.resource.Get());
    auto& info = shared_->slots[slot].tex[kind];
    info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height;
    info.format = t.desc.Format; info.generation = t.generation;
    if (!ext_w || !ext_h) { ext_x = ext_y = 0; ext_w = info.width; ext_h = info.height; }
    info.ext_x = ext_x; info.ext_y = ext_y; info.ext_w = ext_w; info.ext_h = ext_h;
    info.valid = 1;
    push_event(shared_, kEvTag, frame, static_cast<std::uint64_t>(kind));
}

std::uint64_t Producer::begin_present(ID3D12Resource* backbuffer, ID3D12GraphicsCommandList* list) {
    std::lock_guard lock(mutex_);
    if (!ready() || !backbuffer || !list) return 0;
    InterlockedIncrement(&shared_->presents_total);
    // Which frame is being presented: the PCL present-start marker when the game sends it,
    // otherwise the oldest frame still being written (frames present in submission order).
    int slot = -1;
    std::uint64_t via = 0;  // 1: present marker, 2: oldest-frame fallback
    if (have_present_marker_) { slot = slot_for_frame(pending_present_frame_, false); if (slot >= 0) via = 1; }
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
    if (slot < 0) { InterlockedIncrement(&shared_->presents_without_frame); return 0; }
    InterlockedExchange(&shared_->presents_without_frame, 0);
    auto& m = shared_->slots[slot];
    // Frames older than the one being presented will never be presented: release them.
    for (int i = 0; i < kSlots; ++i)
        if (i != slot && shared_->slots[i].state == kWriting && shared_->slots[i].frame_id < m.frame_id)
            InterlockedExchange(&shared_->slots[i].state, kFree);
    collect_retired();
    const auto desc = backbuffer->GetDesc();
    if (ensure_texture(slot, kBackbuffer, desc)) {
        auto& t = textures_[slot][kBackbuffer];
        copy_into(list, backbuffer, D3D12_RESOURCE_STATE_PRESENT, t.resource.Get());
        auto& info = m.tex[kBackbuffer];
        info.width = static_cast<std::uint32_t>(t.desc.Width); info.height = t.desc.Height; info.format = t.desc.Format;
        info.generation = t.generation; info.ext_x = info.ext_y = 0; info.ext_w = info.width; info.ext_h = info.height;
        info.valid = 1;
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
    if (FAILED(queue->Signal(fence_.Get(), ++fence_value_))) return;
    auto& m = shared_->slots[slot];
    m.fence_value = fence_value_;
    InterlockedExchange(&m.state, kReady);
    InterlockedExchange64(&shared_->latest_ready_frame, static_cast<LONG64>(frame));
    ++shared_->hooks.frames_published;
    push_event(shared_, kEvPublished, frame, fence_value_);
}

}  // namespace fw
