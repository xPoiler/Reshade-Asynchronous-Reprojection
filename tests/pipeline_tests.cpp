// End-to-end GPU test of the transport + Latewarp path without a game:
// "game" device A renders a synthetic frame and publishes it through fw::Producer exactly like the
// add-on does (planar D32S8 depth at render resolution, RGBA8 backbuffer); device B runs the
// presenter's Renderer + Latewarp12 and we check the warped marker moves by the expected amount.
#include "addon/producer.hpp"
#include "presenter/pose.hpp"
#include "presenter/renderer.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

using namespace fw;
using Microsoft::WRL::ComPtr;

static int failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31, mant = h & 1023;
    if (exp == 0) return (sign ? -1.f : 1.f) * std::ldexp(float(mant), -24);
    if (exp == 31) return 0.f;
    return (sign ? -1.f : 1.f) * std::ldexp(float(mant + 1024), int(exp) - 25);
}

// Column / row where the marker channel peaks (averaged over the central band).
static double peak(const std::vector<std::uint16_t>& px, std::uint32_t w, std::uint32_t h, bool columns, int channel) {
    double best = -1, pos = -1;
    const std::uint32_t n = columns ? w : h;
    for (std::uint32_t i = 0; i < n; ++i) {
        double sum = 0;
        for (std::uint32_t j = (columns ? h : w) * 2 / 5; j < (columns ? h : w) * 3 / 5; ++j) {
            const std::size_t idx = columns ? (std::size_t(j) * w + i) : (std::size_t(i) * w + j);
            const float r = half_to_float(px[idx * 4 + 0]), g = half_to_float(px[idx * 4 + 1]), b = half_to_float(px[idx * 4 + 2]);
            sum += channel == 0 ? (r - g) : (g + b + r) / 3.0;  // red bar vs white bar
        }
        if (sum > best) { best = sum; pos = i; }
    }
    return pos;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // Optional: pipeline_tests <width> <height> [depth fraction] runs the checks at that size (depth at 75%).
    const std::uint32_t W = argc > 2 ? std::uint32_t(std::atoi(argv[1])) : 1280;
    const std::uint32_t H = argc > 2 ? std::uint32_t(std::atoi(argv[2])) : 720;
    const double depth_fraction = argc > 3 ? std::atof(argv[3]) : 0.75;
    // [colour bits]: 10 makes the game's frame R10G10B10A2 (like HDR10 / 10-bit games), otherwise RGBA8.
    const DXGI_FORMAT kColour = argc > 4 && std::atoi(argv[4]) == 10 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
    const std::uint32_t DW = std::uint32_t(W * depth_fraction), DH = std::uint32_t(H * depth_fraction);
    ComPtr<IDXGIFactory6> factory;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (d.VendorId == 0x10DE) break;
        adapter.Reset();
    }
    if (!adapter) { std::printf("SKIP: no NVIDIA adapter\n"); return 0; }
    ComPtr<ID3D12Device> game;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&game)))) { std::printf("FAIL device\n"); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue; game->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> alloc; game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ComPtr<ID3D12GraphicsCommandList> list;
    game->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));

    WNDCLASSW wc{}; wc.lpfnWndProc = DefWindowProcW; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"FwTest";
    RegisterClassW(&wc);
    HWND window = CreateWindowExW(WS_EX_NOREDIRECTIONBITMAP, wc.lpszClassName, L"FrameWarp test", WS_POPUP, 0, 0, W, H, nullptr, nullptr, wc.hInstance, nullptr);

    Producer producer;
    EXPECT(producer.attach(game.Get()), "producer attach");
    producer.set_swapchain(window, W, H, kColour, 0);

    // Synthetic game frame: dark background, white vertical bar at the center column, red horizontal bar.
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC cd{}; cd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; cd.Width = W; cd.Height = H;
    cd.DepthOrArraySize = 1; cd.MipLevels = 1; cd.Format = kColour; cd.SampleDesc.Count = 1;
    cd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    ComPtr<ID3D12Resource> backbuffer, depth;
    game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &cd, D3D12_RESOURCE_STATE_PRESENT, nullptr, IID_PPV_ARGS(&backbuffer));
    D3D12_RESOURCE_DESC dd = cd; dd.Width = DW; dd.Height = DH; dd.Format = DXGI_FORMAT_R32G8X24_TYPELESS; dd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear{}; clear.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT; clear.DepthStencil.Depth = 0.01f;
    game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(&depth));
    ComPtr<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    game->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap));
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV; game->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv_heap));
    game->CreateRenderTargetView(backbuffer.Get(), nullptr, rtv_heap->GetCPUDescriptorHandleForHeapStart());
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{}; dsv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT; dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    game->CreateDepthStencilView(depth.Get(), &dsv, dsv_heap->GetCPUDescriptorHandleForHeapStart());

    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {backbuffer.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET};
    list->ResourceBarrier(1, &b);
    const float dark[4] = {0.1f, 0.1f, 0.1f, 1}, white[4] = {1, 1, 1, 1}, red[4] = {1, 0, 0, 1};
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    list->ClearRenderTargetView(rtv, dark, 0, nullptr);
    const D3D12_RECT vbar{LONG(W / 2 - 4), 0, LONG(W / 2 + 4), LONG(H)};
    const D3D12_RECT hbar{0, LONG(H / 2 - 4), LONG(W), LONG(H / 2 + 4)};
    list->ClearRenderTargetView(rtv, red, 1, &hbar);
    list->ClearRenderTargetView(rtv, white, 1, &vbar);
    list->ClearDepthStencilView(dsv_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 0.01f, 0, 0, nullptr);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    list->ResourceBarrier(1, &b);

    // UE-style camera: fwd +X, right +Y, up +Z; reversed infinite projection, 90 deg vertical FOV.
    const float aspect = float(W) / float(H), znear = 10.0f;
    Camera cam{};
    const float proj[16] = {1.0f / aspect, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, znear, 0};
    std::memcpy(cam.view_to_clip, proj, sizeof(proj));
    cam.right[1] = 1; cam.up[2] = 1; cam.fwd[0] = 1;
    cam.near_plane = znear; cam.fov = 1.5708f; cam.aspect = aspect; cam.depth_inverted = 1;
    producer.on_constants(7, cam);
    producer.on_tag(7, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
    const auto token = producer.begin_present(backbuffer.Get(), list.Get());
    EXPECT(token != 0, "begin_present found the frame");
    list->Close();
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    producer.finish_present(queue.Get(), token);
    const Shared& sh = *producer.shared();
    int slot = -1;
    for (int i = 0; i < kSlots; ++i) if (sh.slots[i].state == kReady && sh.slots[i].frame_id == 7) slot = i;
    EXPECT(slot >= 0, "slot published");
    if (slot < 0) return 1;
    EXPECT(sh.slots[slot].tex[kDepth].format == DXGI_FORMAT_R32G8X24_TYPELESS, "depth copied in its planar typeless format");

    Renderer renderer;
    std::string error;
    if (!renderer.init(sh.adapter, window, W, H, kColour, 0, error)) { std::printf("FAIL renderer: %s\n", error.c_str()); return 1; }
    std::printf("presenter queue priority: %s\n", renderer.queue_priority());
    EXPECT(renderer.open_session(sh.producer_pid, sh.session, error), "open session: %s", error.c_str());
    Latewarp12 latewarp;
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const auto dir = std::filesystem::path(exe).parent_path();
    if (!latewarp.initialize(renderer.device(), dir, dir / L"logs")) { std::printf("FAIL latewarp: %s\n", latewarp.status().c_str()); return 1; }

    const CameraBasis source{{0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
    Mat4 projection; std::memcpy(projection.data(), proj, sizeof(proj));
    auto run = [&](bool first, double yaw, double pitch, std::vector<std::uint16_t>& px, std::uint32_t& w, std::uint32_t& h, double side = 0.0) {
        auto* l = renderer.begin_frame();
        IngestedSource src{};
        if (first) src = renderer.ingest(sh, slot);
        static IngestedSource kept; if (first) kept = src;
        EXPECT(kept.valid && kept.has_depth, "ingest valid=%d depth=%d", kept.valid, kept.has_depth);
        CameraBasis target = apply_rotation(source, {0, 0, 1}, yaw, pitch, source.pos);
        target.pos = target.pos + target.right * side;  // sideways move: parallax depends on depth
        auto inputs = renderer.latewarp_inputs(kept, true);
        inputs.depth_inverted = true;
        const bool ok = latewarp.evaluate(l, inputs, first, view_matrix(target, {}), view_matrix(source, {}), projection);
        EXPECT(ok, "evaluate: %s", latewarp.status().c_str());
        renderer.finish_frame(ok, 0);
        renderer.read_back(true, px, w, h);
    };
    std::vector<std::uint16_t> px; std::uint32_t w = 0, h = 0;
    run(true, 0, 0, px, w, h);
    const double x0 = peak(px, w, h, true, 1), y0 = peak(px, w, h, false, 0);
    std::printf("identity: bar x=%.0f, red bar y=%.0f\n", x0, y0);
    EXPECT(std::fabs(x0 - W / 2.0) <= 5 && std::fabs(y0 - H / 2.0) <= 5, "identity keeps markers centred");

    const double yaw = 5.0 * 3.14159265 / 180.0;
    const double expected = std::tan(yaw) / aspect * (W / 2.0);
    run(false, yaw, 0, px, w, h);
    const double x1 = peak(px, w, h, true, 1);
    std::printf("yaw +5 deg: bar x=%.0f (shift %.1f px, |expected| %.1f)\n", x1, x1 - x0, expected);
    EXPECT(std::fabs(std::fabs(x1 - x0) - expected) < 4.0, "yaw shift magnitude");
    // Turning right (towards +right) must move the scene left.
    EXPECT(x1 < x0, "yaw direction: scene moves opposite to the turn");

    run(false, 0, yaw, px, w, h);
    const double y1 = peak(px, w, h, false, 0);
    const double expected_y = std::tan(yaw) * (H / 2.0);
    std::printf("pitch +5 deg: red bar y=%.0f (shift %.1f px, |expected| %.1f)\n", y1, y1 - y0, expected_y);
    EXPECT(std::fabs(std::fabs(y1 - y0) - expected_y) < 4.0, "pitch shift magnitude");
    EXPECT(y1 > y0, "looking up moves the scene down");

    // FrameWarp's own warp engine: the same yaw and pitch shifts as Latewarp, and the scene edge revealed
    // by the turn is filled (no black band).
    {
        auto run_own = [&](double yaw_, double pitch_, double side_ = 0.0) {
            renderer.begin_frame();
            const IngestedSource src = renderer.ingest(sh, slot);
            CameraBasis target = apply_rotation(source, {0, 0, 1}, yaw_, pitch_, source.pos);
            target.pos = target.pos + target.right * side_;
            const Mat4 m = clip_source_to_target(projection, view_matrix(source, {}), view_matrix(target, {}));
            const bool ok = renderer.own_warp(src, true, false, m.data(), true);
            EXPECT(ok, "own warp ran");
            renderer.finish_frame(ok, 0);
            renderer.read_back(true, px, w, h);
        };
        run_own(0, 0);
        const double ox0 = peak(px, w, h, true, 1), oy0 = peak(px, w, h, false, 0);
        run_own(yaw, 0);
        const double ox1 = peak(px, w, h, true, 1);
        // The revealed right edge (turning right) must not be black: the dark background is 0.1.
        double edge = 0;
        for (std::uint32_t y = 0; y < h; y += 8) edge += half_to_float(px[(std::size_t(y) * w + (w - 2)) * 4 + 2]);
        edge /= double((h + 7) / 8);
        run_own(0, yaw);
        const double oy1 = peak(px, w, h, false, 0);
        std::printf("own warp engine: yaw bar x %.0f -> %.0f (shift %.1f, Latewarp %.1f); pitch red bar y %.0f -> %.0f (shift %.1f, Latewarp %.1f); "
                    "revealed edge brightness %.2f\n", ox0, ox1, ox1 - ox0, x1 - x0, oy0, oy1, oy1 - oy0, y1 - y0, edge);
        EXPECT(std::fabs(ox0 - x0) <= 1.0 && std::fabs(oy0 - y0) <= 1.0, "own engine: identity matches");
        EXPECT(std::fabs((ox1 - ox0) - (x1 - x0)) <= 1.5, "own engine: yaw shift matches Latewarp");
        EXPECT(std::fabs((oy1 - oy0) - (y1 - y0)) <= 1.5, "own engine: pitch shift matches Latewarp");
        EXPECT(edge > 0.05, "own engine: the revealed edge is filled (%.2f)", edge);
        // Moving sideways: the shift depends on depth (parallax), like Latewarp's.
        run_own(0, 0, 50.0);
        const double oxs = peak(px, w, h, true, 1);
        // The bar is 1000 units away (depth 0.01, near plane 10): moving 50 units right shifts it left by
        // (50 / 1000) / aspect * W/2 px - what Latewarp does in the sideways test below.
        const double own_expected = -(50.0 / 1000.0) / aspect * (W / 2.0);
        std::printf("own warp engine: sideways move 50 units: bar x %.0f -> %.0f (shift %.1f, expected %.1f)\n", ox0, oxs, oxs - ox0, own_expected);
        EXPECT(std::fabs((oxs - ox0) - own_expected) <= 1.5, "own engine: sideways parallax uses depth");
        // GPU time of a presented frame with the own engine vs Latewarp (steady state, no new source).
        renderer.begin_frame();
        const IngestedSource steady_src = renderer.ingest(sh, slot);
        renderer.finish_frame(false, 0);
        double own_ms = 0, late_ms = 0;
        for (int i = 0; i < 40; ++i) {
            auto* l = renderer.begin_frame();
            const CameraBasis target = apply_rotation(source, {0, 0, 1}, 0.001 * i, 0, source.pos);
            if (i & 1) {
                const Mat4 m = clip_source_to_target(projection, view_matrix(source, {}), view_matrix(target, {}));
                renderer.own_warp(steady_src, true, false, m.data(), true);
            } else {
                auto in = renderer.latewarp_inputs(steady_src, true);
                in.depth_inverted = true;
                latewarp.evaluate(l, in, false, view_matrix(target, {}), view_matrix(source, {}), projection);
            }
            renderer.finish_frame(true, 0);
            renderer.wait_idle();
            // The frame timestamps are read back when the frame slot is reused (3 frames later).
            if (i >= 10) ((i - 3) & 1 ? own_ms : late_ms) += renderer.last_gpu_ms();
        }
        std::printf("GPU per presented frame at %ux%u: own engine %.3f ms, Latewarp %.3f ms\n", W, H, own_ms / 15, late_ms / 15);
    }

    // A "rendered frame" evaluation (first use of a new source) must warp exactly like the others:
    // the presenter's first output after every new game frame is such an evaluation.
    run(true, yaw, 0, px, w, h);
    const double xr = peak(px, w, h, true, 1);
    run(false, yaw, 0, px, w, h);
    const double xr2 = peak(px, w, h, true, 1);
    run(false, yaw, 0, px, w, h);
    const double xr3 = peak(px, w, h, true, 1);
    std::printf("yaw +5 deg with IsRenderedFrame=1: bar x=%.0f, then same pose again: %.0f, %.0f\n", xr, xr2, xr3);
    EXPECT(std::fabs(xr - x1) < 3.0, "rendered-frame evaluation warps like any other (got %.0f, expected %.0f)", xr, x1);

    // Parallax: a sideways camera move shifts the scene by (move / distance). The test depth is cleared
    // to 0.01 with near plane 10 (reversed-Z infinite): distance 1000 units.
    const double side = 50.0;
    const double expected_side = (side / 1000.0) / aspect * (W / 2.0);
    run(false, 0, 0, px, w, h, side);
    const double xt = peak(px, w, h, true, 1);
    std::printf("sideways move %.0f units: bar x=%.0f (shift %.1f px, |expected| %.1f)\n", side, xt, xt - x0, expected_side);
    EXPECT(std::fabs(std::fabs(xt - x0) - expected_side) < 4.0, "translation parallax uses depth");

    // The game changes its render resolution (DLSS preset): depth arrives at a new size, possibly
    // smaller. Rotation and depth-dependent parallax must stay correct and the device must survive.
    // Sizes seen in E33: Ultra Performance 1/3, Performance 1/2, Balanced ~0.58, Quality ~0.67, both directions.
    for (const double scale : {0.5, 0.9, 0.334, 0.5, 0.58, 0.668, 0.5}) {
        ComPtr<ID3D12Fence> gf; game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gf));
        queue->Signal(gf.Get(), 1);
        while (gf->GetCompletedValue() < 1) Sleep(1);
        const UINT DW2 = UINT(W * scale), DH2 = UINT(H * scale);
        D3D12_RESOURCE_DESC dd2 = dd; dd2.Width = DW2; dd2.Height = DH2;
        ComPtr<ID3D12Resource> depth2;
        // Different content from the original depth (distance 500 instead of 1000): reading a stale depth
        // texture would show the old parallax.
        D3D12_CLEAR_VALUE clear2 = clear; clear2.DepthStencil.Depth = 0.02f;
        game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &dd2, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear2, IID_PPV_ARGS(&depth2));
        ComPtr<ID3D12DescriptorHeap> dsv2_heap;
        D3D12_DESCRIPTOR_HEAP_DESC hd2{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        game->CreateDescriptorHeap(&hd2, IID_PPV_ARGS(&dsv2_heap));
        game->CreateDepthStencilView(depth2.Get(), &dsv, dsv2_heap->GetCPUDescriptorHandleForHeapStart());
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        list->ClearDepthStencilView(dsv2_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 0.02f, 0, 0, nullptr);
        static std::uint64_t next_frame = 8;
        const std::uint64_t fid = next_frame++;
        producer.on_constants(fid, cam);
        producer.on_tag(fid, kDepth, depth2.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW2, DH2, list.Get());
        const auto token2 = producer.begin_present(backbuffer.Get(), list.Get());
        list->Close();
        queue->ExecuteCommandLists(1, lists);
        producer.finish_present(queue.Get(), token2);
        queue->Signal(gf.Get(), 2);
        while (gf->GetCompletedValue() < 2) Sleep(1);
        for (int i = 0; i < kSlots; ++i) if (sh.slots[i].state == kReady && sh.slots[i].frame_id == fid) slot = i;
        EXPECT(sh.slots[slot].frame_id == fid && sh.slots[slot].tex[kDepth].width == DW2, "resized depth published");
        run(true, yaw, 0, px, w, h);
        run(false, yaw, 0, px, w, h);
        const double xs = peak(px, w, h, true, 1);
        run(false, 0, 0, px, w, h, side);
        const double xts = peak(px, w, h, true, 1);
        const HRESULT removed = renderer.device()->GetDeviceRemovedReason();
        std::printf("depth %ux%u: yaw bar x=%.0f (expected %.0f), sideways shift %.1f px (|expected| %.1f), device %s\n", DW2, DH2, xs, x1,
                    xts - x0, 2 * expected_side, removed == S_OK ? "ok" : "REMOVED");
        EXPECT(removed == S_OK, "presenter device survives a depth resolution change");
        EXPECT(std::fabs(xs - x1) < 3.0, "rotation correct after depth resolution change");
        EXPECT(std::fabs(std::fabs(xts - x0) - 2 * expected_side) < 4.0, "parallax uses the NEW depth after a resolution change");
    }

    // Moving-object extrapolation: static camera, the white bar's motion vectors say it moved 20 render
    // px to the right since the previous frame. One frame ahead it must be 20 render px further right;
    // the red bar (static) must not move.
    {
        ComPtr<ID3D12Fence> gf; game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gf));
        D3D12_RESOURCE_DESC md = cd; md.Width = DW; md.Height = DH; md.Format = DXGI_FORMAT_R16G16_FLOAT;
        ComPtr<ID3D12Resource> motion;
        game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &md, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&motion));
        ComPtr<ID3D12DescriptorHeap> mv_rtv;
        D3D12_DESCRIPTOR_HEAP_DESC mh{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        game->CreateDescriptorHeap(&mh, IID_PPV_ARGS(&mv_rtv));
        game->CreateRenderTargetView(motion.Get(), nullptr, mv_rtv->GetCPUDescriptorHandleForHeapStart());
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        const float none[4] = {0, 0, 0, 0}, moved[4] = {-20.0f / float(DW), 0, 0, 0};  // uv towards the previous frame
        const LONG bx = LONG(DW / 2);
        const D3D12_RECT bar{bx - 6, 0, bx + 6, LONG(DH)};
        list->ClearRenderTargetView(mv_rtv->GetCPUDescriptorHandleForHeapStart(), none, 0, nullptr);
        list->ClearRenderTargetView(mv_rtv->GetCPUDescriptorHandleForHeapStart(), moved, 1, &bar);
        Camera still = cam;
        for (int i = 0; i < 16; ++i) still.clip_to_prev_clip[i] = (i % 5 == 0) ? 1.0f : 0.0f;  // identity: camera did not move
        const std::uint64_t fid = 100;
        producer.on_constants(fid, still);
        producer.on_tag(fid, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
        producer.on_tag(fid, kMotion, motion.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, DW, DH, list.Get());
        const auto token3 = producer.begin_present(backbuffer.Get(), list.Get());
        list->Close();
        queue->ExecuteCommandLists(1, lists);
        producer.finish_present(queue.Get(), token3);
        queue->Signal(gf.Get(), 1);
        while (gf->GetCompletedValue() < 1) Sleep(1);
        for (int i = 0; i < kSlots; ++i) if (sh.slots[i].state == kReady && sh.slots[i].frame_id == fid) slot = i;
        EXPECT(sh.slots[slot].frame_id == fid && sh.slots[slot].tex[kMotion].valid, "frame with motion vectors published");

        auto* l = renderer.begin_frame();
        IngestedSource src = renderer.ingest(sh, slot);
        EXPECT(src.has_motion, "motion vectors ingested");
        {
            // Camera estimation input: the sampled grid returns the motion field (bar -20 px, else 0) and the
            // frame keeps recording after the mid-frame flush.
            std::vector<float> grid;
            const int kGW = int(DW / 4), kGH = 36;  // every 4th render column: samples land on the 12 px bar
            EXPECT(renderer.sample_motion(src, kGW, kGH, grid) && grid.size() == std::size_t(kGW) * kGH * 4, "motion samples read back");
            double on_bar = 0, elsewhere = 0; int nb = 0, ne = 0;
            for (int gy = 0; gy < kGH; ++gy)
                for (int gx = 0; gx < kGW; ++gx) {
                    const float* v = &grid[(gy * kGW + gx) * 4];
                    const double sx = std::floor((gx + 0.5) * DW / double(kGW)) + 0.5;
                    if (std::fabs(sx - DW / 2.0) < 5) { on_bar += v[0] * DW; ++nb; }
                    else if (std::fabs(sx - DW / 2.0) > 12) { elsewhere += std::fabs(v[0]) * DW; ++ne; }
                }
            std::printf("motion samples: bar %.2f px (expected -20), elsewhere %.2f px (%d / %d samples)\n", nb ? on_bar / nb : 0.0,
                        ne ? elsewhere / ne : 0.0, nb, ne);
            EXPECT(nb && std::fabs(on_bar / nb + 20.0) < 0.2, "sampled bar motion");
            EXPECT(ne && elsewhere / ne < 0.01, "sampled static motion");
        }
        renderer.analyze_motion(src, still.clip_to_prev_clip, 1.0f, 1.0f, true);
        auto evaluate = [&](ID3D12GraphicsCommandList* list_now, float alpha, bool rendered) {
            auto inputs = renderer.latewarp_inputs(src, true);
            inputs.depth_inverted = true;
            if (alpha != 0.0f) {
                ID3D12Resource* result = renderer.extrapolate_objects(src, false, alpha, still.clip_to_prev_clip);
                EXPECT(result != nullptr, "extrapolation ran");
                if (result) inputs.backbuffer = inputs.hudless = result;
            }
            latewarp.evaluate(list_now, inputs, rendered, view_matrix(source, {}), view_matrix(source, {}), projection);
            renderer.finish_frame(true, 0);
            renderer.read_back(true, px, w, h);
        };
        evaluate(l, 0.0f, true);
        const double still_x = peak(px, w, h, true, 1), still_red = peak(px, w, h, false, 0);
        MotionFit fit{};
        for (int i = 0; i < 4 && !renderer.take_motion_fit(fit); ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
        evaluate(renderer.begin_frame(), 1.0f, false);
        const double moved_x = peak(px, w, h, true, 1), moved_red = peak(px, w, h, false, 0);
        evaluate(renderer.begin_frame(), 0.5f, false);
        const double half_x = peak(px, w, h, true, 1);
        // Interpolation (what the presenter uses): half a frame back towards the previous position. The
        // part of the bar's current area it has left must show background, not a stretched bar.
        evaluate(renderer.begin_frame(), -0.5f, false);
        const double back_x = peak(px, w, h, true, 1);
        auto white_at = [&](double x) {
            double sum = 0; const std::uint32_t xi = std::uint32_t(x);
            for (std::uint32_t y = h * 2 / 5; y < h * 3 / 5; ++y) {
                const std::size_t i = (std::size_t(y) * w + xi) * 4;
                sum += (half_to_float(px[i]) + half_to_float(px[i + 1]) + half_to_float(px[i + 2])) / 3.0;
            }
            return sum / double(h / 5);
        };
        const double left_behind = white_at(still_x + 4.0 * double(W) / double(DW));  // inside the old bar, outside the moved one
        std::printf("interpolated -1/2 frame: bar x=%.0f (expected %.1f), brightness where it left %.2f\n", back_x,
                    still_x - 10.0 * double(W) / double(DW), left_behind);
        EXPECT(std::fabs((back_x - still_x) + 10.0 * double(W) / double(DW)) < 4.0, "interpolating back moves the object towards its previous position");
        EXPECT(left_behind < 0.5, "the area the object left shows background (%.2f)", left_behind);
        const double expected_move = 20.0 * double(W) / double(DW);
        std::printf("object extrapolation: bar x=%.0f, +1 frame x=%.0f (expected +%.1f), +1/2 frame x=%.0f; red bar y %.0f -> %.0f; moving pixels %.0f of %.0f\n",
                    still_x, moved_x, expected_move, half_x, still_red, moved_red, fit.moving, fit.samples);
        EXPECT(std::fabs((moved_x - still_x) - expected_move) < 4.0, "moving object advances one frame of its own motion");
        EXPECT(std::fabs((half_x - still_x) - expected_move / 2) < 4.0, "half a frame moves it half as far");
        EXPECT(std::fabs(moved_red - still_red) < 1.5, "static geometry stays put");
        EXPECT(fit.moving > 0.5 * 12 * DH && fit.moving < 2.0 * 12 * DH, "moving pixels = the bar (%.0f)", fit.moving);
        // GPU cost per presented frame, with and without the per-frame extrapolation passes.
        auto median_gpu = [&](float alpha) {
            std::vector<float> ms;
            for (int i = 0; i < 12; ++i) {
                auto* lf = renderer.begin_frame();
                auto inputs = renderer.latewarp_inputs(src, true);
                inputs.depth_inverted = true;
                if (alpha != 0.0f) if (ID3D12Resource* r = renderer.extrapolate_objects(src, false, alpha, still.clip_to_prev_clip)) inputs.backbuffer = inputs.hudless = r;
                latewarp.evaluate(lf, inputs, false, view_matrix(source, {}), view_matrix(source, {}), projection);
                renderer.finish_frame(true, 0);
                renderer.wait_idle();
                if (i >= 4) ms.push_back(renderer.last_gpu_ms());
            }
            std::sort(ms.begin(), ms.end());
            return ms[ms.size() / 2];
        };
        const float plain = median_gpu(0.0f), with_objects = median_gpu(0.5f);
        std::printf("GPU per presented frame: %.3f ms plain, %.3f ms with object extrapolation (+%.3f)\n", plain, with_objects, with_objects - plain);
    }

    // No-warp mask for games without HUD layers. The camera moves (uniform screen motion `shift` in uv);
    // (A) a strip whose motion vectors ignore that motion (a first-person weapon) and (B) a patch that
    // stays identical while the scene changes (HUD) must both stay put when the camera turns.
    {
        ComPtr<ID3D12Fence> gf; game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gf));
        UINT64 fence_value = 0;
        D3D12_RESOURCE_DESC md = cd; md.Width = DW; md.Height = DH; md.Format = DXGI_FORMAT_R16G16_FLOAT;
        ComPtr<ID3D12Resource> motion;
        game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &md, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&motion));
        ComPtr<ID3D12DescriptorHeap> mv_rtv;
        D3D12_DESCRIPTOR_HEAP_DESC mh{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        game->CreateDescriptorHeap(&mh, IID_PPV_ARGS(&mv_rtv));
        game->CreateRenderTargetView(motion.Get(), nullptr, mv_rtv->GetCPUDescriptorHandleForHeapStart());
        const float shift = 6.0f / float(W);  // uv the scene moved since the previous frame (6 output px)
        Camera moving_cam = cam;
        for (int i = 0; i < 16; ++i) moving_cam.clip_to_prev_clip[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        moving_cam.clip_to_prev_clip[12] = 2.0f * shift;  // row-vector: prev.x = x + 2*shift*w (clip), i.e. +shift in uv
        const LONG hx0 = LONG(W / 8), hx1 = LONG(W / 4), hy0 = LONG(H / 8), hy1 = LONG(H / 4);
        const float green[4] = {0, 1, 0, 1}, teal[4] = {0, 0.7f, 0.6f, 1};
        int hud_patch_colour = 0;
        bool horizontal_bar = false;
        bool near_wall = false;
        float weapon_mv = 0.0f;
        float strip_depth = 0.5f;  // depth of the camera-attached strip (reversed-Z: near / distance)  // the weapon strip's own motion vector (uv): 0 = stuck to the screen, else animating
        bool repeating = false;  // scenery that repeats every 6 px (= the camera motion per frame): windows, railings  // a wall close to the camera behind the HUD patch (normal motion vectors)
        int scene_offset = -1;  // textured scene (vertical grey stripes) shifting every frame; >= 0 freezes it
        // The upscaler's output (games calling DLSS directly): the same scene before tone mapping (here
        // frame = sqrt(scene / 4) per channel) and without the HUD.
        bool with_scene = false;
        // A bright yellow-white surface the tone mapping washes out towards white (as filmic tone mappers
        // do): scene (4, 4, 0.25) -> per channel (1, 1, 0.5), washed 80% towards its grey 0.964.
        bool washed_patch = false;
        const D3D12_RECT washed_rect{LONG(W * 3 / 4), LONG(H * 5 / 8), LONG(W * 3 / 4) + 64, LONG(H * 5 / 8) + 64};
        ComPtr<ID3D12Resource> scene_tex;
        ComPtr<ID3D12DescriptorHeap> scene_rtv;
        {
            D3D12_RESOURCE_DESC sd = cd; sd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &sd, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&scene_tex));
            D3D12_DESCRIPTOR_HEAP_DESC sh_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            game->CreateDescriptorHeap(&sh_desc, IID_PPV_ARGS(&scene_rtv));
            game->CreateRenderTargetView(scene_tex.Get(), nullptr, scene_rtv->GetCPUDescriptorHandleForHeapStart());
        }
        auto publish = [&](std::uint64_t fid, float bg, LONG bar_dx, bool weapon, bool hud_patch, bool near_strip = true) -> int {
            alloc->Reset();
            list->Reset(alloc.Get(), nullptr);
            list->ResourceBarrier(1, &b);  // backbuffer PRESENT -> RENDER_TARGET
            list->ClearDepthStencilView(dsv_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 0.01f, 0, 0, nullptr);
            const float back[4] = {bg, bg, bg, 1};
            list->ClearRenderTargetView(rtv, back, 0, nullptr);
            // Real scenes are textured: grey stripes that change every frame like scenery under a moving camera.
            const float grey[4] = {bg + 0.15f, bg + 0.15f, bg + 0.15f, 1};
            const LONG phase = scene_offset >= 0 ? scene_offset : LONG((fid * 5) % 16);
            for (LONG x = phase; x < LONG(W); x += 16) {
                const D3D12_RECT stripe{x, 0, x + 4, LONG(H)};
                list->ClearRenderTargetView(rtv, grey, 1, &stripe);
            }
            const D3D12_RECT bar{LONG(W / 2) - 4 + bar_dx, 0, LONG(W / 2) + 4 + bar_dx, LONG(H)};
            list->ClearRenderTargetView(rtv, white, 1, &bar);
            if (horizontal_bar) {
                const D3D12_RECT segment{LONG(W / 2 - W / 8), LONG(H * 3 / 4), LONG(W / 2 + W / 8), LONG(H * 3 / 4) + 6};
                const float yellow[4] = {1, 1, 0, 1};
                list->ClearRenderTargetView(rtv, yellow, 1, &segment);
            }
            if (washed_patch) {
                const float washed[4] = {0.971f, 0.971f, 0.871f, 1};
                list->ClearRenderTargetView(rtv, washed, 1, &washed_rect);
            }
            if (hud_patch) {  // textured like real HUD (text, icons): 2 px stripes
                // 0: opaque green; 1: teal, shifted 2 px (the content changed); 2: green at 50% over the scene
                const float blend[4] = {0.5f * bg, 0.5f * bg + 0.5f, 0.5f * bg, 1};
                const float* colour = hud_patch_colour == 1 ? teal : hud_patch_colour == 2 ? blend : green;
                for (LONG x = hx0 + (hud_patch_colour == 1 ? 2 : 0); x < hx1; x += 4) {
                    const D3D12_RECT stripe{x, hy0, std::min(x + 2, hx1), hy1};
                    list->ClearRenderTargetView(rtv, colour, 1, &stripe);
                }
            }
            if (with_scene) {
                const auto srtv = scene_rtv->GetCPUDescriptorHandleForHeapStart();
                auto lin = [](float v) { return 4.0f * v * v; };
                const float sback[4] = {lin(bg), lin(bg), lin(bg), 1}, sgrey[4] = {lin(bg + 0.15f), lin(bg + 0.15f), lin(bg + 0.15f), 1};
                const float swhite[4] = {4, 4, 4, 1};
                list->ClearRenderTargetView(srtv, sback, 0, nullptr);
                for (LONG x = phase; x < LONG(W); x += 16) {
                    const D3D12_RECT stripe{x, 0, x + 4, LONG(H)};
                    list->ClearRenderTargetView(srtv, sgrey, 1, &stripe);
                }
                list->ClearRenderTargetView(srtv, swhite, 1, &bar);
                if (washed_patch) {
                    const float swashed[4] = {4, 4, 0.25f, 1};
                    list->ClearRenderTargetView(srtv, swashed, 1, &washed_rect);
                }
            }
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            list->ResourceBarrier(1, &b);
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            const float scene_mv[4] = {shift, 0, 0, 0}, none[4] = {weapon_mv, 0, 0, 0};
            list->ClearRenderTargetView(mv_rtv->GetCPUDescriptorHandleForHeapStart(), scene_mv, 0, nullptr);
            if (weapon) {
                const LONG bx = LONG(DW / 2);
                const D3D12_RECT strip{bx - 8, 0, bx + 8, LONG(DH)};
                list->ClearRenderTargetView(mv_rtv->GetCPUDescriptorHandleForHeapStart(), none, 1, &strip);
            }
            producer.on_constants(fid, moving_cam);
            if (repeating) {  // cyan 3 px stripes, period 6 px, in the lower left: the same image every frame
                const float cyan[4] = {0, 0.8f, 0.8f, 1};
                for (LONG x = LONG(W / 16); x < LONG(W / 3); x += 6) {
                    const D3D12_RECT r{x, LONG(H * 5 / 8), x + 3, LONG(H * 7 / 8)};
                    list->ClearRenderTargetView(rtv, cyan, 1, &r);
                }
            }
            if (near_wall) {
                const D3D12_RECT wall{LONG(hx0 * DW / W) - 4, LONG(hy0 * DH / H) - 4, LONG(hx1 * DW / W) + 4, LONG(hy1 * DH / H) + 4};
                list->ClearDepthStencilView(dsv_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 1, &wall);
            }
            if (weapon) {  // reversed-Z depth = near / distance: 0.5 = a weapon 2x the near plane away; the
                           // "sky" variant is 10000x the near plane away
                const LONG bx = LONG(DW / 2);
                const D3D12_RECT strip{bx - 8, 0, bx + 8, LONG(DH)};
                list->ClearDepthStencilView(dsv_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH,
                                            near_strip ? strip_depth : 1e-4f, 0, 1, &strip);
            }
            producer.on_tag(fid, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
            producer.on_tag(fid, kMotion, motion.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, DW, DH, list.Get());
            if (with_scene) producer.on_tag(fid, kScene, scene_tex.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
            const auto t = producer.begin_present(backbuffer.Get(), list.Get());
            list->Close();
            queue->ExecuteCommandLists(1, lists);
            producer.finish_present(queue.Get(), t);
            queue->Signal(gf.Get(), ++fence_value);
            while (gf->GetCompletedValue() < fence_value) Sleep(1);
            int found = -1;
            for (int i = 0; i < kSlots; ++i) if (sh.slots[i].state == kReady && sh.slots[i].frame_id == fid) found = i;
            return found;
        };
        auto green_x = [&]() {  // centroid of the green patch
            double sum = 0, weight = 0;
            for (std::uint32_t y = std::uint32_t(hy0); y < std::uint32_t(hy1); ++y)
                for (std::uint32_t x = 0; x < w / 2; ++x) {
                    const std::size_t i = (std::size_t(y) * w + x) * 4;
                    const float g = half_to_float(px[i + 1]) - half_to_float(px[i]);
                    if (g > 0.3f) { sum += x * g; weight += g; }
                }
            return weight > 0 ? sum / weight : -1.0;
        };
        auto segment_x = [&]() {  // centroid of the yellow segment (red + green, no blue)
            double sum = 0, weight = 0;
            for (std::uint32_t y = h * 3 / 4 - 4; y < h * 3 / 4 + 10; ++y)
                for (std::uint32_t x = 0; x < w; ++x) {
                    const std::size_t i = (std::size_t(y) * w + x) * 4;
                    const float yv = std::min(half_to_float(px[i]), half_to_float(px[i + 1])) - half_to_float(px[i + 2]);
                    if (yv > 0.5f) { sum += x * yv; weight += yv; }
                }
            return weight > 0 ? sum / weight : -1.0;
        };
        // Present one frame of `s` with the mask, rendered, then turned by `yaw`.
        bool own_engine = false;  // warp with FrameWarp's own engine instead of Latewarp
        auto show = [&](const IngestedSource& s, double turn) {
            auto* lf = renderer.begin_frame();
            auto inputs = renderer.latewarp_inputs(s, true);
            inputs.depth_inverted = true;
            inputs.no_warp_mask = renderer.no_warp_mask();
            inputs.mask_rect = s.color_rect;
            const CameraBasis target = apply_rotation(source, {0, 0, 1}, turn, 0, source.pos);
            if (own_engine) {
                const Mat4 m = clip_source_to_target(projection, view_matrix(source, {}), view_matrix(target, {}));
                renderer.own_warp(s, true, inputs.no_warp_mask != nullptr, m.data(), true);
            } else {
                latewarp.evaluate(lf, inputs, turn == 0, view_matrix(target, {}), view_matrix(source, {}), projection);
            }
            renderer.finish_frame(true, 0);
            renderer.read_back(true, px, w, h);
        };
        auto ingest_frame = [&](int s_slot, bool hud, bool weapon) {
            EXPECT(s_slot >= 0, "frame published");
            if (s_slot < 0) return IngestedSource{};
            renderer.set_keep_previous_colour(true);
            renderer.begin_frame();
            IngestedSource s = renderer.ingest(sh, s_slot);
            renderer.analyze_motion(s, moving_cam.clip_to_prev_clip, 1.0f, 1.0f, true);
            renderer.build_no_warp_mask(s, moving_cam.clip_to_prev_clip, hud, weapon, weapon);
            renderer.finish_frame(false, 0);
            renderer.wait_idle();
            return s;
        };
        // (A) first-person weapon from motion vectors.
        renderer.reset_hud_detection();
        IngestedSource weapon_src = ingest_frame(publish(200, 0.1f, 0, true, false), false, true);
        show(weapon_src, 0);
        const double weapon_x0 = peak(px, w, h, true, 1);
        show(weapon_src, yaw);
        const double weapon_x1 = peak(px, w, h, true, 1);
        // (A1) a weapon mid-animation (aiming in while walking): its motion vectors show the animation, not
        // zero, and not the camera motion either. Still attached: not warped.
        weapon_mv = -5.0f * shift;
        IngestedSource anim_src = ingest_frame(publish(210, 0.1f, 0, true, false), false, true);
        weapon_mv = 0.0f;
        show(anim_src, 0);
        const double anim_x0 = peak(px, w, h, true, 1);
        show(anim_src, yaw);
        const double anim_x1 = peak(px, w, h, true, 1);
        EXPECT(std::fabs(anim_x1 - anim_x0) < 2.0, "an animating first-person weapon is not warped (%.0f -> %.0f)", anim_x0, anim_x1);
        // (A1b) a third-person character: attached to the camera too, but a few metres away (300x the near
        // plane). Held as well.
        strip_depth = 1.0f / 300.0f;
        IngestedSource third_src = ingest_frame(publish(220, 0.1f, 0, true, false), false, true);
        strip_depth = 0.5f;
        show(third_src, 0);
        const double third_x0 = peak(px, w, h, true, 1);
        show(third_src, yaw);
        const double third_x1 = peak(px, w, h, true, 1);
        EXPECT(std::fabs(third_x1 - third_x0) < 2.0, "a third-person character a few metres away is not warped (%.0f -> %.0f)", third_x0, third_x1);
        // (A1c) something at that distance moving on its own while the camera turns (a car, a person): its
        // motion is not the camera's and it is not stuck to the screen - it keeps warping.
        strip_depth = 1.0f / 300.0f;
        weapon_mv = -5.0f * shift;
        IngestedSource mover_src = ingest_frame(publish(230, 0.1f, 0, true, false), false, true);
        strip_depth = 0.5f;
        weapon_mv = 0.0f;
        show(mover_src, 0);
        const double mover_x0 = peak(px, w, h, true, 1);
        show(mover_src, yaw);
        const double mover_x1 = peak(px, w, h, true, 1);
        std::printf("attached: third-person character x %.0f -> %.0f; object moving on its own at that distance x %.0f -> %.0f\n",
                    third_x0, third_x1, mover_x0, mover_x1);
        EXPECT(std::fabs(mover_x1 - mover_x0) > 20.0, "an object moving on its own a few metres away still warps (%.0f -> %.0f)", mover_x0, mover_x1);
        // (A2) the same zero-motion strip far away (sky, 10000x the near plane) is not attached: it must warp.
        renderer.reset_hud_detection();
        IngestedSource sky_src{};
        sky_src = ingest_frame(publish(251, 0.1f, 0, true, false, false), false, true);
        show(sky_src, 0);
        const double sky_x0 = peak(px, w, h, true, 1);
        show(sky_src, yaw);
        const double sky_x1 = peak(px, w, h, true, 1);
        // (B) HUD from pixels that stay the same while the scene changes (6 frames).
        renderer.reset_hud_detection();
        IngestedSource hud_src{};
        for (int i = 0; i < 7; ++i) hud_src = ingest_frame(publish(300 + i, (i & 1) ? 0.3f : 0.1f, (i & 1) ? 12 : -12, false, true), true, false);
        show(hud_src, 0);
        const double patch_x0 = green_x(), bar_x0 = peak(px, w, h, true, 1);
        show(hud_src, yaw);
        const double patch_x1 = green_x(), bar_x1 = peak(px, w, h, true, 1);
        // (C) What was HUD changes (its content moved: scenery now): unmasked in the next frame.
        hud_patch_colour = 1;
        IngestedSource changed_src = ingest_frame(publish(310, 0.1f, 12, false, true), true, false);
        show(changed_src, 0);
        const double gone_x0 = green_x();
        show(changed_src, yaw);
        const double gone_x1 = green_x();
        hud_patch_colour = 0;
        // (D) A first-person weapon held still on screen for many frames never enters the HUD map: the
        // frame after it is gone, that area warps.
        renderer.reset_hud_detection();
        for (int i = 0; i < 7; ++i) ingest_frame(publish(400 + i, (i & 1) ? 0.3f : 0.1f, 0, true, false), true, true);
        IngestedSource after_src = ingest_frame(publish(410, 0.1f, 0, false, false), true, true);
        show(after_src, 0);
        const double after_x0 = peak(px, w, h, true, 1);
        show(after_src, yaw);
        const double after_x1 = peak(px, w, h, true, 1);
        // (F) HUD that pops up (RE9 shows it only when needed): masked two game frames after it appears
        // while the camera turns (8 px per frame here).
        renderer.reset_hud_detection();
        for (int i = 0; i < 3; ++i) ingest_frame(publish(420 + i, (i & 1) ? 0.3f : 0.1f, (i & 1) ? 12 : -12, false, false), true, false);
        IngestedSource popup_src{};
        for (int i = 0; i < 3; ++i) popup_src = ingest_frame(publish(430 + i, (i & 1) ? 0.3f : 0.1f, (i & 1) ? 12 : -12, false, true), true, false);
        show(popup_src, 0);
        const double popup_x0 = green_x();
        show(popup_src, yaw);
        const double popup_x1 = green_x();
        EXPECT(popup_x0 > 0 && std::fabs(popup_x1 - popup_x0) < 2.0, "HUD that just appeared is masked after two frames (%.1f -> %.1f)", popup_x0, popup_x1);
        // (G) Scenery with edges only along the motion (a horizontal bar during a horizontal pan) looks the
        // same every frame but is not HUD: it must still warp.
        renderer.reset_hud_detection();
        horizontal_bar = true;
        IngestedSource line_src{};
        for (int i = 0; i < 7; ++i) line_src = ingest_frame(publish(440 + i, 0.1f, (i & 1) ? 12 : -12, false, false), true, false);
        horizontal_bar = false;
        show(line_src, 0);
        const double line_x0 = segment_x();
        show(line_src, yaw);
        const double line_x1 = segment_x();
        EXPECT(line_x0 > 0 && line_x1 < line_x0 - 20.0, "a horizontal edge under a horizontal pan is not taken for HUD (%.1f -> %.1f)", line_x0, line_x1);
        // (H) Repeated game frames (same image, camera data says it moved): nothing may be masked.
        renderer.reset_hud_detection();
        scene_offset = 3;
        IngestedSource repeat_src{};
        for (int i = 0; i < 4; ++i) repeat_src = ingest_frame(publish(450 + i, 0.1f, 0, false, false), true, false);
        scene_offset = -1;
        show(repeat_src, 0);
        const double repeat_x0 = peak(px, w, h, true, 1);
        show(repeat_src, yaw);
        const double repeat_x1 = peak(px, w, h, true, 1);
        EXPECT(std::fabs(repeat_x1 - repeat_x0) > 20.0, "repeated frames do not mask the scene (%.0f -> %.0f)", repeat_x0, repeat_x1);
        // (I) A crosshair over a wall close to the camera (third person aiming at a near wall): HUD, not weapon.
        renderer.reset_hud_detection();
        near_wall = true;
        IngestedSource wall_src{};
        for (int i = 0; i < 7; ++i) wall_src = ingest_frame(publish(460 + i, (i & 1) ? 0.3f : 0.1f, (i & 1) ? 12 : -12, false, true), true, true);
        near_wall = false;
        show(wall_src, 0);
        const double wall_x0 = green_x();
        show(wall_src, yaw);
        const double wall_x1 = green_x();
        EXPECT(wall_x0 > 0 && std::fabs(wall_x1 - wall_x0) < 2.0, "HUD over a near wall is detected (%.1f -> %.1f)", wall_x0, wall_x1);
        // (J) Repeating scenery that moves exactly one period per frame looks unchanged, but moving scenery
        // explains it just as well: it must still warp.
        renderer.reset_hud_detection();
        repeating = true;
        IngestedSource repeat_scene{};
        for (int i = 0; i < 7; ++i) repeat_scene = ingest_frame(publish(470 + i, (i & 1) ? 0.3f : 0.1f, (i & 1) ? 12 : -12, false, false), true, false);
        repeating = false;
        auto cyan_x = [&]() {
            double sum = 0, weight = 0;
            for (std::uint32_t y = h * 5 / 8 + 4; y < h * 7 / 8 - 4; ++y)
                for (std::uint32_t x = 0; x < w / 2; ++x) {
                    const std::size_t i = (std::size_t(y) * w + x) * 4;
                    const float c = std::min(half_to_float(px[i + 1]), half_to_float(px[i + 2])) - half_to_float(px[i]);
                    if (c > 0.5f) { sum += x * c; weight += c; }
                }
            return weight > 0 ? sum / weight : -1.0;
        };
        show(repeat_scene, 0);
        const double rep_x0 = cyan_x();
        show(repeat_scene, yaw);
        const double rep_x1 = cyan_x();
        EXPECT(rep_x0 > 0 && rep_x1 < rep_x0 - 20.0, "repeating scenery is not taken for HUD (%.1f -> %.1f)", rep_x0, rep_x1);
        // (E) Semi-transparent HUD (50% over a scene that changes every frame): detected by its edges.
        renderer.reset_hud_detection();
        hud_patch_colour = 2;
        IngestedSource glass_src{};
        for (int i = 0; i < 7; ++i) glass_src = ingest_frame(publish(500 + i, (i & 1) ? 0.3f : 0.1f, (i & 1) ? 12 : -12, false, true), true, false);
        show(glass_src, 0);
        const double glass_x0 = green_x();
        show(glass_src, yaw);
        const double glass_x1 = green_x();
        hud_patch_colour = 0;
        // (K) HUD from the upscaler's output: found in the very first frame, with the camera still
        // (nothing to learn from), opaque and semi-transparent; the scene around it still warps.
        renderer.reset_hud_detection();
        with_scene = true;
        scene_offset = 3;
        IngestedSource scene_src = ingest_frame(publish(600, 0.1f, 0, false, true), true, false);
        show(scene_src, 0);
        const double sp_x0 = green_x(), sb_x0 = peak(px, w, h, true, 1);
        show(scene_src, yaw);
        const double sp_x1 = green_x(), sb_x1 = peak(px, w, h, true, 1);
        own_engine = true;  // the same frame and mask through FrameWarp's own engine
        show(scene_src, 0);
        const double op_x0 = green_x(), ob_x0 = peak(px, w, h, true, 1);
        show(scene_src, yaw);
        const double op_x1 = green_x(), ob_x1 = peak(px, w, h, true, 1);
        own_engine = false;
        std::printf("own engine with the HUD from the upscaler's output: patch x %.1f -> %.1f, bar x %.0f -> %.0f\n", op_x0, op_x1, ob_x0, ob_x1);
        EXPECT(op_x0 > 0 && std::fabs(op_x1 - op_x0) < 2.0, "own engine holds the HUD found from the upscaler's output (%.1f -> %.1f)", op_x0, op_x1);
        EXPECT(std::fabs(ob_x1 - ob_x0) > 20.0, "own engine still warps the scene next to it (%.0f -> %.0f)", ob_x0, ob_x1);
        hud_patch_colour = 2;
        IngestedSource scene_glass = ingest_frame(publish(601, 0.3f, 0, false, true), true, false);
        show(scene_glass, 0);
        const double sg_x0 = green_x();
        show(scene_glass, yaw);
        const double sg_x1 = green_x();
        hud_patch_colour = 0;
        // GPU cost of the scene HUD passes (per pass, from the renderer's own timestamps).
        {
            const int cost_slot = publish(603, 0.1f, 0, false, true);
            renderer.take_hud_stats();
            for (int i = 0; i < 12; ++i) ingest_frame(cost_slot, true, false);
            for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
            const HudStats hs = renderer.take_hud_stats();
            const double n = std::max(1, hs.scene_frames);
            std::printf("scene HUD GPU ms per frame at %ux%u (%d frames):", W, H, hs.scene_frames);
            for (double v : hs.scene_pass_ms) std::printf(" %.3f", v / n);
            std::printf("\n");
        }
        // (N) Highlights the tone mapping washes out towards white are scenery, not HUD: they warp, while
        // the HUD patch in the same frame is held.
        washed_patch = true;
        IngestedSource washed_src = ingest_frame(publish(730, 0.1f, 0, false, true), true, false);
        washed_patch = false;
        auto washed_x = [&]() {
            double sum = 0, weight = 0;
            for (std::uint32_t y = h * 5 / 8; y < h * 3 / 4; ++y)
                for (std::uint32_t x = w / 2; x < w; ++x) {
                    const std::size_t i = (std::size_t(y) * w + x) * 4;
                    const float r = half_to_float(px[i]), g = half_to_float(px[i + 1]), bl = half_to_float(px[i + 2]);
                    if (r > 0.93f && g > 0.93f && bl > 0.82f && bl < 0.92f) { sum += x; weight += 1; }
                }
            return weight > 0 ? sum / weight : -1.0;
        };
        show(washed_src, 0);
        const double wp_x0 = washed_x(), wh_x0 = green_x();
        show(washed_src, yaw);
        const double wp_x1 = washed_x(), wh_x1 = green_x();
        std::printf("washed-out highlight x %.1f -> %.1f; HUD patch next to it x %.1f -> %.1f\n", wp_x0, wp_x1, wh_x0, wh_x1);
        EXPECT(wp_x0 > 0 && std::fabs(wp_x1 - wp_x0) > 20.0, "a highlight washed out by the tone mapping is not taken for HUD (%.1f -> %.1f)", wp_x0, wp_x1);
        EXPECT(wh_x0 > 0 && std::fabs(wh_x1 - wh_x0) < 2.0, "the HUD next to it is still held (%.1f -> %.1f)", wh_x0, wh_x1);
        // (L) Without HUD, nothing of the scene is taken for HUD: the bar and the stripes all warp.
        IngestedSource scene_clean = ingest_frame(publish(740, 0.1f, 0, false, false), true, false);
        show(scene_clean, 0);
        const double sc_x0 = peak(px, w, h, true, 1);
        show(scene_clean, yaw);
        const double sc_x1 = peak(px, w, h, true, 1);
        // GPU cost of taking in one game frame, split like the presenter's log (HUD from the upscaler's
        // output, masks, motion analysis all on), with the GPU otherwise idle.
        {
            renderer.take_gpu_usage();
            (void)renderer.take_hud_stats();
            for (int i = 0; i < 24; ++i) {
                const int bs = publish(900 + i, 0.1f, 0, true, true);
                if (bs < 0) continue;
                renderer.begin_frame();
                IngestedSource s = renderer.ingest(sh, bs);
                renderer.analyze_motion(s, moving_cam.clip_to_prev_clip, 1.0f, 1.0f, true);
                renderer.build_no_warp_mask(s, moving_cam.clip_to_prev_clip, true, true, true);
                renderer.submit_work();
                renderer.wait_idle();
                if (i == 3) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); renderer.take_gpu_usage(); }
            }
            for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
            const Renderer::GpuUsage u = renderer.take_gpu_usage();
            const HudStats hs = renderer.take_hud_stats();
            double fits = 0;
            for (int i = 0; i + 1 < 18; ++i) fits += hs.scene_pass_ms[i];
            const double n = std::max<double>(1, u.split);
            std::printf("intake GPU at %ux%u (depth %ux%u): %.3f ms per game frame (reaching the textures %.3f, depth+motion %.3f, colour copies %.3f, HUD/masks/motion %.3f); "
                        "HUD from output: fits %.3f ms, final %.3f ms (%u frames)\n",
                        W, H, DW, DH, u.intakes ? u.intake_ms / u.intakes : 0.0, u.access_ms / n, u.depth_ms / n, u.colour_ms / n, u.rest_ms / n,
                        hs.scene_frames ? fits / hs.scene_frames : 0.0, hs.scene_frames ? hs.scene_pass_ms[17] / hs.scene_frames : 0.0, u.intakes);
            if (hs.scene_frames) {
                std::printf("HUD from output: %.6f%% of the screen; GPU ms per pass:", 100.0 * hs.scene_share_sum / hs.scene_frames);
                for (int i = 0; i < 18; ++i) std::printf(" %.3f", hs.scene_pass_ms[i] / hs.scene_frames);
                std::printf("\n");
            }
        }
        with_scene = false;
        scene_offset = -1;
        std::printf("HUD from the upscaler's output: patch x %.1f -> %.1f, bar x %.0f -> %.0f; semi-transparent patch x %.1f -> %.1f; "
                    "no HUD: bar x %.0f -> %.0f\n", sp_x0, sp_x1, sb_x0, sb_x1, sg_x0, sg_x1, sc_x0, sc_x1);
        EXPECT(scene_src.has_scene, "the upscaler's output is ingested");
        EXPECT(sp_x0 > 0 && std::fabs(sp_x1 - sp_x0) < 2.0, "HUD found from the upscaler's output in one frame (%.1f -> %.1f)", sp_x0, sp_x1);
        EXPECT(std::fabs(sb_x1 - sb_x0) > 20.0, "the scene next to that HUD still warps (%.0f -> %.0f)", sb_x0, sb_x1);
        EXPECT(sg_x0 > 0 && std::fabs(sg_x1 - sg_x0) < 2.0, "semi-transparent HUD found from the upscaler's output (%.1f -> %.1f)", sg_x0, sg_x1);
        EXPECT(std::fabs(sc_x1 - sc_x0) > 20.0, "no HUD: nothing held (%.0f -> %.0f)", sc_x0, sc_x1);
        std::printf("no-warp mask: weapon strip x %.0f -> %.0f; HUD patch x %.1f -> %.1f, scene bar x %.0f -> %.0f; changed patch x %.1f -> %.1f; "
                    "after weapon x %.0f -> %.0f; semi-transparent patch x %.1f -> %.1f (5 deg yaw)\n",
                    weapon_x0, weapon_x1, patch_x0, patch_x1, bar_x0, bar_x1, gone_x0, gone_x1, after_x0, after_x1, glass_x0, glass_x1);
        EXPECT(std::fabs(after_x1 - after_x0) > 20.0, "where a held weapon was, the scene warps the next frame (%.0f -> %.0f)", after_x0, after_x1);
        // Over a textured scene only part of a semi-transparent panel is held (its edges): it must move far
        // less than the scene (about 32 px here), not necessarily zero.
        EXPECT(glass_x0 > 0 && std::fabs(glass_x1 - glass_x0) < 8.0, "semi-transparent HUD is mostly held (%.1f -> %.1f)", glass_x0, glass_x1);
        EXPECT(gone_x0 > 0 && gone_x1 < gone_x0 - 20.0, "a former HUD area that changed warps again in the next frame");
        EXPECT(std::fabs(weapon_x1 - weapon_x0) < 2.0, "camera-attached strip (motion vectors ignore the camera) is not warped");
        EXPECT(std::fabs(sky_x1 - sky_x0) > 20.0, "a far zero-motion strip (sky) is still warped (%.0f -> %.0f)", sky_x0, sky_x1);
        EXPECT(patch_x0 > 0 && std::fabs(patch_x1 - patch_x0) < 2.0, "static HUD patch is not warped");
        EXPECT(std::fabs(bar_x1 - bar_x0) > 20.0, "the scene under the HUD mask is still warped");
    }

    // Timing: GPU time of a whole presenter frame (Latewarp + blit), steady state.
    // Every 4th frame takes in a new game frame (ingest + rendered-frame Latewarp), like 30 fps -> 120 Hz.
    double total = 0; int samples = 0;
    std::vector<double> cpu_ingest_tick, cpu_plain_tick, cpu_ingest, cpu_eval_rendered, cpu_eval_plain, cpu_present;
    LARGE_INTEGER qf; QueryPerformanceFrequency(&qf);
    auto ms_since = [&](LARGE_INTEGER a) { LARGE_INTEGER b; QueryPerformanceCounter(&b); return double(b.QuadPart - a.QuadPart) * 1000.0 / double(qf.QuadPart); };
    for (int i = 0; i < 120; ++i) {
        const bool new_source = (i % 4) == 0;
        LARGE_INTEGER t0, t1, t2, t3; QueryPerformanceCounter(&t0);
        auto* l = renderer.begin_frame();
        const CameraBasis target = apply_rotation(source, {0, 0, 1}, 0.001 * i, 0, source.pos);
        IngestedSource steady{};
        QueryPerformanceCounter(&t1);
        if (new_source) steady = renderer.ingest(sh, slot);
        if (new_source && i >= 20) cpu_ingest.push_back(ms_since(t1));
        steady.valid = steady.has_depth = true;
        steady.color_w = W; steady.color_h = H;
        steady.color_rect = {0, 0, W, H}; steady.depth_rect = {0, 0, DW, DH};
        auto inputs = renderer.latewarp_inputs(steady, true);
        inputs.depth_inverted = true;
        QueryPerformanceCounter(&t2);
        latewarp.evaluate(l, inputs, new_source, view_matrix(target, {}), view_matrix(source, {}), projection);
        if (i >= 20) (new_source ? cpu_eval_rendered : cpu_eval_plain).push_back(ms_since(t2));
        QueryPerformanceCounter(&t3);
        renderer.finish_frame(true, 0);
        if (i >= 20) cpu_present.push_back(ms_since(t3));
        if (i >= 20) (new_source ? cpu_ingest_tick : cpu_plain_tick).push_back(ms_since(t0));
        if (i >= 10 && renderer.last_gpu_ms() > 0) { total += renderer.last_gpu_ms(); ++samples; }
    }
    std::printf("%ux%u presenter frame GPU time: %.3f ms average over %d frames\n", W, H, samples ? total / samples : 0.0, samples);
    auto p50 = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    auto pmax = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    std::printf("CPU ms (p50/max): ingest %.3f/%.3f, Latewarp rendered-frame %.3f/%.3f, Latewarp plain %.3f/%.3f, finish+present %.3f/%.3f\n",
                p50(cpu_ingest), pmax(cpu_ingest), p50(cpu_eval_rendered), pmax(cpu_eval_rendered), p50(cpu_eval_plain), pmax(cpu_eval_plain),
                p50(cpu_present), pmax(cpu_present));
    std::printf("CPU ms per tick incl. waits (p50/max): new-frame tick %.3f/%.3f, other tick %.3f/%.3f\n", p50(cpu_ingest_tick), pmax(cpu_ingest_tick),
                p50(cpu_plain_tick), pmax(cpu_plain_tick));
    renderer.wait_idle();
    latewarp.shutdown();
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("pipeline tests passed\n");
    return 0;
}
