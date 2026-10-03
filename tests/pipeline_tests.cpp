// End-to-end GPU test of the transport + Latewarp path without a game:
// "game" device A renders a synthetic frame and publishes it through fw::Producer exactly like the
// add-on does (planar D32S8 depth at render resolution, RGBA8 backbuffer); device B runs the
// presenter's Renderer + Latewarp12 and we check the warped marker moves by the expected amount.
#include "addon/producer.hpp"
#include "presenter/pose.hpp"
#include "presenter/renderer.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    const auto colour_space = kColour == DXGI_FORMAT_R10G10B10A2_UNORM ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
                                                                         : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
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
    producer.set_swapchain(window, W, H, kColour, static_cast<std::uint32_t>(colour_space));

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
    if (!renderer.init(sh.adapter, window, W, H, kColour, static_cast<std::uint32_t>(colour_space), error)) {
        std::printf("FAIL renderer: %s\n", error.c_str()); return 1;
    }
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

    // Moving objects: static camera, the white bar's motion vectors say it moved 20 render px to the right
    // since the previous frame. Shown half a frame back it must be 10 render px to the left, with the
    // area it left showing background; the red bar (static) must not move.
    {
        ComPtr<ID3D12Fence> gf; game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gf));
        UINT64 gfv = 0;
        D3D12_RESOURCE_DESC md = cd; md.Width = DW; md.Height = DH; md.Format = DXGI_FORMAT_R16G16_FLOAT;
        ComPtr<ID3D12Resource> motion;
        game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &md, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&motion));
        ComPtr<ID3D12DescriptorHeap> mv_rtv;
        D3D12_DESCRIPTOR_HEAP_DESC mh{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        game->CreateDescriptorHeap(&mh, IID_PPV_ARGS(&mv_rtv));
        game->CreateRenderTargetView(motion.Get(), nullptr, mv_rtv->GetCPUDescriptorHandleForHeapStart());
        Camera still = cam;
        for (int i = 0; i < 16; ++i) still.clip_to_prev_clip[i] = (i % 5 == 0) ? 1.0f : 0.0f;  // identity: camera did not move
        auto publish_mv = [&](std::uint64_t fid) {
            alloc->Reset();
            list->Reset(alloc.Get(), nullptr);
            const float none[4] = {0, 0, 0, 0}, moved[4] = {-20.0f / float(DW), 0, 0, 0};  // uv towards the previous frame
            const LONG bx = LONG(DW / 2);
            const D3D12_RECT bar{bx - 6, 0, bx + 6, LONG(DH)};
            list->ClearRenderTargetView(mv_rtv->GetCPUDescriptorHandleForHeapStart(), none, 0, nullptr);
            list->ClearRenderTargetView(mv_rtv->GetCPUDescriptorHandleForHeapStart(), moved, 1, &bar);
            producer.on_constants(fid, still);
            producer.on_tag(fid, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
            producer.on_tag(fid, kMotion, motion.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, DW, DH, list.Get());
            const auto t = producer.begin_present(backbuffer.Get(), list.Get());
            list->Close();
            queue->ExecuteCommandLists(1, lists);
            producer.finish_present(queue.Get(), t);
            queue->Signal(gf.Get(), ++gfv);
            while (gf->GetCompletedValue() < gfv) Sleep(1);
            int s_slot = -1;
            for (int i = 0; i < kSlots; ++i) if (sh.slots[i].state == kReady && sh.slots[i].frame_id == fid) s_slot = i;
            EXPECT(s_slot >= 0 && sh.slots[s_slot].tex[kMotion].valid, "frame with motion vectors published");
            return s_slot;
        };
        renderer.set_object_history(true);
        // The previous frame, then the current one (each keeps the one before it).
        IngestedSource src;
        bool moved_ok = false;
        for (std::uint64_t fid = 100; fid <= 101; ++fid) {
            const int s_slot = publish_mv(fid);
            renderer.begin_frame();
            src = renderer.ingest(sh, s_slot);
            EXPECT(src.has_motion, "motion vectors ingested");
            if (fid == 101) {
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
            renderer.analyze_motion(src, still.clip_to_prev_clip, 1.0f, 1.0f, true, true, true, false, fid);
            const bool built = renderer.object_motion(src, still.clip_to_prev_clip, still.view_to_clip, 1.0f, 1.0f);
            if (fid == 100) EXPECT(!built, "no object motion without a previous frame");
            else moved_ok = built;
            renderer.finish_frame(false, 0);
            renderer.wait_idle();
        }
        EXPECT(moved_ok, "object motion built once the previous frame is kept");
        MotionFit fit{};
        // (the fits of both frames come back in order, a few frames later)
        for (int i = 0; i < 8 && fit.frame != 101; ++i)
            if (!renderer.take_motion_fit(fit)) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
        EXPECT(fit.frame == 101, "the motion fit comes back with its game frame (%llu, expected 101)", static_cast<unsigned long long>(fit.frame));
        const Mat4 identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        auto show_back = [&](float frames_back) {
            renderer.begin_frame();
            Renderer::ObjectWarp objects;
            objects.frames_back = frames_back;
            std::memcpy(objects.prev_to_target, identity.data(), sizeof(objects.prev_to_target));
            std::memcpy(objects.view_to_clip, still.view_to_clip, sizeof(objects.view_to_clip));
            const bool ok = renderer.own_warp(src, true, false, identity.data(), true, false, frames_back > 0 ? &objects : nullptr);
            EXPECT(ok, "own warp ran");
            renderer.finish_frame(ok, 0);
            renderer.read_back(true, px, w, h);
        };
        show_back(0.0f);
        const double still_x = peak(px, w, h, true, 1), still_red = peak(px, w, h, false, 0);
        show_back(0.5f);
        const double half_x = peak(px, w, h, true, 1), half_red = peak(px, w, h, false, 0);
        auto white_at = [&](double x) {
            double sum = 0; const std::uint32_t xi = std::uint32_t(x);
            for (std::uint32_t y = h / 5; y < h * 2 / 5; ++y) {
                const std::size_t i = (std::size_t(y) * w + xi) * 4;
                sum += (half_to_float(px[i]) + half_to_float(px[i + 1]) + half_to_float(px[i + 2])) / 3.0;
            }
            return sum / double(h / 5);
        };
        const double left_behind = white_at(still_x);  // where the bar is now; half a frame back it is a bar width further left
        show_back(1.0f);
        const double back_x = peak(px, w, h, true, 1);
        const double step = 20.0 * double(W) / double(DW);
        std::printf("moving objects: bar x=%.0f, half a frame back %.0f (expected %.1f), a whole frame back %.0f (expected %.1f); "
                    "brightness where it left %.2f; red bar y %.0f -> %.0f; moving pixels %.0f of %.0f\n",
                    still_x, half_x, still_x - step / 2, back_x, still_x - step, left_behind, still_red, half_red, fit.moving, fit.samples);
        EXPECT(std::fabs((half_x - still_x) + step / 2) < 4.0, "half a frame back the object is half way to its previous position");
        EXPECT(std::fabs((back_x - still_x) + step) < 4.0, "a whole frame back it is at its previous position");
        EXPECT(left_behind < 0.5, "the area the object left shows background (%.2f)", left_behind);
        EXPECT(std::fabs(half_red - still_red) < 1.5, "static geometry stays put");
        EXPECT(fit.moving > 0.5 * 12 * DH && fit.moving < 2.0 * 12 * DH, "moving pixels = the bar (%.0f)", fit.moving);
        // GPU cost per presented frame, with and without moving objects: refreshes back to back (a GPU left idle
        // between them clocks down and takes many times longer).
        auto median_gpu = [&](float frames_back) {
            Renderer::ObjectWarp objects;
            objects.frames_back = frames_back;
            std::memcpy(objects.prev_to_target, identity.data(), sizeof(objects.prev_to_target));
            std::memcpy(objects.view_to_clip, still.view_to_clip, sizeof(objects.view_to_clip));
            renderer.wait_idle();
            renderer.take_gpu_usage();
            for (int i = 0; i < 200; ++i) {
                renderer.begin_frame();
                renderer.own_warp(src, true, false, identity.data(), true, false, frames_back > 0 ? &objects : nullptr);
                renderer.finish_frame(true, 0);
            }
            renderer.wait_idle();
            for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
            const Renderer::GpuUsage u = renderer.take_gpu_usage();
            return u.warps ? float(u.warp_ms / u.warps) : 0.0f;
        };
        const float plain = median_gpu(0.0f), with_objects = median_gpu(0.5f);
        std::printf("GPU per presented frame: %.3f ms plain, %.3f ms with moving objects (+%.3f)\n", plain, with_objects, with_objects - plain);
        renderer.set_object_history(false);

        // A refresh with nothing new to show is submitted without drawing or presenting (skip_frame): counted as
        // such, and the frame slots go on as usual.
        {
            renderer.wait_idle();
            renderer.take_gpu_usage();
            for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.skip_frame(); }
            for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); }
            renderer.wait_idle();
            const Renderer::GpuUsage u = renderer.take_gpu_usage();
            std::printf("refreshes with nothing new: %u skipped, %u drawn\n", u.skipped, u.warps);
            EXPECT(u.skipped == 3, "skipped refreshes are counted, not timed as drawn ones (%u)", u.skipped);
            EXPECT(renderer.device()->GetDeviceRemovedReason() == S_OK, "the device is fine after skipped refreshes");
        }

        // Frame generation (the game's own, used automatically): its generated image is taken where it is
        // generated, the frame is published once the GPU has run that command list, taken in with the image, and
        // the warp shows the image (its bar 30 px left of the frame's) or the frame.
        {
            ComPtr<ID3D12Resource> gen;
            game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &cd, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&gen));
            ComPtr<ID3D12DescriptorHeap> gen_rtv;
            D3D12_DESCRIPTOR_HEAP_DESC gh{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            game->CreateDescriptorHeap(&gh, IID_PPV_ARGS(&gen_rtv));
            game->CreateRenderTargetView(gen.Get(), nullptr, gen_rtv->GetCPUDescriptorHandleForHeapStart());
            const std::uint64_t fid = 151;
            alloc->Reset();
            list->Reset(alloc.Get(), nullptr);
            const float none[4] = {0, 0, 0, 0}, black[4] = {0, 0, 0, 1}, bright[4] = {1, 1, 1, 1};
            list->ClearRenderTargetView(mv_rtv->GetCPUDescriptorHandleForHeapStart(), none, 0, nullptr);
            list->ClearRenderTargetView(gen_rtv->GetCPUDescriptorHandleForHeapStart(), black, 0, nullptr);
            const D3D12_RECT gen_bar{LONG(still_x) - 30, 0, LONG(still_x) - 30 + 8, LONG(H)};  // (peak: the left edge, as the frame's)
            list->ClearRenderTargetView(gen_rtv->GetCPUDescriptorHandleForHeapStart(), bright, 1, &gen_bar);
            producer.on_constants(fid, still);
            producer.on_tag(fid, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
            producer.on_tag(fid, kMotion, motion.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, DW, DH, list.Get());
            producer.on_present_marker(fid);
            producer.on_generated(0, 1, 1, gen.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, backbuffer.Get(), D3D12_RESOURCE_STATE_PRESENT, list.Get());
            list->Close();
            queue->ExecuteCommandLists(1, lists);  // (the frame is published once the GPU has run the list)
            queue->Signal(gf.Get(), ++gfv);
            while (gf->GetCompletedValue() < gfv) Sleep(1);
            int g_slot = -1;
            for (int wait = 0; wait < 200 && g_slot < 0; ++wait, Sleep(1))
                for (int i = 0; i < kSlots; ++i) if (sh.slots[i].state == kReady && sh.slots[i].frame_id == fid) g_slot = i;
            EXPECT(g_slot >= 0 && sh.slots[g_slot].generated == 1 && sh.slots[g_slot].tex[kBackbuffer].valid,
                   "frame generation: the frame is published with its generated image when its list is submitted");
            EXPECT(producer.generation_active(), "frame generation: presents publish nothing meanwhile");
            if (g_slot >= 0) {
                renderer.begin_frame();
                const IngestedSource gs = renderer.ingest(sh, g_slot);
                EXPECT(gs.generated == 1 && gs.per_frame == 1, "frame generation: the image is taken in with the frame (%d of %d)", gs.generated,
                       gs.per_frame);
                renderer.analyze_motion(gs, still.clip_to_prev_clip, 1.0f, 1.0f, true, true, true, false, fid);
                renderer.prepare_generated(gs, 1.0f, 1.0f, true, true);
                renderer.finish_frame(false, 0);
                renderer.wait_idle();
                auto show_gen = [&](int which) {
                    renderer.begin_frame();
                    const bool ok = renderer.own_warp(gs, true, false, identity.data(), true, false, nullptr, which);
                    renderer.finish_frame(ok, 0);
                    renderer.read_back(true, px, w, h);
                };
                show_gen(0);
                const double gen_x = peak(px, w, h, true, 1);
                show_gen(-1);
                const double real_x = peak(px, w, h, true, 1);
                std::printf("frame generation: generated image bar x %.0f (expected %.0f), the frame's %.0f (expected %.0f)\n", gen_x, still_x - 30,
                            real_x, still_x);
                EXPECT(std::fabs(gen_x - (still_x - 30)) < 2.0, "frame generation: the warp shows the generated image");
                EXPECT(std::fabs(real_x - still_x) < 2.0, "frame generation: and the frame itself after it");
            }
            // 6x (5 images per frame, one call each): 3 are kept, evenly (1, 3, 5), and the frame is published
            // after the last one only.
            {
                const std::uint64_t fid6 = 152;
                alloc->Reset();
                list->Reset(alloc.Get(), nullptr);
                producer.on_constants(fid6, still);
                producer.on_tag(fid6, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
                producer.on_tag(fid6, kMotion, motion.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, DW, DH, list.Get());
                producer.on_present_marker(fid6);
                for (std::uint32_t n = 1; n <= 5; ++n)
                    producer.on_generated(0, n, 5, gen.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, backbuffer.Get(), D3D12_RESOURCE_STATE_PRESENT,
                                          list.Get());
                list->Close();
                queue->ExecuteCommandLists(1, lists);
                queue->Signal(gf.Get(), ++gfv);
                while (gf->GetCompletedValue() < gfv) Sleep(1);
                int s6 = -1;
                for (int wait = 0; wait < 200 && s6 < 0; ++wait, Sleep(1))
                    for (int i = 0; i < kSlots; ++i) if (sh.slots[i].state == kReady && sh.slots[i].frame_id == fid6) s6 = i;
                const auto& m6 = sh.slots[s6 >= 0 ? s6 : 0];
                std::printf("frame generation 6x: %u of %u images kept (%u, %u, %u)\n", m6.generated, m6.per_frame, m6.gen_index[0], m6.gen_index[1],
                            m6.gen_index[2]);
                EXPECT(s6 >= 0 && m6.generated == 3 && m6.per_frame == 5 && m6.gen_index[0] == 1 && m6.gen_index[1] == 3 && m6.gen_index[2] == 5,
                       "frame generation 6x: images 1, 3 and 5 kept, the frame published after the last");
            }
            // Frame generation off: its textures go (video memory), and come back with it.
            EXPECT(renderer.has_generated(), "frame generation: its textures are held while it is on");
            renderer.release_generated();
            EXPECT(!renderer.has_generated(), "frame generation: its textures released once it is off");
            Sleep(600);  // (frame generation's images stop: presents publish frames again)
            EXPECT(!producer.generation_active(), "frame generation: over half a second after its last image");
        }
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
        // The game's HUD-less picture without a UI layer (Cyberpunk 2077 with frame generation): the frame as it
        // is just before the HUD patch is drawn.
        bool with_hudless = false;
        // A bright yellow-white surface the tone mapping washes out towards white (as filmic tone mappers
        // do): scene (4, 4, 0.25) -> per channel (1, 1, 0.5), washed 80% towards its grey 0.964.
        bool washed_patch = false;
        // Scenery the upscaler's output does not have (post-processing after upscaling: neon, bloom,
        // effects): magenta 2 px stripes in the frame only, at this x (< 0: none).
        LONG neon_x = -1;
        const LONG neon_y0 = LONG(H * 3 / 8), neon_y1 = LONG(H / 2);
        bool combined_hud = false;  // HUD from the upscaler's output + camera-motion check
        bool turn_rule = false;     // attached: what stays nearly still on screen while the camera turns
        bool memory = false;        // background memory: updated at ingest, used by the own warp
        LONG strip_half = 8;        // half width of the weapon strip (render px)
        // The weapon's visible outline reaching past what its motion vectors mark: a white band just left of
        // the strip, at the weapon's depth, with the scenery's motion vectors.
        bool fringe = false;
        int stretch = 0;            // render px the scenery around the character/weapon stretches over
        bool memory_consecutive = false;
        LONG stripe_phase = -1;     // >= 0: the scenery stripes at this phase (moving with the camera)
        const Camera* analyze_cam = &moving_cam;  // the camera motion the analysis gets
        bool hud_fill = false;      // own warp: fill behind the HUD from the upscaler's output
        bool near_rule = true;      // attached: near-camera pixels moving against the camera model count
        const D3D12_RECT washed_rect{LONG(W * 3 / 4), LONG(H * 5 / 8), LONG(W * 3 / 4) + 64, LONG(H * 5 / 8) + 64};
        ComPtr<ID3D12Resource> scene_tex, hudless_tex;
        game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &cd, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&hudless_tex));
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
            const LONG phase = stripe_phase >= 0 ? stripe_phase : scene_offset >= 0 ? scene_offset : LONG((fid * 5) % 16);
            for (LONG x = phase; x < LONG(W); x += 16) {
                const D3D12_RECT stripe{x, 0, x + 4, LONG(H)};
                list->ClearRenderTargetView(rtv, grey, 1, &stripe);
            }
            const D3D12_RECT bar{LONG(W / 2) - 4 + bar_dx, 0, LONG(W / 2) + 4 + bar_dx, LONG(H)};
            list->ClearRenderTargetView(rtv, white, 1, &bar);
            if (fringe) {
                const LONG fx = LONG(DW / 2);
                const D3D12_RECT band{LONG((fx - 12) * W / DW), 0, LONG((fx - 9) * W / DW), LONG(H)};
                list->ClearRenderTargetView(rtv, white, 1, &band);
            }
            if (horizontal_bar) {
                const D3D12_RECT segment{LONG(W / 2 - W / 8), LONG(H * 3 / 4), LONG(W / 2 + W / 8), LONG(H * 3 / 4) + 6};
                const float yellow[4] = {1, 1, 0, 1};
                list->ClearRenderTargetView(rtv, yellow, 1, &segment);
            }
            if (washed_patch) {
                const float washed[4] = {0.971f, 0.971f, 0.871f, 1};
                list->ClearRenderTargetView(rtv, washed, 1, &washed_rect);
            }
            if (neon_x >= 0) {
                const float magenta[4] = {1, 0, 1, 1};
                for (LONG x = neon_x; x < neon_x + 48; x += 4) {
                    const D3D12_RECT stripe{x, neon_y0, x + 2, neon_y1};
                    list->ClearRenderTargetView(rtv, magenta, 1, &stripe);
                }
            }
            if (with_hudless) {
                D3D12_RESOURCE_BARRIER hb[2]{};
                hb[0].Type = hb[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                hb[0].Transition = {backbuffer.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE};
                hb[1].Transition = {hudless_tex.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                    D3D12_RESOURCE_STATE_COPY_DEST};
                list->ResourceBarrier(2, hb);
                list->CopyResource(hudless_tex.Get(), backbuffer.Get());
                for (auto& x : hb) std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
                list->ResourceBarrier(2, hb);
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
                const D3D12_RECT strip{bx - strip_half, 0, bx + strip_half, LONG(DH)};
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
                const D3D12_RECT strip{bx - strip_half, 0, bx + strip_half, LONG(DH)};
                list->ClearDepthStencilView(dsv_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH,
                                            near_strip ? strip_depth : 1e-4f, 0, 1, &strip);
                if (fringe) {
                    const D3D12_RECT band{bx - 12, 0, bx - 9, LONG(DH)};
                    list->ClearDepthStencilView(dsv_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, strip_depth, 0, 1, &band);
                }
            }
            producer.on_tag(fid, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
            producer.on_tag(fid, kMotion, motion.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, DW, DH, list.Get());
            if (with_scene) producer.on_tag(fid, kScene, scene_tex.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
            if (with_hudless) producer.on_tag(fid, kHudless, hudless_tex.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
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
                renderer.own_warp(s, true, inputs.no_warp_mask != nullptr, m.data(), true, memory);
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
            renderer.analyze_motion(s, analyze_cam->clip_to_prev_clip, 1.0f, 1.0f, true, true, near_rule, turn_rule);
            renderer.build_no_warp_mask(s, moving_cam.clip_to_prev_clip, hud, weapon, weapon, true, combined_hud, hud_fill, stretch);
            if (memory) renderer.update_memory(s, analyze_cam->clip_to_prev_clip, true, memory_consecutive);
            renderer.finish_frame(false, 0);
            renderer.wait_idle();
            // (FW_MASKHASH: every mask this builds, hashed - to check an optimisation changes nothing)
            if (std::getenv("FW_MASKHASH")) {
                static int built = 0;
                ++built;
                for (const int which : {6, 7, 8, 9, 27}) {
                    std::vector<std::uint16_t> data;
                    std::uint32_t dw = 0, dh = 0;
                    std::uint64_t hash = 1469598103934665603ull;
                    if (renderer.read_back(which, data, dw, dh))
                        for (const std::uint16_t v : data) { hash ^= v; hash *= 1099511628211ull; }
                    else hash = 0;
                    std::printf("maskhash %d %d %016llx\n", built, which, static_cast<unsigned long long>(hash));
                }
            }
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
        // Near-camera rule off (the floor band option): a weapon stuck to the screen is still held.
        near_rule = false;
        IngestedSource far_rule_src = ingest_frame(publish(211, 0.1f, 0, true, false), false, true);
        near_rule = true;
        show(far_rule_src, 0);
        const double nr_x0 = peak(px, w, h, true, 1);
        show(far_rule_src, yaw);
        const double nr_x1 = peak(px, w, h, true, 1);
        std::printf("near-camera rule off: weapon stuck to the screen x %.0f -> %.0f\n", nr_x0, nr_x1);
        EXPECT(std::fabs(nr_x1 - nr_x0) < 2.0, "near-camera rule off: a weapon stuck to the screen is still held (%.0f -> %.0f)", nr_x0, nr_x1);
        // (A1c) A camera orbiting a third-person character: it turns, but moves sideways too, so the character
        // at the pivot (10x the near plane) stays still on screen and follows the camera model. The rules
        // above do not hold it; the turn rule does. Scenery still warps.
        {
            Camera orbit = moving_cam;
            orbit.clip_to_prev_clip[8] = -2.0f * shift / 0.1f;  // prev.x = x + 2*shift*(1 - d / 0.1): no motion at the pivot's depth
            const float old_depth = strip_depth;
            strip_depth = 0.1f;
            weapon_mv = 0.0f;
            analyze_cam = &orbit;
            IngestedSource plain = ingest_frame(publish(212, 0.1f, 0, true, false), false, true);
            show(plain, 0);
            const double p0 = peak(px, w, h, true, 1);
            show(plain, yaw);
            const double p1 = peak(px, w, h, true, 1);
            turn_rule = true;
            IngestedSource held_src = ingest_frame(publish(213, 0.1f, 0, true, false), false, true);
            show(held_src, 0);
            const double h0 = peak(px, w, h, true, 1);
            show(held_src, yaw);
            const double h1 = peak(px, w, h, true, 1);
            // A camera that only turns: the same rule holds nothing of the scenery (the bar warps).
            analyze_cam = &moving_cam;
            IngestedSource turn_src = ingest_frame(publish(214, 0.1f, 0, false, false), false, true);
            turn_rule = false;
            strip_depth = old_depth;
            show(turn_src, 0);
            const double t0 = peak(px, w, h, true, 1);
            show(turn_src, yaw);
            const double t1 = peak(px, w, h, true, 1);
            std::printf("orbited character: without the turn rule x %.0f -> %.0f, with it %.0f -> %.0f; scenery with a plain turn %.0f -> %.0f\n",
                        p0, p1, h0, h1, t0, t1);
            EXPECT(std::fabs(p1 - p0) > 20.0, "the orbited character follows the camera model: not held by the other rules (%.0f -> %.0f)", p0, p1);
            EXPECT(std::fabs(h1 - h0) < 2.0, "the turn rule holds the orbited character (%.0f -> %.0f)", h0, h1);
            EXPECT(std::fabs(t1 - t0) > 20.0, "with a plain turn the turn rule holds no scenery (%.0f -> %.0f)", t0, t1);
        }
        // (A1d) Background memory: the camera turns (the scenery moves 6 px left per frame) past the held
        // weapon strip, so what is behind the strip now was visible to its right a few frames ago. What the
        // turn then uncovers beside the strip comes from memory: close to the real scenery (the same frame
        // rendered without the strip), much closer than the stretch fill.
        {
            own_engine = true;
            auto gap_error = [&](const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
                const LONG x0 = LONG(DW / 2 * W / DW) - LONG(34 * W / 1280), x1 = LONG(DW / 2 * W / DW) - LONG(20 * W / 1280);  // left of the held strip (the turn's shift grows with W)
                double sum = 0; int n = 0;
                for (std::uint32_t y = h / 4; y < h * 3 / 4; y += 2)
                    for (LONG x = x0; x < x1; ++x) {
                        const std::size_t i = (std::size_t(y) * w + std::size_t(x)) * 4;
                        for (int k = 0; k < 3; ++k) sum += std::fabs(half_to_float(a[i + k]) - half_to_float(b[i + k]));
                        n += 3;
                    }
                return n ? sum / n : 0.0;
            };
            auto phase_at = [&](int frame) { return LONG(((12 + 6 * (3 - frame)) % 16 + 16) % 16); };  // frame 3 at phase 12
            stripe_phase = phase_at(3);
            IngestedSource truth_src = ingest_frame(publish(215, 0.1f, -400, false, false), false, true);
            show(truth_src, yaw);
            const std::vector<std::uint16_t> truth = px;
            renderer.reset_hud_detection();  // (also starts the memory over)
            memory = true;
            IngestedSource last{};
            for (int f = 0; f < 4; ++f) {
                stripe_phase = phase_at(f);
                memory_consecutive = f > 0;
                last = ingest_frame(publish(216 + f, 0.1f, -400, true, false), false, true);
            }
            show(last, yaw);
            const double remembered = gap_error(px, truth);
            memory = false;
            show(last, yaw);
            const double stretched = gap_error(px, truth);
            stripe_phase = -1;
            memory_consecutive = false;
            own_engine = false;
            std::printf("background memory: error beside the held strip %.4f (stretch fill %.4f)\n", remembered, stretched);
            EXPECT(remembered < 0.5 * stretched, "the background memory shows what was behind the held strip (%.4f vs %.4f)", remembered, stretched);
        }
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
        // (P) Fill behind the HUD (own engine): where the turn uncovers what the HUD patch hid, the scenery
        // predicted from the upscaler's output, close to the same scenery rendered without the HUD, and
        // closer than stepping past the HUD to its surroundings.
        {
            own_engine = true;
            auto band_error = [&](const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
                double sum = 0; int n = 0;
                for (LONG y = hy0 + 2; y < hy1 - 2; ++y)
                    for (LONG x = std::max<LONG>(0, hx0 - 48); x < hx1 + 48; ++x) {
                        if (x >= hx0 - 2 && x < hx1 + 2) continue;  // the held HUD itself
                        const std::size_t i = (std::size_t(y) * w + std::size_t(x)) * 4;
                        for (int k = 0; k < 3; ++k) sum += std::fabs(half_to_float(a[i + k]) - half_to_float(b[i + k]));
                        n += 3;
                    }
                return n ? sum / n : 0.0;
            };
            IngestedSource clean_src = ingest_frame(publish(604, 0.1f, 0, false, false), true, false);
            show(clean_src, yaw);
            const std::vector<std::uint16_t> truth = px;
            hud_fill = true;
            IngestedSource fill_src = ingest_frame(publish(605, 0.1f, 0, false, true), true, false);
            show(fill_src, yaw);
            const double filled = band_error(px, truth), fp_x = green_x();
            hud_fill = false;
            IngestedSource step_src = ingest_frame(publish(606, 0.1f, 0, false, true), true, false);
            show(step_src, yaw);
            const double stepped = band_error(px, truth);
            own_engine = false;
            std::printf("fill behind the HUD: error where the HUD moved off %.4f (stepping past it %.4f), patch x %.1f\n", filled, stepped, fp_x);
            EXPECT(filled < 0.02 && filled < 0.5 * stepped, "the fill shows the scenery behind the HUD (%.4f vs %.4f)", filled, stepped);
            EXPECT(std::fabs(fp_x - op_x0) < 2.0, "with the fill the HUD itself is still held (%.1f)", fp_x);
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
        // (O) Combined HUD detection. Scenery missing from the upscaler's output is taken for HUD from the
        // output alone; with the camera-motion check it warps, as it follows the world while the camera
        // moves (6 px per frame here). The HUD patch in the same frames stays held.
        auto magenta_x = [&]() {
            double sum = 0, weight = 0;
            for (std::uint32_t y = std::uint32_t(neon_y0); y < std::uint32_t(neon_y1); ++y)
                for (std::uint32_t x = 0; x < w; ++x) {
                    const std::size_t i = (std::size_t(y) * w + x) * 4;
                    const float m = std::min(half_to_float(px[i]), half_to_float(px[i + 2])) - half_to_float(px[i + 1]);
                    if (m > 0.5f) { sum += x * m; weight += m; }
                }
            return weight > 0 ? sum / weight : -1.0;
        };
        renderer.reset_hud_detection();
        combined_hud = true;
        IngestedSource neon_src{};
        for (int k = 0; k < 4; ++k) {
            neon_x = LONG(W * 5 / 8) - 6 * k;
            neon_src = ingest_frame(publish(731 + k, 0.1f, 0, false, true), true, false);
        }
        show(neon_src, 0);
        const double cn_x0 = magenta_x(), ch_x0 = green_x();
        show(neon_src, yaw);
        const double cn_x1 = magenta_x(), ch_x1 = green_x();
        combined_hud = false;
        neon_x = LONG(W * 5 / 8) - 24;
        IngestedSource neon_only = ingest_frame(publish(735, 0.1f, 0, false, true), true, false);
        neon_x = -1;
        show(neon_only, 0);
        const double on_x0 = magenta_x();
        show(neon_only, yaw);
        const double on_x1 = magenta_x();
        std::printf("combined HUD detection: post-processed scenery x %.1f -> %.1f (output only: %.1f -> %.1f), HUD patch x %.1f -> %.1f\n",
                    cn_x0, cn_x1, on_x0, on_x1, ch_x0, ch_x1);
        EXPECT(on_x0 > 0 && std::fabs(on_x1 - on_x0) < 2.0, "the test scenery is taken for HUD from the upscaler's output alone (%.1f -> %.1f)", on_x0, on_x1);
        EXPECT(cn_x0 > 0 && std::fabs(cn_x1 - cn_x0) > 20.0, "combined: scenery that follows the world warps (%.1f -> %.1f)", cn_x0, cn_x1);
        EXPECT(ch_x0 > 0 && std::fabs(ch_x1 - ch_x0) < 2.0, "combined: the HUD is still held (%.1f -> %.1f)", ch_x0, ch_x1);
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
                renderer.pass_stamp(Renderer::kPassStart);
                renderer.analyze_motion(s, moving_cam.clip_to_prev_clip, 1.0f, 1.0f, true);
                renderer.pass_stamp(Renderer::kPassAnalyze);
                renderer.pass_stamp(Renderer::kPassObjects);
                renderer.build_no_warp_mask(s, moving_cam.clip_to_prev_clip, true, true, true);
                renderer.submit_work();
                renderer.wait_idle();
                if (i == 3) {
                    renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); renderer.take_gpu_usage();
                    (void)renderer.take_pass_times();
                }
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
            const Renderer::PassTimes pt = renderer.take_pass_times();
            if (pt.frames)
                std::printf("game frame GPU by pass (ms, %u frames): motion analysis %.3f, HUD %.3f, character/weapon + stretch %.3f, mask %.3f\n", pt.frames,
                            pt.ms[Renderer::kPassAnalyze] / pt.frames, pt.ms[Renderer::kPassHud] / pt.frames, pt.ms[Renderer::kPassCharacter] / pt.frames,
                            pt.ms[Renderer::kPassMask] / pt.frames);
            EXPECT(pt.frames > 0, "the game frame's passes are timed");
            {  // ...and with the camera-motion check of the combined HUD detection (the default)
                (void)renderer.take_pass_times();
                for (int i = 0; i < 24; ++i) {
                    const int bs = publish(930 + i, 0.1f, 0, true, true);
                    if (bs < 0) continue;
                    renderer.begin_frame();
                    IngestedSource s = renderer.ingest(sh, bs);
                    renderer.pass_stamp(Renderer::kPassStart);
                    renderer.analyze_motion(s, moving_cam.clip_to_prev_clip, 1.0f, 1.0f, true);
                    renderer.pass_stamp(Renderer::kPassAnalyze);
                    renderer.pass_stamp(Renderer::kPassObjects);
                    renderer.build_no_warp_mask(s, moving_cam.clip_to_prev_clip, true, true, true, true, true);
                    renderer.submit_work();
                    renderer.wait_idle();
                    if (i == 3) (void)renderer.take_pass_times();
                }
                for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
                const Renderer::PassTimes pc = renderer.take_pass_times();
                if (pc.frames)
                    std::printf("game frame GPU by pass with the camera-motion check (ms, %u frames): HUD %.3f, mask %.3f\n", pc.frames,
                                pc.ms[Renderer::kPassHud] / pc.frames, pc.ms[Renderer::kPassMask] / pc.frames);
                (void)renderer.take_hud_stats();
                renderer.take_gpu_usage();
            }
            if (hs.scene_frames) {
                std::printf("HUD from output: %.6f%% of the screen; GPU ms per pass:", 100.0 * hs.scene_share_sum / hs.scene_frames);
                for (int i = 0; i < 18; ++i) std::printf(" %.3f", hs.scene_pass_ms[i] / hs.scene_frames);
                std::printf("\n");
            }
        }
        // GPU cost of the background memory: the same intake with and without it (the difference is the pass).
        {
            auto intake_rest = [&](bool with_memory, std::uint64_t first_id) {
                renderer.take_gpu_usage();
                for (int i = 0; i < 12; ++i) {
                    const int bs = publish(first_id + i, 0.1f, 0, true, true);
                    if (bs < 0) continue;
                    renderer.begin_frame();
                    IngestedSource s = renderer.ingest(sh, bs);
                    renderer.analyze_motion(s, moving_cam.clip_to_prev_clip, 1.0f, 1.0f, true);
                    renderer.build_no_warp_mask(s, moving_cam.clip_to_prev_clip, true, true, true);
                    if (with_memory) renderer.update_memory(s, moving_cam.clip_to_prev_clip, true, i > 0);
                    renderer.submit_work();
                    renderer.wait_idle();
                    if (i == 3) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); renderer.take_gpu_usage(); }
                }
                for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
                const Renderer::GpuUsage u = renderer.take_gpu_usage();
                return u.split ? u.rest_ms / u.split : 0.0;
            };
            const double without = intake_rest(false, 930), with = intake_rest(true, 942);
            std::printf("background memory GPU at %ux%u: %.3f ms per game frame (HUD/masks/motion %.3f with it, %.3f without)\n",
                        W, H, with - without, with, without);
        }
        // (Q) Split queues: frames taken in on the intake (compute) queue into one set of textures while
        // the warp reads the shown set; the warp switches only once the new set is complete and shown.
        {
            own_engine = true;
            const bool split_ok = renderer.set_split(true);
            EXPECT(split_ok && renderer.split(), "split queues available");
            auto intake_split = [&](int s_slot) {
                EXPECT(s_slot >= 0, "frame published");
                renderer.set_keep_previous_colour(true);
                renderer.begin_intake();
                IngestedSource s = renderer.ingest(sh, s_slot);
                renderer.analyze_motion(s, moving_cam.clip_to_prev_clip, 1.0f, 1.0f, true);
                renderer.build_no_warp_mask(s, moving_cam.clip_to_prev_clip, true, true, true, true, true, true);
                renderer.submit_work();
                renderer.mark_intake_complete();
                return s;
            };
            IngestedSource first = intake_split(publish(960, 0.1f, 0, false, true));
            renderer.wait_for_intake(renderer.intake_submitted());
            const bool shown_a = renderer.show_intake();
            show(first, 0);
            const double a_x = peak(px, w, h, true, 1);
            // A second frame (bar 40 px to the right) is being taken in: until it is shown, the warp keeps
            // showing the first one, in full.
            IngestedSource second = intake_split(publish(961, 0.1f, 40, false, true));
            show(first, 0);
            const double during_x = peak(px, w, h, true, 1);
            renderer.wait_for_intake(renderer.intake_submitted());
            const bool shown_b = renderer.show_intake();
            show(second, 0);
            const double b_x = peak(px, w, h, true, 1);
            show(second, yaw);
            const double b_turned = peak(px, w, h, true, 1);
            // An unwarped refresh shows the shown frame too.
            renderer.begin_frame();
            renderer.finish_frame(false, 0);
            renderer.read_back(1, px, w, h);
            const double plain_x = peak(px, w, h, true, 1);
            // ...exactly the shown frame, and its GPU time per refresh (back to back: a GPU left idle in between
            // clocks down).
            {
                std::vector<std::uint16_t> shown_px;
                std::uint32_t sw = 0, sh2 = 0;
                renderer.read_back(0, shown_px, sw, sh2);
                std::size_t differ = 0;
                for (std::size_t i = 0; i < std::min(px.size(), shown_px.size()); ++i) differ += px[i] != shown_px[i];
                renderer.wait_idle();
                renderer.take_gpu_usage();
                for (int i = 0; i < 200; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); }
                renderer.wait_idle();
                for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
                const Renderer::GpuUsage u = renderer.take_gpu_usage();
                std::printf("unwarped refresh with split queues: %.3f ms GPU, %zu values differ from the shown frame\n",
                            u.warps ? u.warp_ms / u.warps : 0.0, differ);
                EXPECT(sw == w && sh2 == h && differ == 0, "an unwarped refresh shows exactly the shown frame (%zu values differ)", differ);
            }
            // Back to the single queue: the shown frame stays.
            const bool back = renderer.set_split(false);
            show(second, 0);
            const double after_x = peak(px, w, h, true, 1);
            {  // (for comparison: an unwarped refresh on one queue - the blit alone)
                renderer.wait_idle();
                renderer.take_gpu_usage();
                for (int i = 0; i < 200; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); }
                renderer.wait_idle();
                for (int i = 0; i < 3; ++i) { renderer.begin_frame(); renderer.finish_frame(false, 0); renderer.wait_idle(); }
                const Renderer::GpuUsage u = renderer.take_gpu_usage();
                std::printf("unwarped refresh on one queue: %.3f ms GPU\n", u.warps ? u.warp_ms / u.warps : 0.0);
            }
            own_engine = false;
            std::printf("split queues: first frame bar x %.0f; while the second is taken in %.0f; second %.0f (turned %.0f), unwarped %.0f, "
                        "back on one queue %.0f\n", a_x, during_x, b_x, b_turned, plain_x, after_x);
            EXPECT(shown_a && shown_b, "each completed frame is shown");
            EXPECT(std::fabs(during_x - a_x) < 1.0, "while a frame is taken in, the warp shows the previous one (%.0f vs %.0f)", during_x, a_x);
            EXPECT(std::fabs(b_x - a_x - 40.0) < 2.0, "the new frame is shown once complete (%.0f vs %.0f)", b_x, a_x);
            EXPECT(std::fabs(b_turned - b_x) > 20.0, "the shown frame warps (%.0f -> %.0f)", b_x, b_turned);
            EXPECT(std::fabs(plain_x - b_x) < 1.0, "an unwarped refresh shows the shown frame (%.0f)", plain_x);
            EXPECT(back && std::fabs(after_x - b_x) < 1.0, "back on one queue the shown frame stays (%.0f)", after_x);
            // Without the background memory, a held area wider than the fill's search reach keeps its own colour
            // where nothing unheld is found - never anything read from the memory's slot (a red depth value).
            own_engine = true;
            strip_half = 150;
            IngestedSource wide = ingest_frame(publish(962, 0.1f, -400, true, false), false, true);
            strip_half = 8;
            show(wide, yaw * 4.0);  // (a gap wider than the fill's 96 px search reach)
            own_engine = false;
            int reddish = 0;
            for (std::size_t i = 0; i + 3 < px.size(); i += 4)
                if (half_to_float(px[i]) > 0.004f && half_to_float(px[i + 1]) < 0.002f && half_to_float(px[i + 2]) < 0.002f) ++reddish;
            std::printf("wide held area without the background memory: red pixels %d\n", reddish);
            // The stretch: a weapon outline the motion vectors miss slides away from the weapon as a ghost
            // (white pixels left of the strip after the turn) without it; with it, it stays with the weapon.
            own_engine = true;
            fringe = true;
            IngestedSource edge1 = ingest_frame(publish(963, 0.1f, -400, true, false), false, true);
            show(edge1, yaw);
            const LONG strip_l = LONG((DW / 2 - 12) * W / DW);
            auto ghost = [&]() {
                int n = 0;
                for (std::uint32_t y = h / 4; y < h * 3 / 4; ++y)
                    for (LONG x = std::max<LONG>(0, strip_l - LONG(60 * W / 1280)); x < strip_l - 12; ++x) {
                        const std::size_t i = (std::size_t(y) * w + std::size_t(x)) * 4;
                        if (half_to_float(px[i]) > 0.8f && half_to_float(px[i + 1]) > 0.8f && half_to_float(px[i + 2]) > 0.8f) ++n;
                    }
                return n;
            };
            const int ghost1 = ghost();
            auto dump = [&](const char* name) {  // (FW_DUMP: the output as raw half floats, for a look)
                if (!std::getenv("FW_DUMP")) return;
                if (FILE* f = std::fopen(name, "wb")) { std::fwrite(&w, 4, 1, f); std::fwrite(&h, 4, 1, f); std::fwrite(px.data(), 2, px.size(), f); std::fclose(f); }
            };
            dump("stretch_off.f16");
            stretch = 16;
            IngestedSource stretched = ingest_frame(publish(964, 0.1f, -400, true, false), false, true);
            stretch = 0;
            fringe = false;
            show(stretched, yaw);
            const int ghost4 = ghost();
            dump("stretch_on.f16");
            own_engine = false;
            std::printf("stretch: ghost outline pixels %d without, %d with 16 render px\n", ghost1, ghost4);
            EXPECT(ghost1 > 0, "the test outline slides away as a ghost without the stretch (%d)", ghost1);
            EXPECT(ghost4 == 0, "with the stretch the outline stays with the weapon (%d)", ghost4);
            EXPECT(reddish == 0, "without the memory nothing is read from its slot (%d red pixels)", reddish);
            // Captures keep two earlier game frames (stash sets 1 and 0) next to the current one.
            {
                ingest_frame(publish(970, 0.1f, 0, false, false), false, false);
                renderer.stash_source(1);
                ingest_frame(publish(971, 0.1f, 20, false, false), false, false);
                renderer.stash_source(0);
                ingest_frame(publish(972, 0.1f, 40, false, false), false, false);
                double bars[3] = {};
                bool depths = true;
                for (int i = 0; i < 3; ++i) {
                    EXPECT(renderer.read_back(i == 0 ? 15 : i == 1 ? 10 : 0, px, w, h), "kept frame %d read back", i);
                    bars[i] = peak(px, w, h, true, 1);
                    std::vector<std::uint16_t> d; std::uint32_t dw = 0, dh = 0;
                    depths = depths && renderer.read_back(i == 0 ? 17 : i == 1 ? 11 : 3, d, dw, dh) && dw > 0;
                }
                std::printf("captured frames: bar x %.0f, %.0f, %.0f\n", bars[0], bars[1], bars[2]);
                EXPECT(std::fabs(bars[1] - bars[0] - 20.0) < 2.0 && std::fabs(bars[2] - bars[1] - 20.0) < 2.0,
                       "the two kept frames are the earlier ones, oldest first (%.0f, %.0f, %.0f)", bars[0], bars[1], bars[2]);
                EXPECT(depths, "their depth is kept too");
                EXPECT(!renderer.read_back(16, px, w, h), "no HUD-less picture is kept when not asked for");
            }
        }
        with_scene = false;
        scene_offset = -1;
        // (L) HUD from the game's HUD-less picture without a UI layer: exactly where the frame differs from it,
        // in the very first frame (nothing learned), through FrameWarp's own engine; the scene around it warps.
        {
            renderer.reset_hud_detection();
            with_hudless = true;
            IngestedSource hl_src = ingest_frame(publish(980, 0.1f, 0, false, true), true, false);
            with_hudless = false;
            own_engine = true;
            show(hl_src, 0);
            const double hp_x0 = green_x(), hb_x0 = peak(px, w, h, true, 1);
            show(hl_src, yaw);
            const double hp_x1 = green_x(), hb_x1 = peak(px, w, h, true, 1);
            own_engine = false;
            std::printf("HUD from the HUD-less picture: patch x %.1f -> %.1f, bar x %.0f -> %.0f\n", hp_x0, hp_x1, hb_x0, hb_x1);
            EXPECT(hl_src.has_hudless && !hl_src.has_ui, "the HUD-less picture is ingested without a UI layer");
            EXPECT(hp_x0 > 0 && std::fabs(hp_x1 - hp_x0) < 2.0, "HUD found from the HUD-less picture in one frame (%.1f -> %.1f)", hp_x0, hp_x1);
            EXPECT(std::fabs(hb_x1 - hb_x0) > 20.0, "the scene next to that HUD still warps (%.0f -> %.0f)", hb_x0, hb_x1);
        }
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

    // XPAR's own motion estimation: a frame with depth but without motion vectors, and a camera to estimate
    // (ReShade's depth, no DLSS or FSR). A textured picture displaced between two frames: the motion
    // texture must say where every pixel came from, and say so with confidence.
    {
        ComPtr<ID3D12Fence> gf; game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gf));
        UINT64 fence_value = 0;
        auto publish = [&](std::uint64_t fid, LONG ox, LONG oy) -> int {
            alloc->Reset();
            list->Reset(alloc.Get(), nullptr);
            list->ResourceBarrier(1, &b);  // backbuffer PRESENT -> RENDER_TARGET
            const float back[4] = {0.2f, 0.2f, 0.2f, 1};
            list->ClearRenderTargetView(rtv, back, 0, nullptr);
            // The same rectangles every frame, displaced by (ox, oy): large shapes first, then smaller and
            // smaller ones on top (detail at every scale, like a real picture), sized with the picture.
            std::uint32_t seed = 12345;
            auto next = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
            const LONG margin = LONG(W / 4);
            for (int i = 0; i < 2000; ++i) {
                const LONG unit = i < 60 ? LONG(W / 8) : i < 400 ? LONG(W / 32) : LONG(W / 128);
                const LONG x = LONG(next() % (W + 2 * margin)) - margin + ox, y = LONG(next() % (H + 2 * margin)) - margin + oy;
                const LONG rw = unit / 2 + LONG(next() % unit), rh = unit / 2 + LONG(next() % unit);
                const float g = 0.1f + float(next() % 100) / 125.0f;
                const float grey[4] = {g, g, g, 1};
                const D3D12_RECT r{std::clamp<LONG>(x, 0, LONG(W)), std::clamp<LONG>(y, 0, LONG(H)), std::clamp<LONG>(x + rw, 0, LONG(W)),
                                   std::clamp<LONG>(y + rh, 0, LONG(H))};
                if (r.right > r.left && r.bottom > r.top) list->ClearRenderTargetView(rtv, grey, 1, &r);
            }
            list->ClearDepthStencilView(dsv_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 0.01f, 0, 0, nullptr);
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            list->ResourceBarrier(1, &b);
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            Camera to_estimate{};
            to_estimate.valid = 1; to_estimate.estimated = 1; to_estimate.depth_inverted = 1;
            to_estimate.mvec_scale[0] = float(DW); to_estimate.mvec_scale[1] = float(DH);
            producer.on_constants(fid, to_estimate);
            producer.on_tag(fid, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
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
        auto take_in = [&](int s) {
            renderer.begin_frame();
            const IngestedSource src = renderer.ingest(sh, s);
            renderer.submit_work();
            renderer.wait_idle();
            return src;
        };
        auto median = [](std::vector<float> v) { if (v.empty()) return 0.0f; std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
        renderer.reset_hud_detection();  // (no previous frame)
        std::uint64_t fid = 2000;
        const int first = publish(fid, 0, 0);
        EXPECT(first >= 0, "frame without motion vectors published");
        const IngestedSource before = first >= 0 ? take_in(first) : IngestedSource{};
        EXPECT(before.has_motion && before.motion_estimated, "a frame with depth and no motion vectors gets estimated ones");
        // A slow turn, a fast one (past what the refinements alone can follow: the coarse search), and none.
        LONG at_x = 0, at_y = 0;
        for (const auto& [dx, dy] : {std::pair<LONG, LONG>{24, 6}, std::pair<LONG, LONG>{-150, 20}, std::pair<LONG, LONG>{0, 0}}) {
            at_x += dx; at_y += dy;
            const int s = publish(++fid, at_x, at_y);
            if (s < 0) { EXPECT(false, "frame published"); continue; }
            take_in(s);
            std::vector<std::uint16_t> mv, flow;
            std::uint32_t mw = 0, mh = 0, fw2 = 0, fh2 = 0;
            EXPECT(renderer.read_back(4, mv, mw, mh) && renderer.read_back(13, flow, fw2, fh2), "estimated motion read back");
            // (the displacements' texture holds a pyramid; level 0, half the depth grid, is at its origin)
            const std::uint32_t stride = fw2;
            fw2 = std::max(1u, DW / 2); fh2 = std::max(1u, DH / 2);
            // The picture moved by (dx, dy): every pixel was at (-dx, -dy) from where it is, in px of the picture.
            std::vector<float> ex, ey, all_x;
            std::size_t confident = 0;
            for (std::uint32_t y = fh2 / 8; y < fh2 * 7 / 8; ++y)
                for (std::uint32_t x = fw2 / 8; x < fw2 * 7 / 8; ++x) {
                    const std::size_t i = (std::size_t(y) * stride + x) * 4;
                    const float mx = half_to_float(flow[i]) * float(W) / float(fw2), my = half_to_float(flow[i + 1]) * float(H) / float(fh2);
                    all_x.push_back(std::fabs(mx + float(dx)));
                    if (half_to_float(flow[i + 2]) < 0.5f) continue;
                    ++confident;
                    ex.push_back(std::fabs(mx + float(dx))); ey.push_back(std::fabs(my + float(dy)));
                }
            const float tex_x = half_to_float(mv[(std::size_t(mh / 2) * mw + mw / 2) * 4]) * float(W), tex_y = half_to_float(mv[(std::size_t(mh / 2) * mw + mw / 2) * 4 + 1]) * float(H);
            const double share = double(confident) / double(std::max<std::size_t>(all_x.size(), 1));
            std::printf("own motion, picture moved by (%ld, %ld) px: confident %.0f%%, their median error %.2f, %.2f px (all pixels %.2f); motion texture at the centre (%.1f, %.1f)\n",
                        dx, dy, share * 100.0, median(ex), median(ey), median(all_x), tex_x, tex_y);
            // (flat rectangle interiors tell nothing: how much counts depends on the picture's size)
            EXPECT(share > 0.03, "the estimated motion is confident where the picture has detail (%.0f%%)", share * 100.0);
            EXPECT(median(ex) < 0.5f && median(ey) < 0.5f, "the confident motion is right (%.2f, %.2f px off)", median(ex), median(ey));
            EXPECT(std::fabs(tex_x + float(dx)) < 2.0f && std::fabs(tex_y + float(dy)) < 2.0f, "the motion texture holds it, in uv (%.1f, %.1f)", tex_x, tex_y);
            // Where nothing is known (the flat insides of the rectangles) the motion texture takes what is
            // known around it: nearly every pixel of it is right, not only the confident ones.
            std::vector<float> off;
            for (std::uint32_t y = mh / 8; y < mh * 7 / 8; y += 2)
                for (std::uint32_t x = mw / 8; x < mw * 7 / 8; x += 2) {
                    const std::size_t i = (std::size_t(y) * mw + x) * 4;
                    off.push_back(std::hypot(half_to_float(mv[i]) * float(W) + float(dx), half_to_float(mv[i + 1]) * float(H) + float(dy)));
                }
            std::sort(off.begin(), off.end());
            const float p95 = off.empty() ? 0.0f : off[off.size() * 95 / 100];
            if (std::getenv("FW_DUMP")) {  // (where the motion texture is off: share of pixels beyond 2 px per cell of an 8 x 4 grid)
                for (std::uint32_t cy = 0; cy < 4; ++cy) {
                    std::printf("   ");
                    for (std::uint32_t cx = 0; cx < 8; ++cx) {
                        int bad = 0, n = 0;
                        for (std::uint32_t y = cy * mh / 4; y < (cy + 1) * mh / 4; y += 2)
                            for (std::uint32_t x = cx * mw / 8; x < (cx + 1) * mw / 8; x += 2) {
                                const std::size_t i = (std::size_t(y) * mw + x) * 4;
                                ++n;
                                if (std::hypot(half_to_float(mv[i]) * float(W) + float(dx), half_to_float(mv[i + 1]) * float(H) + float(dy)) > 2.0f) ++bad;
                            }
                        std::printf("%4d%%", n ? 100 * bad / n : 0);
                    }
                    std::printf("\n");
                }
                // one cell in detail: the displacements found there (level 0) and the motion texture
                const std::uint32_t x0 = fw2 * 2 / 8, x1 = fw2 * 3 / 8, y0 = 0, y1 = fh2 / 4;
                int known = 0, known_bad = 0, all = 0, all_bad = 0;
                for (std::uint32_t y = y0; y < y1; ++y)
                    for (std::uint32_t x = x0; x < x1; ++x) {
                        const std::size_t i = (std::size_t(y) * stride + x) * 4;
                        const float e = std::hypot(half_to_float(flow[i]) * float(W) / float(fw2) + float(dx), half_to_float(flow[i + 1]) * float(H) / float(fh2) + float(dy));
                        ++all; if (e > 2.0f) ++all_bad;
                        if (half_to_float(flow[i + 2]) > 0.5f) { ++known; if (e > 2.0f) ++known_bad; }
                    }
                const std::size_t mi = (std::size_t(mh / 8) * mw + mw * 5 / 16) * 4;
                std::printf("   cell (row 0, column 2): %d of %d displacements known, %d of those wrong; %d of all wrong; motion texture in it (%.1f, %.1f) px\n",
                            known, all, known_bad, all_bad, half_to_float(mv[mi]) * float(W), half_to_float(mv[mi + 1]) * float(H));
            }
            std::vector<float> conf_err;
            for (std::size_t i = 0; i < ex.size(); ++i) conf_err.push_back(std::hypot(ex[i], ey[i]));
            std::sort(conf_err.begin(), conf_err.end());
            std::printf("   motion texture, every pixel: 95%% within %.2f px; the confident displacements: 95%% within %.2f px, 99%% within %.2f px\n", p95,
                        conf_err.empty() ? 0.0f : conf_err[conf_err.size() * 95 / 100], conf_err.empty() ? 0.0f : conf_err[conf_err.size() * 99 / 100]);
            // (not checked for the fast turn: where this picture has little detail - the large rectangles -
            // the coarse search finds nothing to hold on to, and a whole area is wrong and unknown alike)
            if (std::abs(dx) < 100) EXPECT(p95 < 2.0f, "the motion texture is right where nothing was known too (95%% within %.2f px)", p95);
        }
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
    // A packed HDR UI target has only two alpha bits, but its premultiplied RGB and the final composite
    // still contain enough information to recover the intended alpha.
    {
        if (renderer.split()) {
            renderer.wait_idle();
            EXPECT(renderer.set_split(false), "single queue for UI-alpha test");
        }
        renderer.wait_idle();

        D3D12_RESOURCE_DESC layer_desc = cd;
        layer_desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
        const float hudless_colour[4] = {0.6f, 0.4f, 0.2f, 1.0f};
        const float ui_colour[4] = {0.0f, 0.0f, 0.0f, 0.4f};
        const float final_colour[4] = {0.36f, 0.24f, 0.12f, 1.0f};
        const float bgra_ui_colour[4] = {0.0f, 0.0f, 0.0f, 0.3f};
        const float inverted_alpha_final[4] = {0.18f, 0.12f, 0.06f, 1.0f};
        auto make_layer = [&](DXGI_FORMAT format, D3D12_RESOURCE_STATES state, const float* clear,
                              ComPtr<ID3D12Resource>& resource) {
            D3D12_RESOURCE_DESC desc = cd;
            desc.Format = format;
            D3D12_CLEAR_VALUE cv{}; cv.Format = format;
            std::memcpy(cv.Color, clear, sizeof(cv.Color));
            const HRESULT hr = game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, &cv, IID_PPV_ARGS(&resource));
            EXPECT(SUCCEEDED(hr), "create UI-alpha test layer");
            return SUCCEEDED(hr);
        };
        const float ui_alpha_colour[4] = {0.4f, 0.0f, 0.0f, 0.0f};
        ComPtr<ID3D12Resource> alpha_backbuffer, hudless_layer, ui_layer, ui_alpha_layer, bgra_ui_layer;
        const bool layers_created = make_layer(layer_desc.Format, D3D12_RESOURCE_STATE_RENDER_TARGET, final_colour, alpha_backbuffer) &&
                                    make_layer(layer_desc.Format, D3D12_RESOURCE_STATE_RENDER_TARGET, hudless_colour, hudless_layer) &&
                                    make_layer(layer_desc.Format, D3D12_RESOURCE_STATE_RENDER_TARGET, ui_colour, ui_layer) &&
                                    make_layer(DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_STATE_RENDER_TARGET, ui_alpha_colour, ui_alpha_layer) &&
                                    make_layer(DXGI_FORMAT_B8G8R8A8_UNORM, D3D12_RESOURCE_STATE_RENDER_TARGET, bgra_ui_colour, bgra_ui_layer);
        ComPtr<ID3D12DescriptorHeap> layer_rtvs;
        D3D12_DESCRIPTOR_HEAP_DESC layer_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 5, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        const HRESULT rtv_hr = game->CreateDescriptorHeap(&layer_heap_desc, IID_PPV_ARGS(&layer_rtvs));
        EXPECT(SUCCEEDED(rtv_hr), "create UI-alpha test RTVs");
        ComPtr<ID3D12Fence> alpha_test_fence;
        const HRESULT fence_hr = game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&alpha_test_fence));
        EXPECT(SUCCEEDED(fence_hr), "create UI-alpha test fence");
        if (layers_created && SUCCEEDED(rtv_hr) && SUCCEEDED(fence_hr)) {
            const D3D12_CPU_DESCRIPTOR_HANDLE layer_rtv0 = layer_rtvs->GetCPUDescriptorHandleForHeapStart();
            D3D12_CPU_DESCRIPTOR_HANDLE layer_rtv1 = layer_rtv0, layer_rtv2 = layer_rtv0, layer_rtv3 = layer_rtv0, layer_rtv4 = layer_rtv0;
            const SIZE_T layer_rtv_step = game->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            layer_rtv1.ptr += layer_rtv_step; layer_rtv2.ptr += 2 * layer_rtv_step; layer_rtv3.ptr += 3 * layer_rtv_step;
            layer_rtv4.ptr += 4 * layer_rtv_step;
            game->CreateRenderTargetView(alpha_backbuffer.Get(), nullptr, layer_rtv0);
            game->CreateRenderTargetView(hudless_layer.Get(), nullptr, layer_rtv1);
            game->CreateRenderTargetView(ui_layer.Get(), nullptr, layer_rtv2);
            game->CreateRenderTargetView(ui_alpha_layer.Get(), nullptr, layer_rtv3);
            game->CreateRenderTargetView(bgra_ui_layer.Get(), nullptr, layer_rtv4);

            HANDLE alpha_test_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            EXPECT(alpha_test_event != nullptr, "create UI-alpha test synchronization event");
            if (!alpha_test_event) return 1;
            if (alpha_test_event) {
                const HRESULT event_hr = alpha_test_fence->SetEventOnCompletion(1, alpha_test_event);
                const HRESULT signal_hr = queue->Signal(alpha_test_fence.Get(), 1);
                EXPECT(SUCCEEDED(event_hr) && SUCCEEDED(signal_hr), "subscribe before waiting for prior game work");
                const bool prior_work_complete = WaitForSingleObject(alpha_test_event, 10000) == WAIT_OBJECT_0;
                EXPECT(prior_work_complete, "prior game work completes");
                CloseHandle(alpha_test_event);
                if (!prior_work_complete) return 1;
            }
            alloc->Reset();
            list->Reset(alloc.Get(), nullptr);

            list->ClearRenderTargetView(layer_rtv0, final_colour, 0, nullptr);
            list->ClearRenderTargetView(layer_rtv1, hudless_colour, 0, nullptr);
            list->ClearRenderTargetView(layer_rtv2, ui_colour, 0, nullptr);
            list->ClearRenderTargetView(layer_rtv3, ui_alpha_colour, 0, nullptr);
            D3D12_RESOURCE_BARRIER to_present{};
            to_present.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            to_present.Transition = {alpha_backbuffer.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                     D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT};
            list->ResourceBarrier(1, &to_present);
            constexpr std::uint64_t alpha_test_frame = 10000;
            producer.on_constants(alpha_test_frame, cam);
            producer.on_tag(alpha_test_frame, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
            producer.on_tag(alpha_test_frame, kHudless, hudless_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
            producer.on_tag(alpha_test_frame, kUi, ui_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
            const auto alpha_token = producer.begin_present(alpha_backbuffer.Get(), list.Get());
            EXPECT(alpha_token != 0, "begin present for low-precision UI layer");
            list->Close();
            ID3D12CommandList* alpha_lists[] = {list.Get()};
            queue->ExecuteCommandLists(1, alpha_lists);
            producer.finish_present(queue.Get(), alpha_token);

            HANDLE alpha_test_event2 = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            EXPECT(alpha_test_event2 != nullptr, "create UI-alpha publish event");
            if (!alpha_test_event2) return 1;
            if (alpha_test_event2) {
                const HRESULT event_hr = alpha_test_fence->SetEventOnCompletion(2, alpha_test_event2);
                const HRESULT signal_hr = queue->Signal(alpha_test_fence.Get(), 2);
                EXPECT(SUCCEEDED(event_hr) && SUCCEEDED(signal_hr), "subscribe before waiting for tagged frame copies");
                const bool frame_copies_complete = WaitForSingleObject(alpha_test_event2, 10000) == WAIT_OBJECT_0;
                EXPECT(frame_copies_complete, "tagged frame copies complete");
                CloseHandle(alpha_test_event2);
                if (!frame_copies_complete) return 1;
            }

            int alpha_slot = -1;
            for (int i = 0; i < kSlots; ++i)
                if (sh.slots[i].state == kReady && sh.slots[i].frame_id == alpha_test_frame) alpha_slot = i;
            EXPECT(alpha_slot >= 0, "low-precision UI frame published");
            if (alpha_slot >= 0) {
                renderer.begin_frame();
                const IngestedSource alpha_source = renderer.ingest(sh, alpha_slot);
                EXPECT(alpha_source.has_hudless && alpha_source.has_ui, "low-precision UI tags remain enabled");
                float identity[16]{};
                for (int i = 0; i < 16; ++i) identity[i] = i % 5 == 0 ? 1.0f : 0.0f;
                const bool own_ok = renderer.own_warp(alpha_source, true, false, identity, true);
                EXPECT(own_ok, "own warp accepts the reconstructed UI layer");
                renderer.finish_frame(own_ok, 0);
                EXPECT(renderer.read_back(true, px, w, h), "read back reconstructed XPAR UI");
                const std::size_t centre = (std::size_t(h / 2) * w + w / 2) * 4;
                EXPECT(std::fabs(half_to_float(px[centre]) - final_colour[0]) < 0.015f &&
                       std::fabs(half_to_float(px[centre + 1]) - final_colour[1]) < 0.015f &&
                       std::fabs(half_to_float(px[centre + 2]) - final_colour[2]) < 0.015f,
                       "XPAR's recovered alpha reproduces the black translucent overlay");

                auto* alpha_late_list = renderer.begin_frame();
                auto alpha_inputs = renderer.latewarp_inputs(alpha_source, true);
                alpha_inputs.depth_inverted = true;
                const bool late_ok = latewarp.evaluate(alpha_late_list, alpha_inputs, true, view_matrix(source, {}),
                                                       view_matrix(source, {}), projection);
                EXPECT(late_ok, "Latewarp accepts the same reconstructed UI layer");
                renderer.finish_frame(late_ok, 0);
                EXPECT(renderer.read_back(true, px, w, h), "read back reconstructed Latewarp UI");
                EXPECT(std::fabs(half_to_float(px[centre]) - final_colour[0]) < 0.015f &&
                       std::fabs(half_to_float(px[centre + 1]) - final_colour[1]) < 0.015f &&
                       std::fabs(half_to_float(px[centre + 2]) - final_colour[2]) < 0.015f,
                       "Latewarp preserves the black translucent overlay");
            }

            // Repeat with Streamline's separate, full-precision UI Alpha tag. It must repair the packed
            // UI color layer as well, and the feature remains active in both warp engines.
            constexpr std::uint64_t alpha_tag_frame = 10001;
            HANDLE alpha_test_event3 = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            EXPECT(alpha_test_event3 != nullptr, "create separate UI-alpha synchronization event");
            if (!alpha_test_event3) return 1;
            if (alpha_test_event3) {
                const HRESULT event_hr = alpha_test_fence->SetEventOnCompletion(3, alpha_test_event3);
                EXPECT(SUCCEEDED(event_hr), "subscribe before submitting separate UI-alpha frame");
                alloc->Reset();
                list->Reset(alloc.Get(), nullptr);
                D3D12_RESOURCE_BARRIER to_target{};
                to_target.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                to_target.Transition = {alpha_backbuffer.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                        D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET};
                list->ResourceBarrier(1, &to_target);
                list->ClearRenderTargetView(layer_rtv0, final_colour, 0, nullptr);
                list->ClearRenderTargetView(layer_rtv1, hudless_colour, 0, nullptr);
                list->ClearRenderTargetView(layer_rtv2, ui_colour, 0, nullptr);
                list->ClearRenderTargetView(layer_rtv3, ui_alpha_colour, 0, nullptr);
                std::swap(to_target.Transition.StateBefore, to_target.Transition.StateAfter);
                list->ResourceBarrier(1, &to_target);
                producer.on_constants(alpha_tag_frame, cam);
                producer.on_tag(alpha_tag_frame, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
                producer.on_tag(alpha_tag_frame, kHudless, hudless_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
                producer.on_tag(alpha_tag_frame, kUi, ui_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
                producer.on_tag(alpha_tag_frame, kUiAlpha, ui_alpha_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
                const auto alpha_tag_token = producer.begin_present(alpha_backbuffer.Get(), list.Get());
                EXPECT(alpha_tag_token != 0, "begin present with separate UI Alpha tag");
                list->Close();
                ID3D12CommandList* alpha_tag_lists[] = {list.Get()};
                queue->ExecuteCommandLists(1, alpha_tag_lists);
                producer.finish_present(queue.Get(), alpha_tag_token);
                const HRESULT signal_hr = queue->Signal(alpha_test_fence.Get(), 3);
                EXPECT(SUCCEEDED(signal_hr), "signal completion for separate UI-alpha frame");
                const bool alpha_tag_copies_complete = WaitForSingleObject(alpha_test_event3, 10000) == WAIT_OBJECT_0;
                EXPECT(alpha_tag_copies_complete, "separate UI-alpha copies complete");
                CloseHandle(alpha_test_event3);
                if (!alpha_tag_copies_complete) return 1;

                int alpha_tag_slot = -1;
                for (int i = 0; i < kSlots; ++i)
                    if (sh.slots[i].state == kReady && sh.slots[i].frame_id == alpha_tag_frame) alpha_tag_slot = i;
                EXPECT(alpha_tag_slot >= 0, "frame with separate UI Alpha published");
                if (alpha_tag_slot >= 0) {
                    renderer.begin_frame();
                    const IngestedSource alpha_tag_source = renderer.ingest(sh, alpha_tag_slot);
                    EXPECT(alpha_tag_source.has_hudless && alpha_tag_source.has_ui, "separate UI Alpha reconstructs a HUD layer");
                    float identity[16]{};
                    for (int i = 0; i < 16; ++i) identity[i] = i % 5 == 0 ? 1.0f : 0.0f;
                    const bool alpha_tag_own_ok = renderer.own_warp(alpha_tag_source, true, false, identity, true);
                    EXPECT(alpha_tag_own_ok, "own warp uses separate UI Alpha");
                    renderer.finish_frame(alpha_tag_own_ok, 0);
                    EXPECT(renderer.read_back(true, px, w, h), "read back separate-alpha XPAR UI");
                    const std::size_t tag_centre = (std::size_t(h / 2) * w + w / 2) * 4;
                    EXPECT(std::fabs(half_to_float(px[tag_centre]) - final_colour[0]) < 0.015f &&
                           std::fabs(half_to_float(px[tag_centre + 1]) - final_colour[1]) < 0.015f &&
                           std::fabs(half_to_float(px[tag_centre + 2]) - final_colour[2]) < 0.015f,
                           "XPAR recomposes packed UI color with full-precision alpha");

                    auto* alpha_tag_late_list = renderer.begin_frame();
                    auto alpha_tag_inputs = renderer.latewarp_inputs(alpha_tag_source, true);
                    alpha_tag_inputs.depth_inverted = true;
                    const bool alpha_tag_late_ok = latewarp.evaluate(alpha_tag_late_list, alpha_tag_inputs, true,
                                                                      view_matrix(source, {}), view_matrix(source, {}), projection);
                    EXPECT(alpha_tag_late_ok, "Latewarp uses separate UI Alpha");
                    renderer.finish_frame(alpha_tag_late_ok, 0);
                    EXPECT(renderer.read_back(true, px, w, h), "read back separate-alpha Latewarp UI");
                    EXPECT(std::fabs(half_to_float(px[tag_centre]) - final_colour[0]) < 0.015f &&
                           std::fabs(half_to_float(px[tag_centre + 1]) - final_colour[1]) < 0.015f &&
                           std::fabs(half_to_float(px[tag_centre + 2]) - final_colour[2]) < 0.015f,
                           "Latewarp recomposes packed UI color with full-precision alpha");
                }
            }

            // Reproduce the game-reported BGRA8 UI / HDR10 frame pair. Here alpha is the
            // transparent share (0.3), so the black dim layer leaves 30% of the background.
            if (layer_desc.Format == DXGI_FORMAT_R10G10B10A2_UNORM) {
                constexpr std::uint64_t bgra_frame = 10002;
                HANDLE bgra_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                EXPECT(bgra_event != nullptr, "create BGRA/HDR synchronization event");
                if (!bgra_event) return 1;
                const HRESULT bgra_event_hr = alpha_test_fence->SetEventOnCompletion(4, bgra_event);
                EXPECT(SUCCEEDED(bgra_event_hr), "subscribe before submitting BGRA/HDR frame");
                alloc->Reset();
                list->Reset(alloc.Get(), nullptr);
                D3D12_RESOURCE_BARRIER bgra_to_target{};
                bgra_to_target.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                bgra_to_target.Transition = {alpha_backbuffer.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                             D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET};
                list->ResourceBarrier(1, &bgra_to_target);
                list->ClearRenderTargetView(layer_rtv0, inverted_alpha_final, 0, nullptr);
                list->ClearRenderTargetView(layer_rtv1, hudless_colour, 0, nullptr);
                list->ClearRenderTargetView(layer_rtv4, bgra_ui_colour, 0, nullptr);
                std::swap(bgra_to_target.Transition.StateBefore, bgra_to_target.Transition.StateAfter);
                list->ResourceBarrier(1, &bgra_to_target);
                producer.on_constants(bgra_frame, cam);
                producer.on_tag(bgra_frame, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
                producer.on_tag(bgra_frame, kHudless, hudless_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
                producer.on_tag(bgra_frame, kUi, bgra_ui_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
                const auto bgra_token = producer.begin_present(alpha_backbuffer.Get(), list.Get());
                EXPECT(bgra_token != 0, "begin present with BGRA8 UI on HDR10 frame");
                list->Close();
                ID3D12CommandList* bgra_lists[] = {list.Get()};
                queue->ExecuteCommandLists(1, bgra_lists);
                producer.finish_present(queue.Get(), bgra_token);
                const HRESULT bgra_signal_hr = queue->Signal(alpha_test_fence.Get(), 4);
                EXPECT(SUCCEEDED(bgra_signal_hr), "signal BGRA/HDR frame completion");
                const bool bgra_copies_complete = WaitForSingleObject(bgra_event, 10000) == WAIT_OBJECT_0;
                EXPECT(bgra_copies_complete, "BGRA/HDR UI copies complete");
                CloseHandle(bgra_event);
                if (!bgra_copies_complete) return 1;

                int bgra_slot = -1;
                for (int i = 0; i < kSlots; ++i)
                    if (sh.slots[i].state == kReady && sh.slots[i].frame_id == bgra_frame) bgra_slot = i;
                EXPECT(bgra_slot >= 0, "BGRA/HDR UI frame published");
                if (bgra_slot >= 0) {
                    renderer.begin_frame();
                    const IngestedSource bgra_source = renderer.ingest(sh, bgra_slot);
                    EXPECT(bgra_source.has_hudless && bgra_source.has_ui, "BGRA/HDR tags remain split for reprojection");
                    float identity[16]{};
                    for (int i = 0; i < 16; ++i) identity[i] = i % 5 == 0 ? 1.0f : 0.0f;
                    const bool bgra_own_ok = renderer.own_warp(bgra_source, true, false, identity, true);
                    EXPECT(bgra_own_ok, "XPAR accepts BGRA8 UI on HDR10");
                    renderer.finish_frame(bgra_own_ok, 0);
                    EXPECT(renderer.read_back(true, px, w, h), "read back normalized BGRA/HDR XPAR UI");
                    const std::size_t bgra_centre = (std::size_t(h / 2) * w + w / 2) * 4;
                    EXPECT(std::fabs(half_to_float(px[bgra_centre]) - inverted_alpha_final[0]) < 0.015f &&
                           std::fabs(half_to_float(px[bgra_centre + 1]) - inverted_alpha_final[1]) < 0.015f &&
                           std::fabs(half_to_float(px[bgra_centre + 2]) - inverted_alpha_final[2]) < 0.015f,
                           "XPAR chooses transparency alpha for the black dim layer");
                    std::vector<std::uint16_t> bgra_raw_ui, bgra_normalized_ui;
                    std::uint32_t raw_ui_w = 0, raw_ui_h = 0, normalized_ui_w = 0, normalized_ui_h = 0;
                    const bool raw_ui_read = renderer.read_back(15, bgra_raw_ui, raw_ui_w, raw_ui_h);
                    const bool normalized_ui_read = renderer.read_back(16, bgra_normalized_ui, normalized_ui_w, normalized_ui_h);
                    EXPECT(raw_ui_read && raw_ui_w == W && raw_ui_h == H, "diagnostics expose the raw BGRA8 UI layer");
                    EXPECT(normalized_ui_read && normalized_ui_w == W && normalized_ui_h == H,
                           "diagnostics expose the normalized BGRA8 UI layer");
                    if (raw_ui_read && normalized_ui_read) {
                        const std::size_t raw_centre = (std::size_t(H / 2) * W + W / 2) * 4;
                        const std::size_t normalized_centre = (std::size_t(H / 2) * W + W / 2) * 4;
                        EXPECT(std::fabs(half_to_float(bgra_raw_ui[raw_centre + 3]) - 0.3f) < 0.01f,
                               "raw BGRA8 UI diagnostic retains tagged alpha");
                        EXPECT(std::fabs(half_to_float(bgra_normalized_ui[normalized_centre + 3]) - 0.7f) < 0.01f,
                               "normalized BGRA8 UI diagnostic exposes reconstructed opacity");
                    }

                    auto* bgra_late_list = renderer.begin_frame();
                    auto bgra_inputs = renderer.latewarp_inputs(bgra_source, true);
                    bgra_inputs.depth_inverted = true;
                    const bool bgra_late_ok = latewarp.evaluate(bgra_late_list, bgra_inputs, true,
                                                                view_matrix(source, {}), view_matrix(source, {}), projection);
                    EXPECT(bgra_late_ok, "Latewarp accepts BGRA8 UI on HDR10");
                    renderer.finish_frame(bgra_late_ok, 0);
                    EXPECT(renderer.read_back(true, px, w, h), "read back normalized BGRA/HDR Latewarp UI");
                    EXPECT(std::fabs(half_to_float(px[bgra_centre]) - inverted_alpha_final[0]) < 0.015f &&
                           std::fabs(half_to_float(px[bgra_centre + 1]) - inverted_alpha_final[1]) < 0.015f &&
                           std::fabs(half_to_float(px[bgra_centre + 2]) - inverted_alpha_final[2]) < 0.015f,
                           "Latewarp chooses transparency alpha for the black dim layer");

                    // BGRA8 UI is not PQ-coded merely because the swapchain is HDR. Simulate a colored
                    // SDR UI contribution represented in the final PQ frame; no raw BGRA blend mode fits.
                    renderer.wait_idle();
                    constexpr std::uint64_t colored_frame = 10003;
                    const float colored_hudless[4] = {0.4f, 0.2f, 0.1f, 1.0f};
                    const float colored_ui[4] = {0.8f, 0.3f, 0.5f, 0.4f};
                    const float colored_final[4] = {0.6f, 0.25f, 0.20f, 1.0f};
                    HANDLE colored_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                    EXPECT(colored_event != nullptr, "create colored BGRA/PQ synchronization event");
                    if (!colored_event) return 1;
                    const HRESULT colored_event_hr = alpha_test_fence->SetEventOnCompletion(5, colored_event);
                    EXPECT(SUCCEEDED(colored_event_hr), "subscribe before submitting colored BGRA/PQ frame");
                    alloc->Reset();
                    list->Reset(alloc.Get(), nullptr);
                    D3D12_RESOURCE_BARRIER colored_to_target{};
                    colored_to_target.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    colored_to_target.Transition = {alpha_backbuffer.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                                    D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET};
                    list->ResourceBarrier(1, &colored_to_target);
                    list->ClearRenderTargetView(layer_rtv0, colored_final, 0, nullptr);
                    list->ClearRenderTargetView(layer_rtv1, colored_hudless, 0, nullptr);
                    list->ClearRenderTargetView(layer_rtv4, colored_ui, 0, nullptr);
                    std::swap(colored_to_target.Transition.StateBefore, colored_to_target.Transition.StateAfter);
                    list->ResourceBarrier(1, &colored_to_target);
                    producer.on_constants(colored_frame, cam);
                    producer.on_tag(colored_frame, kDepth, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, 0, 0, DW, DH, list.Get());
                    producer.on_tag(colored_frame, kHudless, hudless_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
                    producer.on_tag(colored_frame, kUi, bgra_ui_layer.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, 0, 0, W, H, list.Get());
                    const auto colored_token = producer.begin_present(alpha_backbuffer.Get(), list.Get());
                    EXPECT(colored_token != 0, "begin present with colored BGRA8 UI in PQ output");
                    list->Close();
                    ID3D12CommandList* colored_lists[] = {list.Get()};
                    queue->ExecuteCommandLists(1, colored_lists);
                    producer.finish_present(queue.Get(), colored_token);
                    const HRESULT colored_signal_hr = queue->Signal(alpha_test_fence.Get(), 5);
                    EXPECT(SUCCEEDED(colored_signal_hr), "signal colored BGRA/PQ frame completion");
                    const bool colored_copies_complete = WaitForSingleObject(colored_event, 10000) == WAIT_OBJECT_0;
                    EXPECT(colored_copies_complete, "colored BGRA/PQ UI copies complete");
                    CloseHandle(colored_event);
                    if (!colored_copies_complete) return 1;

                    int colored_slot = -1;
                    for (int i = 0; i < kSlots; ++i)
                        if (sh.slots[i].state == kReady && sh.slots[i].frame_id == colored_frame) colored_slot = i;
                    EXPECT(colored_slot >= 0, "colored BGRA/PQ UI frame published");
                    if (colored_slot >= 0) {
                        renderer.begin_frame();
                        const IngestedSource colored_source = renderer.ingest(sh, colored_slot);
                        EXPECT(colored_source.has_hudless && colored_source.has_ui, "colored BGRA/PQ tags remain split");
                        const bool colored_own_ok = renderer.own_warp(colored_source, true, false, identity, true);
                        EXPECT(colored_own_ok, "XPAR accepts colored BGRA8 UI on PQ output");
                        renderer.finish_frame(colored_own_ok, 0);
                        EXPECT(renderer.read_back(true, px, w, h), "read back colored BGRA/PQ XPAR UI");
                        const std::size_t colored_centre = (std::size_t(h / 2) * w + w / 2) * 4;
                        EXPECT(std::fabs(half_to_float(px[colored_centre]) - colored_final[0]) < 0.015f &&
                               std::fabs(half_to_float(px[colored_centre + 1]) - colored_final[1]) < 0.015f &&
                               std::fabs(half_to_float(px[colored_centre + 2]) - colored_final[2]) < 0.015f,
                               "XPAR preserves colored SDR UI contribution represented in the HDR frame");
                    }
                }
            }
        }
    }
    renderer.wait_idle();
    latewarp.shutdown();
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("pipeline tests passed\n");
    return 0;
}
