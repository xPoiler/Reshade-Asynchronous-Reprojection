// Direct3D 11 games (Producer::attach_d3d11): the textures and the fence the presenter opens are created on a
// D3D12 device and opened in the game's D3D11 device, which copies into them and signals the fence. This
// checks that path end to end on this machine's GPU: a D3D11 texture (colour, and a depth-like R32F one)
// copied into the shared textures, the fence signalled from the D3D11 immediate context, and what the D3D12
// side reads back after the fence.
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;
static int failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// A shared texture as Producer::ensure_texture makes it.
static ComPtr<ID3D12Resource> shared_texture(ID3D12Device* d12, DXGI_FORMAT format, UINT w, UINT h, HANDLE& handle) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w; desc.Height = h; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = format; desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> r;
    if (FAILED(d12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r)))) {
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        if (FAILED(d12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    }
    if (FAILED(d12->CreateSharedHandle(r.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) return nullptr;
    return r;
}

// Reads a shared texture back on the D3D12 side, once `fence` has reached `value`.
static bool read_back(ID3D12Device* d12, ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64 value, ID3D12Resource* tex,
                      UINT bytes_per_pixel, std::vector<std::uint8_t>& out) {
    const auto desc = tex->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    d12->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, &total);
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> buffer;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(d12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer))) ||
        FAILED(d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list)))) return false;
    // (as the presenter does: its queue waits for the game's fence before touching the textures)
    queue->Wait(fence, value);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {tex, 0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dst{buffer.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src{tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    list->ResourceBarrier(1, &b);
    list->Close();
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> done;
    d12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done));
    queue->Signal(done.Get(), 1);
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    done->SetEventOnCompletion(1, ev);
    const bool ok = WaitForSingleObject(ev, 5000) == WAIT_OBJECT_0;
    CloseHandle(ev);
    if (!ok) return false;
    std::uint8_t* mapped = nullptr;
    if (FAILED(buffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    out.resize(std::size_t(desc.Width) * desc.Height * bytes_per_pixel);
    for (UINT y = 0; y < desc.Height; ++y)
        std::memcpy(&out[std::size_t(y) * desc.Width * bytes_per_pixel], mapped + fp.Offset + std::size_t(y) * fp.Footprint.RowPitch,
                    std::size_t(desc.Width) * bytes_per_pixel);
    buffer->Unmap(0, nullptr);
    return true;
}

static void test_adapter(IDXGIAdapter1* adapter);

int main() {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) { std::printf("no DXGI\n"); return 1; }
    // Every hardware GPU in the machine (a game can run on any of them).
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter)); ++i) {
        DXGI_ADAPTER_DESC1 ad{};
        adapter->GetDesc1(&ad);
        if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        std::printf("GPU %u:\n", i);
        test_adapter(adapter.Get());
    }
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("d3d11 tests passed\n");
    return 0;
}

static void test_adapter(IDXGIAdapter1* adapter) {
    ComPtr<ID3D12Device> d12;
    ComPtr<ID3D11Device> d11_base;
    ComPtr<ID3D11DeviceContext> context;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d12))) ||
        FAILED(D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &d11_base, nullptr, &context))) {
        ++failures; std::printf("FAIL: no devices\n"); return;
    }
    ComPtr<ID3D11Device5> d11;
    ComPtr<ID3D11DeviceContext4> context4;
    EXPECT(SUCCEEDED(d11_base.As(&d11)) && SUCCEEDED(context.As(&context4)), "Direct3D 11.4 (shared fences) available");
    if (!d11 || !context4) return;

    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    d12->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_handle = nullptr;
    EXPECT(SUCCEEDED(d12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) &&
           SUCCEEDED(d12->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &fence_handle)), "shared fence created");
    ComPtr<ID3D11Fence> fence11;
    EXPECT(SUCCEEDED(d11->OpenSharedFence(fence_handle, IID_PPV_ARGS(&fence11))), "the fence opens in D3D11");

    const UINT w = 320, h = 180;
    struct Case { const char* name; DXGI_FORMAT format; UINT bpp; UINT samples; };
    const Case cases[] = {{"colour", DXGI_FORMAT_R8G8B8A8_UNORM, 4, 1}, {"colour, 10-bit", DXGI_FORMAT_R10G10B10A2_UNORM, 4, 1},
                          {"depth (R32F)", DXGI_FORMAT_R32_FLOAT, 4, 1}, {"multisampled colour (resolved)", DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4}};
    UINT64 value = 0;
    for (const Case& k : cases) {
        HANDLE handle = nullptr;
        auto shared = shared_texture(d12.Get(), k.format, w, h, handle);
        ComPtr<ID3D11Texture2D> opened;
        EXPECT(shared && SUCCEEDED(d11->OpenSharedResource1(handle, IID_PPV_ARGS(&opened))), "%s: the shared texture opens in D3D11", k.name);
        if (!opened) continue;
        // The game's texture: a pattern (written through a default texture, as a game renders it).
        std::vector<std::uint32_t> pattern(std::size_t(w) * h);
        for (UINT y = 0; y < h; ++y)
            for (UINT x = 0; x < w; ++x) pattern[std::size_t(y) * w + x] = (k.format == DXGI_FORMAT_R32_FLOAT)
                ? [&] { float f = 0.001f * float(x + y); std::uint32_t u; std::memcpy(&u, &f, 4); return u; }()
                : (x * 7u) | ((y * 5u) << 8) | (((x ^ y) & 0xFFu) << 16) | 0xC0000000u;
        D3D11_TEXTURE2D_DESC gd{};
        gd.Width = w; gd.Height = h; gd.MipLevels = 1; gd.ArraySize = 1; gd.Format = k.format;
        gd.SampleDesc.Count = 1; gd.Usage = D3D11_USAGE_DEFAULT; gd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA init{pattern.data(), w * 4, 0};
        ComPtr<ID3D11Texture2D> game;
        EXPECT(SUCCEEDED(d11->CreateTexture2D(&gd, &init, &game)), "%s: game texture", k.name);
        ComPtr<ID3D11Texture2D> source = game;
        if (k.samples > 1) {
            // A multisampled backbuffer: the pattern drawn into every sample (a copy from the single-sampled
            // texture is not allowed, so a clear to one colour stands in; the resolve must give that colour).
            D3D11_TEXTURE2D_DESC md = gd;
            md.SampleDesc.Count = k.samples; md.BindFlags = D3D11_BIND_RENDER_TARGET;
            ComPtr<ID3D11Texture2D> ms;
            ComPtr<ID3D11RenderTargetView> rtv;
            EXPECT(SUCCEEDED(d11->CreateTexture2D(&md, nullptr, &ms)) && SUCCEEDED(d11->CreateRenderTargetView(ms.Get(), nullptr, &rtv)), "%s: target", k.name);
            if (!rtv) continue;
            const float colour[4] = {1.0f, 0.5f, 0.0f, 1.0f};
            context->ClearRenderTargetView(rtv.Get(), colour);
            source = ms;
            context->ResolveSubresource(opened.Get(), 0, source.Get(), 0, k.format);
            for (auto& p : pattern) p = 0xFF0080FFu;  // (R 255, G 128, B 0, A 255)
        } else {
            context->CopySubresourceRegion(opened.Get(), 0, 0, 0, 0, source.Get(), 0, nullptr);
        }
        EXPECT(SUCCEEDED(context4->Signal(fence11.Get(), ++value)), "%s: fence signalled from D3D11", k.name);
        context->Flush();
        std::vector<std::uint8_t> back;
        EXPECT(read_back(d12.Get(), queue.Get(), fence.Get(), value, shared.Get(), k.bpp, back), "%s: read back after the fence", k.name);
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < pattern.size() && back.size() >= pattern.size() * 4; ++i) {
            std::uint32_t got;
            std::memcpy(&got, &back[i * 4], 4);
            const std::uint32_t want = pattern[i];
            // (a resolve may round the clear colour's 0.5 either way: one step per channel)
            bool same = true;
            for (int ch = 0; ch < 4; ++ch) {
                const int a = int((got >> (8 * ch)) & 0xFF), b = int((want >> (8 * ch)) & 0xFF);
                if (k.samples > 1 ? std::abs(a - b) > 1 : a != b) same = false;
            }
            if (!same && wrong == 0) std::printf("  first difference at pixel %zu: got %08x, expected %08x\n", i, got, want);
            if (!same) ++wrong;
        }
        std::printf("%s: %zu of %zu pixels differ, fence at %llu\n", k.name, wrong, pattern.size(), static_cast<unsigned long long>(fence->GetCompletedValue()));
        EXPECT(wrong == 0, "%s: the D3D12 side sees what D3D11 copied", k.name);
        CloseHandle(handle);
    }
    CloseHandle(fence_handle);
}
