#include "presenter/renderer.hpp"
#include <d3dcompiler.h>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>

namespace fw {
namespace {

const char kShaders[] = R"(
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst4 : register(u0);
RWTexture2D<float> dst1 : register(u0);
cbuffer C : register(b0) { uint2 size; uint marker; uint counter; };

[numthreads(8, 8, 1)] void cs_color(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= size)) return;
    dst4[id.xy] = src.Load(int3(id.xy, 0));
}
[numthreads(8, 8, 1)] void cs_depth(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= size)) return;
    dst1[id.xy] = src.Load(int3(id.xy, 0)).x;
}
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 ps(float4 pos : SV_Position) : SV_Target {
    int2 p = int2(pos.xy);
    // Scale to fit when the frame and the window differ in size (1:1 when they match, as usual).
    uint sw, sh;
    src.GetDimensions(sw, sh);
    const int2 sp = min(int2(pos.xy * float2(sw, sh) / float2(size)), int2(sw, sh) - 1);
    float4 c = src.Load(int3(sp, 0));
    // Debug strip (top-left, 16 cells of 48x48 px): one white cell advances every presented frame,
    // the rest is green while warping / red while showing the original. Only exists in our output.
    if (marker != 0 && p.y < 48 && p.x < 48 * 16) {
        const uint cell = uint(p.x) / 48;
        c = cell == (counter % 16) ? float4(1, 1, 1, 1) : (marker == 1 ? float4(0, 0.6, 0, 1) : float4(0.6, 0, 0, 1));
    }
    return float4(c.rgb, 1);
}
)";

// Moving-object extrapolation. Work happens on the game's render-resolution grid (depth/motion
// size); `rect` is the valid region. Motion is stored towards the previous frame, like the game's.
const char kShadersX[] = R"(
cbuffer X : register(b0) {
    row_major float4x4 clip_to_prev;  // the game's clipToPrevClip (row vector: prev = clip * M)
    uint4 rect;                       // valid render region: x, y, w, h
    uint2 grid;                       // render texture size
    uint2 mv_size;                    // motion vector texture size
    uint2 out_size;                   // colour / output size
    float2 mv_scale;                  // game motion vector -> uv
    float alpha;                      // game frames to move objects forward (negative: back)
    float threshold;                  // render px of own motion before a pixel counts as moving
    uint groups_x;                    // analyze: groups per row; reduce: total groups
    uint flags;                       // bit 0: mv_scale valid
};

float depth_key(float d) { return 1.0 + saturate((log2(max(d, 1e-30)) + 24.0) / 24.0) * 1022.0; }  // reversed-Z: nearer = larger

Texture2D<float> depth_t : register(t0);
Texture2D<float2> motion_t : register(t1);
RWTexture2D<float4> object_u : register(u0);        // xy: own motion (render px, to previous frame), z: depth, w: 1 moving, 2 attached to camera
RWStructuredBuffer<float4> partial_u : register(u1);
groupshared float4 gs_a[64], gs_b[64];
[numthreads(8, 8, 1)] void cs_analyze(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {
    float4 a = 0, b = 0;
    if (all(id.xy < grid)) {
        float4 o = 0;
        if (all(id.xy >= rect.xy) && all(id.xy < rect.xy + rect.zw)) {
            const float2 uv = (float2(id.xy - rect.xy) + 0.5) / float2(rect.zw);
            const float d = depth_t.Load(int3(id.xy, 0));
            const float4 prev = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, d, 1), clip_to_prev);
            const float2 cam = float2(prev.x / prev.w * 0.5 + 0.5, 0.5 - prev.y / prev.w * 0.5) - uv;
            const float2 g = motion_t.Load(int3(min(id.xy, mv_size - 1), 0));
            if (prev.w > 0 && all(abs(g) < 1e4)) {
                const float2 own = (g * mv_scale - cam) * float2(rect.zw);
                const float2 cam_px = cam * float2(rect.zw);
                const float limit = max(threshold, 0.15 * length(cam_px));  // camera-model error grows with camera speed
                const bool moving = (flags & 1) && dot(own, own) > limit * limit;
                // Attached to the camera (first-person weapon, hands): close to the camera and moving in a way the
                // camera motion does not explain - stuck to the screen while the world moves under it, or mid
                // animation (aiming in/out while walking). Nearby walls move exactly as the camera predicts.
                // Only near the camera (reversed-Z: d = near / distance, so d > 1/64 means closer than 64x the
                // near plane): the sky and distant scenery often have motion vectors that ignore the camera too.
                const bool attached = (flags & 1) && (flags & 16) && d > 1.0 / 64.0 &&
                                      length(own) > max(1.0, 0.25 * length(cam_px));
                o = float4(own, d, attached ? 2 : (moving ? 1 : 0));
                // The scale fit only uses pixels whose motion vector points along the camera motion (either
                // sign per axis); attached or independently moving pixels would bias it.
                const float lg = length(g), lc = length(cam);
                const float c1 = abs(dot(g, cam)) / max(lg * lc, 1e-12), c2 = abs(dot(float2(g.x, -g.y), cam)) / max(lg * lc, 1e-12);
                a.w = 1; b.w = moving ? 1 : 0;  // pixel counts always
                if (max(c1, c2) > 0.95) {
                    a.xyz = float3(g.x * cam.x, g.x * g.x, cam.x * cam.x);
                    b.xyz = float3(g.y * cam.y, g.y * g.y, cam.y * cam.y);
                }
            }
        }
        object_u[id.xy] = o;
    }
    gs_a[gi] = a; gs_b[gi] = b;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 32; s > 0; s >>= 1) {
        if (gi < s) { gs_a[gi] += gs_a[gi + s]; gs_b[gi] += gs_b[gi + s]; }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) { const uint g = gid.y * groups_x + gid.x; partial_u[g * 2] = gs_a[0]; partial_u[g * 2 + 1] = gs_b[0]; }
}

RWStructuredBuffer<float4> reduce_in_u : register(u0);
RWStructuredBuffer<float4> reduce_out_u : register(u1);
groupshared float4 rs_a[256], rs_b[256];
[numthreads(256, 1, 1)] void cs_reduce(uint gi : SV_GroupIndex) {
    float4 a = 0, b = 0;
    for (uint i = gi; i < groups_x; i += 256) { a += reduce_in_u[i * 2]; b += reduce_in_u[i * 2 + 1]; }
    rs_a[gi] = a; rs_b[gi] = b;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint s = 128; s > 0; s >>= 1) {
        if (gi < s) { rs_a[gi] += rs_a[gi + s]; rs_b[gi] += rs_b[gi + s]; }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0) { reduce_out_u[0] = rs_a[0]; reduce_out_u[1] = rs_b[0]; }
}

RWTexture2D<uint> dest_u : register(u0);
[numthreads(8, 8, 1)] void cs_clear(uint3 id : SV_DispatchThreadID) { if (all(id.xy < grid)) dest_u[id.xy] = 0; }

// Forward splat: every moving pixel claims its position `alpha` frames ahead; the nearest one wins.
Texture2D<float4> object_t : register(t0);
[numthreads(8, 8, 1)] void cs_splat(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    const float4 o = object_t.Load(int3(id.xy, 0));
    if (o.w < 0.5 || o.w > 1.5) return;
    const float2 move = -o.xy * alpha;
    const int2 dst = int2(floor(float2(id.xy) + move + 0.5));
    if (any(dst < int2(rect.xy)) || any(dst >= int2(rect.xy + rect.zw))) return;
    const int2 off = clamp(dst - int2(id.xy), -1023, 1023);
    const uint key = (uint(depth_key(o.z)) << 22) | (uint(off.x + 1024) << 11) | uint(off.y + 1024);
    InterlockedMax(dest_u[dst], key);
}

// Gather at output resolution. alpha <= 0 interpolates objects towards their previous position (exact
// path from the motion vectors). Pixels an object covers now but has left at this time show the
// previous game frame's background (reprojected by the camera motion); otherwise stretch from behind.
Texture2D<uint> dest_t : register(t0);
Texture2D<float4> object_g : register(t1);
Texture2D<float4> colour_t : register(t2);
Texture2D<float> depth_g : register(t3);
Texture2D<float4> previous_t : register(t4);
RWTexture2D<float4> out_u : register(u0);
float4 bilinear(Texture2D<float4> t, float2 pos) {  // pos in texel centres (0.5 = first texel)
    const float2 p = clamp(pos - 0.5, 0, float2(out_size) - 1.001);
    const int2 i = int2(floor(p));
    const float2 f = p - float2(i);
    const float4 a = t.Load(int3(i, 0)), b = t.Load(int3(i + int2(1, 0), 0));
    const float4 c = t.Load(int3(i + int2(0, 1), 0)), d = t.Load(int3(i + int2(1, 1), 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}
[numthreads(8, 8, 1)] void cs_gather(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float2 scale = float2(out_size) / float2(rect.zw);
    const float2 centre = float2(id.xy) + 0.5;
    const int2 pr = int2(rect.xy) + int2(min(uint2(centre / scale), rect.zw - 1));
    const uint key = dest_t.Load(int3(pr, 0));
    const float4 here = object_g.Load(int3(pr, 0));
    if (key != 0) {
        const int2 off = int2((key >> 11) & 2047, key & 2047) - 1024;
        const bool occluded = (here.w < 0.5 || here.w > 1.5) && depth_key(depth_g.Load(int3(pr, 0))) > float(key >> 22) + 2.0;
        if (!occluded) {
            const float4 o = object_g.Load(int3(pr - off, 0));
            const float2 move = -o.xy * alpha;  // exact, in render px
            out_u[id.xy] = bilinear(colour_t, centre - move * scale);
            return;
        }
    }
    if (here.w > 0.5 && here.w < 1.5 && alpha != 0) {
        if (flags & 2) {
            const float2 uv = (float2(pr - int2(rect.xy)) + 0.5) / float2(rect.zw);
            const float4 pv = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, depth_g.Load(int3(pr, 0)), 1), clip_to_prev);
            if (pv.w > 0) {
                const float2 puv = float2(pv.x / pv.w * 0.5 + 0.5, 0.5 - pv.y / pv.w * 0.5);
                if (all(puv >= 0) && all(puv <= 1)) { out_u[id.xy] = bilinear(previous_t, puv * float2(out_size)); return; }
            }
        }
        const float2 behind = here.xy * alpha;
        const float len = length(behind);
        const float2 dir = behind / max(len, 1e-6);
        const float step = max(1.0, len / 4.0);
        [loop] for (int k = 1; k <= 8; ++k) {
            const int2 q = clamp(pr + int2(round(dir * step * k)), int2(rect.xy), int2(rect.xy + rect.zw) - 1);
            if (abs(object_g.Load(int3(q, 0)).w - 1) > 0.5) { out_u[id.xy] = bilinear(colour_t, centre + dir * step * k * scale); return; }
        }
    }
    out_u[id.xy] = colour_t.Load(int3(id.xy, 0));
}

// HUD score (output resolution). Evidence for "overlay" is a pixel that stays the same at the same
// screen position while the camera moved the scene under it: either the same colour (opaque HUD) or the
// same edge (semi-transparent HUD, whose colour follows the scene but whose outline and text do not).
// A pixel that clearly changed in both colour and edge loses half its score. Pixels in the first-person
// weapon's depth range never count: the weapon is masked per frame from motion vectors instead, so it
// cannot linger in this slower map.
Texture2D<float4> hud_current_t : register(t0);
Texture2D<float4> hud_previous_t : register(t1);
Texture2D<float> hud_depth_t : register(t2);
Texture2D<float4> hud_object_t : register(t3);  // per-pixel analysis: w > 1.5 = attached to the camera
RWTexture2D<float> hud_score_u : register(u0);
float luma(Texture2D<float4> t, int2 p) {
    return dot(t.Load(int3(clamp(p, int2(0, 0), int2(out_size) - 1), 0)).rgb, float3(0.299, 0.587, 0.114));
}
float2 edge(Texture2D<float4> t, int2 p) {  // Sobel luminance gradient
    const float a = luma(t, p + int2(-1, -1)), b = luma(t, p + int2(0, -1)), c = luma(t, p + int2(1, -1));
    const float d = luma(t, p + int2(-1, 0)), f = luma(t, p + int2(1, 0));
    const float g = luma(t, p + int2(-1, 1)), h = luma(t, p + int2(0, 1)), i = luma(t, p + int2(1, 1));
    return float2((c + 2 * f + i) - (a + 2 * d + g), (g + 2 * h + i) - (a + 2 * b + c)) * 0.25;
}
RWByteAddressBuffer hud_counts_u : register(u1);  // [0] evidence pixels, [4] changed pixels (this frame)
// 0: nothing to learn; 1: evidence for HUD (weight); 2: clearly changed where motion would show it;
// 3: first-person weapon range; 4: clearly changed elsewhere.
uint hud_classify(uint2 id, out float weight) {
    weight = 0;
    const float2 uv = (float2(id) + 0.5) / float2(out_size);
    const int2 pr = int2(rect.xy) + int2(min(uint2(uv * float2(rect.zw)), rect.zw - 1));
    const float depth = hud_depth_t.Load(int3(pr, 0));
    // The first-person weapon never enters the HUD map. Exactly the pixels the motion vectors flag as
    // attached to the camera this frame; before the motion-vector scale is known, everything in the
    // weapon's depth range (which in some games reaches metres - a crosshair over a near wall would be
    // missed, so only as a fallback).
    if (flags & 32) { if (hud_object_t.Load(int3(pr, 0)).w > 1.5) return 3; }
    else if ((flags & 16) && depth > 1.0 / 64.0) return 3;
    const int2 p = int2(id);
    const float3 delta = abs(hud_current_t.Load(int3(p, 0)).rgb - hud_previous_t.Load(int3(p, 0)).rgb);
    const float change = max(delta.r, max(delta.g, delta.b));
    const float2 ec = edge(hud_current_t, p), ep = edge(hud_previous_t, p);
    const float lc = length(ec), lp = length(ep);
    const bool same_edge = lc > 0.05 && lp > 0.05 && dot(ec, ep) > 0.9 * lc * lp && lc < 2.0 * lp && lp < 2.0 * lc;
    const bool same_colour = change < 0.03;
    const float4 pv = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1), clip_to_prev);
    const float2 cam_px = pv.w > 0 ? (float2(pv.x / pv.w * 0.5 + 0.5, 0.5 - pv.y / pv.w * 0.5) - uv) * float2(out_size) : float2(0, 0);
    const float moved = length(cam_px);
    // Only detail ALONG the camera motion (>= 3 px here) can show whether a pixel moved: flat areas and
    // edges parallel to the motion (a rooftop during a horizontal pan) look the same either way.
    const bool telling = moved >= 3.0 && abs(dot(ec, cam_px / max(moved, 1e-6))) >= 0.05;
    if (!same_colour && !same_edge) return change > 0.06 ? (telling ? 2u : 4u) : 0u;
    if (!telling) return 0;
    // If moving scenery explains the pixel just as well (it matches where the scene came from), it is no
    // evidence: repeating detail (windows, railings, tiles) can land on an identical copy of itself.
    const int2 q = clamp(int2(round(float2(p) + cam_px)), int2(0, 0), int2(out_size) - 1);
    const float3 scenery = abs(hud_current_t.Load(int3(p, 0)).rgb - hud_previous_t.Load(int3(q, 0)).rgb);
    if (max(scenery.r, max(scenery.g, scenery.b)) < 0.03) return 0;
    const float2 eq = edge(hud_previous_t, q);
    const float lq = length(eq);
    if (lc > 0.05 && lq > 0.05 && dot(ec, eq) > 0.9 * lc * lq && lc < 2.0 * lq && lq < 2.0 * lc) return 0;
    // Stronger evidence for a bigger camera move, but never enough in one frame to reach the mask
    // threshold (0.6): a single repeated or glitched game frame cannot mask scenery.
    weight = min(0.45, 0.1 + 0.6 * saturate(moved / 12.0));
    return 1;
}
[numthreads(1, 1, 1)] void cs_clear_counts(uint3 id : SV_DispatchThreadID) { hud_counts_u.Store2(0, uint2(0, 0)); }
[numthreads(8, 8, 1)] void cs_hud_count(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size) || !(flags & 2)) return;
    float weight;
    const uint kind = hud_classify(id.xy, weight);
    uint ignored;
    if (kind == 1) hud_counts_u.InterlockedAdd(0, 1, ignored);
    else if (kind == 2) hud_counts_u.InterlockedAdd(4, 1, ignored);
}
[numthreads(8, 8, 1)] void cs_hud(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size) || !(flags & 2)) return;
    float weight;
    const uint kind = hud_classify(id.xy, weight);
    if (kind == 3) { hud_score_u[id.xy] = 0; return; }
    if (kind == 2 || kind == 4) { hud_score_u[id.xy] = hud_score_u[id.xy] * 0.5; return; }
    if (kind != 1) return;
    // A real HUD covers a small part of the screen. When most telling pixels look unchanged, the frame
    // itself is suspect (repeated image, hitch, camera data without a new image): learn nothing from it.
    const uint evidence = hud_counts_u.Load(0), changed = hud_counts_u.Load(4);
    if (evidence + changed < 256 || evidence > 0.35 * (evidence + changed)) return;
    hud_score_u[id.xy] = lerp(hud_score_u[id.xy], 1.0, weight);
}

// Motion/depth samples on a grid (mv_size = grid size) over the render rect, for camera estimation.
RWStructuredBuffer<float4> sample_u : register(u0);
[numthreads(8, 8, 1)] void cs_sample(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= mv_size)) return;
    const uint2 p = rect.xy + uint2((float2(id.xy) + 0.5) * float2(rect.zw) / float2(mv_size));
    sample_u[id.y * mv_size.x + id.x] = float4(motion_t.Load(int3(p, 0)), depth_t.Load(int3(p, 0)), 1);
}

[numthreads(8, 8, 1)] void cs_clear_score(uint3 id : SV_DispatchThreadID) { if (all(id.xy < out_size)) hud_score_u[id.xy] = 0; }

// No-warp mask (output resolution, R8): HUD (score, widened by 1 px for anti-aliased edges) and
// camera-attached pixels (widened by 1 render px).
Texture2D<float> mask_score_t : register(t0);
Texture2D<float4> mask_object_t : register(t1);
RWTexture2D<unorm float> mask_u : register(u0);
[numthreads(8, 8, 1)] void cs_mask(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    bool keep = false;
    if (flags & 4) {
        [unroll] for (int y = -1; y <= 1; ++y)
            [unroll] for (int x = -1; x <= 1; ++x)
                keep = keep || mask_score_t.Load(int3(clamp(int2(id.xy) + int2(x, y), int2(0, 0), int2(out_size) - 1), 0)) > 0.6;
    }
    if (!keep && (flags & 8)) {
        const float2 uv = (float2(id.xy) + 0.5) / float2(out_size);
        const int2 pr = int2(rect.xy) + int2(min(uint2(uv * float2(rect.zw)), rect.zw - 1));
        [unroll] for (int y = -1; y <= 1; ++y)
            [unroll] for (int x = -1; x <= 1; ++x) {
                const int2 q = clamp(pr + int2(x, y), int2(rect.xy), int2(rect.xy + rect.zw) - 1);
                keep = keep || mask_object_t.Load(int3(q, 0)).w > 1.5;
            }
    }
    mask_u[id.xy] = keep ? 1.0 : 0.0;
}
)";

// Descriptor layout in the shader-visible heap.
constexpr UINT kSrcSrv = 0;         // + slot * kTexCount + kind: shared textures
constexpr UINT kPrivUav = 32;       // + private id
constexpr UINT kPrivSrv = 48;       // + private id
// Extrapolation tables (6 SRVs t0-t5, 2 UAVs u0-u1 each).
constexpr UINT kXSrvCount = 6;
constexpr UINT kXAnalyzeSrv = 64, kXAnalyzeUav = 70, kXReduceUav = 72, kXSplatSrv = 74, kXSplatUav = 80;
constexpr UINT kXGatherSrvHudless = 82, kXGatherSrvBackbuffer = 88, kXGatherUav = 94;
constexpr UINT kXHudSrv = 96, kXHudUav = 102, kXMaskSrv = 104, kXMaskUav = 110, kXSampleSrv = 112, kXSampleUav = 118;
constexpr UINT kHeapSize = 120;

DXGI_FORMAT srv_format(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
        default: return f;
    }
}

DXGI_FORMAT swapchain_format(DXGI_FORMAT game) {
    switch (srv_format(game)) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R10G10B10A2_UNORM: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
        default: return DXGI_FORMAT_R8G8B8A8_UNORM;
    }
}

ComPtr<ID3DBlob> compile(const char* entry, const char* target, std::string& error, const char* source = kShaders) {
    ComPtr<ID3DBlob> code, messages;
    if (FAILED(D3DCompile(source, std::strlen(source), "framewarp.hlsl", nullptr, nullptr, entry, target,
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &messages))) {
        error = std::string("shader ") + entry + ": " + (messages ? static_cast<const char*>(messages->GetBufferPointer()) : "?");
        return nullptr;
    }
    return code;
}

D3D12_RESOURCE_BARRIER transition_barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    return b;
}

}  // namespace

Renderer::~Renderer() {
    if (queue_ && fence_) wait_idle();
    if (fence_event_) CloseHandle(fence_event_);
    if (waitable_) CloseHandle(waitable_);
}

D3D12_CPU_DESCRIPTOR_HANDLE Renderer::cpu(UINT index) const {
    auto h = heap_->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(index) * descriptor_size_; return h;
}
D3D12_GPU_DESCRIPTOR_HANDLE Renderer::gpu(UINT index) const {
    auto h = heap_->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(index) * descriptor_size_; return h;
}

bool Renderer::init(const LUID& adapter_luid, HWND window, std::uint32_t width, std::uint32_t height, DXGI_FORMAT game_format,
                    std::uint32_t color_space, std::string& error) {
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory_)))) { error = "DXGI factory"; return false; }
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory_->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&adapter)))) { error = "game adapter not found"; return false; }
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_)))) { error = "D3D12 device"; return false; }

    // Highest queue priority we are allowed: realtime needs SeIncreaseBasePriorityPrivilege (admin).
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME; priority_name_ = "realtime";
    if (FAILED(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)))) {
        qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH; priority_name_ = "high";
        if (FAILED(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)))) { error = "command queue"; return false; }
    }
    for (auto& a : allocators_)
        if (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a)))) { error = "allocator"; return false; }
    if (FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators_[0].Get(), nullptr, IID_PPV_ARGS(&list_)))) { error = "command list"; return false; }
    list_->Close();
    if (FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) { error = "fence"; return false; }
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kHeapSize, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_)))) { error = "descriptor heap"; return false; }
    D3D12_DESCRIPTOR_HEAP_DESC rd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(device_->CreateDescriptorHeap(&rd, IID_PPV_ARGS(&rtv_heap_)))) { error = "rtv heap"; return false; }
    descriptor_size_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    rtv_size_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    swap_format_ = swapchain_format(game_format);
    if (!create_pipelines(error)) return false;

    D3D12_QUERY_HEAP_DESC qh{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 6, 0};
    device_->CreateQueryHeap(&qh, IID_PPV_ARGS(&timestamps_));
    D3D12_HEAP_PROPERTIES rb{}; rb.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = 6 * 8; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_));
    queue_->GetTimestampFrequency(&timestamp_frequency_);
    queue_->GetClockCalibration(&calib_gpu_, &calib_cpu_);

    // Composition swapchain: DWM composites it over the game without taking focus or input.
    width_ = width; height_ = height;
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = width; sd.Height = height; sd.Format = swap_format_; sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.BufferCount = 3;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE; sd.Scaling = DXGI_SCALING_STRETCH;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    ComPtr<IDXGISwapChain1> sc1;
    if (FAILED(factory_->CreateSwapChainForComposition(queue_.Get(), &sd, nullptr, &sc1)) || FAILED(sc1.As(&swapchain_))) {
        error = "composition swapchain"; return false;
    }
    swapchain_->SetMaximumFrameLatency(1);
    waitable_ = swapchain_->GetFrameLatencyWaitableObject();
    UINT support = 0;
    const auto cs = static_cast<DXGI_COLOR_SPACE_TYPE>(color_space);
    if (SUCCEEDED(swapchain_->CheckColorSpaceSupport(cs, &support)) && (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT))
        swapchain_->SetColorSpace1(cs);
    if (FAILED(DCompositionCreateDevice(nullptr, IID_PPV_ARGS(&dcomp_))) ||
        FAILED(dcomp_->CreateTargetForHwnd(window, TRUE, &target_)) || FAILED(dcomp_->CreateVisual(&visual_)) ||
        FAILED(visual_->SetContent(swapchain_.Get())) || FAILED(target_->SetRoot(visual_.Get())) || FAILED(dcomp_->Commit())) {
        error = "DirectComposition setup"; return false;
    }
    create_swapchain_views();
    return true;
}

void Renderer::create_swapchain_views() {
    for (UINT i = 0; i < 3; ++i) {
        ComPtr<ID3D12Resource> buffer;
        swapchain_->GetBuffer(i, IID_PPV_ARGS(&buffer));
        auto h = rtv_heap_->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * rtv_size_;
        device_->CreateRenderTargetView(buffer.Get(), nullptr, h);
    }
}

bool Renderer::resize(std::uint32_t width, std::uint32_t height) {
    if (width == width_ && height == height_) return true;
    wait_idle();
    if (FAILED(swapchain_->ResizeBuffers(3, width, height, swap_format_, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT))) return false;
    width_ = width; height_ = height;
    create_swapchain_views();
    return true;
}

bool Renderer::create_pipelines(std::string& error) {
    D3D12_DESCRIPTOR_RANGE srv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_DESCRIPTOR_RANGE uav{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 4};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &srv};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {1, &uav};
    D3D12_ROOT_SIGNATURE_DESC rs{3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob, err;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
        FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_)))) {
        error = "root signature"; return false;
    }
    auto color = compile("cs_color", "cs_5_0", error), depth = compile("cs_depth", "cs_5_0", error);
    auto vs = compile("vs", "vs_5_0", error), ps = compile("ps", "ps_5_0", error);
    if (!color || !depth || !vs || !ps) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC cd{};
    cd.pRootSignature = root_.Get();
    cd.CS = {color->GetBufferPointer(), color->GetBufferSize()};
    if (FAILED(device_->CreateComputePipelineState(&cd, IID_PPV_ARGS(&cs_color_)))) { error = "cs_color pso"; return false; }
    cd.CS = {depth->GetBufferPointer(), depth->GetBufferSize()};
    if (FAILED(device_->CreateComputePipelineState(&cd, IID_PPV_ARGS(&cs_depth_)))) { error = "cs_depth pso"; return false; }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gd{};
    gd.pRootSignature = root_.Get();
    gd.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    gd.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    gd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    gd.SampleMask = UINT_MAX;
    gd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID; gd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gd.NumRenderTargets = 1; gd.RTVFormats[0] = swap_format_;
    gd.SampleDesc.Count = 1;
    if (FAILED(device_->CreateGraphicsPipelineState(&gd, IID_PPV_ARGS(&blit_)))) { error = "blit pso"; return false; }

    // Extrapolation passes: 32 constants, 4 SRVs, 2 UAVs.
    D3D12_DESCRIPTOR_RANGE xsrv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kXSrvCount, 0, 0, 0};
    D3D12_DESCRIPTOR_RANGE xuav{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 0};
    D3D12_ROOT_PARAMETER xp[3]{};
    xp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; xp[0].Constants = {0, 0, 32};
    xp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; xp[1].DescriptorTable = {1, &xsrv};
    xp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; xp[2].DescriptorTable = {1, &xuav};
    D3D12_ROOT_SIGNATURE_DESC xrs{3, xp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    blob.Reset(); err.Reset();
    if (FAILED(D3D12SerializeRootSignature(&xrs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
        FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_x_)))) {
        error = "extrapolation root signature"; return false;
    }
    struct { const char* entry; ComPtr<ID3D12PipelineState>* pso; } xs[] = {
        {"cs_analyze", &cs_analyze_}, {"cs_reduce", &cs_reduce_}, {"cs_clear", &cs_clear_}, {"cs_splat", &cs_splat_}, {"cs_gather", &cs_gather_},
        {"cs_hud", &cs_hud_}, {"cs_mask", &cs_mask_}, {"cs_clear_score", &cs_clear_score_},
        {"cs_hud_count", &cs_hud_count_}, {"cs_clear_counts", &cs_clear_counts_}, {"cs_sample", &cs_sample_}};
    for (auto& x : xs) {
        auto code = compile(x.entry, "cs_5_0", error, kShadersX);
        if (!code) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC xd{};
        xd.pRootSignature = root_x_.Get();
        xd.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (FAILED(device_->CreateComputePipelineState(&xd, IID_PPV_ARGS(x.pso->ReleaseAndGetAddressOf())))) { error = std::string(x.entry) + " pso"; return false; }
    }
    D3D12_HEAP_PROPERTIES def{}; def.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = 2 * 16; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&sums_)))) {
        error = "motion fit buffer"; return false;
    }
    bd.Width = 16;
    if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&hud_counts_)))) {
        error = "HUD counter buffer"; return false;
    }
    D3D12_HEAP_PROPERTIES rbh{}; rbh.Type = D3D12_HEAP_TYPE_READBACK;
    bd.Width = 3 * 2 * 16; bd.Flags = D3D12_RESOURCE_FLAG_NONE;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&fit_readback_)))) {
        error = "motion fit readback"; return false;
    }
    return true;
}

void Renderer::set_x_srv(UINT index, PrivateId id) {
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    const auto& p = private_[id];
    d.Format = p.texture ? p.format : DXGI_FORMAT_R8G8B8A8_UNORM;
    device_->CreateShaderResourceView(p.texture.Get(), &d, cpu(index));
}

void Renderer::set_x_uav(UINT index, PrivateId id) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    const auto& p = private_[id];
    d.Format = p.texture ? p.format : DXGI_FORMAT_R8G8B8A8_UNORM;
    device_->CreateUnorderedAccessView(p.texture.Get(), nullptr, &d, cpu(index));
}

void Renderer::x_dispatch(ID3D12PipelineState* pso, const void* constants, UINT srv_table, UINT uav_table, UINT groups_x, UINT groups_y) {
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetComputeRootSignature(root_x_.Get());
    list_->SetPipelineState(pso);
    list_->SetComputeRoot32BitConstants(0, 32, constants, 0);
    list_->SetComputeRootDescriptorTable(1, gpu(srv_table));
    list_->SetComputeRootDescriptorTable(2, gpu(uav_table));
    list_->Dispatch(groups_x, groups_y, 1);
}

namespace {
struct XConstants {
    float clip_to_prev[16];
    std::uint32_t rect[4], grid[2], mv_size[2], out_size[2];
    float mv_scale[2], alpha, threshold;
    std::uint32_t groups_x, flags;
};
static_assert(sizeof(XConstants) == 32 * 4, "root constants");
}  // namespace

void Renderer::analyze_motion(const IngestedSource& src, const float clip_to_prev_clip[16], float scale_x, float scale_y, bool scale_valid,
                              bool depth_inverted) {
    if (!src.has_depth || !src.has_motion) return;
    const auto& depth = private_[kPDepth];
    const UINT w = depth.width, h = depth.height, gx = (w + 7) / 8, gy = (h + 7) / 8;
    if (!ensure_private(kPObject, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT)) return;
    if (partial_groups_ < gx * gy) {
        wait_idle();
        partials_.Reset();
        D3D12_HEAP_PROPERTIES def{}; def.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = UINT64(gx) * gy * 2 * 16; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&partials_)))) return;
        partial_groups_ = gx * gy;
    }
    // Descriptors are rewritten each time; the resources behind them only change after wait_idle.
    set_x_srv(kXAnalyzeSrv + 0, kPDepth);
    set_x_srv(kXAnalyzeSrv + 1, kPMotion);
    for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXAnalyzeSrv + i, kPDepth);
    set_x_uav(kXAnalyzeUav + 0, kPObject);
    auto buffer_uav = [&](UINT index, ID3D12Resource* r, UINT elements) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; d.Format = DXGI_FORMAT_UNKNOWN;
        d.Buffer.NumElements = elements; d.Buffer.StructureByteStride = 16;
        device_->CreateUnorderedAccessView(r, nullptr, &d, cpu(index));
    };
    buffer_uav(kXAnalyzeUav + 1, partials_.Get(), partial_groups_ * 2);
    buffer_uav(kXReduceUav + 0, partials_.Get(), partial_groups_ * 2);
    buffer_uav(kXReduceUav + 1, sums_.Get(), 2);

    XConstants c{};
    std::memcpy(c.clip_to_prev, clip_to_prev_clip, sizeof(c.clip_to_prev));
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    c.grid[0] = w; c.grid[1] = h;
    c.mv_size[0] = private_[kPMotion].width; c.mv_size[1] = private_[kPMotion].height;
    c.mv_scale[0] = scale_x; c.mv_scale[1] = scale_y;
    c.threshold = 1.0f;
    c.groups_x = gx;
    c.flags = (scale_valid ? 1u : 0u) | (depth_inverted ? 16u : 0u);
    transition(private_[kPObject], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_analyze_.Get(), &c, kXAnalyzeSrv, kXAnalyzeUav, gx, gy);
    D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = partials_.Get();
    list_->ResourceBarrier(1, &uav);
    c.groups_x = gx * gy;
    x_dispatch(cs_reduce_.Get(), &c, kXAnalyzeSrv, kXReduceUav, 1, 1);
    auto to_copy = transition_barrier(sums_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    list_->CopyBufferRegion(fit_readback_.Get(), UINT64(frame_index_) * 32, sums_.Get(), 0, 32);
    std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
    list_->ResourceBarrier(1, &to_copy);
    transition(private_[kPObject], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    fit_pending_[frame_index_] = true;
}

ID3D12Resource* Renderer::extrapolate_objects(const IngestedSource& src, bool from_hudless, float alpha, const float clip_to_prev_clip[16]) {
    if (!src.has_depth || !src.has_motion || !private_[kPObject].texture) return nullptr;
    const UINT w = private_[kPObject].width, h = private_[kPObject].height;
    const PrivateId colour = from_hudless ? kPHudless : kPBackbuffer;
    const UINT ow = private_[colour].width, oh = private_[colour].height;
    if (!ensure_private(kPDest, w, h, DXGI_FORMAT_R32_UINT) || !ensure_private(kPExtrap, ow, oh, DXGI_FORMAT_R16G16B16A16_FLOAT)) return nullptr;
    for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXSplatSrv + i, kPObject);
    set_x_uav(kXSplatUav + 0, kPDest);
    set_x_uav(kXSplatUav + 1, kPDest);
    const UINT gather = from_hudless ? kXGatherSrvHudless : kXGatherSrvBackbuffer;
    set_x_srv(gather + 0, kPDest);
    set_x_srv(gather + 1, kPObject);
    set_x_srv(gather + 2, colour);
    set_x_srv(gather + 3, kPDepth);
    const bool previous = previous_valid_ && previous_from_hudless_ == from_hudless && private_[kPPrevious].texture &&
                          private_[kPPrevious].width == ow && private_[kPPrevious].height == oh;
    set_x_srv(gather + 4, previous ? kPPrevious : colour);
    set_x_srv(gather + 5, colour);
    set_x_uav(kXGatherUav + 0, kPExtrap);
    set_x_uav(kXGatherUav + 1, kPExtrap);

    XConstants c{};
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    c.grid[0] = w; c.grid[1] = h;
    c.out_size[0] = ow; c.out_size[1] = oh;
    c.alpha = alpha;
    std::memcpy(c.clip_to_prev, clip_to_prev_clip, sizeof(c.clip_to_prev));
    c.flags = previous ? 2u : 0u;
    transition(private_[kPDest], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_clear_.Get(), &c, kXSplatSrv, kXSplatUav, (w + 7) / 8, (h + 7) / 8);
    D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = private_[kPDest].texture.Get();
    list_->ResourceBarrier(1, &uav);
    x_dispatch(cs_splat_.Get(), &c, kXSplatSrv, kXSplatUav, (w + 7) / 8, (h + 7) / 8);
    transition(private_[kPDest], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(private_[kPExtrap], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_gather_.Get(), &c, gather, kXGatherUav, (ow + 7) / 8, (oh + 7) / 8);
    transition(private_[kPExtrap], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return private_[kPExtrap].texture.Get();
}

ID3D12Resource* Renderer::build_no_warp_mask(const IngestedSource& src, const float clip_to_prev_clip[16], bool hud, bool attached,
                                             bool depth_inverted) {
    if (!src.has_depth || !private_[kPObject].texture) return nullptr;
    const UINT ow = private_[kPBackbuffer].width, oh = private_[kPBackbuffer].height;
    const bool new_score = !private_[kPHudScore].texture || private_[kPHudScore].width != ow || private_[kPHudScore].height != oh;
    if (!ensure_private(kPHudScore, ow, oh, DXGI_FORMAT_R32_FLOAT) || !ensure_private(kPMask, ow, oh, DXGI_FORMAT_R8_UNORM)) return nullptr;
    XConstants c{};
    std::memcpy(c.clip_to_prev, clip_to_prev_clip, sizeof(c.clip_to_prev));
    const UINT w = private_[kPObject].width, h = private_[kPObject].height;
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    c.grid[0] = w; c.grid[1] = h;
    c.out_size[0] = ow; c.out_size[1] = oh;
    const bool previous = previous_valid_ && !previous_from_hudless_ && private_[kPPrevious].texture &&
                          private_[kPPrevious].width == ow && private_[kPPrevious].height == oh;
    if (new_score || reset_hud_) {
        // New or resized score: start from "not HUD".
        reset_hud_ = false;
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        set_x_uav(kXHudUav + 0, kPHudScore); set_x_uav(kXHudUav + 1, kPHudScore);
        XConstants z = c; z.flags = 0;
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXHudSrv + i, kPBackbuffer);
        x_dispatch(cs_clear_score_.Get(), &z, kXHudSrv, kXHudUav, (ow + 7) / 8, (oh + 7) / 8);
    }
    if (hud && previous) {
        set_x_srv(kXHudSrv + 0, kPBackbuffer);
        set_x_srv(kXHudSrv + 1, kPPrevious);
        set_x_srv(kXHudSrv + 2, kPDepth);
        set_x_srv(kXHudSrv + 3, kPObject);
        for (UINT i = 4; i < kXSrvCount; ++i) set_x_srv(kXHudSrv + i, kPDepth);
        set_x_uav(kXHudUav + 0, kPHudScore);
        D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
        raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
        raw.Buffer.NumElements = 4; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(hud_counts_.Get(), nullptr, &raw, cpu(kXHudUav + 1));
        c.flags = 2u | (depth_inverted ? 16u : 0u) | (attached ? 32u : 0u);
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_RESOURCE_BARRIER counts{}; counts.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; counts.UAV.pResource = hud_counts_.Get();
        x_dispatch(cs_clear_counts_.Get(), &c, kXHudSrv, kXHudUav, 1, 1);
        list_->ResourceBarrier(1, &counts);
        x_dispatch(cs_hud_count_.Get(), &c, kXHudSrv, kXHudUav, (ow + 7) / 8, (oh + 7) / 8);
        list_->ResourceBarrier(1, &counts);
        x_dispatch(cs_hud_.Get(), &c, kXHudSrv, kXHudUav, (ow + 7) / 8, (oh + 7) / 8);
    }
    transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    set_x_srv(kXMaskSrv + 0, kPHudScore);
    set_x_srv(kXMaskSrv + 1, kPObject);
    for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXMaskSrv + i, kPObject);
    set_x_uav(kXMaskUav + 0, kPMask); set_x_uav(kXMaskUav + 1, kPMask);
    c.flags = (hud ? 4u : 0u) | (attached ? 8u : 0u);
    transition(private_[kPMask], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_mask_.Get(), &c, kXMaskSrv, kXMaskUav, (ow + 7) / 8, (oh + 7) / 8);
    transition(private_[kPMask], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    mask_ready_ = true;
    return private_[kPMask].texture.Get();
}

void Renderer::flush_and_wait() {
    list_->Close();
    if (pending_game_wait_ && fence_game_) { queue_->Wait(fence_game_.Get(), pending_game_wait_); pending_game_wait_ = 0; }
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    queue_->Signal(fence_.Get(), ++fence_value_);
    frame_values_[frame_index_] = fence_value_;
    wait_idle();
    allocators_[frame_index_]->Reset();
    list_->Reset(allocators_[frame_index_].Get(), nullptr);
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
}

bool Renderer::sample_motion(const IngestedSource& src, std::uint32_t grid_w, std::uint32_t grid_h, std::vector<float>& out) {
    if (!src.has_depth || !src.has_motion) return false;
    const UINT count = grid_w * grid_h, bytes = count * 16;
    if (!samples_ || samples_count_ < count) {
        wait_idle();
        samples_.Reset(); samples_readback_.Reset();
        D3D12_HEAP_PROPERTIES def{}; def.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = bytes; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&samples_))))
            return false;
        D3D12_HEAP_PROPERTIES rb{}; rb.Type = D3D12_HEAP_TYPE_READBACK;
        bd.Flags = D3D12_RESOURCE_FLAG_NONE;
        if (FAILED(device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&samples_readback_)))) return false;
        samples_count_ = count;
    }
    set_x_srv(kXSampleSrv + 0, kPDepth);
    set_x_srv(kXSampleSrv + 1, kPMotion);
    for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXSampleSrv + i, kPDepth);
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; d.Format = DXGI_FORMAT_UNKNOWN;
    d.Buffer.NumElements = count; d.Buffer.StructureByteStride = 16;
    device_->CreateUnorderedAccessView(samples_.Get(), nullptr, &d, cpu(kXSampleUav + 0));
    device_->CreateUnorderedAccessView(samples_.Get(), nullptr, &d, cpu(kXSampleUav + 1));
    XConstants c{};
    const UINT w = private_[kPDepth].width, h = private_[kPDepth].height;
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    c.grid[0] = w; c.grid[1] = h;
    c.mv_size[0] = grid_w; c.mv_size[1] = grid_h;
    x_dispatch(cs_sample_.Get(), &c, kXSampleSrv, kXSampleUav, (grid_w + 7) / 8, (grid_h + 7) / 8);
    auto to_copy = transition_barrier(samples_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    list_->CopyBufferRegion(samples_readback_.Get(), 0, samples_.Get(), 0, bytes);
    flush_and_wait();
    float* mapped = nullptr;
    D3D12_RANGE range{0, bytes};
    if (FAILED(samples_readback_->Map(0, &range, reinterpret_cast<void**>(&mapped)))) return false;
    out.assign(mapped, mapped + count * 4);
    D3D12_RANGE none{0, 0};
    samples_readback_->Unmap(0, &none);
    return true;
}

bool Renderer::take_motion_fit(MotionFit& fit) {
    if (!fit_ready_) return false;
    fit = fit_latest_;
    fit_ready_ = false;
    return true;
}


bool Renderer::ensure_private(PrivateId id, std::uint32_t w, std::uint32_t h, DXGI_FORMAT format) {
    auto& p = private_[id];
    if (p.texture && p.width == w && p.height == h && p.format == format) return true;
    if (p.texture) wait_idle();
    p = Private{};
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = format; d.SampleDesc.Count = 1; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                nullptr, IID_PPV_ARGS(&p.texture)))) return false;
    p.width = w; p.height = h; p.format = format; p.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    device_->CreateUnorderedAccessView(p.texture.Get(), nullptr, nullptr, cpu(kPrivUav + id));
    device_->CreateShaderResourceView(p.texture.Get(), nullptr, cpu(kPrivSrv + id));
    return true;
}

void Renderer::transition(Private& p, D3D12_RESOURCE_STATES to) {
    if (p.state == to) return;
    const auto b = transition_barrier(p.texture.Get(), p.state, to);
    list_->ResourceBarrier(1, &b);
    p.state = to;
}

bool Renderer::open_session(DWORD pid, std::uint64_t session, std::string& error) {
    wait_idle();
    fence_game_.Reset();
    for (auto& slot : shared_) for (auto& t : slot) t = SharedTex{};
    std::fill(std::begin(srv_resource_), std::end(srv_resource_), nullptr);
    pid_ = pid; session_ = session;
    HANDLE handle = nullptr;
    if (FAILED(device_->OpenSharedHandleByName(object_name(pid, session, L"fence").c_str(), GENERIC_ALL, &handle))) {
        error = "game fence not available yet"; return false;
    }
    const HRESULT hr = device_->OpenSharedHandle(handle, IID_PPV_ARGS(&fence_game_));
    CloseHandle(handle);
    if (FAILED(hr)) { error = "open game fence"; return false; }
    return true;
}

ID3D12Resource* Renderer::shared_texture(int slot, int kind, std::uint32_t generation) {
    auto& t = shared_[slot][kind];
    if (t.resource && t.generation == generation) return t.resource.Get();
    if (t.resource) {
        // The game resized this texture (e.g. DLSS preset change). Frames in flight may still read the old
        // one, and the replacement can be allocated at the same address, so a pointer-keyed descriptor
        // cache would keep describing freed memory (GPU page fault -> DEVICE_HUNG). Drain, then forget it.
        wait_idle();
        srv_resource_[kSrcSrv + slot * kTexCount + kind] = nullptr;
    }
    t = SharedTex{};
    HANDLE handle = nullptr;
    if (FAILED(device_->OpenSharedHandleByName(object_name(pid_, session_, L"tex", slot, kind, generation).c_str(), GENERIC_ALL, &handle)))
        return nullptr;
    const HRESULT hr = device_->OpenSharedHandle(handle, IID_PPV_ARGS(&t.resource));
    CloseHandle(handle);
    if (FAILED(hr)) return nullptr;
    t.generation = generation;
    {
        const auto d = t.resource->GetDesc();
        char note[160];
        std::snprintf(note, sizeof(note), "opened shared %s slot %d gen %u: %llux%u fmt %d", tex_name(kind), slot, generation,
                      static_cast<unsigned long long>(d.Width), d.Height, int(d.Format));
        notes_.push_back(note);
    }
    return t.resource.Get();
}

ID3D12GraphicsCommandList* Renderer::begin_frame() {
    frame_index_ = (frame_index_ + 1) % 3;
    if (!completed(frame_values_[frame_index_])) {
        fence_->SetEventOnCompletion(frame_values_[frame_index_], fence_event_);
        WaitForSingleObject(fence_event_, 1000);
    }
    // GPU time of the frame that last used this allocator.
    if (readback_ && frame_values_[frame_index_]) {
        std::uint64_t* ts = nullptr;
        D3D12_RANGE range{frame_index_ * 16, frame_index_ * 16 + 16};
        if (SUCCEEDED(readback_->Map(0, &range, reinterpret_cast<void**>(&ts)))) {
            const auto a = ts[frame_index_ * 2], b = ts[frame_index_ * 2 + 1];
            if (b > a) gpu_ms_ = static_cast<float>(double(b - a) * 1000.0 / double(timestamp_frequency_));
            // Convert GPU timestamps to CPU QPC with the calibration pair.
            LARGE_INTEGER qf; QueryPerformanceFrequency(&qf);
            auto to_qpc = [&](std::uint64_t gpu) {
                const double seconds = (double(gpu) - double(calib_gpu_)) / double(timestamp_frequency_);
                return static_cast<std::int64_t>(double(calib_cpu_) + seconds * double(qf.QuadPart));
            };
            if (b > a && a > 0) { timing_ = {submit_qpc_[frame_index_], to_qpc(a), to_qpc(b), true}; }
            D3D12_RANGE none{0, 0};
            readback_->Unmap(0, &none);
        }
    }
    // Motion-vector fit sums recorded by analyze_motion in that frame.
    if (fit_pending_[frame_index_] && fit_readback_) {
        fit_pending_[frame_index_] = false;
        float* f = nullptr;
        D3D12_RANGE range{frame_index_ * 32, frame_index_ * 32 + 32};
        if (SUCCEEDED(fit_readback_->Map(0, &range, reinterpret_cast<void**>(&f)))) {
            const float* v = f + frame_index_ * 8;
            fit_latest_.gc[0] = v[0]; fit_latest_.gg[0] = v[1]; fit_latest_.cc[0] = v[2]; fit_latest_.samples = v[3];
            fit_latest_.gc[1] = v[4]; fit_latest_.gg[1] = v[5]; fit_latest_.cc[1] = v[6]; fit_latest_.moving = v[7];
            fit_ready_ = true;
            D3D12_RANGE none{0, 0};
            fit_readback_->Unmap(0, &none);
        }
    }
    allocators_[frame_index_]->Reset();
    list_->Reset(allocators_[frame_index_].Get(), nullptr);
    if (timestamps_) list_->EndQuery(timestamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 2);
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
    reading_.clear();
    return list_.Get();
}

void Renderer::convert(ID3D12Resource* source, DXGI_FORMAT source_format, int slot, int kind, PrivateId target,
                       std::uint32_t w, std::uint32_t h) {
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = srv_format(source_format);
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Texture2D.MipLevels = 1;
    const UINT index = kSrcSrv + slot * kTexCount + kind;
    if (srv_resource_[index] != source) {
        // A frame still in flight may read this descriptor: only rewrite it once the GPU is idle.
        if (srv_resource_[index]) wait_idle();
        device_->CreateShaderResourceView(source, &sv, cpu(index));
        srv_resource_[index] = source;
    }
    auto& p = private_[target];
    transition(p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list_->SetPipelineState(target == kPDepth ? cs_depth_.Get() : cs_color_.Get());
    list_->SetComputeRootSignature(root_.Get());
    const std::uint32_t constants[4] = {w, h, 0, 0};
    list_->SetComputeRoot32BitConstants(0, 4, constants, 0);
    list_->SetComputeRootDescriptorTable(1, gpu(index));
    list_->SetComputeRootDescriptorTable(2, gpu(kPrivUav + target));
    list_->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    transition(p, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

IngestedSource Renderer::ingest(const Shared& shared, int slot) {
    IngestedSource out;
    const auto& m = shared.slots[slot];
    const auto& bb = m.tex[kBackbuffer];
    if (!bb.valid || !fence_game_) return out;
    ID3D12Resource* sources[kTexCount]{};
    for (int k = 0; k < kTexCount; ++k)
        if (m.tex[k].valid) sources[k] = shared_texture(slot, k, m.tex[k].generation);
    if (!sources[kBackbuffer]) return out;
    pending_game_wait_ = std::max(pending_game_wait_, m.fence_value);

    // Shared textures live in COMMON between the two devices' queues.
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    for (auto* r : sources)
        if (r) { barriers.push_back(transition_barrier(r, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)); reading_.push_back(r); }
    list_->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());

    out.color_w = bb.width; out.color_h = bb.height;
    out.color_rect = {0, 0, bb.width, bb.height};
    ensure_private(kPBackbuffer, bb.width, bb.height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    ensure_private(kPHudless, bb.width, bb.height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    ensure_private(kPUi, bb.width, bb.height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    ensure_private(kPZeroUi, bb.width, bb.height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    ensure_private(kPOutput, bb.width, bb.height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (keep_previous_ && ingested_) {
        const PrivateId from = last_had_hudless_ ? kPHudless : kPBackbuffer;
        if (private_[from].texture && ensure_private(kPPrevious, private_[from].width, private_[from].height, DXGI_FORMAT_R16G16B16A16_FLOAT)) {
            transition(private_[from], D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(private_[kPPrevious], D3D12_RESOURCE_STATE_COPY_DEST);
            list_->CopyResource(private_[kPPrevious].texture.Get(), private_[from].texture.Get());
            transition(private_[from], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(private_[kPPrevious], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            previous_valid_ = true;
            previous_from_hudless_ = last_had_hudless_;
        }
    }
    convert(sources[kBackbuffer], static_cast<DXGI_FORMAT>(bb.format), slot, kBackbuffer, kPBackbuffer, bb.width, bb.height);

    const auto& hl = m.tex[kHudless];
    if (sources[kHudless] && hl.width == bb.width && hl.height == bb.height) {
        convert(sources[kHudless], static_cast<DXGI_FORMAT>(hl.format), slot, kHudless, kPHudless, bb.width, bb.height);
        out.has_hudless = true;
    }
    const auto& ui = m.tex[kUi];
    if (sources[kUi] && ui.width == bb.width && ui.height == bb.height) {
        convert(sources[kUi], static_cast<DXGI_FORMAT>(ui.format), slot, kUi, kPUi, bb.width, bb.height);
        out.has_ui = true;
    }
    const auto& dp = m.tex[kDepth];
    if (sources[kDepth]) {
        ensure_private(kPDepth, dp.width, dp.height, DXGI_FORMAT_R32_FLOAT);
        ensure_private(kPMotion, dp.width, dp.height, DXGI_FORMAT_R16G16_FLOAT);
        convert(sources[kDepth], static_cast<DXGI_FORMAT>(dp.format), slot, kDepth, kPDepth, dp.width, dp.height);
        const auto& mv = m.tex[kMotion];
        if (sources[kMotion]) {
            convert(sources[kMotion], static_cast<DXGI_FORMAT>(mv.format), slot, kMotion, kPMotion,
                    std::min(mv.width, dp.width), std::min(mv.height, dp.height));
            out.has_motion = true;
        }
        out.depth_rect = {dp.ext_x, dp.ext_y, dp.ext_w, dp.ext_h};
        out.has_depth = true;
    }
    ingested_ = true;
    last_had_hudless_ = out.has_hudless;
    barriers.clear();
    for (auto* r : reading_) barriers.push_back(transition_barrier(r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON));
    list_->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    reading_.clear();
    out.valid = true;
    return out;
}

LatewarpInputs Renderer::latewarp_inputs(const IngestedSource& src, bool use_ui_tags) {
    LatewarpInputs in;
    const bool split = use_ui_tags && src.has_hudless && src.has_ui;
    in.backbuffer = private_[kPBackbuffer].texture.Get();
    in.hudless = split ? private_[kPHudless].texture.Get() : in.backbuffer;
    in.ui = split ? private_[kPUi].texture.Get() : private_[kPZeroUi].texture.Get();
    in.depth = private_[kPDepth].texture.Get();
    in.motion = private_[kPMotion].texture.Get();
    in.output = private_[kPOutput].texture.Get();
    in.color_rect = src.color_rect;
    in.depth_rect = src.depth_rect;
    if (in.output) transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return in;
}

void Renderer::finish_frame(bool warped, int marker) {
    ComPtr<ID3D12Resource> back;
    const UINT index = swapchain_->GetCurrentBackBufferIndex();
    swapchain_->GetBuffer(index, IID_PPV_ARGS(&back));
    const auto to_rt = transition_barrier(back.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    list_->ResourceBarrier(1, &to_rt);
    auto rtv = rtv_heap_->GetCPUDescriptorHandleForHeapStart(); rtv.ptr += SIZE_T(index) * rtv_size_;
    const PrivateId source = warped ? kPOutput : kPBackbuffer;
    if (private_[source].texture) {
        transition(private_[source], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
        list_->SetDescriptorHeaps(1, heaps);  // NGX may have bound its own heap
        list_->SetPipelineState(blit_.Get());
        list_->SetGraphicsRootSignature(root_.Get());
        const std::uint32_t constants[4] = {width_, height_, static_cast<std::uint32_t>(marker), static_cast<std::uint32_t>(present_counter_++)};
        list_->SetGraphicsRoot32BitConstants(0, 4, constants, 0);
        list_->SetGraphicsRootDescriptorTable(1, gpu(kPrivSrv + source));
        const D3D12_VIEWPORT vp{0, 0, float(width_), float(height_), 0, 1};
        const D3D12_RECT sr{0, 0, LONG(width_), LONG(height_)};
        list_->RSSetViewports(1, &vp);
        list_->RSSetScissorRects(1, &sr);
        list_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list_->DrawInstanced(3, 1, 0, 0);
        transition(private_[source], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    } else {
        const float black[4] = {0, 0, 0, 1};
        list_->ClearRenderTargetView(rtv, black, 0, nullptr);
    }
    const auto to_present = transition_barrier(back.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    list_->ResourceBarrier(1, &to_present);
    if (timestamps_) {
        list_->EndQuery(timestamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 2 + 1);
        list_->ResolveQueryData(timestamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 2, 2, readback_.Get(), frame_index_ * 16);
    }
    list_->Close();
    if (pending_game_wait_ && fence_game_) { queue_->Wait(fence_game_.Get(), pending_game_wait_); pending_game_wait_ = 0; }
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    {
        LARGE_INTEGER q; QueryPerformanceCounter(&q);
        submit_qpc_[frame_index_] = q.QuadPart;
    }
    swapchain_->Present(1, 0);
    {
        DXGI_FRAME_STATISTICS st{};
        present_stats_.hr = swapchain_->GetFrameStatistics(&st);
        swapchain_->GetLastPresentCount(&present_stats_.last_present_count);
        present_stats_.present_count = st.PresentCount; present_stats_.present_refresh = st.PresentRefreshCount;
        present_stats_.sync_refresh = st.SyncRefreshCount; present_stats_.sync_qpc = st.SyncQPCTime.QuadPart;
        static int recalibrate = 0;
        if (++recalibrate % 240 == 0) queue_->GetClockCalibration(&calib_gpu_, &calib_cpu_);
    }
    queue_->Signal(fence_.Get(), ++fence_value_);
    frame_values_[frame_index_] = fence_value_;
}

bool Renderer::read_back(bool output, std::vector<std::uint16_t>& pixels, std::uint32_t& w, std::uint32_t& h) {
    auto& p = private_[output ? kPOutput : kPBackbuffer];
    if (!p.texture) return false;
    wait_idle();
    const auto desc = p.texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, &total);
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> buffer;
    if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) return false;
    allocators_[frame_index_]->Reset();
    list_->Reset(allocators_[frame_index_].Get(), nullptr);
    const auto before = p.state;
    transition(p, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{buffer.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src{p.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    transition(p, before);
    list_->Close();
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    wait_idle();
    w = static_cast<std::uint32_t>(desc.Width); h = desc.Height;
    pixels.resize(std::size_t(w) * h * 4);
    std::uint8_t* mapped = nullptr;
    if (FAILED(buffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    for (std::uint32_t y = 0; y < h; ++y)
        std::memcpy(&pixels[std::size_t(y) * w * 4], mapped + fp.Offset + std::size_t(y) * fp.Footprint.RowPitch, std::size_t(w) * 8);
    buffer->Unmap(0, nullptr);
    return true;
}

void Renderer::wait_idle() {
    if (!queue_ || !fence_) return;
    queue_->Signal(fence_.Get(), ++fence_value_);
    if (fence_->GetCompletedValue() < fence_value_) {
        fence_->SetEventOnCompletion(fence_value_, fence_event_);
        WaitForSingleObject(fence_event_, 2000);
    }
}

}  // namespace fw
