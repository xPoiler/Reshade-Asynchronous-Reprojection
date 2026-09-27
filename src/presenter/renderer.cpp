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
                // Further out (up to 4096x the near plane: a third-person character a few metres away) only what
                // is stuck to the screen while the camera clearly moves - objects moving on their own (cars,
                // people) do not do that, and they must keep warping.
                const float2 screen_px = g * mv_scale * float2(rect.zw);
                const bool attached = (flags & 1) && (flags & 16) &&
                                      ((d > 1.0 / 64.0 && length(own) > max(1.0, 0.25 * length(cam_px))) ||
                                       (d > 1.0 / 4096.0 && length(cam_px) >= 2.0 && length(screen_px) < 0.2 * length(cam_px)));
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

// HUD from the upscaler's output (opt-in). The upscaler's output is the scene before post-processing
// and HUD; the game's frame is that scene tone-mapped and post-processed, plus the HUD. Predict the
// frame from the scene and what the prediction misses is HUD, in this very frame. The model only holds
// what tone mapping generally does:
//  - a tone curve per channel over the whole screen;
//  - highlights washing out towards grey/white (filmic and ACES-style tone mappers): per brightness
//    level, how far the colour moves towards that level's grey - only in bright levels, where it
//    happens; in dark ones HUD over dark scenery would teach it the HUD;
//  - a smooth affine colour correction per 128 px tile (vignette, damage tint, bloom).
// Fits sample every fourth pixel; each is fitted first from every pixel, then again without the
// pixels the previous stage missed (the HUD must not bend them).
// rect: x predictor/curve read, y slot or tile set written, z which pixels count (0 all, 1 near curve 0,
// 2 near tile set 0 over predictor 1); grid: tile counts.
Texture2D<float4> sc_frame_t : register(t0);
Texture2D<float4> sc_scene_t : register(t1);
RWTexture2D<float> sc_hud_u : register(u0);
RWByteAddressBuffer sc_fit_u : register(u1);
static const uint kBins = 64, kTile = 128, kMaxTiles = 4096;
// Byte offsets in the fit buffer.
static const uint kCurveAcc = 0;                                  // 3 curves x 3 ch x 64 bins x (sum, count)
static const uint kMeanBase = kCurveAcc + 3 * kBins * 3 * 2 * 4;   // 3 curves x 3 ch x 64 means
static const uint kDesatAcc = kMeanBase + 3 * kBins * 3 * 4;       // 2 x 64 bins x (grey sum, count, k num, k den)
static const uint kDesatBase = kDesatAcc + 2 * kBins * 4 * 4;      // 2 x 64 bins x (grey, k)
static const uint kTileBase = kDesatBase + 2 * kBins * 2 * 4;      // 2 tile sets x tiles x 3 ch x (a, b)
static const uint kHudCount = kTileBase + 2 * kMaxTiles * 3 * 2 * 4;
static const float3 kLuma = float3(0.2126, 0.7152, 0.0722);
static const float kHighlight = 0.5;  // grey level (display) from which highlights may wash out
float bin_pos(float v) { return (log2(max(v, 1.0 / 4096.0)) + 12.0) * (float(kBins) / 20.0); }
uint bin_of(float v) { return min(uint(max(bin_pos(v), 0.0)), kBins - 1); }
float mean_of(uint slot, uint c, int b) { return asfloat(sc_fit_u.Load(kMeanBase + ((slot * 3 + c) * kBins + uint(b)) * 4)); }
// The curve and its slope per doubling of the scene value.
float curve_slope(uint slot, uint c, float v, out float slope) {
    const float f = bin_pos(v) - 0.5, fl = floor(f);
    const int i0 = clamp(int(fl), 0, int(kBins) - 1), i1 = clamp(int(fl) + 1, 0, int(kBins) - 1);
    const float m0 = mean_of(slot, c, i0), m1 = mean_of(slot, c, i1);
    slope = (m1 - m0) * (float(kBins) / 20.0);
    return lerp(m0, m1, saturate(f - fl));
}
float curve(uint slot, uint c, float v) { float s; return curve_slope(slot, c, v, s); }
float3 curve3(uint slot, float3 s) { return float3(curve(slot, 0, s.r), curve(slot, 1, s.g), curve(slot, 2, s.b)); }
float grey_of(uint d, int b) { return asfloat(sc_fit_u.Load(kDesatBase + ((d * kBins + uint(b)) * 2) * 4)); }
float wash_of(uint d, int b) { return asfloat(sc_fit_u.Load(kDesatBase + ((d * kBins + uint(b)) * 2 + 1) * 4)); }
// Curve, then washed towards the grey of the scene's brightness level.
float3 washed(uint d, uint slot, float3 s) {
    const float3 a = curve3(slot, s);
    const float f = bin_pos(dot(s, kLuma)) - 0.5, fl = floor(f);
    const int i0 = clamp(int(fl), 0, int(kBins) - 1), i1 = clamp(int(fl) + 1, 0, int(kBins) - 1);
    const float t = saturate(f - fl);
    const float grey = lerp(grey_of(d, i0), grey_of(d, i1), t), k = lerp(wash_of(d, i0), wash_of(d, i1), t);
    return a + k * (grey - a);
}
// Predictor 1: curve 1 + wash-out 0; predictor 2: curve 2 + wash-out 1.
float3 predictor(uint sel, float3 s) { return sel == 1 ? washed(0, 1, s) : washed(1, 2, s); }
// Tile correction (a, b per channel), bilinear between tile centres.
void correction(uint set, uint2 p, out float3 a, out float3 b) {
    const float2 t = (float2(p) + 0.5) / float(kTile) - 0.5;
    const float2 fl = floor(t), f = saturate(t - fl);
    const int2 last = int2(grid) - 1;
    a = 0; b = 0;
    [unroll] for (int k = 0; k < 4; ++k) {
        const int2 q = clamp(int2(fl) + int2(k & 1, k >> 1), int2(0, 0), last);
        const float w = ((k & 1) ? f.x : 1 - f.x) * ((k >> 1) ? f.y : 1 - f.y);
        const uint base = kTileBase + ((set * kMaxTiles + uint(q.y) * grid.x + uint(q.x)) * 3) * 8;
        [unroll] for (uint c = 0; c < 3; ++c) {
            a[c] += w * asfloat(sc_fit_u.Load(base + c * 8));
            b[c] += w * asfloat(sc_fit_u.Load(base + c * 8 + 4));
        }
    }
}
float miss(float3 frame, float3 pred) { const float3 d = abs(frame - pred); return max(d.r, max(d.g, d.b)); }
bool counts(uint rule, uint2 p, float3 frame, float3 scene) {
    if (rule == 1) return miss(frame, curve3(0, scene)) < 0.1;
    if (rule == 2) { float3 a, b; correction(0, p, a, b); return miss(frame, a * predictor(1, scene) + b) < 0.08; }
    return true;
}
[numthreads(64, 1, 1)] void cs_scene_clear(uint3 id : SV_DispatchThreadID) {
    for (uint i = id.x; i < 3 * kBins * 3 * 2; i += 64) sc_fit_u.Store(kCurveAcc + i * 4, 0);
    for (uint j = id.x; j < 2 * kBins * 4; j += 64) sc_fit_u.Store(kDesatAcc + j * 4, 0);
    if (id.x == 0) sc_fit_u.Store(kHudCount, 0);
}
bool sample_at(uint3 id, out uint2 p, out float3 frame, out float3 scene) {
    p = id.xy * 4;
    frame = 0; scene = 0;
    if (any(p >= out_size)) return false;
    frame = sc_frame_t.Load(int3(p, 0)).rgb;
    scene = max(sc_scene_t.Load(int3(p, 0)).rgb, 0);
    return true;
}
// Tone curve rect.y: mean frame value per scene-value bin, per channel.
groupshared uint sc_bins[kBins * 3 * 2];
[numthreads(8, 8, 1)] void cs_scene_accum(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    for (uint i = gi; i < kBins * 3 * 2; i += 64) sc_bins[i] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint2 p; float3 frame, scene;
    if (sample_at(id, p, frame, scene) && counts(rect.z, p, frame, scene))
        [unroll] for (uint c = 0; c < 3; ++c) {
            const uint b = bin_of(scene[c]);
            InterlockedAdd(sc_bins[(c * kBins + b) * 2], uint(saturate(frame[c]) * 1024.0 + 0.5));
            InterlockedAdd(sc_bins[(c * kBins + b) * 2 + 1], 1);
        }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = gi; j < kBins * 3 * 2; j += 64)
        if (sc_bins[j]) sc_fit_u.InterlockedAdd(kCurveAcc + (rect.y * kBins * 3 * 2 + j) * 4, sc_bins[j]);
}
// Mean per bin; empty bins take the line between their nearest filled neighbours. `acc`: word index of
// bin 0's (sum, count) pair; `stride`: words between bins.
float filled_mean(uint acc, uint stride, uint b) {
    int lo = -1, hi = -1;
    for (int k = int(b); k >= 0; --k) if (sc_fit_u.Load((acc + uint(k) * stride + 1) * 4)) { lo = k; break; }
    for (int m = int(b); m < int(kBins); ++m) if (sc_fit_u.Load((acc + uint(m) * stride + 1) * 4)) { hi = m; break; }
    if (lo < 0 && hi < 0) return 0;
    const int l = lo >= 0 ? lo : hi, h = hi >= 0 ? hi : lo;
    const uint el = (acc + uint(l) * stride) * 4, eh = (acc + uint(h) * stride) * 4;
    const float vl = float(sc_fit_u.Load(el)) / 1024.0 / float(sc_fit_u.Load(el + 4));
    const float vh = float(sc_fit_u.Load(eh)) / 1024.0 / float(sc_fit_u.Load(eh + 4));
    return h > l ? lerp(vl, vh, float(int(b) - l) / float(h - l)) : vl;
}
[numthreads(64, 3, 1)] void cs_scene_finish(uint3 id : SV_DispatchThreadID) {
    const uint b = id.x, c = id.y;
    const float v = filled_mean(kCurveAcc / 4 + (rect.y * 3 + c) * kBins * 2, 2, b);
    sc_fit_u.Store(kMeanBase + ((rect.y * 3 + c) * kBins + b) * 4, asuint(v));
}
// Wash-out rect.y, step 1: the frame's grey (luminance) per scene-luminance bin.
groupshared uint sc_grey[kBins * 2];
[numthreads(8, 8, 1)] void cs_scene_grey(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    for (uint i = gi; i < kBins * 2; i += 64) sc_grey[i] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint2 p; float3 frame, scene;
    if (sample_at(id, p, frame, scene) && counts(rect.z, p, frame, scene)) {
        const uint b = bin_of(dot(scene, kLuma));
        InterlockedAdd(sc_grey[b * 2], uint(saturate(dot(frame, kLuma)) * 1024.0 + 0.5));
        InterlockedAdd(sc_grey[b * 2 + 1], 1);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = gi; j < kBins * 2; j += 64)
        if (sc_grey[j]) sc_fit_u.InterlockedAdd(kDesatAcc + ((rect.y * kBins + j / 2) * 4 + (j & 1)) * 4, sc_grey[j]);
}
[numthreads(64, 1, 1)] void cs_scene_grey_finish(uint3 id : SV_DispatchThreadID) {
    const uint b = id.x;
    const float grey = filled_mean(kDesatAcc / 4 + rect.y * kBins * 4, 4, b);
    sc_fit_u.Store(kDesatBase + ((rect.y * kBins + b) * 2) * 4, asuint(grey));
}
// Step 2: per bin, how far the curve's colour moves towards that grey (least squares over channels).
groupshared int sc_num[kBins];
groupshared uint sc_den[kBins];
[numthreads(8, 8, 1)] void cs_scene_wash(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    sc_num[gi] = 0; sc_den[gi] = 0;  // 64 threads, 64 bins
    GroupMemoryBarrierWithGroupSync();
    uint2 p; float3 frame, scene;
    if (sample_at(id, p, frame, scene) && counts(rect.z, p, frame, scene)) {
        const uint b = bin_of(dot(scene, kLuma));
        const float3 a = curve3(rect.x, scene), towards = grey_of(rect.y, int(b)) - a;
        InterlockedAdd(sc_num[b], int(round(dot(frame - a, towards) * 1024.0)));
        InterlockedAdd(sc_den[b], uint(dot(towards, towards) * 1024.0 + 0.5));
    }
    GroupMemoryBarrierWithGroupSync();
    if (sc_den[gi]) {
        sc_fit_u.InterlockedAdd(kDesatAcc + ((rect.y * kBins + gi) * 4 + 2) * 4, asuint(sc_num[gi]));
        sc_fit_u.InterlockedAdd(kDesatAcc + ((rect.y * kBins + gi) * 4 + 3) * 4, sc_den[gi]);
    }
}
[numthreads(64, 1, 1)] void cs_scene_wash_finish(uint3 id : SV_DispatchThreadID) {
    const uint b = id.x, acc = kDesatAcc + (rect.y * kBins + b) * 16;
    const float num = float(asint(sc_fit_u.Load(acc + 8))), den = float(sc_fit_u.Load(acc + 12));
    float k = den > 0 ? saturate(num / den) : 0.0;
    if (grey_of(rect.y, int(b)) < kHighlight) k = 0;
    sc_fit_u.Store(kDesatBase + ((rect.y * kBins + b) * 2 + 1) * 4, asuint(k));
}
groupshared float sc_red[256];
groupshared float sc_tot[13];
[numthreads(16, 16, 1)] void cs_scene_tiles(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    float v[13];
    [unroll] for (uint q = 0; q < 13; ++q) v[q] = 0;
    for (uint y = 0; y < 2; ++y)
        for (uint x = 0; x < 2; ++x) {
            const uint2 p = gid.xy * kTile + (tid.xy * 2 + uint2(x, y)) * 4;
            if (any(p >= out_size)) continue;
            const float3 frame = sc_frame_t.Load(int3(p, 0)).rgb, scene = max(sc_scene_t.Load(int3(p, 0)).rgb, 0);
            if (!counts(rect.z, p, frame, scene)) continue;
            const float3 pr = predictor(rect.x, scene);
            [unroll] for (uint c = 0; c < 3; ++c) {
                v[c * 4] += pr[c]; v[c * 4 + 1] += frame[c]; v[c * 4 + 2] += pr[c] * pr[c]; v[c * 4 + 3] += pr[c] * frame[c];
            }
            v[12] += 1;
        }
    [unroll] for (uint k = 0; k < 13; ++k) {
        sc_red[gi] = v[k];
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 128; s > 0; s >>= 1) {
            if (gi < s) sc_red[gi] += sc_red[gi + s];
            GroupMemoryBarrierWithGroupSync();
        }
        if (gi == 0) sc_tot[k] = sc_red[0];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi != 0) return;
    const float n = sc_tot[12];
    const uint base = kTileBase + ((rect.y * kMaxTiles + gid.y * grid.x + gid.x) * 3) * 8;
    [unroll] for (uint c = 0; c < 3; ++c) {
        float a = 1, b = 0;
        if (n >= float(kTile * kTile / 128)) {  // an eighth of the tile's samples
            const float sp = sc_tot[c * 4], sf = sc_tot[c * 4 + 1], spp = sc_tot[c * 4 + 2], spf = sc_tot[c * 4 + 3];
            const float var = spp - sp * sp / n;
            a = var > 1e-4 * n ? clamp((spf - sp * sf / n) / var, 0.4, 2.5) : 1.0;
            b = (sf - a * sp) / n;
        }
        sc_fit_u.Store(base + c * 8, asuint(a));
        sc_fit_u.Store(base + c * 8 + 4, asuint(b));
    }
}
// Final prediction (predictor 2, tile set 1) at full resolution. The tolerance grows with the
// prediction's own gradient (sharpening, sub-pixel differences between the two images): the tone
// curve's slope times the scene's gradient, per channel.
groupshared uint sc_hud_pixels;
[numthreads(8, 8, 1)] void cs_scene_hud(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0) sc_hud_pixels = 0;
    GroupMemoryBarrierWithGroupSync();
    if (all(id.xy < out_size)) {
        float3 a, b;
        correction(1, id.xy, a, b);
        const int2 p = int2(id.xy), last = int2(out_size) - 1;
        const float3 scene = max(sc_scene_t.Load(int3(p, 0)).rgb, 0);
        float3 slope;
        curve_slope(2, 0, scene.r, slope.r); curve_slope(2, 1, scene.g, slope.g); curve_slope(2, 2, scene.b, slope.b);
        const float3 pred = a * predictor(2, scene) + b;
        float3 ls[4];
        const int2 offs[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
        [unroll] for (uint i = 0; i < 4; ++i)
            ls[i] = log2(max(sc_scene_t.Load(int3(clamp(p + offs[i], int2(0, 0), last), 0)).rgb, 1.0 / 4096.0));
        const float3 g = max(abs(ls[1] - ls[0]), abs(ls[3] - ls[2]));
        const float gradient = dot(abs(a * slope) * g, float3(1, 1, 1) / 3.0);
        const bool hud = miss(sc_frame_t.Load(int3(p, 0)).rgb, pred) > 0.06 + 0.5 * gradient;
        sc_hud_u[id.xy] = hud ? 1.0 : 0.0;
        if (hud) InterlockedAdd(sc_hud_pixels, 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && sc_hud_pixels) sc_fit_u.InterlockedAdd(kHudCount, sc_hud_pixels);
}

// FrameWarp's own warp engine (experimental; used when chosen, or when NVIDIA's Latewarp is not available).
// clip_to_prev here maps the rendered frame's clip space (with depth) to the displayed camera's: rotation
// and movement together. For each output pixel we look for the rendered pixel that lands there once put
// at its own depth - a short fixed-point search (start at the same spot, move by the remaining miss),
// which converges in a few steps because the mapping is close to a shift. Rotation needs no depth; the
// depth makes near objects shift more than far ones when the camera moves (parallax), and the sky (at
// infinity) not at all.
// Where the search leaves the rendered frame (screen edges while turning), the edge is extended inwards
// with a short blend. No-warp mask pixels stay where they are, and warped pixels never take their colour
// from under the mask (they step past it to the nearest scene pixel). Games with HUD layers get their UI
// composited on top afterwards.
// rect: the valid depth region; grid: depth texture size. flags: 1 UI layer, 2 mask, 16 depth inverted.
Texture2D<float4> ow_color_t : register(t0);
Texture2D<float4> ow_ui_t : register(t1);
Texture2D<float> ow_mask_t : register(t2);
Texture2D<float> ow_depth_t : register(t3);
RWTexture2D<float4> ow_out_u : register(u0);
float4 ow_bilinear(float2 uv) {
    const float2 p = uv * float2(out_size) - 0.5, fl = floor(p), f = p - fl;
    const int2 last = int2(out_size) - 1, a = clamp(int2(fl), int2(0, 0), last), b = clamp(int2(fl) + 1, int2(0, 0), last);
    return lerp(lerp(ow_color_t.Load(int3(a.x, a.y, 0)), ow_color_t.Load(int3(b.x, a.y, 0)), f.x),
                lerp(ow_color_t.Load(int3(a.x, b.y, 0)), ow_color_t.Load(int3(b.x, b.y, 0)), f.x), f.y);
}
float ow_depth(float2 uv) {
    const uint2 p = rect.xy + min(uint2(saturate(uv) * float2(rect.zw)), rect.zw - 1);
    return ow_depth_t.Load(int3(p, 0));
}
// Rendered-frame uv (at its depth) -> displayed uv; w <= 0: behind the displayed camera.
float3 ow_forward(float2 uv) {
    const float4 f = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, ow_depth(uv), 1), clip_to_prev);
    return float3(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5, f.w);
}
[numthreads(8, 8, 1)] void cs_own_warp(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(out_size);
    float2 src = uv;
    bool valid = true;
    [loop] for (int it = 0; it < 6; ++it) {
        const float3 f = ow_forward(src);
        if (f.z <= 1e-6) { valid = false; break; }
        const float2 miss = uv - f.xy;
        src += miss;
        if (all(abs(miss * float2(out_size)) < 0.25)) break;
    }
    float4 c;
    if (!valid) {
        c = ow_color_t.Load(int3(id.xy, 0));  // nothing sensible to show (e.g. more than 90 degrees away)
    } else {
        const float2 half_px = 0.5 / float2(out_size);
        const float2 inside = clamp(src, half_px, 1.0 - half_px);
        if (all(inside == src)) {
            c = ow_bilinear(src);
            // Held (masked) pixels are no scene: a warped pixel landing on them would copy the HUD or the
            // weapon to a second, moving place. Step past them along the warp to the nearest scene pixel.
            if (flags & 2) {
                const int2 last = int2(out_size) - 1;
                if (ow_mask_t.Load(int3(clamp(int2(src * float2(out_size)), int2(0, 0), last), 0)) > 0.5) {
                    const float2 away = normalize((src - uv) * float2(out_size) + float2(1e-3, 0));
                    bool found = false;
                    [loop] for (int k = 1; k <= 48 && !found; ++k)
                        [unroll] for (int side = 0; side < 2; ++side) {
                            const float2 q = src * float2(out_size) + away * (side ? -k : k) * 2.0;
                            const int2 qi = clamp(int2(q), int2(0, 0), last);
                            if (!found && ow_mask_t.Load(int3(qi, 0)) <= 0.5) { c = ow_color_t.Load(int3(qi, 0)); found = true; }
                        }
                }
            }
        } else {
            // Revealed edge: the nearest edge pixels, blended with a few taps further in along the same
            // direction, so the band reads as a soft continuation rather than streaks.
            const float2 inward = normalize(inside - src + 1e-9) / float2(out_size);
            const float reach = min(length((src - inside) * float2(out_size)), 24.0);
            c = 0;
            [unroll] for (int k = 0; k < 4; ++k) c += ow_bilinear(inside + inward * (reach * 0.25 * k));
            c *= 0.25;
        }
    }
    if ((flags & 2) && ow_mask_t.Load(int3(id.xy, 0)) > 0.5) c = ow_color_t.Load(int3(id.xy, 0));
    if (flags & 1) {
        const float4 ui = ow_ui_t.Load(int3(id.xy, 0));
        c.rgb = c.rgb * (1.0 - ui.a) + ui.rgb;
    }
    ow_out_u[id.xy] = float4(c.rgb, 1);
}

// Debug view on the warped output: masked pixels magenta, HUD score still below the threshold green.
Texture2D<unorm float> tint_mask_t : register(t0);
Texture2D<float> tint_score_t : register(t1);
RWTexture2D<float4> tint_u : register(u0);
[numthreads(8, 8, 1)] void cs_tint(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float m = tint_mask_t.Load(int3(id.xy, 0)), s = tint_score_t.Load(int3(id.xy, 0));
    const float4 c = tint_u[id.xy];
    if (m > 0.5) tint_u[id.xy] = float4(lerp(c.rgb, float3(1, 0, 1), 0.5), c.a);
    else if (s > 0.05) tint_u[id.xy] = float4(lerp(c.rgb, float3(0, 1, 0), 0.5 * saturate(s / 0.6)), c.a);
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
constexpr UINT kXTintSrv = 120, kXTintUav = 126, kXSceneSrv = 128, kXSceneUav = 134, kXOwnSrv = 136, kXOwnUav = 142;
constexpr UINT kHeapSize = 144;
// Scene HUD fit buffer (see cs_scene_*): curve accumulators and curves, wash-out accumulators and
// values, 2 tile sets, HUD pixel count (same layout as the shader's constants).
constexpr UINT kSceneBins = 64, kSceneTile = 128, kSceneMaxTiles = 4096;
constexpr UINT kSceneHudCount = 3 * kSceneBins * 3 * 2 * 4 + 3 * kSceneBins * 3 * 4 + 2 * kSceneBins * 4 * 4 + 2 * kSceneBins * 2 * 4 +
                                2 * kSceneMaxTiles * 3 * 2 * 4;
constexpr UINT kSceneFitBytes = kSceneHudCount + 16;
constexpr UINT kSceneStamps = 19;  // before the first pass and after each of the 18

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
    // GPU time of each scene-HUD pass (one set per frame in flight), for the log.
    D3D12_QUERY_HEAP_DESC sq{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 3 * kSceneStamps, 0};
    device_->CreateQueryHeap(&sq, IID_PPV_ARGS(&scene_stamps_));
    bd.Width = 3 * kSceneStamps * 8;
    device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&scene_stamps_readback_));

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
        {"cs_hud_count", &cs_hud_count_}, {"cs_clear_counts", &cs_clear_counts_}, {"cs_sample", &cs_sample_}, {"cs_tint", &cs_tint_},
        {"cs_scene_clear", &cs_scene_clear_}, {"cs_scene_accum", &cs_scene_accum_}, {"cs_scene_finish", &cs_scene_finish_},
        {"cs_scene_tiles", &cs_scene_tiles_}, {"cs_scene_hud", &cs_scene_hud_}, {"cs_scene_grey", &cs_scene_grey_},
        {"cs_scene_grey_finish", &cs_scene_grey_finish_}, {"cs_scene_wash", &cs_scene_wash_}, {"cs_scene_wash_finish", &cs_scene_wash_finish_},
        {"cs_own_warp", &cs_own_warp_}};
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
    bd.Width = kSceneFitBytes;
    if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&scene_fit_)))) {
        error = "scene HUD buffer"; return false;
    }
    D3D12_HEAP_PROPERTIES rbh{}; rbh.Type = D3D12_HEAP_TYPE_READBACK;
    bd.Width = 3 * 2 * 16; bd.Flags = D3D12_RESOURCE_FLAG_NONE;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&fit_readback_)))) {
        error = "motion fit readback"; return false;
    }
    bd.Width = 3 * 16;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&hud_readback_)))) {
        error = "HUD counter readback"; return false;
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

// Root constants of the compute passes (the renderer's header names the type).
struct XConstants {
    float clip_to_prev[16];
    std::uint32_t rect[4], grid[2], mv_size[2], out_size[2];
    float mv_scale[2], alpha, threshold;
    std::uint32_t groups_x, flags;
};
static_assert(sizeof(XConstants) == 32 * 4, "root constants");

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
                                             bool keep_attached, bool depth_inverted) {
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
    const bool from_scene = hud && src.has_scene && private_[kPScene].texture && private_[kPScene].width == ow && private_[kPScene].height == oh;
    if (from_scene) {
        hud_from_scene_ = true;
        detect_hud_from_scene(c);
    } else if (hud_from_scene_) {
        // The upscaler's output stopped coming: the learned map starts over.
        hud_from_scene_ = false;
        reset_hud_ = true;
    }
    if (hud && previous && !from_scene) {
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
        // The counters go to the log (how often the whole-frame guard holds learning back).
        auto to_copy = transition_barrier(hud_counts_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list_->ResourceBarrier(1, &to_copy);
        list_->CopyBufferRegion(hud_readback_.Get(), UINT64(frame_index_) * 16, hud_counts_.Get(), 0, 8);
        std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
        list_->ResourceBarrier(1, &to_copy);
        hud_pending_[frame_index_] = true;
    }
    transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    set_x_srv(kXMaskSrv + 0, kPHudScore);
    set_x_srv(kXMaskSrv + 1, kPObject);
    for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXMaskSrv + i, kPObject);
    set_x_uav(kXMaskUav + 0, kPMask); set_x_uav(kXMaskUav + 1, kPMask);
    c.flags = (hud ? 4u : 0u) | (attached && keep_attached ? 8u : 0u);
    transition(private_[kPMask], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_mask_.Get(), &c, kXMaskSrv, kXMaskUav, (ow + 7) / 8, (oh + 7) / 8);
    transition(private_[kPMask], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    mask_ready_ = true;
    return private_[kPMask].texture.Get();
}

void Renderer::detect_hud_from_scene(const XConstants& base) {
    const UINT ow = private_[kPBackbuffer].width, oh = private_[kPBackbuffer].height;
    XConstants c = base;
    c.grid[0] = (ow + kSceneTile - 1) / kSceneTile; c.grid[1] = (oh + kSceneTile - 1) / kSceneTile;
    if (c.grid[0] * c.grid[1] > kSceneMaxTiles) return;
    set_x_srv(kXSceneSrv + 0, kPBackbuffer);
    set_x_srv(kXSceneSrv + 1, kPScene);
    for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXSceneSrv + i, kPScene);
    set_x_uav(kXSceneUav + 0, kPHudScore);
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
    raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.Buffer.NumElements = kSceneFitBytes / 4; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device_->CreateUnorderedAccessView(scene_fit_.Get(), nullptr, &raw, cpu(kXSceneUav + 1));
    transition(private_[kPHudScore], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    D3D12_RESOURCE_BARRIER fit{}; fit.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; fit.UAV.pResource = scene_fit_.Get();
    auto run_pass = [&](ID3D12PipelineState* pso, UINT read, UINT write, UINT rule, UINT gx, UINT gy) {
        c.rect[0] = read; c.rect[1] = write; c.rect[2] = rule; c.rect[3] = 0;
        x_dispatch(pso, &c, kXSceneSrv, kXSceneUav, gx, gy);
        list_->ResourceBarrier(1, &fit);
    };
    UINT stamp = 0;
    auto mark = [&]() { if (scene_stamps_) list_->EndQuery(scene_stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * kSceneStamps + stamp++); };
    auto run = [&](ID3D12PipelineState* pso, UINT read, UINT write, UINT rule, UINT gx, UINT gy) {
        run_pass(pso, read, write, rule, gx, gy);
        mark();
    };
    const UINT hx = (ow / 4 + 7) / 8, hy = (oh / 4 + 7) / 8;
    mark();
    run(cs_scene_clear_.Get(), 0, 0, 0, 1, 1);
    run(cs_scene_accum_.Get(), 0, 0, 0, hx, hy);   // curve 0 from every pixel
    run(cs_scene_finish_.Get(), 0, 0, 0, 1, 1);
    run(cs_scene_accum_.Get(), 0, 1, 1, hx, hy);   // curve 1 without what curve 0 misses
    run(cs_scene_finish_.Get(), 0, 1, 0, 1, 1);
    run(cs_scene_grey_.Get(), 0, 0, 0, hx, hy);    // wash-out 0 over curve 1, from every pixel (what the
    run(cs_scene_grey_finish_.Get(), 0, 0, 0, 1, 1);  // curve misses systematically is what it must learn)
    run(cs_scene_wash_.Get(), 1, 0, 0, hx, hy);
    run(cs_scene_wash_finish_.Get(), 0, 0, 0, 1, 1);
    run(cs_scene_tiles_.Get(), 1, 0, 0, c.grid[0], c.grid[1]);  // tile set 0 over predictor 1, every pixel
    run(cs_scene_accum_.Get(), 0, 2, 2, hx, hy);   // then all again without what that still misses:
    run(cs_scene_finish_.Get(), 0, 2, 0, 1, 1);    // curve 2,
    run(cs_scene_grey_.Get(), 0, 1, 2, hx, hy);    // wash-out 1 over curve 2,
    run(cs_scene_grey_finish_.Get(), 0, 1, 0, 1, 1);
    run(cs_scene_wash_.Get(), 2, 1, 2, hx, hy);
    run(cs_scene_wash_finish_.Get(), 0, 1, 0, 1, 1);
    run(cs_scene_tiles_.Get(), 2, 1, 2, c.grid[0], c.grid[1]);  // tile set 1 over predictor 2
    run(cs_scene_hud_.Get(), 2, 1, 0, (ow + 7) / 8, (oh + 7) / 8);
    // The HUD pixel count goes to the log.
    auto to_copy = transition_barrier(scene_fit_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    list_->CopyBufferRegion(hud_readback_.Get(), UINT64(frame_index_) * 16 + 8, scene_fit_.Get(), kSceneHudCount, 4);
    std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
    list_->ResourceBarrier(1, &to_copy);
    if (scene_stamps_)
        list_->ResolveQueryData(scene_stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * kSceneStamps, kSceneStamps,
                                scene_stamps_readback_.Get(), UINT64(frame_index_) * kSceneStamps * 8);
    hud_scene_pending_[frame_index_] = true;
    hud_scene_pixels_ = double(ow) * double(oh);
}

bool Renderer::own_warp(const IngestedSource& src, bool use_ui_tags, bool use_mask, const float source_to_target[16], bool depth_inverted) {
    if (!src.valid || !src.has_depth || !private_[kPOutput].texture || !private_[kPDepth].texture) return false;
    const bool split = use_ui_tags && src.has_hudless && src.has_ui && private_[kPHudless].texture && private_[kPUi].texture;
    const bool mask = use_mask && mask_ready_ && private_[kPMask].texture && private_[kPMask].width == private_[kPOutput].width &&
                      private_[kPMask].height == private_[kPOutput].height;
    const UINT ow = private_[kPOutput].width, oh = private_[kPOutput].height;
    XConstants c{};
    std::memcpy(c.clip_to_prev, source_to_target, sizeof(c.clip_to_prev));
    c.out_size[0] = ow; c.out_size[1] = oh;
    const UINT dw = private_[kPDepth].width, dh = private_[kPDepth].height;
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, dw, dh};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, dw - r.x); c.rect[3] = std::min(r.h, dh - r.y);
    c.grid[0] = dw; c.grid[1] = dh;
    c.flags = (split ? 1u : 0u) | (mask ? 2u : 0u) | (depth_inverted ? 16u : 0u);
    set_x_srv(kXOwnSrv + 0, split ? kPHudless : kPBackbuffer);
    set_x_srv(kXOwnSrv + 1, split ? kPUi : kPZeroUi);
    set_x_srv(kXOwnSrv + 2, mask ? kPMask : kPDepth);
    set_x_srv(kXOwnSrv + 3, kPDepth);
    for (UINT i = 4; i < kXSrvCount; ++i) set_x_srv(kXOwnSrv + i, kPBackbuffer);
    set_x_uav(kXOwnUav + 0, kPOutput); set_x_uav(kXOwnUav + 1, kPOutput);
    transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_own_warp_.Get(), &c, kXOwnSrv, kXOwnUav, (ow + 7) / 8, (oh + 7) / 8);
    return true;
}

void Renderer::tint_mask() {
    if (!mask_ready_ || !private_[kPOutput].texture || !private_[kPHudScore].texture) return;
    const UINT ow = private_[kPOutput].width, oh = private_[kPOutput].height;
    if (private_[kPMask].width != ow || private_[kPMask].height != oh) return;
    XConstants c{};
    c.out_size[0] = ow; c.out_size[1] = oh;
    set_x_srv(kXTintSrv + 0, kPMask);
    set_x_srv(kXTintSrv + 1, kPHudScore);
    for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXTintSrv + i, kPMask);
    set_x_uav(kXTintUav + 0, kPOutput); set_x_uav(kXTintUav + 1, kPOutput);
    transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    D3D12_RESOURCE_BARRIER written{}; written.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; written.UAV.pResource = private_[kPOutput].texture.Get();
    list_->ResourceBarrier(1, &written);  // after Latewarp's writes
    x_dispatch(cs_tint_.Get(), &c, kXTintSrv, kXTintUav, (ow + 7) / 8, (oh + 7) / 8);
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
    LARGE_INTEGER a, b, f; QueryPerformanceCounter(&a);
    flush_and_wait();
    QueryPerformanceCounter(&b); QueryPerformanceFrequency(&f);
    last_flush_ms_ = float(double(b.QuadPart - a.QuadPart) * 1000.0 / double(f.QuadPart));
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
            auto a = ts[frame_index_ * 2], b = ts[frame_index_ * 2 + 1];
            const float ms = b > a ? static_cast<float>(double(b - a) * 1000.0 / double(timestamp_frequency_)) : 0.0f;
            if (intake_slot_[frame_index_]) {  // a new game frame taken in between refreshes
                if (b > a) intake_gpu_ms_ = ms;
                b = a;  // not a presented frame: no present timing from it
            } else if (b > a) gpu_ms_ = ms;
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
    if (hud_scene_pending_[frame_index_] && hud_readback_) {
        hud_scene_pending_[frame_index_] = false;
        std::uint32_t* v = nullptr;
        D3D12_RANGE range{frame_index_ * 16 + 8, frame_index_ * 16 + 12};
        if (SUCCEEDED(hud_readback_->Map(0, &range, reinterpret_cast<void**>(&v)))) {
            ++hud_stats_.scene_frames;
            hud_stats_.scene_share_sum += double(v[frame_index_ * 4 + 2]) / std::max(1.0, hud_scene_pixels_);
            D3D12_RANGE none{0, 0};
            hud_readback_->Unmap(0, &none);
        }
        std::uint64_t* ts = nullptr;
        D3D12_RANGE tr{frame_index_ * kSceneStamps * 8, (frame_index_ + 1) * kSceneStamps * 8};
        if (scene_stamps_readback_ && SUCCEEDED(scene_stamps_readback_->Map(0, &tr, reinterpret_cast<void**>(&ts)))) {
            const std::uint64_t* t = ts + frame_index_ * kSceneStamps;
            for (UINT i = 0; i + 1 < kSceneStamps; ++i)
                if (t[i + 1] >= t[i]) hud_stats_.scene_pass_ms[i] += double(t[i + 1] - t[i]) * 1000.0 / double(timestamp_frequency_);
            D3D12_RANGE none{0, 0};
            scene_stamps_readback_->Unmap(0, &none);
        }
    }
    if (hud_pending_[frame_index_] && hud_readback_) {
        hud_pending_[frame_index_] = false;
        std::uint32_t* v = nullptr;
        D3D12_RANGE range{frame_index_ * 16, frame_index_ * 16 + 8};
        if (SUCCEEDED(hud_readback_->Map(0, &range, reinterpret_cast<void**>(&v)))) {
            const std::uint32_t evidence = v[frame_index_ * 4], changed = v[frame_index_ * 4 + 1];
            ++hud_stats_.frames;
            // Same rule as cs_hud.
            if (evidence + changed < 256) ++hud_stats_.too_little;
            else if (evidence > 0.35 * (evidence + changed)) ++hud_stats_.guarded;
            else { ++hud_stats_.learned; hud_stats_.share_sum += double(evidence) / double(evidence + changed); }
            D3D12_RANGE none{0, 0};
            hud_readback_->Unmap(0, &none);
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

IngestedSource Renderer::ingest(const Shared& shared, int slot, const std::function<void(const IngestedSource&)>& after_depth) {
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
    // Depth and motion vectors are recorded first: a caller that needs them on the CPU (camera
    // estimation) can flush here and wait for this small amount of work, before the 4K colour work.
    if (after_depth) after_depth(out);

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
    const auto& sc = m.tex[kScene];
    if (sources[kScene] && sc.width == bb.width && sc.height == bb.height &&
        ensure_private(kPScene, bb.width, bb.height, DXGI_FORMAT_R16G16B16A16_FLOAT)) {
        convert(sources[kScene], static_cast<DXGI_FORMAT>(sc.format), slot, kScene, kPScene, bb.width, bb.height);
        out.has_scene = true;
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

void Renderer::submit_work() {
    if (timestamps_) {
        list_->EndQuery(timestamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 2 + 1);
        list_->ResolveQueryData(timestamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 2, 2, readback_.Get(), frame_index_ * 16);
    }
    list_->Close();
    if (pending_game_wait_ && fence_game_) { queue_->Wait(fence_game_.Get(), pending_game_wait_); pending_game_wait_ = 0; }
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    queue_->Signal(fence_.Get(), ++fence_value_);
    frame_values_[frame_index_] = fence_value_;
    intake_slot_[frame_index_] = true;
}

void Renderer::finish_frame(bool warped, int marker) {
    intake_slot_[frame_index_] = false;
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

bool Renderer::read_back(int which, std::vector<std::uint16_t>& pixels, std::uint32_t& w, std::uint32_t& h) {
    auto& p = private_[which == 1 ? kPOutput : which == 2 ? kPScene : kPBackbuffer];
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
