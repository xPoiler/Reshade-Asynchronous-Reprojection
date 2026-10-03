#include "presenter/renderer.hpp"
#include "presenter/pose.hpp"
#include <cstdlib>
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

// Compute passes on the game's render-resolution grid (depth/motion size) and at output resolution;
// `rect` is the valid render region. Motion is stored towards the previous frame, like the game's.
const char kShadersX[] = R"(
cbuffer X : register(b0) {
    row_major float4x4 clip_to_prev;  // the game's clipToPrevClip (row vector: prev = clip * M)
    uint4 rect;                       // valid render region: x, y, w, h
    uint2 grid;                       // render texture size
    uint2 mv_size;                    // motion vector texture size
    uint2 out_size;                   // colour / output size
    float2 mv_scale;                  // game motion vector -> uv
    float alpha;                      // moving objects: game frames back towards the previous frame
    float threshold;                  // render px of own motion before a pixel counts as moving
    uint groups_x;                    // analyze: groups per row; reduce: total groups
    uint flags;                       // bit 0: mv_scale valid
};


Texture2D<float> depth_t : register(t0);
Texture2D<float2> motion_t : register(t1);
Texture2D<float4> analyze_flow_t : register(t2);  // (flag 2: XPAR's own motion, z = trusted)
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
                // XPAR's own motion (flag 2), estimated from the picture: only where it is trusted can a pixel be
                // seen moving on its own. Elsewhere (flat or dark areas, a soft TAA picture) the motion is filled
                // in from around it and its error would pass for movement: up to a quarter of the picture, warped
                // along with that error, shook (RE2 with TAA).
                bool trusted = true;
                if (flags & 2) trusted = analyze_flow_t.Load(int3(min(uint2((float2(id.xy) + 0.5) * float2(out_size) / float2(grid)), out_size - 1), 0)).z > 0.5;
                const bool moving = (flags & 1) && trusted && dot(own, own) > limit * limit;
                // Attached to the camera (first-person weapon, hands): close to the camera and moving in a way the
                // camera motion does not explain - stuck to the screen while the world moves under it, or mid
                // animation (aiming in/out while walking). Nearby walls move exactly as the camera predicts.
                // Only near the camera (reversed-Z: d = near / distance, so d > 1/64 means closer than 64x the
                // near plane): the sky and distant scenery often have motion vectors that ignore the camera too.
                // Near the camera the same threshold as "moving": with a separate, higher one, the band of a
                // weapon whose motion vectors differ from the camera model by between the two (DOOM Eternal:
                // an oval over the gun) was neither kept still nor warped as scenery.
                // Further out (up to 4096x the near plane: a third-person character a few metres away) only what
                // is stuck to the screen while the camera clearly moves - objects moving on their own (cars,
                // people) do not do that, and they must keep warping. The near rule can be switched off (flag
                // 64): with an estimated camera the nearby floor can miss the camera model while strafing.
                const float2 screen_px = g * mv_scale * float2(rect.zw);
                // Still compared with the turn alone (flag 128, on by default): a camera orbiting a third-person
                // character keeps it nearly still on screen. The camera model includes the orbit, so the
                // character follows it and the rule above does not hold it - but the warp turns the view
                // around the camera and would swing it away until the next frame snaps it back. Hold what
                // moves under 20% of what the turn alone (the camera motion at infinite distance) moves.
                bool turn_still = false;
                if (flags & 128) {
                    const float4 far = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), clip_to_prev);
                    const float2 turn_px = far.w > 0 ? (float2(far.x / far.w * 0.5 + 0.5, 0.5 - far.y / far.w * 0.5) - uv) * float2(rect.zw)
                                                     : float2(0, 0);
                    turn_still = d > 1.0 / 4096.0 && length(turn_px) >= 2.0 && length(screen_px) < 0.2 * length(turn_px);
                }
                const bool attached = (flags & 1) && (flags & 16) &&
                                      ((!(flags & 64) && d > 1.0 / 64.0 && moving) ||
                                       (d > 1.0 / 4096.0 && length(cam_px) >= 2.0 && length(screen_px) < 0.2 * length(cam_px)) ||
                                       turn_still);
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

// Moving objects (option, XPAR engine). The passes below share a second constant block (b1).
cbuffer W : register(b1) {
    row_major float4x4 wmat;  // cs_own_warp: the previous frame's clip -> the displayed camera's
    float4 wproj;             // the projection's depth terms (a, b, c, e): clip z = vz a + b, clip w = vz c + e
};
// View distance (clip w) of a point from its stored depth, whichever way round depth is stored.
float view_w(float d) {
    const float den = d * wproj.z - wproj.x;
    if (abs(den) < 1e-20) return -1;
    const float vz = (wproj.y - d * wproj.w) / den;
    return vz * wproj.z + wproj.w;
}
// Moving objects: a pixel's offset to the previous frame (cs_obj_move) is either a world object's (cars,
// people: clip space, the warp's camera applies) or, w = kHeld, a held one's (character/weapon kept still:
// on-screen motion in NDC in xy, no camera warp).
static const float kHeld = 1e9;
bool ob_world(float4 o) { return any(o != 0) && o.w < kHeld * 0.5; }
bool ob_held(float4 o) { return o.w > kHeld * 0.5; }
// Nearer = larger, from the view distance (2^-16 .. 2^16 units, 32 steps per doubling).
float near_key(float d) {
    const float w = view_w(d);
    return w > 0 ? 1.0 + saturate((16.0 - log2(w)) / 32.0) * 1022.0 : 1.0;
}

// Frame generation (the game's own, with moving objects on), once per game frame per generated image. Its
// depth: this frame's, moved back along the motion vectors to the image's moment (alpha: 0 the previous
// frame .. 1 this one): the point at q in the image is the one at p with p + (1 - alpha) motion(p) = q. Flag 1:
// the motion vector scale is known (otherwise this frame's depth as it is).
Texture2D<float> gd_depth_t : register(t0);
Texture2D<float2> gd_motion_t : register(t1);
RWTexture2D<float> gd_out_u : register(u0);
[numthreads(8, 8, 1)] void cs_gen_depth(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    int2 p = int2(id.xy);
    if ((flags & 1) && all(id.xy >= rect.xy) && all(id.xy < rect.xy + rect.zw)) {
        const float2 q = (float2(id.xy - rect.xy) + 0.5) / float2(rect.zw);
        const int2 lo = int2(rect.xy), hi = int2(rect.xy + rect.zw) - 1;
        float2 uv = q;
        [unroll] for (int i = 0; i < 3; ++i) {
            const int2 at = clamp(lo + int2(floor(uv * float2(rect.zw))), lo, hi);
            uv = q - (1.0 - alpha) * gd_motion_t.Load(int3(min(uint2(at), mv_size - 1), 0)) * mv_scale;
        }
        p = clamp(lo + int2(floor(uv * float2(rect.zw))), lo, hi);
    }
    gd_out_u[id.xy] = gd_depth_t.Load(int3(p, 0));
}
// ...and, with the HUD-less picture and UI layer, the UI the image shows taken back out (the warp puts the
// frame's UI back over it): c = scene (1 - ui.a) + ui.rgb solved for the scene where the UI lets enough
// through, this frame's HUD-less picture where it covers it (blended in between).
Texture2D<float4> gu_ui_t : register(t0);
Texture2D<float4> gu_hudless_t : register(t1);
RWTexture2D<float4> gu_colour_u : register(u0);
[numthreads(8, 8, 1)] void cs_gen_unui(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float4 ui = gu_ui_t.Load(int3(id.xy, 0));
    if (all(ui == 0)) return;
    const float4 c = gu_colour_u[id.xy];
    const float3 under = gu_hudless_t.Load(int3(id.xy, 0)).rgb;
    const float3 scene = ui.a < 0.97 ? max((c.rgb - ui.rgb) / (1.0 - ui.a), 0.0) : under;
    gu_colour_u[id.xy] = float4(lerp(scene, under, saturate((ui.a - 0.5) / 0.45)), c.a);
}

// Moving objects, once per game frame before anything else: one object, one way of moving. The motion analysis
// tells pixel by pixel what moves with the camera (attached: kept still and moved along its own on-screen
// motion) and what moves on its own (moved in the world, then warped with the camera); a third-person
// character is often both - torso attached, legs and arms on their own - and the two ways disagree by the
// camera's motion, tearing it apart. So what moves on its own and touches something attached at about the same
// distance becomes attached too: jumps of 64, 32, ... 1 render px between pixels within 15% of each other's
// distance spread it over the whole object; the result goes back into the analysis, so the mask holds it too.
Texture2D<float4> oj_object_t : register(t0);
Texture2D<float> oj_kind_t : register(t1);
Texture2D<float> oj_depth_t : register(t2);
RWTexture2D<float> oj_kind_u : register(u0);
[numthreads(8, 8, 1)] void cs_obj_join_init(uint3 id : SV_DispatchThreadID) {
    if (all(id.xy < grid)) oj_kind_u[id.xy] = oj_object_t.Load(int3(id.xy, 0)).w;
}
[numthreads(8, 8, 1)] void cs_obj_join(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    float k = oj_kind_t.Load(int3(id.xy, 0));
    if (abs(k - 1.0) < 0.5) {
        const float w = view_w(oj_depth_t.Load(int3(id.xy, 0)));
        const int step = int(rect.x);
        [unroll] for (int y = -1; y <= 1; ++y)
            [unroll] for (int x = -1; x <= 1; ++x) {
                const int2 q = clamp(int2(id.xy) + int2(x, y) * step, int2(0, 0), int2(grid) - 1);
                if (oj_kind_t.Load(int3(q, 0)) > 1.5 && w > 0 && abs(view_w(oj_depth_t.Load(int3(q, 0))) - w) <= 0.15 * w) k = 2.0;
            }
    }
    oj_kind_u[id.xy] = k;
}
RWTexture2D<float4> oj_object_u : register(u0);
[numthreads(8, 8, 1)] void cs_obj_join_final(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    float4 o = oj_object_u[id.xy];
    if (o.w < 1.5 && oj_kind_t.Load(int3(id.xy, 0)) > 1.5) { o.w = 2.0; oj_object_u[id.xy] = o; }
}

// Once per game frame: where each moving pixel's point was in the previous frame, as an offset in this
// frame's clip space (normalised so that this frame's point is (x, y, depth, 1)): o = previous - now.
// Moving the point by s * o is a straight line in 3D between the two frames, whatever the camera did,
// so an object next to a camera that moves with it (a car's interior) stays where it belongs. Only
// pixels the motion analysis found moving on their own (om_object_t.w = 1); everything else 0.
// clip_to_prev: the previous frame's clip -> this frame's (the inverse of the game's clipToPrevClip).
Texture2D<float> om_depth_t : register(t0);
Texture2D<float2> om_motion_t : register(t1);
Texture2D<float4> om_object_t : register(t2);
Texture2D<float> om_prevz_t : register(t3);
Texture2D<float4> om_flow_t : register(t4);  // (flag 2) the motion seen in the picture: level-0 displacements, half the grid; z: confident
RWTexture2D<float4> om_out_u : register(u0);
RWTexture2D<uint> om_tiles_u : register(u1);  // per 8x8 block: anything moving (the refresh passes skip the rest)
groupshared uint om_any;
[numthreads(8, 8, 1)] void cs_obj_move(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {
    if (gi == 0) om_any = 0;
    GroupMemoryBarrierWithGroupSync();
    float4 o = 0;
    if (all(id.xy < grid)) {
        const float kind = om_object_t.Load(int3(id.xy, 0)).w;  // 1 moving on its own, 2 attached to the camera
        float2 motion = om_motion_t.Load(int3(min(id.xy, mv_size - 1), 0)) * mv_scale;
        // What the picture shows moving differently from the game's motion vectors (flag 2): shadows and lights
        // cast by moving things, glare, markers the game draws on them - the game's motion vectors there are the
        // ground's or none. Only where the picture's motion is confident (detail in both directions).
        bool picture = false;
        if (kind < 1.5 && (flags & 2)) {
            const float4 fl = om_flow_t.Load(int3(min(id.xy / 2, out_size - 1), 0));
            if (fl.z > 0.5) {
                const float2 seen = fl.xy * 2.0, game = motion * float2(rect.zw);  // (render px)
                const float2 dd = seen - game;
                if (dot(dd, dd) > max(4.0, 0.09 * dot(game, game))) { motion = seen / float2(rect.zw); picture = true; }
            }
        }
        if (all(id.xy >= rect.xy) && all(id.xy < rect.xy + rect.zw) && (kind > 0.5 || picture)) {
            const float2 uv = (float2(id.xy - rect.xy) + 0.5) / float2(rect.zw);
            const float d = om_depth_t.Load(int3(id.xy, 0));
            const float2 prev = uv + motion;
            // Attached and kept still (flag 1): its on-screen motion, shown without the camera's warp.
            if (kind > 1.5 && (flags & 1) && !picture) {
                if (all(prev >= 0) && all(prev < 1)) o = float4((prev.x - uv.x) * 2, (uv.y - prev.y) * 2, 0, kHeld);
            } else if (all(prev >= 0) && all(prev < 1)) {
                // The previous depth where it lands: of the 3x3 pixels there, the one closest to this depth (the
                // same surface, not what is next to it).
                const int2 pp = int2(rect.xy) + int2(prev * float2(rect.zw));
                float dp = 0, best = 1e30;
                [unroll] for (int y = -1; y <= 1; ++y)
                    [unroll] for (int x = -1; x <= 1; ++x) {
                        const int2 q = clamp(pp + int2(x, y), int2(rect.xy), int2(rect.xy + rect.zw) - 1);
                        const float v = om_prevz_t.Load(int3(q, 0));
                        const float e = abs(v - d) / max(abs(d), 1e-20) + 1e-3 * (abs(x) + abs(y));
                        if (e < best) { best = e; dp = v; }
                    }
                const float4 h = mul(float4(prev.x * 2 - 1, 1 - prev.y * 2, dp, 1), clip_to_prev);
                const float wc = view_w(d);
                if (h.w > 0 && wc > 0) {
                    const float4 hn = h / h.w;
                    const float wp = view_w(hn.z);
                    if (wp > 0) o = hn * (wp / wc) - float4(uv.x * 2 - 1, 1 - uv.y * 2, d, 1);
                }
            }
        }
        if (!all(isfinite(o))) o = 0;
        om_out_u[id.xy] = o;
        if (any(o != 0)) InterlockedOr(om_any, 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0) om_tiles_u[gid.xy] = om_any;
}

// A copy for the next game frame (read in place: the shown textures stay readable for both queues).
Texture2D<float4> cp_in4_t : register(t0);
RWTexture2D<float4> cp_out4_u : register(u0);
[numthreads(8, 8, 1)] void cs_copy4(uint3 id : SV_DispatchThreadID) { if (all(id.xy < grid)) cp_out4_u[id.xy] = cp_in4_t.Load(int3(id.xy, 0)); }
Texture2D<float> cp_in1_t : register(t0);
RWTexture2D<float> cp_out1_u : register(u0);
[numthreads(8, 8, 1)] void cs_copy1(uint3 id : SV_DispatchThreadID) { if (all(id.xy < grid)) cp_out1_u[id.xy] = cp_in1_t.Load(int3(id.xy, 0)); }

// Per refresh: every moving pixel claims the output pixels it covers at the displayed moment (`alpha`
// game frames back towards the previous frame, along its straight line, then the camera's warp); the
// nearest claim wins. Key: nearness (10 bits) | the claiming render pixel relative to the one under the
// output pixel (11 + 11 bits).
RWTexture2D<uint> os_keys_u : register(u0);
RWTexture2D<uint> os_out_tiles_u : register(u1);  // per 8x8 output block: claimed or marked (the other passes skip the rest)
// Everything (a new texture), or only the blocks the previous refresh touched (one thread per block).
[numthreads(8, 8, 1)] void cs_obj_clear(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    os_keys_u[id.xy] = 0;
    if (all((id.xy & 7) == 0)) os_out_tiles_u[id.xy / 8] = 0;
}
[numthreads(8, 8, 1)] void cs_obj_clear_tiles(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= (out_size + 7) / 8) || os_out_tiles_u[id.xy] == 0) return;
    os_out_tiles_u[id.xy] = 0;
    for (uint y = 0; y < 8; ++y)
        for (uint x = 0; x < 8; ++x)
            if (all(id.xy * 8 + uint2(x, y) < out_size)) os_keys_u[id.xy * 8 + uint2(x, y)] = 0;
}
Texture2D<float> os_mask_t : register(t2);
Texture2D<float> os_depth_t : register(t3);
Texture2D<float4> os_move_t : register(t6);
Texture2D<uint> os_tiles_t : register(t11);
[numthreads(8, 8, 1)] void cs_obj_splat(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 p = int2(rect.xy + id.xy);
    if (os_tiles_t.Load(int3(p / 8, 0)) == 0) return;
    const float4 o = os_move_t.Load(int3(p, 0));
    if (all(o == 0)) return;
    const float2 scale = float2(out_size) / float2(rect.zw);
    const float2 uv = (float2(id.xy) + 0.5) / float2(rect.zw);
    // Held pixels (the HUD) stay where they are: never a second, moving copy of them - except the character/weapon
    // with its own on-screen motion.
    const bool held = ob_held(o);
    if (!held && (flags & 2) && os_mask_t.Load(int3(min(uint2(uv * float2(out_size)), out_size - 1), 0)) > 0.95) return;
    const float d = os_depth_t.Load(int3(p, 0));
    const float2 half_size = min(scale, 4.0) * 0.5 + 0.75;  // (overlapping a little: no cracks; cs_obj_fix drops claims that miss)
    const uint near = uint(near_key(d)) << 22;
    // 0: where it is at the displayed moment (its claim); 1: where it would be if it stood still
    // (key 1: the warp may show it there, but it has moved on - cs_obj_fix looks again).
    [unroll] for (int which = 0; which < 2; ++which) {
        const float4 q = float4(uv.x * 2 - 1, 1 - uv.y * 2, d, 1);
        const float4 f = held ? float4(q.xy + (which == 0 ? alpha : 0.0) * o.xy, 0, 1) : mul(q + (which == 0 ? alpha : 0.0) * o, clip_to_prev);
        if (f.w <= 1e-6) continue;
        const float2 c = float2(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5) * float2(out_size);
        const int2 lo = int2(ceil(c - half_size - 0.5)), hi = int2(ceil(c + half_size - 0.5)) - 1;
        for (int y = max(lo.y, 0); y <= min(hi.y, int(out_size.y) - 1); ++y)
            for (int x = max(lo.x, 0); x <= min(hi.x, int(out_size.x) - 1); ++x) {
                const int2 under = int2(rect.xy) + int2(min(uint2((float2(x, y) + 0.5) / scale), rect.zw - 1));
                const int2 off = clamp(p - under, -1023, 1023);
                InterlockedMax(os_keys_u[int2(x, y)], which == 0 ? near | (uint(off.x + 1024) << 11) | uint(off.y + 1024) : 1u);
                InterlockedOr(os_out_tiles_u[int2(x, y) / 8], 1u);
            }
    }
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
// Combined HUD detection: what the upscaler-output detector finds is HUD only where the pixel does not
// follow the world. Two things per pixel, packed in one float (world evidence in the fraction, see
// pack_world):
//  - world evidence: towards 1 where the pixel changed exactly as the camera moved the scenery under it
//    (post-processed scenery: bloom, lights, effects), towards 0 where it stayed put while the camera
//    moved (same evidence and guard as cs_hud);
//  - persistence: how far (px) the camera has moved while the detector kept finding HUD at this very
//    spot. The HUD stays where it is on screen; scenery the detector catches moves on and loses it. A
//    see-through HUD panel changes with the world behind it and can look like it follows the world, but
//    it stays found at the same spot: past 1/12 of the screen width of camera movement it is HUD whatever
//    the world evidence says. When the detector stops finding it, the persistence halves every frame.
Texture2D<float> hud_found_t : register(t4);  // the upscaler-output detector's result this frame
float world_part(float v) { return frac(v); }
float persist_part(float v) { return floor(v); }
float pack_world(float persist, float world) { return floor(clamp(persist, 0.0, 4095.0)) + clamp(world, 0.0, 0.999); }
// Where the scenery at this pixel was in the previous frame (output px); z <= 0: unknown.
float3 hud_came_from(uint2 id) {
    const float2 uv = (float2(id) + 0.5) / float2(out_size);
    const int2 pr = int2(rect.xy) + int2(min(uint2(uv * float2(rect.zw)), rect.zw - 1));
    const float depth = hud_depth_t.Load(int3(pr, 0));
    const float4 pv = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1), clip_to_prev);
    if (pv.w <= 0) return float3(0, 0, 0);
    return float3(float2(id) + (float2(pv.x / pv.w * 0.5 + 0.5, 0.5 - pv.y / pv.w * 0.5) - uv) * float2(out_size), 1);
}
bool hud_follows_world(uint2 id) {
    const float3 came = hud_came_from(id);
    if (came.z <= 0) return false;
    const float2 from = came.xy;
    if (length(from - float2(id)) < 3.0) return false;
    // The camera rarely moves whole pixels: the best of the 4 pixels around where the scenery came from.
    const float3 c = hud_current_t.Load(int3(id, 0)).rgb;
    const int2 base = int2(floor(from - 0.5));
    float best = 1e9;
    [unroll] for (int i = 0; i < 4; ++i) {
        const int2 q = clamp(base + int2(i & 1, i >> 1), int2(0, 0), int2(out_size) - 1);
        const float3 d = abs(c - hud_previous_t.Load(int3(q, 0)).rgb);
        best = min(best, max(d.r, max(d.g, d.b)));
    }
    return best < 0.04;
}
[numthreads(8, 8, 1)] void cs_hud_world(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size) || !(flags & 2)) return;
    const float packed = hud_score_u[id.xy];
    float world = world_part(packed), persist = persist_part(packed);
    const float3 came = hud_came_from(id.xy);
    const float moved = came.z > 0 ? length(came.xy - float2(id.xy)) : 0.0;
    persist = hud_found_t.Load(int3(id.xy, 0)) > 0.5 ? persist + moved : floor(persist * 0.5);
    float weight;
    const uint kind = hud_classify(id.xy, weight);
    if (kind == 1) {
        const uint evidence = hud_counts_u.Load(0), changed = hud_counts_u.Load(4);
        if (evidence + changed >= 256 && evidence <= 0.35 * (evidence + changed)) world = lerp(world, 0.0, weight);
    } else if ((kind == 2 || kind == 4) && hud_follows_world(id.xy)) {
        world = lerp(world, 1.0, 0.5);
    }
    hud_score_u[id.xy] = pack_world(persist, world);
}

// Motion/depth samples on a grid (mv_size = grid size) over the render rect, for camera estimation.
// flags bit 1: the motion vectors are XPAR's own (see cs_flow_lk) - a sample counts only where they are
// confident (t2: the displacements, out_size: their size).
Texture2D<float4> sample_flow_t : register(t2);
RWStructuredBuffer<float4> sample_u : register(u0);
[numthreads(8, 8, 1)] void cs_sample(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= mv_size)) return;
    const uint2 p = rect.xy + uint2((float2(id.xy) + 0.5) * float2(rect.zw) / float2(mv_size));
    float valid = 1;
    if (flags & 2) valid = sample_flow_t.Load(int3(min(uint2((float2(id.xy) + 0.5) * float2(out_size) / float2(mv_size)), out_size - 1), 0)).z;
    sample_u[id.y * mv_size.x + id.x] = float4(motion_t.Load(int3(p, 0)), depth_t.Load(int3(p, 0)), valid);
}

[numthreads(8, 8, 1)] void cs_clear_score(uint3 id : SV_DispatchThreadID) { if (all(id.xy < out_size)) hud_score_u[id.xy] = 0; }

// HUD from the game's HUD-less picture (a game that sends it without a UI layer - Cyberpunk 2077 with frame
// generation): exactly where the frame differs from it, every frame (the mask widens it by a pixel). Unless
// they differ over half the screen or more: then the HUD-less picture is not this frame's picture without its
// HUD (an effect drawn after the HUD, another frame) and gives no HUD. The picture itself is the scene behind
// the HUD (the fill, alpha 1 on HUD pixels).
Texture2D<float4> hh_frame_t : register(t0);
Texture2D<float4> hh_hudless_t : register(t1);
RWByteAddressBuffer hh_counts_u : register(u1);  // [8] pixels that differ
bool hh_differs(uint2 p) {
    const float3 a = hh_frame_t.Load(int3(p, 0)).rgb, b = hh_hudless_t.Load(int3(p, 0)).rgb;
    const float3 d = abs(a - b);
    return max(d.r, max(d.g, d.b)) > max(threshold, 0.02 * max(max(b.r, b.g), b.b));
}
[numthreads(1, 1, 1)] void cs_hudless_clear(uint3 id : SV_DispatchThreadID) { hh_counts_u.Store(8, 0); }
[numthreads(8, 8, 1)] void cs_hudless_count(uint3 id : SV_DispatchThreadID) {
    uint ignored;
    if (all(id.xy < out_size) && hh_differs(id.xy)) hh_counts_u.InterlockedAdd(8, 1, ignored);
}
[numthreads(8, 8, 1)] void cs_hudless_score(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const bool matches = hh_counts_u.Load(8) * 2 < out_size.x * out_size.y;
    hud_score_u[id.xy] = matches && hh_differs(id.xy) ? 1.0 : 0.0;
}
Texture2D<float4> hh_fill_from_t : register(t0);  // (the fill pass: the HUD-less picture, then the score)
Texture2D<float> hh_score_t : register(t1);
RWTexture2D<float4> hh_fill_u : register(u0);
[numthreads(8, 8, 1)] void cs_hudless_fill(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    hh_fill_u[id.xy] = float4(hh_fill_from_t.Load(int3(id.xy, 0)).rgb, hh_score_t.Load(int3(id.xy, 0)) > 0.5 ? 1.0 : 0.0);
}

// Camera-attached pixels widened by 1 render px, at render resolution (R8), for the mask below.
Texture2D<float4> att_object_t : register(t0);
RWTexture2D<unorm float> att_u : register(u0);
[numthreads(8, 8, 1)] void cs_attached(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 pr = int2(rect.xy + id.xy);
    bool keep = false;
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x) {
            const int2 q = clamp(pr + int2(x, y), int2(rect.xy), int2(rect.xy + rect.zw) - 1);
            keep = keep || att_object_t.Load(int3(q, 0)).w > 1.5;
        }
    att_u[pr] = keep ? 1.0 : 0.0;
}

// Stretch around the character/weapon (option, XPAR engine): how far each render pixel is from the
// camera-attached ones, as a share of the stretch width mv_size.x (render px, up to 32): 0 next to them,
// 1 from the width on. Two passes: the distance along rows (px / 255), then the Euclidean one.
Texture2D<unorm float> ramp_att_t : register(t0);
Texture2D<unorm float> ramp_row_t : register(t0);
RWTexture2D<unorm float> ramp_u : register(u0);
[numthreads(8, 8, 1)] void cs_ramp_rows(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 pr = int2(rect.xy + id.xy);
    const int b = int(clamp(mv_size.x, 1u, 32u));
    int best = b;
    [loop] for (int x = -b; x <= b; ++x) {
        const int px = pr.x + x;
        if (px >= int(rect.x) && px < int(rect.x + rect.z) && abs(x) < best && ramp_att_t.Load(int3(px, pr.y, 0)) > 0.5)
            best = abs(x);
    }
    ramp_u[pr] = float(best) / 255.0;
}
[numthreads(8, 8, 1)] void cs_ramp(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 pr = int2(rect.xy + id.xy);
    const int b = int(clamp(mv_size.x, 1u, 32u));
    float best = float(b * b);
    [loop] for (int y = -b; y <= b; ++y) {
        const int py = pr.y + y;
        if (py < int(rect.y) || py >= int(rect.y + rect.w)) continue;
        const float dx = round(ramp_row_t.Load(int3(pr.x, py, 0)) * 255.0);
        best = min(best, dx * dx + float(y * y));
    }
    ramp_u[pr] = saturate(sqrt(best) / float(b));
}

// No-warp mask (output resolution, R8): HUD (score, widened by 1 px for anti-aliased edges) and
// camera-attached pixels (widened by 1 render px, see cs_attached). A second one for the XPAR warp (u1):
// 1 held; elsewhere 0.9 * (1 - the share of the warp's motion applied there) - with the stretch (flag
// 128, t3 the distance share from cs_ramp) less and less towards the character/weapon, otherwise all.
Texture2D<float> mask_score_t : register(t0);
Texture2D<float> mask_attached_t : register(t1);
Texture2D<float> mask_world_t : register(t2);  // combined HUD detection (flag 64): see cs_hud_world
Texture2D<unorm float> mask_ramp_t : register(t3);
bool mask_follows_world(int3 q) {
    const float v = mask_world_t.Load(q);
    return world_part(v) >= 0.5 && persist_part(v) < float(out_size.x) / 12.0;
}
RWTexture2D<unorm float> mask_u : register(u0);
RWTexture2D<unorm float> warp_mask_u : register(u1);
[numthreads(8, 8, 1)] void cs_mask(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(out_size);
    const int2 pr = int2(rect.xy) + int2(min(uint2(uv * float2(rect.zw)), rect.zw - 1));
    bool keep = false;
    if (flags & 4) {
        [unroll] for (int y = -1; y <= 1; ++y)
            [unroll] for (int x = -1; x <= 1; ++x) {
                const int3 q = int3(clamp(int2(id.xy) + int2(x, y), int2(0, 0), int2(out_size) - 1), 0);
                keep = keep || (mask_score_t.Load(q) > 0.6 && (!(flags & 64) || !mask_follows_world(q)));
            }
    }
    if (!keep && (flags & 8)) keep = mask_attached_t.Load(int3(pr, 0)) > 0.5;
    mask_u[id.xy] = keep ? 1.0 : 0.0;
    const float follow = (flags & 128) ? mask_ramp_t.Load(int3(pr, 0)) : 1.0;
    warp_mask_u[id.xy] = keep ? 1.0 : 0.9 * (1.0 - follow);
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
// What every pixel of the group reads, loaded once per group: curve 2, wash-out 1, and tile set 1 at the
// group's four tile centres (a group of 8x8 never straddles a tile centre: 64 is a multiple of 8).
groupshared float sh_mean[3 * kBins], sh_grey[kBins], sh_wash[kBins], sh_tile[4 * 6];
float sh_curve_slope(uint c, float v, out float slope) {
    const float f = bin_pos(v) - 0.5, fl = floor(f);
    const int i0 = clamp(int(fl), 0, int(kBins) - 1), i1 = clamp(int(fl) + 1, 0, int(kBins) - 1);
    const float m0 = sh_mean[c * kBins + uint(i0)], m1 = sh_mean[c * kBins + uint(i1)];
    slope = (m1 - m0) * (float(kBins) / 20.0);
    return lerp(m0, m1, saturate(f - fl));
}
float3 sh_predictor(float3 s) {  // predictor(2): curve 2, then wash-out 1
    float unused;
    const float3 a = float3(sh_curve_slope(0, s.r, unused), sh_curve_slope(1, s.g, unused), sh_curve_slope(2, s.b, unused));
    const float f = bin_pos(dot(s, kLuma)) - 0.5, fl = floor(f);
    const int i0 = clamp(int(fl), 0, int(kBins) - 1), i1 = clamp(int(fl) + 1, 0, int(kBins) - 1);
    const float t = saturate(f - fl);
    const float grey = lerp(sh_grey[i0], sh_grey[i1], t), k = lerp(sh_wash[i0], sh_wash[i1], t);
    return a + k * (grey - a);
}
void sh_load(uint3 gid, uint gi) {
    for (uint i = gi; i < 3 * kBins; i += 64) sh_mean[i] = mean_of(2, i / kBins, int(i % kBins));
    sh_grey[gi] = grey_of(1, int(gi));  // 64 threads, 64 bins
    sh_wash[gi] = wash_of(1, int(gi));
    const float2 t0 = (float2(gid.xy * 8) + 0.5) / float(kTile) - 0.5;
    const int2 fl0 = int2(floor(t0));
    if (gi < 24) {  // corner k, channel c, a or b
        const uint k = gi / 6, c = (gi % 6) / 2, ab = gi & 1;
        const int2 q = clamp(fl0 + int2(k & 1, k >> 1), int2(0, 0), int2(grid) - 1);
        sh_tile[gi] = asfloat(sc_fit_u.Load(kTileBase + ((1 * kMaxTiles + uint(q.y) * grid.x + uint(q.x)) * 3) * 8 + c * 8 + ab * 4));
    }
}
// Tile correction (set 1), bilinear between tile centres.
void sh_tile_ab(uint2 id, out float3 a, out float3 b) {
    const float2 t = (float2(id) + 0.5) / float(kTile) - 0.5;
    const float2 f = saturate(t - floor(t));
    a = 0; b = 0;
    [unroll] for (int k = 0; k < 4; ++k) {
        const float w = ((k & 1) ? f.x : 1 - f.x) * ((k >> 1) ? f.y : 1 - f.y);
        [unroll] for (uint c = 0; c < 3; ++c) { a[c] += w * sh_tile[k * 6 + c * 2]; b[c] += w * sh_tile[k * 6 + c * 2 + 1]; }
    }
}
[numthreads(8, 8, 1)] void cs_scene_hud(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    if (gi == 0) sc_hud_pixels = 0;
    sh_load(gid, gi);
    GroupMemoryBarrierWithGroupSync();
    if (all(id.xy < out_size)) {
        float3 a, b;
        sh_tile_ab(id.xy, a, b);
        const int2 p = int2(id.xy), last = int2(out_size) - 1;
        const float3 scene = max(sc_scene_t.Load(int3(p, 0)).rgb, 0);
        float3 slope;
        sh_curve_slope(0, scene.r, slope.r); sh_curve_slope(1, scene.g, slope.g); sh_curve_slope(2, scene.b, slope.b);
        const float3 pred = a * sh_predictor(scene) + b;
        float3 ls[4];
        const int2 offs[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
        [unroll] for (uint j = 0; j < 4; ++j)
            ls[j] = log2(max(sc_scene_t.Load(int3(clamp(p + offs[j], int2(0, 0), last), 0)).rgb, 1.0 / 4096.0));
        const float3 g = max(abs(ls[1] - ls[0]), abs(ls[3] - ls[2]));
        const float gradient = dot(abs(a * slope) * g, float3(1, 1, 1) / 3.0);
        const bool hud = miss(sc_frame_t.Load(int3(p, 0)).rgb, pred) > 0.06 + 0.5 * gradient;
        sc_hud_u[id.xy] = hud ? 1.0 : 0.0;
        if (hud) InterlockedAdd(sc_hud_pixels, 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && sc_hud_pixels) sc_fit_u.InterlockedAdd(kHudCount, sc_hud_pixels);
}
// Fill behind the HUD (opt-in): where the HUD was found, the same prediction - the scene the HUD covers,
// in the frame's colours - for the warp to show where the HUD moves away from (instead of stretching the
// surroundings). Effects the game adds after upscaling (bloom, grain) are only there as far as the
// colour model holds them.
Texture2D<float> sc_score_t : register(t2);
RWTexture2D<float4> sc_fill_u : register(u0);
// The HUD as the mask holds it: found here or next to it (the mask widens the HUD by 1 px, cs_mask).
bool hud_near(Texture2D<float> score, int2 p) {
    bool near = false;
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x)
            near = near || score.Load(int3(clamp(p + int2(x, y), int2(0, 0), int2(out_size) - 1), 0)) > 0.5;
    return near;
}
[numthreads(8, 8, 1)] void cs_scene_fill(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    sh_load(gid, gi);
    GroupMemoryBarrierWithGroupSync();
    if (any(id.xy >= out_size)) return;
    // Every pixel is written (alpha 1: filled here), so the warp needs nothing but this texture.
    if (!hud_near(sc_score_t, int2(id.xy))) { sc_fill_u[id.xy] = float4(0, 0, 0, 0); return; }
    float3 a, b;
    sh_tile_ab(id.xy, a, b);
    const float3 scene = max(sc_scene_t.Load(int3(id.xy, 0)).rgb, 0);
    sc_fill_u[id.xy] = float4(max(a * sh_predictor(scene) + b, 0), 1);
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
// With the fill behind the HUD (flag 4), a warped pixel landing on the HUD takes the scene predicted
// there from the upscaler's output instead (see cs_scene_fill; alpha 1 where filled).
// With the stretch (flag 128), the scenery around the character/weapon follows the warp less and less
// towards it (see cs_mask), so what the warp uncovers beside it is covered by stretched scenery instead of
// being filled, and the outline its motion vectors miss stays with it.
// Moving objects (flag 256, option; cs_obj_fix, after this pass): pixels the motion analysis found moving
// on their own are placed where they are at the displayed moment (cs_obj_splat claims, nearest wins), and
// what they uncover shows
// the previous frame, warped to the displayed camera with its own depth (wmat: its clip -> the displayed
// camera's); where the previous frame shows the object there too, the still scenery next to it is stretched.
// rect: the valid depth region; grid: depth texture size. flags: 1 UI layer, 2 mask, 4 fill behind the
// HUD, 16 depth inverted, 32 depth-independent (rotation only), 64 background memory (t4), 128 stretch,
// 256 moving objects (t6-t9; alpha: game frames back towards the previous frame).
// t2: the XPAR warp's mask (cs_mask u1): held above 0.95.
Texture2D<float4> ow_color_t : register(t0);
Texture2D<float4> ow_ui_t : register(t1);
Texture2D<float> ow_mask_t : register(t2);
Texture2D<float> ow_depth_t : register(t3);
Texture2D<float4> ow_fill_t : register(t5);  // the scene behind the HUD (alpha 1 where filled)
Texture2D<float4> ow_move_t : register(t6);   // moving objects: offset to the previous frame (cs_obj_move)
Texture2D<uint> ow_keys_t : register(t7);     // ...their claims on the output pixels (cs_obj_splat)
Texture2D<float4> ow_prev_t : register(t8);   // ...the previous frame's picture
Texture2D<float> ow_prevz_t : register(t9);   // ...and depth
Texture2D<uint> ow_out_tiles_t : register(t10); // ...the 8x8 output blocks they touch
RWTexture2D<float4> ow_out_u : register(u0);
SamplerState ow_linear : register(s0);  // bilinear, clamped to the edge
float4 ow_bilinear(float2 uv) { return ow_color_t.SampleLevel(ow_linear, uv, 0); }
bool ow_held(int3 p) { return ow_mask_t.Load(p) > 0.95; }
// The share of the warp's motion applied at this spot (the stretch).
float ow_follow(float2 uv) { return saturate(1.0 - ow_mask_t.SampleLevel(ow_linear, uv, 0) / 0.9); }
float ow_depth(float2 uv) {
    if (flags & 32) return 0;  // the warp does not depend on depth (the camera only turns): no reads
    const uint2 p = rect.xy + min(uint2(saturate(uv) * float2(rect.zw)), rect.zw - 1);
    return ow_depth_t.Load(int3(p, 0));
}
// Background memory (flag 64, see cs_memory): the scenery last seen at this spot of the source frame -
// behind held pixels, or outside the frame - if seen within the last 60 game frames. Bilinear where all
// four texels hold something, the nearest texel otherwise.
static const float kMemMargin = 0.125;  // the memory covers the screen plus this much on every side
static const float kMemForget = 60.0;   // game frames
Texture2D<float4> ow_mem_t : register(t4);
bool ow_from_memory(float2 s, out float4 c) {
    c = 0;
    const float2 mu = (s + kMemMargin) / (1.0 + 2.0 * kMemMargin);
    if (any(mu < 0) || any(mu > 1)) return false;
    const float4 ages = ow_mem_t.GatherAlpha(ow_linear, mu);
    if (max(max(ages.x, ages.y), max(ages.z, ages.w)) < kMemForget) {
        c = float4(ow_mem_t.SampleLevel(ow_linear, mu, 0).rgb, 1);
        return true;
    }
    uint mw, mh;
    ow_mem_t.GetDimensions(mw, mh);
    const float4 v = ow_mem_t.Load(int3(min(uint2(mu * float2(mw, mh)), uint2(mw, mh) - 1), 0));
    if (v.a >= kMemForget) return false;
    c = float4(v.rgb, 1);
    return true;
}
// Rendered-frame uv (at its depth) -> displayed uv; w <= 0: behind the displayed camera.
float3 ow_forward(float2 uv) {
    const float4 f = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, ow_depth(uv), 1), clip_to_prev);
    return float3(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5, f.w);
}
// Moving objects: how much of a spot of the rendered frame is a moving object, bilinear between the four
// render pixels around it (the object's outline at output resolution, not in render-pixel steps).
float ow_moving(float2 uv) {
    const float2 p = clamp(uv * float2(rect.zw) - 0.5, 0, float2(rect.zw) - 1.001);
    const int2 i = int2(rect.xy) + int2(floor(p));
    const float2 f = p - floor(p);
    const float a = ob_world(ow_move_t.Load(int3(i, 0))) ? 1.0 : 0.0, b = ob_world(ow_move_t.Load(int3(i + int2(1, 0), 0))) ? 1.0 : 0.0;
    const float c = ob_world(ow_move_t.Load(int3(i + int2(0, 1), 0))) ? 1.0 : 0.0, d = ob_world(ow_move_t.Load(int3(i + int2(1, 1), 0))) ? 1.0 : 0.0;
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}
// Is a spot a world object: half covered (smooth outlines), or right on one of its render pixels (thin
// parts - antennas, rails - are never half covered and would otherwise be neither moved nor cleared).
bool ow_on_world(float2 uv) {
    const int2 r = int2(rect.xy) + int2(min(uint2(saturate(uv) * float2(rect.zw)), rect.zw - 1));
    return ob_world(ow_move_t.Load(int3(r, 0))) || ow_moving(uv) >= 0.5;
}
// ...and of a held one with its own motion.
float ow_held_moving(float2 uv) {
    const float2 p = clamp(uv * float2(rect.zw) - 0.5, 0, float2(rect.zw) - 1.001);
    const int2 i = int2(rect.xy) + int2(floor(p));
    const float2 f = p - floor(p);
    const float a = ob_held(ow_move_t.Load(int3(i, 0))) ? 1.0 : 0.0, b = ob_held(ow_move_t.Load(int3(i + int2(1, 0), 0))) ? 1.0 : 0.0;
    const float c = ob_held(ow_move_t.Load(int3(i + int2(0, 1), 0))) ? 1.0 : 0.0, d = ob_held(ow_move_t.Load(int3(i + int2(1, 1), 0))) ? 1.0 : 0.0;
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}
// Moving objects: the colour at output pixel p of the rendered frame, carried along with the moving object
// there (alpha game frames back along its motion); c when nothing moves there.
float4 ow_carried(int2 p, float4 c) {
    const float2 u = (float2(p) + 0.5) / float2(out_size);
    const int2 r = int2(rect.xy) + int2(min(uint2(u * float2(rect.zw)), rect.zw - 1));
    const float4 o = ow_move_t.Load(int3(r, 0));
    if (!ob_world(o)) return c;
    const float4 a = float4(u.x * 2 - 1, 1 - u.y * 2, ow_depth_t.Load(int3(r, 0)), 1), b = a + alpha * o;
    if (b.w <= 1e-6) return c;
    return ow_bilinear(u - float2(b.x / b.w - a.x, a.y - b.y / b.w) * 0.5);
}
// The rendered pixel that lands on uv; false: none (behind the displayed camera).
bool ow_find(float2 uv, out float2 src) {
    src = uv;
    const int steps = (flags & 128) ? 12 : 6;
    [loop] for (int it = 0; it < steps; ++it) {
        const float3 f = ow_forward(src);
        if (f.z <= 1e-6) return false;
        float2 lands = f.xy;
        if (flags & 128) lands = src + (f.xy - src) * ow_follow(src);
        const float2 miss = uv - lands;
        src += miss;
        if (all(abs(miss * float2(out_size)) < 0.25)) break;
    }
    return true;
}
// What the warp shows at an output pixel that is not held (src: where its search landed).
float4 ow_scene(uint2 id, float2 uv, out float2 src, out bool valid) {
    valid = ow_find(uv, src);
    float4 c;
    if (!valid) {
        c = ow_color_t.Load(int3(id.xy, 0));  // nothing sensible to show (e.g. more than 90 degrees away)
    } else {
        const float2 half_px = 0.5 / float2(out_size);
        const float2 inside = clamp(src, half_px, 1.0 - half_px);
        if (all(inside == src)) {
            c = ow_bilinear(src);
            // (moving objects: where the warp shows one without the moving-object pass looking again - beside the
            // character/weapon, where the scenery follows the warp less - it moves with the object)
            if (flags & 256) c = ow_carried(int2(src * float2(out_size)), c);
            // Held (masked) pixels are no scene: a warped pixel landing on them would copy the HUD or the
            // weapon to a second, moving place. Step past them along the warp to the nearest scene pixel.
            if (flags & 2) {
                const int2 last = int2(out_size) - 1;
                const int3 si = int3(clamp(int2(src * float2(out_size)), int2(0, 0), last), 0);
                const float4 filled = (flags & 4) ? ow_fill_t.Load(si) : float4(0, 0, 0, 0);
                if (filled.a > 0.5 && ow_held(si)) {
                    c = float4(filled.rgb, 1);
                } else if (ow_held(si)) {
                    const float2 away = normalize((src - uv) * float2(out_size) + float2(1e-3, 0));
                    // (HLSL evaluates both sides of &&: an explicit branch, so nothing is read without the memory)
                    bool found = false;
                    if (flags & 64) {
                        float4 remembered;
                        if (ow_from_memory(src, remembered)) { c = remembered; found = true; }
                    }
                    [loop] for (int k = 1; k <= 48 && !found; ++k)
                        [unroll] for (int side = 0; side < 2; ++side) {
                            const float2 q = src * float2(out_size) + away * (side ? -k : k) * 2.0;
                            const int2 qi = clamp(int2(q), int2(0, 0), last);
                            if (!found && !ow_held(int3(qi, 0))) {
                                c = ow_color_t.Load(int3(qi, 0));
                                found = true;
                                // Moving objects (flag 256): a moving object's pixel is taken from where the object
                                // is at the displayed moment, so the copy moves with it instead of splitting from it.
                                if (flags & 256) c = ow_carried(qi, c);
                            }
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
            if (flags & 64) {
                float4 remembered;
                if (ow_from_memory(src, remembered)) c = remembered;
            }
        }
    }
    return c;
}
[numthreads(8, 8, 1)] void cs_own_warp(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(out_size);
    float2 src;
    bool valid;
    float4 c = ow_scene(id.xy, uv, src, valid);
    if ((flags & 2) && ow_held(int3(id.xy, 0))) c = ow_color_t.Load(int3(id.xy, 0));
    if (flags & 1) {
        const float4 ui = ow_ui_t.Load(int3(id.xy, 0));
        c.rgb = c.rgb * (1.0 - ui.a) + ui.rgb;
    }
    ow_out_u[id.xy] = float4(c.rgb, 1);
}

// Moving objects (option), after the warp: only the output pixels a moving pixel claimed (cs_obj_splat)
// or may have left (key 1: where it would land if it stood still) are looked at again.
[numthreads(8, 8, 1)] void cs_obj_fix(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size) || ow_out_tiles_t.Load(int3(id.xy / 8, 0)) == 0) return;
    const uint key = ow_keys_t.Load(int3(id.xy, 0));
    if (key == 0) return;
    const float2 rscale = float2(out_size) / float2(rect.zw);
    const int2 under = int2(rect.xy) + int2(min(uint2((float2(id.xy) + 0.5) / rscale), rect.zw - 1));
    // Held here: the HUD (and anything held without its own motion) stays as the warp shows it; the
    // character/weapon is revisited (it may have moved away from here, or something else moved in) - also the
    // pixel or two the mask is widened by around it (a render pixel next to this one is held with its motion).
    const bool held_here = (flags & 2) && ow_held(int3(id.xy, 0));
    if (held_here) {
        bool moving_held = false;
        [unroll] for (int y = -1; y <= 1; ++y)
            [unroll] for (int x = -1; x <= 1; ++x)
                moving_held = moving_held || ob_held(ow_move_t.Load(int3(clamp(under + int2(x, y), int2(rect.xy), int2(rect.xy + rect.zw) - 1), 0)));
        if (!moving_held) return;
    }
    const float2 uv = (float2(id.xy) + 0.5) / float2(out_size);
    float2 src;
    bool valid;
    const float4 scene = ow_scene(id.xy, uv, src, valid);
    float4 c = 0;
    bool changed = false;
    // Where the search landed: a moving pixel (its object has moved on by the displayed moment) or not.
    const int2 rs = int2(rect.xy) + int2(min(uint2(saturate(src) * float2(rect.zw)), rect.zw - 1));
    const float4 landed_move = valid ? ow_move_t.Load(int3(rs, 0)) : float4(0, 0, 0, 0);
    const bool vacated = valid && ow_on_world(src);
    const float landed_key = valid && !vacated ? near_key(ow_depth_t.Load(int3(rs, 0))) : 0.0;
    bool claimed = false;
    if (key > 1 && float(key >> 22) + 2.0 >= landed_key) {
        // A moving object is here: its exact spot in the rendered frame. From the claiming pixel (any of the
        // object's pixels around here: several claim with the same nearness), a few steps across the object's
        // surface, each with the motion and depth of the pixel it stands on; never off the object. Claims
        // overlap a little (no cracks between them); one whose object does not reach this pixel (its outline)
        // is dropped.
        const int2 sp = under + int2(int((key >> 11) & 2047u) - 1024, int(key & 2047u) - 1024);
        float4 o = ow_move_t.Load(int3(sp, 0));
        float d = ow_depth_t.Load(int3(sp, 0));
        float2 s = (float2(sp - int2(rect.xy)) + 0.5) / float2(rect.zw);
        if (ob_held(o)) {
            // The character/weapon: along its own on-screen motion (no camera warp).
            [unroll] for (int it = 0; it < 3; ++it) {
                const float2 next = uv - alpha * o.xy * float2(0.5, -0.5);
                const int2 np = int2(rect.xy) + int2(min(uint2(saturate(next) * float2(rect.zw)), rect.zw - 1));
                const float4 no = ow_move_t.Load(int3(np, 0));
                if (!ob_held(no) || ow_held_moving(next) < 0.5) break;
                s = next; o = no;
            }
            const float2 miss = (uv - (s + alpha * o.xy * float2(0.5, -0.5))) * float2(out_size);
            if (all(abs(miss) < 0.75) && ow_held_moving(s) >= 0.5) {
                c = ow_bilinear(s);
                changed = claimed = true;
            }
        } else {
        [unroll] for (int it = 0; it < 4; ++it) {
            const float4 f = mul(float4(s.x * 2 - 1, 1 - s.y * 2, d, 1) + alpha * o, clip_to_prev);
            if (f.w <= 1e-6) break;
            const float2 next = s + uv - float2(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5);
            const int2 np = int2(rect.xy) + int2(min(uint2(saturate(next) * float2(rect.zw)), rect.zw - 1));
            const float4 no = ow_move_t.Load(int3(np, 0));
            if (!ob_world(no) || !ow_on_world(next)) break;
            s = next; o = no; d = ow_depth_t.Load(int3(np, 0));
        }
        const float4 f = mul(float4(s.x * 2 - 1, 1 - s.y * 2, d, 1) + alpha * o, clip_to_prev);
        const float2 miss = (uv - float2(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5)) * float2(out_size);
        if (f.w > 1e-6 && all(abs(miss) < 0.75) && ow_on_world(s)) {
            c = ow_bilinear(s);
            changed = claimed = true;
        } else {
            // The steps across the surface found no exact spot (fine structure: a mesh, foliage, where the depth
            // changes every pixel or two): the claiming pixel's own displacement, if that still lands on the
            // moving object - everything there moves with it.
            const float2 c0 = (float2(sp - int2(rect.xy)) + 0.5) / float2(rect.zw);
            const float4 f0 = mul(float4(c0.x * 2 - 1, 1 - c0.y * 2, ow_depth_t.Load(int3(sp, 0)), 1) + alpha * ow_move_t.Load(int3(sp, 0)), clip_to_prev);
            if (f0.w > 1e-6) {
                const float2 s0 = uv - (float2(f0.x / f0.w * 0.5 + 0.5, 0.5 - f0.y / f0.w * 0.5) - c0);
                if (ow_on_world(s0)) {
                    c = ow_bilinear(s0);
                    changed = claimed = true;
                }
            }
        }
        }
    }
    if (!claimed && vacated) {
        changed = true;
        // Uncovered: what was behind the object, from the previous frame (found as the warp finds any
        // pixel, with the previous frame's own depth).
        float2 q = uv;
        bool ok = true;
        [loop] for (int it = 0; it < 6; ++it) {
            const uint2 pz = rect.xy + min(uint2(saturate(q) * float2(rect.zw)), rect.zw - 1);
            const float4 f = mul(float4(q.x * 2 - 1, 1 - q.y * 2, ow_prevz_t.Load(int3(pz, 0)), 1), wmat);
            if (f.w <= 1e-6) { ok = false; break; }
            const float2 miss = uv - float2(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5);
            q += miss;
            if (all(abs(miss * float2(out_size)) < 0.25)) break;
        }
        ok = ok && all(q >= 0) && all(q <= 1);
        const float object_key = near_key(ow_depth_t.Load(int3(rs, 0)));
        const uint2 qz = rect.xy + min(uint2(saturate(q) * float2(rect.zw)), rect.zw - 1);
        if (ok && near_key(ow_prevz_t.Load(int3(qz, 0))) + 2.0 < object_key) {
            c = ow_prev_t.SampleLevel(ow_linear, q, 0);
            if (flags & 512) c = float4(0, 0, 1, 1);
        } else {
            // The previous frame shows the object there too: the nearest still scenery along its motion.
            const float d = ow_depth_t.Load(int3(rs, 0));
            const float4 a = mul(float4(src.x * 2 - 1, 1 - src.y * 2, d, 1), clip_to_prev);
            const float4 b = mul(float4(src.x * 2 - 1, 1 - src.y * 2, d, 1) + landed_move, clip_to_prev);
            float2 dir = (b.xy / b.w - a.xy / a.w) * float2(0.5, -0.5) * float2(out_size);
            dir = all(isfinite(dir)) && dot(dir, dir) > 1e-6 ? normalize(dir) : float2(1, 0);
            const int2 last = int2(out_size) - 1;
            bool found = false;
            [loop] for (int k = 1; k <= 48 && !found; ++k)
                [unroll] for (int side = 0; side < 2; ++side) {
                    const float2 t = src * float2(out_size) + dir * (side ? -k : k) * 2.0;
                    const int2 ti = clamp(int2(t), int2(0, 0), last);
                    const int2 tr = int2(rect.xy) + int2(min(uint2((float2(ti) + 0.5) / rscale), rect.zw - 1));
                    if (!found && all(ow_move_t.Load(int3(tr, 0)) == 0) && !((flags & 2) && ow_held(int3(ti, 0)))) {
                        c = ow_color_t.Load(int3(ti, 0));
                        found = true;
                    }
                }
            if (!found) {
                // None within reach (the object fills the view along its motion, or the character stands next
                // to it): the object's own picture, carried along with it, so a piece of it the warp copied here
                // moves with the object instead of staying behind.
                const float4 a2 = float4(src.x * 2 - 1, 1 - src.y * 2, d, 1), b2 = a2 + alpha * landed_move;
                const float2 shift = b2.w > 1e-6 ? float2(b2.x / b2.w - a2.x, a2.y - b2.y / b2.w) * 0.5 : float2(0, 0);
                c = ow_bilinear(src - shift);
            }
            changed = true;
            if (flags & 512) c = found ? float4(1, 0, 0, 1) : float4(1, 0, 1, 1);
        }
    }
    // Held here, unclaimed and not uncovering an object: the character/weapon has moved away - the scene behind
    // it, as the warp shows it next to held pixels.
    // (only where the warp finds the scenery itself: what it would fill in by stepping across held pixels comes
    // from elsewhere - while driving, streaks through the windscreen; then the held pixel stays as it is)
    if (!changed && held_here && valid && all(src >= 0) && all(src <= 1) &&
        !ow_held(int3(min(uint2(src * float2(out_size)), out_size - 1), 0))) {
        c = scene;
        changed = true;
        if (flags & 512) c = float4(0, 1, 1, 1);
    }
    // (debug, flag 512: green claimed, blue previous frame, red still scenery stretched, magenta none found (the
    // object carried along), cyan the scene where the character/weapon moved away, yellow a claim lost to
    // nearer scenery, grey only marked)
    if ((flags & 512) && claimed) c = float4(0, 1, 0, 1);
    if ((flags & 512) && !changed) { c = key > 1 ? float4(1, 1, 0, 1) : float4(0.4, 0.4, 0.4, 1); changed = true; }
    if (!changed) return;
    if (flags & 1) {
        const float4 ui = ow_ui_t.Load(int3(id.xy, 0));
        c.rgb = c.rgb * (1.0 - ui.a) + ui.rgb;
    }
    ow_out_u[id.xy] = float4(c.rgb, 1);
}

// Background memory (option): the scenery around the view as last seen - behind held pixels (weapon,
// character, HUD) and up to kMemMargin beyond the frame's edges - at half resolution. Once per game
// frame: what is visible and not held now is written in; everything else is carried over from the
// previous memory, moved with the camera at its remembered depth (a short fixed-point search, as the
// warp does), one frame older; after kMemForget frames it is forgotten. rgb: colour, a: game frames since
// seen (255: nothing); a second texture holds the depth. flags: 1 previous memory valid, 16 depth inverted;
// grid: memory size; rect: the depth region; clip_to_prev: this frame's clip -> the previous frame's.
Texture2D<float4> mem_frame_t : register(t0);
Texture2D<float> mem_mask_t : register(t1);
Texture2D<float> mem_depth_t : register(t2);
Texture2D<float4> mem_prev_t : register(t3);
Texture2D<float> mem_prevz_t : register(t4);
RWTexture2D<float4> mem_u : register(u0);
RWTexture2D<float> memz_u : register(u1);
[numthreads(8, 8, 1)] void cs_memory(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    const float2 s = (float2(id.xy) + 0.5) / float2(grid) * (1.0 + 2.0 * kMemMargin) - kMemMargin;
    if (all(s >= 0) && all(s <= 1) && mem_mask_t.Load(int3(min(uint2(s * float2(out_size)), out_size - 1), 0)) <= 0.5) {
        mem_u[id.xy] = float4(mem_frame_t.SampleLevel(ow_linear, s, 0).rgb, 0);
        memz_u[id.xy] = mem_depth_t.Load(int3(rect.xy + min(uint2(s * float2(rect.zw)), rect.zw - 1), 0));
        return;
    }
    const float far = (flags & 16) ? 0.0 : 1.0;
    float4 v = float4(0, 0, 0, 255);
    float z = far;
    if (flags & 1) {
        float d = far;
        bool ok = true;
        uint2 pi = uint2(0, 0);
        [loop] for (int it = 0; it < 3 && ok; ++it) {
            const float4 p = mul(float4(s.x * 2 - 1, 1 - s.y * 2, d, 1), clip_to_prev);
            ok = p.w > 0;
            const float2 pu = (float2(p.x / p.w * 0.5 + 0.5, 0.5 - p.y / p.w * 0.5) + kMemMargin) / (1.0 + 2.0 * kMemMargin);
            ok = ok && all(pu >= 0) && all(pu < 1);
            pi = min(uint2(saturate(pu) * float2(grid)), grid - 1);
            if (ok) d = mem_prevz_t.Load(int3(pi, 0));
        }
        if (ok) {
            const float4 prev = mem_prev_t.Load(int3(pi, 0));
            if (prev.a + 1.0 < kMemForget) { v = float4(prev.rgb, prev.a + 1.0); z = d; }
        }
    }
    mem_u[id.xy] = v;
    memz_u[id.xy] = z;
}

// XPAR's own motion estimation, for games that give depth but no motion vectors (no DLSS or FSR: depth
// from ReShade). From the picture of this game frame and of the one before: for every pixel of a grid
// half the render grid's size, where that pixel was in the previous frame.
//  - Both pictures as brightness, in a pyramid of 5 levels (each half the one before; all in one texture,
//    level 0 at the origin and the others stacked to its right).
//  - Coarse to fine. At the coarsest level every pixel first tries whole-pixel displacements of its 5x5
//    window (fast turns: up to 12 px there, 384 px of the render grid). Then, level by level, the
//    displacement is refined (Lucas-Kanade, inverse-compositional: the brightness gradients of this frame,
//    one bilinear sample of the previous frame per window pixel, the window sharing one displacement),
//    with a 3x3 median after every level.
//  - A pixel's displacement counts as known (confident) where its window has detail in both directions
//    and the two frames agree there once matched; rain, water, flat sky and moving things do not.
// Motion vectors in uv to the previous frame (as a game's) go to the motion texture; the camera is
// estimated from the confident ones.
// grid: level 0 size; rect.x: level; rect.y: 1 = take the displacement from the coarser level (x2).
static const uint kFlowLevels = 5;
static const int kFlowWindow = 2;        // 5x5
static const float kFlowNoise = 1e-5;    // brightness gradient energy below which a window has no detail
static const float kFlowRatio = 0.8;     // left-over difference allowed per unit of detail
uint2 fl_size(uint k) { return max(grid >> k, uint2(1, 1)); }
uint2 fl_origin(uint k) {
    if (k == 0) return uint2(0, 0);
    uint y = 0;
    for (uint j = 1; j < k; ++j) y += fl_size(j).y;
    return uint2(grid.x, y);
}
float2 fl_atlas() { return float2(grid.x + fl_size(1).x, grid.y); }
// (flags bit 0: the picture is linear HDR - compressed so that dark and bright detail weigh alike)
float fl_tone(float3 c) {
    const float l = dot(max(c, 0), float3(0.299, 0.587, 0.114));
    return (flags & 1) ? sqrt(l / (1.0 + l)) : l;
}
Texture2D<float4> fl_color_t : register(t0);
RWTexture2D<float> fl_luma_u : register(u0);
[numthreads(8, 8, 1)] void cs_flow_luma(uint3 id : SV_DispatchThreadID) {
    uint k = 0;
    uint2 p = id.xy;
    if (id.x >= grid.x) {
        bool found = false;
        for (uint j = 1; j < kFlowLevels && !found; ++j) {
            const uint2 o = fl_origin(j), s = fl_size(j);
            if (all(id.xy >= o) && all(id.xy < o + s)) { k = j; p = id.xy - o; found = true; }
        }
        if (!found) return;
    } else if (id.y >= grid.y) return;
    // The part of the picture this pixel stands for, averaged (bilinear taps two source pixels apart).
    const float2 span = float2(out_size) / float2(grid) * float(1u << k);
    const int n = clamp(int(ceil(max(span.x, span.y) * 0.5)), 1, 32);
    float sum = 0;
    [loop] for (int j = 0; j < n; ++j)
        [loop] for (int i = 0; i < n; ++i)
            sum += fl_tone(fl_color_t.SampleLevel(ow_linear, (float2(p) + (float2(i, j) + 0.5) / float(n)) * span / float2(out_size), 0).rgb);
    fl_luma_u[id.xy] = sum / float(n * n);
}
// Brightness and its gradients (x: brightness, y/z: central differences, 0 on a level's border).
Texture2D<float> fl_luma_t : register(t0);
RWTexture2D<float4> fl_feat_u : register(u0);
[numthreads(8, 8, 1)] void cs_flow_feat(uint3 id : SV_DispatchThreadID) {
    uint k = 0;
    int2 p = int2(id.xy);
    if (id.x >= grid.x) {
        bool found = false;
        for (uint j = 1; j < kFlowLevels && !found; ++j) {
            const uint2 o = fl_origin(j), s = fl_size(j);
            if (all(id.xy >= o) && all(id.xy < o + s)) { k = j; p = int2(id.xy - o); found = true; }
        }
        if (!found) return;
    } else if (id.y >= grid.y) return;
    const int2 o = int2(fl_origin(k)), last = int2(fl_size(k)) - 1;
    const float gx = (p.x > 0 && p.x < last.x) ? 0.5 * (fl_luma_t.Load(int3(o + p + int2(1, 0), 0)) - fl_luma_t.Load(int3(o + p - int2(1, 0), 0))) : 0.0;
    const float gy = (p.y > 0 && p.y < last.y) ? 0.5 * (fl_luma_t.Load(int3(o + p + int2(0, 1), 0)) - fl_luma_t.Load(int3(o + p - int2(0, 1), 0))) : 0.0;
    fl_feat_u[id.xy] = float4(fl_luma_t.Load(int3(o + p, 0)), gx, gy, 0);
}
Texture2D<float4> fl_feat_t : register(t0);   // this frame: brightness and gradients
Texture2D<float> fl_prev_t : register(t1);    // the previous frame's brightness
Texture2D<float4> fl_flow_t : register(t2);   // displacement so far (px of its level), z: confident
RWTexture2D<float4> fl_flow_u : register(u0);
// The previous frame's brightness at a position of level k (pixel centres at +0.5), clamped to the level.
float fl_prev(uint k, float2 pos) {
    const float2 s = float2(fl_size(k));
    return fl_prev_t.SampleLevel(ow_linear, (float2(fl_origin(k)) + clamp(pos, 0.5, s - 0.5)) / fl_atlas(), 0);
}
[numthreads(8, 8, 1)] void cs_flow_search(uint3 id : SV_DispatchThreadID) {
    const uint k = rect.x;
    const int2 size = int2(fl_size(k)), o = int2(fl_origin(k));
    if (any(int2(id.xy) >= size)) return;
    float c[25];
    [unroll] for (int j = 0; j < 25; ++j)
        c[j] = fl_feat_t.Load(int3(o + clamp(int2(id.xy) + int2(j % 5 - 2, j / 5 - 2), int2(0, 0), size - 1), 0)).x;
    const int reach = int(rect.z);
    float best = 1e9;
    int2 shift = int2(0, 0);
    [loop] for (int sy = -reach; sy <= reach; ++sy)
        [loop] for (int sx = -reach; sx <= reach; ++sx) {
            float e = 1e-4 * float(abs(sx) + abs(sy)) * 25.0;  // (equal matches: the smaller displacement)
            [unroll] for (int j = 0; j < 25; ++j) {
                const int2 q = clamp(clamp(int2(id.xy) + int2(j % 5 - 2, j / 5 - 2), int2(0, 0), size - 1) + int2(sx, sy), int2(0, 0), size - 1);
                e += abs(c[j] - fl_prev_t.Load(int3(o + q, 0)));
            }
            if (e < best) { best = e; shift = int2(sx, sy); }
        }
    fl_flow_u[id.xy] = float4(shift, 0, 0);
}
// One refinement of a pixel's displacement; z of the result: confident (written by the last pass, flags bit 1).
[numthreads(8, 8, 1)] void cs_flow_lk(uint3 id : SV_DispatchThreadID) {
    const uint k = rect.x;
    const int2 size = int2(fl_size(k)), o = int2(fl_origin(k));
    if (any(int2(id.xy) >= size)) return;
    float2 u;
    if (rect.y) {
        uint tw, th;
        fl_flow_t.GetDimensions(tw, th);
        const float2 coarse = float2(fl_size(k + 1));
        u = 2.0 * fl_flow_t.SampleLevel(ow_linear, clamp((float2(id.xy) + 0.5) / float2(size) * coarse, 0.5, coarse - 0.5) / float2(tw, th), 0).xy;
    } else {
        u = fl_flow_t.Load(int3(id.xy, 0)).xy;
    }
    float sxx = 0, sxy = 0, syy = 0, bx = 0, by = 0, rr = 0;
    [loop] for (int dy = -kFlowWindow; dy <= kFlowWindow; ++dy)
        [loop] for (int dx = -kFlowWindow; dx <= kFlowWindow; ++dx) {
            const int2 q = clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), size - 1);
            const float3 f = fl_feat_t.Load(int3(o + q, 0)).xyz;
            const float d = f.x - fl_prev(k, float2(q) + 0.5 + u);
            sxx += f.y * f.y; sxy += f.y * f.z; syy += f.z * f.z;
            bx += f.y * d; by += f.z * d; rr += d * d;
        }
    const float n = float((2 * kFlowWindow + 1) * (2 * kFlowWindow + 1));
    sxx /= n; sxy /= n; syy /= n; bx /= n; by /= n; rr /= n;
    if (flags & 2) {
        // Confident: detail in both directions (the smaller eigenvalue of the window's gradient matrix) and
        // little left between the two frames once matched.
        const float tr = sxx + syy;
        const float lmin = 0.5 * (tr - sqrt(max(tr * tr - 4.0 * (sxx * syy - sxy * sxy), 0.0)));
        fl_flow_u[id.xy] = float4(u, (lmin > kFlowNoise && rr < kFlowRatio * lmin + kFlowNoise) ? 1.0 : 0.0, 0);
        return;
    }
    const float lam = 1e-7;
    const float det = (sxx + lam) * (syy + lam) - sxy * sxy;
    float2 du = float2((syy + lam) * bx - sxy * by, (sxx + lam) * by - sxy * bx) / det;
    du *= min(1.0, threshold / max(length(du), 1e-9));  // (at most `threshold` px per refinement)
    fl_flow_u[id.xy] = float4(u + du, 0, 0);
}
[numthreads(8, 8, 1)] void cs_flow_median(uint3 id : SV_DispatchThreadID) {
    const int2 size = int2(fl_size(rect.x));
    if (any(int2(id.xy) >= size)) return;
    float2 v[9];
    [unroll] for (int j = 0; j < 9; ++j) v[j] = fl_flow_t.Load(int3(clamp(int2(id.xy) + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), size - 1), 0)).xy;
    // (a sorting network for the median of 9, per component)
#define FL_SORT(a, b) { const float2 lo = min(v[a], v[b]); v[b] = max(v[a], v[b]); v[a] = lo; }
    FL_SORT(1, 2) FL_SORT(4, 5) FL_SORT(7, 8) FL_SORT(0, 1) FL_SORT(3, 4) FL_SORT(6, 7) FL_SORT(1, 2) FL_SORT(4, 5) FL_SORT(7, 8)
    FL_SORT(0, 3) FL_SORT(5, 8) FL_SORT(4, 7) FL_SORT(3, 6) FL_SORT(1, 4) FL_SORT(2, 5) FL_SORT(4, 7) FL_SORT(4, 2) FL_SORT(6, 4) FL_SORT(4, 2)
#undef FL_SORT
    fl_flow_u[id.xy] = float4(v[4], 0, 0);
}
// Where nothing is known (flat sand, sky, a plain wall) the displacement found is a guess, and the masks
// would take it at its word: ground that seems to stand still while the camera turns counts as attached
// to the camera. Those pixels take the displacement of the known ones around them instead: the known
// displacements are averaged down a pyramid (levels laid out like the brightness pyramid, in the
// displacement textures; z: how much of a pixel is known), and on the way back up every pixel that knows
// nothing takes the coarser level's. Displacements stay in level-0 px at every level.
// During a turn the side of the picture that has just come into view has no previous position at all,
// yet something there always matches something: once the pyramid's top says how the picture moved as a
// whole, a pixel that would have come from outside the previous picture is not known after all (mode 3),
// and the pyramid is built again without those.
// rect.x: the level written; rect.y: 0 average the level above it, 1 copy, 2 own where known, else from
// the coarser level, 3 level 0 without what came from outside (rect.z: the top level).
[numthreads(8, 8, 1)] void cs_flow_fill(uint3 id : SV_DispatchThreadID) {
    const uint k = rect.x;
    const int2 size = int2(fl_size(k)), o = int2(fl_origin(k));
    if (any(int2(id.xy) >= size)) return;
    float4 r = 0;
    if (rect.y == 0) {
        const int2 fs = int2(fl_size(k - 1)), fo = int2(fl_origin(k - 1));
        float3 sum = 0;
        [unroll] for (int j = 0; j < 4; ++j) {
            const float4 v = fl_flow_t.Load(int3(fo + min(int2(id.xy) * 2 + int2(j & 1, j >> 1), fs - 1), 0));
            sum += float3(v.xy * v.z, v.z);
        }
        if (sum.z > 0) r = float4(sum.xy / sum.z, sum.z * 0.25, 0);
    } else if (rect.y == 1) {
        r = fl_flow_t.Load(int3(o + int2(id.xy), 0));
    } else if (rect.y == 3) {
        r = fl_flow_t.Load(int3(id.xy, 0));
        const int2 ts = int2(fl_size(rect.z)), to = int2(fl_origin(rect.z));
        float3 whole = 0;
        [loop] for (int y = 0; y < ts.y; ++y)
            [loop] for (int x = 0; x < ts.x; ++x) {
                const float4 v = fl_flow_t.Load(int3(to + int2(x, y), 0));
                whole += float3(v.xy * v.z, v.z);
            }
        if (whole.z > 0) {
            const float2 from = float2(id.xy) + 0.5 + whole.xy / whole.z;
            if (any(from < 0.0) || any(from > float2(size))) r.z = 0;
        }
    } else {
        r = fl_flow_t.Load(int3(o + int2(id.xy), 0));
        if (r.z <= 0) {
            uint tw, th;
            fl_flow_t.GetDimensions(tw, th);
            const float2 cs = float2(fl_size(k + 1));
            const float2 at = float2(fl_origin(k + 1)) + clamp((float2(id.xy) + 0.5) / float2(size) * cs, 0.5, cs - 0.5);
            r = float4(fl_flow_t.SampleLevel(ow_linear, at / float2(tw, th), 0).xy, 0, 0);
        }
    }
    fl_flow_u[o + int2(id.xy)] = r;
}
// The motion texture (mv_size: its size, the render grid): uv to the previous frame. flags bit 2: no
// previous frame yet - no motion.
RWTexture2D<float2> fl_motion_u : register(u0);
[numthreads(8, 8, 1)] void cs_flow_motion(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= mv_size)) return;
    if (flags & 4) { fl_motion_u[id.xy] = float2(0, 0); return; }
    uint tw, th;
    fl_flow_t.GetDimensions(tw, th);
    const float2 g = float2(grid);
    const float2 u = fl_flow_t.SampleLevel(ow_linear, clamp((float2(id.xy) + 0.5) / float2(mv_size) * g, 0.5, g - 0.5) / float2(tw, th), 0).xy;
    fl_motion_u[id.xy] = u / g;
}

// Debug view on the warped output: masked pixels magenta, HUD score still below the threshold green
// (learned HUD only, flag 1: with the upscaler's output there is nothing being learned).
Texture2D<unorm float> tint_mask_t : register(t0);
Texture2D<float> tint_score_t : register(t1);
RWTexture2D<float4> tint_u : register(u0);
[numthreads(8, 8, 1)] void cs_tint(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float m = tint_mask_t.Load(int3(id.xy, 0)), s = tint_score_t.Load(int3(id.xy, 0));
    const float4 c = tint_u[id.xy];
    if (m > 0.5) tint_u[id.xy] = float4(lerp(c.rgb, float3(1, 0, 1), 0.5), c.a);
    else if ((flags & 1) && s > 0.05) tint_u[id.xy] = float4(lerp(c.rgb, float3(0, 1, 0), 0.5 * saturate(s / 0.6)), c.a);
}
)";

// Descriptor layout in the shader-visible heap.
constexpr UINT kSrcSrv = 0;         // + slot * kTexCount + kind: shared textures
constexpr UINT kPrivMax = 56;       // private textures at most
constexpr UINT kPrivUav = 40;       // + private id
constexpr UINT kPrivSrv = kPrivUav + kPrivMax;  // + private id
// Compute pass tables (6 SRVs t0-t5, 2 UAVs u0-u1 each).
constexpr UINT kXSrvCount = 6;
constexpr UINT kX = kPrivSrv + kPrivMax;  // first table
constexpr UINT kXAnalyzeSrv = kX + 0, kXAnalyzeUav = kX + 6, kXReduceUav = kX + 8;  // (kX + 10 .. 31: free)
constexpr UINT kXHudSrv = kX + 32, kXHudUav = kX + 38, kXMaskSrv = kX + 40, kXMaskUav = kX + 46, kXSampleSrv = kX + 48, kXSampleUav = kX + 54;
constexpr UINT kXTintSrv = kX + 56, kXTintUav = kX + 62, kXSceneSrv = kX + 64, kXSceneUav = kX + 70;
constexpr UINT kXHudlessSrv = kX + 72, kXHudlessUav = kX + 78;
constexpr UINT kXAttSrv = kX + 80, kXAttUav = kX + 86, kXWorldUav = kX + 88, kXFillSrv = kX + 90, kXFillUav = kX + 96;
constexpr UINT kXMemSrv = kX + 98, kXMemUav = kX + 104, kXRowsSrv = kX + 106, kXRowsUav = kX + 112;
constexpr UINT kXRampSrv = kX + 114, kXRampUav = kX + 120;
// Own motion estimation: one table of 6 SRVs + 2 UAVs per pass and direction (see estimate_motion).
constexpr UINT kXFlow = kX + 122;
enum FlowTable : UINT { kFlowLuma, kFlowFeat, kFlowSearch, kFlowMedianAB, kFlowMedianBA, kFlowLkAB, kFlowLkBA, kFlowMotion, kFlowFillAB, kFlowFillBA, kFlowTables };
// Passes on the second root signature (the XPAR warp and the moving objects): 12 SRVs t0-t11 and 2 UAVs
// per table.
constexpr UINT kWSrvCount = 12, kWTableSize = kWSrvCount + 2;
constexpr UINT kW = kXFlow + kFlowTables * 8;
enum WTable : UINT { kWWarp, kWShown, kWClear, kWSplat, kWMove, kWCopyColour, kWCopyDepth, kWJoinInit, kWJoinAB, kWJoinBA, kWJoinFinal,
                     kWGenDepth0, kWGenDepth1, kWGenDepth2, kWGenUi0, kWGenUi1, kWGenUi2, kWTables };
constexpr UINT kWSrv(UINT table) { return kW + table * kWTableSize; }
constexpr UINT kWUav(UINT table) { return kW + table * kWTableSize + kWSrvCount; }
constexpr UINT kHeapSize = kW + kWTables * kWTableSize;
constexpr float kMemMargin = 0.125f;  // as in the shaders
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
    if (queue_ && fences_[0]) wait_idle();
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
    // Development: FW_D3D_DEBUG=1 enables the D3D12 debug layer (errors printed by print_debug_messages).
    if (std::getenv("FW_D3D_DEBUG")) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
    }
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
    // The intake queue (split queues): compute, normal priority - the realtime queue's warp goes first.
    // Without it (creation failed) everything stays on the realtime queue.
    D3D12_COMMAND_QUEUE_DESC iqd{};
    iqd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    iqd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    if (FAILED(device_->CreateCommandQueue(&iqd, IID_PPV_ARGS(&iqueue_)))) iqueue_.Reset();
    for (int i = 0; i < kRing; ++i) {
        const auto type = i < 3 ? D3D12_COMMAND_LIST_TYPE_DIRECT : D3D12_COMMAND_LIST_TYPE_COMPUTE;
        if (i >= 3 && !iqueue_) break;
        if (FAILED(device_->CreateCommandAllocator(type, IID_PPV_ARGS(&allocators_[i])))) { error = "allocator"; return false; }
    }
    if (FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators_[0].Get(), nullptr, IID_PPV_ARGS(&lists_[0])))) { error = "command list"; return false; }
    lists_[0]->Close();
    if (iqueue_ && FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocators_[3].Get(), nullptr, IID_PPV_ARGS(&lists_[1])))) iqueue_.Reset();
    if (lists_[1]) lists_[1]->Close();
    list_ = lists_[0];
    for (auto& f : fences_)
        if (FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f)))) { error = "fence"; return false; }
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    static_assert(kSrcSrv + kSlots * kTexCount <= kPrivUav && kPrivUav + kPCount <= kPrivSrv && kPrivSrv + kPCount <= kX,
                  "descriptor ranges overlap");
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kHeapSize, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_)))) { error = "descriptor heap"; return false; }
    D3D12_DESCRIPTOR_HEAP_DESC rd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(device_->CreateDescriptorHeap(&rd, IID_PPV_ARGS(&rtv_heap_)))) { error = "rtv heap"; return false; }
    descriptor_size_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    rtv_size_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    swap_format_ = swapchain_format(game_format);
    if (!create_pipelines(error)) return false;

    D3D12_QUERY_HEAP_DESC qh{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2 * kRing, 0};
    device_->CreateQueryHeap(&qh, IID_PPV_ARGS(&timestamps_));
    D3D12_HEAP_PROPERTIES rb{}; rb.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = 2 * kRing * 8; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_));
    queue_->GetTimestampFrequency(&timestamp_frequency_);
    timestamp_frequencies_[0] = timestamp_frequency_;
    if (!iqueue_ || FAILED(iqueue_->GetTimestampFrequency(&timestamp_frequencies_[1]))) timestamp_frequencies_[1] = timestamp_frequency_;
    queue_->GetClockCalibration(&calib_gpu_, &calib_cpu_);
    D3D12_QUERY_HEAP_DESC stq{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 3 * kRing, 0};
    device_->CreateQueryHeap(&stq, IID_PPV_ARGS(&stage_stamps_));
    bd.Width = 3 * kRing * 8;
    device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&stage_readback_));
    // GPU time of each scene-HUD pass (one set per frame in flight), for the log.
    D3D12_QUERY_HEAP_DESC sq{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, kRing * kSceneStamps, 0};
    device_->CreateQueryHeap(&sq, IID_PPV_ARGS(&scene_stamps_));
    bd.Width = kRing * kSceneStamps * 8;
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
    D3D12_STATIC_SAMPLER_DESC linear{};
    linear.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    linear.AddressU = linear.AddressV = linear.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    linear.MaxLOD = D3D12_FLOAT32_MAX;
    linear.ShaderRegister = 0;
    linear.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC xrs{3, xp, 1, &linear, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    blob.Reset(); err.Reset();
    if (FAILED(D3D12SerializeRootSignature(&xrs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
        FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_x_)))) {
        error = "extrapolation root signature"; return false;
    }
    struct { const char* entry; ComPtr<ID3D12PipelineState>* pso; } xs[] = {
        {"cs_analyze", &cs_analyze_}, {"cs_reduce", &cs_reduce_},
        {"cs_hud", &cs_hud_}, {"cs_hud_world", &cs_hud_world_}, {"cs_mask", &cs_mask_}, {"cs_clear_score", &cs_clear_score_}, {"cs_hudless_clear", &cs_hudless_clear_}, {"cs_hudless_count", &cs_hudless_count_},
        {"cs_hudless_score", &cs_hudless_score_}, {"cs_hudless_fill", &cs_hudless_fill_},
        {"cs_hud_count", &cs_hud_count_}, {"cs_clear_counts", &cs_clear_counts_}, {"cs_sample", &cs_sample_}, {"cs_tint", &cs_tint_},
        {"cs_scene_clear", &cs_scene_clear_}, {"cs_scene_accum", &cs_scene_accum_}, {"cs_scene_finish", &cs_scene_finish_},
        {"cs_scene_tiles", &cs_scene_tiles_}, {"cs_scene_hud", &cs_scene_hud_}, {"cs_scene_fill", &cs_scene_fill_}, {"cs_scene_grey", &cs_scene_grey_},
        {"cs_scene_grey_finish", &cs_scene_grey_finish_}, {"cs_scene_wash", &cs_scene_wash_}, {"cs_scene_wash_finish", &cs_scene_wash_finish_},
        {"cs_attached", &cs_attached_}, {"cs_memory", &cs_memory_}, {"cs_ramp_rows", &cs_ramp_rows_}, {"cs_ramp", &cs_ramp_},
        {"cs_flow_luma", &cs_flow_luma_}, {"cs_flow_feat", &cs_flow_feat_}, {"cs_flow_search", &cs_flow_search_}, {"cs_flow_lk", &cs_flow_lk_},
        {"cs_flow_median", &cs_flow_median_}, {"cs_flow_motion", &cs_flow_motion_}, {"cs_flow_fill", &cs_flow_fill_}};
    for (auto& x : xs) {
        auto code = compile(x.entry, "cs_5_0", error, kShadersX);
        if (!code) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC xd{};
        xd.pRootSignature = root_x_.Get();
        xd.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (FAILED(device_->CreateComputePipelineState(&xd, IID_PPV_ARGS(x.pso->ReleaseAndGetAddressOf())))) { error = std::string(x.entry) + " pso"; return false; }
    }
    // The XPAR warp and the moving objects: the same 32 constants (b0), 20 more (b1), 12 SRVs, 2 UAVs.
    D3D12_DESCRIPTOR_RANGE wsrv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kWSrvCount, 0, 0, 0};
    D3D12_ROOT_PARAMETER wp[4]{};
    wp[0] = xp[0];
    wp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; wp[1].Constants = {1, 0, 20};
    wp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; wp[2].DescriptorTable = {1, &wsrv};
    wp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; wp[3].DescriptorTable = {1, &xuav};
    D3D12_ROOT_SIGNATURE_DESC wrs{4, wp, 1, &linear, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    blob.Reset(); err.Reset();
    if (FAILED(D3D12SerializeRootSignature(&wrs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
        FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_w_)))) {
        error = "warp root signature"; return false;
    }
    struct { const char* entry; ComPtr<ID3D12PipelineState>* pso; } ws[] = {
        {"cs_own_warp", &cs_own_warp_}, {"cs_obj_fix", &cs_obj_fix_}, {"cs_obj_move", &cs_obj_move_}, {"cs_gen_depth", &cs_gen_depth_}, {"cs_gen_unui", &cs_gen_unui_}, {"cs_obj_join_init", &cs_obj_join_init_}, {"cs_obj_join", &cs_obj_join_}, {"cs_obj_join_final", &cs_obj_join_final_}, {"cs_obj_clear", &cs_obj_clear_}, {"cs_obj_clear_tiles", &cs_obj_clear_tiles_}, {"cs_obj_splat", &cs_obj_splat_},
        {"cs_copy4", &cs_copy4_}, {"cs_copy1", &cs_copy1_}};
    for (auto& x : ws) {
        auto code = compile(x.entry, "cs_5_0", error, kShadersX);
        if (!code) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC xd{};
        xd.pRootSignature = root_w_.Get();
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
    bd.Width = kRing * 2 * 16; bd.Flags = D3D12_RESOURCE_FLAG_NONE;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&fit_readback_)))) {
        error = "motion fit readback"; return false;
    }
    bd.Width = kRing * 16;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&hud_readback_)))) {
        error = "HUD counter readback"; return false;
    }
    return true;
}

void Renderer::set_x_srv(UINT index, PrivateId id, bool from_shown) {
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    const auto& p = from_shown ? shown(id) : private_[id];
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

void Renderer::w_dispatch(ID3D12PipelineState* pso, const void* constants, const float w[20], UINT table, UINT groups_x, UINT groups_y) {
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetComputeRootSignature(root_w_.Get());
    list_->SetPipelineState(pso);
    list_->SetComputeRoot32BitConstants(0, 32, constants, 0);
    list_->SetComputeRoot32BitConstants(1, 20, w, 0);
    list_->SetComputeRootDescriptorTable(2, gpu(kWSrv(table)));
    list_->SetComputeRootDescriptorTable(3, gpu(kWUav(table)));
    list_->Dispatch(groups_x, groups_y, 1);
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
                              bool depth_inverted, bool near_rule, bool turn_rule, std::uint64_t frame) {
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
    set_x_srv(kXAnalyzeSrv + 2, own_motion_ ? kPFlowB : kPDepth);  // (XPAR's own motion: which of it is trusted)
    for (UINT i = 3; i < kXSrvCount; ++i) set_x_srv(kXAnalyzeSrv + i, kPDepth);
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
    c.flags = (scale_valid ? 1u : 0u) | (depth_inverted ? 16u : 0u) | (near_rule ? 0u : 64u) | (turn_rule ? 128u : 0u);
    if (own_motion_) { c.flags |= 2u; c.out_size[0] = std::max(1u, w / 2); c.out_size[1] = std::max(1u, h / 2); }  // (level 0 of the displacements)
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
    fit_frame_[frame_index_] = frame;
}

// The projection's depth terms (row vectors: clip z = vz a + b, clip w = vz c + e) for view_w in the shaders.
static void projection_terms(const float view_to_clip[16], float out[4]) {
    out[0] = view_to_clip[10]; out[1] = view_to_clip[14]; out[2] = view_to_clip[11]; out[3] = view_to_clip[15];
}

bool Renderer::has_generated() const {
    for (int id = kPGen0; id <= kPGenZ2; ++id)
        if (private_[id].texture || front_[id].texture) return true;
    for (int slot = 0; slot < kSlots; ++slot)
        for (int i = 0; i < kMaxGenerated; ++i) if (shared_[slot][kGen0 + i].resource) return true;
    return false;
}

void Renderer::release_generated() {
    wait_idle();
    for (int id = kPGen0; id <= kPGenZ2; ++id) private_[id] = Private{}, front_[id] = Private{};
    for (int slot = 0; slot < kSlots; ++slot)
        for (int i = 0; i < kMaxGenerated; ++i) {
            shared_[slot][kGen0 + i] = SharedTex{};
            srv_resource_[kSrcSrv + slot * kTexCount + kGen0 + i] = nullptr;
        }
}

void Renderer::prepare_generated(const IngestedSource& src, float scale_x, float scale_y, bool scale_valid, bool use_ui_tags) {
    if (!src.generated || !src.has_depth || !private_[kPDepth].texture) return;
    const auto& depth = private_[kPDepth];
    const UINT w = depth.width, h = depth.height;
    const bool motion = src.has_motion && !src.motion_estimated && private_[kPMotion].texture;
    const bool unui = use_ui_tags && src.has_hudless && src.has_ui && private_[kPHudless].texture && private_[kPUi].texture;
    XConstants c{};
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    c.grid[0] = w; c.grid[1] = h;
    c.mv_size[0] = motion ? private_[kPMotion].width : 1; c.mv_size[1] = motion ? private_[kPMotion].height : 1;
    c.mv_scale[0] = scale_x; c.mv_scale[1] = scale_y;
    c.flags = motion && scale_valid ? 1u : 0u;
    float wc[20] = {};
    for (int i = 0; i < src.generated; ++i) {
        const auto z = static_cast<PrivateId>(kPGenZ0 + i), colour = static_cast<PrivateId>(kPGen0 + i);
        if (!ensure_private(z, w, h, DXGI_FORMAT_R32_FLOAT) || !private_[colour].texture) return;
        c.alpha = float(src.gen_index[i]) / float(src.per_frame + 1);
        const UINT d = kWGenDepth0 + i;
        for (UINT k = 0; k < kWSrvCount; ++k) set_x_srv(kWSrv(d) + k, kPDepth);
        if (motion) set_x_srv(kWSrv(d) + 1, kPMotion);
        set_x_uav(kWUav(d) + 0, z); set_x_uav(kWUav(d) + 1, z);
        transition(private_[z], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        w_dispatch(cs_gen_depth_.Get(), &c, wc, d, (w + 7) / 8, (h + 7) / 8);
        transition(private_[z], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (unui && private_[kPUi].width == private_[colour].width && private_[kPUi].height == private_[colour].height) {
            XConstants u = c;
            u.out_size[0] = private_[colour].width; u.out_size[1] = private_[colour].height;
            const UINT t = kWGenUi0 + i;
            for (UINT k = 0; k < kWSrvCount; ++k) set_x_srv(kWSrv(t) + k, kPUi);
            set_x_srv(kWSrv(t) + 1, kPHudless);
            set_x_uav(kWUav(t) + 0, colour); set_x_uav(kWUav(t) + 1, colour);
            transition(private_[colour], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            w_dispatch(cs_gen_unui_.Get(), &u, wc, t, (u.out_size[0] + 7) / 8, (u.out_size[1] + 7) / 8);
            transition(private_[colour], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
    }
}

bool Renderer::object_motion(const IngestedSource& src, const float clip_to_prev_clip[16], const float view_to_clip[16], float scale_x,
                             float scale_y, bool held, bool picture) {
    objects_built_ = false;
    const auto& depth = private_[kPDepth];
    if (!src.has_depth || !src.has_motion || !history_valid_ || !private_[kPObject].texture || !depth.texture) return false;
    const UINT w = depth.width, h = depth.height;
    if (private_[kPPrevDepth].width != w || private_[kPPrevDepth].height != h || private_[kPObject].width != w || private_[kPObject].height != h)
        return false;
    Mat4 m{}, inv{};
    std::memcpy(m.data(), clip_to_prev_clip, sizeof(float) * 16);
    if (!mat_inverse(m, inv) || !ensure_private(kPObjMove, w, h, DXGI_FORMAT_R32G32B32A32_FLOAT) ||
        !ensure_private(kPObjTiles, (w + 7) / 8, (h + 7) / 8, DXGI_FORMAT_R32_UINT))
        return false;
    XConstants c{};
    std::memcpy(c.clip_to_prev, inv.data(), sizeof(c.clip_to_prev));
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    c.grid[0] = w; c.grid[1] = h;
    c.mv_size[0] = private_[kPMotion].width; c.mv_size[1] = private_[kPMotion].height;
    c.mv_scale[0] = scale_x; c.mv_scale[1] = scale_y;
    const bool flow = picture && private_[kPFlowB].texture;
    c.flags = (held ? 1u : 0u) | (flow ? 2u : 0u);
    c.out_size[0] = std::max(1u, w / 2); c.out_size[1] = std::max(1u, h / 2);  // (level 0 of the picture's motion)
    float wc[20] = {};
    projection_terms(view_to_clip, wc + 16);
    if (held && ensure_private(kPObjKindA, w, h, DXGI_FORMAT_R16_FLOAT) && ensure_private(kPObjKindB, w, h, DXGI_FORMAT_R16_FLOAT)) {
        // One object, one way of moving (cs_obj_join): written back into the analysis before the mask is built.
        XConstants j{};
        j.grid[0] = w; j.grid[1] = h;
        auto table = [&](UINT t, PrivateId kind_in, PrivateId out) {
            for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(t) + i, kPDepth);
            set_x_srv(kWSrv(t) + 0, kPObject);
            set_x_srv(kWSrv(t) + 1, kind_in);
            set_x_srv(kWSrv(t) + 2, kPDepth);
            set_x_uav(kWUav(t) + 0, out); set_x_uav(kWUav(t) + 1, out);
        };
        auto barrier = [&](PrivateId id) {
            D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource = private_[id].texture.Get();
            list_->ResourceBarrier(1, &b);
        };
        table(kWJoinInit, kPObjKindB, kPObjKindA);
        table(kWJoinAB, kPObjKindA, kPObjKindB);
        table(kWJoinBA, kPObjKindB, kPObjKindA);
        transition(private_[kPObjKindA], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        w_dispatch(cs_obj_join_init_.Get(), &j, wc, kWJoinInit, (w + 7) / 8, (h + 7) / 8);
        bool in_a = true;
        for (const UINT step : {64u, 32u, 16u, 8u, 4u, 2u, 1u}) {
            const PrivateId from = in_a ? kPObjKindA : kPObjKindB, to = in_a ? kPObjKindB : kPObjKindA;
            transition(private_[from], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(private_[to], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            barrier(to);
            j.rect[0] = step;
            w_dispatch(cs_obj_join_.Get(), &j, wc, in_a ? kWJoinAB : kWJoinBA, (w + 7) / 8, (h + 7) / 8);
            in_a = !in_a;
        }
        const PrivateId result = in_a ? kPObjKindA : kPObjKindB;
        transition(private_[result], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        table(kWJoinFinal, result, kPObject);
        transition(private_[kPObject], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        w_dispatch(cs_obj_join_final_.Get(), &j, wc, kWJoinFinal, (w + 7) / 8, (h + 7) / 8);
        transition(private_[kPObject], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    set_x_srv(kWSrv(kWMove) + 0, kPDepth);
    set_x_srv(kWSrv(kWMove) + 1, kPMotion);
    set_x_srv(kWSrv(kWMove) + 2, kPObject);
    set_x_srv(kWSrv(kWMove) + 3, kPPrevDepth);
    set_x_srv(kWSrv(kWMove) + 4, flow ? kPFlowB : kPDepth);
    for (UINT i = 5; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWMove) + i, kPDepth);
    set_x_uav(kWUav(kWMove) + 0, kPObjMove); set_x_uav(kWUav(kWMove) + 1, kPObjTiles);
    transition(private_[kPObjMove], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(private_[kPObjTiles], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_obj_move_.Get(), &c, wc, kWMove, (w + 7) / 8, (h + 7) / 8);
    transition(private_[kPObjMove], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(private_[kPObjTiles], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    objects_built_ = true;
    if (!split_) shown_objects_ = true;
    return true;
}

// Moving objects: the previous game frame's picture and depth, copied before this frame's replace them
// (with split queues from the shown set, read in place; the copies go to the set this frame fills). Only
// when this frame has the same sizes: otherwise the previous frame is of no use, and this frame's intake
// replaces the textures the copies would read before the GPU gets to them.
void Renderer::keep_history(PrivateId colour, std::uint32_t depth_w, std::uint32_t depth_h, std::uint32_t colour_w, std::uint32_t colour_h,
                            DXGI_FORMAT colour_format) {
    history_valid_ = false;
    const Private& d = shown(kPDepth);
    const Private& c = shown(colour);
    if (!object_history_ || !ingested_ || !d.texture || !c.texture || d.width != depth_w || d.height != depth_h || c.width != colour_w ||
        c.height != colour_h || c.format != colour_format)
        return;
    auto copy = [&](PrivateId from_id, PrivateId to, ID3D12PipelineState* pso, UINT table) {
        const Private& from = shown(from_id);
        if (!ensure_private(to, from.width, from.height, from.format)) return false;
        XConstants k{};
        k.grid[0] = from.width; k.grid[1] = from.height;
        const float wc[20] = {};
        for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(table) + i, from_id, true);
        set_x_uav(kWUav(table) + 0, to); set_x_uav(kWUav(table) + 1, to);
        transition(private_[to], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        w_dispatch(pso, &k, wc, table, (from.width + 7) / 8, (from.height + 7) / 8);
        transition(private_[to], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        return true;
    };
    history_valid_ = copy(kPDepth, kPPrevDepth, cs_copy1_.Get(), kWCopyDepth) && copy(colour, kPPrevColour, cs_copy4_.Get(), kWCopyColour);
}

ID3D12Resource* Renderer::build_no_warp_mask(const IngestedSource& src, const float clip_to_prev_clip[16], bool hud, bool attached,
                                             bool keep_attached, bool depth_inverted, bool combined, bool fill, int stretch) {
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
    const bool reset = new_score || reset_hud_;
    if (reset) {
        // New or resized score: start from "not HUD".
        reset_hud_ = false;
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        set_x_uav(kXHudUav + 0, kPHudScore); set_x_uav(kXHudUav + 1, kPHudScore);
        XConstants z = c; z.flags = 0;
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXHudSrv + i, kPBackbuffer);
        x_dispatch(cs_clear_score_.Get(), &z, kXHudSrv, kXHudUav, (ow + 7) / 8, (oh + 7) / 8);
    }
    fill_ready_ = false;
    // A HUD-less picture without a UI layer: the HUD is where the frame differs from it (see cs_hudless_*).
    const bool from_hudless = hud && src.has_hudless && !src.has_ui && private_[kPHudless].texture && private_[kPHudless].width == ow &&
                              private_[kPHudless].height == oh;
    const bool from_scene = hud && !from_hudless && src.has_scene && private_[kPScene].texture && private_[kPScene].width == ow &&
                            private_[kPScene].height == oh;
    if (from_hudless) {
        hud_from_scene_ = true;  // (the learned map starts over when this stops)
        XConstants h = c;
        h.threshold = 3.0f / 255.0f;
        set_x_srv(kXHudlessSrv + 0, kPBackbuffer);
        set_x_srv(kXHudlessSrv + 1, kPHudless);
        for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXHudlessSrv + i, kPHudless);
        set_x_uav(kXHudlessUav + 0, kPHudScore);
        D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
        raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
        raw.Buffer.NumElements = 4; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(hud_counts_.Get(), nullptr, &raw, cpu(kXHudlessUav + 1));
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_RESOURCE_BARRIER counts{}; counts.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; counts.UAV.pResource = hud_counts_.Get();
        x_dispatch(cs_hudless_clear_.Get(), &h, kXHudlessSrv, kXHudlessUav, 1, 1);
        list_->ResourceBarrier(1, &counts);
        x_dispatch(cs_hudless_count_.Get(), &h, kXHudlessSrv, kXHudlessUav, (ow + 7) / 8, (oh + 7) / 8);
        list_->ResourceBarrier(1, &counts);
        x_dispatch(cs_hudless_score_.Get(), &h, kXHudlessSrv, kXHudlessUav, (ow + 7) / 8, (oh + 7) / 8);
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const DXGI_FORMAT fill_format = private_[kPBackbuffer].format == DXGI_FORMAT_R11G11B10_FLOAT ? DXGI_FORMAT_R16G16B16A16_FLOAT
                                                                                                    : private_[kPBackbuffer].format;
        if (fill && ensure_private(kPFill, ow, oh, fill_format)) {
            set_x_srv(kXFillSrv + 0, kPHudless);
            set_x_srv(kXFillSrv + 1, kPHudScore);
            for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXFillSrv + i, kPHudless);
            set_x_uav(kXFillUav + 0, kPFill); set_x_uav(kXFillUav + 1, kPFill);
            transition(private_[kPFill], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            x_dispatch(cs_hudless_fill_.Get(), &h, kXFillSrv, kXFillUav, (ow + 7) / 8, (oh + 7) / 8);
            transition(private_[kPFill], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            fill_ready_ = true;
        }
    } else if (from_scene) {
        hud_from_scene_ = true;
        detect_hud_from_scene(c, fill);
    } else if (hud_from_scene_) {
        // The upscaler's output stopped coming: the learned map starts over.
        hud_from_scene_ = false;
        reset_hud_ = true;
    }
    // Combined detection: where the pixels follow the world (world evidence, cs_hud_world), from the
    // previous colour. Its own UAV table: the score's may already be in use for this frame's clear.
    bool world = false;
    if (from_scene && combined && previous) {
        const bool new_world = !private_[kPWorld].texture || private_[kPWorld].width != ow || private_[kPWorld].height != oh;
        if (ensure_private(kPWorld, ow, oh, DXGI_FORMAT_R32_FLOAT)) {
            world = true;
            set_x_srv(kXHudSrv + 0, kPBackbuffer);
            set_x_srv(kXHudSrv + 1, kPPrevious);
            set_x_srv(kXHudSrv + 2, kPDepth);
            set_x_srv(kXHudSrv + 3, kPObject);
            set_x_srv(kXHudSrv + 4, kPHudScore);
            for (UINT i = 5; i < kXSrvCount; ++i) set_x_srv(kXHudSrv + i, kPDepth);
            transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            set_x_uav(kXWorldUav + 0, kPWorld);
            D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
            raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
            raw.Buffer.NumElements = 4; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            device_->CreateUnorderedAccessView(hud_counts_.Get(), nullptr, &raw, cpu(kXWorldUav + 1));
            transition(private_[kPWorld], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = private_[kPWorld].texture.Get();
            if (new_world || reset) {
                XConstants z = c; z.flags = 0;
                x_dispatch(cs_clear_score_.Get(), &z, kXHudSrv, kXWorldUav, (ow + 7) / 8, (oh + 7) / 8);
                list_->ResourceBarrier(1, &uav);
            }
            c.flags = 2u | (depth_inverted ? 16u : 0u) | (attached ? 32u : 0u);
            D3D12_RESOURCE_BARRIER counts{}; counts.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; counts.UAV.pResource = hud_counts_.Get();
            x_dispatch(cs_clear_counts_.Get(), &c, kXHudSrv, kXWorldUav, 1, 1);
            list_->ResourceBarrier(1, &counts);
            x_dispatch(cs_hud_count_.Get(), &c, kXHudSrv, kXWorldUav, (ow + 7) / 8, (oh + 7) / 8);
            list_->ResourceBarrier(1, &counts);
            x_dispatch(cs_hud_world_.Get(), &c, kXHudSrv, kXWorldUav, (ow + 7) / 8, (oh + 7) / 8);
            transition(private_[kPWorld], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
    }
    if (hud && previous && !from_scene && !from_hudless) {
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
    // Camera-attached pixels, widened once at render resolution (the mask reads one value per pixel).
    const bool keep_att = attached && keep_attached && ensure_private(kPAttached, w, h, DXGI_FORMAT_R8_UNORM);
    if (keep_att) {
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXAttSrv + i, kPObject);
        set_x_uav(kXAttUav + 0, kPAttached); set_x_uav(kXAttUav + 1, kPAttached);
        transition(private_[kPAttached], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        x_dispatch(cs_attached_.Get(), &c, kXAttSrv, kXAttUav, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
        transition(private_[kPAttached], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    // The stretch around them (option): the distance share, in two passes.
    const bool ramp = keep_att && stretch > 0 && ensure_private(kPRampRows, w, h, DXGI_FORMAT_R8_UNORM) &&
                      ensure_private(kPRamp, w, h, DXGI_FORMAT_R8_UNORM);
    if (ramp) {
        XConstants a = c;
        a.mv_size[0] = static_cast<std::uint32_t>(std::clamp(stretch, 1, 32));
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXRowsSrv + i, kPAttached);
        set_x_uav(kXRowsUav + 0, kPRampRows); set_x_uav(kXRowsUav + 1, kPRampRows);
        transition(private_[kPRampRows], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        x_dispatch(cs_ramp_rows_.Get(), &a, kXRowsSrv, kXRowsUav, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
        transition(private_[kPRampRows], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXRampSrv + i, kPRampRows);
        set_x_uav(kXRampUav + 0, kPRamp); set_x_uav(kXRampUav + 1, kPRamp);
        transition(private_[kPRamp], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        x_dispatch(cs_ramp_.Get(), &a, kXRampSrv, kXRampUav, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
        transition(private_[kPRamp], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    if (!ensure_private(kPWarpMask, ow, oh, DXGI_FORMAT_R8_UNORM)) return nullptr;
    set_x_srv(kXMaskSrv + 0, kPHudScore);
    set_x_srv(kXMaskSrv + 1, keep_att ? kPAttached : kPHudScore);
    set_x_srv(kXMaskSrv + 2, world ? kPWorld : kPHudScore);
    set_x_srv(kXMaskSrv + 3, ramp ? kPRamp : kPHudScore);
    for (UINT i = 4; i < kXSrvCount; ++i) set_x_srv(kXMaskSrv + i, kPHudScore);
    set_x_uav(kXMaskUav + 0, kPMask); set_x_uav(kXMaskUav + 1, kPWarpMask);
    c.flags = (hud ? 4u : 0u) | (keep_att ? 8u : 0u) | (world ? 64u : 0u) | (ramp ? 128u : 0u);
    stretch_built_ = ramp;
    transition(private_[kPMask], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(private_[kPWarpMask], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_mask_.Get(), &c, kXMaskSrv, kXMaskUav, (ow + 7) / 8, (oh + 7) / 8);
    transition(private_[kPMask], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(private_[kPWarpMask], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    mask_ready_ = true;
    built_mask_ = true;
    return private_[kPMask].texture.Get();
}

void Renderer::detect_hud_from_scene(const XConstants& base, bool fill) {
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
    // (alpha marks the filled pixels: the game's format when it has one, half-float otherwise)
    const DXGI_FORMAT fill_format = private_[kPBackbuffer].format == DXGI_FORMAT_R11G11B10_FLOAT ? DXGI_FORMAT_R16G16B16A16_FLOAT
                                                                                                : private_[kPBackbuffer].format;
    if (fill && ensure_private(kPFill, ow, oh, fill_format)) {
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        set_x_srv(kXFillSrv + 0, kPBackbuffer);
        set_x_srv(kXFillSrv + 1, kPScene);
        set_x_srv(kXFillSrv + 2, kPHudScore);
        for (UINT i = 3; i < kXSrvCount; ++i) set_x_srv(kXFillSrv + i, kPScene);
        set_x_uav(kXFillUav + 0, kPFill);
        device_->CreateUnorderedAccessView(scene_fit_.Get(), nullptr, &raw, cpu(kXFillUav + 1));
        transition(private_[kPFill], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        c.rect[0] = 2; c.rect[1] = 1; c.rect[2] = 0; c.rect[3] = 0;
        x_dispatch(cs_scene_fill_.Get(), &c, kXFillSrv, kXFillUav, (ow + 7) / 8, (oh + 7) / 8);
        transition(private_[kPFill], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        fill_ready_ = true;
    }
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

bool Renderer::own_warp(const IngestedSource& src, bool use_ui_tags, bool use_mask, const float source_to_target[16], bool depth_inverted,
                        bool memory, const ObjectWarp* objects, int generated) {
    // (the shown set with split queues)
    if (!src.valid || !src.has_depth || !private_[kPOutput].texture || !shown(kPDepth).texture) return false;
    const bool split = use_ui_tags && src.has_hudless && src.has_ui && shown(kPHudless).texture && shown(kPUi).texture;
    const bool mask_ready = split_ ? shown_mask_ : mask_ready_;
    const bool mask = use_mask && mask_ready && shown(kPWarpMask).texture && shown(kPWarpMask).width == private_[kPOutput].width &&
                      shown(kPWarpMask).height == private_[kPOutput].height;
    const UINT ow = private_[kPOutput].width, oh = private_[kPOutput].height;
    XConstants c{};
    std::memcpy(c.clip_to_prev, source_to_target, sizeof(c.clip_to_prev));
    c.out_size[0] = ow; c.out_size[1] = oh;
    const UINT dw = shown(kPDepth).width, dh = shown(kPDepth).height;
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, dw, dh};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, dw - r.x); c.rect[3] = std::min(r.h, dh - r.y);
    c.grid[0] = dw; c.grid[1] = dh;
    // Row 2 (the rendered depth) does not reach x, y or w: a pure turn, the same for every depth.
    const float* m = source_to_target;
    const float scale = std::max({std::fabs(m[0]), std::fabs(m[5]), std::fabs(m[12]), std::fabs(m[13]), std::fabs(m[14]), std::fabs(m[15]), 1e-6f});
    // Moving objects: what the shown frame's object pass left, with its previous frame kept, same sizes.
    const bool moving = objects && objects->frames_back > 0 && shown_objects_ && shown(kPObjMove).texture &&
                        shown(kPObjMove).width == dw && shown(kPObjMove).height == dh && shown(kPPrevDepth).width == dw &&
                        shown(kPPrevDepth).height == dh && shown(kPPrevColour).width == ow && shown(kPPrevColour).height == oh &&
                        shown(kPObjTiles).width == (dw + 7) / 8 && shown(kPObjTiles).height == (dh + 7) / 8;
    const bool no_depth = std::fabs(m[8]) <= 1e-6f * scale && std::fabs(m[9]) <= 1e-6f * scale && std::fabs(m[11]) <= 1e-6f * scale;
    const bool fill_ready = split_ ? shown_fill_ : fill_ready_;
    const bool fill = mask && !split && fill_ready && shown(kPFill).width == ow && shown(kPFill).height == oh;
    const bool remembered = memory && mask && mem_shown_ >= 0 && private_[kPMem0 + mem_shown_].texture;
    const bool stretched = mask && (split_ ? shown_stretch_ : stretch_built_);
    c.flags = (split ? 1u : 0u) | (mask ? 2u : 0u) | (fill ? 4u : 0u) | (depth_inverted ? 16u : 0u) | (no_depth ? 32u : 0u) |
              (remembered ? 64u : 0u) | (stretched ? 128u : 0u) | (moving ? 256u : 0u);
    float wc[20] = {};
    const UINT t = kWSrv(kWWarp);
    // Frame generation: one of the game's generated images, with its own depth (prepare_generated).
    const bool gen = generated >= 0 && generated < src.generated && shown(static_cast<PrivateId>(kPGen0 + generated)).width == ow &&
                     shown(static_cast<PrivateId>(kPGenZ0 + generated)).width == dw;
    set_x_srv(t + 0, gen ? static_cast<PrivateId>(kPGen0 + generated) : split ? kPHudless : kPBackbuffer, true);
    set_x_srv(t + 1, split ? kPUi : kPBackbuffer, true);  // (only read with a UI layer)
    set_x_srv(t + 2, mask ? kPWarpMask : kPDepth, true);
    set_x_srv(t + 3, gen ? static_cast<PrivateId>(kPGenZ0 + generated) : kPDepth, true);
    if (remembered) set_x_srv(t + 4, static_cast<PrivateId>(kPMem0 + mem_shown_));
    else set_x_srv(t + 4, kPDepth, true);
    set_x_srv(t + 5, fill ? kPFill : kPBackbuffer, true);
    if (moving) {
        // The moving pixels' claims on the output pixels first: the blocks the previous refresh touched are
        // cleared (all of it for new textures), then the nearest claim wins.
        const UINT tw = (ow + 7) / 8, th = (oh + 7) / 8;
        ID3D12Resource* const keys_before = private_[kPObjKeys].texture.Get();
        ID3D12Resource* const tiles_before = private_[kPObjOutTiles].texture.Get();
        if (!ensure_private(kPObjKeys, ow, oh, DXGI_FORMAT_R32_UINT) || !ensure_private(kPObjOutTiles, tw, th, DXGI_FORMAT_R32_UINT)) return false;
        const bool fresh = private_[kPObjKeys].texture.Get() != keys_before || private_[kPObjOutTiles].texture.Get() != tiles_before;
        c.alpha = std::clamp(objects->frames_back, 0.0f, 1.0f);
        std::memcpy(wc, objects->prev_to_target, sizeof(float) * 16);
        projection_terms(objects->view_to_clip, wc + 16);
        for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWClear) + i, kPDepth, true);
        set_x_uav(kWUav(kWClear) + 0, kPObjKeys); set_x_uav(kWUav(kWClear) + 1, kPObjOutTiles);
        transition(private_[kPObjKeys], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(private_[kPObjOutTiles], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (fresh) w_dispatch(cs_obj_clear_.Get(), &c, wc, kWClear, (ow + 7) / 8, (oh + 7) / 8);
        else w_dispatch(cs_obj_clear_tiles_.Get(), &c, wc, kWClear, (tw + 7) / 8, (th + 7) / 8);
        D3D12_RESOURCE_BARRIER uav[2]{};
        uav[0].Type = uav[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav[0].UAV.pResource = private_[kPObjKeys].texture.Get(); uav[1].UAV.pResource = private_[kPObjOutTiles].texture.Get();
        list_->ResourceBarrier(2, uav);
        const UINT st = kWSrv(kWSplat);
        for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(st + i, kPDepth, true);
        set_x_srv(st + 2, mask ? kPWarpMask : kPDepth, true);
        set_x_srv(st + 6, kPObjMove, true);
        set_x_srv(st + 11, kPObjTiles, true);
        set_x_uav(kWUav(kWSplat) + 0, kPObjKeys); set_x_uav(kWUav(kWSplat) + 1, kPObjOutTiles);
        w_dispatch(cs_obj_splat_.Get(), &c, wc, kWSplat, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
        transition(private_[kPObjKeys], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(private_[kPObjOutTiles], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        set_x_srv(t + 6, kPObjMove, true);
        set_x_srv(t + 7, kPObjKeys);
        set_x_srv(t + 8, kPPrevColour, true);
        set_x_srv(t + 9, kPPrevDepth, true);
        set_x_srv(t + 10, kPObjOutTiles);
        set_x_srv(t + 11, kPDepth, true);
    } else {
        for (UINT i = 6; i < kWSrvCount; ++i) set_x_srv(t + i, kPDepth, true);
    }
    set_x_uav(kWUav(kWWarp) + 0, kPOutput); set_x_uav(kWUav(kWWarp) + 1, kPOutput);
    transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_own_warp_.Get(), &c, wc, kWWarp, (ow + 7) / 8, (oh + 7) / 8);
    if (moving) {
        // The pixels moving objects claimed or left, looked at again.
        D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = private_[kPOutput].texture.Get();
        list_->ResourceBarrier(1, &uav);
        XConstants fix = c;
        if (objects->debug) fix.flags |= 512;
        w_dispatch(cs_obj_fix_.Get(), &fix, wc, kWWarp, (ow + 7) / 8, (oh + 7) / 8);
    }
    return true;
}

void Renderer::tint_mask() {
    // (with split queues the HUD score belongs to the intake queue: magenta only)
    if (!(split_ ? shown_mask_ : mask_ready_) || !private_[kPOutput].texture || !private_[kPHudScore].texture) return;
    const UINT ow = private_[kPOutput].width, oh = private_[kPOutput].height;
    if (shown(kPMask).width != ow || shown(kPMask).height != oh) return;
    XConstants c{};
    c.out_size[0] = ow; c.out_size[1] = oh;
    c.flags = hud_from_scene_ || split_ ? 0u : 1u;
    set_x_srv(kXTintSrv + 0, kPMask, true);
    if (split_) set_x_srv(kXTintSrv + 1, kPMask, true);
    else set_x_srv(kXTintSrv + 1, kPHudScore);
    for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXTintSrv + i, kPMask, true);
    set_x_uav(kXTintUav + 0, kPOutput); set_x_uav(kXTintUav + 1, kPOutput);
    transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    D3D12_RESOURCE_BARRIER written{}; written.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; written.UAV.pResource = private_[kPOutput].texture.Get();
    list_->ResourceBarrier(1, &written);  // after Latewarp's writes
    x_dispatch(cs_tint_.Get(), &c, kXTintSrv, kXTintUav, (ow + 7) / 8, (oh + 7) / 8);
}

void Renderer::execute() {
    list_->Close();
    ID3D12CommandQueue* q = context_queue();
    std::uint64_t& game_wait = pending_game_waits_[context_];
    if (game_wait && fence_game_) { q->Wait(fence_game_.Get(), game_wait); game_wait = 0; }
    // The intake rewrites the set the warp showed until the last swap: after the warps that read it.
    if (context_ == 1 && release_wait_) q->Wait(fences_[0].Get(), release_wait_);
    ID3D12CommandList* lists[] = {list_.Get()};
    q->ExecuteCommandLists(1, lists);
}

void Renderer::signal() {
    context_queue()->Signal(fences_[context_].Get(), ++fence_values_[context_]);
    frame_values_[frame_index_] = fence_values_[context_];
}

void Renderer::flush_and_wait() {
    execute();
    signal();
    wait_idle();
    allocators_[frame_index_]->Reset();
    list_->Reset(allocators_[frame_index_].Get(), nullptr);
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
}

bool Renderer::record_motion_samples(const IngestedSource& src, std::uint32_t grid_w, std::uint32_t grid_h) {
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
    set_x_srv(kXSampleSrv + 2, own_motion_ ? kPFlowB : kPDepth);  // (the displacements found, with their confidence)
    for (UINT i = 3; i < kXSrvCount; ++i) set_x_srv(kXSampleSrv + i, kPDepth);
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
    if (own_motion_) { c.flags = 2; c.out_size[0] = std::max(1u, w / 2); c.out_size[1] = std::max(1u, h / 2); }  // (level 0 of the displacements)
    x_dispatch(cs_sample_.Get(), &c, kXSampleSrv, kXSampleUav, (grid_w + 7) / 8, (grid_h + 7) / 8);
    auto to_copy = transition_barrier(samples_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    list_->CopyBufferRegion(samples_readback_.Get(), 0, samples_.Get(), 0, bytes);
    samples_pending_bytes_ = bytes;
    last_flush_ms_ = 0;
    return true;
}

bool Renderer::read_motion_samples(std::vector<float>& out) {
    if (!samples_readback_ || !samples_pending_bytes_) return false;
    const UINT bytes = samples_pending_bytes_;
    samples_pending_bytes_ = 0;
    float* mapped = nullptr;
    D3D12_RANGE range{0, bytes};
    if (FAILED(samples_readback_->Map(0, &range, reinterpret_cast<void**>(&mapped)))) return false;
    out.assign(mapped, mapped + bytes / 4);
    D3D12_RANGE none{0, 0};
    samples_readback_->Unmap(0, &none);
    return true;
}

bool Renderer::sample_motion(const IngestedSource& src, std::uint32_t grid_w, std::uint32_t grid_h, std::vector<float>& out) {
    if (!record_motion_samples(src, grid_w, grid_h)) return false;
    LARGE_INTEGER a, b, f; QueryPerformanceCounter(&a);
    flush_and_wait();
    QueryPerformanceCounter(&b); QueryPerformanceFrequency(&f);
    last_flush_ms_ = float(double(b.QuadPart - a.QuadPart) * 1000.0 / double(f.QuadPart));
    return read_motion_samples(out);
}

void Renderer::wait_for(std::uint64_t value) {
    if (completed(value)) return;
    fences_[0]->SetEventOnCompletion(value, fence_event_);
    WaitForSingleObject(fence_event_, 1000);
}

int Renderer::print_debug_messages() {
    ComPtr<ID3D12InfoQueue> info;
    if (!device_ || FAILED(device_.As(&info))) { std::printf("D3D12 debug layer: not available\n"); return 0; }
    int errors = 0;
    const UINT64 n = info->GetNumStoredMessages();
    for (UINT64 i = 0; i < n; ++i) {
        SIZE_T size = 0;
        info->GetMessage(i, nullptr, &size);
        std::vector<char> buffer(size);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(buffer.data());
        if (FAILED(info->GetMessage(i, m, &size))) continue;
        if (m->Severity > D3D12_MESSAGE_SEVERITY_ERROR) continue;
        ++errors;
        if (errors <= 20) std::printf("D3D12 %s: %s\n", m->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ? "CORRUPTION" : "ERROR", m->pDescription);
    }
    std::printf("D3D12 debug layer: %llu messages, %d errors\n", static_cast<unsigned long long>(n), errors);
    info->ClearStoredMessages();
    return errors;
}

void Renderer::wait_for_intake(std::uint64_t value) {
    ID3D12Fence* f = fences_[split_ ? 1 : 0].Get();
    if (f->GetCompletedValue() >= value) return;
    f->SetEventOnCompletion(value, fence_event_);
    WaitForSingleObject(fence_event_, 1000);
}

// ---- split queues
void Renderer::swap_shown() {
    for (int i = 0; i < kPCount; ++i) {
        const auto id = static_cast<PrivateId>(i);
        if (!shown_id(id)) continue;
        std::swap(private_[id], front_[id]);
        // The per-texture descriptors describe the texture the intake writes next.
        if (private_[id].texture) {
            device_->CreateUnorderedAccessView(private_[id].texture.Get(), nullptr, nullptr, cpu(kPrivUav + id));
            device_->CreateShaderResourceView(private_[id].texture.Get(), nullptr, cpu(kPrivSrv + id));
        }
    }
}

bool Renderer::set_split(bool on) {
    if (on && !iqueue_) on = false;
    if (on == split_) return false;
    wait_idle();
    // Turning on: what was taken in so far becomes the shown set. Turning off: the shown set goes back to
    // where the single queue reads everything.
    swap_shown();
    if (on) { shown_mask_ = mask_ready_; shown_fill_ = fill_ready_; shown_stretch_ = stretch_built_; }
    else { mask_ready_ = shown_mask_; fill_ready_ = shown_fill_; }
    private_[kPPrevious] = Private{};  // (with split queues an alias of a shown texture)
    previous_valid_ = false;
    history_valid_ = objects_built_ = shown_objects_ = false;
    pending_show_ = 0; release_wait_ = 0;
    mem_last_ = mem_shown_ = mem_pending_ = -1;
    split_ = on;
    return true;
}

ID3D12GraphicsCommandList* Renderer::begin_intake() {
    if (!split_) return begin_frame();
    built_mask_ = false;
    fill_ready_ = false;
    mem_pending_ = -1;
    return begin(1);
}

bool Renderer::show_intake() {
    if (!split_ || !pending_show_ || fences_[1]->GetCompletedValue() < pending_show_) return false;
    swap_shown();
    shown_mask_ = built_mask_;
    shown_objects_ = objects_built_;
    shown_stretch_ = stretch_built_;
    shown_fill_ = fill_ready_;
    mem_shown_ = mem_pending_;
    // Warps recorded so far may still read the set the intake writes next.
    release_wait_ = fence_values_[0];
    pending_show_ = 0;
    return true;
}

bool Renderer::update_memory(const IngestedSource& src, const float clip_to_prev_clip[16], bool depth_inverted, bool consecutive) {
    const UINT ow = private_[kPBackbuffer].width, oh = private_[kPBackbuffer].height;
    auto fail = [&]() { mem_last_ = mem_pending_ = -1; if (!split_) mem_shown_ = -1; return false; };
    if (!src.valid || !src.has_depth || !private_[kPDepth].texture || !private_[kPMask].texture ||
        private_[kPMask].width != ow || private_[kPMask].height != oh) return fail();
    const UINT mw = static_cast<UINT>(std::ceil(ow * (1.0f + 2.0f * kMemMargin) * 0.5f));
    const UINT mh = static_cast<UINT>(std::ceil(oh * (1.0f + 2.0f * kMemMargin) * 0.5f));
    const int r = mem_last_, wi = r == 0 ? 1 : 0;
    const auto wc = static_cast<PrivateId>(kPMem0 + wi), wz = static_cast<PrivateId>(kPMemZ0 + wi);
    if (!ensure_private(wc, mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT) || !ensure_private(wz, mw, mh, DXGI_FORMAT_R32_FLOAT)) return fail();
    const bool prev = consecutive && r >= 0 && private_[kPMem0 + r].width == mw && private_[kPMem0 + r].height == mh &&
                      private_[kPMemZ0 + r].width == mw;
    XConstants c{};
    std::memcpy(c.clip_to_prev, clip_to_prev_clip, sizeof(c.clip_to_prev));
    c.out_size[0] = ow; c.out_size[1] = oh;
    c.grid[0] = mw; c.grid[1] = mh;
    const UINT dw = private_[kPDepth].width, dh = private_[kPDepth].height;
    const Rect2 dr = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, dw, dh};
    c.rect[0] = dr.x; c.rect[1] = dr.y; c.rect[2] = std::min(dr.w, dw - dr.x); c.rect[3] = std::min(dr.h, dh - dr.y);
    c.flags = (prev ? 1u : 0u) | (depth_inverted ? 16u : 0u);
    // (the HUD-less picture when the game sends one: the memory is scenery)
    const PrivateId frame = src.has_hudless && private_[kPHudless].texture ? kPHudless : kPBackbuffer;
    set_x_srv(kXMemSrv + 0, frame);
    set_x_srv(kXMemSrv + 1, kPMask);
    set_x_srv(kXMemSrv + 2, kPDepth);
    set_x_srv(kXMemSrv + 3, prev ? static_cast<PrivateId>(kPMem0 + r) : frame);
    set_x_srv(kXMemSrv + 4, prev ? static_cast<PrivateId>(kPMemZ0 + r) : kPDepth);
    set_x_srv(kXMemSrv + 5, kPDepth);
    set_x_uav(kXMemUav + 0, wc); set_x_uav(kXMemUav + 1, wz);
    transition(private_[wc], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(private_[wz], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_memory_.Get(), &c, kXMemSrv, kXMemUav, (mw + 7) / 8, (mh + 7) / 8);
    transition(private_[wc], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(private_[wz], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    mem_last_ = mem_pending_ = wi;
    if (!split_) mem_shown_ = wi;
    return true;
}

void Renderer::copy_shown_to_output() {
    // An unwarped refresh: the shown frame as it is, through the warp shader with the identity (the
    // shown textures stay in the shader-read state both queues use).
    const auto& bb = shown(kPBackbuffer);
    if (!bb.texture || !private_[kPOutput].texture || bb.width != private_[kPOutput].width || bb.height != private_[kPOutput].height) return;
    XConstants c{};
    for (int i = 0; i < 16; ++i) c.clip_to_prev[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    c.out_size[0] = bb.width; c.out_size[1] = bb.height;
    c.rect[2] = c.rect[3] = 1; c.grid[0] = c.grid[1] = 1;
    c.flags = 32u;  // depth-independent: no depth reads
    const float wc[20] = {};
    for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWShown) + i, kPBackbuffer, true);
    set_x_uav(kWUav(kWShown) + 0, kPOutput); set_x_uav(kWUav(kWShown) + 1, kPOutput);
    transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_own_warp_.Get(), &c, wc, kWShown, (bb.width + 7) / 8, (bb.height + 7) / 8);
}

bool Renderer::take_motion_fit(MotionFit& fit) {
    if (!fit_ready_) return false;
    fit = fit_latest_;
    fit_ready_ = false;
    return true;
}


DXGI_FORMAT Renderer::colour_format(DXGI_FORMAT source) {
    DXGI_FORMAT compact = DXGI_FORMAT_UNKNOWN;
    switch (srv_format(source)) {
        case DXGI_FORMAT_R10G10B10A2_UNORM: compact = DXGI_FORMAT_R10G10B10A2_UNORM; break;
        case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM: compact = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case DXGI_FORMAT_R11G11B10_FLOAT: compact = DXGI_FORMAT_R11G11B10_FLOAT; break;
        default: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    }
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{compact};
    constexpr auto kNeed2 = D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
    if (FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))) ||
        !(fs.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) || (fs.Support2 & kNeed2) != kNeed2)
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    return compact;
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
    pending_show_ = 0; release_wait_ = 0; shown_mask_ = shown_fill_ = false;
    mem_last_ = mem_shown_ = mem_pending_ = -1;
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
    const HRESULT named = device_->OpenSharedHandleByName(object_name(pid_, session_, L"tex", slot, kind, generation).c_str(), GENERIC_ALL, &handle);
    if (FAILED(named)) {
        char note[160];
        std::snprintf(note, sizeof(note), "could not open shared %s slot %d gen %u by name: 0x%08lX", tex_name(kind), slot, generation, static_cast<unsigned long>(named));
        notes_.push_back(note);
        return nullptr;
    }
    const HRESULT hr = device_->OpenSharedHandle(handle, IID_PPV_ARGS(&t.resource));
    CloseHandle(handle);
    if (FAILED(hr)) {
        char note[160];
        std::snprintf(note, sizeof(note), "could not open shared %s slot %d gen %u: 0x%08lX", tex_name(kind), slot, generation, static_cast<unsigned long>(hr));
        notes_.push_back(note);
        return nullptr;
    }
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

ID3D12GraphicsCommandList* Renderer::begin_frame() { return begin(0); }

ID3D12GraphicsCommandList* Renderer::begin(int context) {
    context_ = context;
    list_ = lists_[context];
    ring_pos_[context] = (ring_pos_[context] + 1) % 3;
    frame_index_ = static_cast<std::uint32_t>(context * 3 + ring_pos_[context]);
    const std::uint64_t timestamp_frequency_ = timestamp_frequencies_[context];  // (this slot's queue)
    if (fences_[context]->GetCompletedValue() < frame_values_[frame_index_]) {
        fences_[context]->SetEventOnCompletion(frame_values_[frame_index_], fence_event_);
        WaitForSingleObject(fence_event_, 1000);
    }
    // GPU time of the frame that last used this allocator (a skipped refresh has none).
    const bool skipped = skipped_slot_[frame_index_];
    skipped_slot_[frame_index_] = false;
    if (skipped) ++usage_.skipped;
    if (readback_ && frame_values_[frame_index_] && !skipped) {
        std::uint64_t* ts = nullptr;
        D3D12_RANGE range{frame_index_ * 16, frame_index_ * 16 + 16};
        if (SUCCEEDED(readback_->Map(0, &range, reinterpret_cast<void**>(&ts)))) {
            auto a = ts[frame_index_ * 2], b = ts[frame_index_ * 2 + 1];
            const float ms = b > a ? static_cast<float>(double(b - a) * 1000.0 / double(timestamp_frequency_)) : 0.0f;
            if (intake_slot_[frame_index_]) {  // a new game frame taken in between refreshes
                if (b > a && continuation_slot_[frame_index_]) {
                    intake_gpu_ms_ += ms;  // same frame, second submission
                    usage_.intake_ms += ms; usage_.rest_ms += ms;
                } else if (b > a) {
                    intake_gpu_ms_ = ms;
                    ++usage_.intakes; usage_.intake_ms += ms;
                    std::uint64_t* st = nullptr;
                    D3D12_RANGE sr{frame_index_ * 24, frame_index_ * 24 + 24};
                    if (stage_valid_[frame_index_] && stage_readback_ && SUCCEEDED(stage_readback_->Map(0, &sr, reinterpret_cast<void**>(&st)))) {
                        const std::uint64_t sa = st[frame_index_ * 3], s0 = st[frame_index_ * 3 + 1], s1 = st[frame_index_ * 3 + 2];
                        if (a <= sa && sa <= s0 && s0 <= s1 && s1 <= b) {
                            const double k = 1000.0 / double(timestamp_frequency_);
                            ++usage_.split;
                            usage_.access_ms += double(sa - a) * k;
                            usage_.depth_ms += double(s0 - sa) * k; usage_.colour_ms += double(s1 - s0) * k; usage_.rest_ms += double(b - s1) * k;
                        }
                        D3D12_RANGE none0{0, 0};
                        stage_readback_->Unmap(0, &none0);
                    }
                }
                b = a;  // not a presented frame: no present timing from it
            } else if (b > a) { gpu_ms_ = ms; ++usage_.warps; usage_.warp_ms += ms; }
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
            fit_latest_.frame = fit_frame_[frame_index_];
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
    stages_marked_ = 0;
    stage_valid_[frame_index_] = false;
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

bool Renderer::estimate_motion(DXGI_FORMAT colour, bool write_motion) {
    const UINT W = private_[kPDepth].width, H = private_[kPDepth].height;
    const UINT w0 = std::max(1u, W / 2), h0 = std::max(1u, H / 2);  // level 0: half the render grid
    constexpr UINT kLevels = 5;
    auto level_w = [&](UINT k) { return std::max(1u, w0 >> k); };
    auto level_h = [&](UINT k) { return std::max(1u, h0 >> k); };
    const UINT aw = w0 + level_w(1), ah = h0;  // the pyramid in one texture (see cs_flow_luma)
    const bool same = private_[kPLumaA].texture && private_[kPLumaA].width == aw && private_[kPLumaA].height == ah;
    if (!ensure_private(kPLumaA, aw, ah, DXGI_FORMAT_R16_FLOAT) || !ensure_private(kPLumaB, aw, ah, DXGI_FORMAT_R16_FLOAT) ||
        !ensure_private(kPFeat, aw, ah, DXGI_FORMAT_R16G16B16A16_FLOAT) || !ensure_private(kPFlowA, aw, ah + 8, DXGI_FORMAT_R16G16B16A16_FLOAT) ||
        !ensure_private(kPFlowB, aw, ah + 8, DXGI_FORMAT_R16G16B16A16_FLOAT)) return false;  // (+8: room for the fill's smallest levels)
    if (!same) flow_previous_ = false;
    const bool had_previous = flow_previous_;
    const PrivateId now = flow_luma_ ? kPLumaA : kPLumaB, before = flow_luma_ ? kPLumaB : kPLumaA;  // (flow_luma_: which one holds the previous frame)
    XConstants c{};
    c.grid[0] = w0; c.grid[1] = h0;
    c.out_size[0] = private_[kPBackbuffer].width; c.out_size[1] = private_[kPBackbuffer].height;
    c.mv_size[0] = W; c.mv_size[1] = H;
    c.threshold = 1.5f;
    const std::uint32_t tone = colour == DXGI_FORMAT_R16G16B16A16_FLOAT ? 1u : 0u;
    auto table = [&](FlowTable t, PrivateId s0, PrivateId s1, PrivateId s2, PrivateId out) {
        const UINT base = kXFlow + t * 8;
        set_x_srv(base + 0, s0); set_x_srv(base + 1, s1); set_x_srv(base + 2, s2);
        for (UINT i = 3; i < kXSrvCount; ++i) set_x_srv(base + i, s0);
        set_x_uav(base + 6, out); set_x_uav(base + 7, out);
        return base;
    };
    auto run = [&](ID3D12PipelineState* pso, UINT base, PrivateId out, UINT w, UINT h) {
        transition(private_[out], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        x_dispatch(pso, &c, base, base + 6, (w + 7) / 8, (h + 7) / 8);
        transition(private_[out], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    };
    // This frame's brightness pyramid and gradients.
    c.flags = tone;
    run(cs_flow_luma_.Get(), table(kFlowLuma, kPBackbuffer, kPBackbuffer, kPBackbuffer, now), now, aw, ah);
    c.flags = 0;
    run(cs_flow_feat_.Get(), table(kFlowFeat, now, now, now, kPFeat), kPFeat, aw, ah);
    const UINT motion = table(kFlowMotion, kPFeat, before, kPFlowA, kPMotion);  // (the filled displacements end in A)
    if (flow_previous_) {
        const UINT lk_ab = table(kFlowLkAB, kPFeat, before, kPFlowA, kPFlowB), lk_ba = table(kFlowLkBA, kPFeat, before, kPFlowB, kPFlowA);
        const UINT med_ab = table(kFlowMedianAB, kPFeat, before, kPFlowA, kPFlowB), med_ba = table(kFlowMedianBA, kPFeat, before, kPFlowB, kPFlowA);
        bool in_a = true;  // which texture holds the displacements so far
        auto step = [&](ID3D12PipelineState* pso, UINT k) {
            run(pso, pso == cs_flow_median_.Get() ? (in_a ? med_ab : med_ba) : (in_a ? lk_ab : lk_ba), in_a ? kPFlowB : kPFlowA, level_w(k), level_h(k));
            in_a = !in_a;
        };
        // Coarsest level: whole-pixel search, then the refinements; every level ends with a median.
        c.rect[0] = kLevels - 1; c.rect[1] = 0; c.rect[2] = 12;
        run(cs_flow_search_.Get(), table(kFlowSearch, kPFeat, before, kPFeat, kPFlowA), kPFlowA, level_w(kLevels - 1), level_h(kLevels - 1));
        step(cs_flow_median_.Get(), kLevels - 1);
        static constexpr int kRefinements[kLevels] = {1, 2, 3, 3, 4};
        for (int k = kLevels - 1; k >= 0; --k) {
            c.rect[0] = static_cast<std::uint32_t>(k);
            for (int i = 0; i < kRefinements[k]; ++i) {
                c.rect[1] = (i == 0 && k != int(kLevels) - 1) ? 1u : 0u;  // from the coarser level
                step(cs_flow_lk_.Get(), static_cast<UINT>(k));
            }
            c.rect[1] = 0;
            step(cs_flow_median_.Get(), static_cast<UINT>(k));
        }
        // Which displacements are known (always ends in A: 1 + 13 refinements + 5 medians + this = 20 passes).
        c.rect[0] = 0; c.rect[1] = 0; c.flags = 2;
        step(cs_flow_lk_.Get(), 0);
        c.flags = 0;
        // Filling in what is not known (see cs_flow_fill). A level's own displacements are in one texture
        // for even levels and in the other for odd ones; the filled level j is written into the texture
        // level j is not in, where the filled level j + 1 it needs also went. So every pass reads one
        // texture and writes the other, and level 0 ends own and confident in B, for the camera estimate,
        // and filled in A, for the motion texture.
        const UINT fill_ab = table(kFlowFillAB, kPFeat, before, kPFlowA, kPFlowB), fill_ba = table(kFlowFillBA, kPFeat, before, kPFlowB, kPFlowA);
        auto fill = [&](UINT level, UINT mode, bool from_a) {
            c.rect[0] = level; c.rect[1] = mode;
            run(cs_flow_fill_.Get(), from_a ? fill_ab : fill_ba, from_a ? kPFlowB : kPFlowA, level_w(level), level_h(level));
        };
        constexpr UINT kTop = 8;  // (an even level, a few pixels large: it ends in the texture level 0 is in)
        for (UINT j = 1; j <= kTop; ++j) fill(j, 0, (j - 1) % 2 == 0);
        // Level 0 again, in B, without what would have come from outside the previous picture; from here on
        // B holds the even levels and the confident displacements the camera is estimated from.
        c.rect[2] = kTop;
        fill(0, 3, true);
        for (UINT j = 1; j <= kTop; ++j) fill(j, 0, (j - 1) % 2 == 1);
        fill(kTop, 1, false);
        for (int j = int(kTop) - 1; j >= 0; --j) fill(static_cast<UINT>(j), 2, j % 2 == 1);
        c.rect[0] = 0; c.rect[1] = 0;
    } else {
        // No previous frame: no motion, nothing known (a search that reaches nowhere writes zeros).
        c.rect[0] = 0; c.rect[1] = 0; c.rect[2] = 0;
        run(cs_flow_search_.Get(), table(kFlowSearch, kPFeat, before, kPFeat, kPFlowA), kPFlowA, w0, h0);
        run(cs_flow_search_.Get(), table(kFlowFillAB, kPFeat, before, kPFeat, kPFlowB), kPFlowB, w0, h0);
        c.flags = 4;
    }
    if (write_motion) run(cs_flow_motion_.Get(), motion, kPMotion, W, H);
    flow_luma_ ^= 1;
    flow_previous_ = true;
    return had_previous;
}

bool Renderer::picture_motion() {
    if (!private_[kPDepth].texture || !private_[kPBackbuffer].texture) return false;
    return estimate_motion(private_[kPBackbuffer].format, false);
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
    pending_game_waits_[context_] = std::max(pending_game_waits_[context_], m.fence_value);

    // Shared textures live in COMMON between the two devices' queues.
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    for (auto* r : sources)
        if (r) { barriers.push_back(transition_barrier(r, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)); reading_.push_back(r); }
    list_->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    mark_stage();  // shared textures made readable

    const DXGI_FORMAT colour = colour_format(static_cast<DXGI_FORMAT>(bb.format));
    // The game's picture (and the copy of the previous one the HUD detection compares it with).
    auto take_colour = [&]() {
    ensure_private(kPBackbuffer, bb.width, bb.height, colour);
    ensure_private(kPOutput, bb.width, bb.height, colour);  // the warped frame: the same values as the game's
    if (keep_previous_ && ingested_ && split_) {
        // Split queues: the previous frame is the shown one - read in place (both queues only read it).
        const PrivateId from = last_had_hudless_ ? kPHudless : kPBackbuffer;
        if (front_[from].texture) {
            private_[kPPrevious] = front_[from];
            previous_valid_ = true;
            previous_from_hudless_ = last_had_hudless_;
        }
    } else if (keep_previous_ && ingested_) {
        const PrivateId from = last_had_hudless_ ? kPHudless : kPBackbuffer;
        if (private_[from].texture && ensure_private(kPPrevious, private_[from].width, private_[from].height, private_[from].format)) {
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
    };

    const auto& dp = m.tex[kDepth];
    // Depth without motion vectors, and no camera from the game (ReShade's depth, no DLSS or FSR): the
    // motion is estimated from the picture, which is then taken first.
    const bool estimate = sources[kDepth] && !sources[kMotion] && m.camera.estimated != 0;
    keep_history(last_had_hudless_ ? kPHudless : kPBackbuffer, sources[kDepth] ? dp.width : 0, sources[kDepth] ? dp.height : 0, bb.width, bb.height,
                 last_had_hudless_ ? DXGI_FORMAT_R16G16B16A16_FLOAT : colour);
    objects_built_ = false;
    own_motion_ = false;
    if (sources[kDepth]) {
        ensure_private(kPDepth, dp.width, dp.height, DXGI_FORMAT_R32_FLOAT);
        ensure_private(kPMotion, dp.width, dp.height, DXGI_FORMAT_R16G16_FLOAT);
        convert(sources[kDepth], static_cast<DXGI_FORMAT>(dp.format), slot, kDepth, kPDepth, dp.width, dp.height);
        const auto& mv = m.tex[kMotion];
        if (sources[kMotion]) {
            convert(sources[kMotion], static_cast<DXGI_FORMAT>(mv.format), slot, kMotion, kPMotion,
                    std::min(mv.width, dp.width), std::min(mv.height, dp.height));
            out.has_motion = true;
        } else if (estimate) {
            take_colour();
            estimate_motion(colour);
            out.has_motion = true;
            out.motion_estimated = true;
            own_motion_ = true;
        }
        out.depth_rect = {dp.ext_x, dp.ext_y, dp.ext_w, dp.ext_h};
        out.has_depth = true;
    }
    // Depth and motion vectors are recorded first: a caller that needs them on the CPU (camera
    // estimation) can flush here and wait for this small amount of work, before the 4K colour work.
    if (after_depth) after_depth(out);
    mark_stage();

    out.color_w = bb.width; out.color_h = bb.height;
    out.color_rect = {0, 0, bb.width, bb.height};
    if (!own_motion_) take_colour();

    const auto& hl = m.tex[kHudless];
    // HUD layers (games that send them) in half-float: the UI's alpha needs more than 2 bits.
    if (sources[kHudless] && hl.width == bb.width && hl.height == bb.height &&
        ensure_private(kPHudless, bb.width, bb.height, DXGI_FORMAT_R16G16B16A16_FLOAT)) {
        convert(sources[kHudless], static_cast<DXGI_FORMAT>(hl.format), slot, kHudless, kPHudless, bb.width, bb.height);
        out.has_hudless = true;
    }
    const auto& ui = m.tex[kUi];
    if (sources[kUi] && ui.width == bb.width && ui.height == bb.height && ensure_private(kPUi, bb.width, bb.height, DXGI_FORMAT_R16G16B16A16_FLOAT)) {
        convert(sources[kUi], static_cast<DXGI_FORMAT>(ui.format), slot, kUi, kPUi, bb.width, bb.height);
        out.has_ui = true;
    }
    const auto& sc = m.tex[kScene];
    if (sources[kScene] && sc.width == bb.width && sc.height == bb.height &&
        ensure_private(kPScene, bb.width, bb.height, colour_format(static_cast<DXGI_FORMAT>(sc.format)))) {
        convert(sources[kScene], static_cast<DXGI_FORMAT>(sc.format), slot, kScene, kPScene, bb.width, bb.height);
        out.has_scene = true;
    }
    // Frame generation: the images generated before this frame, in order (with its depth only).
    for (int i = 0; i < std::min<int>(int(m.generated), kMaxGenerated) && out.has_depth; ++i) {
        const auto& g = m.tex[kGen0 + i];
        const auto id = static_cast<PrivateId>(kPGen0 + i);
        if (!sources[kGen0 + i] || g.width != bb.width || g.height != bb.height ||
            !ensure_private(id, bb.width, bb.height, colour_format(static_cast<DXGI_FORMAT>(g.format)))) break;
        convert(sources[kGen0 + i], static_cast<DXGI_FORMAT>(g.format), slot, kGen0 + i, id, bb.width, bb.height);
        out.gen_index[i] = int(m.gen_index[i]) >= 1 && m.gen_index[i] <= m.per_frame ? int(m.gen_index[i]) : i + 1;
        out.generated = i + 1;
    }
    if (out.generated) out.per_frame = std::max<int>(int(m.per_frame), out.gen_index[out.generated - 1]);
    ingested_ = true;
    last_had_hudless_ = out.has_hudless;
    mark_stage();
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
    // Latewarp always takes a UI layer: an empty one (zero-initialised, 8-bit) for games without.
    if (!split && private_[kPBackbuffer].texture)
        ensure_private(kPZeroUi, private_[kPBackbuffer].width, private_[kPBackbuffer].height, DXGI_FORMAT_R8G8B8A8_UNORM);
    in.ui = split ? private_[kPUi].texture.Get() : private_[kPZeroUi].texture.Get();
    in.depth = private_[kPDepth].texture.Get();
    in.motion = private_[kPMotion].texture.Get();
    in.output = private_[kPOutput].texture.Get();
    in.color_rect = src.color_rect;
    in.depth_rect = src.depth_rect;
    if (in.output) transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return in;
}

void Renderer::submit_work(bool continuation) {
    if (timestamps_) {
        list_->EndQuery(timestamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 2 + 1);
        list_->ResolveQueryData(timestamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 2, 2, readback_.Get(), frame_index_ * 16);
    }
    if (stages_marked_ == 3) {
        list_->ResolveQueryData(stage_stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * 3, 3, stage_readback_.Get(), frame_index_ * 24);
        stage_valid_[frame_index_] = true;
    }
    execute();
    signal();
    intake_slot_[frame_index_] = true;
    continuation_slot_[frame_index_] = continuation;
}

void Renderer::skip_frame() {
    intake_slot_[frame_index_] = false;
    skipped_slot_[frame_index_] = true;
    execute();
    signal();
}

void Renderer::finish_frame(bool warped, int marker) {
    intake_slot_[frame_index_] = false;
    // With split queues the unwarped frame is the shown one, copied (the warp never changes the state of
    // the textures the intake queue may read at the same time).
    if (split_ && !warped && private_[kPOutput].texture) { copy_shown_to_output(); warped = true; }
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
    execute();
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
    signal();
}

void Renderer::stash_source(int set, bool hudless) {
    static const PrivateId kFrom[] = {kPBackbuffer, kPDepth, kPMotion, kPHudless};
    static const PrivateId kTo[2][4] = {{kPStashColor, kPStashDepth, kPStashMotion, kPStashHudless},
                                        {kPStash2Color, kPStash2Depth, kPStash2Motion, kPStash2Hudless}};
    const PrivateId* to_set = kTo[set ? 1 : 0];
    wait_idle();
    context_ = 0; list_ = lists_[0]; frame_index_ = static_cast<std::uint32_t>(ring_pos_[0]);
    for (int i = 0; i < 4; ++i) {
        const auto& from = split_ && shown_id(kFrom[i]) ? front_[kFrom[i]] : private_[kFrom[i]];
        if (from.texture && (i < 3 || hudless)) ensure_private(to_set[i], from.width, from.height, from.format);
        else private_[to_set[i]] = Private{};
    }
    allocators_[frame_index_]->Reset();
    list_->Reset(allocators_[frame_index_].Get(), nullptr);
    for (int i = 0; i < 4; ++i) {
        auto& from = split_ && shown_id(kFrom[i]) ? front_[kFrom[i]] : private_[kFrom[i]];
        auto& to = private_[to_set[i]];
        if (!from.texture || !to.texture) continue;
        const auto before = from.state;
        transition(from, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(to, D3D12_RESOURCE_STATE_COPY_DEST);
        list_->CopyResource(to.texture.Get(), from.texture.Get());
        transition(from, before);
        transition(to, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    list_->Close();
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    wait_idle();
}

bool Renderer::read_back(int which, std::vector<std::uint16_t>& pixels, std::uint32_t& w, std::uint32_t& h) {
    static const PrivateId kWhich[] = {kPBackbuffer, kPOutput, kPScene, kPDepth, kPMotion, kPObject, kPMask, kPHudScore, kPWorld, kPFill,
                                       kPStashColor, kPStashDepth, kPStashMotion, kPFlowB, kPStashHudless, kPStash2Color, kPStash2Hudless,
                                       kPStash2Depth, kPStash2Motion, kPHudless, kPGen0, kPGen1, kPGen2, kPGenZ0, kPGenZ1, kPGenZ2, kPUi};
    if (which < 0 || which >= int(std::size(kWhich))) return false;
    auto& p = split_ && shown_id(kWhich[which]) ? front_[kWhich[which]] : private_[kWhich[which]];
    if (!p.texture) return false;
    wait_idle();
    context_ = 0; list_ = lists_[0]; frame_index_ = static_cast<std::uint32_t>(ring_pos_[0]);
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
    // Always handed out as half-float RGBA, whatever the private format.
    auto to_half = [](float v) -> std::uint16_t {
        std::uint32_t x; std::memcpy(&x, &v, 4);
        const std::uint32_t sign = (x >> 16) & 0x8000u;
        const int e = int((x >> 23) & 0xFF) - 127 + 15;
        const std::uint32_t mant = x & 0x7FFFFFu;
        // (small values - distant reversed depth - as subnormal halves instead of zero)
        if (e <= 0) return e < -10 ? std::uint16_t(sign) : std::uint16_t(sign | (((mant | 0x800000u) >> (1 - e)) + 0x1000u) >> 13);
        if (e >= 31) return std::uint16_t(sign | 0x7C00u);
        return std::uint16_t(sign | (std::uint32_t(e) << 10) | ((mant + 0x1000u) >> 13));
    };
    auto small_float = [](std::uint32_t bits, int mant_bits) {  // R11G11B10: 5-bit exponent, no sign
        const std::uint32_t e = bits >> mant_bits, m = bits & ((1u << mant_bits) - 1);
        if (e == 0) return std::ldexp(float(m), -14 - mant_bits);
        return std::ldexp(1.0f + float(m) / float(1u << mant_bits), int(e) - 15);
    };
    for (std::uint32_t y = 0; y < h; ++y) {
        const std::uint8_t* row = mapped + fp.Offset + std::size_t(y) * fp.Footprint.RowPitch;
        std::uint16_t* out = &pixels[std::size_t(y) * w * 4];
        if (desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) { std::memcpy(out, row, std::size_t(w) * 8); continue; }
        if (desc.Format == DXGI_FORMAT_R16G16_FLOAT) {
            for (std::uint32_t x = 0; x < w; ++x) {
                std::memcpy(out + std::size_t(x) * 4, row + std::size_t(x) * 4, 4);
                out[std::size_t(x) * 4 + 2] = 0; out[std::size_t(x) * 4 + 3] = 0x3C00;
            }
            continue;
        }
        for (std::uint32_t x = 0; x < w; ++x) {
            std::uint32_t v = 0;
            if (desc.Format == DXGI_FORMAT_R8_UNORM) v = row[x];
            else std::memcpy(&v, row + std::size_t(x) * 4, 4);
            float c[4] = {0, 0, 0, 1};
            if (desc.Format == DXGI_FORMAT_R32_FLOAT) std::memcpy(&c[0], &v, 4);
            else if (desc.Format == DXGI_FORMAT_R8_UNORM) c[0] = float(v) / 255.0f;
            else if (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM) {
                for (int k = 0; k < 4; ++k) c[k] = float((v >> (8 * k)) & 0xFF) / 255.0f;
                if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM) std::swap(c[0], c[2]);
            } else if (desc.Format == DXGI_FORMAT_R10G10B10A2_UNORM) {
                for (int k = 0; k < 3; ++k) c[k] = float((v >> (10 * k)) & 0x3FF) / 1023.0f;
                c[3] = float(v >> 30) / 3.0f;
            } else if (desc.Format == DXGI_FORMAT_R11G11B10_FLOAT) {
                c[0] = small_float(v & 0x7FF, 6); c[1] = small_float((v >> 11) & 0x7FF, 6); c[2] = small_float(v >> 22, 5);
            }
            for (int k = 0; k < 4; ++k) out[std::size_t(x) * 4 + k] = to_half(c[k]);
        }
    }
    buffer->Unmap(0, nullptr);
    return true;
}

void Renderer::wait_idle() {
    if (!queue_ || !fences_[0]) return;
    if (iqueue_ && fences_[1]) {
        iqueue_->Signal(fences_[1].Get(), ++fence_values_[1]);
        if (fences_[1]->GetCompletedValue() < fence_values_[1]) {
            fences_[1]->SetEventOnCompletion(fence_values_[1], fence_event_);
            WaitForSingleObject(fence_event_, 2000);
        }
    }
    queue_->Signal(fences_[0].Get(), ++fence_values_[0]);
    if (fences_[0]->GetCompletedValue() < fence_values_[0]) {
        fences_[0]->SetEventOnCompletion(fence_values_[0], fence_event_);
        WaitForSingleObject(fence_event_, 2000);
    }
}

}  // namespace fw
