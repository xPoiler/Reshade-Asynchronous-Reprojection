#include "presenter/renderer.hpp"
#include "presenter/pose.hpp"
#include <cmath>
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
    // (marker, counter: the source's size when it differs - motion vectors at output resolution over the render
    // grid: the matching position; 0: the same pixel)
    const uint2 source = uint2(marker, counter);
    const uint2 at = all(source > 0) ? min(uint2((float2(id.xy) + 0.5) * float2(source) / float2(size)), source - 1) : id.xy;
    dst4[id.xy] = src.Load(int3(at, 0));
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
    const uint strip = marker & 3u;
    if (strip != 0 && p.y < 48 && p.x < 48 * 16) {
        const uint cell = uint(p.x) / 48;
        c = cell == (counter % 16) ? float4(1, 1, 1, 1) : (strip == 1 ? float4(0, 0.6, 0, 1) : float4(0.6, 0, 0, 1));
    }
    // A sequence recording (development, see SequenceRecorder): a red dot in the top-right corner while it records (marker
    // 4), amber and blinking while it is written out (8) - only in our output, never in the game's picture.
    if (marker & 12u) {
        const float r = max(10.0, float(size.y) / 90.0);
        const float d = length(pos.xy - float2(float(size.x) - 3.0 * r, 3.0 * r));
        const bool lit = (marker & 4u) || ((counter / 30u) & 1u) == 0;
        if (d < r * 1.25 && lit) c = d < r ? ((marker & 4u) ? float4(1, 0.05, 0.05, 1) : float4(1, 0.65, 0, 1)) : float4(0, 0, 0, 1);
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
// The game's picture for the HUD detections, on the scale their thresholds assume: values 0..1 roughly as bright as they
// look (SDR, HDR10 PQ). A linear scRGB picture (flag 32768, set for every pass while the game outputs scRGB: values up to
// 10-25, 1.0 = 80 nits) is encoded with the HDR10 curve (PQ) first - without it bright sky passed for HUD everywhere
// (Cyberpunk in scRGB: 40-58% of the screen "HUD", 2-13% in HDR10).
float3 seen(float3 c) {
    if (!(flags & 32768)) return c;
    const float3 l = pow(max(c, 0) * (80.0 / 10000.0), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * l) / (1.0 + 18.6875 * l), 78.84375);
}


Texture2D<float> depth_t : register(t0);
Texture2D<float2> motion_t : register(t1);
Texture2D<float4> analyze_flow_t : register(t2);  // (flag 2: XPAR's own motion, z = trusted)
RWTexture2D<float4> object_u : register(u0);        // xy: own motion (render px, to previous frame), z: depth, w: 1 moving, 2 attached to camera
RWStructuredBuffer<float4> partial_u : register(u1);
groupshared float4 gs_a[64], gs_b[64];
// A floor at p (reversed-Z depth): getting closer down the screen, evenly (a plane, not a bump or an edge), by at
// least 60% of its own depth value per screen height - the ground under a third-person camera looking down at it.
// An upright body facing the camera (a character) barely changes down the screen. k: render rows (~1/190 height).
bool floor_at(int2 p, int k) {
    const int2 lo = int2(rect.xy), hi = int2(rect.xy + rect.zw) - 1;
    const float a = depth_t.Load(int3(clamp(p - int2(0, k), lo, hi), 0));
    const float c = depth_t.Load(int3(clamp(p, lo, hi), 0));
    const float b = depth_t.Load(int3(clamp(p + int2(0, k), lo, hi), 0));
    const float slope = (b - a) / float(2 * k) * float(rect.w);
    return slope > 0 && slope > 0.6 * max(c, 1e-6) && abs(b - 2 * c + a) < 0.25 * abs(b - a);
}
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
            // Behind where the camera was a frame ago (the camera moved back past it - only when moving backwards,
            // only for what is very close: the back of a first-person gun): no point of the scene can be seen now
            // and have been behind the camera then, so it moves with the camera - attached. (It had no camera motion
            // at all to compare with and was left unclassified: warped with the scenery, DOOM Eternal walking back.)
            if (prev.w <= 0 && all(abs(g) < 1e4))
                o = float4(0, 0, d, ((flags & 1) && (flags & 16) && d > 1.0 / 4096.0) ? 2 : 0);
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
                // The camera's turn alone here (its motion at infinite distance).
                const float4 far = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), clip_to_prev);
                const float2 turn_px = far.w > 0 ? (float2(far.x / far.w * 0.5 + 0.5, 0.5 - far.y / far.w * 0.5) - uv) * float2(rect.zw)
                                                 : float2(0, 0);
                // Part of the camera rig (flag 2048): what stays on screen because it moves with the camera, not because
                // the player keeps it there (a car or a person followed by turning the camera stays on screen too, and
                // must warp - held, it split from the scenery, worse still with XPAR frame generation). Two ways of being
                // carried by the camera: where the camera's motion at this spot comes mostly from its moving, not its
                // turning (walking with a third-person character: the scenery shifts by its distance), or where the turn
                // is cancelled at this distance (the centre an orbiting camera turns around: the character). A followed
                // object far from both stays on screen against a turn that is not cancelled there.
                const float cam_len = length(cam_px), turn_len = length(turn_px);
                const bool carried = !(flags & 2048) || length(cam_px - turn_px) >= 0.5 * cam_len;
                // (XPAR's own motion, flag 2: a camera estimated without any movement of its own here - the estimate found none,
                // with only distant scenery to measure it on - can not tell an orbit's centre from a followed object: what stays
                // on the screen while it turns is taken as the orbit's centre. Dark Souls: its character moved 3-32 px against a
                // turn of 40-426 and was never held - warped with the turn, it snapped back every frame)
                const bool unmoved = (flags & 2) && length(cam_px - turn_px) < 0.05 * turn_len;
                const bool orbit_centre = !(flags & 2048) || cam_len < 0.5 * turn_len || unmoved;
                // Still compared with the turn alone (flag 128, on by default): a camera orbiting a third-person
                // character keeps it nearly still on screen. The camera model includes the orbit, so the
                // character follows it and the rule above does not hold it - but the warp turns the view
                // around the camera and would swing it away until the next frame snaps it back. Hold what
                // moves under 20% of what the turn alone (the camera motion at infinite distance) moves.
                bool turn_still = false;
                if (flags & 128) {
                    turn_still = trusted && d > 1.0 / 4096.0 && turn_len >= 2.0 && length(screen_px) < 0.2 * turn_len && orbit_centre;
                    // Floor release (flag 512, debug test; alpha: the depth value of the orbit pivot, only set while turns
                    // are predicted around a measured pivot): the ground around the orbited character barely moves on
                    // screen either, but the warp moves it right now - not held when it is floor (two of three spots
                    // across agree) within half the pivot's distance.
                    if (turn_still && (flags & 512) && (flags & 16) && alpha > 0 && abs(1.0 / d - 1.0 / alpha) < 0.5 / alpha) {
                        const int k = max(2, int(round(float(rect.w) / 190.0)));
                        const int2 p = int2(id.xy);
                        const int votes = (floor_at(p, k) ? 1 : 0) + (floor_at(p - int2(k, 0), k) ? 1 : 0) + (floor_at(p + int2(k, 0), k) ? 1 : 0);
                        if (votes >= 2) turn_still = false;
                    }
                }
                // (near the camera - the first-person weapon and hands - the rig needs no proof. Stuck to the screen only
                // where the motion is known: XPAR's own estimate, filled in where the picture has no detail, can come out
                // near zero there - a dark floor or a flat wall was held, up to 29% of the screen in RE2)
                const bool attached = (flags & 1) && (flags & 16) &&
                                      ((!(flags & 64) && d > 1.0 / 64.0 && moving) ||
                                       (trusted && d > 1.0 / 4096.0 && cam_len >= 2.0 && length(screen_px) < 0.2 * cam_len &&
                                        (d > 1.0 / 64.0 || carried)) ||
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

// XPAR frame generation (option): the images between the previous game frame and this one, made as frame generation
// makes them and shown through the same path as the game's own generated images (own_warp with each image's depth,
// its camera taken part of the way back: generated_to_target). Whole pictures, no kinds of pixels: every point of
// this frame goes back along its motion to where it is at that moment (alpha: 0 the previous frame .. 1 this one; a
// point of this frame at p is at p + (1 - alpha) mv(p) then), every point of the previous frame forward along its way
// to this frame, the nearest winning each spot. Where both frames show the same surface there, their colours are
// blended by time; otherwise the nearer one is shown, and where neither lands, the farthest scenery found around.
// Motion: the game's motion vectors, those of the nearest surface in the 3x3 around (the anti-aliased edge of an
// object has the motion of what is behind it), or the picture's own where it is sure and they disagree (flag 2:
// shadows, glare, see-through things). The HUD (flag 4: the no-warp mask's HUD, 1) stays as this frame has it.
// clip_to_prev: the game's clipToPrevClip; wmat: its inverse (the previous frame's clip -> this frame's).
Texture2D<float4> fg_cur_t : register(t0);
Texture2D<float4> fg_prev_t : register(t1);
Texture2D<float> fg_depth_t : register(t2);
Texture2D<float> fg_prevz_t : register(t3);
Texture2D<float2> fg_mv_t : register(t4);
Texture2D<unorm float> fg_mask_t : register(t5);
Texture2D<float4> fg_flow_t : register(t6);
Texture2D<float2> fg_motion_t : register(t7);  // this frame's motion per render pixel (cs_fg_motion)
Texture2D<uint> fg_inv_t : register(t8);       // the previous frame's pixels, claimed by this frame's points there (cs_fg_inverse)
Texture2D<uint> fg_k1_t : register(t9);        // this frame's points at that moment (cs_fg_splat_cur)
Texture2D<uint> fg_k0_t : register(t10);       // the previous frame's points at that moment (cs_fg_splat_prev)
RWTexture2D<float2> fg_motion_u : register(u0);
RWTexture2D<uint> fg_keys_u : register(u0);
RWTexture2D<uint> fg_keys2_u : register(u1);
RWTexture2D<float4> fg_colour_u : register(u0);
RWTexture2D<float> fg_z_u : register(u0);
SamplerState fg_linear : register(s0);
int2 fg_px(float2 uv) { return int2(rect.xy) + int2(min(uint2(saturate(uv) * float2(rect.zw)), rect.zw - 1)); }
float2 fg_uv(int2 r) { return (float2(r - int2(rect.xy)) + 0.5) / float2(rect.zw); }
bool fg_inside(int2 r) { return all(r >= int2(rect.xy)) && all(r < int2(rect.xy + rect.zw)); }
float fg_z(int2 r) { return fg_depth_t.Load(int3(r, 0)); }
float fg_pz(int2 r) { return fg_prevz_t.Load(int3(r, 0)); }
// The game's motion vector at a render pixel (uv, towards the previous frame), or the picture's own.
float2 fg_game_mv(int2 r) {
    float2 m = fg_mv_t.Load(int3(min(uint2(r), mv_size - 1), 0)) * mv_scale;
    if (flags & 2) {
        uint fw, fh;
        fg_flow_t.GetDimensions(fw, fh);
        const float4 fl = fg_flow_t.Load(int3(min(uint2(r) / 2, uint2(fw, fh) - 1), 0));
        if (fl.z > 0.5) {
            const float2 seen = fl.xy * 2.0, game = m * float2(rect.zw);  // (render px)
            const float2 dd = seen - game;
            if (dot(dd, dd) > max(4.0, 0.09 * dot(game, game))) m = seen / float2(rect.zw);
        }
    }
    return m;
}
// The HUD and 3 output px around it (its glow and anti-aliased edges, which the mask leaves out: moved like scenery, they
// were left beside it as broken copies): taken as this frame has it.
bool fg_hud(float2 uv) {
    if (!(flags & 4)) return false;
    uint mw, mh;
    fg_mask_t.GetDimensions(mw, mh);
    const int2 c = int2(min(uint2(saturate(uv) * float2(mw, mh)), uint2(mw, mh) - 1));
    [unroll] for (int y = -3; y <= 3; y += 3)
        [unroll] for (int x = -3; x <= 3; x += 3)
            if (fg_mask_t.Load(int3(clamp(c + int2(x, y), int2(0, 0), int2(mw, mh) - 1), 0)) > 0.9) return true;
    return false;
}
// Claims: nearness << 22 | offset from the claimed render pixel to the claimant (11 bits each, +1024).
uint fg_key(float nk, int2 at, int2 from) {
    const int2 off = at - from;
    return (min(uint(max(nk, 1.0)), 1023u) << 22) | (uint(off.x + 1024) << 11) | uint(off.y + 1024);
}
int2 fg_claimant(uint key, int2 at) { return at - int2(int((key >> 11) & 2047u) - 1024, int(key & 2047u) - 1024); }

// This frame's motion per render pixel: that of the nearest surface in the 3x3 around. (u1: the claims on the previous
// frame, cleared.)
// (The HUD has none of its own and lends none: the game's motion vectors there are those of the scenery behind it.)
[numthreads(8, 8, 1)] void cs_fg_motion(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 r = int2(rect.xy + id.xy);
    fg_keys2_u[r] = 0;
    if (fg_hud(fg_uv(r))) { fg_motion_u[r] = float2(0, 0); return; }
    int2 from = r;
    float nearest = near_key(fg_z(r));
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x) {
            const int2 n = r + int2(x, y);
            if (!fg_inside(n) || fg_hud(fg_uv(n))) continue;
            const float k = near_key(fg_z(n));
            if (k > nearest) { nearest = k; from = n; }
        }
    fg_motion_u[r] = fg_game_mv(from);
}
// Every point of this frame claims where it was in the previous frame (3x3 render px, the nearest winning): what each
// pixel of the previous frame has become. (Not the HUD: it was where it is, and its motion vectors are the scenery's.)
[numthreads(8, 8, 1)] void cs_fg_inverse(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 r = int2(rect.xy + id.xy);
    if (fg_hud(fg_uv(r))) return;
    const float2 was = fg_uv(r) + fg_motion_t.Load(int3(r, 0));
    if (any(was < 0) || any(was > 1)) return;
    const int2 rw = fg_px(was);
    const float nk = near_key(fg_z(r));
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x) {
            const int2 c = rw + int2(x, y);
            if (!fg_inside(c) || any(abs(c - r) > 1000)) continue;
            uint ignored;
            InterlockedMax(fg_keys_u[c], fg_key(nk, c, r), ignored);
        }
}
// Where a pixel of the previous frame is in this frame: at the point of this frame that claimed it on the same surface,
// otherwise (covered in this frame) moved with the camera alone, at its own depth.
float2 fg_prev_to_cur(int2 rq) {
    const float2 q = fg_uv(rq);
    const uint key = fg_inv_t.Load(int3(rq, 0));
    if (key != 0 && abs(near_key(fg_pz(rq)) - float(key >> 22)) <= 8.0) return q - fg_motion_t.Load(int3(fg_claimant(key, rq), 0));
    const float4 f = mul(float4(q.x * 2 - 1, 1 - q.y * 2, fg_pz(rq), 1), wmat);
    return f.w > 1e-6 ? float2(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5) : q;
}
[numthreads(8, 8, 1)] void cs_fg_clear(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    fg_keys_u[id.xy] = 0;
    fg_keys2_u[id.xy] = 0;
}
// This frame's points at that moment (the HUD where it is, in front of everything).
[numthreads(8, 8, 1)] void cs_fg_splat_cur(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 r = int2(rect.xy + id.xy);
    const float2 p = fg_uv(r);
    const bool hud = fg_hud(p);
    const float2 then = hud ? p : p + (1.0 - alpha) * fg_motion_t.Load(int3(r, 0));
    if (any(then < 0) || any(then > 1)) return;
    const int2 rt = fg_px(then);
    const float nk = hud ? 1023.0 : min(near_key(fg_z(r)), 1022.0);
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x) {
            const int2 c = rt + int2(x, y);
            if (!fg_inside(c) || any(abs(c - r) > 1000)) continue;
            uint ignored;
            InterlockedMax(fg_keys_u[c], fg_key(nk, c, r), ignored);
        }
}
// ...and the previous frame's (not where this frame shows the HUD: it is there in both).
[numthreads(8, 8, 1)] void cs_fg_splat_prev(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 r = int2(rect.xy + id.xy);
    const float2 q = fg_uv(r);
    if (fg_hud(q)) return;
    const float2 then = q + alpha * (fg_prev_to_cur(r) - q);
    if (any(then < 0) || any(then > 1)) return;
    const int2 rt = fg_px(then);
    const float nk = min(near_key(fg_pz(r)), 1022.0);
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x) {
            const int2 c = rt + int2(x, y);
            if (!fg_inside(c) || any(abs(c - r) > 1000)) continue;
            uint ignored;
            InterlockedMax(fg_keys_u[c], fg_key(nk, c, r), ignored);
        }
}
// The point of this frame at x then (its claimant, one more step with the motion found at the spot on the same
// surface) and its depth...
bool fg_from_cur(float2 x, out float2 p, out float z) {
    p = x; z = 0;
    const int2 rx = fg_px(x);
    const uint key = fg_k1_t.Load(int3(rx, 0));
    if (key == 0) return false;
    const int2 c = fg_claimant(key, rx);
    z = fg_z(c);
    if ((key >> 22) == 1023u) { p = fg_uv(c) + (x - fg_uv(rx)); return true; }  // (the HUD: where it is)
    p = x - (1.0 - alpha) * fg_motion_t.Load(int3(c, 0));
    const int2 rp = fg_px(p);
    if (abs(near_key(fg_z(rp)) - near_key(z)) <= 2.0) p = x - (1.0 - alpha) * fg_motion_t.Load(int3(rp, 0));
    return all(p >= 0) && all(p <= 1);
}
// ...and of the previous frame (the same way along to this frame as its claimant's).
bool fg_from_prev(float2 x, out float2 q, out float z) {
    q = x; z = 0;
    const int2 rx = fg_px(x);
    const uint key = fg_k0_t.Load(int3(rx, 0));
    if (key == 0) return false;
    const int2 c = fg_claimant(key, rx);
    z = fg_pz(c);
    const float2 qc = fg_uv(c);
    q = x - alpha * (fg_prev_to_cur(c) - qc);
    return all(q >= 0) && all(q <= 1);
}
float4 fg_sample(bool prev, float2 uv) { return prev ? fg_prev_t.SampleLevel(fg_linear, uv, 0) : fg_cur_t.SampleLevel(fg_linear, uv, 0); }
// The picture at x then: both frames on the same surface (depth within ~19%) blended by time, otherwise the nearer;
// where neither lands, the farthest point found in the nearest ring around (up to 16 render px), continued.
bool fg_pick(float2 x, out float4 c, out float z) {
    c = 0; z = 0;
    float2 p, q;
    float z1, z0;
    const bool a = fg_from_cur(x, p, z1), b = fg_from_prev(x, q, z0);
    if (a && b) {
        const float k1 = near_key(z1), k0 = near_key(z0);
        if (abs(k1 - k0) <= 8.0) { c = lerp(fg_sample(true, q), fg_sample(false, p), saturate(alpha)); z = z1; }
        else if (k1 > k0) { c = fg_sample(false, p); z = z1; }
        else { c = fg_sample(true, q); z = z0; }
        return true;
    }
    if (a) { c = fg_sample(false, p); z = z1; return true; }
    if (b) { c = fg_sample(true, q); z = z0; return true; }
    float best = 1e9;
    [loop] for (int k = 1; k <= 16 && best > 1e8; ++k)
        [loop] for (int d = 0; d < 8; ++d) {
            const float2 o = float2(d == 0 || d == 1 || d == 7 ? 1 : d == 3 || d == 4 || d == 5 ? -1 : 0,
                                    d == 1 || d == 2 || d == 3 ? 1 : d == 5 || d == 6 || d == 7 ? -1 : 0);
            const float2 s = x + o * float(k) / float2(rect.zw);
            if (any(s < 0) || any(s > 1)) continue;
            if (fg_from_cur(s, p, z1) && near_key(z1) < best) { best = near_key(z1); c = fg_sample(false, p + (x - s)); z = z1; }
            if (fg_from_prev(s, q, z0) && near_key(z0) < best) { best = near_key(z0); c = fg_sample(true, q + (x - s)); z = z0; }
        }
    return best < 1e8;
}
[numthreads(8, 8, 1)] void cs_fg_resolve(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float2 x = (float2(id.xy) + 0.5) / float2(out_size);
    float4 c = fg_cur_t.Load(int3(id.xy, 0));
    float z;
    float4 picked;
    if (!fg_hud(x) && fg_pick(x, picked, z)) c = picked;
    fg_colour_u[id.xy] = float4(c.rgb, 1);
}
// ...and its depth, at render resolution (for the warp).
[numthreads(8, 8, 1)] void cs_fg_depth(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    float d = fg_depth_t.Load(int3(id.xy, 0));
    if (fg_inside(int2(id.xy))) {
        const float2 x = fg_uv(int2(id.xy));
        float4 c;
        float z;
        if (!fg_hud(x) && fg_pick(x, c, z)) d = z;
    }
    fg_z_u[id.xy] = d;
}

// Moving objects, once per game frame before anything else: one object, one way of moving. The motion analysis
// tells pixel by pixel what moves with the camera (attached: kept still and moved along its own on-screen
// motion) and what moves on its own (moved in the world, then warped with the camera); a third-person
// character is often both - torso attached, legs and arms on their own - and the two ways disagree by the
// camera's motion, tearing it apart. So what moves on its own and touches something attached at about the same
// distance becomes attached too: jumps of 64, 32, ... 1 render px between pixels within 15% of each other's
// distance spread it over the whole object; the result goes back into the analysis, so the mask holds it too.
// Holding whole characters (option, rect.y 1 - see Renderer::hold_characters): the same spreading, but only into spots
// that have stayed in place on screen lately (t3, the score of cs_still: swinging limbs do, grass, walls and floor the
// character runs past do not - a different bit of them there every frame, moving with the camera), and spots that have
// stayed in place for several frames are attached themselves (seeds).
Texture2D<float4> oj_object_t : register(t0);
Texture2D<float> oj_kind_t : register(t1);
Texture2D<float> oj_depth_t : register(t2);
Texture2D<float> oj_still_t : register(t3);
RWTexture2D<float> oj_kind_u : register(u0);
[numthreads(8, 8, 1)] void cs_obj_join_init(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    float k = oj_object_t.Load(int3(id.xy, 0)).w;
    if ((uint(rect.y) & 1) && oj_still_t.Load(int3(id.xy, 0)) >= 0.6) k = 2.0;
    oj_kind_u[id.xy] = k;
}
[numthreads(8, 8, 1)] void cs_obj_join(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    float k = oj_kind_t.Load(int3(id.xy, 0));
    if (abs(k - 1.0) < 0.5 && (!(uint(rect.y) & 1) || oj_still_t.Load(int3(id.xy, 0)) >= 0.3)) {
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
// How much each spot of the screen has stayed in place lately while the camera moved (render px; 0..1, a quarter of the way
// towards this frame's verdict every frame the camera moves at least 2 px there; unchanged when it does not). This
// frame's verdict: on screen under half of what the camera's motion would move it, and held the way the motion analysis
// holds (near the camera, or carried by the camera's moving, or at the centre an orbit turns around - not an object
// followed by turning), where the motion is known (XPAR's own motion: its trusted spots). A character jittering a few
// pixels each frame - the camera following with lag, limbs swinging - stays in place this way; scenery passes by.
// rect: the depth region; flags 1 scale known, 2 own motion (t2: its trust, out_size its size), 8 start over,
// 16 depth inverted.
Texture2D<float4> st_object_t : register(t0);
Texture2D<float> st_depth_t : register(t1);
Texture2D<float4> st_flow_t : register(t2);
RWTexture2D<float> st_score_u : register(u0);
[numthreads(8, 8, 1)] void cs_still(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 pr = int2(rect.xy + id.xy);
    float s = (flags & 8) ? 0.0 : st_score_u[pr];
    const float4 o = st_object_t.Load(int3(pr, 0));
    const float d = st_depth_t.Load(int3(pr, 0));
    const float2 uv = (float2(id.xy) + 0.5) / float2(rect.zw);
    const float4 prev = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, d, 1), clip_to_prev);
    const float4 far = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), clip_to_prev);
    if ((flags & 1) && (flags & 16) && prev.w > 0 && far.w > 0 && d > 1.0 / 4096.0) {
        const float2 cam_px = (float2(prev.x / prev.w * 0.5 + 0.5, 0.5 - prev.y / prev.w * 0.5) - uv) * float2(rect.zw);
        const float2 turn_px = (float2(far.x / far.w * 0.5 + 0.5, 0.5 - far.y / far.w * 0.5) - uv) * float2(rect.zw);
        const float cam_len = length(cam_px), turn_len = length(turn_px);
        if (cam_len >= 2.0) {
            bool trusted = true;
            if (flags & 2) trusted = st_flow_t.Load(int3(min(uint2((float2(pr) + 0.5) * float2(out_size) / float2(grid)), out_size - 1), 0)).z > 0.5;
            const bool rig = d > 1.0 / 64.0 || length(cam_px - turn_px) >= 0.5 * cam_len || cam_len < 0.5 * turn_len;
            const float2 screen_px = o.xy + cam_px;  // (o.xy: its own motion against the camera's, cs_analyze)
            const bool still = trusted && rig && any(o != 0) && length(screen_px) < 0.5 * cam_len;
            s = lerp(s, still ? 1.0 : 0.0, 0.25);
        }
    }
    st_score_u[pr] = s;
}
// The character/weapon by how much of the camera's turn each spot follows (Settings::follow_share, Renderer::follow_share):
// 1 the scenery, 0 what the camera carries (the third-person character it orbits or follows, the first-person weapon), in
// between what partly does (the ground around an orbited character). The least-squares fit of a spot's motion on screen
// against the turn's there over its recent frames, carried along with the spot (back along its motion to where it was a
// frame ago), this frame's added, capped by information, not time: a character on screen while the camera rests keeps its
// share until the camera turns again. The warp moves each spot by its share of the warp's motion (ow_lands): no
// held/not-held edge for the character to split along.
// Two shares (Settings::share_from_input, flag 32): how a spot follows the camera's turn in frames with input (I) and in
// frames without (X: a car's steering, a camera turning by itself) - each frame's whole turn, the game's own, goes to one
// of the two (flag 64: it had input), so no model of the input is needed (measured against the model's turn instead, the
// shares were as wrong as the model: Cyberpunk's guns warped). The warp applies each share to its part of the predicted
// turn (a car's interior follows the mouse-look but not the steering). Least squares on the two: sums s.I, I.I, I.X, s.X,
// X.X (s: the spot's motion). Without it, against the camera's turn alone (X none, the rest's share = the turn's).
// Either way a frame counts less the more it disagrees with what the spot has followed so far (a gun's own animation while
// turning, its bob while walking).
// A frame counts less where the motion is mostly across the turn (one share: something moving on its own); not at all where
// the motion is unknown (XPAR's own motion: its untrusted spots), far away (the sky) or without a turn.
// flags: 1 scale known, 2 own motion (t2 its trust, out_size its size), 4 against the camera's whole motion (the warp turns
// around an orbit), 8 start over, 16 depth inverted, 32 two shares (above). Units: 1080p px.
// The scenery's shares are then taken by distance alone (bands of depth, see cs_share_bins): measured spot by spot they came
// out in blotches where the motion estimate has few confident spots (RE2: 19-34%), and each blotch moved by its own share -
// the scenery rippled while turning. Per spot the measurement only tells what the camera carries: under half of its band's
// share of the input's turn.
static const float kShareMax = 2000.0;   // (20 frames of a 10 px turn)
static const float kSharePrior = 20.0;   // (no measurement: the scenery's 1)
static const int kShareBins = 24;        // bands: half octaves of (reversed) depth, 1/4096 to 1
int share_bin(float d) { return int(floor((log2(max(d, 1e-9)) + 12.0) * 2.0)); }  // (< 0: farther - the sky)
// The sums of a spot: s.I, I.I, I.X, s.X, X.X, and for the camera's movement s.T, T.T (and the depth they were measured
// at), as half floats in a uint4.
// The camera's movement (T: its motion at the spot minus its turn's - walking, a car driving) has a share of its own: the
// warp moves each spot by its shares of the turn and of the movement apart (ow_lands_by) - the movement's at most the
// turn's (see share_measure_at: only what the turn does not explain teaches it). With one share for both, measured
// on the turn alone, what follows the turn but not the movement - a car's interior while looking around - was moved by the
// car's driving as if it were the scenery flying past (torn apart while the mouse moved, Cyberpunk), and a gun with a
// share of 0.2 left from before it was drawn by a fifth of the walking's shift at its distance (torn while walking until a
// turn taught it). T = 0 against the camera's whole motion (flag 4: the warp turns around an orbit, the movement with it).
struct ShareSums { float sI, II, IX, sX, XX, sT, TT; };
uint4 share_pack(ShareSums a, float depth) {
    return uint4(f32tof16(a.sI) | (f32tof16(a.II) << 16), f32tof16(a.IX) | (f32tof16(depth) << 16),
                 f32tof16(a.sX) | (f32tof16(a.XX) << 16), f32tof16(a.sT) | (f32tof16(a.TT) << 16));
}
ShareSums share_unpack(uint4 v, out float depth) {
    ShareSums a;
    a.sI = f16tof32(v.x & 0xFFFF); a.II = f16tof32(v.x >> 16);
    a.IX = f16tof32(v.y & 0xFFFF); depth = abs(f16tof32(v.y >> 16));
    a.sX = f16tof32(v.z & 0xFFFF); a.XX = f16tof32(v.z >> 16);
    a.sT = f16tof32(v.w & 0xFFFF); a.TT = f16tof32(v.w >> 16);
    return a;
}
ShareSums share_scaled(ShareSums a, float k) {
    a.sI *= k; a.II *= k; a.IX *= k; a.sX *= k; a.XX *= k; a.sT *= k; a.TT *= k;
    return a;
}
// What the spot is (the depth's sign, see share_measure_at): not scenery - attached to the camera or moving on its own.
bool share_own(uint4 v) { return (v.y >> 31) != 0; }
// (the spread's weight between two spots: one of each kind - a character and the floor at its distance - counts as little
// as a spot at another distance)
float share_kind_w(bool a, bool b) { return a == b ? 1.0 : 0.02; }
// (capped by information: the turn's sums and the movement's each on their own)
ShareSums share_capped(ShareSums a) {
    const float info = a.II + a.XX, k = info > kShareMax ? kShareMax / info : 1.0;
    a.sI *= k; a.II *= k; a.IX *= k; a.sX *= k; a.XX *= k;
    const float kt = a.TT > kShareMax ? kShareMax / a.TT : 1.0;
    a.sT *= kt; a.TT *= kt;
    return a;
}
// The movement's share (the scenery's 1 unmeasured).
float share_moved(ShareSums a) { return (a.sT + kSharePrior) / (a.TT + kSharePrior); }
// The two shares from the sums, with a prior of `prior` towards the scenery's 1 for each (a spot or band not measured: 1)
// and - two shares - one towards the two being the same, a tenth of what is known: only frames that tell the input's turn
// from the rest set them apart (mouse-look alone, steering alone). Turning both ways at once - the mouse and a car's
// steering the same way - leaves the two parts in step, and the shares came out anywhere between them (0.3 to 1 across
// RE2's scenery in a replay with the input half of every turn).
float2 share_solve(ShareSums a, float prior, bool two) {
    const float same = two ? 0.1 * (a.II + a.XX) : 0.0;
    const float m00 = a.II + prior + same, m11 = a.XX + prior + same, m01 = a.IX - same;
    const float r0 = a.sI + prior, r1 = a.sX + prior;
    const float det = m00 * m11 - m01 * m01;
    if (det <= 1e-6) return float2(r0 / m00, r1 / m11);
    return float2((r0 * m11 - m01 * r1) / det, (m00 * r1 - m01 * r0) / det);
}
Texture2D<float4> sm_object_t : register(t0);
Texture2D<float> sm_depth_t : register(t1);
Texture2D<float4> sm_flow_t : register(t2);
Texture2D<uint4> sm_prev_t : register(t3);  // (the sums as they were, share_pack)
RWTexture2D<uint4> sm_acc_u : register(u0);
// This frame's sums per band (fixed point, 4 per px^2, signed - kept as uint, two's complement: [s * 24 + band] for the
// sums s.I, I.I, I.X, s.X, X.X - every second spot each way, each at most a 16 px turn's worth), gathered per group first.
// [120, 240): the bands' sums so far (floats, same order); [240, 264) their share of the input's turn, [264, 288) of the rest.
// [288, 300): diagnostics (flag 128, recorded diagnostics: share.csv) - near-camera spots (closer than 64x the near plane),
// every fourth each way: 0 with motion, 1 measured, 2 sum of turn^2, 3 of motion.turn, 4 and 5 the same weighted, 6 the
// weights x1000, 7 the spots' shares x250, 8 motion lengths x10, 9 turn lengths x10, 10 information, 11 motion over twice the turn
RWByteAddressBuffer sm_bins_u : register(u1);
groupshared int sm_bins_gs[5 * kShareBins];
groupshared int sm_diag_gs[12];
float share_band_so_far(int b) {  // (a band not measured yet: the scenery's 1)
    return asfloat(sm_bins_u.Load((120 + 24 + uint(b)) * 4)) > 0 ? asfloat(sm_bins_u.Load((240 + uint(b)) * 4)) : 1.0;
}
// Half resolution: one spot per 2x2 block of render pixels (cell c: the render pixel 2c of the rect, its sample), the
// shares put back together at render resolution by distance (cs_share_up) - the shares are smooth over ~3% of the screen
// anyway; at render resolution the passes cost ~0.9 ms per game frame at 1080p render (RE2).
uint2 share_cells() { return (rect.zw + 1) / 2; }
int2 share_sample(uint2 cell) { return int2(rect.xy) + int2(min(cell * 2, rect.zw - 1)); }
// (bilinear; p in render px of the rect. Only from where the same surface was - within ~40% of this spot's distance: a spot
// just uncovered from behind something nearer - the wall beside a gun or a character moving past it - took that one's sums
// and stayed partly with it, a trail beside its outline; with none, it starts over)
ShareSums sm_prev_at(float2 p, float depth, out float own) {
    const float2 q = clamp((p - 0.5) * 0.5, float2(0, 0), float2(share_cells()) - 1.001);
    const int2 i = int2(floor(q));
    const float2 f = q - floor(q);
    const float l = log2(max(depth, 1e-9));
    ShareSums sum = (ShareSums)0;
    float w = 0;
    own = 0;
    [unroll] for (int k = 0; k < 4; ++k) {
        const int2 o = int2(k & 1, k >> 1);
        float dz;
        const uint4 packed = sm_prev_t.Load(int3(i + o, 0));
        const ShareSums v = share_unpack(packed, dz);
        const float wk = (o.x ? f.x : 1.0 - f.x) * (o.y ? f.y : 1.0 - f.y);
        if (abs(log2(max(dz, 1e-9)) - l) < 0.5) {
            sum.sI += wk * v.sI; sum.II += wk * v.II; sum.IX += wk * v.IX; sum.sX += wk * v.sX; sum.XX += wk * v.XX;
            sum.sT += wk * v.sT; sum.TT += wk * v.TT;
            w += wk;
            own += share_own(packed) ? wk : 0.0;
        }
    }
    own = w > 1e-4 ? own / w : 0.0;
    if (w > 1e-4) return share_scaled(sum, 1.0 / w);
    return (ShareSums)0;
}
void share_measure_at(uint2 id);
[numthreads(8, 8, 1)] void cs_share_measure(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    for (uint k = gi; k < uint(5 * kShareBins); k += 64) sm_bins_gs[k] = 0;
    if (gi < 12) sm_diag_gs[gi] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (all(id.xy < share_cells())) share_measure_at(id.xy);
    GroupMemoryBarrierWithGroupSync();
    uint ignored;
    for (uint k2 = gi; k2 < uint(5 * kShareBins); k2 += 64)
        if (sm_bins_gs[k2] != 0) sm_bins_u.InterlockedAdd(k2 * 4, uint(sm_bins_gs[k2]), ignored);
    if ((flags & 128) && gi < 12 && sm_diag_gs[gi] != 0) sm_bins_u.InterlockedAdd((288 + gi) * 4, uint(sm_diag_gs[gi]), ignored);
}
void share_measure_at(uint2 id) {
    const int2 pr = share_sample(id);
    const float4 o = sm_object_t.Load(int3(pr, 0));
    const float d = sm_depth_t.Load(int3(pr, 0));
    const float2 uv = (float2(pr - int2(rect.xy)) + 0.5) / float2(rect.zw);
    const float4 prev = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, d, 1), clip_to_prev);
    const float4 far = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), clip_to_prev);
    ShareSums acc = (ShareSums)0;
    float was_own = 0;
    bool told = false;  // (this frame tells scenery from the rest: a turn to measure)
    const bool diag = (flags & 128) && (flags & 16) && d > 1.0 / 64.0 && ((id.x | id.y) & 1) == 0 && prev.w > 0 && any(o != 0);
    if (diag) InterlockedAdd(sm_diag_gs[0], 1);
    if (prev.w > 0 && any(o != 0)) {
        const float2 cam_px = (float2(prev.x / prev.w * 0.5 + 0.5, 0.5 - prev.y / prev.w * 0.5) - uv) * float2(rect.zw);
        const float2 screen_px = o.xy + cam_px;  // (o.xy: its own motion against the camera's, cs_analyze)
        if (!(flags & 8)) acc = sm_prev_at(float2(pr - int2(rect.xy)) + 0.5 + screen_px, d, was_own);
        if ((flags & 1) && (flags & 16) && far.w > 0 && d > 1.0 / 4096.0) {
            bool trusted = true;
            if (flags & 2) trusted = sm_flow_t.Load(int3(min(uint2((float2(pr) + 0.5) * float2(out_size) / float2(grid)), out_size - 1), 0)).z > 0.5;
            const float2 turn_px = (float2(far.x / far.w * 0.5 + 0.5, 0.5 - far.y / far.w * 0.5) - uv) * float2(rect.zw);
            // (two shares: the frame's whole turn is the input's - flag 64 - or the rest's)
            const float2 whole_px = (flags & 4) ? cam_px : turn_px;
            const bool rest = (flags & 32) && !(flags & 64);
            const float unit = 1080.0 / float(rect.w);
            const float2 r = rest ? float2(0, 0) : whole_px * unit, x = rest ? whole_px * unit : float2(0, 0), s = screen_px * unit;
            const float rr = dot(r, r), xx = dot(x, x), all_r = rr + xx;
            told = all_r > 0.25;
            // The camera's movement here (its motion minus its turn's). Its share is learned only from the movement the turn
            // does not explain: what the turn's known shares leave of the spot's motion, against the part of the movement
            // across the frame's turn. An orbiting camera moves as it turns - the movement along the turn, collinear with it:
            // the two can not be told apart there, and learned freely the pair drifted apart (Elden Ring: the ground near the
            // camera at a turn's share of 0.6-0.9 instead of ~0.3, its movement's share in blotches of 0); such frames teach
            // it nothing, the turn's share keeps what it absorbs of the movement, as before. Walking or driving straight -
            // the movement alone - teaches it (a gun, a car's interior: 0).
            const float2 t = (flags & 4) ? float2(0, 0) : (cam_px - turn_px) * unit;
            const float2 turn_w = r + x;
            const float ww = dot(turn_w, turn_w);
            const float2 t_across = ww > 0.25 ? t - turn_w * (dot(t, turn_w) / ww) : t;
            const float ta = dot(t_across, t_across), moved = share_moved(acc);
            if (trusted && ta > 0.25) {
                // (counting less the more it disagrees with what the spot has followed so far, as the turn's)
                const float2 known_t = share_solve(acc, kSharePrior, (flags & 32) != 0);
                const float2 rest_t = s - r * known_t.x - x * ((flags & 32) ? known_t.y : known_t.x);
                const float2 miss_t = rest_t - t_across * moved;
                const float ct = lerp(1.0, 1.0 / (1.0 + dot(miss_t, miss_t) / (0.25 * ta + 4.0)), acc.TT / (acc.TT + 200.0));
                acc.sT += ct * dot(rest_t, t_across); acc.TT += ct * ta;
            }
            if (trusted && all_r > 0.25) {
                // (how far this frame is from what the spot has followed so far, against the turns' size - counting less the
                // more the spot already knows: a gun bobbing 8-15 px while walking against a head-bob turn of 1.5 px taught it
                // to follow 1.5-2.3 times the turn, Cyberpunk; a gun's own animation while turning likewise)
                const float2 known = share_solve(acc, kSharePrior, (flags & 32) != 0);
                const float2 miss = s - r * known.x - x * ((flags & 32) ? known.y : known.x);
                const float known_info = acc.II + acc.XX;
                float c = lerp(1.0, 1.0 / (1.0 + dot(miss, miss) / (0.25 * all_r + 4.0)), known_info / (known_info + 200.0));
                if (!(flags & 32)) {
                    const float2 across = s - r * (dot(s, r) / rr);
                    c *= rr / (rr + 4.0 * dot(across, across));
                }
                ShareSums f;
                f.sI = dot(s, r); f.II = rr; f.IX = dot(r, x); f.sX = dot(s, x); f.XX = xx;
                acc.sI += c * f.sI; acc.II += c * f.II; acc.IX += c * f.IX; acc.sX += c * f.sX; acc.XX += c * f.XX;
                if (diag) {
                    const float cap = min(all_r, 1024.0), kq = cap / all_r, sd = clamp((f.sI + f.sX) * kq, -3.0 * cap, 3.0 * cap);
                    InterlockedAdd(sm_diag_gs[1], 1);
                    InterlockedAdd(sm_diag_gs[2], int(round(cap)));
                    InterlockedAdd(sm_diag_gs[3], int(round(sd)));
                    InterlockedAdd(sm_diag_gs[4], int(round(c * cap)));
                    InterlockedAdd(sm_diag_gs[5], int(round(c * sd)));
                    InterlockedAdd(sm_diag_gs[6], int(round(c * 1000.0)));
                    InterlockedAdd(sm_diag_gs[8], int(round(min(length(s), 200.0) * 10.0)));
                    InterlockedAdd(sm_diag_gs[9], int(round(sqrt(cap) * 10.0)));
                    if (dot(s, s) > 4.0 * all_r) InterlockedAdd(sm_diag_gs[11], 1);
                }
                // (into its band only if it follows at least half of the band's share of the input's turn so far: what the
                // camera carries - the character filling most of its own band - would take the band down with it, and the
                // grass around it too. And only scenery - what this frame's motion analysis finds the camera explains, o.w 0:
                // at a distance with nothing else, the band learned from what the camera carries itself - a first-person gun,
                // a car's interior - swung between 0.05 and 1 within a second, and the spots above half of it took its value:
                // the gun a patchwork of 0 and 0.13-0.5, the car's interior torn apart by distance, Cyberpunk)
                const int b = share_bin(d);
                const ShareSums spot = share_capped(acc);
                if (b >= 0 && b < kShareBins && o.w < 0.5 &&
                    share_solve(spot, kSharePrior, (flags & 32) != 0).x >= 0.5 * share_band_so_far(b)) {
                    // (each at most a 16 px turn's worth; the motion at most 3 turns: the products bounded, in 32 bits)
                    const float k = c * min(1.0, 256.0 / all_r);
                    const float lim = 3.0 * all_r * k;
                    InterlockedAdd(sm_bins_gs[0 * kShareBins + b], int(round(clamp(f.sI * k, -lim, lim) * 4.0)));
                    InterlockedAdd(sm_bins_gs[1 * kShareBins + b], int(round(f.II * k * 4.0)));
                    InterlockedAdd(sm_bins_gs[2 * kShareBins + b], int(round(f.IX * k * 4.0)));
                    InterlockedAdd(sm_bins_gs[3 * kShareBins + b], int(round(clamp(f.sX * k, -lim, lim) * 4.0)));
                    InterlockedAdd(sm_bins_gs[4 * kShareBins + b], int(round(f.XX * k * 4.0)));
                }
            }
        }
    } else if (o.w > 1.5) {
        acc.II = acc.XX = 0.5 * kShareMax;  // (behind where the camera was a frame ago: carried by it, see cs_analyze)
        acc.TT = kShareMax;
    }
    const float info = acc.II + acc.XX;
    acc = share_capped(acc);
    if (diag) {
        InterlockedAdd(sm_diag_gs[7], int(round(clamp(share_solve(acc, kSharePrior, (flags & 32) != 0).x, -4.0, 4.0) * 250.0)));
        InterlockedAdd(sm_diag_gs[10], int(round(min(info, kShareMax))));
    }
    // What the spot is, in the depth's sign: what this frame's analysis finds attached to the camera or moving on its own
    // (o.w; the bands leave it out too), or what it was where this frame can not tell - no turn: everything stands still
    // against the camera's motion. The spread keeps the two apart (share_kind_w): spread by distance alone, a character took
    // the floor's share where the floor beside it is at its distance - RE2's orbiting camera, the floor at Claire's distance
    // following 0.23 of the turn: her head against the far wall carried, from the waist down 0.24, torn at the border.
    const bool own = o.w > 0.5 || (!told && was_own > 0.5);
    sm_acc_u[id] = share_pack(acc, own ? -d : d);
}
// The bands, once per game frame: this frame's sums carried on (capped by information, as the spots'), their shares, this
// frame's cleared (layout: see sm_bins_u). flags 8: start over.
static const float kShareBinMax = 1.0e8, kShareBinPrior = 400.0;
RWByteAddressBuffer sbn_u : register(u0);
[numthreads(32, 1, 1)] void cs_share_bins(uint3 id : SV_DispatchThreadID) {
    if (id.x >= uint(kShareBins)) return;
    const uint b = id.x;
    if ((flags & 128) && b < 12) sbn_u.Store((288 + b) * 4, 0);  // (the diagnostics, copied out before: see follow_share)
    float v[5];
    float info = 0;
    [unroll] for (uint k = 0; k < 5; ++k) {
        const float now = float(int(sbn_u.Load((k * 24 + b) * 4))) / 4.0;
        v[k] = ((flags & 8) ? 0.0 : asfloat(sbn_u.Load((120 + k * 24 + b) * 4))) + now;
        sbn_u.Store((k * 24 + b) * 4, 0);
    }
    info = v[1] + v[4];
    const float scale = info > kShareBinMax ? kShareBinMax / info : 1.0;
    [unroll] for (uint k2 = 0; k2 < 5; ++k2) { v[k2] *= scale; sbn_u.Store((120 + k2 * 24 + b) * 4, asuint(v[k2])); }
    ShareSums a;
    a.sI = v[0]; a.II = v[1]; a.IX = v[2]; a.sX = v[3]; a.XX = v[4];
    const float2 sh = saturate(share_solve(a, kShareBinPrior, (flags & 32) != 0));
    sbn_u.Store((240 + b) * 4, asuint(sh.x));
    sbn_u.Store((264 + b) * 4, asuint(sh.y));
}
// ...spread over its surroundings at its own distance (two passes over the cells, across then down: 17 taps groups_x cells apart,
// Gaussian; neighbours nearer or farther count less and less - a fifth by ~20%): a character's spots come to one value,
// and spots the motion estimate missed take their surroundings'. Across (cs_share_blur): u0 s.I, I.I, I.X; u1 s.X, X.X,
// s.T, T.T. Down (cs_share_final) gives the shares (u1: the input's turn, the rest's, the movement's): the turn's their
// band's (u0, interpolated by distance; the sky's 1) where the spot follows most of it, none where it follows under a
// tenth (carried by the camera), in between what it measured (share_kept); the movement's the spot's own, 0 under 0.15,
// 1 over 0.45 (what the camera
// carries or the scenery).
// flags 4: against the camera's whole motion (the movement's share = the turn's); 16: depth inverted (otherwise nothing is
// measured: 1); 32 two shares (otherwise the rest's = the input's).
Texture2D<uint4> sb_packed_t : register(t0);
Texture2D<float> sb_depth_t : register(t1);
RWTexture2D<float4> sb_acc_u : register(u0);
RWTexture2D<float4> sb_acc2_u : register(u1);
[numthreads(8, 8, 1)] void cs_share_blur(uint3 id : SV_DispatchThreadID) {
    const uint2 cells = share_cells();
    if (any(id.xy >= cells)) return;
    float d0;
    const uint4 packed0 = sb_packed_t.Load(int3(id.xy, 0));
    share_unpack(packed0, d0);
    const bool own0 = share_own(packed0);
    const float l0 = log2(max(d0, 1e-9));
    ShareSums sum = (ShareSums)0;
    float wsum = 0;
    [unroll] for (int k = -8; k <= 8; ++k) {
        const int2 q = clamp(int2(id.xy) + int2(int(groups_x), 0) * k, int2(0, 0), int2(cells) - 1);
        float dz;
        const uint4 packed = sb_packed_t.Load(int3(q, 0));
        const ShareSums v = share_unpack(packed, dz);
        const float rel = log2(max(dz, 1e-9)) - l0;
        const float g = exp(-float(k * k) / 32.0 - rel * rel * 25.0) * share_kind_w(own0, share_own(packed));
        sum.sI += g * v.sI; sum.II += g * v.II; sum.IX += g * v.IX; sum.sX += g * v.sX; sum.XX += g * v.XX;
        sum.sT += g * v.sT; sum.TT += g * v.TT;
        wsum += g;
    }
    sum = share_scaled(sum, 1.0 / wsum);
    sb_acc_u[id.xy] = float4(sum.sI, sum.II, sum.IX, own0 ? -d0 : d0);  // (w: the cell's depth and kind, for the pass down)
    sb_acc2_u[id.xy] = float4(sum.sX, sum.XX, sum.sT, sum.TT);
}
// (a cell's depth and kind in one: the depth, negative where it is not scenery - share_own)
bool share_own_w(float w) { return (asuint(w) >> 31) != 0; }
Texture2D<float4> sf_acc_t : register(t0);
Texture2D<float4> sf_acc2_t : register(t2);
RWByteAddressBuffer sf_bins_u : register(u0);
RWTexture2D<float4> sf_half_u : register(u1);   // (per cell: held for the input's turn, for the rest's, for the movement; its depth)
RWTexture2D<unorm float4> sf_share_u : register(u1);  // (cs_share_up: the shares at render resolution)
float2 share_band(float d) {
    const float x = (log2(max(d, 1e-9)) + 12.0) * 2.0 - 0.5;  // (band centres at b + 0.5)
    if (x < -0.5) return float2(1, 1);
    const int b0 = clamp(int(floor(x)), 0, kShareBins - 1), b1 = min(b0 + 1, kShareBins - 1);
    const float2 v0 = float2(asfloat(sf_bins_u.Load((240 + uint(b0)) * 4)), asfloat(sf_bins_u.Load((264 + uint(b0)) * 4)));
    const float2 v1 = float2(asfloat(sf_bins_u.Load((240 + uint(b1)) * 4)), asfloat(sf_bins_u.Load((264 + uint(b1)) * 4)));
    return lerp(v0, v1, saturate(x - floor(x)));
}
float share_kept(float r) { return saturate(min(2.0 * (r - 0.1), max(r, 2.0 * r - 0.6))); }
[numthreads(8, 8, 1)] void cs_share_final(uint3 id : SV_DispatchThreadID) {
    const uint2 cells = share_cells();
    if (any(id.xy >= cells)) return;
    const float w0 = sf_acc_t.Load(int3(id.xy, 0)).w, d0 = abs(w0);
    const bool own0 = share_own_w(w0);
    const float l0 = log2(max(d0, 1e-9));
    float4 sum = 0, sum2 = 0;
    float wsum = 0, w_at = 0, w_char = 0;
    [unroll] for (int k = -8; k <= 8; ++k) {
        const int2 q = clamp(int2(id.xy) + int2(0, int(groups_x)) * k, int2(0, 0), int2(cells) - 1);
        const float4 t = sf_acc_t.Load(int3(q, 0));
        const float rel = log2(max(abs(t.w), 1e-9)) - l0;
        const float g_at = exp(-float(k * k) / 32.0 - rel * rel * 25.0);  // (at its distance, any kind)
        w_at += g_at;
        w_char += share_own_w(t.w) ? g_at : 0.0;
        const float g = g_at * share_kind_w(own0, share_own_w(t.w));
        sum += g * t;
        sum2 += g * sf_acc2_t.Load(int3(q, 0));
        wsum += g;
    }
    ShareSums a;
    a.sI = sum.x / wsum; a.II = sum.y / wsum; a.IX = sum.z / wsum; a.sX = sum2.x / wsum; a.XX = sum2.y / wsum;
    a.sT = sum2.z / wsum; a.TT = sum2.w / wsum;
    const float2 spot = share_solve(a, kSharePrior, (flags & 32) != 0);
    float2 band = (flags & 16) ? share_band(d0) : float2(1, 1);
    if (!(flags & 32)) band.y = band.x;
    // (two shares: the rest's also held where its own says so - a car's interior follows the mouse-look, its band's 1, but
    // not the steering: held by the input's alone, the rest's took its band's 1 too and the steering moved the interior. Held
    // by its own alone, a character - frames without input rare, the rest's share near the prior's 1 - followed 0.3 of the
    // rest and of the movement, RE2: so held where either holds it)
    // (in between, what the spot measured: under 0.1 of its band's share none - carried, the prior's pull leaves a carried
    // thing a little over 0; from 0.2 to 0.6 its own share; over 0.8 its band's - the scenery, its own measurement only
    // ripples; no steeper than twice the measured anywhere. A ramp from 0 at 0.15 to the band at 0.45 stretched what one
    // thing measures across itself - a gun's sway, 0.2 here and 0.35 there - into 0.07 against 0.74 of the turn, and a
    // firmer cut between 0.35 and 0.5 put RE2's Claire on the edge: carried above the waist, the floor's below. Against the
    // real previous frame, replayed: Stalker 2's rifle 0.041 -> 0.035, Cyberpunk 2077's car 0.022 -> 0.020)
    // (flag 256: a character - what the analysis finds attached or moving on its own, or a cell mostly among such at its
    // distance - carried as one, none of the turn: measured spot by spot it came out in patches, 0 here and 0.3 there, and
    // the patches split apart whenever the camera turned. What it does beyond being carried - legs that go with the floor,
    // a gun's sway - is its own motion, carried on smoothly (cs_share_resid))
    const bool character = (flags & 256) && (own0 || w_char > 0.6 * w_at);
    const float kept = character ? 0.0 : share_kept(spot.x / max(band.x, 0.05));
    const float kept_rest = character ? 0.0 : (flags & 32) ? min(kept, share_kept(spot.y / max(band.y, 0.05))) : kept;
    sf_half_u[id.xy] = float4(kept, kept_rest, (flags & 16) ? smoothstep(0.15, 0.45, share_moved(a)) : 1.0, w0);
}
// ...at render resolution: each pixel from the four cells around it, by their distance against its own (as the spread; none
// at its distance - a thin thing between cells: the nearest in distance), the bands by its own distance.
Texture2D<float4> su_half_t : register(t0);
Texture2D<float4> su_object_t : register(t3);  // (this frame's analysis: w the pixel's kind, as the cells')
[numthreads(8, 8, 1)] void cs_share_up(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 pr = int2(rect.xy + id.xy);
    const float d0 = sb_depth_t.Load(int3(pr, 0));
    const float l0 = log2(max(d0, 1e-9));
    const bool own0 = su_object_t.Load(int3(pr, 0)).w > 0.5;
    const uint2 cells = share_cells();
    const float2 u = clamp(float2(id.xy) * 0.5, float2(0, 0), float2(cells) - 1.001);
    const int2 i = int2(floor(u));
    const float2 f = u - floor(u);
    float3 kept = 0, nearest = 0;
    float wsum = 0, best = 1e9;
    [unroll] for (int k = 0; k < 4; ++k) {
        const int2 o = int2(k & 1, k >> 1);
        const float4 v = su_half_t.Load(int3(min(i + o, int2(cells) - 1), 0));
        const float rel = abs(log2(max(abs(v.w), 1e-9)) - l0);
        const float g = (o.x ? f.x : 1.0 - f.x) * (o.y ? f.y : 1.0 - f.y) * exp(-rel * rel * 25.0) * share_kind_w(own0, share_own_w(v.w));
        kept += g * v.xyz; wsum += g;
        if (rel < best) { best = rel; nearest = v.xyz; }
    }
    kept = wsum > 1e-3 ? kept / wsum : nearest;
    float2 band = (flags & 16) ? share_band(d0) : float2(1, 1);
    if (!(flags & 32)) band.y = band.x;
    const float2 turned = band * kept.xy;
    // (the movement moves a spot by the turn's share - the input's, or the rest's with two shares - as before, at most: its
    // own share only ever lowers it, where walking or driving showed it does not follow)
    const float moved = (flags & 4) ? turned.y : (flags & 16) ? min(turned.y, kept.z) : turned.y;
    // (w: the pixel's nearness, 32 steps per doubling of the distance as ow_near, halved - for the outline - plus 0.5 where
    // it follows under half of its band's share: carried by the camera, not scenery that follows little because the camera
    // orbits - see cs_mask)
    const float x = (flags & 16) ? d0 : 1.0 - d0;
    sf_share_u[pr] = float4(turned, moved, 0.5 * saturate((log2(max(x, 1e-12)) + 30.0) / 32.0) + (kept.x < 0.5 ? 0.5 : 0.0));
}
// ...and what each spot of a character does on its own (cs_share_resid, after cs_share_up): its motion this frame less what
// the shares the warp applies (kPShare: the input's turn, the rest's, the camera's movement) explain - render px towards the
// previous frame, per cell; z 1 where the analysis finds it not scenery (attached to the camera or moving on its own), w its
// depth. The warp carries it on past the frame (ow_own). Held by their share alone, a character walking or turning its body -
// RE2's Claire, Armored Core VI's machine - stayed where the game frame had it until the next one: the parts of it the shares
// moved apart from the rest, the gap where it had been filled with pieces of it.
Texture2D<float4> sr_object_t : register(t0);
Texture2D<float> sr_depth_t : register(t1);
Texture2D<float4> sr_share_t : register(t2);
RWTexture2D<float4> sr_resid_u : register(u0);
[numthreads(8, 8, 1)] void cs_share_resid(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= share_cells())) return;
    const int2 pr = share_sample(id.xy);
    const float4 o = sr_object_t.Load(int3(pr, 0));
    const float d = sr_depth_t.Load(int3(pr, 0));
    // (flag 512: only where the share carries it, too - kPShare.w over 0.5: XPAR's own motion estimate marks scenery
    // edges moving during fast turns, and carried on they were shredded - RE2's desk)
    const bool own = o.w > 0.5 && (!(flags & 512) || sr_share_t.Load(int3(pr, 0)).w >= 0.5);
    float2 resid = 0;
    if (own && (flags & 1) && (flags & 16)) {
        const float2 uv = (float2(pr - int2(rect.xy)) + 0.5) / float2(rect.zw);
        const float4 prev = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, d, 1), clip_to_prev);
        const float4 far = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), clip_to_prev);
        if (prev.w > 0 && far.w > 0) {
            const float2 cam_px = (float2(prev.x / prev.w * 0.5 + 0.5, 0.5 - prev.y / prev.w * 0.5) - uv) * float2(rect.zw);
            const float2 turn_px = (float2(far.x / far.w * 0.5 + 0.5, 0.5 - far.y / far.w * 0.5) - uv) * float2(rect.zw);
            const float2 whole = (flags & 4) ? cam_px : turn_px;
            const float2 t = (flags & 4) ? float2(0, 0) : cam_px - turn_px;
            const float4 sh = sr_share_t.Load(int3(pr, 0));
            const bool rest = (flags & 32) && !(flags & 64);
            resid = (o.xy + cam_px) - whole * (rest ? sh.y : sh.x) - t * sh.z;
        }
    }
    sr_resid_u[id.xy] = float4(resid, own ? 1.0 : 0.0, d);
}
// ...then spread over the character's cells at about its own distance (cs_resid_blur: across, flag 1 down; 17 taps groups_x
// cells apart, Gaussian; only cells that are characters themselves): measured cell by cell, a body's parts were carried on
// by slightly different motions and pulled apart.
Texture2D<float4> rb_in_t : register(t0);
RWTexture2D<float4> rb_out_u : register(u0);
[numthreads(8, 8, 1)] void cs_resid_blur(uint3 id : SV_DispatchThreadID) {
    const uint2 cells = share_cells();
    if (any(id.xy >= cells)) return;
    const float4 c0 = rb_in_t.Load(int3(id.xy, 0));
    const int2 dir = (flags & 1) ? int2(0, 1) : int2(1, 0);
    const float l0 = log2(max(c0.w, 1e-9));
    float2 sum = 0;
    float wsum = 0, wall = 0;
    [unroll] for (int k = -8; k <= 8; ++k) {
        const int2 q = clamp(int2(id.xy) + dir * k * int(max(groups_x, 1u)), int2(0, 0), int2(cells) - 1);
        const float4 v = rb_in_t.Load(int3(q, 0));
        const float rel = log2(max(v.w, 1e-9)) - l0;
        const float g = exp(-float(k * k) / 32.0 - ((flags & 2) ? 0.0 : rel * rel * 25.0));  // (flag 2: at any distance)
        wall += g;
        if (v.z < 0.5) continue;
        sum += g * v.xy;
        wsum += g;
    }
    // (a cell the analysis took for scenery, among a character's at its distance - mostly theirs - is the character's too:
    // flat cloth it could not tell from the camera's motion this frame stayed behind while the rest of the body was carried
    // on, and the cracks filled with what is behind it - dark specks through Claire's jacket)
    if (c0.z < 0.5 && wsum <= 0.6 * wall) { rb_out_u[id.xy] = c0; return; }
    rb_out_u[id.xy] = float4(sum / max(wsum, 1e-6), 1.0, c0.w);
}
// ...and per rendered pixel (cs_own_up): how far it is carried on per game frame, render px - from the four cells around it,
// by their distance against its own, of a character only, faded in by how much of what is around at its distance is one
// (smoothstep 0.25-0.75 of the weight), the nearest in distance where none matches (a fold, a strand of hair, an arm before
// the body: carried on and left behind side by side, neighbours tore the body into specks). Once per game frame, for the
// warp's every refresh to read with one sample (ow_own): worked out there, every rendered pixel's four cells cost the warp a
// quarter more GPU time.
Texture2D<float4> ou_resid_t : register(t0);
Texture2D<float> ou_depth_t : register(t1);
RWTexture2D<float2> ou_move_u : register(u0);
[numthreads(8, 8, 1)] void cs_own_up(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const int2 pr = int2(rect.xy + id.xy);
    const uint2 cells = share_cells();
    const float2 q = clamp(float2(id.xy) * 0.5, float2(0, 0), float2(cells) - 1.001);
    const int2 i = int2(floor(q));
    const float2 fq = q - floor(q);
    const float l0 = log2(max(ou_depth_t.Load(int3(pr, 0)), 1e-9));
    float2 sum = 0, nearest = 0;
    float w_own = 0, w_all = 0, best = 1e9, nearest_own = 0;
    [unroll] for (int k = 0; k < 4; ++k) {
        const int2 o = int2(k & 1, k >> 1);
        const float4 v = ou_resid_t.Load(int3(min(i + o, int2(cells) - 1), 0));
        const float rel = log2(max(v.w, 1e-9)) - l0;
        const float g = (o.x ? fq.x : 1.0 - fq.x) * (o.y ? fq.y : 1.0 - fq.y) * exp(-rel * rel * 25.0);
        w_all += g;
        if (v.z > 0.5) { sum += g * v.xy; w_own += g; }
        if (abs(rel) < best) { best = abs(rel); nearest = v.xy; nearest_own = v.z; }
    }
    float2 move = 0;
    if (w_all < 1e-3) move = -nearest * nearest_own;
    else move = -(sum / max(w_own, 1e-6)) * smoothstep(0.25, 0.75, w_own / w_all);
    ou_move_u[pr] = move;
}
// ...and holes in it (cs_hold_fill): a spot not held, enclosed by held pixels at its own distance (within 15%) in at least
// 7 of 8 directions within 3% of the screen height (the frame's edge counts, a character cut off by it), before anything
// nearer or farther comes in the way, is part of what is held. Flat skin or cloth XPAR's own motion estimate cannot
// follow (RE2) left the character full of small holes that warped with the scenery: it fell apart into specks. The
// floor at its feet or a wall beside it touch it on one side only; what is seen between an arm and the body is farther.
// Into a kind texture (u0, 2 held), back into the analysis by cs_obj_join_final. rect.x: the step (render px), rect.y the
// steps.
static const int2 kHoldDirs[8] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(1, -1), int2(-1, 1), int2(-1, -1)};
[numthreads(8, 8, 1)] void cs_hold_fill(uint3 id : SV_DispatchThreadID) {
    // (one thread per 2x2 block - holes are blobs, not single pixels: a quarter of the walks; the block's pixels at its
    // first one's distance are filled with it)
    const int2 b = int2(id.xy) * 2;
    if (any(b >= int2(grid))) return;
    const float4 o = oj_object_t.Load(int3(b, 0));
    const float w0 = view_w(oj_depth_t.Load(int3(b, 0)));
    bool fill = false;
    if (w0 > 0) {
        const int step = max(1, int(rect.x)), steps = int(rect.y);
        uint enclosed = 0, open = 0;
        [loop] for (uint i = 0; i < 8 && open < 2; ++i) {  // (two open directions: not enclosed - most spots stop here)
            const int2 dir = kHoldDirs[i];
            bool hit = false;
            [loop] for (int n = 1; n <= steps; ++n) {
                const int2 q = b + dir * (n * step);
                if (any(q < 0) || any(q >= int2(grid))) { hit = true; break; }
                const float wq = view_w(oj_depth_t.Load(int3(q, 0)));
                if (wq <= 0 || abs(wq - w0) > 0.15 * w0) break;
                if (oj_object_t.Load(int3(q, 0)).w > 1.5) { hit = true; break; }
            }
            if (hit) ++enclosed; else ++open;
        }
        fill = enclosed >= 7;
    }
    [unroll] for (int k = 0; k < 4; ++k) {
        const int2 p = b + int2(k & 1, k >> 1);
        if (any(p >= int2(grid))) continue;
        float kind = oj_object_t.Load(int3(p, 0)).w;
        if (fill && kind < 1.5) {
            const float wp = view_w(oj_depth_t.Load(int3(p, 0)));
            if (wp > 0 && abs(wp - w0) <= 0.15 * w0) kind = 2.0;
        }
        oj_kind_u[p] = kind;
    }
}
// Growing what is held along depth (hold_characters), only where XPAR's own motion estimate is not trusted - where the
// motion cannot tell (dark hair, a black gun stock: RE2 has no motion vectors): held spreads, one cell of half the
// render grid per pass, into such spots whose distance continues from a held neighbour's (within 15%: the same object -
// what is seen behind it is farther), never into the floor (it gets nearer evenly down the screen: the ground under the
// feet). As many passes as a tenth of the held area's size (rect.x passes; the area counted here, read back a few frames
// later). Games with motion vectors never come here: every pixel's motion is known.
// Cells of 2 * groups_x render px (out_size; groups_x 1: the half grid): x 1 held, 0 not held where the estimate is not
// trusted somewhere in the cell (growable), -1 not held and trusted; y depth (of the cell's held pixel if any, else its
// nearest). Not grown from the HUD (flag 32, t1: the last no-warp mask, 1 on the HUD): it is held too, but its depth is
// the scenery's behind it - grown from, it spread over the grass (Dark Souls).
Texture2D<float> og_mask_t : register(t1);
Texture2D<float4> og_object_t : register(t0);
Texture2D<float> og_depth_t : register(t2);
Texture2D<float4> og_flow_t : register(t4);
Texture2D<float2> og_half_t : register(t5);
RWTexture2D<float2> og_half_u : register(u0);
RWByteAddressBuffer og_count_u : register(u1);
groupshared uint og_n;
[numthreads(8, 8, 1)] void cs_grow_init(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0) og_n = 0;
    GroupMemoryBarrierWithGroupSync();
    if (all(id.xy < out_size)) {
        const uint f = max(groups_x, 1u), s = 2 * f;
        float held = 0, d_held = 0, d_near = (flags & 16) ? 0.0 : 1.0;
        uint n_held = 0;
        [loop] for (uint k = 0; k < s * s; ++k) {
            const int2 p = min(int2(id.xy * s + uint2(k % s, k / s)), int2(grid) - 1);
            const float d = og_depth_t.Load(int3(p, 0));
            d_near = (flags & 16) ? max(d_near, d) : min(d_near, d);
            bool hud = false;
            if (flags & 32) {
                uint mw, mh;
                og_mask_t.GetDimensions(mw, mh);
                hud = og_mask_t.Load(int3(min(uint2((float2(p) + 0.5) * float2(mw, mh) / float2(grid)), uint2(mw, mh) - 1), 0)) > 0.9;
            }
            if (!hud && og_object_t.Load(int3(p, 0)).w > 1.5) { ++n_held; d_held = d; }
        }
        // (a seed: most of the cell held - a character's held parts come in patches; single blades of grass held next to
        // it, grown from, came out as blobs of held grass)
        held = 4 * n_held >= 3 * s * s ? 1.0 : 0.0;
        bool open = false;
        const int2 half_last = max(int2(grid) / 2, int2(1, 1)) - 1;
        [loop] for (uint j = 0; j < f * f && !open; ++j)
            open = og_flow_t.Load(int3(min(int2(id.xy * f + uint2(j % f, j / f)), half_last), 0)).z < 0.5;
        og_half_u[id.xy] = float2(held > 0 ? 1.0 : open ? 0.0 : -1.0, held > 0 ? d_held : d_near);
        if (held > 0 && f == 1) { uint ignored; InterlockedAdd(og_n, 1, ignored); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && og_n) { uint ignored; og_count_u.InterlockedAdd(0, og_n, ignored); }
}
[numthreads(1, 1, 1)] void cs_grow_clear(uint3 id : SV_DispatchThreadID) { og_count_u.Store(0, 0); }
// (floor: the depth value grows down the screen, evenly - as floor_at, on the half grid)
bool grow_floor(int2 p) {
    const int k = max(1, int(round(float(out_size.y) / 190.0)));
    const int2 hi = int2(out_size) - 1;
    const float a = og_half_t.Load(int3(clamp(p - int2(0, k), int2(0, 0), hi), 0)).y;
    const float c = og_half_t.Load(int3(p, 0)).y;
    const float b = og_half_t.Load(int3(clamp(p + int2(0, k), int2(0, 0), hi), 0)).y;
    const float slope = (b - a) / float(2 * k) * float(out_size.y);
    return (flags & 16) && slope > 0 && slope > 0.6 * max(c, 1e-6) && abs(b - 2 * c + a) < 0.25 * abs(b - a);
}
[numthreads(8, 8, 1)] void cs_grow(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    float2 v = og_half_t.Load(int3(id.xy, 0));
    if (abs(v.x) < 0.5) {
        const float w = view_w(v.y);
        if (w > 0 && !grow_floor(int2(id.xy))) {
            [unroll] for (int k = 0; k < 8; ++k) {
                const int2 q = clamp(int2(id.xy) + kHoldDirs[k], int2(0, 0), int2(out_size) - 1);
                const float2 n = og_half_t.Load(int3(q, 0));
                if (n.x > 0.5 && abs(view_w(n.y) - w) <= 0.15 * w) v.x = 1;
            }
        }
    }
    og_half_u[id.xy] = v;
}
// Back into the analysis: a pixel of a grown cell at its distance (within 15%), where the motion is not trusted.
RWTexture2D<float4> og_object_u : register(u0);
[numthreads(8, 8, 1)] void cs_grow_final(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    float4 o = og_object_u[id.xy];
    if (o.w > 1.5) return;
    const int2 h = min(int2(id.xy) / int(2 * max(groups_x, 1u)), int2(out_size) - 1);
    const float2 v = og_half_t.Load(int3(h, 0));
    if (v.x < 0.5 || og_flow_t.Load(int3(min(int2(id.xy) / 2, max(int2(grid) / 2, int2(1, 1)) - 1), 0)).z >= 0.5) return;
    const float w = view_w(og_depth_t.Load(int3(id.xy, 0))), wc = view_w(v.y);
    if (w > 0 && wc > 0 && abs(w - wc) <= 0.15 * wc) { o.w = 2.0; og_object_u[id.xy] = o; }
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
    if (!held && (flags & 2) && os_mask_t.Load(int3(min(uint2(uv * float2(out_size)), out_size - 1), 0)) > 0.885) return;
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
    return dot(seen(t.Load(int3(clamp(p, int2(0, 0), int2(out_size) - 1), 0)).rgb), float3(0.299, 0.587, 0.114));
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
    const float3 delta = abs(seen(hud_current_t.Load(int3(p, 0)).rgb) - seen(hud_previous_t.Load(int3(p, 0)).rgb));
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
    const float3 scenery = abs(seen(hud_current_t.Load(int3(p, 0)).rgb) - seen(hud_previous_t.Load(int3(q, 0)).rgb));
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
// The classification of each pixel, made once (cs_hud_count) for the passes after it: kind, weight - in one uint, exactly:
// kind 1 (the only one with a weight, 0.1-0.45) as the weight's bits; any other kind as the top bit and the kind, weight 0.
// (32 bits per pixel instead of 64: 32 MB less at 4K)
RWTexture2D<uint> hud_kind_u : register(u0);  // (cs_hud_count)
Texture2D<uint> hud_kind_t : register(t5);     // (cs_hud, cs_hud_world)
uint hud_kind_pack(uint kind, float weight) { return kind == 1 ? asuint(weight) : (0x80000000u | kind); }
uint hud_kind_of(uint2 id, out float weight) {
    const uint v = hud_kind_t.Load(int3(id, 0));
    if (v == 0 || (v & 0x80000000u)) { weight = 0; return v & 0xFFu; }  // (0: never written - kind 0, as the float pair read)
    weight = asfloat(v);
    return 1;
}
[numthreads(8, 8, 1)] void cs_hud_count(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size) || !(flags & 2)) return;
    float weight;
    const uint kind = hud_classify(id.xy, weight);
    hud_kind_u[id.xy] = hud_kind_pack(kind, weight);
    uint ignored;
    if (kind == 1) hud_counts_u.InterlockedAdd(0, 1, ignored);
    else if (kind == 2) hud_counts_u.InterlockedAdd(4, 1, ignored);
}
[numthreads(8, 8, 1)] void cs_hud(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size) || !(flags & 2)) return;
    float weight;
    const uint kind = hud_kind_of(id.xy, weight);
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
// ...and what it finds is HUD only where it joins HUD the learned map confirms (cs_hud's score, kept alongside in this mode):
// tiles of 8 output px - 1 holding some confirmed HUD, 0.5 holding found HUD that has stayed put (cs_hud_tiles) - the
// confirmation spreading through touching tiles of the latter (cs_hud_spread; see cs_mask). Glare, bloom, lit signs and
// foliage - drawn after the upscaler, so the upscaler-output detector takes them for HUD - never stay put on the screen
// as the camera moves, so none of them is confirmed; text, icons and gauges are, and a panel's plain inside joins the text
// in it. Stayed put: found while the camera moved past 1/12 of the screen width (the persistence, cs_hud_world) - in a
// lit facade the found tiles touch one another, and the confirmation of the crosshair ran through all of them (a fixed
// reach instead held every light within 1/30 of the screen height of it); the minimap's and a gauge's tiles have
// stayed put (93-98%), a facade's lights mostly not (18%).
Texture2D<float> tiles_learned_t : register(t0);
Texture2D<float> tiles_found_t : register(t1);
Texture2D<float> tiles_world_t : register(t2);  // (persistence and world evidence, packed: see pack_world)
RWTexture2D<unorm float> tiles_u : register(u0);
[numthreads(8, 8, 1)] void cs_hud_tiles(uint3 id : SV_DispatchThreadID) {
    const uint2 tiles = (out_size + 7) / 8;
    if (any(id.xy >= tiles)) return;
    bool confirmed = false, steady = false;
    [loop] for (uint y = 0; y < 8; ++y)
        [loop] for (uint x = 0; x < 8; ++x) {
            const uint2 p = id.xy * 8 + uint2(x, y);
            if (any(p >= out_size)) continue;
            confirmed = confirmed || tiles_learned_t.Load(int3(p, 0)) > 0.6;
            steady = steady || (tiles_found_t.Load(int3(p, 0)) > 0.6 && floor(tiles_world_t.Load(int3(p, 0))) >= float(out_size.x) / 12.0);
        }
    tiles_u[id.xy] = confirmed ? 1.0 : steady ? 0.5 : 0.0;
}
// The spreading, in blocks of 16x16 tiles (32 steps: all of a block), the blocks shifted by rect.xy tiles on every other
// pass so that it crosses their borders (in place: each block reads and writes only its own tiles).
groupshared float spread_gs[16 * 16];
[numthreads(16, 16, 1)] void cs_hud_spread(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uint gi : SV_GroupIndex) {
    uint tw, th;
    tiles_u.GetDimensions(tw, th);
    const int2 t = int2(gid.xy * 16 + tid.xy) - int2(rect.xy);
    const bool inside = all(t >= 0) && all(t < int2(tw, th));
    spread_gs[gi] = inside ? tiles_u[t] : 0.0;
    GroupMemoryBarrierWithGroupSync();
    [loop] for (int i = 0; i < 32; ++i) {
        float v = spread_gs[gi];
        if (v > 0.25 && v < 0.75) {
            [unroll] for (int y = -1; y <= 1; ++y)
                [unroll] for (int x = -1; x <= 1; ++x) {
                    const int2 n = int2(tid.xy) + int2(x, y);
                    if (all(n >= 0) && all(n < 16) && spread_gs[n.y * 16 + n.x] > 0.75) v = 1.0;
                }
        }
        GroupMemoryBarrierWithGroupSync();
        spread_gs[gi] = v;
        GroupMemoryBarrierWithGroupSync();
    }
    if (inside) tiles_u[t] = spread_gs[gi];
}
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
    const float3 c = seen(hud_current_t.Load(int3(id, 0)).rgb);
    const int2 base = int2(floor(from - 0.5));
    float best = 1e9;
    [unroll] for (int i = 0; i < 4; ++i) {
        const int2 q = clamp(base + int2(i & 1, i >> 1), int2(0, 0), int2(out_size) - 1);
        const float3 d = abs(c - seen(hud_previous_t.Load(int3(q, 0)).rgb));
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
    const uint kind = hud_kind_of(id.xy, weight);
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
// HUD learning (score_hud_probe): the probe - where the final picture differs from the copy taken at the candidate point -
// in tiles of 16x16 output px (mv_size), the share of each that differs.
Texture2D<unorm float> pr_probe_t : register(t0);
RWTexture2D<float> pr_tiles_u : register(u0);
[numthreads(8, 8, 1)] void cs_probe_tiles(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= mv_size)) return;
    float s = 0, n = 0;
    [loop] for (uint k = 0; k < 256; ++k) {
        const uint2 p = id.xy * 16 + uint2(k % 16, k / 16);
        if (all(p < out_size)) { s += pr_probe_t.Load(int3(p, 0)); n += 1; }
    }
    pr_tiles_u[id.xy] = n > 0 ? s / n : 0;
}
// ...and against the previous frame's tiles of the same candidate, where the camera's motion moves a tile by at least a
// tile and a half (closer, the two can not be told apart): the overlap at the same spot (what stays on the screen - the
// HUD) and where the scenery came from (what moves with it). u1: [0] the share that differs (all tiles), [1]/[2] min/max at
// the same spot, [3]/[4] min/max where the scenery came from (x256).
Texture2D<float> ps_now_t : register(t0);
Texture2D<float> ps_prev_t : register(t1);
Texture2D<float> ps_depth_t : register(t2);
RWByteAddressBuffer ps_sums_u : register(u1);
[numthreads(1, 1, 1)] void cs_probe_clear(uint3 id : SV_DispatchThreadID) {
    ps_sums_u.Store4(0, uint4(0, 0, 0, 0));
    ps_sums_u.Store4(16, uint4(0, 0, 0, 0));
}
[numthreads(8, 8, 1)] void cs_probe_score(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= mv_size)) return;
    const float h = ps_now_t.Load(int3(id.xy, 0));
    uint ignored;
    ps_sums_u.InterlockedAdd(0, uint(h * 256.0), ignored);
    const float2 uv = (float2(id.xy) * 16.0 + 8.0) / float2(out_size);
    const int2 pr = int2(rect.xy) + int2(min(uint2(saturate(uv) * float2(rect.zw)), rect.zw - 1));
    const float4 pv = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, ps_depth_t.Load(int3(pr, 0)), 1), clip_to_prev);
    if (pv.w <= 0) return;
    const float2 from = float2(pv.x / pv.w * 0.5 + 0.5, 0.5 - pv.y / pv.w * 0.5);
    const float2 shift = (from - uv) * float2(out_size) / 16.0;
    if (length(shift) < 1.5) return;
    const int2 q = int2(round(float2(id.xy) + shift));
    if (any(q < 0) || any(q >= int2(mv_size))) return;
    const float f = ps_prev_t.Load(int3(id.xy, 0)), w = ps_prev_t.Load(int3(q, 0));
    ps_sums_u.InterlockedAdd(4, uint(min(h, f) * 256.0), ignored);
    ps_sums_u.InterlockedAdd(8, uint(max(h, f) * 256.0), ignored);
    ps_sums_u.InterlockedAdd(12, uint(min(h, w) * 256.0), ignored);
    ps_sums_u.InterlockedAdd(16, uint(max(h, w) * 256.0), ignored);
}

// HUD from the game's HUD layer, without a HUD-less picture (the add-on takes the layer where the game blends it onto the
// picture - Cyberpunk 2077 draws its whole HUD into one: see ui_layer.hpp): exactly where the layer is not transparent,
// every frame - nothing guessed (lights, glare, foliage behind the crosshair are not in it).
Texture2D<float4> hl_layer_t : register(t0);
RWTexture2D<float> hl_score_u : register(u0);
[numthreads(8, 8, 1)] void cs_layer_hud(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    hl_score_u[id.xy] = hl_layer_t.Load(int3(id.xy, 0)).a > 0.02 ? 1.0 : 0.0;
}
// Is that layer the game's HUD (see Renderer::layer_trust)? Everything of a HUD - fixed or anchored in the world - is drawn
// over the finished picture, after the upscaler: where the layer clearly covers, the frame differs from the upscaler's
// output (cs_scene_hud's verdict, t1). Every second pixel each way: [0] covered, [4] covered and differing (a pixel
// next to one that differs counts: the verdict misses anti-aliased edges).
Texture2D<float4> lc_layer_t : register(t0);
Texture2D<float> lc_found_t : register(t1);
RWByteAddressBuffer lc_counts_u : register(u0);
groupshared uint lc_n[2];
[numthreads(1, 1, 1)] void cs_layer_check_clear(uint3 id : SV_DispatchThreadID) { lc_counts_u.Store2(0, uint2(0, 0)); }
[numthreads(8, 8, 1)] void cs_layer_check(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0) { lc_n[0] = 0; lc_n[1] = 0; }
    GroupMemoryBarrierWithGroupSync();
    const int2 p = int2(id.xy * 2);
    if (all(p < int2(out_size)) && lc_layer_t.Load(int3(p, 0)).a > 0.5) {
        uint ignored;
        InterlockedAdd(lc_n[0], 1, ignored);
        bool found = false;
        [unroll] for (int k = 0; k < 5; ++k) {
            const int2 q = clamp(p + int2(k == 1 ? 1 : k == 2 ? -1 : 0, k == 3 ? 1 : k == 4 ? -1 : 0), int2(0, 0), int2(out_size) - 1);
            found = found || lc_found_t.Load(int3(q, 0)) > 0.5;
        }
        if (found) InterlockedAdd(lc_n[1], 1, ignored);
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && lc_n[0]) { uint ignored; lc_counts_u.InterlockedAdd(0, lc_n[0], ignored); lc_counts_u.InterlockedAdd(4, lc_n[1], ignored); }
}

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

// Sight reticles (option): a reticle the game draws without depth or motion vectors of its own (a red dot on a
// sight's glass) has the scenery's behind it, so the warp moves it with the scenery and every game frame snaps it
// back onto the gun - it skips while turning. What is on the glass (reticle, its glow, reflections) is treated as
// a layer fixed to the gun and added on top of the scenery seen through it: estimated per game frame inside the
// sight's window (enclosed by the held character/weapon), and the warp moves only the scenery under it (the frame
// minus the layer) and adds the layer back where the gun shows it. Nothing is held: no mask edges.
// Sizes are shares of the screen height (the values that worked at 4K, scaled): coarse cells of groups_x render px
// (0.55% of the height; mv_size: their grid).
static const int kRetTrack = 8, kRetTrackN = (2 * 8 + 1) * (2 * 8 + 1);  // (cs_ret_align's shifts)
static const int2 kRetDirs[8] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(1, -1), int2(-1, 1), int2(-1, -1)};
// Per cell: the held pixels' mean on-screen motion to the previous frame (output px), how many there are, and the nearest
// one's nearness (32 steps per doubling of the distance, as the share's: flag 16 reversed depth).
Texture2D<float4> rc_object_t : register(t0);
Texture2D<float> rc_depth_t : register(t1);
RWTexture2D<float4> rc_cells_u : register(u0);
[numthreads(8, 8, 1)] void cs_ret_cells(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= mv_size)) return;
    float2 sum = 0;
    float n = 0, nearest = 0;
    [loop] for (uint y = 0; y < groups_x; ++y)
        [loop] for (uint x = 0; x < groups_x; ++x) {
            const uint2 p = id.xy * groups_x + uint2(x, y);
            if (any(p >= rect.zw)) continue;
            const int2 pr = int2(rect.xy + p);
            const float4 o = rc_object_t.Load(int3(pr, 0));
            if (o.w < 1.5) continue;
            const float2 uv = (float2(p) + 0.5) / float2(rect.zw);
            const float d = rc_depth_t.Load(int3(pr, 0));
            nearest = max(nearest, saturate((log2(max((flags & 16) ? d : 1.0 - d, 1e-12)) + 30.0) / 32.0));
            const float4 pv = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, d, 1), clip_to_prev);
            const float2 cam = pv.w > 0 ? float2(pv.x / pv.w * 0.5 + 0.5, 0.5 - pv.y / pv.w * 0.5) - uv : float2(0, 0);
            sum += (o.xy / float2(rect.zw) + cam) * float2(out_size);  // its own motion plus the camera's (see cs_analyze)
            n += 1;
        }
    rc_cells_u[id.xy] = float4(n > 0 ? sum / n : float2(0, 0), n, nearest);
}
// Per cell: enclosed by held cells (z 1), the gun's motion there (xy: the mean of the 8 held cells met) and how near the
// frame around it is (w: the farthest of the 8 cells met, each by its nearest pixel).
// threshold: the reach in cells.
Texture2D<float4> re_cells_t : register(t0);
RWTexture2D<float4> re_encl_u : register(u0);
RWByteAddressBuffer re_buf_u : register(u1);  // (the window cells counted: see cs_ret_args)
[numthreads(8, 8, 1)] void cs_ret_enclose(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= mv_size)) return;
    float4 r = 0;
    if (re_cells_t.Load(int3(id.xy, 0)).z < 0.5) {
        const int reach = int(threshold);
        float2 g = 0;
        float frame = 1.0;
        uint hits = 0;
        [loop] for (uint i = 0; i < 8 && hits == i; ++i) {
            [loop] for (int k = 1; k <= reach; ++k) {
                const int2 q = int2(id.xy) + kRetDirs[i] * k;
                if (any(q < 0) || any(q >= int2(mv_size))) break;
                const float4 c = re_cells_t.Load(int3(q, 0));
                if (c.z > 0.5) { g += c.xy; frame = min(frame, c.w); ++hits; break; }
            }
        }
        if (hits == 8) {
            r = float4(g / 8.0, 1, frame);
            uint ignored;
            re_buf_u.InterlockedAdd(uint(kRetTrackN + 3) * 4, 1, ignored);
        }
    }
    re_encl_u[id.xy] = r;
}
// Where the reticle went: it does not move with the sight's frame (projected to infinity, it follows where the gun
// points, not the frame's sway and bob - measured at 4K: 0-4 px while the frame moved up to 11). The previous layer
// is lined up with this frame: for each of 17 x 17 shifts within +-0.37% of the screen height (+-8 px at 4K), the
// layer's weight times this frame's brightness where it would be now, summed (fixed point, one sum per shift, and
// the weight in the last); cs_ret_pick takes the best, refined to a fraction of a step, and cs_ret_shift hands it to
// the cells (xy: the motion to the previous frame).
float ret_track_step() { return float(out_size.y) / 2160.0; }  // output px per shift step
SamplerState ra_linear : register(s0);
Texture2D<float4> ra_cur_t : register(t0);
Texture2D<float4> ra_hist_t : register(t1);
RWByteAddressBuffer ra_buf_u : register(u1);
groupshared uint ra_sums[kRetTrackN + 1];
groupshared uint ra_any;
[numthreads(8, 8, 1)] void cs_ret_align(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    // every second pixel each way, where the previous layer has something - a group without any stops here (the layer
    // is a sight's glass: nearly all of the screen has none, and each group cleared and summed its 290 counters anyway)
    float w = 0;
    if (all(id.xy < out_size) && ((id.x | id.y) & 1) == 0) {
        const float4 l = ra_hist_t.Load(int3(id.xy, 0));
        w = max(max(l.r, l.g), l.b);
    }
    if (gi == 0) ra_any = 0;
    GroupMemoryBarrierWithGroupSync();
    if (w > 0.05) InterlockedOr(ra_any, 1u);
    GroupMemoryBarrierWithGroupSync();
    if (ra_any == 0) return;
    for (uint k = gi; k <= uint(kRetTrackN); k += 64) ra_sums[k] = 0;
    GroupMemoryBarrierWithGroupSync();
    {
        if (w > 0.05) {
            uint ignored;
            InterlockedAdd(ra_sums[kRetTrackN], uint(w * 1024.0), ignored);
            [loop] for (int dy = -kRetTrack; dy <= kRetTrack; ++dy)
                [loop] for (int dx = -kRetTrack; dx <= kRetTrack; ++dx) {
                    const float2 q = (float2(id.xy) + 0.5 + float2(dx, dy) * ret_track_step()) / float2(out_size);
                    const float v = dot(seen(ra_cur_t.SampleLevel(ra_linear, q, 0).rgb), float3(0.299, 0.587, 0.114));
                    InterlockedAdd(ra_sums[(dy + kRetTrack) * (2 * kRetTrack + 1) + dx + kRetTrack], uint(w * saturate(v) * 1024.0), ignored);
                }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k2 = gi; k2 <= uint(kRetTrackN); k2 += 64) {
        uint ignored;
        if (ra_sums[k2] != 0) ra_buf_u.InterlockedAdd(k2 * 4, ra_sums[k2], ignored);
    }
}
// The two passes over the whole screen (cs_ret_align, cs_ret_layer) only while a sight's window has been seen in the last 3
// game frames (the layer's two textures take turns: both written once more with nothing in them, then left so): their
// dispatches (u0, at 0 and 16) from the window cells cs_ret_enclose counted (u1 at kRetTrackN + 3; + 4: the frames
// seen, a bit each). flags 4: run them anyway (just switched on - the textures may hold an old layer).
RWByteAddressBuffer ry_args_u : register(u0);
RWByteAddressBuffer ry_buf_u : register(u1);
[numthreads(1, 1, 1)] void cs_ret_args(uint3 id : SV_DispatchThreadID) {
    const uint seen = ((ry_buf_u.Load(uint(kRetTrackN + 4) * 4) << 1) | (ry_buf_u.Load(uint(kRetTrackN + 3) * 4) > 0 ? 1u : 0u)) & 7u;
    ry_buf_u.Store(uint(kRetTrackN + 4) * 4, (flags & 4) ? 7u : seen);
    ry_buf_u.Store(uint(kRetTrackN + 3) * 4, 0);
    const uint3 groups = (seen != 0 || (flags & 4)) ? uint3((out_size + 7) / 8, 1) : uint3(0, 0, 0);
    ry_args_u.Store3(0, groups);
    ry_args_u.Store3(16, groups);
}
RWByteAddressBuffer rp_buf_u : register(u0);
[numthreads(1, 1, 1)] void cs_ret_pick(uint3 id : SV_DispatchThreadID) {
    const int n = 2 * kRetTrack + 1;
    uint best = 0;
    int at = (kRetTrackN - 1) / 2;
    [loop] for (int k = 0; k < kRetTrackN; ++k) {
        const uint v = rp_buf_u.Load(uint(k) * 4);
        if (v > best) { best = v; at = k; }
    }
    float2 shift = 0;
    // (a layer to line up: ~256 px of it at 4K, every fourth one sampled - fewer at lower resolutions)
    const float step = ret_track_step();
    const bool enough = float(rp_buf_u.Load(uint(kRetTrackN) * 4)) > max(64.0 * step * step, 8.0) * 1024.0;
    if (enough) {
        const int bx = at % n, by = at / n;
        shift = float2(bx - kRetTrack, by - kRetTrack);
        // a fraction of a pixel from the neighbours (parabola through three)
        if (bx > 0 && bx < n - 1) {
            const float l = rp_buf_u.Load(uint(at - 1) * 4), c = best, r = rp_buf_u.Load(uint(at + 1) * 4);
            const float den = l - 2 * c + r;
            if (den < 0) shift.x += clamp(0.5 * (l - r) / den, -0.5, 0.5);
        }
        if (by > 0 && by < n - 1) {
            const float u = rp_buf_u.Load(uint(at - n) * 4), c = best, d = rp_buf_u.Load(uint(at + n) * 4);
            const float den = u - 2 * c + d;
            if (den < 0) shift.y += clamp(0.5 * (u - d) / den, -0.5, 0.5);
        }
    }
    rp_buf_u.Store2(uint(kRetTrackN + 1) * 4, asuint(shift * step));
    [loop] for (int k2 = 0; k2 <= kRetTrackN; ++k2) rp_buf_u.Store(uint(k2) * 4, 0);
}
RWTexture2D<float4> rs_encl_u : register(u0);
RWByteAddressBuffer rs_buf_u : register(u1);
[numthreads(8, 8, 1)] void cs_ret_shift(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= mv_size)) return;
    const float2 shift = asfloat(rs_buf_u.Load2(uint(kRetTrackN + 1) * 4));
    float4 e = rs_encl_u[id.xy];
    e.xy = -shift;  // (the motion to the previous frame)
    rs_encl_u[id.xy] = e;
}
// The layer per output pixel (rgb; a: how well it has agreed lately), in this game frame's screen space. t0/t1:
// this game frame's picture and the previous one's (both HUD-less when the previous one is), t2 depth, t3 the held
// character/weapon (render px, widened), t4 the cells above, t5 the previous frame's layer. u0: the layer - the next
// frame's history, and what the warp reads (the two take turns: the one written last is the frame's). Flag 2: no
// history yet.
// The residual - this frame minus the previous one where the scenery came from - is the layer plus what that
// prediction got wrong (scenery just uncovered, blur, grain). The layer, carried along with the gun, takes in a
// residual that agrees with it, and one that does not the less the longer it has agreed: a new reticle is learned
// within a few frames, a one-frame error barely touches a learned one. Not where the scenery came from clearly more
// layer than this pixel has (taking out the previous frame's reticle there, slightly off, left a copy of it one
// scenery motion away - the same offset frame after frame in a steady turn, so it agreed and was learned: ghost
// rings); the reticle's own pixels keep learning over its glow. Light added only (reticles and their glow).
Texture2D<float4> rl_cur_t : register(t0);
Texture2D<float4> rl_prev_t : register(t1);
Texture2D<float> rl_depth_t : register(t2);
Texture2D<unorm float> rl_att_t : register(t3);
Texture2D<float4> rl_encl_t : register(t4);
Texture2D<float4> rl_hist_t : register(t5);
RWTexture2D<float4> rl_layer_u : register(u0);
// (u1: per cell, the last game frame this texture held a layer there - mv_scale: asuint, x the game frame this texture was
// last written in, y this one. A group of pixels whose cells - and their neighbours, as the test per pixel below - are
// neither in a sight's window nor hold anything from that last time has nothing to do: all of the screen but a window.)
RWTexture2D<uint> rl_dirty_u : register(u1);
SamplerState rl_linear : register(s0);
groupshared uint rl_needed;
groupshared uint rl_marks[4];  // (the group's cells - 2 x 2 at most - that got a layer this frame: marked once each)
void rl_store(uint2 pid, int2 cell_first);
[numthreads(8, 8, 1)] void cs_ret_layer(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {
    if (gi == 0) { rl_needed = 0; rl_marks[0] = rl_marks[1] = rl_marks[2] = rl_marks[3] = 0; }
    GroupMemoryBarrierWithGroupSync();
    {
        // (a cell each, up to 8 x 8 of them)
        uint cw, ch;
        rl_encl_t.GetDimensions(cw, ch);
        const float2 to_render = float2(rect.zw) / float2(out_size);
        const int2 c0 = max(int2(float2(gid.xy * 8) * to_render) / int(groups_x) - 1, int2(0, 0));
        const int2 c1 = min(int2((float2(gid.xy * 8) + 8.0) * to_render) / int(groups_x) + 1, int2(cw, ch) - 1);
        const int2 q = c0 + int2(gi & 7, gi >> 3);
        if (all(q <= c1) && (rl_encl_t.Load(int3(q, 0)).z > 0.5 || rl_dirty_u[q] >= asuint(mv_scale.x))) rl_needed = 1;
    }
    GroupMemoryBarrierWithGroupSync();
    if (rl_needed == 0) return;
    const int2 cell_first = int2(float2(gid.xy * 8) * float2(rect.zw) / float2(out_size)) / int(groups_x);
    rl_store(id.xy, cell_first);
    GroupMemoryBarrierWithGroupSync();
    if (gi < 4 && rl_marks[gi] != 0) InterlockedMax(rl_dirty_u[cell_first + int2(gi & 1, gi >> 1)], asuint(mv_scale.y));
}
void rl_store(uint2 pid, int2 cell_first) {
    const uint3 id = uint3(pid, 0);
    if (any(id.xy >= out_size)) return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(out_size);
    const int2 pr = int2(rect.xy) + int2(min(uint2(uv * float2(rect.zw)), rect.zw - 1));
    const int2 cell = int2(uint2(pr - int2(rect.xy)) / groups_x);
    float4 en = rl_encl_t.Load(int3(cell, 0));
    // (the cells next to the window count too, where they are not gun themselves: reticle parts touching the gun - a
    // chevron on the front post - are in cells the gun also reaches)
    if (en.z < 0.5) {
        uint cw, ch;
        rl_encl_t.GetDimensions(cw, ch);
        [unroll] for (uint i = 0; i < 8; ++i) {
            const int2 q = clamp(cell + kRetDirs[i], int2(0, 0), int2(cw, ch) - 1);
            if (rl_encl_t.Load(int3(q, 0)).z > 0.5) en.z = 1;
        }
    }
    float4 L = 0;
    if (en.z > 0.5 && rl_att_t.Load(int3(pr, 0)) < 0.5) {
        // carried along with the reticle (en.xy: its motion to the previous frame, output px, see cs_ret_align)
        if (!(flags & 2)) L = rl_hist_t.SampleLevel(rl_linear, uv + en.xy / float2(out_size), 0);
        const float4 pv = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, rl_depth_t.Load(int3(pr, 0)), 1), clip_to_prev);
        if (pv.w > 0) {
            const float2 from = float2(pv.x / pv.w * 0.5 + 0.5, 0.5 - pv.y / pv.w * 0.5);  // where the scenery was
            const int2 fr = int2(rect.xy) + int2(min(uint2(saturate(from) * float2(rect.zw)), rect.zw - 1));
            // (not scenery that was behind the gun or off the screen: the previous frame does not show it)
            const float3 before = (flags & 2) ? float3(0, 0, 0) : rl_hist_t.SampleLevel(rl_linear, from, 0).rgb;
            const float here = max(max(L.r, L.g), L.b);
            if (all(from > 0) && all(from < 1) && rl_att_t.Load(int3(fr, 0)) < 0.5 && max(max(before.r, before.g), before.b) < 0.03 + 0.5 * here) {
                const float3 r = rl_cur_t.Load(int3(id.xy, 0)).rgb - (rl_prev_t.SampleLevel(rl_linear, from, 0).rgb - before);
                const float3 d = r - L.rgb;
                const float tol = 0.06 + 0.15 * max(max(abs(L.r), abs(L.g)), abs(L.b));
                if (max(max(abs(d.r), abs(d.g)), abs(d.b)) < tol) {
                    L.rgb += 0.5 * d;
                    L.a = min(1.0, L.a + 0.2);
                } else {
                    L.rgb += (0.5 * (1.0 - L.a) * (1.0 - L.a) + 0.05) * d;
                    L.a *= 0.7;
                }
                L.rgb = max(L.rgb, 0.0);
            }
        }
    }
    rl_layer_u[id.xy] = L;
    if (any(L != 0)) {
        const int2 k = clamp(cell - cell_first, int2(0, 0), int2(1, 1));
        rl_marks[k.y * 2 + k.x] = 1;
    }
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
// (t3: the stretch's distance share - flag 128, R8, in x - or, with the share, the sight windows - flag 2048, see mask_in_sight)
Texture2D<float4> mask_ramp_t : register(t3);
Texture2D<unorm float> mask_tiles_t : register(t4);  // (flag 256) tiles of HUD the learned map confirms (cs_hud_spread)
// (flag 512: t5, the shares of the input's turn and of the rest each render pixel follows, Renderer::follow_share - bilinear
// at uv)
Texture2D<float4> mask_share_t : register(t5);
// (flag 2048, with the share: t3 - mask_ramp_t - the sight windows found this frame, cs_ret_enclose, per cell of groups_x render px)
// Inside a sight's window - within 2 cells of one - what this frame's analysis finds attached (widened by a render pixel,
// t1): a reticle drawn small on the glass (S.T.A.L.K.E.R. 2's red dot: the gun's motion at its centre, the scenery's motion
// and distance in its glow and ring) takes the scenery's share where the share is spread at that distance - it moved
// with the turn, a second dot and ring and the frame's inner edge beside the real ones. Held there, as before the share.
// Only what is farther than the window's frame, by 10% or more (the frame's farthest side, its nearest pixels): on the glass
// is what the game draws at the distance of the scenery behind, or past it (the front sight's ring). A gap in a character
// enclosed by itself - between strands of hair, fingers - is a window too, but what borders it is as near as what encloses
// it: held, RE2 Claire's hair and hands came apart in straight-edged cracks while the rest of her moved with her share.
float mask_near(float w);
bool mask_in_sight(int2 pr) {
    if (!(flags & 2048) || !(flags & 8) || mask_attached_t.Load(int3(pr, 0)) < 0.5) return false;
    uint cw, ch;
    mask_ramp_t.GetDimensions(cw, ch);
    const int2 c = (pr - int2(rect.xy)) / int(max(groups_x, 1u));
    const float near = mask_near(mask_share_t.Load(int3(pr, 0)).w);
    [loop] for (int dy = -2; dy <= 2; ++dy)
        [loop] for (int dx = -2; dx <= 2; ++dx) {
            const int2 q = clamp(c + int2(dx, dy), int2(0, 0), int2(cw, ch) - 1);
            const float4 e = mask_ramp_t.Load(int3(q, 0));
            if (e.z > 0.5 && near < e.w - 0.15 / 32.0) return true;
        }
    return false;
}
// Next to what the share holds (a render pixel away, any of the 8): the outline. Its pixels blend the held thing's colour
// with what is behind, and many carry what is behind's distance and motion - spread by distance, they took the scenery's
// share and slid off with it, a ghost of the outline left in the scenery (Cyberpunk's gun and hand while turning, RE2's
// character, a car's dashboard ornament). Held with it, as the widened attached mask held them before the share.
// (the share's w: the nearness, and whether the camera carries it - see cs_share_up)
float mask_near(float w) { return 2.0 * (w >= 0.5 ? w - 0.5 : w); }
bool mask_carried(int2 pr) { return mask_share_t.Load(int3(pr, 0)).w >= 0.5; }
bool mask_next_to_carried(int2 pr) {
    const int2 lo = int2(rect.xy), hi = int2(rect.xy + rect.zw) - 1;
    // (only where what the share holds is in front - nearer by 10% or more: an outline against what is behind it. Inside
    // one surface - a gun whose sway gives it 0.05 here and 0.2 there, a body - a border of the share is no outline, and
    // holding it drew lines across the gun while turning, Stalker 2's rifle, RE2's Claire)
    const float near = mask_near(mask_share_t.Load(int3(pr, 0)).w);
    [unroll] for (int k = 0; k < 9; ++k) {
        if (k == 4) continue;
        const int2 q = clamp(pr + int2(k % 3 - 1, k / 3 - 1), lo, hi);
        const float4 n = mask_share_t.Load(int3(q, 0));
        if (n.x < 0.1 && mask_near(n.w) > near + 0.15 / 32.0) return true;
    }
    return false;
}
float4 mask_share(float2 uv) {
    const float2 p = clamp(uv * float2(rect.zw) - 0.5, 0, float2(rect.zw) - 1.001);
    const int2 i = int2(rect.xy) + int2(floor(p));
    const float2 f = p - floor(p);
    const float4 a = mask_share_t.Load(int3(i, 0)), b = mask_share_t.Load(int3(i + int2(1, 0), 0));
    const float4 c = mask_share_t.Load(int3(i + int2(0, 1), 0)), d = mask_share_t.Load(int3(i + int2(1, 1), 0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}
bool mask_follows_world(int3 q) {
    const float v = mask_world_t.Load(q);
    return world_part(v) >= 0.5 && persist_part(v) < float(out_size.x) / 12.0;
}
RWTexture2D<unorm float> mask_u : register(u0);
RWTexture2D<unorm float4> warp_mask_u : register(u1);  // (x as above; y the share of the rest of the warp's turn, z of the camera's movement - see ow_lands)
groupshared uint mask_hud_gs[10 * 10];  // the group's pixels and a 1 px border: HUD kept there (see cs_mask)
[numthreads(8, 8, 1)] void cs_mask(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {
    // Whether the HUD is kept at each pixel of the group and its 1 px border, worked out once per pixel
    // (positions clamped to the screen, as the 3x3 test below reads them).
    if (flags & 4) {
        for (uint k = gi; k < 100; k += 64) {
            const int3 q = int3(clamp(int2(gid.xy * 8) + int2(k % 10, k / 10) - 1, int2(0, 0), int2(out_size) - 1), 0);
            // (combined detection, flag 256: only in a tile the learned map's confirmation reached, cs_hud_spread)
            const bool confirmed = !(flags & 256) || mask_tiles_t.Load(int3(q.xy / 8, 0)) > 0.75;
            mask_hud_gs[k] = (confirmed && mask_score_t.Load(q) > 0.6 && (!(flags & 64) || !mask_follows_world(q))) ? 1u : 0u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (any(id.xy >= out_size)) return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(out_size);
    const int2 pr = int2(rect.xy) + int2(min(uint2(uv * float2(rect.zw)), rect.zw - 1));
    bool keep = false;
    if (flags & 4) {
        const uint2 l = id.xy - gid.xy * 8;
        [unroll] for (int y = 0; y <= 2; ++y)
            [unroll] for (int x = 0; x <= 2; ++x) keep = keep || mask_hud_gs[(l.y + y) * 10 + l.x + x] != 0;
    }
    const bool hud = keep;
    if (flags & 512) {
        // The share of the turn followed: the warp moves the spot by that much of its motion; held only the HUD. (As
        // character/weapon where it follows under half of its band's share: kept out of the background memory, XPAR frame
        // generation - 0.75 where held, under 0.1, 0.7 where it follows a little: the mask view shows only what is held, see
        // cs_tint. Not where the scenery itself follows little: the ground at an orbited character's distance - Elden Ring
        // 0.12-0.16 of the turn, RE2's floor beside Claire 0.23 - was marked by its share alone, under half, and the memory
        // never took it in: 39% of the screen in RE2, the gaps beside her filled with streaks of her outline instead)
        const float4 share = mask_share(uv);
        // (held in a sight: 0.96 - held like the HUD by the warp, which reads > 0.9 from this mask - see ow_mask_at; an
        // outline 0.95 - not held, see ow_held)
        const bool sight = !hud && mask_in_sight(pr);
        // (the outline: not where the share holds already - only where it would slide off)
        // (flag 4096: only on what the share counts as scenery - not carried, see mask_carried: an edge against the road
        // that would slide off with it, Cyberpunk's hand. Within a character that follows part of the turn - hair, hands -
        // what borders a nearer strand is the character itself: held, it parted from the rest of it, RE2 Claire)
        const bool outline = !hud && !sight && share.x >= 0.1 && (!(flags & 4096) || !mask_carried(pr)) && mask_next_to_carried(pr);
        const bool held = sight || outline;
        // (flag 8192, default: 0.73 - carried, held by the warp - share under 1/60, see ow_mask_at - beside a carried render
        // pixel it moves a little: not held, see ow_held. Held, it stayed where the frame had it while the part beside it moved
        // a few pixels with its share - a crack with the floor behind, RE2 Claire's hair and hands while turning; scored
        // between game frames, character false edges -1% to -9%. Held elsewhere: carried pixels freed everywhere left a
        // striped copy of Cyberpunk's hand edge on the road (capture 19) - its hand does not border a moving part)
        float carried = !mask_carried(pr) ? 0.0 : share.x < 0.1 ? 0.75 : 0.7;
        if ((flags & 8192) && carried > 0.7 && mask_share_t.Load(int3(pr, 0)).x < 1.0 / 60.0) {
            const int2 lo = int2(rect.xy), hi = int2(rect.xy + rect.zw) - 1;
            [unroll] for (int k = 0; k < 9; ++k) {
                if (k == 4) continue;
                const float4 n = mask_share_t.Load(int3(clamp(pr + int2(k % 3 - 1, k / 3 - 1), lo, hi), 0));
                if (n.w >= 0.5 && n.x >= 1.0 / 60.0) carried = 0.73;
            }
        }
        mask_u[id.xy] = hud ? 1.0 : sight ? 0.96 : outline ? 0.95 : carried;
        if (!(flags & 1024)) warp_mask_u[id.xy] = (hud || held) ? float4(1, 0, 0, 0) : float4(0.9 * (1.0 - share.x), share.y, share.z, 0);
        return;
    }
    if (!keep && (flags & 8)) keep = mask_attached_t.Load(int3(pr, 0)) > 0.5;
    // (1 the HUD, 0.75 the character/weapon: held alike - above 0.5 - except by XPAR frame generation, which keeps
    // only the HUD where it is)
    mask_u[id.xy] = hud ? 1.0 : keep ? 0.75 : 0.0;
    const float follow = (flags & 128) ? mask_ramp_t.Load(int3(pr, 0)).x : 1.0;
    warp_mask_u[id.xy] = keep ? float4(1, 0, 0, 0) : float4(0.9 * (1.0 - follow), follow, follow, 0);
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
    frame = seen(sc_frame_t.Load(int3(p, 0)).rgb);
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
            const float3 frame = seen(sc_frame_t.Load(int3(p, 0)).rgb), scene = max(sc_scene_t.Load(int3(p, 0)).rgb, 0);
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
        const bool hud = miss(seen(sc_frame_t.Load(int3(p, 0)).rgb), pred) > 0.06 + 0.5 * gradient;
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
// 256 moving objects (t6-t9; alpha: game frames back towards the previous frame), 1024 soft edge (ow_soft_edge).
// (512 is the moving-objects pass's debug colours, cs_obj_fix: the soft edge had it first and painted them in.)
// Flag 4096: forward splat (t12, cs_warp_claim) - every rendered pixel first claims the spot it lands on, the nearest
// winning, and each output pixel takes the nearest claimant around it whose surface reaches it (see ow_find).
// t2: the XPAR warp's mask (cs_mask u1): held above 0.95. Flag 2048: sight reticles (t11, see cs_ret_layer) - the layer
// on a sight's glass: where it covers the glass (the reticle, ow_cover) the game frame's own pixel is shown, as
// held; where it is faint (glow) it is taken out where the warp takes the scenery from and added back here. The
// scenery is never taken from under the reticle: the warp steps past it, as past held pixels.
Texture2D<float4> ow_color_t : register(t0);
Texture2D<float4> ow_ui_t : register(t1);
Texture2D<float4> ow_mask_t : register(t2);  // (x: held / the input's share, y: the rest's share, z: the movement's - see cs_mask)
Texture2D<float> ow_depth_t : register(t3);
Texture2D<float4> ow_fill_t : register(t5);  // the scene behind the HUD (alpha 1 where filled)
Texture2D<float4> ow_move_t : register(t6);   // moving objects: offset to the previous frame (cs_obj_move)
Texture2D<uint> ow_keys_t : register(t7);     // ...their claims on the output pixels (cs_obj_splat)
Texture2D<float4> ow_prev_t : register(t8);   // ...the previous frame's picture
Texture2D<float> ow_prevz_t : register(t9);   // ...and depth
Texture2D<uint> ow_out_tiles_t : register(t10); // ...the 8x8 output blocks they touch
Texture2D<float4> ow_layer_t : register(t11);  // sight reticles: the layer on the glass (cs_ret_layer)
Texture2D<uint> ow_claims_t : register(t12);   // forward splat: the claims on the displayed spots (cs_warp_claim)
Texture2D<float> ow_memz_t : register(t13);    // background memory's depth (flag 64)
// How much the layer covers the glass at a spot: 0 up to faint glow, 1 on the reticle itself.
float ow_cover(float4 l) { return saturate((max(max(l.r, l.g), l.b) - 0.03) / 0.2); }
RWTexture2D<float4> ow_out_u : register(u0);
SamplerState ow_linear : register(s0);  // bilinear, clamped to the edge
float4 ow_bilinear(float2 uv) { return ow_color_t.SampleLevel(ow_linear, uv, 0); }
// (held: the HUD - 1 - and, with the follow share, what follows none of the turn - 0.9: shown as the frame has it and never
// taken as scenery for what the turn uncovers beside it, as when it was held; following any of it, the mask is at most
// 0.9 * (1 - 1/64))
// (flag 16384, the follow share: no warp mask at output resolution - t2 is the no-warp mask, R8: 1 the HUD; t14 the shares at
// render resolution, as cs_share_up wrote them. The same values the warp mask held - 0.9 * (1 - the input's share), the
// rest's, the movement's; 1, 0, 0 on the HUD - with the shares interpolated once instead of twice: 64 MB less at 4K, and
// as much again for the shown copy)
Texture2D<float4> ow_share_t : register(t14);
float4 ow_mask_at(float2 s) {
    if (!(flags & 16384)) return ow_mask_t.SampleLevel(ow_linear, s, 0);
    uint sw, sh;
    ow_share_t.GetDimensions(sw, sh);
    const float2 size = float2(sw, sh);
    const float2 su = clamp((float2(rect.xy) + s * float2(rect.zw)) / size, (float2(rect.xy) + 0.5) / size, (float2(rect.xy + rect.zw) - 0.5) / size);
    const float4 share = ow_share_t.SampleLevel(ow_linear, su, 0);
    // (the HUD, bilinear as the mask had it: the four output pixels around, gathered - w top-left, z top-right, x and y below)
    const float2 p = s * float2(out_size) - 0.5;
    const float2 f = p - floor(p);
    const float4 g = ow_mask_t.GatherRed(ow_linear, (floor(p) + 1.0) / float2(out_size));
    const float4 hud = float4(g > 0.9);
    const float h = lerp(lerp(hud.w, hud.z, f.x), lerp(hud.x, hud.y, f.x), f.y);
    return float4(lerp(0.9 * (1.0 - share.x), 1.0, h), share.y * (1.0 - h), share.z * (1.0 - h), 0);
}
float2 ow_own(float2 s);
bool ow_held(int3 p) {
    if (flags & 16384) {
        const float2 uv = (float2(p.xy) + 0.5) / float2(out_size);
        if (ow_mask_at(uv).x <= 0.885) return false;
        // (held with the character - an outline, a sight's glass, 0.95 - when characters are carried on along their own motion
        // (flag 65536): warped as what the camera carries instead - the mask's share there is none - and carried on with the
        // character beside it, if it moves; held, a copy of the outline was left where it had been, or a seam where it moved a
        // little. The HUD, 1, stays held)
        // (flag 131072 - set_outlines_held(false), off by default: outlines, 0.95, the same without own motion. Held they
        // stay (Cyberpunk capture 19: free, the hand's edge left a wide striped copy on the road - error 0.023 held, 0.056
        // free; RE2's false edges on Claire +2-14% held, the big cracks were the false sight windows below). Held, the outline pixels learned through hair and hands sat in patches beside pixels carried by the
        // share - the two moved apart and the floor showed through in straight-edged lines, RE2 Claire's hair; scored
        // between game frames (seq_replay FW_STRIDE=2), not holding them: character false edges -3% to -14%, error the
        // same. What is on a sight's glass, 0.96, stays held: at the distance of the scenery behind the glass but not moving
        // with it, the warp's search over that surface took the red dot and the front sight's ring along with the scenery -
        // doubled, Stalker 2)
        // (dev: flag 131072 frees the outlines alone, 262144 the carried pixels alone - share ~0, mask 0.7/0.75, held by
        // ow_mask_at's 0.9 - own motion, 65536, both)
        const float mv = ow_mask_t.Load(p).x;
        if ((flags & 65536) && mv < 0.955) return false;
        if ((flags & 131072) && mv >= 0.94 && mv < 0.955) return false;
        if ((flags & 262144) && mv < 0.94) return false;
        if ((flags & 524288) && mv > 0.72 && mv < 0.74) return false;  // (0.73: see cs_mask, flag 8192)
        return true;
    }
    return ow_mask_t.Load(p).x > 0.885;
}
// The share of the warp's motion applied at this spot (the stretch).
float ow_follow(float2 uv) { return saturate(1.0 - ow_mask_at(uv).x / 0.9); }
float ow_depth(float2 uv) {
    if (flags & 32) return 0;  // the warp does not depend on depth (the camera only turns): no reads
    const uint2 p = rect.xy + min(uint2(saturate(uv) * float2(rect.zw)), rect.zw - 1);
    return ow_depth_t.Load(int3(p, 0));
}
// Background memory (flag 64, see cs_memory): the scenery last seen at this spot of the source frame -
// behind held pixels, or outside the frame - if seen within the last 60 game frames. Bilinear where all
// four texels hold something, the nearest texel otherwise.
// Nearness from the stored depth alone (32 steps per doubling of the distance; reversed or standard depth, flag 16), so
// depths compare without the projection (the forward splat's claims, the background memory's depth).
float ow_near(float d) {
    const float x = (flags & 16) ? d : 1.0 - d;
    return 1.0 + saturate((log2(max(x, 1e-12)) + 30.0) / 32.0) * 1022.0;
}
static const float kMemMargin = 0.125;  // the memory covers the screen plus this much on every side
static const float kMemForget = 60.0;   // game frames
Texture2D<float4> ow_mem_t : register(t4);
// ...or only what it keeps from before (not seen now: behind what is in front, or held, or outside the frame), and only
// where what it keeps is the scenery behind (far_near: its nearness, ow_near) - not a copy of what has moved on from
// there (the character a few frames ago: a ghost of it in the gap it leaves).
bool ow_from_memory_behind(float2 s, float far_near, out float4 c) {
    c = 0;
    const float2 mu = (s + kMemMargin) / (1.0 + 2.0 * kMemMargin);
    if (any(mu < 0) || any(mu > 1)) return false;
    {
        uint zw, zh;
        ow_memz_t.GetDimensions(zw, zh);
        const float z = ow_memz_t.Load(int3(min(uint2(mu * float2(zw, zh)), uint2(zw, zh) - 1), 0));
        if (abs(ow_near(z) - far_near) > 10.0) return false;
    }
    const float4 ages = ow_mem_t.GatherAlpha(ow_linear, mu);
    if (min(min(ages.x, ages.y), min(ages.z, ages.w)) >= 1.0 && max(max(ages.x, ages.y), max(ages.z, ages.w)) < kMemForget) {
        c = float4(ow_mem_t.SampleLevel(ow_linear, mu, 0).rgb, 1);
        return true;
    }
    uint mw, mh;
    ow_mem_t.GetDimensions(mw, mh);
    const float4 v = ow_mem_t.Load(int3(min(uint2(mu * float2(mw, mh)), uint2(mw, mh) - 1), 0));
    if (v.a < 1.0 || v.a >= kMemForget) return false;
    c = float4(v.rgb, 1);
    return true;
}
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
float3 ow_forward_at(float2 uv, float d) {
    const float4 f = mul(float4(uv.x * 2 - 1, 1 - uv.y * 2, d, 1), clip_to_prev);
    return float3(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5, f.w);
}
float3 ow_forward(float2 uv) { return ow_forward_at(uv, ow_depth(uv)); }
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
// Soft edge (flag 1024): beyond the rendered frame, a blur of the pixels along the edge that widens with the
// distance past it (no streaks: neighbouring rows mix). Neither darkened nor desaturated: either made the band
// stand out more.
float4 ow_soft_edge(float2 src, float2 inside) {
    const float2 px = float2(out_size);
    const float2 beyond = (src - inside) * px;  // from the edge to this pixel, in output pixels
    const float d = length(beyond);
    const float2 inward = -beyond / max(d, 1e-3);
    const float2 along = float2(-inward.y, inward.x);
    const float spread = min(4.0 + 0.6 * d, 192.0);
    const float2 lo = 0.5 / px, hi = 1.0 - lo;
    float4 c = 0;
    float total = 0;
    [unroll] for (int i = -3; i <= 3; ++i)
        [unroll] for (int j = 0; j < 3; ++j) {
            const float2 q = inside + (along * (float(i) * spread / 3.0) + inward * (float(j) * spread * 0.5 + 0.5)) / px;
            const float w = 1.0 / (1.0 + 0.25 * float(i * i) + float(j));
            c += ow_bilinear(clamp(q, lo, hi)) * w;
            total += w;
        }
    return c / total;
}
// Where the point at rendered-frame uv s is shown (f: ow_forward(s)), with the stretch's share of the warp.
// (two shares, flag 8192: the input's part of the warp's turn - wmat: the rendered frame's clip -> the camera turned by the
// input alone, rotation only, so any depth - by the first, the rest of it by the second; see follow_share)
float2 ow_input_at(float2 s) {
    const float4 f = mul(float4(s.x * 2 - 1, 1 - s.y * 2, 0, 1), wmat);
    return f.w > 1e-6 ? float2(f.x / f.w * 0.5 + 0.5, 0.5 - f.y / f.w * 0.5) : s;
}
// (the camera's movement apart, with a share of its own: f is where the spot lands with the turn and the movement, the turn
// alone moves it as it moves what is infinitely far - fr; see ShareSums)
float2 ow_lands_by(float2 s, float3 f, float input_share, float rest_share, float moved_share) {
    const float4 inf = mul(float4(s.x * 2 - 1, 1 - s.y * 2, (flags & 16) ? 0.0 : 1.0, 1), clip_to_prev);
    const float2 fr = inf.w > 1e-6 ? float2(inf.x / inf.w * 0.5 + 0.5, 0.5 - inf.y / inf.w * 0.5) : f.xy;
    const float2 moved = (f.xy - fr) * moved_share;
    if (flags & 8192) {
        const float2 fi = ow_input_at(s);
        return s + (fi - s) * input_share + (fr - fi) * rest_share + moved;
    }
    return s + (fr - s) * input_share + moved;
}
// (flag 65536: a character's own motion carried on - cs_own_up, t15, render px per game frame - for wproj.w game frames past
// the shown frame)
Texture2D<float2> ow_own_t : register(t15);
float2 ow_own(float2 s) {
    if (!(flags & 65536)) return float2(0, 0);
    uint mw, mh;
    ow_own_t.GetDimensions(mw, mh);
    const float2 size = float2(mw, mh);
    const float2 su = clamp((float2(rect.xy) + s * float2(rect.zw)) / size, (float2(rect.xy) + 0.5) / size, (float2(rect.xy + rect.zw) - 0.5) / size);
    return ow_own_t.SampleLevel(ow_linear, su, 0) / float2(rect.zw) * wproj.w;
}
// (whether a rendered spot is a character carried on along its own motion - cs_own_up writes none elsewhere)
bool ow_moves_on(float2 s) {
    uint mw, mh;
    ow_own_t.GetDimensions(mw, mh);
    const float2 size = float2(mw, mh);
    const int2 p = int2(rect.xy) + int2(min(uint2(saturate(s) * float2(rect.zw)), rect.zw - 1));
    return any(abs(ow_own_t.Load(int3(p, 0)).xy) > 0.01);
}
float2 ow_lands(float2 s, float3 f) {
    if (!(flags & 128)) return f.xy;
    const float4 m = ow_mask_at(s);
    return ow_lands_by(s, f, saturate(1.0 - m.x / 0.9), m.y, m.z) + ow_own(s);
}
// The rendered pixel that lands on uv, searched from it; false: none (behind the displayed camera). Where several
// surfaces land on uv (a near object passing in front of the scenery) or none does (what it uncovers), the search
// ends on whichever its steps happen to reach: neighbouring pixels can take different ones (doubled edges).
bool ow_search(float2 uv, out float2 src) {
    src = uv;
    const int steps = (flags & 128) ? 12 : 6;
    [loop] for (int it = 0; it < steps; ++it) {
        const float3 f = ow_forward(src);
        if (f.z <= 1e-6) return false;
        const float2 miss = uv - ow_lands(src, f);
        src += miss;
        if (all(abs(miss * float2(out_size)) < 0.25)) break;
    }
    return true;
}
// Forward splat (flag 4096): see ow_near.
// Every rendered pixel (not held) claims the spot it lands on, in rendered-pixel cells of the displayed picture:
// nearness << 22 | the claimant's offset from the cell (11 bits each, +1024). The nearest claim wins.
RWTexture2D<uint> wc_claims_u : register(u0);
[numthreads(8, 8, 1)] void cs_warp_claims_clear(uint3 id : SV_DispatchThreadID) {
    if (all(id.xy < rect.zw)) wc_claims_u[id.xy] = 0;
}
[numthreads(8, 8, 1)] void cs_warp_claim(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= rect.zw)) return;
    const float2 s = (float2(id.xy) + 0.5) / float2(rect.zw);
    if ((flags & 2) && ow_held(int3(min(uint2(s * float2(out_size)), out_size - 1), 0))) return;
    const float3 f = ow_forward(s);
    if (f.z <= 1e-6) return;
    const float2 l = ow_lands(s, f);
    if (any(l < 0) || any(l >= 1)) return;
    const int2 cell = int2(l * float2(rect.zw));
    const int2 d = int2(id.xy) - cell;
    if (any(abs(d) > 1023)) return;
    InterlockedMax(wc_claims_u[cell], (uint(ow_near(ow_depth(s))) << 22) | (uint(d.x + 1024) << 11) | uint(d.y + 1024));
}
// Is the rendered frame's spot s on the surface of nearness nk: any of the four rendered pixels around it within 6
// steps (~14%). (One rendered pixel spans several output pixels: tested on the nearest alone, a near object's outline
// came out in steps of whole rendered pixels; this way its anti-aliased edge goes with it, as the frame shows it.)
bool ow_on_surface(float2 s, float nk) {
    const float2 p = clamp(s * float2(rect.zw) - 0.5, 0, float2(rect.zw) - 1.001);
    const int2 i = int2(rect.xy) + int2(floor(p));
    bool on = false;
    [unroll] for (int k = 0; k < 4; ++k)
        on = on || abs(ow_near(ow_depth_t.Load(int3(i + int2(k & 1, k >> 1), 0))) - nk) <= 6.0;
    return on;
}
// A claim: its claimant's rendered pixel (the cell it claimed + the offset in the key).
int2 ow_claimant(int2 cell, uint key) { return cell + int2(int((key >> 11) & 2047u) - 1024, int(key & 2047u) - 1024); }
// From a claimant, the steps of the search over its own surface: true if they end on uv (< 0.5 px), still on that surface.
// (the steps at the claimant's own depth: the claimant is within a rendered pixel of uv, so its surface is as good as
// flat there - and steady where the depth changes every rendered pixel: with the depth under each step, hair and foliage
// against what is behind them made the steps jump between the two, and the claim was dropped - the hair came out
// speckled with the scenery behind it)
// (...and the claimant's own share of the warp's motion, flag 128 - where it follows by depth, a grass blade and the
// ground behind it each their own: read at every step, the steps jumped between the two and most of the grass was left to
// the full warp)
bool ow_refine(float2 uv, int2 p, float nk, out float2 s) {
    s = (float2(p) + 0.5) / float2(rect.zw);
    const float dp = ow_depth(s);
    const float4 m = (flags & 128) ? ow_mask_at(s) : float4(0, 1, 1, 0);
    const float fo = (flags & 128) ? saturate(1.0 - m.x / 0.9) : 1.0, fr = (flags & 128) ? m.y : 1.0, fm = (flags & 128) ? m.z : 1.0;
    const float2 own = (flags & 128) ? ow_own(s) : float2(0, 0);  // (and its own motion, as its share: read once)
    [loop] for (int it = 0; it < 6; ++it) {
        const float3 f = ow_forward_at(s, dp);
        if (f.z <= 1e-6) return false;
        const float2 miss = uv - ow_lands_by(s, f, fo, fr, fm) - own;
        s += miss;
        if (all(abs(miss * float2(out_size)) < 0.25)) break;
    }
    const float3 f = ow_forward_at(s, dp);
    return f.z > 1e-6 && all(abs((uv - ow_lands_by(s, f, fo, fr, fm) - own) * float2(out_size)) < 0.5) && ow_on_surface(s, nk);
}
// Uncovered at uv: the farthest claimant's surface (the scenery behind), carried on to uv at its own depth - as far as the
// rendered frame still shows that surface on the way there; beyond, the nearer thing in front hides it: stopped at its
// edge (stretched scenery, not a copy of what is in front, nor blocks of one pixel).
// (with background memory, flag 64: where a nearer thing hides that surface in the rendered frame, the memory may have
// seen it before - ow_hidden is set, and ow_scene shows the memory's)
static bool ow_hidden = false;
static float4 ow_hidden_c = 0;
// (flag 32768, debug: the way each output pixel was found, written instead of its colour - r the path, g 1. 1 the nearest
// claim, 2 a farther claim, 3 uncovered: the scenery behind stretched (ow_behind), 4 uncovered: the memory, 5 uncovered
// next to held pixels: the search's way, 6 no claims around: past the frame or held pixels (the search), 7 the ring: a
// claim reaching it, 8 the ring: the scenery behind, 9 no claims and no ring: the search, 10 stepped past held pixels,
// 11 the memory past held pixels, 12 the frame's edge, 13 none (behind the camera), 14 a held pixel)
static uint ow_path = 0;
float2 ow_behind(float2 uv, int2 p, float far_near) {
    const float2 s0 = (float2(p) + 0.5) / float2(rect.zw);
    const float far_d = ow_depth(s0);
    float2 s = s0;
    [loop] for (int it = 0; it < 4; ++it) {
        const float3 f = ow_forward_at(s, far_d);
        if (f.z <= 1e-6) break;
        s += uv - ow_lands(s, f);
    }
    if (all(s >= 0) && all(s <= 1) && abs(ow_near(ow_depth(s)) - far_near) > 6.0) {
        if (flags & 64) ow_hidden = ow_from_memory_behind(s, far_near, ow_hidden_c);
        float lo = 0, hi = 1;
        [unroll] for (int b = 0; b < 5; ++b) {
            const float m = 0.5 * (lo + hi);
            if (abs(ow_near(ow_depth(lerp(s0, s, m))) - far_near) <= 6.0) lo = m; else hi = m;
        }
        s = lerp(s0, s, lo);
    }
    return s;
}
// The claims around uv (3x3 cells), nearest first: the first claimant whose surface reaches uv (ow_refine) is what is
// shown there - the nearest surface that reaches it. None: uncovered here (hole: ow_behind from the farthest), or no
// claims around at all (false).
static bool ow_first = false;  // (ow_claimed: the nearest claim around was the one taken)
bool ow_claimed(float2 uv, out float2 src, out bool hole) {
    src = uv;
    hole = false;
    ow_first = false;
    const int2 c0 = int2(uv * float2(rect.zw));
    uint keys[9];
    uint far_key = 0xFFFFFFFFu;
    int2 far_cell = c0;
    [unroll] for (int k = 0; k < 9; ++k) {
        const int2 cell = clamp(c0 + int2(k % 3 - 1, k / 3 - 1), int2(0, 0), int2(rect.zw) - 1);
        keys[k] = ow_claims_t.Load(int3(cell, 0));
        if (keys[k] && (keys[k] >> 22) < (far_key >> 22)) { far_key = keys[k]; far_cell = cell; }
    }
    if (far_key == 0xFFFFFFFFu) return false;
    float2 first_s = uv;  // (the nearest claimant's own steps' end, reaching uv or not)
    [loop] for (int attempt = 0; attempt < 3; ++attempt) {
        uint best = 0;
        int bi = 0;
        [unroll] for (int k = 0; k < 9; ++k) if (keys[k] > best) { best = keys[k]; bi = k; }
        if (best == 0) break;
        const int2 pb = ow_claimant(clamp(c0 + int2(bi % 3 - 1, bi / 3 - 1), int2(0, 0), int2(rect.zw) - 1), best);
        // (the same claimant may hold several cells around: tried once)
        [unroll] for (int k = 0; k < 9; ++k)
            if (keys[k] && all(ow_claimant(clamp(c0 + int2(k % 3 - 1, k / 3 - 1), int2(0, 0), int2(rect.zw) - 1), keys[k]) == pb)) keys[k] = 0;
        float2 s;
        if (ow_refine(uv, pb, float(best >> 22), s)) { src = s; ow_first = attempt == 0; ow_path = attempt == 0 ? 1 : 2; return true; }
        if (attempt == 0) first_s = s;
    }
    // (none reaches it, and even the farthest around is a character carried on along its own motion - flag 65536: a crack
    // inside the moving body, not what it uncovers: the nearest claimant's surface, stretched. Taken from behind, the
    // farthest - in hair, the strands and scalp beneath - drew dark lines through it that flickered from refresh to refresh)
    if ((flags & 65536) && ow_moves_on((float2(ow_claimant(far_cell, far_key)) + 0.5) / float2(rect.zw))) { src = first_s; ow_path = 2; return true; }
    src = ow_behind(uv, ow_claimant(far_cell, far_key), float(far_key >> 22));
    ow_path = ow_hidden ? 4 : 3;
    hole = true;
    return true;
}
// No claims in the 3x3 around uv: an uncovered area wider than that (a near object moved off it), or a surface stretched
// that much. Looked for along the way near things move against far ones here (the camera's moving: what the depth at uv
// does against infinity), first behind - where the scenery the gap uncovers continues - then ahead, then all 8
// directions (no such way: a turn alone): the same way for every pixel of the gap, so they agree (picked per pixel
// among 8 directions, neighbours took different scenery: a criss-cross). The nearest claimant found if its surface
// reaches uv (stretched), otherwise the scenery behind from the farthest (ow_behind).
bool ow_ring(float2 uv, out float2 src) {
    src = uv;
    const int2 c0 = int2(uv * float2(rect.zw));
    static const int kRadii[9] = {2, 3, 4, 6, 8, 12, 16, 24, 32};
    uint near_key = 0, far_key = 0xFFFFFFFFu;
    int2 near_cell = c0, far_cell = c0;
    const float3 fn = ow_forward(uv), ff = ow_forward_at(uv, (flags & 16) ? 0.0 : 1.0);
    float2 way = (ow_lands(uv, fn) - ow_lands(uv, ff)) * float2(rect.zw);
    const bool along = fn.z > 1e-6 && ff.z > 1e-6 && dot(way, way) > 0.25;
    way = along ? normalize(way) : float2(1, 0);
    [loop] for (int way_try = 0; way_try < 3 && near_key == 0; ++way_try) {
        if (way_try < 2 && !along) continue;
        [loop] for (int r = 0; r < 9 && near_key == 0; ++r)
            [loop] for (int d = 0; d < (way_try < 2 ? 1 : 8); ++d) {
                const float2 dir = way_try == 0 ? -way : way_try == 1 ? way
                                 : float2(d == 0 || d == 1 || d == 7 ? 1 : d == 3 || d == 4 || d == 5 ? -1 : 0,
                                          d == 1 || d == 2 || d == 3 ? 1 : d == 5 || d == 6 || d == 7 ? -1 : 0);
                const int2 cell = c0 + int2(round(dir * float(kRadii[r])));
                if (any(cell < 0) || any(cell >= int2(rect.zw))) continue;
                const uint key = ow_claims_t.Load(int3(cell, 0));
                if (!key) continue;
                if (key > near_key) { near_key = key; near_cell = cell; }
                if ((key >> 22) < (far_key >> 22)) { far_key = key; far_cell = cell; }
            }
    }
    if (near_key == 0) return false;
    float2 s;
    if (ow_refine(uv, ow_claimant(near_cell, near_key), float(near_key >> 22), s)) { src = s; ow_path = 7; return true; }
    src = ow_behind(uv, ow_claimant(far_cell, far_key), float(far_key >> 22));
    ow_path = ow_hidden ? 4 : 8;
    return true;
}
// The rendered pixel shown at uv; false: none (behind the displayed camera). ow_plain: found as the nearest claim around,
// nothing else involved (the 2x2 blocks' shortcut, see cs_own_warp, builds on those only).
static bool ow_plain = false;
bool ow_find(float2 uv, out float2 src) {
    ow_plain = false;
    if (flags & 4096) {
        bool hole;
        if (ow_claimed(uv, src, hole)) {
            if (!hole) { ow_plain = ow_first; return true; }
            // Uncovered. Next to held pixels (the character/weapon, the HUD: they claim nothing) the search's own
            // way - background memory, or stepping past them - shows what is behind them; elsewhere the scenery behind.
            float2 b;
            if ((flags & 2) && ow_search(uv, b) && all(b >= 0) && all(b <= 1) &&
                ow_held(int3(min(uint2(b * float2(out_size)), out_size - 1), 0))) { src = b; ow_hidden = false; ow_path = 5; }
            return true;
        }
        // No claims around: past the frame's edge or behind held pixels the search's way (soft edge, background memory,
        // stepping past); otherwise uncovered wider than the 3x3 (ow_ring) - the search would end on what moved off it
        // and copy it there (a second, ghost copy beside a near character when strafing).
        if (!ow_search(uv, src)) { ow_path = 13; return false; }
        ow_path = 6;
        if (any(src < 0) || any(src > 1) || ((flags & 2) && ow_held(int3(min(uint2(src * float2(out_size)), out_size - 1), 0)))) return true;
        ow_path = 9;
        float2 r;
        if (ow_ring(uv, r)) src = r;
        return true;
    }
    return ow_search(uv, src);
}
// What the warp shows at an output pixel that is not held (src: where its search landed).
float4 ow_scene_from(uint2 id, float2 uv, float2 src, bool valid);
float4 ow_scene(uint2 id, float2 uv, out float2 src, out bool valid) {
    ow_hidden = false;
    valid = ow_find(uv, src);
    return ow_scene_from(id, uv, src, valid);
}
float4 ow_scene_from(uint2 id, float2 uv, float2 src, bool valid) {
    if (valid && ow_hidden) return ow_hidden_c;  // (uncovered, the memory has what is behind)
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
                    ow_path = 10;
                    const float2 away = normalize((src - uv) * float2(out_size) + float2(1e-3, 0));
                    // (HLSL evaluates both sides of &&: an explicit branch, so nothing is read without the memory)
                    bool found = false;
                    if (flags & 64) {
                        float4 remembered;
                        if (ow_from_memory(src, remembered)) { c = remembered; found = true; ow_path = 11; }
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
            ow_path = 12;
            // Revealed edge: the nearest edge pixels, blended with a few taps further in along the same
            // direction, so the band reads as a soft continuation rather than streaks.
            if (flags & 1024) {
                c = ow_soft_edge(src, inside);
            } else {
                const float2 inward = normalize(inside - src + 1e-9) / float2(out_size);
                const float reach = min(length((src - inside) * float2(out_size)), 24.0);
                c = 0;
                [unroll] for (int k = 0; k < 4; ++k) c += ow_bilinear(inside + inward * (reach * 0.25 * k));
                c *= 0.25;
            }
            if (flags & 64) {
                float4 remembered;
                if (ow_from_memory(src, remembered)) c = remembered;
            }
        }
    }
    return c;
}
// The output pixel's colour from where its search landed: the scene there, held pixels as the frame shows them, the
// sight-reticle layer, the UI layer.
void ow_finish(uint2 pid, float2 uv, float2 src, bool valid) {
    const uint3 id = uint3(pid, 0);
    float4 c = ow_scene_from(id.xy, uv, src, valid);
    if ((flags & 2) && ow_held(int3(id.xy, 0))) c = ow_color_t.Load(int3(id.xy, 0));
    else if ((flags & 2048) && valid) {
        const float4 here = ow_layer_t.Load(int3(id.xy, 0)), there = ow_layer_t.SampleLevel(ow_linear, src, 0);
        float3 scene = c.rgb - there.rgb;
        const float under = ow_cover(there);
        if (under > 0) {
            // the scenery was taken from under the reticle: the nearest scenery past it along the warp instead
            const int2 last = int2(out_size) - 1;
            const float2 away = normalize((src - uv) * float2(out_size) + float2(1e-3, 0));
            bool found = false;
            float3 past = scene;
            [loop] for (int k = 1; k <= 32 && !found; ++k)
                [unroll] for (int side = 0; side < 2; ++side) {
                    // (steps of 1/1080 of the height: up to 3% of it)
                    const int2 qi = clamp(int2(src * float2(out_size) + away * (side ? -k : k) * max(1.0, float(out_size.y) / 1080.0)),
                                          int2(0, 0), last);
                    const float4 lq = ow_layer_t.Load(int3(qi, 0));
                    if (!found && ow_cover(lq) <= 0 && !((flags & 2) && ow_held(int3(qi, 0)))) {
                        past = ow_color_t.Load(int3(qi, 0)).rgb - lq.rgb;
                        found = true;
                    }
                }
            scene = lerp(scene, past, under);
        }
        c.rgb = lerp(scene + here.rgb, ow_color_t.Load(int3(id.xy, 0)).rgb, ow_cover(here));
    }
    if (flags & 1) {
        const float4 ui = ow_ui_t.Load(int3(id.xy, 0));
        c.rgb = c.rgb * (1.0 - ui.a) + ui.rgb;
    }
    if (flags & 32768) c = float4(float((flags & 2) && ow_held(int3(id.xy, 0)) ? 14 : ow_path), 1, 0, 1);
    ow_out_u[id.xy] = float4(c.rgb, 1);
}
// One thread per output pixel (search warp, a pure turn).
[numthreads(8, 8, 1)] void cs_own_warp(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= out_size)) return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(out_size);
    float2 src;
    ow_hidden = false;
    ow_path = 0;
    const bool valid = ow_find(uv, src);
    ow_finish(id.xy, uv, src, valid);
}
// Forward splat, in two passes. The warp as one shader needs so many registers - for its rare cases: holes, the ring,
// memory, soft edges, reticles - that the GPU runs few threads at a time and waits on every read (1.0 ms at 4K, where the
// same reads and writes in a small shader take 0.1). cs_own_warp_lean: every output pixel whose full warp is plain - the
// nearest claim around lands on it (ow_claimed's first try), on a spot inside the frame and not held, no reticle under
// it - gets exactly what the full warp gives it; held pixels the game frame's own. The rest of each 2x2 block goes on
// the list (u1) for cs_own_warp_rest, the full warp, packed: whole groups of threads on the rare cases only.
RWByteAddressBuffer ow_list_u : register(u1);  // [0] how many, [16..] per block: x | y << 12 | its pixels left << 24
RWByteAddressBuffer ow_args_u : register(u0);  // cs_own_warp_args: the full warp's dispatch (groups x, y, z)
[numthreads(1, 1, 1)] void cs_own_warp_reset(uint3 id : SV_DispatchThreadID) { ow_list_u.Store(0, 0); }
// The nearest claim around uv, if its surface reaches uv (ow_claimed's first try: the same claim, the same steps).
bool ow_lean_claim(float2 uv, out float2 src) {
    src = uv;
    const int2 c0 = int2(uv * float2(rect.zw));
    uint best = 0;
    int bi = 0;
    [unroll] for (int k = 0; k < 9; ++k) {
        const uint key = ow_claims_t.Load(int3(clamp(c0 + int2(k % 3 - 1, k / 3 - 1), int2(0, 0), int2(rect.zw) - 1), 0));
        if (key > best) { best = key; bi = k; }
    }
    if (best == 0) return false;
    const int2 pb = ow_claimant(clamp(c0 + int2(bi % 3 - 1, bi / 3 - 1), int2(0, 0), int2(rect.zw) - 1), best);
    return ow_refine(uv, pb, float(best >> 22), src);
}
// Output pixel p the plain way: true if written (what ow_scene + ow_finish give it), false: left for the full warp.
bool ow_lean(uint2 p) {
    const int3 pi = int3(p, 0);
    float4 c;
    if ((flags & 2) && ow_held(pi)) {
        c = ow_color_t.Load(pi);
    } else {
        const float2 uv = (float2(p) + 0.5) / float2(out_size);
        float2 src;
        if (!ow_lean_claim(uv, src)) return false;
        const float2 half_px = 0.5 / float2(out_size);
        if (any(src < half_px) || any(src > 1.0 - half_px)) return false;
        if ((flags & 2) && ow_held(int3(clamp(int2(src * float2(out_size)), int2(0, 0), int2(out_size) - 1), 0))) return false;
        c = ow_bilinear(src);
        if (flags & 2048) {
            const float4 here = ow_layer_t.Load(pi), there = ow_layer_t.SampleLevel(ow_linear, src, 0);
            if (ow_cover(there) > 0) return false;
            c.rgb = lerp(c.rgb - there.rgb + here.rgb, ow_color_t.Load(pi).rgb, ow_cover(here));
        }
    }
    if (flags & 1) {
        const float4 ui = ow_ui_t.Load(pi);
        c.rgb = c.rgb * (1.0 - ui.a) + ui.rgb;
    }
    ow_out_u[p] = float4(c.rgb, 1);
    return true;
}
[numthreads(8, 8, 1)] void cs_own_warp_lean(uint3 id : SV_DispatchThreadID) {
    const uint2 p0 = id.xy * 2;
    if (any(p0 >= out_size)) return;
    uint rest = 0;
    [loop] for (uint k = 0; k < 4; ++k) {
        const uint2 p = p0 + uint2(k & 1, k >> 1);
        if (all(p < out_size) && !ow_lean(p)) rest |= 1u << k;
    }
    if (rest) {
        uint i;
        ow_list_u.InterlockedAdd(0, 1, i);
        ow_list_u.Store(16 + 4 * i, id.x | (id.y << 12) | (rest << 24));
    }
}
[numthreads(1, 1, 1)] void cs_own_warp_args(uint3 id : SV_DispatchThreadID) {
    ow_args_u.Store3(0, uint3((ow_list_u.Load(0) * 4 + 63) / 64, 1, 1));  // (a thread per pixel of each block)
}
[numthreads(64, 1, 1)] void cs_own_warp_rest(uint3 id : SV_DispatchThreadID) {
    // (a thread per pixel of each listed block: its four in parallel, those not left idle)
    if (id.x / 4 >= ow_list_u.Load(0)) return;
    const uint e = ow_list_u.Load(16 + 4 * (id.x / 4)), k = id.x & 3u;
    if (!((e >> (24 + k)) & 1u)) return;
    const uint2 p = uint2(e & 4095u, (e >> 12) & 4095u) * 2 + uint2(k & 1, k >> 1);
    const float2 uv = (float2(p) + 0.5) / float2(out_size);
    float2 src;
    ow_hidden = false;
    const bool valid = ow_find(uv, src);
    ow_finish(p, uv, src, valid);
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
// What is visible does not replace remembered scenery clearly farther away (a person walking past, a pole the camera
// moves by): the memory keeps what is behind it, as last seen, for the warp to show where moving the view uncovers it
// (see ow_find); with age it is forgotten, and what is in front written in.
Texture2D<float4> mem_frame_t : register(t0);
Texture2D<float> mem_mask_t : register(t1);
Texture2D<float> mem_depth_t : register(t2);
Texture2D<float4> mem_prev_t : register(t3);
Texture2D<float> mem_prevz_t : register(t4);
RWTexture2D<float4> mem_u : register(u0);
RWTexture2D<float> memz_u : register(u1);
// Nearness of a stored depth (larger = nearer), whichever way round depth is stored: ~32 steps per doubling.
float mem_near(float d) {
    const float x = (flags & 16) ? d : 1.0 - d;
    return log2(max(x, 1e-12)) * 32.0;
}
[numthreads(8, 8, 1)] void cs_memory(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    const float2 s = (float2(id.xy) + 0.5) / float2(grid) * (1.0 + 2.0 * kMemMargin) - kMemMargin;
    const bool visible = all(s >= 0) && all(s <= 1) && mem_mask_t.Load(int3(min(uint2(s * float2(out_size)), out_size - 1), 0)) <= 0.5;
    const float far = (flags & 16) ? 0.0 : 1.0;
    if (visible) {
        const float vz = mem_depth_t.Load(int3(rect.xy + min(uint2(s * float2(rect.zw)), rect.zw - 1), 0));
        // (what was remembered here, carried with the camera at its own depth: kept if clearly farther than what is
        // visible now - more than ~25% - and not too old)
        bool keep = false;
        float4 kept = 0;
        float kz = far;
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
                keep = prev.a + 1.0 < kMemForget && mem_near(d) + 10.0 < mem_near(vz);
                kept = float4(prev.rgb, prev.a + 1.0); kz = d;
            }
        }
        mem_u[id.xy] = keep ? kept : float4(mem_frame_t.SampleLevel(ow_linear, s, 0).rgb, 0);
        memz_u[id.xy] = keep ? kz : vz;
        return;
    }
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
// Level 0 from the picture (the part of it each pixel stands for, averaged: bilinear taps two source pixels apart), the
// coarser levels each from the 2x2 pixels of the level below, in the group's shared memory: one 16x16 group of level 0
// makes one pixel of level 4. (Each level from the picture itself read the whole picture once per level - up to 1024
// taps per pixel at the coarsest, a few threads holding up the pass: 0.7 ms at 4K. Where a level's span is a whole
// number of taps - the render grid half the output, as usual - the averages are the same.)
groupshared float fl_gs[256];
[numthreads(16, 16, 1)] void cs_flow_luma(uint3 id : SV_DispatchThreadID, uint3 gt : SV_GroupThreadID) {
    float v = 0;
    if (all(id.xy < grid)) {
        const float2 span = float2(out_size) / float2(grid);
        const int n = clamp(int(ceil(max(span.x, span.y) * 0.5)), 1, 32);
        [loop] for (int j = 0; j < n; ++j)
            [loop] for (int i = 0; i < n; ++i)
                v += fl_tone(fl_color_t.SampleLevel(ow_linear, (float2(id.xy) + (float2(i, j) + 0.5) / float(n)) * span / float2(out_size), 0).rgb);
        v /= float(n * n);
        fl_luma_u[id.xy] = v;
    }
    const uint g = gt.y * 16 + gt.x;
    fl_gs[g] = v;
    [unroll] for (uint k = 1; k < kFlowLevels; ++k) {
        GroupMemoryBarrierWithGroupSync();
        const uint h = 1u << (k - 1);
        const bool lead = all((gt.xy & ((h << 1) - 1)) == 0);
        float r = 0;
        if (lead) r = 0.25 * (fl_gs[g] + fl_gs[g + h] + fl_gs[g + h * 16] + fl_gs[g + h * 16 + h]);
        GroupMemoryBarrierWithGroupSync();
        if (lead) {
            fl_gs[g] = r;
            const uint2 q = id.xy >> k;
            if (all(q < fl_size(k))) fl_luma_u[fl_origin(k) + q] = r;
        }
    }
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
    if (m > 0.72) tint_u[id.xy] = float4(lerp(c.rgb, float3(1, 0, 1), 0.5), c.a);  // (held: not 0.7, see cs_mask)
    else if ((flags & 1) && s > 0.05) tint_u[id.xy] = float4(lerp(c.rgb, float3(0, 1, 0), 0.5 * saturate(s / 0.6)), c.a);
}
// Sequence recording (development, Renderer::record_frame): the game's picture shrunk by a whole factor (groups_x) to about
// 1080 lines - a box average, grid its size, out_size the picture's - and with flag 1 the HUD layer's coverage likewise.
Texture2D<float4> rec_colour_t : register(t0);
Texture2D<float4> rec_ui_t : register(t1);
RWTexture2D<float4> rec_colour_u : register(u0);
RWTexture2D<unorm float> rec_ui_u : register(u1);
[numthreads(8, 8, 1)] void cs_rec_down(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= grid)) return;
    const uint f = max(groups_x, 1u);
    float3 c = 0;
    float a = 0;
    for (uint y = 0; y < f; ++y)
        for (uint x = 0; x < f; ++x) {
            const int3 q = int3(min(id.xy * f + uint2(x, y), out_size - 1), 0);
            c += rec_colour_t.Load(q).rgb;
            if (flags & 1) a += rec_ui_t.Load(q).a;
        }
    rec_colour_u[id.xy] = float4(c / float(f * f), 1);
    rec_ui_u[id.xy] = (flags & 1) ? a / float(f * f) : 0.0;
}
)";

// Descriptor layout in the shader-visible heap.
constexpr UINT kSrcSrv = 0;         // + slot * kTexCount + kind: shared textures
constexpr UINT kPrivMax = 80;       // private textures at most
constexpr UINT kPrivUav = 40;       // + private id
constexpr UINT kPrivSrv = kPrivUav + kPrivMax;  // + private id
// Compute pass tables (6 SRVs t0-t5, 2 UAVs u0-u1 each).
constexpr UINT kXSrvCount = 6;
constexpr UINT kX = kPrivSrv + kPrivMax;  // first table
constexpr UINT kXAnalyzeSrv = kX + 0, kXAnalyzeUav = kX + 6, kXReduceUav = kX + 8;
constexpr UINT kXKindUav = kX + 10, kXHudAfterSrv = kX + 12;
constexpr UINT kXRetCellsSrv = kX + 18, kXRetCellsUav = kX + 24, kXRetPickUav = kX + 26, kXRetShiftUav = kX + 28, kXRetArgsUav = kX + 30;
constexpr UINT kXHudSrv = kX + 32, kXHudUav = kX + 38, kXMaskSrv = kX + 40, kXMaskUav = kX + 46, kXSampleSrv = kX + 48, kXSampleUav = kX + 54;
constexpr UINT kXTintSrv = kX + 56, kXTintUav = kX + 62, kXSceneSrv = kX + 64, kXSceneUav = kX + 70;
constexpr UINT kXHudlessSrv = kX + 72, kXHudlessUav = kX + 78;
constexpr UINT kXAttSrv = kX + 80, kXAttUav = kX + 86, kXWorldUav = kX + 88, kXFillSrv = kX + 90, kXFillUav = kX + 96;
constexpr UINT kXMemSrv = kX + 98, kXMemUav = kX + 104, kXRowsSrv = kX + 106, kXRowsUav = kX + 112;
constexpr UINT kXRampSrv = kX + 114, kXRampUav = kX + 120;
constexpr UINT kXRetEnclSrv = kX + 122, kXRetEnclUav = kX + 128, kXRetSrv = kX + 130, kXRetUav = kX + 136;
constexpr UINT kXRetAlignSrv = kX + 138, kXRetAlignUav = kX + 144;
constexpr UINT kRetTrackN = 17 * 17;  // shifts tried by cs_ret_align (+-8 px)
// Own motion estimation: one table of 6 SRVs + 2 UAVs per pass and direction (see estimate_motion).
constexpr UINT kXFlow = kX + 146;
enum FlowTable : UINT { kFlowLuma, kFlowFeat, kFlowSearch, kFlowMedianAB, kFlowMedianBA, kFlowLkAB, kFlowLkBA, kFlowMotion, kFlowFillAB, kFlowFillBA, kFlowTables };
// Passes on the second root signature (the XPAR warp and the moving objects): 12 SRVs t0-t11 and 2 UAVs
// per table.
constexpr UINT kWSrvCount = 16, kWTableSize = kWSrvCount + 2;
// Combined HUD detection: the learned score kept alongside (cs_hud: u0 the score, u1 the counts) and its tiles (6 + 2).
constexpr UINT kXLearnedUav = kXFlow + kFlowTables * 8, kXTilesSrv = kXLearnedUav + 2, kXTilesUav = kXTilesSrv + 6;
// HUD learning (score_hud_probe): the probe into tiles, the sums cleared, the score (6 + 2 each).
constexpr UINT kXProbeTilesSrv = kXTilesUav + 2, kXProbeTilesUav = kXProbeTilesSrv + 6, kXProbeScoreSrv = kXProbeTilesUav + 2,
               kXProbeScoreUav = kXProbeScoreSrv + 6;
// The HUD layer's coverage where it can not be checked (cs_layer_check on the layer alone, see build_no_warp_mask; 6 + 2).
constexpr UINT kXCoverSrv = kXProbeScoreUav + 2, kXCoverUav = kXCoverSrv + 6;
// Sequence recording (record_frame): a table per slot in flight (6 + 2 each).
constexpr UINT kXRec = kXCoverUav + 2;
constexpr UINT kW = kXRec + Renderer::kRecSlots * 8;
enum WTable : UINT { kWWarp, kWShown, kWClear, kWSplat, kWMove, kWCopyColour, kWCopyDepth, kWJoinInit, kWJoinAB, kWJoinBA, kWJoinFinal,
                     kWGenDepth0, kWGenDepth1, kWGenDepth2, kWGenUi0, kWGenUi1, kWGenUi2, kWFgMotion, kWFgInverse, kWClaim,
                     kWStill, kWHoldInit, kWHoldAB, kWHoldBA, kWHoldFinal, kWHoldFill, kWHoldFillFinal, kWGrowInit, kWGrowAB, kWGrowBA,
                     kWGrowFinal, kWWarpArgs, kWShareMeasure, kWShareH, kWShareV, kWShareBins, kWShareUp, kWShareResid, kWResidBlurH, kWResidBlurV, kWOwnUp, kWFgImages,
                     kWTables = kWFgImages + 5 * 3 };
// XPAR frame generation: each image's own tables (descriptors are read when the GPU runs the pass, so one set reused
// for every image would have them all write the last one).
constexpr UINT kWFgClear = 0, kWFgSplatCur = 1, kWFgSplatPrev = 2, kWFgResolve = 3, kWFgDepth = 4;
constexpr UINT kWFg(int image, UINT pass) { return kWFgImages + UINT(image) * 5 + pass; }
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
    D3D12_QUERY_HEAP_DESC ptq{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, kPassStamps * kRing, 0};
    device_->CreateQueryHeap(&ptq, IID_PPV_ARGS(&pass_stamps_));
    {
        D3D12_RESOURCE_DESC pd = bd; pd.Width = kPassStamps * kRing * 8;
        device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &pd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&pass_readback_));
    }
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
    apply_color_space(color_space);
    if (FAILED(DCompositionCreateDevice(nullptr, IID_PPV_ARGS(&dcomp_))) ||
        FAILED(dcomp_->CreateTargetForHwnd(window, TRUE, &target_)) || FAILED(dcomp_->CreateVisual(&visual_)) ||
        FAILED(visual_->SetContent(swapchain_.Get())) || FAILED(target_->SetRoot(visual_.Get())) || FAILED(dcomp_->Commit())) {
        error = "DirectComposition setup"; return false;
    }
    create_swapchain_views();
    return true;
}

// The game's colour space on our swap chain (HDR10 PQ, scRGB, sRGB), if the swap chain takes it - noted for the log
// either way: a refused HDR colour space shows the game's HDR pixels as SDR (grey, washed out).
void Renderer::apply_color_space(std::uint32_t color_space) {
    UINT support = 0;
    const auto cs = static_cast<DXGI_COLOR_SPACE_TYPE>(color_space);
    const bool can = SUCCEEDED(swapchain_->CheckColorSpaceSupport(cs, &support)) && (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT);
    const bool set = can && SUCCEEDED(swapchain_->SetColorSpace1(cs));
    color_space_ = color_space;
    char note[200];
    std::snprintf(note, sizeof(note), "output: format %u, colour space %u (%s) - %s", unsigned(swap_format_), unsigned(color_space),
                  color_space == 12 ? "HDR10 PQ" : color_space == 1 ? "scRGB" : color_space == 0 ? "sRGB" : "other",
                  set ? "set on the swap chain" : can ? "SetColorSpace1 FAILED" : "NOT SUPPORTED by the swap chain (shown as sRGB)");
    notes_.push_back(note);
}

// The game changed its picture format or colour space (HDR switched on or off, often after the swap chain was set
// up): the swap chain's buffers, the final pass and the colour space follow.
bool Renderer::set_output(DXGI_FORMAT game_format, std::uint32_t color_space) {
    const DXGI_FORMAT format = swapchain_format(game_format);
    if (format == swap_format_ && color_space == color_space_) return true;
    wait_idle();
    if (format != swap_format_) {
        if (FAILED(swapchain_->ResizeBuffers(3, width_, height_, format, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT))) {
            notes_.push_back("output: could not switch the swap chain to format " + std::to_string(unsigned(format)));
            return false;
        }
        swap_format_ = format;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC gd{};
        gd.pRootSignature = root_.Get();
        gd.VS = {blit_vs_->GetBufferPointer(), blit_vs_->GetBufferSize()};
        gd.PS = {blit_ps_->GetBufferPointer(), blit_ps_->GetBufferSize()};
        gd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        gd.SampleMask = UINT_MAX;
        gd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID; gd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        gd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        gd.NumRenderTargets = 1; gd.RTVFormats[0] = swap_format_;
        gd.SampleDesc.Count = 1;
        ComPtr<ID3D12PipelineState> blit;
        if (SUCCEEDED(device_->CreateGraphicsPipelineState(&gd, IID_PPV_ARGS(&blit)))) blit_ = blit;
        create_swapchain_views();
    }
    apply_color_space(color_space);
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
    blit_vs_ = vs; blit_ps_ = ps;

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
        {"cs_hudless_score", &cs_hudless_score_}, {"cs_hudless_fill", &cs_hudless_fill_}, {"cs_layer_hud", &cs_layer_hud_}, {"cs_layer_check_clear", &cs_layer_check_clear_}, {"cs_layer_check", &cs_layer_check_},
        {"cs_hud_count", &cs_hud_count_}, {"cs_clear_counts", &cs_clear_counts_}, {"cs_sample", &cs_sample_}, {"cs_tint", &cs_tint_}, {"cs_rec_down", &cs_rec_down_}, {"cs_hud_tiles", &cs_hud_tiles_}, {"cs_hud_spread", &cs_hud_spread_},
        {"cs_scene_clear", &cs_scene_clear_}, {"cs_scene_accum", &cs_scene_accum_}, {"cs_scene_finish", &cs_scene_finish_},
        {"cs_scene_tiles", &cs_scene_tiles_}, {"cs_scene_hud", &cs_scene_hud_}, {"cs_scene_fill", &cs_scene_fill_}, {"cs_scene_grey", &cs_scene_grey_},
        {"cs_scene_grey_finish", &cs_scene_grey_finish_}, {"cs_scene_wash", &cs_scene_wash_}, {"cs_scene_wash_finish", &cs_scene_wash_finish_},
        {"cs_attached", &cs_attached_}, {"cs_probe_tiles", &cs_probe_tiles_}, {"cs_probe_clear", &cs_probe_clear_}, {"cs_probe_score", &cs_probe_score_}, {"cs_memory", &cs_memory_}, {"cs_ramp_rows", &cs_ramp_rows_}, {"cs_ramp", &cs_ramp_},
        {"cs_flow_luma", &cs_flow_luma_}, {"cs_flow_feat", &cs_flow_feat_}, {"cs_flow_search", &cs_flow_search_}, {"cs_flow_lk", &cs_flow_lk_},
        {"cs_flow_median", &cs_flow_median_}, {"cs_flow_motion", &cs_flow_motion_}, {"cs_flow_fill", &cs_flow_fill_},
        {"cs_ret_cells", &cs_ret_cells_}, {"cs_ret_enclose", &cs_ret_enclose_}, {"cs_ret_layer", &cs_ret_layer_},
        {"cs_ret_align", &cs_ret_align_}, {"cs_ret_pick", &cs_ret_pick_}, {"cs_ret_args", &cs_ret_args_}, {"cs_ret_shift", &cs_ret_shift_}};
    for (auto& x : xs) {
        auto code = compile(x.entry, "cs_5_0", error, kShadersX);
        if (!code) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC xd{};
        xd.pRootSignature = root_x_.Get();
        xd.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (FAILED(device_->CreateComputePipelineState(&xd, IID_PPV_ARGS(x.pso->ReleaseAndGetAddressOf())))) { error = std::string(x.entry) + " pso"; return false; }
    }
    // The XPAR warp and the moving objects: the same 32 constants (b0), 20 more (b1), 14 SRVs, 2 UAVs.
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
        {"cs_own_warp", &cs_own_warp_}, {"cs_own_warp_reset", &cs_own_warp_reset_}, {"cs_own_warp_lean", &cs_own_warp_lean_}, {"cs_own_warp_args", &cs_own_warp_args_}, {"cs_own_warp_rest", &cs_own_warp_rest_}, {"cs_warp_claims_clear", &cs_warp_claims_clear_}, {"cs_warp_claim", &cs_warp_claim_}, {"cs_obj_fix", &cs_obj_fix_}, {"cs_obj_move", &cs_obj_move_}, {"cs_gen_depth", &cs_gen_depth_}, {"cs_gen_unui", &cs_gen_unui_}, {"cs_obj_join_init", &cs_obj_join_init_}, {"cs_obj_join", &cs_obj_join_}, {"cs_obj_join_final", &cs_obj_join_final_}, {"cs_still", &cs_still_}, {"cs_hold_fill", &cs_hold_fill_}, {"cs_grow_init", &cs_grow_init_}, {"cs_grow_clear", &cs_grow_clear_}, {"cs_grow", &cs_grow_}, {"cs_grow_final", &cs_grow_final_}, {"cs_obj_clear", &cs_obj_clear_}, {"cs_obj_clear_tiles", &cs_obj_clear_tiles_}, {"cs_obj_splat", &cs_obj_splat_}, {"cs_share_measure", &cs_share_measure_}, {"cs_share_blur", &cs_share_blur_}, {"cs_share_bins", &cs_share_bins_}, {"cs_share_final", &cs_share_final_}, {"cs_share_up", &cs_share_up_}, {"cs_share_resid", &cs_share_resid_}, {"cs_resid_blur", &cs_resid_blur_}, {"cs_own_up", &cs_own_up_},
        {"cs_copy4", &cs_copy4_}, {"cs_copy1", &cs_copy1_}, {"cs_fg_motion", &cs_fg_motion_}, {"cs_fg_inverse", &cs_fg_inverse_},
        {"cs_fg_clear", &cs_fg_clear_}, {"cs_fg_splat_cur", &cs_fg_splat_cur_}, {"cs_fg_splat_prev", &cs_fg_splat_prev_},
        {"cs_fg_resolve", &cs_fg_resolve_}, {"cs_fg_depth", &cs_fg_depth_}};
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
    bd.Width = 16;  // the held area's size (cs_grow_init)
    if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&grow_counts_)))) {
        error = "held area buffer"; return false;
    }
    bd.Width = 16;  // the HUD layer's check (cs_layer_check)
    if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&layer_counts_)))) {
        error = "HUD layer check buffer"; return false;
    }
    bd.Width = (kRetTrackN + 8) * 4;  // sight reticles: a sum per shift tried, the weight, the shift chosen (cs_ret_align), the window cells (cs_ret_args)
    if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ret_buf_)))) {
        error = "reticle tracking buffer"; return false;
    }
    bd.Width = 300 * 4;  // the follow share's depth bands (cs_share_bins), and its diagnostics
    if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&share_bins_)))) {
        error = "follow share bands buffer"; return false;
    }
    bd.Width = 32;  // ...and the dispatches of its passes over the whole screen (cs_ret_args)
    if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&ret_args_)))) {
        error = "reticle dispatch buffer"; return false;
    }
    if (!dispatch_sig_) {
        D3D12_INDIRECT_ARGUMENT_DESC arg{}; arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC sd{}; sd.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS); sd.NumArgumentDescs = 1; sd.pArgumentDescs = &arg;
        if (FAILED(device_->CreateCommandSignature(&sd, nullptr, IID_PPV_ARGS(&dispatch_sig_)))) { error = "dispatch signature"; return false; }
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
    bd.Width = kRing * 4;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&grow_readback_)))) {
        error = "held area readback"; return false;
    }
    bd.Width = kRing * 32;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&probe_readback_)))) {
        error = "HUD learning readback"; return false;
    }
    {
        D3D12_RESOURCE_DESC pb = bd;
        pb.Width = 32; pb.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &pb, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&probe_sums_)))) {
            error = "HUD learning buffer"; return false;
        }
    }
    bd.Width = kRing * 8;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&layer_readback_)))) {
        error = "HUD layer check readback"; return false;
    }
    bd.Width = kRing * 16;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&hud_readback_)))) {
        error = "HUD counter readback"; return false;
    }
    bd.Width = kRing * 32;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&ret_args_readback_)))) {
        error = "reticle dispatch readback"; return false;
    }
    bd.Width = kRing * kShareDiagBytes;
    if (FAILED(device_->CreateCommittedResource(&rbh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&share_readback_)))) {
        error = "follow share diagnostics readback"; return false;
    }
    return true;
}

void Renderer::set_x_srv(UINT index, PrivateId id, bool from_shown) {
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    const auto& p = from_shown ? shown(id) : private_[id];
    d.Format = p.texture ? p.format : DXGI_FORMAT_R8G8B8A8_UNORM;
    // (the UI layer kept as its coverage, one channel: read as (0, 0, 0, a), as a UI layer)
    if (id == kPUi && p.format == DXGI_FORMAT_R8_UNORM)
        d.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
            D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
            D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0);
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

void Renderer::w_dispatch_indirect(ID3D12PipelineState* pso, const void* constants, const float w[20], UINT table, ID3D12Resource* args) {
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetComputeRootSignature(root_w_.Get());
    list_->SetPipelineState(pso);
    list_->SetComputeRoot32BitConstants(0, 32, constants, 0);
    list_->SetComputeRoot32BitConstants(1, 20, w, 0);
    list_->SetComputeRootDescriptorTable(2, gpu(kWSrv(table)));
    list_->SetComputeRootDescriptorTable(3, gpu(kWUav(table)));
    list_->ExecuteIndirect(dispatch_sig_.Get(), 1, args, 0, nullptr, 0);
}

// The full warp's list (cs_own_warp_lean: a count, then one entry per 2x2 block of the output) and its dispatch.
bool Renderer::ensure_warp_list(UINT ow, UINT oh) {
    const UINT64 bytes = 16 + 4ull * ((ow + 1) / 2) * ((oh + 1) / 2);
    if (warp_list_ && warp_list_->GetDesc().Width >= bytes && warp_args_ && dispatch_sig_) return true;
    D3D12_HEAP_PROPERTIES def{}; def.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = bytes; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (!warp_list_ || warp_list_->GetDesc().Width < bytes) {
        warp_list_old_ = warp_list_;  // (a refresh in flight may still use it)
        warp_list_.Reset();
        if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&warp_list_)))) return false;
    }
    if (!warp_args_) {
        bd.Width = 16;
        if (FAILED(device_->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&warp_args_)))) return false;
    }
    if (!dispatch_sig_) {
        D3D12_INDIRECT_ARGUMENT_DESC arg{}; arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC sd{}; sd.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS); sd.NumArgumentDescs = 1; sd.pArgumentDescs = &arg;
        if (FAILED(device_->CreateCommandSignature(&sd, nullptr, IID_PPV_ARGS(&dispatch_sig_)))) return false;
    }
    return true;
}

void Renderer::x_dispatch(ID3D12PipelineState* pso, const void* constants, UINT srv_table, UINT uav_table, UINT groups_x, UINT groups_y,
                          ID3D12Resource* args, UINT64 args_offset) {
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetComputeRootSignature(root_x_.Get());
    list_->SetPipelineState(pso);
    // (flags, the last constant, bit 32768: the game's picture is linear scRGB - see seen() in the shaders)
    std::uint32_t c[32];
    std::memcpy(c, constants, sizeof(c));
    if (color_space_ == 1) c[31] |= 32768u;
    list_->SetComputeRoot32BitConstants(0, 32, c, 0);
    list_->SetComputeRootDescriptorTable(1, gpu(srv_table));
    list_->SetComputeRootDescriptorTable(2, gpu(uav_table));
    if (args) list_->ExecuteIndirect(dispatch_sig_.Get(), 1, args, args_offset, nullptr, 0);
    else list_->Dispatch(groups_x, groups_y, 1);
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
                              bool depth_inverted, bool near_rule, bool turn_rule, std::uint64_t frame, float floor_pivot_depth) {
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
    c.flags = (scale_valid ? 1u : 0u) | (depth_inverted ? 16u : 0u) | (near_rule ? 0u : 64u) | (turn_rule ? 128u : 0u) |
              (floor_pivot_depth > 0 ? 512u : 0u) | (rig_check_ ? 2048u : 0u);
    c.alpha = floor_pivot_depth;  // (the floor release's pivot depth; alpha is unused by this pass otherwise)
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

int Renderer::generate_frames(const IngestedSource& src, int count, const float clip_to_prev_clip[16], const float view_to_clip[16],
                              float scale_x, float scale_y, bool use_ui_tags, bool flow, bool hud) {
    count = std::clamp(count, 0, kMaxGenerated);
    if (!count || !src.valid || !src.has_depth || !src.has_motion || !history_valid_) return 0;
    // (the HUD-less picture with the UI layer the warp puts back over it; HUD-less without one would lose the HUD)
    if (src.has_hudless && !(use_ui_tags && src.has_ui)) return 0;
    const PrivateId colour = src.has_hudless ? kPHudless : kPBackbuffer;
    const auto &cur = private_[colour], &prev = private_[kPPrevColour], &z1 = private_[kPDepth], &z0 = private_[kPPrevDepth], &mv = private_[kPMotion];
    if (!cur.texture || !prev.texture || !z1.texture || !z0.texture || !mv.texture || prev.width != cur.width || prev.height != cur.height ||
        prev.format != cur.format || z0.width != z1.width || z0.height != z1.height)
        return 0;
    const UINT ow = cur.width, oh = cur.height, dw = z1.width, dh = z1.height;
    if (!ensure_private(kPFgMotion, dw, dh, DXGI_FORMAT_R32G32_FLOAT) || !ensure_private(kPFgInv, dw, dh, DXGI_FORMAT_R32_UINT) ||
        !ensure_private(kPFgK1, dw, dh, DXGI_FORMAT_R32_UINT) || !ensure_private(kPFgK0, dw, dh, DXGI_FORMAT_R32_UINT))
        return 0;
    for (int i = 0; i < count; ++i)
        if (!ensure_private(static_cast<PrivateId>(kPGen0 + i), ow, oh, cur.format) ||
            !ensure_private(static_cast<PrivateId>(kPGenZ0 + i), dw, dh, DXGI_FORMAT_R32_FLOAT))
            return 0;
    Mat4 to_prev{}, from_prev{};
    std::memcpy(to_prev.data(), clip_to_prev_clip, sizeof(float) * 16);
    if (!mat_inverse(to_prev, from_prev)) return 0;
    XConstants c{};
    std::memcpy(c.clip_to_prev, clip_to_prev_clip, sizeof(c.clip_to_prev));
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, dw, dh};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, dw - r.x); c.rect[3] = std::min(r.h, dh - r.y);
    c.grid[0] = dw; c.grid[1] = dh;
    c.mv_size[0] = mv.width; c.mv_size[1] = mv.height;
    c.out_size[0] = ow; c.out_size[1] = oh;
    c.mv_scale[0] = scale_x; c.mv_scale[1] = scale_y;
    const bool use_flow = flow && private_[kPFlowB].texture;
    const bool use_hud = hud && private_[kPMask].texture && private_[kPMask].width == ow && private_[kPMask].height == oh;
    c.flags = (use_flow ? 2u : 0u) | (use_hud ? 4u : 0u);
    float wc[20] = {};
    std::memcpy(wc, from_prev.data(), sizeof(float) * 16);  // (the previous frame's clip space -> this frame's)
    projection_terms(view_to_clip, wc + 16);
    auto bind = [&](UINT t) {
        for (UINT k = 0; k < kWSrvCount; ++k) set_x_srv(kWSrv(t) + k, kPDepth);
        set_x_srv(kWSrv(t) + 0, colour);
        set_x_srv(kWSrv(t) + 1, kPPrevColour);
        set_x_srv(kWSrv(t) + 3, kPPrevDepth);
        set_x_srv(kWSrv(t) + 4, kPMotion);
        if (use_hud) set_x_srv(kWSrv(t) + 5, kPMask);
        if (use_flow) set_x_srv(kWSrv(t) + 6, kPFlowB);
        set_x_srv(kWSrv(t) + 7, kPFgMotion);
        set_x_srv(kWSrv(t) + 8, kPFgInv);
        set_x_srv(kWSrv(t) + 9, kPFgK1);
        set_x_srv(kWSrv(t) + 10, kPFgK0);
    };
    auto uavs = [&](UINT t, PrivateId a, PrivateId b) { set_x_uav(kWUav(t) + 0, a); set_x_uav(kWUav(t) + 1, b); };
    auto uav_barrier = [&](PrivateId id) {
        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource = private_[id].texture.Get();
        list_->ResourceBarrier(1, &b);
    };
    const UINT rx = (c.rect[2] + 7) / 8, ry = (c.rect[3] + 7) / 8;
    // This frame's motion (and the claims on the previous frame cleared), then those claims.
    bind(kWFgMotion); uavs(kWFgMotion, kPFgMotion, kPFgInv);
    transition(private_[kPFgMotion], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(private_[kPFgInv], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_fg_motion_.Get(), &c, wc, kWFgMotion, rx, ry);
    transition(private_[kPFgMotion], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    uav_barrier(kPFgInv);
    bind(kWFgInverse); uavs(kWFgInverse, kPFgInv, kPFgInv);
    w_dispatch(cs_fg_inverse_.Get(), &c, wc, kWFgInverse, rx, ry);
    transition(private_[kPFgInv], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // Each image: the claims of both frames at its moment, then its picture and depth.
    for (int i = 0; i < count; ++i) {
        c.alpha = float(i + 1) / float(count + 1);
        const auto gen = static_cast<PrivateId>(kPGen0 + i), genz = static_cast<PrivateId>(kPGenZ0 + i);
        transition(private_[kPFgK1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(private_[kPFgK0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const UINT t_clear = kWFg(i, kWFgClear), t_cur = kWFg(i, kWFgSplatCur), t_prev = kWFg(i, kWFgSplatPrev), t_res = kWFg(i, kWFgResolve),
                   t_depth = kWFg(i, kWFgDepth);
        bind(t_clear); uavs(t_clear, kPFgK1, kPFgK0);
        w_dispatch(cs_fg_clear_.Get(), &c, wc, t_clear, (dw + 7) / 8, (dh + 7) / 8);
        uav_barrier(kPFgK1); uav_barrier(kPFgK0);
        bind(t_cur); uavs(t_cur, kPFgK1, kPFgK1);
        w_dispatch(cs_fg_splat_cur_.Get(), &c, wc, t_cur, rx, ry);
        bind(t_prev); uavs(t_prev, kPFgK0, kPFgK0);
        w_dispatch(cs_fg_splat_prev_.Get(), &c, wc, t_prev, rx, ry);
        transition(private_[kPFgK1], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(private_[kPFgK0], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        bind(t_res); uavs(t_res, gen, gen);
        transition(private_[gen], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        w_dispatch(cs_fg_resolve_.Get(), &c, wc, t_res, (ow + 7) / 8, (oh + 7) / 8);
        transition(private_[gen], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        bind(t_depth); uavs(t_depth, genz, genz);
        transition(private_[genz], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        w_dispatch(cs_fg_depth_.Get(), &c, wc, t_depth, (dw + 7) / 8, (dh + 7) / 8);
        transition(private_[genz], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    return count;
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

void Renderer::hold_characters(const IngestedSource& src, const float clip_to_prev_clip[16], const float view_to_clip[16], bool scale_valid,
                                bool depth_inverted) {
    const auto& depth = private_[kPDepth];
    if (!src.has_depth || !src.has_motion || !private_[kPObject].texture || !depth.texture) return;
    const UINT w = depth.width, h = depth.height;
    if (private_[kPObject].width != w || private_[kPObject].height != h) return;
    const bool fresh = !private_[kPStill].texture || private_[kPStill].width != w || private_[kPStill].height != h;
    if (!ensure_private(kPStill, w, h, DXGI_FORMAT_R16_FLOAT) || !ensure_private(kPObjKindA, w, h, DXGI_FORMAT_R16_FLOAT) ||
        !ensure_private(kPObjKindB, w, h, DXGI_FORMAT_R16_FLOAT))
        return;
    float wc[20] = {};
    projection_terms(view_to_clip, wc + 16);
    // How much each spot has stayed in place lately (cs_still).
    XConstants c{};
    std::memcpy(c.clip_to_prev, clip_to_prev_clip, sizeof(c.clip_to_prev));
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    c.grid[0] = w; c.grid[1] = h;
    c.flags = (scale_valid ? 1u : 0u) | (depth_inverted ? 16u : 0u) | (fresh || still_reset_ ? 8u : 0u);
    if (own_motion_ && private_[kPFlowB].texture) { c.flags |= 2u; c.out_size[0] = std::max(1u, w / 2); c.out_size[1] = std::max(1u, h / 2); }
    still_reset_ = false;
    for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWStill) + i, kPDepth);
    set_x_srv(kWSrv(kWStill) + 0, kPObject);
    set_x_srv(kWSrv(kWStill) + 1, kPDepth);
    if (c.flags & 2u) set_x_srv(kWSrv(kWStill) + 2, kPFlowB);
    set_x_uav(kWUav(kWStill) + 0, kPStill); set_x_uav(kWUav(kWStill) + 1, kPStill);
    transition(private_[kPStill], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_still_.Get(), &c, wc, kWStill, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
    transition(private_[kPStill], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // One character, held whole (cs_obj_join with the score): written back into the analysis before the mask is built.
    XConstants j{};
    j.grid[0] = w; j.grid[1] = h;
    j.rect[1] = 1;
    auto table = [&](UINT t, PrivateId kind_in, PrivateId out) {
        for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(t) + i, kPDepth);
        set_x_srv(kWSrv(t) + 0, kPObject);
        set_x_srv(kWSrv(t) + 1, kind_in);
        set_x_srv(kWSrv(t) + 2, kPDepth);
        set_x_srv(kWSrv(t) + 3, kPStill);
        set_x_uav(kWUav(t) + 0, out); set_x_uav(kWUav(t) + 1, out);
    };
    auto barrier = [&](PrivateId id) {
        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource = private_[id].texture.Get();
        list_->ResourceBarrier(1, &b);
    };
    table(kWHoldInit, kPObjKindB, kPObjKindA);
    table(kWHoldAB, kPObjKindA, kPObjKindB);
    table(kWHoldBA, kPObjKindB, kPObjKindA);
    transition(private_[kPObjKindA], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_obj_join_init_.Get(), &j, wc, kWHoldInit, (w + 7) / 8, (h + 7) / 8);
    bool in_a = true;
    for (const UINT step : {64u, 32u, 16u, 8u, 4u, 2u, 1u}) {
        const PrivateId from = in_a ? kPObjKindA : kPObjKindB, to = in_a ? kPObjKindB : kPObjKindA;
        transition(private_[from], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(private_[to], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        barrier(to);
        j.rect[0] = step;
        w_dispatch(cs_obj_join_.Get(), &j, wc, in_a ? kWHoldAB : kWHoldBA, (w + 7) / 8, (h + 7) / 8);
        in_a = !in_a;
    }
    const PrivateId result = in_a ? kPObjKindA : kPObjKindB;
    transition(private_[result], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    table(kWHoldFinal, result, kPObject);
    transition(private_[kPObject], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_obj_join_final_.Get(), &j, wc, kWHoldFinal, (w + 7) / 8, (h + 7) / 8);
    transition(private_[kPObject], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // Holes in what is held (cs_hold_fill): 3% of the render height, in 12 steps.
    const UINT reach = std::max(4u, static_cast<UINT>(std::lround(0.03 * double(c.rect[3]))));
    XConstants f = j;
    f.rect[0] = std::max(1u, (reach + 11) / 12); f.rect[1] = std::min(12u, reach);
    const PrivateId scratch = in_a ? kPObjKindB : kPObjKindA;  // (the one the join did not end in)
    table(kWHoldFill, kPDepth, scratch);
    transition(private_[scratch], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_hold_fill_.Get(), &f, wc, kWHoldFill, (w / 2 + 8) / 8, (h / 2 + 8) / 8);
    transition(private_[scratch], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    table(kWHoldFillFinal, scratch, kPObject);
    transition(private_[kPObject], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_obj_join_final_.Get(), &j, wc, kWHoldFillFinal, (w + 7) / 8, (h + 7) / 8);
    transition(private_[kPObject], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // Growing along depth where the motion is unknown (cs_grow_*): only with XPAR's own motion estimate, on the half grid
    // (its size); as many passes as a tenth of the held area's size (its cell count, read back a few frames ago).
    const UINT hw = std::max(1u, w / 2), hh = std::max(1u, h / 2);
    // (the estimate's texture holds its level 0 - half the render grid - at its origin, the coarser levels beside and below)
    if (!(c.flags & 2u) || private_[kPFlowB].width < hw || private_[kPFlowB].height < hh ||
        !ensure_private(kPGrowA, hw, hh, DXGI_FORMAT_R16G16_FLOAT) || !ensure_private(kPGrowB, hw, hh, DXGI_FORMAT_R16G16_FLOAT))
        return;
    const UINT passes = std::min(24u, static_cast<UINT>(std::lround(0.1 * std::sqrt(double(grow_cells_)))));
    XConstants g{};
    g.grid[0] = w; g.grid[1] = h;
    g.out_size[0] = hw; g.out_size[1] = hh;
    g.flags = c.flags & 16u;
    g.groups_x = 1;
    auto gtable = [&](UINT t, PrivateId half_in, PrivateId out) {
        for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(t) + i, kPDepth);
        set_x_srv(kWSrv(t) + 0, kPObject);
        set_x_srv(kWSrv(t) + 2, kPDepth);
        set_x_srv(kWSrv(t) + 4, kPFlowB);
        set_x_srv(kWSrv(t) + 5, half_in);
        set_x_uav(kWUav(t) + 0, out);
    };
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
    raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.Buffer.NumElements = 4; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    gtable(kWGrowInit, kPGrowB, kPGrowA);
    XConstants gi = g;
    if (private_[kPMask].texture) { set_x_srv(kWSrv(kWGrowInit) + 1, kPMask); gi.flags |= 32u; }  // (not from the HUD)
    device_->CreateUnorderedAccessView(grow_counts_.Get(), nullptr, &raw, cpu(kWUav(kWGrowInit) + 1));
    D3D12_RESOURCE_BARRIER counts{}; counts.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; counts.UAV.pResource = grow_counts_.Get();
    transition(private_[kPGrowA], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_grow_clear_.Get(), &g, wc, kWGrowInit, 1, 1);
    list_->ResourceBarrier(1, &counts);
    w_dispatch(cs_grow_init_.Get(), &gi, wc, kWGrowInit, (hw + 7) / 8, (hh + 7) / 8);
    auto to_copy = transition_barrier(grow_counts_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    list_->CopyBufferRegion(grow_readback_.Get(), UINT64(frame_index_) * 4, grow_counts_.Get(), 0, 4);
    std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
    list_->ResourceBarrier(1, &to_copy);
    grow_pending_[frame_index_] = true;
    gtable(kWGrowAB, kPGrowA, kPGrowB);
    gtable(kWGrowBA, kPGrowB, kPGrowA);
    for (UINT kWG : {kWGrowAB, kWGrowBA}) set_x_uav(kWUav(kWG) + 1, kWG == kWGrowAB ? kPGrowB : kPGrowA);
    set_x_uav(kWUav(kWGrowInit) + 0, kPGrowA);
    bool grow_a = true;  // (the result is in A)
    for (UINT n = 0; n < passes; ++n) {
        const PrivateId from = grow_a ? kPGrowA : kPGrowB, to = grow_a ? kPGrowB : kPGrowA;
        transition(private_[from], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(private_[to], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        w_dispatch(cs_grow_.Get(), &g, wc, grow_a ? kWGrowAB : kWGrowBA, (hw + 7) / 8, (hh + 7) / 8);
        grow_a = !grow_a;
    }
    const PrivateId grown = grow_a ? kPGrowA : kPGrowB;
    transition(private_[grown], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gtable(kWGrowFinal, grown, kPObject);
    set_x_uav(kWUav(kWGrowFinal) + 1, kPObject);
    transition(private_[kPObject], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_grow_final_.Get(), &g, wc, kWGrowFinal, (w + 7) / 8, (h + 7) / 8);
    transition(private_[kPObject], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

void Renderer::follow_share(const IngestedSource& src, const float clip_to_prev_clip[16], bool scale_valid, bool depth_inverted,
                            bool whole_motion, int share_mode) {
    share_built_ = false;
    const auto& depth = private_[kPDepth];
    if (!src.has_depth || !src.has_motion || !private_[kPObject].texture || !depth.texture) return;
    const UINT w = depth.width, h = depth.height;
    if (private_[kPObject].width != w || private_[kPObject].height != h) return;
    float wc[20] = {};
    XConstants c{};
    std::memcpy(c.clip_to_prev, clip_to_prev_clip, sizeof(c.clip_to_prev));
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    // (the cells: one per 2x2 block of the rect's render pixels - see share_cells)
    const UINT cw = (c.rect[2] + 1) / 2, ch = (c.rect[3] + 1) / 2;
    const bool fresh = !private_[kPShareA].texture || private_[kPShareA].width != cw || private_[kPShareA].height != ch;
    if (!ensure_private(kPShareA, cw, ch, DXGI_FORMAT_R32G32B32A32_UINT) || !ensure_private(kPShareB, cw, ch, DXGI_FORMAT_R32G32B32A32_UINT) ||
        !ensure_private(kPShareT, cw, ch, DXGI_FORMAT_R16G16B16A16_FLOAT) || !ensure_private(kPShareT2, cw, ch, DXGI_FORMAT_R16G16B16A16_FLOAT) ||
        !ensure_private(kPShareHalf, cw, ch, DXGI_FORMAT_R16G16B16A16_FLOAT) || !ensure_private(kPShare, w, h, DXGI_FORMAT_R16G16B16A16_UNORM))
        return;
    c.grid[0] = w; c.grid[1] = h;
    c.flags = (scale_valid ? 1u : 0u) | (whole_motion ? 4u : 0u) | (depth_inverted ? 16u : 0u) | (fresh || share_reset_ ? 8u : 0u);
    if (own_motion_ && private_[kPFlowB].texture) { c.flags |= 2u; c.out_size[0] = std::max(1u, w / 2); c.out_size[1] = std::max(1u, h / 2); }
    if (share_mode) c.flags |= 32u | (share_mode == 1 ? 64u : 0u);
    if (share_diag_ && share_readback_) c.flags |= 128u;
    share_reset_ = false;
    // (the sums ping-pong between A and B: last frame's read, this frame's written)
    const PrivateId from = share_flip_ ? kPShareB : kPShareA, to = share_flip_ ? kPShareA : kPShareB;
    share_flip_ = !share_flip_;
    for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWShareMeasure) + i, kPDepth);
    set_x_srv(kWSrv(kWShareMeasure) + 0, kPObject);
    if (c.flags & 2u) set_x_srv(kWSrv(kWShareMeasure) + 2, kPFlowB);
    set_x_srv(kWSrv(kWShareMeasure) + 3, from);
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
    raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.Buffer.NumElements = 300; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    set_x_uav(kWUav(kWShareMeasure) + 0, to);
    device_->CreateUnorderedAccessView(share_bins_.Get(), nullptr, &raw, cpu(kWUav(kWShareMeasure) + 1));
    transition(private_[from], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(private_[to], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_share_measure_.Get(), &c, wc, kWShareMeasure, (cw + 7) / 8, (ch + 7) / 8);
    transition(private_[to], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    // The bands (cs_share_bins).
    D3D12_RESOURCE_BARRIER bins{}; bins.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; bins.UAV.pResource = share_bins_.Get();
    list_->ResourceBarrier(1, &bins);
    // (diagnostics: the bands' shares so far and this frame's near-camera sums, read back a few frames later - share.csv)
    if (c.flags & 128u) {
        auto to_copy = transition_barrier(share_bins_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list_->ResourceBarrier(1, &to_copy);
        list_->CopyBufferRegion(share_readback_.Get(), UINT64(frame_index_) * kShareDiagBytes, share_bins_.Get(), 240 * 4, kShareDiagBytes);
        std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
        list_->ResourceBarrier(1, &to_copy);
        share_diag_pending_[frame_index_] = true;
        share_diag_meta_[frame_index_].frame = share_diag_frame_;
        share_diag_meta_[frame_index_].mode = share_mode;
    }
    for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWShareBins) + i, kPDepth);
    device_->CreateUnorderedAccessView(share_bins_.Get(), nullptr, &raw, cpu(kWUav(kWShareBins) + 0));
    device_->CreateUnorderedAccessView(share_bins_.Get(), nullptr, &raw, cpu(kWUav(kWShareBins) + 1));
    w_dispatch(cs_share_bins_.Get(), &c, wc, kWShareBins, 1, 1);
    list_->ResourceBarrier(1, &bins);
    // Spread over the cells: across into T, then down into the cells' verdicts (taps 1/270 of the render height apart: +-3%
    // of it); then back to render resolution (cs_share_up).
    XConstants b = c;
    b.groups_x = std::max(1u, static_cast<UINT>(std::lround(double(c.rect[3]) / 270.0)));
    b.flags = 0;
    for (UINT i = 0; i < kWSrvCount; ++i) {
        set_x_srv(kWSrv(kWShareH) + i, kPDepth); set_x_srv(kWSrv(kWShareV) + i, kPDepth); set_x_srv(kWSrv(kWShareUp) + i, kPDepth);
    }
    set_x_srv(kWSrv(kWShareH) + 0, to);
    set_x_uav(kWUav(kWShareH) + 0, kPShareT); set_x_uav(kWUav(kWShareH) + 1, kPShareT2);
    set_x_srv(kWSrv(kWShareV) + 0, kPShareT);
    set_x_srv(kWSrv(kWShareV) + 2, kPShareT2);
    device_->CreateUnorderedAccessView(share_bins_.Get(), nullptr, &raw, cpu(kWUav(kWShareV) + 0));
    set_x_uav(kWUav(kWShareV) + 1, kPShareHalf);
    set_x_srv(kWSrv(kWShareUp) + 0, kPShareHalf);
    device_->CreateUnorderedAccessView(share_bins_.Get(), nullptr, &raw, cpu(kWUav(kWShareUp) + 0));
    set_x_uav(kWUav(kWShareUp) + 1, kPShare);
    set_x_srv(kWSrv(kWShareUp) + 3, kPObject);  // (the pixels' kind, see cs_share_up)
    transition(private_[kPShareT], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(private_[kPShareT2], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_share_blur_.Get(), &b, wc, kWShareH, (cw + 7) / 8, (ch + 7) / 8);
    transition(private_[kPShareT], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(private_[kPShareT2], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(private_[kPShareHalf], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    b.flags = (whole_motion ? 4u : 0u) | (depth_inverted ? 16u : 0u) | (share_mode ? 32u : 0u) | (chars_carried_ ? 256u : 0u);
    w_dispatch(cs_share_final_.Get(), &b, wc, kWShareV, (cw + 7) / 8, (ch + 7) / 8);
    transition(private_[kPShareHalf], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(private_[kPShare], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_share_up_.Get(), &b, wc, kWShareUp, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
    transition(private_[kPShare], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    share_built_ = true;
    // What the characters do on their own, per cell (cs_share_resid): with the measuring pass's constants (this frame's
    // camera motion, its flags) against the shares just made.
    if (ensure_private(kPShareResid, cw, ch, DXGI_FORMAT_R16G16B16A16_FLOAT)) {
        for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWShareResid) + i, kPDepth);
        set_x_srv(kWSrv(kWShareResid) + 0, kPObject);
        set_x_srv(kWSrv(kWShareResid) + 2, kPShare);
        set_x_uav(kWUav(kWShareResid) + 0, kPShareResid); set_x_uav(kWUav(kWShareResid) + 1, kPShareResid);
        transition(private_[kPShareResid], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        XConstants rc = c;
        if (own_carried_only_) rc.flags |= 512u;
        w_dispatch(cs_share_resid_.Get(), &rc, wc, kWShareResid, (cw + 7) / 8, (ch + 7) / 8);
        transition(private_[kPShareResid], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        // (spread over each character, set_own_smooth cells apart: across into T - free after the spread - then down back)
        if (own_smooth_ > 0) {
            XConstants r = c;
            r.groups_x = UINT(own_smooth_);
            for (UINT i = 0; i < kWSrvCount; ++i) { set_x_srv(kWSrv(kWResidBlurH) + i, kPShareResid); set_x_srv(kWSrv(kWResidBlurV) + i, kPShareT); }
            set_x_uav(kWUav(kWResidBlurH) + 0, kPShareT); set_x_uav(kWUav(kWResidBlurH) + 1, kPShareT);
            set_x_uav(kWUav(kWResidBlurV) + 0, kPShareResid); set_x_uav(kWUav(kWResidBlurV) + 1, kPShareResid);
            transition(private_[kPShareT], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            r.flags = own_depth_blind_ ? 2u : 0u;
            w_dispatch(cs_resid_blur_.Get(), &r, wc, kWResidBlurH, (cw + 7) / 8, (ch + 7) / 8);
            transition(private_[kPShareT], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(private_[kPShareResid], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            r.flags = own_depth_blind_ ? 3u : 1u;
            w_dispatch(cs_resid_blur_.Get(), &r, wc, kWResidBlurV, (cw + 7) / 8, (ch + 7) / 8);
            transition(private_[kPShareResid], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        // (per rendered pixel, for the warp: cs_own_up)
        if (ensure_private(kPOwnMove, w, h, DXGI_FORMAT_R16G16_FLOAT)) {
            for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWOwnUp) + i, kPDepth);
            set_x_srv(kWSrv(kWOwnUp) + 0, kPShareResid);
            set_x_uav(kWUav(kWOwnUp) + 0, kPOwnMove); set_x_uav(kWUav(kWOwnUp) + 1, kPOwnMove);
            transition(private_[kPOwnMove], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            w_dispatch(cs_own_up_.Get(), &c, wc, kWOwnUp, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
            transition(private_[kPOwnMove], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
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
                                             bool keep_attached, bool depth_inverted, bool combined, bool fill, int stretch, bool reticles,
                                             bool share) {
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
    // A HUD layer without a HUD-less picture (taken by the add-on, see cs_layer_hud): the HUD exactly, nothing to detect -
    // unless the check found it is no HUD (layer_trust_ 2).
    const bool layer_here = hud && !from_hudless && src.has_ui && !src.has_hudless && private_[kPUi].texture && private_[kPUi].width == ow &&
                            private_[kPUi].height == oh && layer_trust_ != 2;
    const bool scene_here = src.has_scene && private_[kPScene].texture && private_[kPScene].width == ow && private_[kPScene].height == oh;
    // ...where it can not be checked (no upscaler output: RE4's FSR 2), its coverage counted instead, every frame; a frame
    // after ones where it covered 90% of the screen or more does not take it (the learned detection instead). RE4: the layer
    // the add-on found is opaque over the whole picture - all of it held as HUD, the scenery at the game's frame rate. Not
    // for good (a game's full-screen menu is a real HUD layer: Requiem's pause menu), and not where it is checked.
    const bool cover_only = layer_here && layer_trust_ == 0 && !scene_here;
    const bool from_layer = layer_here && !(cover_only && layer_full_);
    // A frame without it once it has been checked (layer_trust_ 1): the game hides its HUD (Requiem, when not needed) -
    // no HUD, nothing to detect (detecting brought back lights taken for HUD, and its cost).
    const bool hidden = hud && !from_hudless && !from_layer && !src.has_hudless && layer_trust_ == 1;
    // Checking the layer (see cs_layer_check): with the upscaler's output, not on a frame that starts the score over.
    const bool check = from_layer && layer_trust_ == 0 && !reset && scene_here;
    if (cover_only && !layer_pending_[frame_index_]) {
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXCoverSrv + i, kPUi);
        D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
        raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
        raw.Buffer.NumElements = 4; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(layer_counts_.Get(), nullptr, &raw, cpu(kXCoverUav + 0));
        device_->CreateUnorderedAccessView(layer_counts_.Get(), nullptr, &raw, cpu(kXCoverUav + 1));
        D3D12_RESOURCE_BARRIER counts{}; counts.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; counts.UAV.pResource = layer_counts_.Get();
        x_dispatch(cs_layer_check_clear_.Get(), &c, kXCoverSrv, kXCoverUav, 1, 1);
        list_->ResourceBarrier(1, &counts);
        x_dispatch(cs_layer_check_.Get(), &c, kXCoverSrv, kXCoverUav, (ow / 2 + 7) / 8, (oh / 2 + 7) / 8);
        auto to_copy = transition_barrier(layer_counts_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list_->ResourceBarrier(1, &to_copy);
        list_->CopyBufferRegion(layer_readback_.Get(), UINT64(frame_index_) * 8, layer_counts_.Get(), 0, 8);
        std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
        list_->ResourceBarrier(1, &to_copy);
        layer_pending_[frame_index_] = true;
        layer_cover_only_[frame_index_] = true;
        layer_frame_samples_ = double(ow / 2) * double(oh / 2);
    }
    if (!layer_here) layer_full_ = false;
    const bool from_scene = hud && !from_hudless && !from_layer && !hidden && src.has_scene && private_[kPScene].texture &&
                            private_[kPScene].width == ow && private_[kPScene].height == oh;
    // The detections' own textures: only while the HUD is detected. Where the game hands its HUD over (its HUD-less
    // picture, its HUD layer) they are released after 300 game frames without (not at once: a game that alternates
    // would have them made again and again, each time waiting for the GPU).
    if (hud && !from_hudless && !from_layer && !hidden) {
        detection_idle_ = 0;
        if (!ensure_private(kPHudKind, ow, oh, DXGI_FORMAT_R32_UINT)) return nullptr;
    } else if (check) {
        detection_idle_ = 0;
    } else if (++detection_idle_ == 300) {
        // (released once the GPU is past them - no waiting here: no hitch)
        for (const PrivateId id : {kPHudKind, kPWorld, kPHudLearned, kPHudTiles}) retire(id);
    }
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
    } else if (from_layer) {
        hud_from_scene_ = true;  // (the learned map starts over when this stops)
        // (the check first: the upscaler-output detection into a texture of its own - kPWorld, unused without detection)
        const bool checking = check && ensure_private(kPWorld, ow, oh, DXGI_FORMAT_R32_FLOAT) && detect_hud_from_scene(c, false, kPWorld);
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXHudlessSrv + i, kPUi);
        set_x_uav(kXHudlessUav + 0, kPHudScore); set_x_uav(kXHudlessUav + 1, kPHudScore);
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        x_dispatch(cs_layer_hud_.Get(), &c, kXHudlessSrv, kXHudlessUav, (ow + 7) / 8, (oh + 7) / 8);
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (checking) {
            // (its own tables: the learned detection's, which never runs on a frame with the layer)
            transition(private_[kPWorld], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXHudSrv + i, i == 0 ? kPUi : kPWorld);
            D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
            raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
            raw.Buffer.NumElements = 4; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            device_->CreateUnorderedAccessView(layer_counts_.Get(), nullptr, &raw, cpu(kXHudUav + 0));
            device_->CreateUnorderedAccessView(layer_counts_.Get(), nullptr, &raw, cpu(kXHudUav + 1));
            D3D12_RESOURCE_BARRIER counts{}; counts.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; counts.UAV.pResource = layer_counts_.Get();
            x_dispatch(cs_layer_check_clear_.Get(), &c, kXHudSrv, kXHudUav, 1, 1);
            list_->ResourceBarrier(1, &counts);
            x_dispatch(cs_layer_check_.Get(), &c, kXHudSrv, kXHudUav, (ow / 2 + 7) / 8, (oh / 2 + 7) / 8);
            auto to_copy = transition_barrier(layer_counts_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list_->ResourceBarrier(1, &to_copy);
            list_->CopyBufferRegion(layer_readback_.Get(), UINT64(frame_index_) * 8, layer_counts_.Get(), 0, 8);
            std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
            list_->ResourceBarrier(1, &to_copy);
            layer_pending_[frame_index_] = true;
            layer_cover_only_[frame_index_] = false;
            layer_frame_samples_ = double(ow / 2) * double(oh / 2);
        }
    } else if (hidden) {
        // (the checked layer is not there: the game shows no HUD now)
        hud_from_scene_ = true;
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        set_x_uav(kXHudUav + 0, kPHudScore); set_x_uav(kXHudUav + 1, kPHudScore);
        XConstants z = c; z.flags = 0;
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXHudSrv + i, kPBackbuffer);
        x_dispatch(cs_clear_score_.Get(), &z, kXHudSrv, kXHudUav, (ow + 7) / 8, (oh + 7) / 8);
        transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
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
    bool world = false, confirmed_tiles = false;
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
            // (the pass after the count reads each pixel's classification at t5)
            for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXHudAfterSrv + i, i == 0 ? kPBackbuffer : i == 1 ? kPPrevious : i == 2 ? kPDepth :
                                                                                i == 3 ? kPObject : i == 4 ? kPHudScore : kPHudKind);
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
            set_x_uav(kXKindUav + 0, kPHudKind);
            device_->CreateUnorderedAccessView(hud_counts_.Get(), nullptr, &raw, cpu(kXKindUav + 1));
            transition(private_[kPHudKind], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            x_dispatch(cs_hud_count_.Get(), &c, kXHudSrv, kXKindUav, (ow + 7) / 8, (oh + 7) / 8);
            list_->ResourceBarrier(1, &counts);
            transition(private_[kPHudKind], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            x_dispatch(cs_hud_world_.Get(), &c, kXHudAfterSrv, kXWorldUav, (ow + 7) / 8, (oh + 7) / 8);
            transition(private_[kPWorld], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            // The learned score alongside (the same classification), and its tiles: what the upscaler-output
            // detector finds is HUD only where it joins HUD it confirms (see cs_hud_tiles, cs_hud_spread, cs_mask).
            const UINT tw = (ow + 7) / 8, th = (oh + 7) / 8;
            const bool new_learned = !private_[kPHudLearned].texture || private_[kPHudLearned].width != ow || private_[kPHudLearned].height != oh;
            if (ensure_private(kPHudLearned, ow, oh, DXGI_FORMAT_R32_FLOAT) && ensure_private(kPHudTiles, tw, th, DXGI_FORMAT_R8_UNORM)) {
                set_x_uav(kXLearnedUav + 0, kPHudLearned);
                device_->CreateUnorderedAccessView(hud_counts_.Get(), nullptr, &raw, cpu(kXLearnedUav + 1));
                transition(private_[kPHudLearned], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                if (new_learned || reset) {
                    XConstants z = c; z.flags = 0;
                    x_dispatch(cs_clear_score_.Get(), &z, kXHudSrv, kXLearnedUav, (ow + 7) / 8, (oh + 7) / 8);
                    D3D12_RESOURCE_BARRIER lu{}; lu.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; lu.UAV.pResource = private_[kPHudLearned].texture.Get();
                    list_->ResourceBarrier(1, &lu);
                }
                x_dispatch(cs_hud_.Get(), &c, kXHudAfterSrv, kXLearnedUav, (ow + 7) / 8, (oh + 7) / 8);
                transition(private_[kPHudLearned], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                // (its counters to the log, as with the learned detection alone)
                auto to_copy = transition_barrier(hud_counts_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list_->ResourceBarrier(1, &to_copy);
                list_->CopyBufferRegion(hud_readback_.Get(), UINT64(frame_index_) * 16, hud_counts_.Get(), 0, 8);
                std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
                list_->ResourceBarrier(1, &to_copy);
                hud_pending_[frame_index_] = true;
                for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXTilesSrv + i, i == 1 ? kPHudScore : i == 2 ? kPWorld : kPHudLearned);
                set_x_uav(kXTilesUav + 0, kPHudTiles); set_x_uav(kXTilesUav + 1, kPHudTiles);
                transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(private_[kPHudTiles], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                x_dispatch(cs_hud_tiles_.Get(), &c, kXTilesSrv, kXTilesUav, (tw + 7) / 8, (th + 7) / 8);
                // (six passes, blocks shifted by half a block every other one: a panel up to ~5 blocks of 128 px across)
                D3D12_RESOURCE_BARRIER tu{}; tu.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; tu.UAV.pResource = private_[kPHudTiles].texture.Get();
                for (int pass = 0; pass < 6; ++pass) {
                    list_->ResourceBarrier(1, &tu);
                    XConstants sp = c;
                    const UINT off = (pass & 1) ? 8u : 0u;
                    sp.rect[0] = sp.rect[1] = off;
                    x_dispatch(cs_hud_spread_.Get(), &sp, kXTilesSrv, kXTilesUav, (tw + off + 15) / 16, (th + off + 15) / 16);
                }
                transition(private_[kPHudTiles], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                // (the confirmation only while the learned detection is learning here: in a game where it barely does - dark
                // scenes held back by its guard, DOOM Eternal: 0-49 of 300 frames - it confirmed almost nothing, and almost
                // all of the HUD found was dropped and warped. Then what the upscaler's output finds is taken as it is,
                // with the camera-motion check alone, as before the confirmation; see learning_stalled_)
                confirmed_tiles = !learning_stalled_;
            }
        }
    }
    // (the learned detection: not over a HUD the game hands over - it cost ~2.4 ms per game frame at 4K for nothing, and
    // could add to the layer's exact HUD what it learned)
    if (hud && previous && !from_scene && !from_hudless && !from_layer && !hidden) {
        set_x_srv(kXHudSrv + 0, kPBackbuffer);
        set_x_srv(kXHudSrv + 1, kPPrevious);
        set_x_srv(kXHudSrv + 2, kPDepth);
        set_x_srv(kXHudSrv + 3, kPObject);
        for (UINT i = 4; i < kXSrvCount; ++i) set_x_srv(kXHudSrv + i, kPDepth);
        // (the pass after the count reads each pixel's classification at t5)
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXHudAfterSrv + i, i == 0 ? kPBackbuffer : i == 1 ? kPPrevious : i == 2 ? kPDepth :
                                                                            i == 3 ? kPObject : i == 4 ? kPDepth : kPHudKind);
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
        set_x_uav(kXKindUav + 0, kPHudKind);
        device_->CreateUnorderedAccessView(hud_counts_.Get(), nullptr, &raw, cpu(kXKindUav + 1));
        transition(private_[kPHudKind], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        x_dispatch(cs_hud_count_.Get(), &c, kXHudSrv, kXKindUav, (ow + 7) / 8, (oh + 7) / 8);
        list_->ResourceBarrier(1, &counts);
        transition(private_[kPHudKind], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        x_dispatch(cs_hud_.Get(), &c, kXHudAfterSrv, kXHudUav, (ow + 7) / 8, (oh + 7) / 8);
        // The counters go to the log (how often the whole-frame guard holds learning back).
        auto to_copy = transition_barrier(hud_counts_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list_->ResourceBarrier(1, &to_copy);
        list_->CopyBufferRegion(hud_readback_.Get(), UINT64(frame_index_) * 16, hud_counts_.Get(), 0, 8);
        std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
        list_->ResourceBarrier(1, &to_copy);
        hud_pending_[frame_index_] = true;
    }
    transition(private_[kPHudScore], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    pass_stamp(kPassHud);
    // The share of the turn followed (follow_share, this frame's), in place of the attached pixels and the stretch.
    const bool shared = share && attached && keep_attached && share_built_ && private_[kPShare].width == w && private_[kPShare].height == h;
    // Camera-attached pixels, widened once at render resolution (the mask reads one value per pixel; with the share only
    // for the sight reticles).
    const bool keep_att = attached && keep_attached && (!shared || reticles) && ensure_private(kPAttached, w, h, DXGI_FORMAT_R8_UNORM);
    if (keep_att) {
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXAttSrv + i, kPObject);
        set_x_uav(kXAttUav + 0, kPAttached); set_x_uav(kXAttUav + 1, kPAttached);
        transition(private_[kPAttached], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        x_dispatch(cs_attached_.Get(), &c, kXAttSrv, kXAttUav, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
        transition(private_[kPAttached], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    // Sight reticles drawn without depth (option, see cs_ret_layer): with the character/weapon held and the previous
    // game frame's picture to compare this one with (the HUD-less pictures when the previous one is HUD-less).
    bool reticle = false;
    UINT sight_cell = 0;  // (the sight windows found this frame: their cells' size - 0, none looked for)
    if (reticles && keep_att && previous_valid_ && private_[kPPrevious].texture && private_[kPPrevious].width == ow &&
        private_[kPPrevious].height == oh) {
        const PrivateId current = previous_from_hudless_ ? kPHudless : kPBackbuffer;
        // cells of 0.55% of the render height (4 px at 720p)
        const UINT cell = std::max(1u, static_cast<UINT>(std::lround(0.0055 * double(c.rect[3]))));
        const UINT cw = (c.rect[2] + cell - 1) / cell, ch = (c.rect[3] + cell - 1) / cell;
        ID3D12Resource* const history_before = private_[kPRetA].texture.Get();
        const bool layers = ret_layers_ready(ow, oh);
        reticle = private_[current].texture && private_[current].width == ow && private_[current].height == oh &&
                  ensure_private(kPRetCells, cw, ch, DXGI_FORMAT_R16G16B16A16_FLOAT) &&
                  ensure_private(kPRetEncl, cw, ch, DXGI_FORMAT_R16G16B16A16_FLOAT);
        ID3D12Resource* const dirty_before = private_[kPRetDirtyA].texture.Get();
        reticle = reticle && ensure_private(kPRetDirtyA, cw, ch, DXGI_FORMAT_R32_UINT) && ensure_private(kPRetDirtyB, cw, ch, DXGI_FORMAT_R32_UINT);
        if (reticle) {
            sight_cell = cell;
            const bool fresh = !layers || private_[kPRetA].texture.Get() != history_before;
            // (new cell maps or new layers: nothing known of what the layers hold - every cell taken as holding something, once each)
            if (private_[kPRetDirtyA].texture.Get() != dirty_before || fresh) ret_written_[0] = ret_written_[1] = 0;
            XConstants k = c;
            k.mv_size[0] = cw; k.mv_size[1] = ch;
            k.groups_x = cell;
            // reach: 25% of the screen height, in cells (sight windows of any size; the gun is needed in all 8 directions)
            k.threshold = std::ceil(0.25f * float(c.rect[3]) / float(cell));
            k.flags = (fresh ? 2u : 0u) | (depth_inverted ? 16u : 0u);  // (16: cs_ret_cells' nearness)
            for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXRetCellsSrv + i, i == 1 ? kPDepth : kPObject);
            set_x_uav(kXRetCellsUav + 0, kPRetCells); set_x_uav(kXRetCellsUav + 1, kPRetCells);
            transition(private_[kPRetCells], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            x_dispatch(cs_ret_cells_.Get(), &k, kXRetCellsSrv, kXRetCellsUav, (cw + 7) / 8, (ch + 7) / 8);
            transition(private_[kPRetCells], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
            raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
            raw.Buffer.NumElements = kRetTrackN + 8; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            D3D12_RESOURCE_BARRIER buf{}; buf.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; buf.UAV.pResource = ret_buf_.Get();
            for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXRetEnclSrv + i, kPRetCells);
            set_x_uav(kXRetEnclUav + 0, kPRetEncl);
            device_->CreateUnorderedAccessView(ret_buf_.Get(), nullptr, &raw, cpu(kXRetEnclUav + 1));
            transition(private_[kPRetEncl], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            x_dispatch(cs_ret_enclose_.Get(), &k, kXRetEnclSrv, kXRetEnclUav, (cw + 7) / 8, (ch + 7) / 8);
            // The passes over the whole screen only while a sight's window was seen lately (cs_ret_args): no cost elsewhere.
            list_->ResourceBarrier(1, &buf);
            raw.Buffer.NumElements = 8;
            device_->CreateUnorderedAccessView(ret_args_.Get(), nullptr, &raw, cpu(kXRetArgsUav + 0));
            raw.Buffer.NumElements = kRetTrackN + 8;
            device_->CreateUnorderedAccessView(ret_buf_.Get(), nullptr, &raw, cpu(kXRetArgsUav + 1));
            XConstants a = k;
            // (flag 4 - run the full-screen passes anyway - clears what old layers hold; without layers there is nothing to
            // clear, and it would read back as a window seen: the layers made in every game)
            a.flags = (ret_ran_ || !layers) ? 0u : 4u;
            x_dispatch(cs_ret_args_.Get(), &a, kXRetEnclSrv, kXRetArgsUav, 1, 1);
            ret_ran_ = true;
            list_->ResourceBarrier(1, &buf);
            // (no layers yet: whether a sight's window shows, read back - the layers are made once it does, see ret_layers_ready)
            if (!layers) {
                auto to_copy = transition_barrier(ret_args_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list_->ResourceBarrier(1, &to_copy);
                list_->CopyBufferRegion(ret_args_readback_.Get(), UINT64(frame_index_) * 32, ret_args_.Get(), 0, 32);
                std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
                list_->ResourceBarrier(1, &to_copy);
                ret_args_pending_[frame_index_] = true;
            }
        }
        reticle = reticle && layers;
        if (reticle) {
            const bool fresh = private_[kPRetA].texture.Get() != history_before;
            const PrivateId from = ret_flip_ ? kPRetB : kPRetA, to = ret_flip_ ? kPRetA : kPRetB;
            XConstants k = c;
            k.mv_size[0] = cw; k.mv_size[1] = ch;
            k.groups_x = cell;
            k.threshold = std::ceil(0.25f * float(c.rect[3]) / float(cell));
            k.flags = fresh ? 2u : 0u;
            D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
            raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
            raw.Buffer.NumElements = kRetTrackN + 8; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            D3D12_RESOURCE_BARRIER buf{}; buf.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; buf.UAV.pResource = ret_buf_.Get();
            auto to_args = transition_barrier(ret_args_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            list_->ResourceBarrier(1, &to_args);
            // Where the reticle went since the previous frame (cs_ret_align, cs_ret_pick), into the cells (cs_ret_shift).
            if (!fresh) {
                for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXRetAlignSrv + i, i == 1 ? from : current);
                set_x_uav(kXRetAlignUav + 0, kPRetCells);
                device_->CreateUnorderedAccessView(ret_buf_.Get(), nullptr, &raw, cpu(kXRetAlignUav + 1));
                x_dispatch(cs_ret_align_.Get(), &k, kXRetAlignSrv, kXRetAlignUav, 0, 0, ret_args_.Get(), 0);
                list_->ResourceBarrier(1, &buf);
            }
            device_->CreateUnorderedAccessView(ret_buf_.Get(), nullptr, &raw, cpu(kXRetPickUav + 0));
            device_->CreateUnorderedAccessView(ret_buf_.Get(), nullptr, &raw, cpu(kXRetPickUav + 1));
            x_dispatch(cs_ret_pick_.Get(), &k, kXRetCellsSrv, kXRetPickUav, 1, 1);
            list_->ResourceBarrier(1, &buf);
            set_x_uav(kXRetShiftUav + 0, kPRetEncl);
            device_->CreateUnorderedAccessView(ret_buf_.Get(), nullptr, &raw, cpu(kXRetShiftUav + 1));
            D3D12_RESOURCE_BARRIER encl{}; encl.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; encl.UAV.pResource = private_[kPRetEncl].texture.Get();
            list_->ResourceBarrier(1, &encl);  // (the enclosure written above, read and rewritten here)
            x_dispatch(cs_ret_shift_.Get(), &k, kXRetEnclSrv, kXRetShiftUav, (cw + 7) / 8, (ch + 7) / 8);
            transition(private_[kPRetEncl], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            const PrivateId srvs[kXSrvCount] = {current, kPPrevious, kPDepth, kPAttached, kPRetEncl, from};
            for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXRetSrv + i, srvs[i]);
            const PrivateId dirty = to == kPRetA ? kPRetDirtyA : kPRetDirtyB;
            set_x_uav(kXRetUav + 0, to); set_x_uav(kXRetUav + 1, dirty);
            transition(private_[to], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition(private_[dirty], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            XConstants lk = k;
            std::uint32_t& written = ret_written_[to == kPRetA ? 0 : 1];
            const std::uint32_t frames[2] = {written, ++ret_frame_};
            std::memcpy(lk.mv_scale, frames, sizeof(frames));
            written = ret_frame_;
            x_dispatch(cs_ret_layer_.Get(), &lk, kXRetSrv, kXRetUav, 0, 0, ret_args_.Get(), 16);
            transition(private_[dirty], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            std::swap(to_args.Transition.StateBefore, to_args.Transition.StateAfter);
            list_->ResourceBarrier(1, &to_args);
            transition(private_[to], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ret_flip_ = !ret_flip_;
            ret_latest_ = to;
        }
    }
    if (!reticle) ret_ran_ = false;
    layer_built_ = reticle;
    // The stretch around them (option): the distance share, in two passes.
    const bool ramp = keep_att && !shared && stretch > 0 && ensure_private(kPRampRows, w, h, DXGI_FORMAT_R8_UNORM) &&
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
    pass_stamp(kPassCharacter);
    // (16 bits: the follow share moves pixels by up to hundreds of px; x, y the turn's shares, z the movement's - see cs_mask.
    // Not with the share: the warp reads the shares and the HUD themselves - see ow_mask_at)
    if (shared) {
        if (private_[kPWarpMask].texture) retire(kPWarpMask);
    } else if (!ensure_private(kPWarpMask, ow, oh, DXGI_FORMAT_R16G16B16A16_UNORM)) {
        return nullptr;
    }
    set_x_srv(kXMaskSrv + 0, kPHudScore);
    set_x_srv(kXMaskSrv + 1, keep_att ? kPAttached : kPHudScore);
    set_x_srv(kXMaskSrv + 2, world ? kPWorld : kPHudScore);
    set_x_srv(kXMaskSrv + 3, ramp ? kPRamp : shared && sight_cell ? kPRetEncl : kPHudScore);
    for (UINT i = 4; i < kXSrvCount; ++i) set_x_srv(kXMaskSrv + i, kPHudScore);
    if (confirmed_tiles) set_x_srv(kXMaskSrv + 4, kPHudTiles);
    if (shared) set_x_srv(kXMaskSrv + 5, kPShare);  // (t5: the two shares, see cs_mask)
    set_x_uav(kXMaskUav + 0, kPMask); set_x_uav(kXMaskUav + 1, shared ? kPMask : kPWarpMask);
    c.flags = (hud ? 4u : 0u) | (keep_att ? 8u : 0u) | (world ? 64u : 0u) | (ramp ? 128u : 0u) | (confirmed_tiles ? 256u : 0u) |
              (shared ? 512u | 1024u : 0u) | (shared && sight_cell ? 2048u : 0u) | (outline_scenery_ ? 4096u : 0u) | (carried_edge_ ? 8192u : 0u);
    c.groups_x = sight_cell;
    if (shared && sight_cell) transition(private_[kPRetEncl], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);  // (no layers yet: left as written)
    stretch_built_ = ramp || shared;  // (the warp's follow share: flag 128)
    share_mask_built_ = shared;
    transition(private_[kPMask], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!shared) transition(private_[kPWarpMask], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_mask_.Get(), &c, kXMaskSrv, kXMaskUav, (ow + 7) / 8, (oh + 7) / 8);
    transition(private_[kPMask], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!shared) transition(private_[kPWarpMask], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    mask_ready_ = true;
    built_mask_ = true;
    pass_stamp(kPassMask);
    return private_[kPMask].texture.Get();
}

bool Renderer::detect_hud_from_scene(const XConstants& base, bool fill, PrivateId out) {
    const UINT ow = private_[kPBackbuffer].width, oh = private_[kPBackbuffer].height;
    XConstants c = base;
    c.grid[0] = (ow + kSceneTile - 1) / kSceneTile; c.grid[1] = (oh + kSceneTile - 1) / kSceneTile;
    if (c.grid[0] * c.grid[1] > kSceneMaxTiles) return false;
    set_x_srv(kXSceneSrv + 0, kPBackbuffer);
    set_x_srv(kXSceneSrv + 1, kPScene);
    for (UINT i = 2; i < kXSrvCount; ++i) set_x_srv(kXSceneSrv + i, kPScene);
    set_x_uav(kXSceneUav + 0, out);
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
    raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.Buffer.NumElements = kSceneFitBytes / 4; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device_->CreateUnorderedAccessView(scene_fit_.Get(), nullptr, &raw, cpu(kXSceneUav + 1));
    transition(private_[out], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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
    return true;
}

bool Renderer::own_warp(const IngestedSource& src, bool use_ui_tags, bool use_mask, const float source_to_target[16], bool depth_inverted,
                        bool memory, const ObjectWarp* objects, int generated, bool soft_edge, const float* input_to_target) {
    // (the shown set with split queues)
    if (!src.valid || !src.has_depth || !private_[kPOutput].texture || !shown(kPDepth).texture) return false;
    const bool split = use_ui_tags && src.has_hudless && src.has_ui && shown(kPHudless).texture && shown(kPUi).texture;
    const bool mask_ready = split_ ? shown_mask_ : mask_ready_;
    const UINT ow = private_[kPOutput].width, oh = private_[kPOutput].height;
    // (with the follow share: the no-warp mask and the shares at render resolution - see ow_mask_at)
    const bool share_mask = (split_ ? shown_share_mask_ : share_mask_built_) && shown(kPMask).texture && shown(kPMask).width == ow &&
                            shown(kPMask).height == oh && shown(kPShare).texture && shown(kPShare).width == shown(kPDepth).width &&
                            shown(kPShare).height == shown(kPDepth).height;
    const bool mask = use_mask && mask_ready && (share_mask || (shown(kPWarpMask).texture && shown(kPWarpMask).width == ow &&
                                                                shown(kPWarpMask).height == oh));
    const PrivateId mask_id = share_mask ? kPMask : kPWarpMask;
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
    // (a pure turn - nothing in the camera's matrix depends on depth - was warped without depth, by the search, as if every
    // pixel moved alike. With the forward splat on, never: with the follow share each spot moves by its own share - a
    // character none of it, the scenery all - and the search lost what is carried while the scenery swept past it: a gun in
    // a character's hand broke into pieces whenever the camera turned, RE2 - only in the refreshes between game frames, so
    // no capture ever showed it as bad as it looked. Pure turns are most of RE2's: the forward splat, with depth)
    const bool no_depth = std::fabs(m[8]) <= 1e-6f * scale && std::fabs(m[9]) <= 1e-6f * scale && std::fabs(m[11]) <= 1e-6f * scale &&
                          !(warp_splat_ && splat_turns_);
    const bool fill_ready = split_ ? shown_fill_ : fill_ready_;
    const bool fill = mask && !split && fill_ready && shown(kPFill).width == ow && shown(kPFill).height == oh;
    const bool remembered = memory && mask && mem_shown_ >= 0 && private_[kPMem0 + mem_shown_].texture;
    const bool stretched = mask && (split_ ? shown_stretch_ : stretch_built_);
    c.flags = (split ? 1u : 0u) | (mask ? 2u : 0u) | (fill ? 4u : 0u) | (depth_inverted ? 16u : 0u) | (no_depth ? 32u : 0u) |
              (remembered ? 64u : 0u) | (stretched ? 128u : 0u) | (moving ? 256u : 0u) | (soft_edge ? 1024u : 0u) |
              (mask && share_mask ? 16384u : 0u) | (warp_paths_ ? 32768u : 0u);
    float wc[20] = {};
    // (the characters carried on along their own motion - see ow_own: the cells of the shown frame, this many game frames on)
    const bool own_motion = mask && share_mask && stretched && warp_ahead_ > 0 && !moving && shown(kPOwnMove).texture &&
                            shown(kPOwnMove).width == dw && shown(kPOwnMove).height == dh;
    if (own_motion) { c.flags |= 65536u; wc[19] = warp_ahead_; }
    if (!outlines_held_) c.flags |= 131072u;
    if (carried_free_) c.flags |= 262144u;
    if (carried_edge_) c.flags |= 524288u;
    const UINT t = kWSrv(kWWarp);
    // Frame generation: one of the game's generated images, with its own depth (prepare_generated).
    const bool gen = generated >= 0 && generated < src.generated && shown(static_cast<PrivateId>(kPGen0 + generated)).width == ow &&
                     shown(static_cast<PrivateId>(kPGenZ0 + generated)).width == dw;
    set_x_srv(t + 0, gen ? static_cast<PrivateId>(kPGen0 + generated) : split ? kPHudless : kPBackbuffer, true);
    set_x_srv(t + 1, split ? kPUi : kPBackbuffer, true);  // (only read with a UI layer)
    set_x_srv(t + 2, mask ? mask_id : kPDepth, true);
    set_x_srv(t + 14, mask && share_mask ? kPShare : kPDepth, true);
    set_x_srv(t + 15, own_motion ? kPOwnMove : kPDepth, true);
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
        set_x_srv(st + 2, mask ? mask_id : kPDepth, true);
        set_x_srv(st + 14, mask && share_mask ? kPShare : kPDepth, true);
        set_x_srv(st + 15, own_motion ? kPOwnMove : kPDepth, true);
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
        for (UINT i = 6; i < 14; ++i) set_x_srv(t + i, kPDepth, true);
    }
    // Two shares (follow_share with the input's turn): the input's part of the turn on its own (wmat - free without moving
    // objects), each part by its share (ow_lands).
    if (input_to_target && !moving && stretched) { std::memcpy(wc, input_to_target, sizeof(float) * 16); c.flags |= 8192u; }
    // Sight reticles (option): the layer on the glass of the shown frame (not with the game's generated images: it
    // belongs to the frame itself).
    // (the history texture written last by the shown frame: the next intake writes the other one, and the one after
    // waits for the warps that read this, as for every shown texture - release_wait_)
    const PrivateId ret = split_ ? ret_shown_ : ret_latest_;
    const bool layer = mask && !gen && (split_ ? shown_layer_ : layer_built_) && private_[ret].texture &&
                       private_[ret].width == ow && private_[ret].height == oh;
    if (layer) { c.flags |= 2048u; set_x_srv(t + 11, ret); }
    // Forward splat (warp_splat_): the claims first, from the same depth, mask and camera (not for a pure turn: nothing
    // to sort out without depth).
    set_x_srv(t + 12, kPDepth, true);
    // (the memory's depth, t13: what it keeps is the scenery behind - see ow_from_memory_behind)
    if (remembered) set_x_srv(t + 13, static_cast<PrivateId>(kPMemZ0 + mem_shown_));
    else set_x_srv(t + 13, kPDepth, true);
    if (warp_splat_ && !no_depth && ensure_private(kPWarpClaims, dw, dh, DXGI_FORMAT_R32_UINT)) {
        const UINT ct = kWSrv(kWClaim);
        for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(ct + i, kPDepth, true);
        set_x_srv(ct + 2, mask ? mask_id : kPDepth, true);
        set_x_srv(ct + 14, mask && share_mask ? kPShare : kPDepth, true);
        set_x_srv(ct + 15, own_motion ? kPOwnMove : kPDepth, true);
        set_x_srv(ct + 3, gen ? static_cast<PrivateId>(kPGenZ0 + generated) : kPDepth, true);
        set_x_uav(kWUav(kWClaim) + 0, kPWarpClaims); set_x_uav(kWUav(kWClaim) + 1, kPWarpClaims);
        transition(private_[kPWarpClaims], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        w_dispatch(cs_warp_claims_clear_.Get(), &c, wc, kWClaim, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
        D3D12_RESOURCE_BARRIER cleared{}; cleared.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; cleared.UAV.pResource = private_[kPWarpClaims].texture.Get();
        list_->ResourceBarrier(1, &cleared);
        w_dispatch(cs_warp_claim_.Get(), &c, wc, kWClaim, (c.rect[2] + 7) / 8, (c.rect[3] + 7) / 8);
        transition(private_[kPWarpClaims], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        set_x_srv(t + 12, kPWarpClaims);
        c.flags |= 4096u;
    }
    if (warp_stop_ <= 1) return true;
    set_x_uav(kWUav(kWWarp) + 0, kPOutput); set_x_uav(kWUav(kWWarp) + 1, kPOutput);
    transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // (forward splat: the plain pixels first, the rest packed for the full warp - see cs_own_warp_lean)
    if ((c.flags & 4096) && warp_blocks_ && !moving && ensure_warp_list(ow, oh)) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
        raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        raw.Buffer.NumElements = UINT(warp_list_->GetDesc().Width / 4);
        device_->CreateUnorderedAccessView(warp_list_.Get(), nullptr, &raw, cpu(kWUav(kWWarp) + 1));
        for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWWarpArgs) + i, kPDepth, true);
        device_->CreateUnorderedAccessView(warp_list_.Get(), nullptr, &raw, cpu(kWUav(kWWarpArgs) + 1));
        raw.Buffer.NumElements = 4;
        device_->CreateUnorderedAccessView(warp_args_.Get(), nullptr, &raw, cpu(kWUav(kWWarpArgs) + 0));
        D3D12_RESOURCE_BARRIER lb{}; lb.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; lb.UAV.pResource = warp_list_.Get();
        w_dispatch(cs_own_warp_reset_.Get(), &c, wc, kWWarp, 1, 1);
        list_->ResourceBarrier(1, &lb);
        w_dispatch(cs_own_warp_lean_.Get(), &c, wc, kWWarp, ((ow + 1) / 2 + 7) / 8, ((oh + 1) / 2 + 7) / 8);
        if (warp_stop_ <= 2) return true;
        list_->ResourceBarrier(1, &lb);
        w_dispatch(cs_own_warp_args_.Get(), &c, wc, kWWarpArgs, 1, 1);
        auto to_args = transition_barrier(warp_args_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        list_->ResourceBarrier(1, &to_args);
        w_dispatch_indirect(cs_own_warp_rest_.Get(), &c, wc, kWWarp, warp_args_.Get());
        std::swap(to_args.Transition.StateBefore, to_args.Transition.StateAfter);
        list_->ResourceBarrier(1, &to_args);
    } else {
        w_dispatch(cs_own_warp_.Get(), &c, wc, kWWarp, (ow + 7) / 8, (oh + 7) / 8);
    }

    if (moving) {
        // The pixels moving objects claimed or left, looked at again.
        D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = private_[kPOutput].texture.Get();
        list_->ResourceBarrier(1, &uav);
        XConstants fix = c;
        fix.flags = (fix.flags & ~512u) | (objects->debug ? 512u : 0u);  // (its debug colours: only when asked for)
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

void Renderer::pass_stamp(IntakePass pass) {
    int& n = pass_count_[frame_index_];
    if (!pass_stamps_ || !pass_readback_ || n >= kPassStamps) return;
    list_->EndQuery(pass_stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * kPassStamps + n);
    pass_ids_[frame_index_][n++] = static_cast<std::uint8_t>(pass);
}

void Renderer::execute() {
    if (const int n = pass_count_[frame_index_]; n > 0 && pass_stamps_)
        list_->ResolveQueryData(pass_stamps_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame_index_ * kPassStamps, n, pass_readback_.Get(),
                                UINT64(frame_index_) * kPassStamps * 8);
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
    for (auto& r : rec_)
        if (r.state.load(std::memory_order_acquire) == 1 && r.context == context_) {
            r.fence = fence_values_[context_];
            r.state.store(2, std::memory_order_release);
        }
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
    if (on) { shown_mask_ = mask_ready_; shown_fill_ = fill_ready_; shown_stretch_ = stretch_built_; shown_share_mask_ = share_mask_built_; shown_layer_ = layer_built_; ret_shown_ = ret_latest_; }
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
    shown_share_mask_ = share_mask_built_;
    shown_layer_ = layer_built_;
    ret_shown_ = ret_latest_;
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
    // An unwarped refresh: the shown frame as it is, copied by a compute shader (the shown textures stay in the
    // shader-read state both queues use, so they cannot be blitted or copied by the copy engine directly). A
    // plain copy: through the warp shader with the identity, this cost as much as a warp.
    const auto& bb = shown(kPBackbuffer);
    if (!bb.texture || !private_[kPOutput].texture || bb.width != private_[kPOutput].width || bb.height != private_[kPOutput].height) return;
    XConstants c{};
    c.grid[0] = bb.width; c.grid[1] = bb.height;
    const float wc[20] = {};
    for (UINT i = 0; i < kWSrvCount; ++i) set_x_srv(kWSrv(kWShown) + i, kPBackbuffer, true);
    set_x_uav(kWUav(kWShown) + 0, kPOutput); set_x_uav(kWUav(kWShown) + 1, kPOutput);
    transition(private_[kPOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    w_dispatch(cs_copy4_.Get(), &c, wc, kWShown, (bb.width + 7) / 8, (bb.height + 7) / 8);
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

void Renderer::prepare_ret_layers(std::uint32_t w, std::uint32_t h) {
    ret_window_seen_ = true;
    ret_layers_ready(w, h);
    while (ret_create_ && !ret_create_->done.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

bool Renderer::ret_layers_ready(std::uint32_t w, std::uint32_t h) {
    auto& a = private_[kPRetA];
    if (a.texture && a.width == w && a.height == h && private_[kPRetB].texture) return true;
    if (a.texture) { retire(kPRetA); retire(kPRetB); }  // (another output size: made again, in the background)
    if (ret_create_ && ret_create_->done.load()) {
        // (taken out under the job's lock, the job let go of after it: its lock must not outlive it)
        ComPtr<ID3D12Resource> made[2];
        bool fits = false;
        {
            std::lock_guard lock(ret_create_->mutex);
            fits = !ret_create_->failed && ret_create_->w == w && ret_create_->h == h;
            made[0] = ret_create_->a; made[1] = ret_create_->b;
        }
        ret_create_.reset();
        if (fits) {
            const PrivateId ids[2] = {kPRetA, kPRetB};
            for (int i = 0; i < 2; ++i) {
                auto& p = private_[ids[i]];
                p = Private{};
                p.texture = made[i]; p.width = w; p.height = h; p.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                p.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                device_->CreateUnorderedAccessView(p.texture.Get(), nullptr, nullptr, cpu(kPrivUav + ids[i]));
                device_->CreateShaderResourceView(p.texture.Get(), nullptr, cpu(kPrivSrv + ids[i]));
            }
            return true;
        }
    }
    if (ret_window_seen_ && !ret_create_) {
        auto job = std::make_shared<RetCreate>();
        job->w = w; job->h = h;
        ret_create_ = job;
        ComPtr<ID3D12Device> device = device_;
        std::thread([job, device]() {
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC d{};
            d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = job->w; d.Height = job->h; d.DepthOrArraySize = 1; d.MipLevels = 1;
            d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; d.SampleDesc.Count = 1; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            ComPtr<ID3D12Resource> a, b;
            const bool ok = SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                                      nullptr, IID_PPV_ARGS(&a))) &&
                            SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                                      nullptr, IID_PPV_ARGS(&b)));
            {
                std::lock_guard lock(job->mutex);
                job->a = a; job->b = b; job->failed = !ok;
            }
            job->done = true;
        }).detach();
    }
    return false;
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
    pending_show_ = 0; release_wait_ = 0; shown_mask_ = shown_fill_ = shown_layer_ = false;
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

void Renderer::retire_texture(ComPtr<ID3D12Resource>& r) {
    if (r) retired_.push_back({std::move(r), fence_values_[0] + 1, fence_values_[1] + 1});
    r.Reset();
}
void Renderer::retire(PrivateId id) {
    retire_texture(private_[id].texture);
    private_[id] = Private{};
}
void Renderer::collect_retired() {
    if (retired_.empty() || !fences_[0]) return;
    const std::uint64_t done0 = fences_[0]->GetCompletedValue(), done1 = fences_[1] ? fences_[1]->GetCompletedValue() : 0;
    // (past both queues' next submission after it was retired; the intake queue only counts with split queues - without,
    // everything runs on the first)
    retired_.erase(std::remove_if(retired_.begin(), retired_.end(), [&](const Retired& r) {
                       return done0 >= r.after0 && (!split_ || done1 >= r.after1);
                   }), retired_.end());
}

ID3D12GraphicsCommandList* Renderer::begin(int context) {
    collect_retired();
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
    if (probe_pending_[frame_index_] && probe_readback_) {
        probe_pending_[frame_index_] = false;
        std::uint32_t* v = nullptr;
        D3D12_RANGE range{frame_index_ * 32, frame_index_ * 32 + 32};
        if (SUCCEEDED(probe_readback_->Map(0, &range, reinterpret_cast<void**>(&v)))) {
            const std::uint32_t* u = v + frame_index_ * 8;
            auto& sc = probe_scores_[probe_pending_cand_[frame_index_]];
            ++sc.frames;
            sc.area += double(u[0]) / 256.0 / std::max(1u, probe_pending_tiles_[frame_index_]);
            if (u[2] + u[4] > 0) {
                ++sc.scored;
                sc.same_min += u[1] / 256.0; sc.same_max += u[2] / 256.0;
                sc.world_min += u[3] / 256.0; sc.world_max += u[4] / 256.0;
            }
            // The learned point in use, checked over its recent frames: stays on the screen (0.8) and clearly more than
            // it moves with the scenery (0.1), over 60 scored frames - else reported, and its scores dropped.
            const std::uint32_t cand = probe_pending_cand_[frame_index_];
            if (probe_pending_locked_[frame_index_]) {
                if (cand != probe_locked_) { probe_locked_ = cand; probe_check_n_ = 0; probe_check_same_ = 1; probe_check_world_ = 0; }
                if (u[2] > 0 && u[4] > 0) {
                    const double a = 1.0 / std::min(60.0, double(++probe_check_n_));
                    probe_check_same_ += a * (u[1] / double(u[2]) - probe_check_same_);
                    probe_check_world_ += a * (u[3] / double(u[4]) - probe_check_world_);
                    if (probe_check_n_ >= 60 && (probe_check_same_ < 0.8 || probe_check_same_ - probe_check_world_ < 0.1)) {
                        probe_fail_ = cand;
                        probe_scores_.erase(cand);
                        probe_check_n_ = 0;
                    }
                }
            } else if (probe_fail_ && cand != probe_fail_) {
                probe_fail_ = 0;  // (learning again)
            }
            D3D12_RANGE none{0, 0};
            probe_readback_->Unmap(0, &none);
        }
    }
    if (grow_pending_[frame_index_] && grow_readback_) {
        grow_pending_[frame_index_] = false;
        std::uint32_t* v = nullptr;
        D3D12_RANGE range{frame_index_ * 4, frame_index_ * 4 + 4};
        if (SUCCEEDED(grow_readback_->Map(0, &range, reinterpret_cast<void**>(&v)))) {
            grow_cells_ = v[frame_index_];
            D3D12_RANGE none{0, 0};
            grow_readback_->Unmap(0, &none);
        }
    }
    if (layer_pending_[frame_index_] && layer_readback_) {
        layer_pending_[frame_index_] = false;
        std::uint32_t* v = nullptr;
        D3D12_RANGE range{frame_index_ * 8, frame_index_ * 8 + 8};
        if (SUCCEEDED(layer_readback_->Map(0, &range, reinterpret_cast<void**>(&v)))) {
            const double frame = std::max(1.0, layer_frame_samples_);
            // (a frame where the layer covers most of the screen - a menu, a full-screen overlay - leaves no 3D picture to
            // compare it with: it counts neither way. Requiem's pause menu: a darkened copy of the scene, which the
            // upscaler-output model predicts - taken for "not drawn over", the layer was rejected from menu frames alone)
            if (layer_cover_only_[frame_index_]) {
                // (coverage alone - a layer that can not be checked: see build_no_warp_mask)
                layer_full_ = double(v[frame_index_ * 2]) >= 0.9 * frame;
                layer_cover_only_[frame_index_] = false;
            } else if (layer_trust_ == 0 && v[frame_index_ * 2] && double(v[frame_index_ * 2]) < 0.5 * frame) {
                layer_covered_ += v[frame_index_ * 2]; layer_drawn_over_ += v[frame_index_ * 2 + 1];
                ++layer_frames_;
                // Trusted: at least 30 frames with some of it, as much of it seen as a tenth of a frame, and most of it drawn
                // over the picture. Not a HUD: 60 frames, a quarter of a frame's worth seen, little of it drawn over (a
                // texture of the 3D picture). In between: more frames.
                if (layer_frames_ >= 30 && double(layer_covered_) >= 0.1 * frame && layer_drawn_over_ >= 0.6 * double(layer_covered_))
                    layer_trust_ = 1;
                else if (layer_frames_ >= 60 && double(layer_covered_) >= 0.25 * frame && layer_drawn_over_ < 0.25 * double(layer_covered_))
                    layer_trust_ = 2;
            }
            D3D12_RANGE none{0, 0};
            layer_readback_->Unmap(0, &none);
        }
    }
    if (hud_pending_[frame_index_] && hud_readback_) {
        hud_pending_[frame_index_] = false;
        std::uint32_t* v = nullptr;
        D3D12_RANGE range{frame_index_ * 16, frame_index_ * 16 + 8};
        if (SUCCEEDED(hud_readback_->Map(0, &range, reinterpret_cast<void**>(&v)))) {
            const std::uint32_t evidence = v[frame_index_ * 4], changed = v[frame_index_ * 4 + 1];
            ++hud_stats_.frames;
            hud_stats_.evidence_sum += evidence; hud_stats_.changed_sum += changed;
            // Same rule as cs_hud.
            if (evidence + changed < 256) ++hud_stats_.too_little;
            else if (evidence > 0.35 * (evidence + changed)) ++hud_stats_.guarded;
            else { ++hud_stats_.learned; hud_stats_.share_sum += double(evidence) / double(evidence + changed); }
            // (is it learning here: at least a tenth of every 300 frames learned from)
            ++learn_frames_;
            if (evidence + changed >= 256 && evidence <= 0.35 * (evidence + changed)) ++learn_learned_;
            if (learn_frames_ >= 300) {
                learning_stalled_ = learn_learned_ < 30;
                learn_frames_ = learn_learned_ = 0;
            }
            D3D12_RANGE none{0, 0};
            hud_readback_->Unmap(0, &none);
        }
    }
    if (ret_args_pending_[frame_index_] && ret_args_readback_) {
        ret_args_pending_[frame_index_] = false;
        std::uint32_t* v = nullptr;
        D3D12_RANGE range{SIZE_T(frame_index_) * 32, SIZE_T(frame_index_ + 1) * 32};
        if (SUCCEEDED(ret_args_readback_->Map(0, &range, reinterpret_cast<void**>(&v)))) {
            const std::uint32_t* a = v + frame_index_ * 8;
            if (a[0] && a[1]) ret_window_seen_ = true;  // (the passes over the screen dispatched: a sight's window seen lately)
            D3D12_RANGE none{0, 0};
            ret_args_readback_->Unmap(0, &none);
        }
    }
    if (share_diag_pending_[frame_index_] && share_readback_) {
        share_diag_pending_[frame_index_] = false;
        std::uint32_t* v = nullptr;
        D3D12_RANGE range{SIZE_T(frame_index_) * kShareDiagBytes, SIZE_T(frame_index_ + 1) * kShareDiagBytes};
        if (SUCCEEDED(share_readback_->Map(0, &range, reinterpret_cast<void**>(&v)))) {
            const std::uint32_t* p = v + frame_index_ * (kShareDiagBytes / 4);
            ShareDiag d = share_diag_meta_[frame_index_];
            for (int b = 0; b < 12; ++b) { std::memcpy(&d.band_a[b], &p[12 + b], 4); std::memcpy(&d.band_b[b], &p[36 + b], 4); }
            for (int k = 0; k < 12; ++k) d.v[k] = static_cast<std::int32_t>(p[48 + k]);
            if (share_diags_.size() < 600) share_diags_.push_back(d);
            D3D12_RANGE none{0, 0};
            share_readback_->Unmap(0, &none);
        }
    }
    // The game frame's passes recorded in this slot last time (see pass_stamp).
    if (pass_count_[frame_index_] > 1 && pass_readback_ && frame_values_[frame_index_]) {
        std::uint64_t* t = nullptr;
        D3D12_RANGE pr{SIZE_T(frame_index_) * kPassStamps * 8, SIZE_T(frame_index_ + 1) * kPassStamps * 8};
        if (SUCCEEDED(pass_readback_->Map(0, &pr, reinterpret_cast<void**>(&t)))) {
            const std::uint64_t* s = t + frame_index_ * kPassStamps;
            bool ordered = true;
            for (int k = 1; k < pass_count_[frame_index_]; ++k) ordered = ordered && s[k] >= s[k - 1];
            if (ordered) {
                for (int k = 1; k < pass_count_[frame_index_]; ++k)
                    pass_times_.ms[pass_ids_[frame_index_][k]] += double(s[k] - s[k - 1]) * 1000.0 / double(timestamp_frequency_);
                ++pass_times_.frames;
            }
            D3D12_RANGE none{0, 0};
            pass_readback_->Unmap(0, &none);
        }
    }
    pass_count_[frame_index_] = 0;
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
                       std::uint32_t w, std::uint32_t h, std::uint32_t source_w, std::uint32_t source_h) {
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = srv_format(source_format);
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // (a one-channel UI layer is the coverage of the game's HUD layer, as the add-on hands it over: read as (0, 0, 0, a) -
    // unless it is kept one-channel, see set_x_srv)
    if (kind == kUi && source_format == DXGI_FORMAT_R8_UNORM && private_[target].format != DXGI_FORMAT_R8_UNORM)
        sv.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
            D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
            D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0);
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
    const bool resample = target != kPDepth && source_w && source_h && (source_w != w || source_h != h);
    const std::uint32_t constants[4] = {w, h, resample ? source_w : 0u, resample ? source_h : 0u};
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
    {  // (one 16x16 group per pixel of level 4, see cs_flow_luma)
        const UINT base = table(kFlowLuma, kPBackbuffer, kPBackbuffer, kPBackbuffer, now);
        transition(private_[now], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        x_dispatch(cs_flow_luma_.Get(), &c, base, base + 6, (w0 + 15) / 16, (h0 + 15) / 16);
        transition(private_[now], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
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

void Renderer::score_hud_probe(const IngestedSource& src, const float clip_to_prev_clip[16]) {
    const std::uint32_t cand = src.has_probe ? src.hud_candidate : 0;
    const auto& depth = private_[kPDepth];
    if (!cand || !private_[kPProbe].texture || !depth.texture || !probe_sums_) { probe_prev_ = 0; return; }
    const UINT ow = private_[kPProbe].width, oh = private_[kPProbe].height, tw = (ow + 15) / 16, th = (oh + 15) / 16;
    if (!ensure_private(kPProbeTilesA, tw, th, DXGI_FORMAT_R16_FLOAT) || !ensure_private(kPProbeTilesB, tw, th, DXGI_FORMAT_R16_FLOAT)) {
        probe_prev_ = 0;
        return;
    }
    const PrivateId now = probe_flip_ ? kPProbeTilesA : kPProbeTilesB, prev = probe_flip_ ? kPProbeTilesB : kPProbeTilesA;
    XConstants c{};
    std::memcpy(c.clip_to_prev, clip_to_prev_clip, sizeof(c.clip_to_prev));
    const UINT w = depth.width, h = depth.height;
    const Rect2 r = src.depth_rect.w ? src.depth_rect : Rect2{0, 0, w, h};
    c.rect[0] = r.x; c.rect[1] = r.y; c.rect[2] = std::min(r.w, w - r.x); c.rect[3] = std::min(r.h, h - r.y);
    c.grid[0] = w; c.grid[1] = h;
    c.mv_size[0] = tw; c.mv_size[1] = th;
    c.out_size[0] = ow; c.out_size[1] = oh;
    for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXProbeTilesSrv + i, kPProbe);
    set_x_uav(kXProbeTilesUav + 0, now); set_x_uav(kXProbeTilesUav + 1, now);
    transition(private_[now], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    x_dispatch(cs_probe_tiles_.Get(), &c, kXProbeTilesSrv, kXProbeTilesUav, (tw + 7) / 8, (th + 7) / 8);
    transition(private_[now], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (probe_prev_ == cand && private_[prev].texture) {
        for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(kXProbeScoreSrv + i, kPDepth);
        set_x_srv(kXProbeScoreSrv + 0, now);
        set_x_srv(kXProbeScoreSrv + 1, prev);
        set_x_srv(kXProbeScoreSrv + 2, kPDepth);
        D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
        raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; raw.Format = DXGI_FORMAT_R32_TYPELESS;
        raw.Buffer.NumElements = 8; raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(probe_sums_.Get(), nullptr, &raw, cpu(kXProbeScoreUav + 0));
        device_->CreateUnorderedAccessView(probe_sums_.Get(), nullptr, &raw, cpu(kXProbeScoreUav + 1));
        D3D12_RESOURCE_BARRIER sums{}; sums.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; sums.UAV.pResource = probe_sums_.Get();
        x_dispatch(cs_probe_clear_.Get(), &c, kXProbeScoreSrv, kXProbeScoreUav, 1, 1);
        list_->ResourceBarrier(1, &sums);
        x_dispatch(cs_probe_score_.Get(), &c, kXProbeScoreSrv, kXProbeScoreUav, (tw + 7) / 8, (th + 7) / 8);
        auto to_copy = transition_barrier(probe_sums_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list_->ResourceBarrier(1, &to_copy);
        list_->CopyBufferRegion(probe_readback_.Get(), UINT64(frame_index_) * 32, probe_sums_.Get(), 0, 32);
        std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
        list_->ResourceBarrier(1, &to_copy);
        probe_pending_[frame_index_] = true;
        probe_pending_cand_[frame_index_] = cand;
        probe_pending_locked_[frame_index_] = src.hud_learned == 1;
        probe_pending_tiles_[frame_index_] = tw * th;
    }
    probe_prev_ = cand;
    probe_flip_ = !probe_flip_;
}

// The pick: of the candidates scored on 30 frames or more that change under 60% of the screen, stay on the screen (0.9
// or more) and clearly more than they move with the scenery (by 0.2), the one changing the most - the earliest point
// after which only the HUD is drawn.
std::uint32_t Renderer::hud_learn_pick() const {
    std::uint32_t pick = 0;
    double pick_area = -1;
    for (const auto& [cand, sc] : probe_scores_) {
        if (sc.scored < 30 || !sc.frames || sc.same_max <= 0 || sc.world_max <= 0) continue;
        const double area = sc.area / sc.frames, same = sc.same_min / sc.same_max, world = sc.world_min / sc.world_max;
        if (area < 0.6 && same >= 0.9 && same - world >= 0.2 && area > pick_area) { pick = cand; pick_area = area; }
    }
    return pick;
}

std::string Renderer::hud_learn_report() const {
    std::string out;
    char item[160];
    if (const std::uint32_t pick = hud_learn_pick()) { std::snprintf(item, sizeof(item), "pick %08x | ", pick); out += item; }
    else out += "no pick yet | ";
    if (probe_locked_) {
        std::snprintf(item, sizeof(item), "in use %08x: stays on the screen %.2f, moves with the scenery %.2f (%u) | ", probe_locked_, probe_check_same_,
                      probe_check_world_, probe_check_n_);
        out += item;
    }
    for (const auto& [cand, sc] : probe_scores_) {
        std::snprintf(item, sizeof(item), "%s%08x: %u frames, differs on %.1f%%, stays on the screen %.2f, moves with the scenery %.2f (%u scored)",
                      out.empty() ? "" : " | ", cand, sc.frames, sc.frames ? 100.0 * sc.area / sc.frames : 0.0,
                      sc.same_max > 0 ? sc.same_min / sc.same_max : 0.0, sc.world_max > 0 ? sc.world_min / sc.world_max : 0.0, sc.scored);
        out += item;
    }
    return out;
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
            // (motion vectors at another resolution than the depth - output resolution: sampled at the matching
            // positions over the depth's grid, not cropped to its corner)
            if (mv.width > dp.width || mv.height > dp.height)
                convert(sources[kMotion], static_cast<DXGI_FORMAT>(mv.format), slot, kMotion, kPMotion, dp.width, dp.height, mv.width, mv.height);
            else
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
    // (the coverage of the game's HUD layer, as the add-on hands it over, stays one channel: 8 MB at 4K instead of 64)
    const DXGI_FORMAT ui_format = static_cast<DXGI_FORMAT>(ui.format) == DXGI_FORMAT_R8_UNORM ? DXGI_FORMAT_R8_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (sources[kUi] && ui.width == bb.width && ui.height == bb.height && ensure_private(kPUi, bb.width, bb.height, ui_format)) {
        convert(sources[kUi], static_cast<DXGI_FORMAT>(ui.format), slot, kUi, kPUi, bb.width, bb.height);
        out.has_ui = true;
    }
    const auto& pb = m.tex[kHudProbe];
    if (sources[kHudProbe] && m.hud_candidate && pb.width == bb.width && pb.height == bb.height &&
        ensure_private(kPProbe, bb.width, bb.height, DXGI_FORMAT_R8_UNORM)) {
        convert(sources[kHudProbe], static_cast<DXGI_FORMAT>(pb.format), slot, kHudProbe, kPProbe, bb.width, bb.height);
        out.has_probe = true;
        out.hud_candidate = m.hud_candidate;
    }
    out.hud_learned = out.has_ui ? m.hud_learned : 0;
    const auto& sc = m.tex[kScene];
    if (sources[kScene] && sc.width == bb.width && sc.height == bb.height &&
        ensure_private(kPScene, bb.width, bb.height, colour_format(static_cast<DXGI_FORMAT>(sc.format)))) {
        convert(sources[kScene], static_cast<DXGI_FORMAT>(sc.format), slot, kScene, kPScene, bb.width, bb.height);
        out.has_scene = true;
        scene_idle_ = 0;
    } else if (++scene_idle_ == 300) {
        // (the upscaler's output no longer comes - the add-on leaves it out while the game hands its HUD over: released
        // once the GPU is past it, no waiting)
        retire(kPScene);
        for (int s = 0; s < kSlots; ++s) {  // (and the add-on's slot textures it had open: theirs are released too)
            retire_texture(shared_[s][kScene].resource);
            shared_[s][kScene] = SharedTex{};
            srv_resource_[kSrcSrv + s * kTexCount + kScene] = nullptr;
        }
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

LatewarpInputs Renderer::latewarp_inputs(const IngestedSource& src, bool use_ui_tags, bool for_latewarp) {
    LatewarpInputs in;
    const bool split = use_ui_tags && src.has_hudless && src.has_ui;
    in.backbuffer = private_[kPBackbuffer].texture.Get();
    in.hudless = split ? private_[kPHudless].texture.Get() : in.backbuffer;
    // Latewarp always takes a UI layer: an empty one (zero-initialised, 8-bit, screen size) for games without - only
    // while Latewarp is the engine (32 MB at 4K).
    if (for_latewarp && !split && private_[kPBackbuffer].texture)
        ensure_private(kPZeroUi, private_[kPBackbuffer].width, private_[kPBackbuffer].height, DXGI_FORMAT_R8G8B8A8_UNORM);
    else if (!for_latewarp && private_[kPZeroUi].texture) {
        retire(kPZeroUi);
    }
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

void Renderer::finish_frame(bool warped, int marker, bool newest) {
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
        last_submit_qpc_ = q.QuadPart;
    }
    {
        LARGE_INTEGER a, b, f;
        QueryPerformanceCounter(&a);
        swapchain_->Present(newest ? 0 : 1, 0);
        QueryPerformanceCounter(&b); QueryPerformanceFrequency(&f);
        last_present_call_ms_ = float(double(b.QuadPart - a.QuadPart) * 1000.0 / double(f.QuadPart));
    }
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

void Renderer::release_stash() {
    wait_idle();
    for (const PrivateId id : {kPStashColor, kPStashDepth, kPStashMotion, kPStashHudless, kPStash2Color, kPStash2Hudless, kPStash2Depth, kPStash2Motion})
        private_[id] = Private{};
}

std::string Renderer::memory_report() {
    static const char* const kNames[] = {"Backbuffer", "Hudless", "Ui", "Depth", "Motion", "ZeroUi", "Output", "Object", "ObjKeys", "ObjMove", "ObjTiles", "ObjOutTiles", "ObjKindA", "ObjKindB", "Gen0", "Gen1", "Gen2", "GenZ0", "GenZ1", "GenZ2", "PrevColour", "PrevDepth", "Previous", "HudScore", "Mask", "Scene", "Attached", "World", "Fill", "Mem0", "Mem1", "MemZ0", "MemZ1", "RampRows", "Ramp", "WarpMask", "HudKind", "StashColor", "StashDepth", "StashMotion", "StashHudless", "Stash2Color", "Stash2Hudless", "Stash2Depth", "Stash2Motion", "LumaA", "LumaB", "Feat", "FlowA", "FlowB", "RetCells", "RetEncl", "RetA", "RetB", "RetLayer", "FgMotion", "FgInv", "FgK1", "FgK0", "HudLearned", "HudTiles", "WarpClaims", "Still", "GrowA", "GrowB", "Probe", "ProbeTilesA", "ProbeTilesB", "ShareA", "ShareB", "ShareT", "Share", "RetDirtyA", "RetDirtyB", "ShareT2", "ShareHalf", "RecColour", "RecUi", "ShareResid", "OwnMove"};
    static_assert(sizeof(kNames) / sizeof(kNames[0]) == kPCount, "a name for every private texture");
    struct Item { std::string name; double mb; };
    std::vector<Item> items;
    double total = 0;
    auto add = [&](const std::string& name, ID3D12Resource* r) {
        if (!r) return;
        const auto desc = r->GetDesc();
        const double mb = double(device_->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes) / 1048576.0;
        char text[96];
        std::snprintf(text, sizeof(text), "%s %llux%u", name.c_str(), static_cast<unsigned long long>(desc.Width), desc.Height);
        items.push_back({text, mb});
        total += mb;
    };
    for (int i = 0; i < kPCount; ++i) {
        add(kNames[i], private_[i].texture.Get());
        add(std::string(kNames[i]) + " (shown)", front_[i].texture.Get());
    }
    double shared = 0;
    for (int s = 0; s < kSlots; ++s)
        for (int t = 0; t < kTexCount; ++t)
            if (ID3D12Resource* r = shared_[s][t].resource.Get()) {
                const auto desc = r->GetDesc();
                shared += double(device_->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes) / 1048576.0;
            }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.mb > b.mb; });
    std::string out;
    char text[160];
    std::snprintf(text, sizeof(text), "own textures %.0f MB (%zu), game frame slots %.0f MB; largest:", total, items.size(), shared);
    out = text;
    for (std::size_t i = 0; i < items.size() && i < 14; ++i) {
        std::snprintf(text, sizeof(text), " %s %.0f,", items[i].name.c_str(), items[i].mb);
        out += text;
    }
    if (!out.empty() && out.back() == ',') out.pop_back();
    return out;
}

std::uint32_t Renderer::warp_rest_blocks() {
    if (!warp_list_) return 0;
    wait_idle();
    context_ = 0; list_ = lists_[0]; frame_index_ = static_cast<std::uint32_t>(ring_pos_[0]);
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = 256; bd.Height = 1;
    bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> buffer;
    if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) return 0;
    allocators_[frame_index_]->Reset();
    list_->Reset(allocators_[frame_index_].Get(), nullptr);
    list_->CopyBufferRegion(buffer.Get(), 0, warp_list_.Get(), 0, 4);
    list_->Close();
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    wait_idle();
    std::uint32_t* mapped = nullptr, n = 0;
    if (SUCCEEDED(buffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) { n = mapped[0]; buffer->Unmap(0, nullptr); }
    return n;
}

bool Renderer::read_back(int which, std::vector<std::uint16_t>& pixels, std::uint32_t& w, std::uint32_t& h) {
    static const PrivateId kWhich[] = {kPBackbuffer, kPOutput, kPScene, kPDepth, kPMotion, kPObject, kPMask, kPHudScore, kPWorld, kPFill,
                                       kPStashColor, kPStashDepth, kPStashMotion, kPFlowB, kPStashHudless, kPStash2Color, kPStash2Hudless,
                                       kPStash2Depth, kPStash2Motion, kPHudless, kPGen0, kPGen1, kPGen2, kPGenZ0, kPGenZ1, kPGenZ2, kPUi, kPWarpMask, kPRetLayer, kPFgMotion, kPFgK1, kPFgInv, kPHudLearned, kPHudTiles, kPHudKind, kPShare, kPRetEncl, kPAttached, kPMem0};
    if (which < 0 || which >= int(std::size(kWhich))) return false;
    // (the sight reticles' layer: the history texture written last)
    // (the background memory: the one written last)
    auto& p = kWhich[which] == kPRetLayer ? private_[ret_latest_] : kWhich[which] == kPMem0 ? private_[kPMem0 + std::max(mem_shown_, 0)]
            : split_ && shown_id(kWhich[which]) ? front_[kWhich[which]] : private_[kWhich[which]];
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
        if (desc.Format == DXGI_FORMAT_R16G16B16A16_UNORM) {
            for (std::uint32_t x = 0; x < w * 4; ++x) {
                std::uint16_t v16; std::memcpy(&v16, row + std::size_t(x) * 2, 2);
                out[x] = to_half(float(v16) / 65535.0f);
            }
            continue;
        }
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
            else if (desc.Format == DXGI_FORMAT_R16_UNORM) { std::uint16_t h16; std::memcpy(&h16, row + std::size_t(x) * 2, 2); v = h16; }
            else if (desc.Format == DXGI_FORMAT_R16G16_UNORM) std::memcpy(&v, row + std::size_t(x) * 4, 4);
            else std::memcpy(&v, row + std::size_t(x) * 4, 4);
            float c[4] = {0, 0, 0, 1};
            if (desc.Format == DXGI_FORMAT_R32_FLOAT) std::memcpy(&c[0], &v, 4);
            else if (desc.Format == DXGI_FORMAT_R8_UNORM) c[0] = float(v) / 255.0f;
            else if (desc.Format == DXGI_FORMAT_R16_UNORM) c[0] = float(v) / 65535.0f;
            else if (desc.Format == DXGI_FORMAT_R16G16_UNORM) { c[0] = float(v & 0xFFFFu) / 65535.0f; c[1] = float(v >> 16) / 65535.0f; }
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

// Sequence recording (development): the frame just taken in - its picture shrunk to about 1080 lines (R11G11B10), its depth
// and motion vectors as they are, the HUD layer's coverage if asked - copied into read-back buffers on the list being
// built (after the intake's own work), never waited for: take_recorded hands them over once the GPU is done. False: all
// slots still in flight (the frame is not recorded).
bool Renderer::record_frame(std::uint64_t tag, bool ui) {
    const auto& c0 = private_[kPBackbuffer];
    if (!c0.texture || !private_[kPDepth].texture || !private_[kPMotion].texture || !cs_rec_down_) return false;
    int idx = -1;
    for (int k = 0; k < kRecSlots && idx < 0; ++k) {
        const int i = (rec_next_ + k) % kRecSlots;
        if (rec_[i].state.load(std::memory_order_acquire) == 0) idx = i;
    }
    if (idx < 0) return false;
    rec_next_ = (idx + 1) % kRecSlots;
    RecSlot& r = rec_[idx];
    const UINT f = std::max(1u, static_cast<UINT>(std::lround(double(c0.height) / 1080.0)));
    const UINT cw = (c0.width + f - 1) / f, ch = (c0.height + f - 1) / f;
    ui = ui && private_[kPUi].texture && private_[kPUi].width == c0.width && private_[kPUi].height == c0.height;
    if (!ensure_private(kPRecColour, cw, ch, DXGI_FORMAT_R11G11B10_FLOAT) || !ensure_private(kPRecUi, cw, ch, DXGI_FORMAT_R8_UNORM)) return false;
    // (the read-back buffers, made again when a size changes)
    const PrivateId ids[4] = {kPRecColour, kPDepth, kPMotion, kPRecUi};
    for (int k = 0; k < 4; ++k) {
        const auto desc = private_[ids[k]].texture->GetDesc();
        UINT64 total = 0;
        device_->GetCopyableFootprints(&desc, 0, 1, 0, &r.foot[k], nullptr, nullptr, &total);
        if (!r.buffer[k] || r.buffer[k]->GetDesc().Width < total) {
            r.buffer[k].Reset();
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
            bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r.buffer[k]))))
                return false;
        }
        r.width[k] = static_cast<std::uint32_t>(desc.Width); r.height[k] = desc.Height;
    }
    const UINT t = kXRec + UINT(idx) * 8;
    for (UINT i = 0; i < kXSrvCount; ++i) set_x_srv(t + i, kPBackbuffer);
    if (ui) set_x_srv(t + 1, kPUi);
    set_x_uav(t + kXSrvCount, kPRecColour);
    set_x_uav(t + kXSrvCount + 1, kPRecUi);
    transition(private_[kPRecColour], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(private_[kPRecUi], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    XConstants c{};
    c.grid[0] = cw; c.grid[1] = ch; c.out_size[0] = c0.width; c.out_size[1] = c0.height; c.groups_x = f; c.flags = ui ? 1u : 0u;
    x_dispatch(cs_rec_down_.Get(), &c, t, t + kXSrvCount, (cw + 7) / 8, (ch + 7) / 8);
    for (int k = 0; k < 4; ++k) {
        auto& src = private_[ids[k]];
        const auto before = src.state;
        transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst{r.buffer[k].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = r.foot[k];
        D3D12_TEXTURE_COPY_LOCATION from{src.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        list_->CopyTextureRegion(&dst, 0, 0, 0, &from, nullptr);
        transition(src, k == 0 || k == 3 ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : before);
    }
    r.tag = tag; r.has_ui = ui; r.context = context_; r.fence = 0;
    r.state.store(1, std::memory_order_release);
    return true;
}

// The oldest recorded frame the GPU has finished, in tight rows (colour R11G11B10, depth R32F, motion R16G16F, coverage R8).
// (one other thread may call it - the presenter's collector - while the render thread records: the slots' states are
// atomic, 0 free, 1 queued, 2 submitted)
bool Renderer::take_recorded(RecordedFrame& out) {
    int idx = -1;
    for (int i = 0; i < kRecSlots; ++i)
        if (rec_[i].state.load(std::memory_order_acquire) == 2 && (idx < 0 || rec_[i].tag < rec_[idx].tag)) idx = i;
    if (idx < 0) return false;
    RecSlot& r = rec_[idx];
    if (fences_[r.context]->GetCompletedValue() < r.fence) return false;
    static const std::uint32_t kBytes[4] = {4, 4, 4, 1};
    std::vector<std::uint8_t>* dst[4] = {&out.colour, &out.depth, &out.motion, &out.ui};
    for (int k = 0; k < 4; ++k) {
        dst[k]->clear();
        if (k == 3 && !r.has_ui) continue;
        std::uint8_t* mapped = nullptr;
        if (FAILED(r.buffer[k]->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) { r.state.store(0, std::memory_order_release); return false; }
        const std::size_t row = std::size_t(r.width[k]) * kBytes[k];
        dst[k]->resize(row * r.height[k]);
        for (std::uint32_t y = 0; y < r.height[k]; ++y)
            std::memcpy(dst[k]->data() + y * row, mapped + r.foot[k].Offset + std::size_t(y) * r.foot[k].Footprint.RowPitch, row);
        const D3D12_RANGE none{0, 0};
        r.buffer[k]->Unmap(0, &none);
    }
    out.tag = r.tag;
    out.cw = r.width[0]; out.ch = r.height[0]; out.dw = r.width[1]; out.dh = r.height[1];
    r.state.store(0, std::memory_order_release);
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
