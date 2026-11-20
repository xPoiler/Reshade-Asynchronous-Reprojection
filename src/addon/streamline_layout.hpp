#pragma once
// sl::Constants and tag geometry as plain functions (no hooks, no devices): streamline_hooks.cpp uses them,
// tests/streamline_tests.cpp checks them.
#include "shared/protocol.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace fw::sl_layout {

// sl::Constants after its BaseStructure, as far as it is read (Streamline 2.x, sl_consts.h).
constexpr std::size_t kConstantsBytes = 424;
constexpr std::size_t kViewToClip = 0, kClipToView = 64, kClipToPrevClip = 192, kJitter = 320, kMvecScale = 328, kPos = 344, kUp = 356,
                      kRight = 368, kFwd = 380, kNear = 392, kFar = 396, kFov = 400, kAspect = 404, kDepthInverted = 412, kReset = 415;

// Streamline marks a value the game did not give with sl::INVALID_FLOAT (FLT_MAX).
inline bool is_set(float v) { return std::isfinite(v) && std::fabs(v) < 1e30f; }

// Every field, verbatim. The camera is the game's only when its basis is one (is_unit_basis in streamline_hooks.cpp).
inline void parse_constants(const std::uint8_t* raw, Camera& cam) {
    std::memcpy(cam.view_to_clip, raw + kViewToClip, 64);
    std::memcpy(cam.clip_to_view, raw + kClipToView, 64);
    std::memcpy(cam.clip_to_prev_clip, raw + kClipToPrevClip, 64);
    std::memcpy(cam.jitter, raw + kJitter, 8);
    std::memcpy(cam.mvec_scale, raw + kMvecScale, 8);
    std::memcpy(cam.pos, raw + kPos, 12);
    std::memcpy(cam.up, raw + kUp, 12);
    std::memcpy(cam.right, raw + kRight, 12);
    std::memcpy(cam.fwd, raw + kFwd, 12);
    std::memcpy(&cam.near_plane, raw + kNear, 4);
    std::memcpy(&cam.far_plane, raw + kFar, 4);
    std::memcpy(&cam.fov, raw + kFov, 4);
    std::memcpy(&cam.aspect, raw + kAspect, 4);
    cam.depth_inverted = raw[kDepthInverted] == 1;
    cam.reset = raw[kReset] == 1;
}

// Constants that carry no camera (a game that gives the upscaler only what DLSS needs: jitter and the motion vector
// scale, its matrices and camera vectors left unset) as an estimated camera: the presenter works the camera out from
// the motion vectors, as for DLSS without Streamline. False when not even jitter and the motion vector scale are set.
// mvec_scale is still Streamline's (motion vector -> share of the screen): see scale_to_pixels.
inline bool estimated_camera(const std::uint8_t* raw, Camera& cam) {
    Camera c{};
    parse_constants(raw, c);
    if (!is_set(c.jitter[0]) || !is_set(c.jitter[1]) || !is_set(c.mvec_scale[0]) || !is_set(c.mvec_scale[1])) return false;
    cam = Camera{};
    cam.jitter[0] = c.jitter[0]; cam.jitter[1] = c.jitter[1];
    cam.mvec_scale[0] = c.mvec_scale[0] != 0 ? c.mvec_scale[0] : 1.0f;
    cam.mvec_scale[1] = c.mvec_scale[1] != 0 ? c.mvec_scale[1] : 1.0f;
    cam.near_plane = is_set(c.near_plane) ? c.near_plane : 0.0f;
    cam.far_plane = is_set(c.far_plane) ? c.far_plane : 0.0f;
    cam.fov = is_set(c.fov) ? c.fov : 0.0f;
    cam.aspect = is_set(c.aspect) ? c.aspect : 0.0f;
    cam.depth_inverted = c.depth_inverted;
    cam.reset = c.reset;
    cam.valid = 1;
    cam.estimated = 1;
    return true;
}

// sl::Constants::mvecScale takes the motion vectors to a share of the screen (Streamline multiplies by the render size
// for DLSS); the presenter's estimated camera wants pixels of the depth it is given, as the NGX and FFX hooks hand it.
inline void scale_to_pixels(Camera& cam, std::uint32_t depth_w, std::uint32_t depth_h) {
    cam.mvec_scale[0] *= static_cast<float>(depth_w);
    cam.mvec_scale[1] *= static_cast<float>(depth_h);
}

struct Rect { std::uint32_t x = 0, y = 0, w = 0, h = 0; };
inline bool operator==(const Rect& a, const Rect& b) { return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h; }

// A tag's size: its extent when it has one, else its texture's.
inline Rect tag_size(Rect extent, std::uint32_t tex_w, std::uint32_t tex_h) {
    return extent.w && extent.h ? Rect{0, 0, extent.w, extent.h} : Rect{0, 0, tex_w, tex_h};
}

// The part of a tagged depth texture that holds the frame. Its extent when tagged with one (kept inside the texture).
// Tagged without one but larger than the render size: the render size from the top left - the texture is made for the
// largest size (dynamic resolution, an upscaler mode switched at run time) and the frame drawn into its top left, which
// is where an upscaler reads it. Else all of it.
inline Rect depth_area(std::uint32_t tex_w, std::uint32_t tex_h, Rect extent, std::uint32_t render_w, std::uint32_t render_h) {
    if (extent.w && extent.h && extent.x < tex_w && extent.y < tex_h) {
        extent.w = extent.w < tex_w - extent.x ? extent.w : tex_w - extent.x;
        extent.h = extent.h < tex_h - extent.y ? extent.h : tex_h - extent.y;
        return extent;
    }
    if (render_w && render_h && render_w <= tex_w && render_h <= tex_h && (render_w < tex_w || render_h < tex_h))
        return {0, 0, render_w, render_h};
    return {0, 0, tex_w, tex_h};
}

}  // namespace fw::sl_layout
