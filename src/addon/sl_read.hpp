#pragma once
// Reading what a Streamline game hands over (streamline_hooks.cpp), as pure functions so tests/sl_read_tests.cpp can check them:
// a frame's sl::Constants (a camera, none, or not at this base offset) and the area of the depth that holds the frame.
#include "shared/protocol.hpp"
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace fw::sl {

constexpr std::size_t kConstantsBytes = 424;  // (sl::Constants after its BaseStructure, as far as it is read)

// What sl::Constants at one base offset hold.
enum class Constants { Camera, NoCamera, NotHere };

inline bool unit_basis(const float* up, const float* right, const float* fwd) {
    auto len = [](const float* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    for (const float* v : {up, right, fwd})
        if (!(std::fabs(len(v) - 1.0f) < 0.02f)) return false;
    return std::fabs(dot(up, right)) < 0.02f && std::fabs(dot(up, fwd)) < 0.02f && std::fabs(dot(right, fwd)) < 0.02f;
}

// raw: kConstantsBytes of sl::Constants read at a base offset. Camera: a camera basis (cam filled). NoCamera: the matrices and the camera
// vectors left at sl::INVALID_FLOAT (FLT_MAX) - a game that hands over no camera; cam filled with what the constants still say (jitter,
// vector scale, depth direction, reset) and marked estimated. NotHere: neither - the constants do not start at this offset.
inline Constants read_constants(const std::uint8_t* raw, Camera& cam) {
    cam = Camera{};
    std::memcpy(cam.view_to_clip, raw + 0, 64);
    std::memcpy(cam.clip_to_view, raw + 64, 64);
    std::memcpy(cam.clip_to_prev_clip, raw + 192, 64);
    std::memcpy(cam.jitter, raw + 320, 8);
    std::memcpy(cam.mvec_scale, raw + 328, 8);
    std::memcpy(cam.pos, raw + 344, 12);
    std::memcpy(cam.up, raw + 356, 12);
    std::memcpy(cam.right, raw + 368, 12);
    std::memcpy(cam.fwd, raw + 380, 12);
    std::memcpy(&cam.near_plane, raw + 392, 4);
    std::memcpy(&cam.far_plane, raw + 396, 4);
    std::memcpy(&cam.fov, raw + 400, 4);
    std::memcpy(&cam.aspect, raw + 404, 4);
    cam.depth_inverted = raw[412] == 1;
    cam.reset = raw[415] == 1;
    if (unit_basis(cam.up, cam.right, cam.fwd)) return Constants::Camera;
    // (no camera: every float of the matrices' first row and of the camera's basis is INVALID_FLOAT, and the flags are 0 / 1 bytes)
    auto invalid = [](const float* v, int n) { for (int i = 0; i < n; ++i) if (v[i] != FLT_MAX) return false; return true; };
    const bool flags = raw[412] <= 1 && raw[413] <= 1 && raw[414] <= 1 && raw[415] <= 1;
    const bool scale = std::isfinite(cam.mvec_scale[0]) && std::isfinite(cam.mvec_scale[1]) && cam.mvec_scale[0] != 0 && cam.mvec_scale[1] != 0 &&
                       std::fabs(cam.mvec_scale[0]) < 1e4f && std::fabs(cam.mvec_scale[1]) < 1e4f;
    if (invalid(cam.view_to_clip, 4) && invalid(cam.clip_to_prev_clip, 4) && invalid(cam.up, 3) && invalid(cam.right, 3) && invalid(cam.fwd, 3) && flags && scale) {
        cam.estimated = 1;
        return Constants::NoCamera;
    }
    return Constants::NotHere;
}

// The area of a tagged texture that holds the frame: {x, y, w, h}.
struct Area { std::uint32_t x = 0, y = 0, w = 0, h = 0; };

// The depth's area. Its extent when tagged (kept inside the texture). Without one: the whole texture - unless it is bigger than the frame
// the motion vectors hold (their extent, or the whole of them), then the top left of that size, where a game that allocates its depth at
// the output size draws a smaller frame. Nothing known of the frame's size (no vectors tagged): the whole texture.
inline Area depth_area(std::uint32_t depth_w, std::uint32_t depth_h, const Area* depth_extent, std::uint32_t motion_w, std::uint32_t motion_h,
                       const Area* motion_extent) {
    if (depth_extent && depth_extent->w && depth_extent->h && depth_extent->x + depth_extent->w <= depth_w && depth_extent->y + depth_extent->h <= depth_h)
        return *depth_extent;
    Area frame{0, 0, motion_w, motion_h};
    if (motion_extent && motion_extent->w && motion_extent->h) frame = *motion_extent;
    if (frame.w && frame.h && frame.w <= depth_w && frame.h <= depth_h && (frame.w < depth_w || frame.h < depth_h)) return {0, 0, frame.w, frame.h};
    return {0, 0, depth_w, depth_h};
}

}  // namespace fw::sl
