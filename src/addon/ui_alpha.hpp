#pragma once
#include <algorithm>
#include <cmath>
#include <dxgiformat.h>

namespace fw {

inline constexpr bool needs_ui_alpha_reconstruction(DXGI_FORMAT format) {
    return format == DXGI_FORMAT_R10G10B10A2_TYPELESS || format == DXGI_FORMAT_R10G10B10A2_UNORM;
}

inline constexpr bool is_ui_alpha_format(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R8_TYPELESS:
        case DXGI_FORMAT_R8_UNORM:
        case DXGI_FORMAT_R16_TYPELESS:
        case DXGI_FORMAT_R16_UNORM:
        case DXGI_FORMAT_R16_FLOAT:
        case DXGI_FORMAT_R32_TYPELESS:
        case DXGI_FORMAT_R32_FLOAT:
            return true;
        default:
            return false;
    }
}

inline constexpr bool needs_ui_blend_validation(DXGI_FORMAT ui_format, DXGI_FORMAT frame_format) {
    const bool hdr10 = frame_format == DXGI_FORMAT_R10G10B10A2_TYPELESS || frame_format == DXGI_FORMAT_R10G10B10A2_UNORM;
    const bool byte_ui = ui_format == DXGI_FORMAT_R8G8B8A8_TYPELESS || ui_format == DXGI_FORMAT_R8G8B8A8_UNORM ||
                         ui_format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || ui_format == DXGI_FORMAT_B8G8R8A8_TYPELESS ||
                         ui_format == DXGI_FORMAT_B8G8R8A8_UNORM || ui_format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    return hdr10 && byte_ui;
}

struct UiAlphaEstimate {
    float alpha;
    bool recovered;
};

enum class UiBlendMode { premultiplied, straight, inverted_premultiplied, inverted_straight, frame_residual };

struct UiBlendResult {
    float rgb[3];
    float alpha;
    UiBlendMode mode;
    float error;
};

inline float ui_blend_error(const float final_rgb[3], const float hudless_rgb[3], const float ui_rgb[3], float alpha) {
    float error = 0.0f;
    for (int channel = 0; channel < 3; ++channel) {
        error = std::max(error, std::fabs(ui_rgb[channel] + (1.0f - alpha) * hudless_rgb[channel] - final_rgb[channel]));
    }
    return error;
}

inline UiBlendResult normalize_ui_blend(const float final_rgb[3], const float hudless_rgb[3], const float ui_rgb[3],
                                        float source_alpha) {
    UiBlendResult best{{ui_rgb[0], ui_rgb[1], ui_rgb[2]}, source_alpha, UiBlendMode::premultiplied,
                       ui_blend_error(final_rgb, hudless_rgb, ui_rgb, source_alpha)};
    auto consider = [&](UiBlendMode mode, float alpha, float scale) {
        float rgb[3] = {ui_rgb[0] * scale, ui_rgb[1] * scale, ui_rgb[2] * scale};
        const float error = ui_blend_error(final_rgb, hudless_rgb, rgb, alpha);
        if (error <= 0.02f && error + 0.002f < best.error) {
            best = {{rgb[0], rgb[1], rgb[2]}, alpha, mode, error};
        }
    };
    consider(UiBlendMode::straight, source_alpha, source_alpha);
    consider(UiBlendMode::inverted_premultiplied, 1.0f - source_alpha, 1.0f);
    consider(UiBlendMode::inverted_straight, 1.0f - source_alpha, 1.0f - source_alpha);
    if (best.error > 0.02f) {
        float normal[3], inverted[3];
        for (int channel = 0; channel < 3; ++channel) {
            normal[channel] = final_rgb[channel] - (1.0f - source_alpha) * hudless_rgb[channel];
            inverted[channel] = final_rgb[channel] - source_alpha * hudless_rgb[channel];
        }
        auto out_of_range = [](const float rgb[3]) {
            float penalty = 0.0f;
            for (int channel = 0; channel < 3; ++channel) {
                penalty = std::max(penalty, std::max(-rgb[channel], rgb[channel] - 1.0f));
            }
            return std::max(penalty, 0.0f);
        };
        const float normal_penalty = out_of_range(normal);
        const float inverted_penalty = out_of_range(inverted);
        const bool use_inverted = inverted_penalty + 0.002f < normal_penalty;
        const float* residual = use_inverted ? inverted : normal;
        best = {{residual[0], residual[1], residual[2]},
                use_inverted ? 1.0f - source_alpha : source_alpha,
                UiBlendMode::frame_residual, 0.0f};
    }
    return best;
}

// Streamline's premultiplied blend gives Final = UI + (1 - alpha) * Hudless.
// Recover alpha from that relation when the tagged UI texture only stores two
// alpha bits. The renderer uses the same least-squares estimate in its shader.
inline UiAlphaEstimate recover_ui_alpha(const float final_rgb[3], const float hudless_rgb[3], const float ui_rgb[3],
                                        float fallback_alpha) {
    float denominator = 0.0f, numerator = 0.0f;
    for (int channel = 0; channel < 3; ++channel) {
        denominator += hudless_rgb[channel] * hudless_rgb[channel];
        numerator += (final_rgb[channel] - ui_rgb[channel]) * hudless_rgb[channel];
    }
    if (denominator <= 1e-6f) return {fallback_alpha, false};

    const float estimate = 1.0f - numerator / denominator;
    const float alpha = estimate < 0.0f ? 0.0f : estimate > 1.0f ? 1.0f : estimate;
    float error = 0.0f;
    for (int channel = 0; channel < 3; ++channel) {
        const float reconstructed = ui_rgb[channel] + (1.0f - alpha) * hudless_rgb[channel];
        error = std::max(error, std::fabs(reconstructed - final_rgb[channel]));
    }
    return error <= 0.01f ? UiAlphaEstimate{alpha, true} : UiAlphaEstimate{fallback_alpha, false};
}

}  // namespace fw
