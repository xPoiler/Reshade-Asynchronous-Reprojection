#include "addon/ui_alpha.hpp"
#include <cmath>
#include <cstdio>

static int failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); } } while (0)

int main() {
    EXPECT(fw::needs_ui_alpha_reconstruction(DXGI_FORMAT_R10G10B10A2_UNORM),
           "10-bit color's two-bit alpha is reconstructed");
    EXPECT(fw::needs_ui_alpha_reconstruction(DXGI_FORMAT_R10G10B10A2_TYPELESS),
           "typeless 10-bit color's two-bit alpha is reconstructed");
    EXPECT(fw::needs_ui_blend_validation(DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_R10G10B10A2_UNORM),
           "BGRA8 UI blended into HDR10 is validated");
    EXPECT(!fw::needs_ui_alpha_reconstruction(DXGI_FORMAT_R8G8B8A8_UNORM),
           "8-bit UI alpha is used directly");
    EXPECT(!fw::needs_ui_alpha_reconstruction(DXGI_FORMAT_R16G16B16A16_FLOAT),
           "half-float UI alpha is used directly");
    EXPECT(fw::is_ui_alpha_format(DXGI_FORMAT_R8_UNORM), "8-bit single-channel UI alpha is accepted");
    EXPECT(fw::is_ui_alpha_format(DXGI_FORMAT_R16_FLOAT), "half-float single-channel UI alpha is accepted");
    EXPECT(!fw::is_ui_alpha_format(DXGI_FORMAT_R8G8B8A8_UNORM), "color textures are not accepted as alpha-only");
    EXPECT(fw::needs_ui_blend_validation(DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_R10G10B10A2_UNORM),
           "BGRA8 UI on an HDR10 frame is validated");
    EXPECT(!fw::needs_ui_blend_validation(DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM),
           "same-precision SDR UI keeps the documented blend path");

    const float hudless[3] = {0.6f, 0.4f, 0.2f};
    const float black_ui[3] = {0.0f, 0.0f, 0.0f};
    const float dimmed_frame[3] = {0.36f, 0.24f, 0.12f};
    const fw::UiAlphaEstimate black = fw::recover_ui_alpha(dimmed_frame, hudless, black_ui, 1.0f / 3.0f);
    EXPECT(black.recovered && std::fabs(black.alpha - 0.4f) < 0.002f,
           "translucent black alpha is recovered above two-bit precision");
    for (int i = 0; i < 3; ++i)
        EXPECT(std::fabs(black_ui[i] + (1.0f - black.alpha) * hudless[i] - dimmed_frame[i]) < 0.002f,
               "recovered black overlay recomposes to the final frame");

    const float premultiplied_ui[3] = {0.32f, 0.08f, 0.16f};
    const float colored_frame[3] = {0.68f, 0.32f, 0.28f};
    const fw::UiAlphaEstimate colored = fw::recover_ui_alpha(colored_frame, hudless, premultiplied_ui, 1.0f / 3.0f);
    EXPECT(colored.recovered && std::fabs(colored.alpha - 0.4f) < 0.002f,
           "translucent colored UI alpha is recovered");

    const float straight_ui[3] = {0.8f, 0.2f, 0.4f};
    const fw::UiBlendResult straight = fw::normalize_ui_blend(colored_frame, hudless, straight_ui, 0.4f);
    EXPECT(straight.mode == fw::UiBlendMode::straight && std::fabs(straight.alpha - 0.4f) < 0.001f &&
           std::fabs(straight.rgb[0] - 0.32f) < 0.001f, "straight-alpha BGRA is premultiplied before reprojection");

    const float inverted_black_frame[3] = {0.42f, 0.28f, 0.14f};
    const fw::UiBlendResult inverted_black = fw::normalize_ui_blend(inverted_black_frame, hudless, black_ui, 0.7f);
    EXPECT(inverted_black.mode == fw::UiBlendMode::inverted_premultiplied &&
           std::fabs(inverted_black.alpha - 0.3f) < 0.001f,
           "inverted alpha on a black translucent dim layer becomes opacity");
    const float inverted_straight_ui[3] = {0.8f, 0.2f, 0.4f};
    const fw::UiBlendResult inverted_straight =
        fw::normalize_ui_blend(colored_frame, hudless, inverted_straight_ui, 0.6f);
    EXPECT(inverted_straight.mode == fw::UiBlendMode::inverted_straight &&
           std::fabs(inverted_straight.alpha - 0.4f) < 0.001f,
           "inverted straight-alpha UI is normalized");

    const float pq_hudless[3] = {0.4f, 0.2f, 0.1f};
    const float pq_ui[3] = {0.8f, 0.3f, 0.5f};
    const float pq_frame[3] = {0.6f, 0.25f, 0.20f};
    const fw::UiBlendResult pq_residual = fw::normalize_ui_blend(pq_frame, pq_hudless, pq_ui, 0.4f);
    EXPECT(pq_residual.mode == fw::UiBlendMode::frame_residual && std::fabs(pq_residual.alpha - 0.4f) < 0.001f,
           "PQ-colored UI falls back to its observed frame residual when byte-color candidates do not fit");
    EXPECT(std::fabs(pq_residual.rgb[0] - 0.36f) < 0.001f && std::fabs(pq_residual.rgb[1] - 0.13f) < 0.001f &&
           std::fabs(pq_residual.rgb[2] - 0.14f) < 0.001f,
           "PQ-colored UI residual reproduces the captured final frame");

    const float pq_inverted_hudless[3] = {0.6f, 0.4f, 0.2f};
    const float pq_inverted_ui[3] = {0.9f, 0.3f, 0.2f};
    const float pq_inverted_frame[3] = {0.28f, 0.16f, 0.08f};
    const fw::UiBlendResult pq_inverted =
        fw::normalize_ui_blend(pq_inverted_frame, pq_inverted_hudless, pq_inverted_ui, 0.3f);
    EXPECT(pq_inverted.mode == fw::UiBlendMode::frame_residual && std::fabs(pq_inverted.alpha - 0.7f) < 0.001f,
           "PQ-colored UI with inverted tagged alpha selects the in-range residual polarity");
    EXPECT(std::fabs(pq_inverted.rgb[0] - 0.10f) < 0.001f && std::fabs(pq_inverted.rgb[1] - 0.04f) < 0.001f &&
           std::fabs(pq_inverted.rgb[2] - 0.02f) < 0.001f,
           "inverted PQ residual preserves the colored UI contribution");

    const float dark[3] = {0.0f, 0.0f, 0.0f};
    const fw::UiAlphaEstimate no_contrast = fw::recover_ui_alpha(dark, dark, black_ui, 1.0f / 3.0f);
    EXPECT(!no_contrast.recovered && no_contrast.alpha == 1.0f / 3.0f,
           "uninformative pixels retain their tagged alpha");

    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::printf("UI alpha format tests passed\n");
    return 0;
}
