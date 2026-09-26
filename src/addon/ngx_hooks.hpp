#pragma once
#include "addon/producer.hpp"

namespace fw {
// Hooks the NVIDIA driver's NGX entry points (_nvngx.dll: NVSDK_NGX_D3D12_CreateFeature and
// NVSDK_NGX_D3D12_EvaluateFeature). Games that use DLSS without Streamline call these every frame
// with depth, motion vectors, jitter and the render size attached. Safe to call repeatedly: installs
// once the driver's NGX module is loaded.
void install_ngx_hooks(Producer* producer);
}
