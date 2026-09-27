#pragma once
#include "addon/producer.hpp"

namespace fw {
// Hooks AMD FidelityFX API entry points (ffxCreateContext / ffxDispatch) in the game's FSR DLL:
// amd_fidelityfx_dx12.dll (FSR 3.1), amd_fidelityfx_loader_dx12.dll or amd_fidelityfx_upscaler_dx12.dll
// (FSR 4). An FSR upscale dispatch carries depth, motion vectors, jitter, the render size, the output
// image and the camera's near/far planes and vertical field of view. Games without a Streamline camera
// (and not using DLSS at the time) publish their frames from it, like the DLSS route. Safe to call
// repeatedly: installs once one of the DLLs is loaded.
void install_ffx_hooks(Producer* producer);
}
