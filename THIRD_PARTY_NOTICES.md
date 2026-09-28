# Third-party notices

FrameWarp uses the following third-party components.

* **ReShade add-on API headers.** Copyright 2014 Patrick Mours. BSD 3-Clause License; see
  `third_party/reshade/LICENSE.md` in the source repository. https://github.com/crosire/reshade
* **Dear ImGui headers.** Copyright (c) 2014-2025 Omar Cornut. MIT License; see
  `third_party/imgui/LICENSE.txt`. https://github.com/ocornut/imgui
* **Vulkan headers** (Khronos Vulkan-Headers). Copyright 2015-2024 The Khronos Group Inc. Apache License 2.0;
  see `third_party/vulkan/LICENSE.md`. https://github.com/KhronosGroup/Vulkan-Headers
* **NVIDIA NGX SDK** (the `nvsdk_ngx_d.lib` loader, linked into `FrameWarpPresenter.exe`). Copyright (c)
  NVIDIA Corporation, used under NVIDIA's SDK license terms. It is not part of this source repository;
  get it from https://github.com/NVIDIA/DLSS.
* **NVIDIA Reflex 2 Frame Warp (`nvngx_latewarp.dll`).** Copyright (c) NVIDIA Corporation. It is
  **not included** and optional; users who want it supply it. FrameWarp loads it at run time through NGX.
* **NVIDIA Streamline.** FrameWarp includes no Streamline code or binaries. It hooks the game's own
  `sl.interposer.dll` at run time to read the data the game already provides.
* **AMD FidelityFX (FSR).** FrameWarp includes no FidelityFX code or binaries. It hooks the game's own
  FSR DLLs at run time; the data layouts it reads follow AMD's public FidelityFX SDK and FSR 2 headers
  (Copyright (c) Advanced Micro Devices, Inc., MIT License). https://github.com/GPUOpen-LibrariesAndSDKs

NVIDIA, RTX, DLSS and Reflex are trademarks of NVIDIA Corporation. AMD, FidelityFX and FSR are
trademarks of Advanced Micro Devices, Inc. FrameWarp is an independent project and is not affiliated
with or endorsed by NVIDIA, AMD, the ReShade project, or any game developer.
