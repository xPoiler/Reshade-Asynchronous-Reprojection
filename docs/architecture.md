# Architecture

Reviewed against the working tree on 2026-11-11. This describes the current source, including development changes; it is not a claim that every path has been validated in game. Settings and implementation details are defined by the linked source files. Release snapshots describe their own versions.

## Processes and frame transport

FrameWarp runs a ReShade add-on in the game and normally renders through a separate presenter process. The presenter uses D3D12 on the game's adapter and draws into a DirectComposition overlay. The tree also contains an in-process presenter DLL and development paths; their existence does not make them the default deployment.

```text
Game / ReShade add-on
  camera, depth, motion, HUD and frame capture
                 |
       shared textures + fence
       shared frame metadata
                 |
Presenter intake queue
  ingest, motion/camera analysis, HUD, mask, memory
                 |
       completed shown-frame state
                 |
Presenter refresh queue
  predicted camera -> warp -> composition overlay
```

The shared contract is [protocol.hpp](../src/shared/protocol.hpp). Four slots move through `Free -> Writing -> Ready -> Reading -> Free`. Texture generations distinguish recreated resources; frame metadata and fences associate captures with GPU completion. The presenter promotes completed intake work as a coherent shown-frame state, including its camera and textures.

D3D12 resources use shared handles. The D3D11 transport opens compatible D3D12 textures and shared fences on the game's adapter. Vulkan uses the interop implementation in [vk_transport.cpp](../src/addon/vk_transport.cpp). The overlay can appear on a display driven by another adapter through Windows composition, with potentially substantial extra cost.

## Data acquisition

| Source | Current role |
|---|---|
| [streamline_hooks.cpp](../src/addon/streamline_hooks.cpp) | Camera constants, tagged depth/motion/HUD resources and frame markers where the game supplies them. |
| [ngx_hooks.cpp](../src/addon/ngx_hooks.cpp) | DLSS inputs/output and supported generated images, including games without Streamline camera data. |
| [ffx_hooks.cpp](../src/addon/ffx_hooks.cpp) | Supported FSR upscaler and frame-generation interfaces. FSR 3.0 upscaling support does not imply support for its older frame-generation interface. |
| [reshade_feed.cpp](../src/addon/reshade_feed.cpp), [depth_choice.cpp](../src/addon/depth_choice.cpp), [XPAR.fx](../shaders/XPAR.fx) | Depth selection, orientation and frame feed when no hooked upscaler supplies the needed data. |
| [mv_probe.cpp](../src/addon/mv_probe.cpp), [motion_search.hpp](../src/presenter/motion_search.hpp) | Find and validate game motion-vector targets on D3D11/D3D12. Vulkan target searching is disabled (`kVulkanSearch = false`); Vulkan upscaler-provided vectors are a separate path. |
| [camera_search.cpp](../src/addon/camera_search.cpp), [camera_search.hpp](../src/presenter/camera_search.hpp) | Collect candidate matrices and check their predicted motion against observed motion on D3D11, D3D12 and Vulkan. |
| [ui_layer.cpp](../src/addon/ui_layer.cpp), [hud_learn.cpp](../src/addon/hud_learn.cpp) | Capture a HUD layer or a picture before HUD drawing, and learn suitable drawing points when direct patterns do not suffice. |

Game camera data and found camera matrices are checked against motion evidence. A valid found camera is followed directly; otherwise [camera_estimator.hpp](../src/presenter/camera_estimator.hpp) estimates camera motion from usable vectors or picture-derived flow. Camera search can account for constants supplied a frame ahead. Do not assume every game's callbacks arrive in the E33 ordering observed in early experiments.

Partial targets containing full motion only on drawn objects are supported. The remaining pixels initially use estimated flow; a validated camera can supply camera-plus-depth motion through `Renderer::fill_from_camera`. Targets containing only object motion relative to camera motion are detected but not used yet. The camera must be judged against independent motion evidence before using it to fill missing vectors.

With no usable scene depth, only camera rotation can be reprojected. Data source changes reset relevant learned state; depth selection and search can take time after loading or changing settings.

## Intake and refresh scheduling

[main.cpp](../src/presenter/main.cpp) coordinates the shown frame, camera model and display clock. [renderer.cpp](../src/presenter/renderer.cpp) implements GPU processing and presentation.

With paced XPAR rendering, intake runs on a separate compute queue while refresh work reads the completed shown set. Separate queues do **not** provide GPU execution isolation: intake, refreshes and the game compete for GPU time.

The intake gate (`Renderer::set_intake_gate`, enabled unless `IntakeBetweenWarps=0`) splits intake into pieces and schedules them after chosen refresh fences, using measured piece durations to fit gaps. It has release paths when refreshes stop. This reduces overlap; it does not guarantee an uninterrupted refresh or zero added source latency.

The display clock uses the active display mode and DXGI statistics. Automatic present lead adjusts when rendering starts to meet refreshes. A refresh that would show an unchanged picture can reuse the displayed result without a new draw/present. Neither queue priority nor a fixed amount of GPU work guarantees a particular refresh rate under all game loads.

The D3D11/D3D12 frame cap in [frame_cap.cpp](../src/addon/frame_cap.cpp) normally paces the game to reduce queueing. It stands aside for Reflex and frame generation. When GPU-bound, it instead limits queue depth by waiting for the preceding game frame's fence, with a bounded wait; `QueueLimit=0` disables that fallback. The add-on filters present callbacks to those associated with a ReShade game frame.

## Warp and image handling

Defaults come from `default_settings` in [producer.cpp](../src/addon/producer.cpp) and runtime setup in [main.cpp](../src/presenter/main.cpp). Saved per-game settings and developer overrides can change them.

| Setting/path | Current default |
|---|---|
| Engine | XPAR; NVIDIA Latewarp is optional. |
| Continuous displacement field | On (`ContinuousWarp` / `FW_TETHER` controls the development override). |
| Forward splat | Off (`Settings::warp_splat = 0`). |
| Follow share | On; input-driven and other camera motion are distinguished. |
| Background memory | On, at 1/4 of the picture's width and height. |
| Uncovered screen edges | Dithered. |
| Camera latency | Lowest latency without edge fill (`auto_prediction = 6`). |
| Present lead | Automatic; 6 ms is the manual setting's initial value. |
| Detailed diagnostics | Off unless enabled or forced by a developer override. |

Motion analysis and follow-share history identify how regions move relative to the camera. The continuous warp builds a displacement field that limits separation between neighboring regions, reducing tears while stretching nearby scenery. It does not reconstruct unseen geometry; large turns and thin strands can still show artifacts.

With splat off, each output pixel searches the field for its source. With splat on, rendered pixels claim output locations and nearer surfaces win. Both use the continuous field when enabled; splat off is not equivalent to disabling continuous warp. The current search path can separate ordinary pixels from the more expensive remainder (`cs_own_warp_seek` / `cs_own_warp_rest`), with fallback paths for reticle layers, debug paths and other conditions. This optimization's in-game validation is tracked separately from its implementation.

The HUD is composited from game-provided layers, captured/learned drawing points or detection, depending on available data. Layer validation currently depends on upscaler output in relevant paths. The D3D11 direct-back-buffer fourth capture pattern remains a support gap.

Reticles without their own depth use a learned sight layer. The layer is only applied while a sight window has been seen recently, and its history is restarted after a gap. Character and weapon pixels under continuous warp generally follow the field rather than forming the old hard held mask. The near-camera and turn rules still contribute to their classification when follow share is enabled.

Background memory retains eligible scenery and is updated with each game frame. Under the default continuous warp it mainly supplies screen-edge and HUD fills. Memory validity, visibility and guard state affect reuse; there is no universal promise that every remembered pixel expires after exactly one second or 60 frames.

The old XPAR moving-objects/frame-generation implementation was removed. The presenter keeps `set_warp_ahead(0)`; do not infer that the development residual-motion path is active. Supported DLSS and FSR 3.1 generated images from the game are still consumed and reprojected. Unsupported generation modes make FrameWarp pause.

NVIDIA Latewarp remains an optional engine requiring its DLL and suitable hardware. Its path registers a new rendered frame with a throwaway evaluation before the target-camera evaluation. Performance measurements from that path do not describe XPAR's current warp.

## Camera prediction and diagnostics

[pose.hpp](../src/presenter/pose.hpp) learns the camera response to mouse/controller input and predicts the display camera. [camera_motion.hpp](../src/shared/camera_motion.hpp) handles matrix-derived motion, and [frame_clock.hpp](../src/presenter/frame_clock.hpp) regularizes frame timing when simulation timestamps are unavailable. Use the model tests and recordings to validate changes; E33's early gain, delay and orbit measurements are not universal constants. The current presenter disables the measured-orbit warp rather than exposing the old manual orbit controls as an active default.

Routine diagnostics and log-only measurements are opt-in. Log rotation occurs when diagnostics first open in a run; crash and add-on initialization-failure reports are exceptions. See [testing and diagnostics](../dev/docs/testing-and-diagnostics.md) for tests, recording controls and verification limits.
