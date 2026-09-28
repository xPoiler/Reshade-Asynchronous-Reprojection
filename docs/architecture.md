# Architecture

```
 GAME PROCESS                                               PRESENTER PROCESS (FrameWarpPresenter.exe)
 ReShade + FrameWarp.addon64                                 own D3D12 device on the same GPU
 ─────────────────────────────────                           ───────────────────────────────────────────
 inline hooks in sl.interposer.dll                           main thread: overlay window, raw mouse
   slSetConstants ─► camera (+ integrated position)            (RIDEV_INPUTSINK), window tracking
   slSetTag/ForFrame ─► GPU copy of depth / MV /             render thread (TIME_CRITICAL), per vblank:
                        HUD-less / UI into slot textures        1. newest slot whose game fence completed
 PCL marker hook ─► sim-start time, present frame id            2. convert shared → private textures
 reshade_present ─► GPU copy of backbuffer (incl. ReShade UI)   3. PoseModel.predict(now)
                    signal shared fence, publish slot           4. own warp engine, or NVIDIA Latewarp
 NGX / FSR hooks ─► depth / MV / upscaler output               5. blit → DirectComposition swapchain
          ═══ shared memory "Local\FrameWarp_<pid>" ═══        GPU priority: realtime process class
          ═══ shared D3D12 textures + fence (NT handles) ═══     + HIGH queue
```

## Source layout

| Path | Role |
|---|---|
| `src/shared/protocol.hpp` | Shared-memory contract (slots, settings, status, diagnostics, event timeline). Version-checked. |
| `src/shared/camera_motion.hpp` | Recovers frame-to-frame camera rotation + translation from `viewToClip` and `clipToPrevClip`. |
| `src/addon/addon.cpp` | ReShade registration, `reshade_present` capture, presenter launch, ImGui panel. |
| `src/addon/streamline_hooks.cpp` | Hooks for Streamline; runtime layout detection (BaseStructure size, ResourceTag stride); buffer classification. |
| `src/addon/producer.cpp` | Slot management, GPU copies into shared textures, shared fence, camera position integration, event log. |
| `src/addon/ngx_hooks.cpp` | Hooks on the driver's NGX exports (`CreateFeature`/`EvaluateFeature`): DLSS parameters, and for games without Streamline, depth, motion vectors and the DLSS output per evaluation. |
| `src/addon/ffx_hooks.cpp` | Hooks on AMD FSR: the FFX API (`amd_fidelityfx_*_dx12.dll`, FSR 3.1/4), the FidelityFX SDK 1.0 upscaler (`ffx_fsr3upscaler_x64.dll`, FSR 3.0) and FSR 2 (`ffx_fsr2_api_x64.dll`). Depth, motion vectors and the upscaled output per dispatch. |
| `src/addon/game_probe.cpp` | Byte-verified probes into Expedition 33's Unreal Streamline plugin (forces `ForceTagStreamlineBuffers()` true, counts calls). |
| `src/common/inline_hook.cpp` | Minimal x64 inline hook (prologue relocation incl. RIP-relative, near relay, atomic patch). |
| `src/presenter/main.cpp` | Presenter process: overlay window, raw input, render loop, logging, profile persistence. |
| `src/presenter/renderer.cpp` | D3D12 device/queue, composition swapchain, shared-texture ingest + format conversion, blit, timing. |
| `src/presenter/latewarp12.cpp` | NGX Latewarp on D3D12 (parameter names from `nvngx_latewarp.dll`); optional. |
| `src/presenter/pose.hpp` | Camera model: fitting, prediction/interpolation, handoff blending, cursor gate. |
| `src/presenter/camera_estimator.hpp` | Games without camera data: rotation, translation and field of view fitted to the motion vectors. |

## Game side (add-on)

* **Hooks.** `slSetConstants`, `slSetTag`, `slSetTagForFrame` (exports of `sl.interposer.dll`) and the
  PCL marker function (via `slGetFeatureFunction`) are inline-hooked. The add-on module pins itself
  because ReShade unloads/reloads add-ons during startup while hooks stay installed.
* **Frame association.** Everything is keyed by Streamline frame token. In Expedition 33 the constants, tags, the
  PCL present marker and the present call arrive in order on one thread per frame.
* **Slots.** 4 slots, state machine `Free → Writing (game) → Ready (fence signalled) → Reading
  (presenter) → Free`, with CAS transitions. The game never touches a slot being read.
* **Copies only.** Tagged resources are copied with `CopyTextureRegion` in the game's own command
  list at tag time (transition from the state Streamline reports → COPY_SOURCE and back). Depth is
  copied in its planar typeless format. The backbuffer is copied at `reshade_present` (after ReShade's
  effects and menu) and the shared fence is signalled on the game's queue.
* **Camera position.** Streamline's `cameraPos` is camera-relative (always 0), so the add-on integrates
  an absolute position every frame from `clipToPrevClip` (`camera_motion.hpp`); discontinuities bump
  `position_epoch`.
* **Games without Streamline, and FSR.** After 60 upscaler calls with no Streamline camera, each DLSS
  evaluation or FSR dispatch publishes its depth and motion vectors as a frame marked `estimated`; the
  presenter works out the camera (`camera_estimator.hpp`: a robust fit of rotation and translation to
  an 80x45 grid of motion vectors; the field of view learned by votes, or the one the game gives FSR
  once checked against the picture). A game that keeps sending its Streamline camera but no Streamline
  depth (Cyberpunk 2077 with FSR) gets FSR's depth and motion vectors attached to its own frames.
* **Upscaler output.** Unless *Find the HUD* is set to *Learned from camera motion*: copied right after the
  upscaler writes it (after `slEvaluateFeature` for Streamline games, after the NGX evaluation or the
  FSR dispatch otherwise).
* **Counters.** Presented images and rendered frames are counted for frame-generation detection.

## Presenter

* **Device and priority.** Own D3D12 device on the game's adapter (by LUID), direct queue at HIGH
  priority, process GPU scheduling class REALTIME (falls back to HIGH), process CPU class HIGH, render
  thread TIME_CRITICAL.
* **Never waits on the game GPU.** A slot is taken only when the game's shared fence has already
  completed (CPU check). Waiting on the GPU made every new frame cost 3 refreshes.
* **Ingest.** Shared textures (opened by name per generation) are converted by a compute shader into
  typed private textures (colour in the game's own 4-byte format when it holds the values exactly,
  RGBA16F otherwise; R32F depth; RG16F motion). A new
  game frame is taken in between refreshes, as its own GPU submission, so a refresh only warps and
  presents; one arriving too close to a refresh waits until just after it.
* **No-warp mask** (R8, used by the warp), built once per game frame:
  * *character/weapon* (every game): pixels whose motion vectors the camera motion does not explain,
    close to the camera, or stuck to the screen while the camera moves further out;
  * *HUD* (games without HUD layers): learned from pixels that stay the same at the same place while
    the camera moves the scene under them; or, with the option on, predicted per frame from the DLSS
    output (tone curve per channel, highlight wash-out, smooth per-tile correction) - what the
    prediction misses is HUD.
* **Frame generation.** When the game presents 1.6 or more images per rendered frame for a second, the
  presenter steps aside (overlay hidden, no GPU work) until the ratio is back near 1.
* **Warp engines.** The own engine (default, any D3D12 GPU): for each output pixel a short fixed-point
  search finds the rendered pixel that lands there at its own depth; masked pixels stay put and are
  never used as a source for others; revealed screen edges are filled with a short inward blend; no
  work while the camera has not moved. NVIDIA Latewarp (optional, `nvngx_latewarp.dll`, NVIDIA GPUs):
  on a new game frame a throwaway `IsRenderedFrame=1` evaluation registers it, followed by the real
  evaluation with the predicted camera. View matrices are built relative to the source camera
  position for precision.
* **Pacing.** A vblank clock is built from our swapchain's DXGI frame statistics. With present lead
  > 0 (default 6 ms) the swapchain allows one queued frame and the render thread wakes `lead` ms
  before each DWM composition deadline (vblank + 2.5 ms), rendering one frame per refresh; the
  schedule resets whenever the overlay becomes visible again. Lead 0: frame latency 1, render when
  the swapchain frees a buffer.
* **Output.** A layered, transparent, no-activate, topmost window follows the game's client area; a
  DirectComposition visual holds a flip-model composition swapchain (3 buffers, max latency 1,
  waitable). The window is hidden when the game is not in front or no frames are available. The
  monitor may be driven by another GPU: rendering stays on the game's GPU and Windows composes the
  overlay onto whichever display shows it.

## Camera model (`pose.hpp`)

Per axis (yaw about world up, pitch):

```
theta(t) = theta_N + tau*w_N*(1 - exp(-h/tau)) + g * sum_i dc_i * (1 - exp(-(t - s_i)/tau))
```

* `theta_N`, `w_N`: angle and angular velocity of the newest game frame; `dc_i`: raw mouse counts at
  `s_i` (shifted by input delay `d`); `tau`: the game's camera smoothing; `g`: radians per count.
* `tau` and `d` by grid search, `g` by least squares, refitted every 30 frames on the game's own camera
  history (frames with a visible cursor excluded).
* Displayed camera time = `now − latency + setting`, where `latency` is the measured sim-to-ingest time.
  Negative `h` interpolates between the last two real frames (angles and position exactly).
* Translation: residual camera velocity (orbit + walking) extrapolated; manual orbit pivot optional.
* Handoff: when a new frame arrives, the angular difference to the previous prediction is blended out
  over 50 ms (positions are not blended — tested worse).

## Shared memory protocol

`Shared` (see `protocol.hpp`) contains: producer identity/session/adapter, backbuffer format and
colour space, 4 `SlotMeta` (state, frame id, fence value, timestamps, `Camera` incl. matrices and
position, per-texture info), `Settings` (written by the UI), `PresenterStatus` (written by the
presenter), `HookStats` (diagnostics) and a 4096-entry event timeline ring. `kVersion` must match on
both sides; object names embed the game PID and a per-device session id.
