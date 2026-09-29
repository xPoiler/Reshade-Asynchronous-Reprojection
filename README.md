# FrameWarp

**Asynchronous camera reprojection for PC games.** FrameWarp shows camera motion at your display's
refresh rate, whatever frame rate the game renders at. For example, a game locked to 30 fps turns and
moves at 120 Hz on a 120 Hz display. In ReShade it appears as **XPAR** (xPoiler's Asynchronous
Reprojection).

On every display refresh, FrameWarp takes the game's newest frame and re-projects it to where the
camera is *now*. It runs as a ReShade add-on and reads what it needs from the game's upscaler:

* **DLSS**: camera, depth and HUD data from the game's NVIDIA Streamline integration, or depth and
  motion vectors from DLSS called directly, without Streamline.
* **AMD FSR 3** (3.0 and 3.1): depth and motion vectors from FSR. FSR 4 uses the same interface as
  FSR 3.1 and should work too. **FSR 2** should work in games that ship it as a DLL (untested).

Without the game's own camera data, FrameWarp works out the camera's movement from the motion
vectors. It then draws the result in a click-through overlay above the game, with its own warp engine
(or, optionally, NVIDIA's Latewarp).

## Tested games

| Game | Notes |
|---|---|
| *Clair Obscur: Expedition 33* (Steam, DLSS) | Add the launch options `-slforcetagging -slviewextension` (see Install). The game provides HUD layers, so the HUD stays still; FrameWarp also holds the character while the camera turns around it. |
| *Resident Evil Requiem* (Steam, DLSS or FSR) | **Requires [REFramework](https://github.com/praydog/REFramework)**: without it the game's DRM crashes with ReShade. The game doesn't provide HUD layers, so FrameWarp detects the HUD itself. With FSR, turn the camera for a second or two after loading while FrameWarp checks the field of view. |
| *Cyberpunk 2077* (Steam, DLSS or FSR 3) | No launch options needed. The game doesn't provide HUD layers: FrameWarp detects the HUD and V's weapon itself. Semi-transparent HUD panels may still move slightly. The game's FSR 2.1 option is built into the game itself and can't be used. |
| *Returnal* (Steam, DLSS) | No launch options needed. The game calls DLSS without Streamline, so FrameWarp works out the camera from DLSS's motion vectors; turn the camera for a few seconds after loading while it calibrates. |

Other DLSS and FSR games may work but are untested.

## Requirements

* A DirectX 12 game with **DLSS or FSR** (see above) turned on, or a Vulkan game with **DLSS** (tested with DOOM Eternal).
* A GPU for DirectX 12. FrameWarp is tested on NVIDIA GPUs; its own warp engine uses plain DirectX 12,
  so AMD and Intel GPUs should work too, but they are untested (reports welcome).
* Windows 11 (tested; Windows 10 may work) with **Hardware-accelerated GPU scheduling** turned on (Settings > System >
  Display > Graphics, then restart). Without it the output may not reach the refresh rate; the XPAR
  panel warns you if it is off.
* [ReShade](https://reshade.me) 6.x **with full add-on support**, installed for the game.
* **Frame generation off** (DLSS, FSR or XeSS frame generation). FrameWarp does the same job, filling
  your display's refresh rate, and the two can't be combined. While a game has frame generation on,
  FrameWarp pauses and says so in its panel.
* The [Microsoft Visual C++ Redistributable 2015–2022 (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

Optional: **NVIDIA Latewarp** (`nvngx_latewarp.dll`, NVIDIA GPUs only), NVIDIA's Reflex 2 Frame Warp
engine, as an alternative to FrameWarp's own engine. It is NVIDIA's file and cannot be included. It
comes bundled with software that uses Reflex 2 Frame Warp, such as community-made Reflex 2 demos, so
search for it by file name. FrameWarp was tested with version **310.2.0.0**: the file description is
"NVIDIA Latewarp" and the SHA256 is `81e63b4a9dcfb99b2b71c3ddfd3ea4a0e9780b778bd47e9a6f7f010a7a6c0f20`.
To check the version, right-click the file and open Properties > Details.

## Install

1. Download the latest `FrameWarp-<version>.zip` from
   [Releases](https://github.com/xPoiler/Reshade-Asynchronous-Reprojection/releases) and
   extract it.
2. Optional: put `nvngx_latewarp.dll` next to `install.bat` to have NVIDIA Latewarp available.
3. Close the game and double-click `install.bat`. It lists your Steam games that have ReShade; type
   the number of the game and press Enter. For a non-Steam game, choose **P** and type its folder.
   The installer copies FrameWarp next to the game's ReShade. For Unreal Engine games, it also
   enables Streamline resource tagging in the game's `Engine.ini`, keeping a backup of the file.
4. **Unreal Engine games with DLSS** (including Expedition 33): add these launch options (Steam >
   game > Properties > Launch Options):
   ```
   -slforcetagging -slviewextension
   ```

To uninstall, double-click `uninstall.bat` and choose the game. It reverts everything the installer
changed.

Advanced: both files also accept a game name or folder directly (`install.bat "Expedition 33"`), and
`install.ps1` accepts `-Game`, `-GameDir`, `-Latewarp <dll>`, `-EngineIni <path>`, `-NoCvar` and
`-Uninstall`.

## Use

1. Run the game in **borderless** mode, with DLSS or FSR on and frame generation off.
2. Open the ReShade overlay (Home key) and go to **Add-ons > XPAR**. The presenter starts
   automatically with the game.

| Setting | What it does |
|---|---|
| **Enable reprojection (N Hz)** | Turns reprojection on or off. N is your display's measured refresh rate. |
| **Show original (A/B)** | Shows the game's own frames, for comparison. |
| **Auto latency** | How far behind the newest game frame the displayed camera sits. **Auto** (default) uses 1/2 frame, or 1/4 frame in games without HUD layers, where a shorter warp keeps any undetected HUD steadier. You can also pick **1 frame** (smoothest), **1/2** or **1/4** (less latency, cleaner screen edges), or turn it **Off** to set the latency by hand. |
| **Present lead (ms)** | How early each frame is rendered before the display refresh (default 6). Raise it if the output drops below your refresh rate. |
| **Keep HUD still** | Warps the scene but not the HUD, using the HUD layers the game provides (Expedition 33). |
| **Keep still** | What is not warped. **HUD + character/weapon** (default), **HUD only**, **Character/weapon only** or **Off**. *HUD*: in games without HUD layers, FrameWarp finds the HUD itself (see *Find the HUD*). *Character/weapon*: what moves with the camera, such as a first-person weapon or a third-person character, found from the game's motion vectors in every game; it starts after a few seconds of turning the camera. |
| **Keep near-camera motion still** | On by default, remembered per game. Part of *Character/weapon*: things close to the camera that move against it (a weapon mid-animation, hands) are kept still. Turn it off if the floor near the camera is kept still while strafing; a weapon that stays put on screen is still kept still. |
| **Find the HUD** | Remembered per game. **Upscaler output + camera motion check** (default): compares the final picture with the upscaler's (DLSS or FSR) own output, which has no HUD, and leaves out what is seen moving with the world when the camera turns (bright lights, neon or effects drawn after upscaling). **From the upscaler output**: the comparison alone. **Learned from camera motion**: finds the HUD from what stays put on screen while the camera moves; it needs a few seconds of camera movement, and is used whenever there is no upscaler output. |
| **Fill behind the HUD from the upscaler output** | Off by default, remembered per game; XPAR engine, with the HUD found from the upscaler output. When the camera turns, the scenery the HUD covered is taken from the upscaler's output (colour-matched to the game's picture) instead of being smeared from the surroundings, so the HUD leaves no trail. Effects the game adds after upscaling, such as bloom or film grain, may be missing in those spots. |
| **Show the mask** | Tints what is kept still (magenta), and with *Learned from camera motion* the HUD still being learned (green), to check the options above. |
| **Warp engine** | **XPAR** (default): FrameWarp's own engine, any GPU. **NVIDIA Latewarp**: selectable when `nvngx_latewarp.dll` is installed and the GPU is NVIDIA's. |
| **Presenter GPU priority** | Keep **Realtime**. Lower priorities cannot hold the refresh rate while the game loads the GPU. |
| **Record detailed diagnostics** | Off by default, remembered per game. Writes frame-by-frame recordings to the logs folder, for troubleshooting. |
| **Reset camera model** | Re-learns how the game's camera responds to your mouse. This also happens automatically within seconds of play. |
| **Start presenter** | Restarts the presenter if it was closed. |

The remaining options (extrapolation, orbit distance, manual mouse gain, debug strip, Streamline
diagnostics) are for fine-tuning and troubleshooting.

## Known limitations

* Frame rates that swing a lot degrade the result, because the camera model assumes a steady frame
  cadence.
* During fast turns, the screen edges show fill for areas the game never rendered.
* Without the game's own camera data (DLSS called directly, most FSR games), the camera is estimated
  from motion vectors, which is a little less exact than the camera data Streamline games provide.
* Upscalers built into a game's executable instead of a DLL (Cyberpunk 2077's FSR 2.1) can't be used.
* Very fast motion goes beyond what re-projecting a single frame can hide. In Expedition 33's
  overworld, very fast camera turns can make the picture shake slightly.
* The Expedition 33 support targets the current Steam build. It switches itself off safely if a game
  update changes the relevant code.

## Troubleshooting

If the XPAR panel says frame generation is on, turn frame generation off in the game's graphics
settings; FrameWarp resumes by itself.

Logs are written to `FrameWarp\logs\`, next to the game's ReShade. The previous run is kept in
`logs\previous\`. When something looks wrong in game, press **Ctrl+Shift+M** to mark that moment in
the logs. For a problem report, tick **Record detailed diagnostics** in the XPAR panel, reproduce the
problem, and include both log folders.

## Building from source

Requirements: Visual Studio 2022 or later (C++ x64 workload), the Windows SDK and CMake 3.24 or later.
The NVIDIA NGX SDK is fetched from NVIDIA's DLSS repository.

```
git clone --depth 1 https://github.com/NVIDIA/DLSS third_party/DLSS
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Optional: put `nvngx_latewarp.dll` in `third_party\latewarp\` (or pass `-DLATEWARP_DLL=<path>`), and
the build copies it next to the presenter. `tools\package.ps1` builds, tests and writes the release
zip to `dist\`. [docs/architecture.md](docs/architecture.md) explains how the add-on and the presenter
work.

## License

FrameWarp is licensed under the GNU General Public License v3.0; see [LICENSE](LICENSE). Third-party
components keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

NVIDIA, RTX, DLSS and Reflex are trademarks of NVIDIA Corporation. AMD, FidelityFX and FSR are
trademarks of Advanced Micro Devices, Inc. FrameWarp is an independent project and is not affiliated
with or endorsed by NVIDIA, AMD, the ReShade project, or any game developer.
