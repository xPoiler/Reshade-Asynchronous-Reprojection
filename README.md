# FrameWarp

**Asynchronous camera reprojection for PC games.** FrameWarp shows camera motion at your display's
refresh rate, whatever frame rate the game renders at. For example, a game locked to 30 fps turns and
moves at 120 Hz on a 120 Hz display.

On every display refresh, FrameWarp takes the game's newest frame and re-projects it to where the
camera is *now*, using NVIDIA's Reflex 2 Frame Warp engine. It runs as a ReShade add-on and reads the
camera, depth and HUD data from the game's own NVIDIA Streamline (DLSS/Reflex) integration. It then
draws the result in a click-through overlay above the game.

## Tested games

| Game | Notes |
|---|---|
| *Clair Obscur: Expedition 33* (Steam) | Add the launch options `-slforcetagging -slviewextension` (see Install). The HUD stays still while the scene is warped. |
| *Resident Evil Requiem* (Steam) | **Requires [REFramework](https://github.com/praydog/REFramework)**: without it the game's DRM crashes with ReShade. The game doesn't provide HUD layers, so FrameWarp detects the HUD itself (see *Detect HUD and first-person weapon*). |
| *Cyberpunk 2077* (Steam) | No launch options needed. The game doesn't provide HUD layers: FrameWarp detects the HUD and V's weapon itself. Semi-transparent HUD panels may still move slightly. |

Other games that use NVIDIA Streamline for DLSS may work but are untested.

## Requirements

* An NVIDIA RTX GPU rendering the game. FrameWarp takes its data from the game's DLSS and Reflex, so the
  GPU must support both (DLSS needs an RTX card), and NVIDIA's Frame Warp engine runs on RTX GPUs too.
* Windows 11 (tested; Windows 10 may work) with **Hardware-accelerated GPU scheduling** turned on (Settings > System >
  Display > Graphics, then restart). Without it the output may not reach the refresh rate; the FrameWarp
  panel warns you if it is off.
* [ReShade](https://reshade.me) 6.x **with full add-on support**, installed for the game.
* DLSS and Reflex enabled in the game.
* **`nvngx_latewarp.dll`**, NVIDIA's Reflex 2 Frame Warp (Latewarp) DLL. It is NVIDIA's file and
  cannot be included. It comes bundled with software that uses Reflex 2 Frame Warp, such as
  community-made Reflex 2 demos, so search for it by file name. FrameWarp was tested with version
  **310.2.0.0**: the file description is "NVIDIA Latewarp" and the SHA256 is
  `81e63b4a9dcfb99b2b71c3ddfd3ea4a0e9780b778bd47e9a6f7f010a7a6c0f20`. To check the version,
  right-click the file and open Properties > Details.
* The [Microsoft Visual C++ Redistributable 2015–2022 (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

## Install

1. Download the latest `FrameWarp-<version>.zip` from
   [Releases](https://github.com/xPoiler/Reshade-Asynchronous-Reprojection/releases) and
   extract it.
2. Put `nvngx_latewarp.dll` next to `install.bat`.
3. Close the game and double-click `install.bat`. It lists your Steam games that have ReShade; type
   the number of the game and press Enter. For a non-Steam game, choose **P** and type its folder.
   The installer copies FrameWarp next to the game's ReShade. For Unreal Engine games, it also
   enables Streamline resource tagging in the game's `Engine.ini`, keeping a backup of the file.
4. **Unreal Engine games** (including Expedition 33): add these launch options (Steam > game >
   Properties > Launch Options):
   ```
   -slforcetagging -slviewextension
   ```

To uninstall, double-click `uninstall.bat` and choose the game. It reverts everything the installer
changed.

Advanced: both files also accept a game name or folder directly (`install.bat "Expedition 33"`), and
`install.ps1` accepts `-Game`, `-GameDir`, `-Latewarp <dll>`, `-EngineIni <path>`, `-NoCvar` and
`-Uninstall`.

## Use

1. Run the game in **borderless** mode, with DLSS and Reflex on.
2. Open the ReShade overlay (Home key) and go to **Add-ons > FrameWarp**. The presenter starts
   automatically with the game.

| Setting | What it does |
|---|---|
| **Enable reprojection (N Hz)** | Turns FrameWarp on or off. N is your display's measured refresh rate. |
| **Show original (A/B)** | Shows the game's own frames, for comparison. |
| **Auto latency** | How far behind the newest game frame the displayed camera sits. **Auto** (default) uses 1/2 frame, or 1/4 frame in games without HUD layers, where a shorter warp keeps any undetected HUD steadier. You can also pick **1 frame** (smoothest), **1/2** or **1/4** (less latency, cleaner screen edges), or turn it **Off** to set the latency by hand. |
| **Present lead (ms)** | How early each frame is rendered before the display refresh (default 6). Raise it if the output drops below your refresh rate. |
| **Keep HUD still** | Warps the scene but not the HUD, using the HUD layers the game provides (Expedition 33). |
| **Detect HUD and first-person weapon** | For games without HUD layers (on by default): finds the HUD from what stays put on screen while the camera moves, and a first-person weapon from its motion, and keeps both unwarped. It needs a few seconds of camera movement to learn. |
| **Presenter GPU priority** | Keep **Realtime**. Lower priorities cannot hold the refresh rate while the game loads the GPU. |
| **Reset camera model** | Re-learns how the game's camera responds to your mouse. This also happens automatically within seconds of play. |
| **Start presenter** | Restarts the presenter if it was closed. |

The remaining options (extrapolation, orbit distance, manual mouse gain, debug strip, Streamline
diagnostics) are for fine-tuning and troubleshooting.

## Known limitations

* Frame rates that swing a lot degrade the result, because the camera model assumes a steady frame
  cadence.
* During fast turns, the screen edges show fill for areas the game never rendered.
* Very fast motion goes beyond what re-projecting a single frame can hide.
* The Expedition 33 support targets the current Steam build. It switches itself off safely if a game
  update changes the relevant code.

## Troubleshooting

Logs are written to `FrameWarp\logs\`, next to the game's ReShade. The previous run is kept in
`logs\previous\`. When something looks wrong in game, press **Ctrl+Shift+M** to mark that moment in
the logs. Include both log folders when you report a problem.

## Building from source

Requirements: Visual Studio 2022 or later (C++ x64 workload), the Windows SDK and CMake 3.24 or later.
The NVIDIA NGX SDK is fetched from NVIDIA's DLSS repository.

```
git clone --depth 1 https://github.com/NVIDIA/DLSS third_party/DLSS
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Put `nvngx_latewarp.dll` in `third_party\latewarp\` (or pass `-DLATEWARP_DLL=<path>`), and the build
copies it next to the presenter. `tools\package.ps1` builds, tests and writes the release zip to
`dist\`. [docs/architecture.md](docs/architecture.md) explains how the add-on and the presenter work.

## License

FrameWarp is licensed under the GNU General Public License v3.0; see [LICENSE](LICENSE). Third-party
components keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

NVIDIA, RTX, DLSS and Reflex are trademarks of NVIDIA Corporation. FrameWarp is an independent project
and is not affiliated with or endorsed by NVIDIA, the ReShade project, or any game developer.
