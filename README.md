# FrameWarp

**Asynchronous camera reprojection for PC games.** FrameWarp shows camera motion at your display's
refresh rate, whatever frame rate the game renders at. For example, a game locked to 30 fps turns and
moves at 120 Hz on a 120 Hz display. In ReShade it appears as **XPAR** (xPoiler's Asynchronous
Reprojection).

On every display refresh, FrameWarp takes the game's newest frame and re-projects it to where the
camera is *now*. It runs as a ReShade add-on and reads what it needs from the game's upscaler, or,
in games without one, from ReShade and the picture itself:

* **DLSS**: camera, depth and HUD data from the game's NVIDIA Streamline integration, or depth and
  motion vectors from DLSS called directly, without Streamline (DirectX 12 and Vulkan games).
* **AMD FSR 3** (3.0 and 3.1): depth and motion vectors from FSR. FSR 4 uses the same interface as
  FSR 3.1 and should work too. **FSR 2** should work in games that ship it as a DLL (untested).
* **No DLSS or FSR** (DirectX 12 and DirectX 11, experimental): the depth from ReShade's depth buffer,
  which FrameWarp picks and sets up by itself, and the motion worked out from the picture itself. This
  is also used when a game's DLSS and FSR are switched off, and FrameWarp switches back by itself when
  they are switched on again. In a game without a usable depth buffer, camera turns are still
  reprojected (walking and strafing move at the game's frame rate).

**HDR works with it** (HDR10 and scRGB), also when HDR is switched on or off in the game's menus.

**Frame generation works with it**: with the game's DLSS Frame Generation or FSR 3.1 frame generation on,
FrameWarp shows the generated images too, each moved to where the camera is now, so moving things get
frame generation's smoothness while the camera keeps FrameWarp's low latency.

Without the game's own camera data, FrameWarp works out the camera's movement from the motion
vectors. It does the same when a game's camera data turns out not to match its motion vectors, which
it checks by itself while the camera moves. It then draws the result in a click-through overlay above the game, with its own warp engine
(or, optionally, NVIDIA's Latewarp).

## Tested games

| Game | Notes |
|---|---|
| *Clair Obscur: Expedition 33* (Steam, DLSS) | Add the launch options `-slforcetagging -slviewextension` (see Install). The game provides HUD layers, so the HUD stays still; FrameWarp also holds the character while the camera turns around it. |
| *Resident Evil Requiem* (Steam, DLSS or FSR) | **Requires [REFramework](https://github.com/praydog/REFramework)**: without it the game's DRM crashes with ReShade. The game doesn't provide HUD layers, so FrameWarp detects the HUD itself. With FSR, turn the camera for a second or two after loading while FrameWarp checks the field of view. DLSS Frame Generation (2x to 4x) and FSR 3.1 frame generation work with FrameWarp; with FSR frame generation the game hands over a HUD-less picture and the HUD is found exactly from it. |
| *Cyberpunk 2077* (Steam, DLSS or FSR 3) | No launch options needed. The game doesn't provide HUD layers: FrameWarp detects the HUD and V's weapon itself. Semi-transparent HUD panels may still move slightly. The game's FSR 2.1 option is built into the game itself and can't be used. DLSS Frame Generation and FSR 3 frame generation work with FrameWarp; with frame generation on, the game provides a HUD-less picture and FrameWarp finds the HUD exactly from it. |
| *DOOM Eternal* (Steam, Vulkan, DLSS) | Install ReShade for Vulkan (see Install). The game calls DLSS without Streamline, so FrameWarp works out the camera from DLSS's motion vectors; turn the camera for a few seconds after loading. FrameWarp finds the HUD and holds the weapon itself. If the ReShade menu hides behind the game, turn off *Present From Compute* in the game's advanced video settings. |
| *Assassin's Creed Black Flag Resynced* (DLSS or FSR) | No launch options needed. The game's camera data doesn't match its picture: FrameWarp notices within the first seconds of camera movement on the first launch (the picture may warp wrongly until then), switches to working out the camera from the motion vectors, and remembers that for the game. Turn the camera for a few seconds after loading while it calibrates; FrameWarp then also measures the point the camera circles around (*Orbit pivot*). With the orbit measured, the ground at the character's feet moves with it (*Let the ground around the character move with the orbit*, on by default). |
| *S.T.A.L.K.E.R. 2* (Steam, DLSS) | No launch options needed. With DLSS Frame Generation on, the game provides its camera and a HUD-less picture; without it, the game calls DLSS directly and FrameWarp works out the camera from DLSS's motion vectors (turn the camera for a minute after loading while it calibrates). The game's NVIDIA Reflex works with FrameWarp. |
| *Red Dead Redemption 2* (DirectX 12, DLSS) | Choose DirectX 12 in the game's settings. The game calls DLSS without Streamline, so FrameWarp works out the camera from DLSS's motion vectors; turn the camera for a few seconds after loading while it calibrates. The third-person camera circles the character: keep *Let the ground around the character move with the orbit* on (the default). The game crashes when the ReShade menu opens unless ReShade's Generic Depth add-on is off (see Troubleshooting). Installed under Program Files, run `install.bat` as administrator. |
| *Returnal* (Steam, DLSS) | No launch options needed. The game calls DLSS without Streamline, so FrameWarp works out the camera from DLSS's motion vectors; turn the camera for a few seconds after loading while it calibrates. |
| *Resident Evil 2* (Steam, DirectX 11, no DLSS or FSR) | Choose DirectX 11 in the game's settings. FrameWarp finds the depth buffer and works out the motion and the camera from the picture; turn the camera for a few seconds after loading while it calibrates. Very fast turns and dark scenes are the hardest. |
| *Assassin's Creed Black Flag Resynced* (upscaler off) | The same as above, in DirectX 12. |
| *Metro 2033 Redux* (Steam, DirectX 11, no DLSS or FSR) | The game only runs in exclusive fullscreen: tick **Force a borderless window** in the XPAR panel and restart the game. FrameWarp finds the depth buffer and works out the motion from the picture. |
| *Cyber Hook* (Steam, DirectX 11, no DLSS or FSR) | The game has no usable depth buffer: FrameWarp reprojects camera turns only (walking and strafing move at the game's frame rate). |

Other games may work but are untested.

## Requirements

* A DirectX 12 game with **DLSS or FSR** (see above) turned on, a Vulkan game with **DLSS** (tested with
  DOOM Eternal), or a DirectX 12 or 11 game **without** them (experimental). Without DLSS or FSR,
  ReShade has to be able to see the game's depth buffer, as for any depth effect (online games
  often block it). DirectX 11 games need Windows 10 version 1703 or later.
* About 1 GB of free video memory at 4K for the presenter, a little more with frame generation. The XPAR
  panel warns when video memory is nearly full.
* A GPU for DirectX 12. FrameWarp is tested on NVIDIA GPUs; its own warp engine uses plain DirectX 12,
  so AMD and Intel GPUs should work too, but they are untested (reports welcome).
* Windows 11 (tested; Windows 10 may work) with **Hardware-accelerated GPU scheduling** turned on (Settings > System >
  Display > Graphics, then restart). Without it the output may not reach the refresh rate; the XPAR
  panel warns you if it is off.
* [ReShade](https://reshade.me) 6.x **with full add-on support**, installed for the game.
* Frame generation: **DLSS Frame Generation and FSR 3.1 frame generation work with FrameWarp** (its own
  warp engine): each generated image is moved to the current camera like any frame. Other kinds (FSR 3.0,
  XeSS) can't be combined with it: while one is on, FrameWarp pauses and says so in its panel.
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
   The installer copies FrameWarp next to the game's ReShade, and `XPAR.fx` (for games without DLSS
   or FSR) into the game's ReShade shader folder. For Unreal Engine games, it also
   enables Streamline resource tagging in the game's `Engine.ini`, keeping a backup of the file.
   **Vulkan games** (DOOM Eternal): install ReShade for the game choosing **Vulkan** as the rendering
   API. ReShade then runs as a system-wide Vulkan layer and puts only a `ReShade.ini` in the game's
   folder; the installer finds the game by that file.
4. **Unreal Engine games with DLSS** (including Expedition 33): add these launch options (Steam >
   game > Properties > Launch Options):
   ```
   -slforcetagging -slviewextension
   ```

After an update, the settings XPAR remembers per game go back to the defaults on the first launch, so
the new version's improved defaults apply right away (the learned camera model is kept).

To uninstall, double-click `uninstall.bat` and choose the game. It reverts everything the installer
changed.

Advanced: both files also accept a game name or folder directly (`install.bat "Expedition 33"`), and
`install.ps1` accepts `-Game`, `-GameDir`, `-Latewarp <dll>`, `-EngineIni <path>`, `-NoCvar` and
`-Uninstall`.

## Use

1. Run the game in **borderless** mode, with DLSS or FSR on if the game has them. Frame generation can be
   on (DLSS Frame Generation or FSR 3.1 frame generation) or off. For a game that only runs in exclusive
   fullscreen, tick **Force a borderless window** and restart the game.
2. Open the ReShade overlay (Home key) and go to **Add-ons > XPAR**. The presenter starts
   automatically with the game.

The panel has four sections: **Camera motion**, **Latency**, **Warp** and **Debug**; rest the mouse on an option
to see what it does. The settings are remembered per game (in the game's `ReShade.ini`), except **Enable
reprojection** and the comparison and debug switches (Show original, Debug strip, Invert warp, Show the mask).
Updating FrameWarp resets them to the defaults (except *Record detailed diagnostics*), and **Reset all settings
to defaults** at the top of the panel does the same at any time.

| Setting | What it does |
|---|---|
| **Enable reprojection (N Hz)** | Turns reprojection on or off. N is the refresh rate of the display the game is on, from its current display mode. |
| **Show original (A/B)** | Shows the game's own frames, for comparison. |
| **Auto latency** | How far behind the newest game frame the displayed camera sits. **Lowest latency without edge fill** (default): every refresh, the camera as late as it can be without revealing anything the game hasn't rendered - the camera as of now (your mouse up to this moment) while you aim or turn slowly, falling back as far as one game frame during fast turns, where what the warp reveals comes from the game's last two frames (keep *Background memory* on). **Camera as of now (edge-limited)**: always as close to now as the uncovered screen edge allows (*Extra uncovered edge*: how much more of the edge than *Auto* may be uncovered; *No edge limit*: always now). **Auto**: 1/2 game frame behind, 1/4 in games without HUD layers, a whole frame in games without DLSS or FSR. **1 frame**, **1/2**, **1/4**, or **Off** to set the latency by hand. |
| **Orbit pivot** | **Measured** (default): in games that give DLSS or FSR their motion vectors but no camera, turns are predicted around the point the camera circles (a third-person character), as FrameWarp measures it, while the turns agree on one point; first-person cameras, games that send their camera and games without DLSS or FSR turn on the spot. **Off**, or **Manual** with a distance. |
| **Let the ground around the character move with the orbit** | On by default, remembered per game; only shown while a measured orbit is in use. *Keep still what the camera turns around* no longer holds the ground at the character's feet. |
| **Automatic present lead** | On by default. FrameWarp renders each refresh as late as its warp still makes it (measured), for the freshest camera, and backs off when a frame misses. Untick it to set **Present lead (ms)** by hand: how early each frame is rendered before the display refresh. With the XPAR engine and a lead above 0, new game frames are taken in on their own GPU queue, so their processing never delays a refresh (about 50–100 MB more video memory at 4K). |
| **Prevent GPU queueing** | On by default, remembered per game. Not a frame rate limit you set: FrameWarp finds the rate the game's GPU sustains and holds the game just below it, so no frame waits in the GPU's queue and the game reads your input later (what NVIDIA Reflex does). It stands aside by itself while the game's own Reflex is on and during frame generation. DirectX 12 and 11. |
| **Keep HUD still** | Warps the scene but not the HUD, using the HUD layers the game provides (Expedition 33). |
| **Keep still** | What is not warped. **HUD + character/weapon** (default), **HUD only**, **Character/weapon only** or **Off**. *HUD*: in games without HUD layers, FrameWarp finds the HUD itself (see *Find the HUD*). *Character/weapon*: what moves with the camera, such as a first-person weapon or a third-person character, found from the game's motion vectors in every game; it starts after a few seconds of turning the camera. |
| **Keep near-camera motion still** | On by default, remembered per game. Part of *Character/weapon*: things close to the camera that move against it (a weapon mid-animation, hands) are kept still. Turn it off if the floor near the camera is kept still while strafing; a weapon that stays put on screen is still kept still. |
| **Keep still what the camera turns around (third-person)** | On by default, remembered per game. Part of *Character/weapon*: while the camera turns, anything that moves less than a fifth of what the turn alone would move it on screen is kept still. Over-the-shoulder cameras circle the character, so it barely moves on screen while the room swings around it; the warp turns the view around the camera and would otherwise swing the character away until the next game frame snaps it back. |
| **Hold sight reticles** | On by default, remembered per game, XPAR engine. Part of *Character/weapon*: red dot and holographic sights whose reticle the game draws without depth of its own (it would skip with the scenery behind the glass while turning). Inside the window the held weapon encloses, what is on the glass (reticle, glow) is learned as a layer that follows the reticle within a few frames of turning, and only the scenery under it is warped. Sights drawn with depth are held anyway. |
| **Stretch around character/weapon** | 0-32 render pixels, 1 by default (0 turns it off), remembered per game, XPAR engine. Where the scenery slides away from a kept-still character or weapon, the warp uncovers a gap and fills it from the nearest scenery pixel, which draws streaks and smears the outline its motion vectors miss (anti-aliasing, the upscaler's softening). With a width set, the scenery that close follows the warp less and less towards the character or weapon: the gap is covered by slightly stretched scenery, and the outline stays with it. |
| **Move objects at the display rate even without frame generation** | Off by default (experimental), remembered per game, XPAR engine, games with DLSS or FSR. The game's own DLSS or FSR 3.1 frame generation is used by itself when it is on; this is XPAR's own frame generation for what moves on its own, for when it is not: cars, people and anything else moving are shown one game frame late and move at the display rate, between two real frames, instead of at the game's frame rate; the camera is not delayed. What they uncover shows the previous frame. The character/weapon kept still moves at the display rate too (one frame late, still not warped with the camera). Shadows stay where the game drew them. |
| **Background memory for uncovered areas** | On by default, remembered per game, XPAR engine. The scenery last seen around the kept-still character or weapon is remembered (half resolution, up to a second) and shown where the warp uncovers it while turning, instead of stretched edge pixels. About 80 MB of video memory at 4K and about 0.1 ms of GPU time per game frame, on the intake queue. |
| **Uncovered screen edges** | XPAR engine, remembered per game. **Soft** (default): what a fast turn uncovers past the edge of the game's frame, where background memory has nothing, is a blur of the scenery along the edge, wider further out. **Extended edge pixels**: the edge repeated (sharper close to it, streaks further out). |
| **Find the HUD** | Remembered per game. **Upscaler output + camera motion check** (default): compares the final picture with the upscaler's (DLSS or FSR) own output, which has no HUD, and leaves out what is seen moving with the world when the camera turns (bright lights, neon or effects drawn after upscaling). **From the upscaler output**: the comparison alone. **Learned from camera motion**: finds the HUD from what stays put on screen while the camera moves; it needs a few seconds of camera movement, and is used whenever there is no upscaler output. |
| **Fill behind the HUD from the upscaler output** | On by default, remembered per game; XPAR engine, with the HUD found from the upscaler output. When the camera turns, the scenery the HUD covered is taken from the upscaler's output (colour-matched to the game's picture) instead of being smeared from the surroundings, so the HUD leaves no trail. Effects the game adds after upscaling, such as bloom or film grain, may be missing in those spots. |
| **Show the mask** | Tints what is kept still (magenta), to check the options above. With *Learned from camera motion* and NVIDIA Latewarp (or *Present lead* 0), the HUD still being learned shows green too. |
| **Warp engine** | **XPAR** (default): FrameWarp's own engine, any GPU. **NVIDIA Latewarp**: selectable when `nvngx_latewarp.dll` is installed and the GPU is NVIDIA's. |
| **Presenter GPU priority** | Keep **Realtime**. Lower priorities cannot hold the refresh rate while the game loads the GPU. |
| **Controller right stick drives rotation** | On by default, remembered per game. The right stick of a controller (XInput: Xbox pads, or others through Steam Input) moves the reprojected camera between the game's frames, as the mouse does; how fast the stick turns the camera and the game's response curve are learned from the game's own camera (shown under the camera model). |
| **Force a borderless window** | Off by default, remembered per game, applies when the game is restarted. For games that only run in exclusive fullscreen, where nothing can be shown over the game: the game stays in a window that covers the screen and is told it is fullscreen. |
| **Record detailed diagnostics** | Off by default, remembered per game. Writes frame-by-frame recordings to the logs folder, for troubleshooting, and turns on the **Ctrl+Shift+M** (mark a moment in the log) and **Ctrl+Shift+D** (capture the current frames; the game pauses for about a second) keys. |
| **Reset camera model** | Re-learns how the game's camera responds to your mouse. This also happens automatically within seconds of play. |
| **Start presenter** | Restarts the presenter if it was closed. |

The remaining options (extrapolation, manual mouse gain, debug strip, Streamline diagnostics) are for
fine-tuning and troubleshooting.

## Known limitations

* Frame rates that swing a lot degrade the result, because the camera model assumes a steady frame
  cadence.
* With frame generation, moving things are shown one game frame late, as frame generation always does
  (the camera is not delayed). FSR 3.0 frame generation (the older FidelityFX SDK) and XeSS frame
  generation aren't used: FrameWarp pauses while they are on. Beyond 4x, three generated images per game
  frame are shown, spread evenly. Frame generation and FrameWarp together need GPU headroom: at 3x and 4x
  the output may not reach the refresh rate on mid-range GPUs.
* During very fast turns, the screen edges can briefly show fill for areas the game never rendered (with the
  default latency only at the start of a flick, as a soft blur).
* VRR (G-SYNC, FreeSync): while FrameWarp shows its frames, the screen runs at its maximum refresh rate.
  FrameWarp's window is always composited by Windows, so VRR can't follow it, also with G-SYNC for windowed
  mode. FrameWarp paces itself to the screen's refreshes instead, so no frames queue at the maximum.
* Without the game's own camera data (DLSS called directly, most FSR games, games whose camera data
  doesn't match their motion vectors), the camera is estimated from motion vectors, which is a little
  less exact than the camera data Streamline games provide.
* Upscalers built into a game's executable instead of a DLL (Cyberpunk 2077's FSR 2.1) can't be used.
* Without DLSS or FSR, the motion is worked out from the picture: very fast turns, dark or blurry
  scenes and flat surfaces are where it is least sure, and the result is less exact than with an
  upscaler's motion vectors. When the depth reads empty, FrameWarp switches on ReShade's depth copy and
  asks for one restart of the game. In a game whose depth buffer ReShade can't see, or that has none
  belonging to the picture, only camera turns are reprojected. DirectX 9, DirectX 10 and OpenGL games aren't supported, and Vulkan games only with DLSS.
* Very fast motion goes beyond what re-projecting a single frame can hide. In Expedition 33's
  overworld, very fast camera turns can make the picture shake slightly.
* The Expedition 33 support targets the current Steam build. It switches itself off safely if a game
  update changes the relevant code.

## Troubleshooting

If the XPAR panel says frame generation is on and XPAR is paused, the game uses a kind of frame
generation XPAR can't take (or NVIDIA Latewarp is the warp engine): turn frame generation off in the
game's graphics settings, or switch to DLSS or FSR 3.1 frame generation; FrameWarp resumes by itself.

If the XPAR panel warns that video memory is nearly full, the game and FrameWarp don't both fit and both
stutter: lower the game's texture quality or resolution, or turn off frame generation. scRGB HDR doubles the
memory FrameWarp needs for the game's picture (choose HDR10/PQ if the game offers both), and with HDR on in
Windows the desktop itself takes more video memory on the graphics card that drives the screen.

If the output is stuck at a low frame rate on an NVIDIA GPU, check **Background Application Max Frame
Rate** in the NVIDIA Control Panel (Manage 3D settings). FrameWarp's window is never the focused one, as
the game keeps your keyboard and mouse, so the driver treats it as a background application and caps
it; the game it covers can be capped too. Turn the setting off, for all programs or for
`FrameWarpPresenter.exe` and the game. The XPAR panel warns when its output is held back by a frame
rate cap like this.

If the game crashes when you open the ReShade menu (Red Dead Redemption 2 does, also without FrameWarp),
turn off ReShade's **Generic Depth** add-on: add `DisabledAddons=Generic Depth` under `[ADDON]` in the game's
`ReShade.ini` (it then shows unticked in ReShade's Add-ons tab). Games with DLSS or FSR don't need it: FrameWarp takes the depth from the upscaler.

Logs are written to `FrameWarp\logs\`, next to the game's ReShade. The previous run is kept in
`logs\previous\`. For a problem report, tick **Record detailed diagnostics** in the XPAR panel,
reproduce the problem (pressing **Ctrl+Shift+M** marks the moment it looks wrong in the logs), and
include both log folders. Without DLSS or FSR, `ReShade.log` also shows which depth buffer FrameWarp
picked and why.

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
