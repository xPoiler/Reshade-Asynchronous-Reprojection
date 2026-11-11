DISABLE GLOBAL FRAME LIMITERS, INCLUDING BACKGROUND FRAME LIMITERS (CHECK NVIDIA CONTROL PANEL) OR THE REPROJECTION WON'T WORK.

 
# FrameWarp

This README describes the current source checkout. Documentation bundled with a release describes that release.

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
* **No DLSS or FSR** (DirectX 12, DirectX 11 and Vulkan): the depth from ReShade's depth buffer, which
  FrameWarp picks and sets up by itself, the game's own motion vectors, which it finds among the game's
  render targets (DirectX 12 and 11), and otherwise the motion worked out from the picture itself. This
  is also used when a game's DLSS and FSR are switched off, and FrameWarp switches back by itself when
  they are switched on again. In a game without a usable depth buffer, camera turns are still
  reprojected (walking and strafing move at the game's frame rate).

**HDR works with it** (HDR10 and scRGB), also when HDR is switched on or off in the game's menus.

**Frame generation works with it**: with the game's DLSS Frame Generation or FSR 3.1 frame generation on,
FrameWarp shows the generated images too, each moved to where the camera is now, so moving things get
frame generation's smoothness while the camera keeps FrameWarp's low latency.

**The HUD and your character or weapon stay right**: the HUD is taken from the step where the game draws it,
which FrameWarp finds by itself in most tested games (exact from the first frame), and characters, weapons and
car interiors are moved by how much of the camera's turn they really follow, learned as you play.

When the game doesn't hand its camera over, FrameWarp **finds the game's own camera** in the data the game
gives the GPU, checking each candidate against the motion it sees, and follows it (DirectX 12, DirectX 11
and Vulkan). Where none is found, it works out the camera's movement from the motion vectors or the
picture. It does the same when a game's camera data turns out not to match its motion vectors, which it
checks by itself while the camera moves. It then draws the result in a click-through overlay above the
game, with its own warp engine (or, optionally, NVIDIA's Latewarp).

## Tested games

| Game | Notes |
|---|---|
| *Clair Obscur: Expedition 33* (Steam, DLSS) | Add the launch options `-slforcetagging -slviewextension` (see Install). The game provides HUD layers, so the HUD stays still; FrameWarp also holds the character while the camera turns around it. |
| *Resident Evil Requiem* (Steam, DLSS or FSR) | **Requires [REFramework](https://github.com/praydog/REFramework)**: without it the game's DRM crashes with ReShade. FrameWarp takes the HUD from the game's own HUD layer, which it finds by itself, and checks it (frames where the game hides its HUD show none). With FSR, turn the camera for a second or two after loading while FrameWarp checks the field of view. DLSS Frame Generation (2x to 4x) and FSR 3.1 frame generation work with FrameWarp; with FSR frame generation the game hands over a HUD-less picture and the HUD is found exactly from it. |
| *Cyberpunk 2077* (Steam, DLSS, FSR 3, XeSS or none) | No launch options needed. FrameWarp takes the HUD from the game's own HUD layer, which it finds by itself. V's weapon is held, and a car's interior stays put while you steer and follows when you look around. DLSS Frame Generation and FSR 3 frame generation work with FrameWarp; with frame generation on, the game provides a HUD-less picture and FrameWarp finds the HUD exactly from it. With XeSS or no upscaler (the game's FSR 2.1 is built into the game and works the same way), FrameWarp finds the game's own camera and motion vectors by itself. |
| *DOOM Eternal* (Steam, Vulkan, DLSS or none) | Install ReShade for Vulkan (see Install). The game calls DLSS without Streamline and gives it no camera: FrameWarp finds the game's own camera (turn the camera for a few seconds after loading). FrameWarp takes the HUD from the step where the game blends it on, and holds the weapon itself. If the ReShade menu hides behind the game, turn off *Present From Compute* in the game's advanced video settings. |
| *Armored Core VI: Fires of Rubicon* (Steam, DirectX 12, no DLSS or FSR) | FrameWarp finds the depth buffer and the game's own camera and motion vectors by itself. The HUD is taken from the game's own HUD layer with its colour, so its ring and crosshair stay put while the scenery under them moves. |
| *Enshrouded* (Steam, Vulkan, DLSS, FSR 3 or none) | Install ReShade for Vulkan (see Install). The game gives DLSS no camera: FrameWarp finds the game's own camera (turn the camera for a few seconds after loading). With the game's FSR 3 (built into the game) or no upscaler, FrameWarp takes the depth from ReShade and works out the motion from the picture. The HUD is found from the HUD-less picture the game provides with DLSS. |
| *The Blood of Dawnwalker* (DirectX 12, DLSS) | The game hands over a HUD-less picture without a HUD layer: FrameWarp takes the HUD from the two pictures, so a faint backdrop (the dark panel behind the quest text) stays put while the scenery under it moves. |
| *Assassin's Creed Black Flag Resynced* (DLSS or FSR) | No launch options needed. The game's camera data doesn't match its picture: FrameWarp notices within the first seconds of camera movement on the first launch (the picture may warp wrongly until then), switches to working out the camera from the motion vectors, and remembers that for the game. Turn the camera for a few seconds after loading while it calibrates. |
| *S.T.A.L.K.E.R. 2* (Steam, DLSS) | No launch options needed. With DLSS Frame Generation on, the game provides its camera and a HUD-less picture; without it, the game calls DLSS directly and FrameWarp works out the camera from DLSS's motion vectors (turn the camera for a minute after loading while it calibrates). The game's NVIDIA Reflex works with FrameWarp. |
| *Red Dead Redemption 2* (DirectX 12, DLSS) | Choose DirectX 12 in the game's settings. The game calls DLSS without Streamline, so FrameWarp works out the camera from DLSS's motion vectors; turn the camera for a few seconds after loading while it calibrates. The game crashes when the ReShade menu opens unless ReShade's Generic Depth add-on is off (see Troubleshooting). Installed under Program Files, run `install.bat` as administrator. |
| *Returnal* (Steam, DLSS) | No launch options needed. The game calls DLSS without Streamline, so FrameWarp works out the camera from DLSS's motion vectors; turn the camera for a few seconds after loading while it calibrates. |
| *Resident Evil 2* (Steam, DirectX 11, no DLSS or FSR) | FrameWarp finds the depth buffer and the game's own camera and motion vectors by itself. The HUD is taken from a copy of the picture made just before the game draws it. |
| *Resident Evil 4 Remake* (Steam, DirectX 12) | The game's FSR 2 is built into its executable and can't be used: FrameWarp takes the depth from ReShade and works out the motion and the camera from the picture (the depth buffer is found within seconds; turn the camera for a few seconds after loading). |
| *Elden Ring* (Steam, DirectX 12, no DLSS or FSR) | Like any ReShade setup for this game, it has to run without Easy Anti-Cheat (offline). FrameWarp finds the depth buffer and works out the motion and the camera from the picture; the camera circles the character, and the character and the ground around it follow just as much of each turn as they do in the game. When the GPU is the limit, *Prevent GPU queueing* stands aside by itself. |
| *Dark Souls Remastered* (Steam, DirectX 11, no DLSS or FSR) | FrameWarp finds the depth buffer, the game's own camera and its motion vectors (drawn only on what moves by itself; the scenery's motion comes from the camera), and takes the HUD from a copy made just before the game draws it. |
| *Hi-Fi Rush* (Steam, DirectX 12) | FrameWarp works out the motion from the picture and takes the HUD from a copy of the picture made just before the game draws it onto the screen. |
| *Assassin's Creed Black Flag Resynced* (upscaler off) | The same as above, in DirectX 12. |
| *Metro 2033 Redux* (Steam, DirectX 11, no DLSS or FSR) | The game only runs in exclusive fullscreen: tick **Force a borderless window** in the XPAR panel and restart the game. FrameWarp finds the depth buffer and works out the motion from the picture. |
| *Cyber Hook* (Steam, DirectX 11, no DLSS or FSR) | The game has no usable depth buffer: FrameWarp reprojects camera turns only (walking and strafing move at the game's frame rate). |

Other games may work but are untested.

## Requirements

* A DirectX 12 game with **DLSS or FSR** (see above) turned on, a Vulkan game with **DLSS** (tested with
  DOOM Eternal and Enshrouded), or a DirectX 12, DirectX 11 or Vulkan game **without** them. Without DLSS
  or FSR, ReShade has to be able to see the game's depth buffer, as for any depth effect (online games
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
   To update every FrameWarp it finds to the installer's version at once, choose **A** (shown when an
   older one is found: in your Steam libraries and in every folder the installer has installed to).
   The installer keeps that list of folders since version 2.3, so a game outside your Steam libraries
   (Epic, GOG, a folder of your own) that was installed with an earlier version isn't found by **A**:
   choose it once with **P**, and **A** finds it from then on.
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
`install.ps1` accepts `-Game`, `-GameDir`, `-Latewarp <dll>`, `-EngineIni <path>`, `-NoCvar`,
`-Uninstall`, `-UpdateAll` (every older FrameWarp found, as **A** does) and `-SteamPath <folder>`.

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
| **Frame counter** | On by default, remembered per game. A small grey counter in a corner of FrameWarp's picture: **GAME**, the game's frames per second that FrameWarp takes in, and **OUT**, the refreshes per second it shows. Pick the corner (top right by default) or turn it off. Counters drawn into the game, such as the Steam overlay's, are hidden under FrameWarp's picture. |
| **Auto latency** | How far behind the newest game frame the displayed camera sits. **Lowest latency without edge fill** (default): every refresh, the camera as late as it can be without revealing anything the game hasn't rendered - the camera as of now (your mouse up to this moment) while you aim or turn slowly, falling back as far as one game frame during fast turns, where what the warp reveals comes from the game's last two frames (keep *Background memory* on). **Camera as of now (edge-limited)**: always as close to now as the uncovered screen edge allows (*Extra uncovered edge*: how much more of the edge than *Auto* may be uncovered; *No edge limit*: always now). **Auto**: 1/2 game frame behind, 1/4 in games without HUD layers, a whole frame in games without DLSS or FSR. **1 frame**, **1/2**, **1/4**, or **Off** to set the latency by hand. |
| **Prevent GPU queueing** | On by default, remembered per game. FrameWarp measures the rate the game can sustain and paces its frames to reduce GPU queueing. It stands aside for the game's Reflex and frame generation. When the GPU is the limit, it instead limits the game to one queued frame, then periodically retries pacing. DirectX 12 and 11. Performance still depends on the GPU load shared by the game and presenter. |
| **Uncovered screen edges** | XPAR engine, remembered per game. What a fast turn uncovers past the edge of the game's frame, where background memory has nothing. **Dithered** (default): the game's frame as it is there, mixed into the warped picture in a fine pattern near the edge - noisy, not smeared. **Soft**: a blur of the scenery along the edge, wider further out. **Extended edge pixels**: the edge repeated (sharper close to it, streaks further out). |
| **Controller right stick drives rotation** | On by default, remembered per game. The right stick of a controller (XInput: Xbox pads, or others through Steam Input) moves the reprojected camera between the game's frames, as the mouse does; how fast the stick turns the camera and the game's response curve are learned from the game's own camera (shown under the camera model). |
| **Force a borderless window** | Off by default, remembered per game, applies when the game is restarted. For games that only run in exclusive fullscreen, where nothing can be shown over the game: the game stays in a window that covers the screen and is told it is fullscreen. |
| **Show the mask** | Shows the pixels the active warp holds still. With the default continuous warp, characters and weapons follow the displacement field, so the held-pixel view mainly shows the HUD. The view differs with other warp paths. |
| **Record detailed diagnostics** | Off by default, remembered per game. Enables routine presenter/add-on diagnostics, performance measurements for the logs, and frame-by-frame CSV recordings. Also enables **Ctrl+Shift+M** (mark a moment), **Ctrl+Shift+D** (capture frames), and **Ctrl+Shift+F8** (record a sequence). Capturing frames can briefly pause the output. Crash reports and add-on initialization-failure reports are separate. |
| **Reset camera model** | Re-learns how the game's camera responds to your mouse. This also happens automatically within seconds of play. |
| **Start presenter** | Restarts the presenter if it was closed. |

Settings that the current path does not use are greyed out; their tooltips explain why.

The panel shows only the settings above; the others keep working at their defaults, or at what
`ReShade.ini` remembers. To see and change them, add `ShowAllSettings=1` under `[FrameWarp]` in the
game's `ReShade.ini`:

| Setting | What it does |
|---|---|
| **Automatic present lead** | On by default. FrameWarp measures how early it must render to reach each refresh and adjusts the lead. Untick it to set **Present lead (ms)** manually. With the XPAR engine and a lead above zero, new frames are processed on a separate GPU queue; intake work is scheduled between refresh warps where possible. Both queues still share the GPU, so heavy work can delay a refresh. |
| **Keep HUD still** | Warps the scene but not the HUD, using the HUD layers the game provides (Expedition 33). |
| **Keep still** | Selects the HUD and character/weapon handling: **HUD + character/weapon** (default), **HUD only**, **Character/weapon only** or **Off**. HUD layers are used where available; otherwise the HUD is detected. Character/weapon motion comes from game vectors or picture-derived estimates. With the default continuous warp, these regions guide the displacement field rather than all being held at a fixed screen position. |
| **Character/weapon by how much it follows the turn** | On by default, remembered per game. Learns how much each part of the picture follows the camera's turn and movement, so scenery, a character and a vehicle interior can move differently. The default continuous warp smooths the displacement around them to reduce tears, at the cost of stretching nearby scenery. *Hold the whole character* and *Stretch around character/weapon* apply when this option is off. The near-camera and turn rules still contribute to character classification. |
| **...measured against the turn from your input** | On by default, remembered per game. A turn from your mouse or stick and a turn the camera makes by itself (a car's steering) are learned separately, and the warp applies each to its own part of the predicted turn: a car's interior stays put while you steer and follows when you look around. |
| **Keep near-camera motion still** | On by default, remembered per game. Helps classify nearby things moving against the camera, such as an animated weapon or hands. The active warp uses that classification for character/weapon handling. Try disabling it if nearby scenery is incorrectly treated as attached to the camera. |
| **Keep still what the camera turns around (third-person)** | On by default, remembered per game. Part of character/weapon classification: identifies things that move much less on screen than the camera turn would move scenery. With follow share on, this still helps identify attached regions; it does not enable the separate measured-orbit camera model. |
| **Hold the whole character** | On by default, remembered per game; only used with *Character/weapon by how much it follows the turn* off. A character the camera follows (running, its limbs swinging, the camera lagging a little) is held as one piece: what stays in place on screen over several frames while the camera moves, and its moving parts touching it. |
| **Hold sight reticles** | On by default, remembered per game, XPAR engine. Part of *Character/weapon*: red dot and holographic sights whose reticle the game draws without depth of its own (it would skip with the scenery behind the glass while turning). Inside the window the held weapon encloses, what is on the glass (reticle, glow) is learned as a layer that follows the reticle within a few frames of turning, and only the scenery under it is warped. Sights drawn with depth are held anyway. |
| **Stretch around character/weapon** | 0-32 render pixels, 1 by default (0 turns it off), remembered per game, XPAR engine; only used with *Character/weapon by how much it follows the turn* off. Where the scenery slides away from a kept-still character or weapon, the warp uncovers a gap and fills it from the nearest scenery pixel, which draws streaks and smears the outline its motion vectors miss (anti-aliasing, the upscaler's softening). With a width set, the scenery that close follows the warp less and less towards the character or weapon: the gap is covered by slightly stretched scenery, and the outline stays with it. |
| **Background memory for uncovered areas** | On by default, remembered per game, XPAR engine. Remembers visible scenery and uses it where the warp reveals areas it has seen before. With the default continuous warp, it mainly helps at screen edges and behind the HUD; the field handles stretching beside characters. GPU and memory costs depend on resolution and scene content. |
| **Background memory resolution** | **1/4** (default), **1/2** or **Full**, remembered per game, relative to the game's picture dimensions. Full preserves finer detail in remembered fills; lower resolutions reduce GPU work and video memory. Doubling each dimension uses roughly four times the memory for these textures. |
| **Find the HUD** | Remembered per game. **Upscaler output + camera motion check** (default): compares the final picture with the upscaler's (DLSS or FSR) own output, which has no HUD, and leaves out what is seen moving with the world when the camera turns (bright lights, neon or effects drawn after upscaling). **From the upscaler output**: the comparison alone. **Learned from camera motion**: finds the HUD from what stays put on screen while the camera moves; it needs a few seconds of camera movement, and is used whenever there is no upscaler output. |
| **Use the game's own HUD layer when found** | On by default, remembered per game; DirectX 12, DirectX 11 and Vulkan games. Where the game draws its whole HUD into a layer of its own and blends it on at the end, or draws it straight onto its final picture (XPAR then takes a copy of the picture from just before), XPAR takes the HUD from that step instead of finding it: exact from the first frame, and bright lights or neon next to the HUD are never mistaken for it. Games that hand their HUD layers over themselves (through their frame generation integration) use those anyway. In games with DLSS or FSR, a HUD layer is first checked against the upscaler's output (everything of a HUD, markers anchored in the world too, is drawn over the finished picture; full-screen menus do not count): once it passes, a frame without it is one where the game hides its HUD, and nothing is detected there; a layer that fails is ignored and the HUD is detected. Without DLSS or FSR the layer can't be checked, and it isn't used in frames where it covers 90% of the screen or more (that is no HUD). The panel says where the HUD comes from and how the check went. Turn it off if a game's HUD warps or parts of the scenery are held still with it: the HUD is then found as chosen in **Find the HUD**. |
| **Learn where the game draws its HUD** | On by default, remembered per game. For a game where none of the steps above is found, XPAR tries the places the game could be drawing its HUD, scores them against the picture and keeps the one that holds the HUD: learned once and saved per game (**Relearn** starts over). It rests while the game hands its HUD over another way. |
| **Fill behind the HUD from the upscaler output** | On by default, remembered per game; XPAR engine, with the HUD found from the upscaler output. When the camera turns, the scenery the HUD covered is taken from the upscaler's output (colour-matched to the game's picture) instead of being smeared from the surroundings, so the HUD leaves no trail. Effects the game adds after upscaling, such as bloom or film grain, may be missing in those spots. |
| **Warp engine** | **XPAR** (default): FrameWarp's own engine, any GPU. **NVIDIA Latewarp**: selectable when `nvngx_latewarp.dll` is installed and the GPU is NVIDIA's. |
| **Forward-splat warp** | Off by default, remembered per game, XPAR engine. Off: searches from each output pixel for its source using the active warp field. On: rendered pixels claim output locations, with the nearer surface winning. Both paths support the continuous warp; switching splat off does not switch back to the old character-holding method. |
| **Presenter GPU priority** | Keep **Realtime**. Lower priorities cannot hold the refresh rate while the game loads the GPU. |

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
  default settings, as a dithered transition to the unwarped frame).
* Overlays drawn into the game, such as the Steam overlay (Shift+Tab) and its FPS counter, are covered by
  FrameWarp's picture while it shows its frames. Use the **Frame counter** instead.
* VRR (G-SYNC, FreeSync): while FrameWarp shows its frames, the screen runs at its maximum refresh rate.
  FrameWarp's window is always composited by Windows, so VRR can't follow it, also with G-SYNC for windowed
  mode. FrameWarp paces itself to the screen's refreshes instead, so no frames queue at the maximum.
* Where the game hands over no camera and FrameWarp doesn't find it (it looks for it in every game, and
  can take a minute to find it after loading), the camera is estimated from motion vectors or the picture,
  which is a little less exact. Games whose camera data doesn't match their motion vectors are estimated too.
* Upscalers built into a game's executable instead of a DLL (Cyberpunk 2077's FSR 2.1, Resident Evil 4
  Remake's FSR 2), XeSS and any other unsupported upscaler aren't used directly: FrameWarp works as in a
  game without DLSS or FSR.
* Without DLSS or FSR, where the game's own motion vectors aren't found (Vulkan games are not searched),
  the motion is worked out from the picture: very fast turns, dark or blurry scenes and flat surfaces are
  where it is least sure. When the depth reads empty, FrameWarp switches on ReShade's depth copy and asks
  for one restart of the game. In a game whose depth buffer ReShade can't see, or that has none belonging
  to the picture, only camera turns are reprojected. DirectX 9, DirectX 10 and OpenGL games aren't
  supported.
* Turning DLSS on in the middle of a session, on a graphics card with little video memory (8 GB at 4K),
  can leave the game at a very low frame rate until it is restarted (seen in DOOM Eternal and
  Enshrouded).
* Very fast motion goes beyond what re-projecting a single frame can hide. In Expedition 33's
  overworld, very fast camera turns can make the picture shake slightly.
* Expedition 33's optional executable probes use addresses and expected bytes from a previously tested
  Steam executable. Probes whose bytes no longer match are skipped; this is not a compatibility guarantee
  for the latest game build.

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

FrameWarp asks Windows Explorer to start its presenter, so that the Steam overlay is not loaded into it (it
could freeze the presenter). If security software blocks that, or the game runs as administrator, the
presenter is started by the game instead, up to 10 seconds later. If the presenter's picture ever stops
updating for 2 seconds, it closes itself and the game carries on without reprojection; `ReShade.log` and,
with diagnostics on, `presenter.log` say what happened.

If the game crashes when you open the ReShade menu (Red Dead Redemption 2 does, also without FrameWarp),
turn off ReShade's **Generic Depth** add-on: add `DisabledAddons=Generic Depth` under `[ADDON]` in the game's
`ReShade.ini` (it then shows unticked in ReShade's Add-ons tab). Games with DLSS or FSR don't need it: FrameWarp takes the depth from the upscaler.

For a problem report, turn on **Record detailed diagnostics** in the XPAR panel before reproducing
it. Routine logs are written to `FrameWarp\logs\`, next to the presenter; when diagnostics first open
in a run, the preceding logs are moved to `logs\previous\`. A read-only installation falls back to
`%LOCALAPPDATA%\FrameWarp\<game-folder>\`. Press **Ctrl+Shift+M** to mark a bad moment, then include the
current and previous logs and the game's `ReShade.log`. With diagnostics enabled, the latter includes
FrameWarp's depth-choice messages. Routine FrameWarp diagnostics are disabled when the option is off;
crash and add-on initialization-failure reports remain available. Developer test installations may
force diagnostics on with `DiagnosticsOn=1` under `[FrameWarp]` in `ReShade.ini`.

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
