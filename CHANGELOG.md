# Changelog

## 1.5.0

* **Steady refresh rate in demanding games:** with the XPAR engine, new game frames are taken in on their
  own GPU queue while the warp keeps showing the previous frame, so the several milliseconds of HUD and
  mask work per game frame no longer make refreshes late (DOOM Eternal, Resident Evil Requiem now hold
  120 Hz). Uses about 50-100 MB more video memory at 4K. Latewarp and *Present lead* 0 keep the single
  queue.
* **Find the HUD** (replaces the *Find the HUD from the upscaler output* checkbox, remembered per game):
  the new default, **Upscaler output + camera motion check**, finds the HUD from the upscaler's output
  and leaves out what is seen moving with the world when the camera turns, so bright lights, neon and
  effects drawn after upscaling keep warping. *From the upscaler output* and *Learned from camera
  motion* remain; without an upscaler output the learned detection is used.
* **Fill behind the HUD from the upscaler output** (new option, on by default, remembered per game, XPAR
  engine): the scenery the HUD covered is shown from the upscaler's output when the camera turns, instead
  of a smeared trail.
* **Keep near-camera motion still** (new option, on by default, remembered per game): turn it off if
  the floor near the camera is kept still while strafing.
* Estimated camera: a first-person weapon no longer counts as scenery when working out the camera's
  movement. It made up a small camera move tied to turning, which left holes in the weapon's mask,
  held parts of the nearby floor, and could get the strafing direction wrong.
* A weapon or character close to the camera whose motion only slightly differs from the camera's is now
  kept still too (it could show holes in the mask).
* Vulkan: frames are copied only after the game has finished drawing them (the lower part of the picture
  could come from another frame, and the ReShade menu could flicker).
* **Vulkan support** (tested with DOOM Eternal, DLSS): depth, motion vectors and the frame are passed
  from Vulkan games to the presenter. The installer recognises games that use ReShade's Vulkan layer
  (a ReShade.ini next to the game's executable, no ReShade DLL).
* The camera model keeps what it learned while the mouse barely moves (menus, standing still, the
  ReShade menu open) instead of dropping the mouse input until it is learned again.
* See-through HUD panels stay held with the camera motion check: the scenery moving behind them could
  make parts of them look like world and warp (doubled numbers, HUD colours in the gaps around the weapon).
* Estimated camera: the camera shook when facing certain parts of a room after a few minutes of play (its
  sense of "up" slowly drifted); it now follows the drift. In menus before any turning the estimated camera
  no longer warps slightly.
* The mouse is no longer applied while the game's camera clearly does not follow it (menus with a cursor
  the game draws itself). Low mouse sensitivity and gamepads are not affected.
* Returning to a game's main menu no longer freezes the view until alt-tab (games whose frame numbers
  start over there).
* The presenter writes its log, recordings and camera profile in the background: a slow disk no longer
  costs refreshes.
* **Ctrl+Shift+D** (development) also saves depth, motion vectors, the masks and the fill behind the HUD.
* **Show the mask**: green (HUD being learned) only with *Learned from camera motion*.

## 1.4.0

* **No NVIDIA RTX GPU or DLSS required any more:** FrameWarp works with DLSS and with AMD FSR, and its own
  warp engine runs on any DirectX 12 GPU (tested on NVIDIA; AMD and Intel untested).
* **NVIDIA Latewarp is optional.** The installer no longer asks for `nvngx_latewarp.dll`; with it,
  Latewarp can be chosen in the **Warp engine** setting, which is locked to XPAR's engine otherwise.
* The installer recognises FSR games too.
* **Record detailed diagnostics** (new option, off by default, remembered per game): the frame-by-frame
  recordings in the logs folder are only written when it is on. `presenter.log` is always written.
* The ReShade panel, tab and add-on are now called **XPAR** (xPoiler's Asynchronous Reprojection).
  File names, folders and settings are unchanged.
* **FSR support** (FSR 3.0 and 3.1 tested; FSR 4, and FSR 2 where the game ships its DLL, should work
  but are untested): FrameWarp reads the depth and motion vectors FSR is given.
  Games that still send their camera through Streamline with FSR (Cyberpunk 2077) keep their own
  camera; otherwise FrameWarp works out the camera's movement from the motion vectors. Switching
  between DLSS and FSR in a game's menu is followed on the fly. **Find the HUD from the upscaler
  output** now works with FSR as well. (FSR built into a game's executable, like Cyberpunk 2077's
  FSR 2.1, cannot be reached; FSR 2 is supported where the game ships its DLL.)
* **XPAR warp engine** (new **Warp engine** setting, now the default): FrameWarp's own warp, also used
  when NVIDIA Latewarp is not available. It uses depth, so near and far objects move apart correctly
  when you strafe, keeps the HUD and character/weapon still like Latewarp, costs less GPU, and does no
  work at all while the camera stands still. NVIDIA Latewarp stays available in the setting.
* The ReShade menu stays visible while moving the mouse (Clair Obscur: Expedition 33): FrameWarp pauses
  warping while the menu is open and ignores the mouse meanwhile.
* Estimated camera (games without their own camera data):
  - zooming is no longer taken for moving forward or back;
  - it follows games whose depth runs the standard way round;
  - the field of view a game reports is checked against the picture, because some games report the
    horizontal one. Resident Evil Requiem with FSR was choppy in fast turns because of this.
* **Lighter on the GPU:** FrameWarp keeps its copies of the game's picture in the game's own format
  (half the memory traffic for 8- and 10-bit games, the same picture), no longer copies the previous
  frame while the HUD comes from the upscaler output, and no longer stalls the GPU mid-frame when it
  estimates the camera (FSR could miss the odd refresh because of this). HUD detection from the
  upscaler output is about 20% cheaper for the same result, and textures a game does not use (HUD
  layers, the empty UI layer) are no longer allocated (about 200 MB less VRAM at 4K). The log now
  shows every 10 seconds how much of the GPU FrameWarp uses and on what.
* Game folders that only administrators may write to (some launchers install games that way): the
  installer says to run it as administrator, and FrameWarp keeps its logs and calibration in
  `%LOCALAPPDATA%\FrameWarp\<game folder>` instead.
* Installer: DLSS games without Streamline are recognised (no more "Streamline not found" warning).
* If ReShade refuses to load the add-on, the reason FrameWarp can see is written to
  `%LOCALAPPDATA%\FrameWarp\addon-init.log`.

## 1.3.0

* **Returnal support**, and games that call DLSS directly without NVIDIA Streamline: FrameWarp reads
  DLSS's depth and motion vectors and works out the camera's movement from them (no launch options
  needed).
* **Frame generation safeguard:** FrameWarp and frame generation (DLSS, FSR or XeSS) do the same job and
  cannot be combined. When a game has frame generation on, FrameWarp now pauses, leaves the game
  untouched and says so in its panel; it resumes by itself when frame generation is turned off.
  Before, the picture could freeze or break up.
* **Keep still** dropdown: HUD + character/weapon (default), HUD only, character/weapon only, or off.
  Character/weapon detection now works in every game, including games with their own HUD layers
  (Clair Obscur: Expedition 33), and also holds a third-person character that stays put on screen
  while the camera turns. Objects moving on their own keep warping.
* New option **Find the HUD from the DLSS output** (off by default, remembered per game): the HUD is
  found in every frame by comparing the final picture with DLSS's own output, which has no HUD. No
  learning, and it follows HUD that fades in and out. Excellent in Resident Evil Requiem and Returnal;
  in some games bright lights and neon may be kept still too, so check it with **Show the mask**.
* **Show the mask** (debug): tints what is kept still.
* Smoother output: new game frames are prepared between display refreshes instead of inside one, so
  taking in a frame no longer makes the output miss a refresh.
* FrameWarp steps aside in menus and loading screens that skip the game's usual rendering.
* Fixed: on displays with Windows scaling (laptop screens at 125-150%) the overlay showed only the
  top-left part of the picture, enlarged.
* The FrameWarp panel warns when Windows' hardware-accelerated GPU scheduling is off on the game's
  GPU. Without it, the output may not reach the refresh rate.
* Installer: the game list shows which FrameWarp version is installed, and it asks before replacing an
  existing install (it says so for the same version or a downgrade; `-Force` skips the question).
* Installer: updates are safer. Staged copies of the add-on (RE9's `_storage_`) are replaced too, and
  a presenter left running from a crashed game session is stopped instead of blocking the update.
  Logs, calibration and the Latewarp DLL are kept.
* README: explains why an RTX GPU is required.

## 1.2.0

* **Cyberpunk 2077 support.**
* **HUD and first-person weapon detection** for games that don't provide HUD layers (Cyberpunk 2077,
  Resident Evil Requiem). It's on by default and keeps both unwarped:
  * the weapon and hands are recognised by their motion (they move with the camera);
  * the HUD is recognised by what stays put on screen while the scene moves under it. That includes
    HUD that appears only when needed and a crosshair over nearby walls; semi-transparent panels are
    held by their edges and may still move slightly. Moving or repeating scenery is not mistaken for
    HUD.
* **Auto latency** is the new default: 1/2 game frame, or 1/4 in games without HUD layers.
* Fixed: the view fighting the game's camera when it zooms, for example when stopping aiming while
  walking backwards in Resident Evil Requiem.
* Installer: finds the `Engine.ini` of Unreal Engine 4 games too, including ones stored under an extra
  store folder.

## 1.1.0

* **Resident Evil Requiem support** (requires REFramework).
* Works with right-handed engines such as RE Engine: the camera axis convention is now read from the
  game's projection, which fixes a doubled, smeared image during camera motion there. Unreal Engine
  games behave exactly as before.
* The add-on also finds the presenter next to the game's executable. This fixes "presenter not
  found" in games that load their DLLs from a staging folder (RE9's `_storage_`).
* Installer: double-click `install.bat` or `uninstall.bat` to choose the game from a numbered list of
  detected Steam games, or type a folder for non-Steam games.
* Installer: picks the live ReShade when a mod manager keeps backup copies. Uninstall also removes
  the add-on copies a game makes in its staging folder.
* The "presenter failed to start" message now shows the Windows error and the paths that were tried.

## 1.0.0

First release.

* Camera motion at the display's refresh rate, reprojected with NVIDIA Reflex 2 Frame Warp from the
  game's Streamline data.
* Camera prediction from the game's camera history and raw mouse input, with automatic calibration.
* Depth-correct parallax for camera movement (third-person orbit, walking), and a HUD that stays
  still.
* Auto latency at 1, 1/2 or 1/4 of a game frame, and frame pacing locked to the display.
* Safe when the DLSS preset (render resolution) changes during play.
* An installer that finds the game's ReShade, sets up Unreal Engine games, and uninstalls cleanly.
