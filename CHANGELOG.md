# Changelog

## Unreleased

* **Lower output latency, steadier timing.** The present lead is automatic (new default): XPAR measures how
  long its warp takes and renders as late as it still makes the refresh, backing off when a frame misses
  (Stalker 2: 2.5-3.5 ms instead of 6, present to screen 16 ms instead of 19.6). The camera is taken at the
  planned moment before each refresh, so lead changes and wake-up jitter no longer move it. XPAR's frames no
  longer queue behind a late one: the newest finished frame is shown at each refresh, so one missed refresh
  costs that refresh only (before, every frame after it stayed a refresh later, up to 24.5 ms). The slider
  remains as a manual setting.
* The Output fps figure counts refreshes only (with Present lead 0 it counted idle checks: 373 fps shown).
* The first-person weapon is held still while walking backwards too: its nearest part (behind where the
  camera was a frame earlier) went unrecognised and warped with the scenery (DOOM Eternal).
* **Games that hand DLSS full-resolution motion vectors work properly** (DLSS without Streamline tags, no
  "low-resolution motion vectors" flag - Stalker 2 without frame generation). XPAR read only a quarter of
  them: the camera estimate saw a narrower field of view, rejected most frames, and the mouse model learned
  from a camera that barely moved. They are now taken whole and scaled to render pixels.
* **The mouse model no longer unlearns the mouse.** Stretches where the camera clearly does not follow the
  mouse (inventories, maps, dialogues with the game's own cursor) are left out of its fit, a learned model is
  only refitted with enough usable data, and frames a camera estimate cannot explain are not learned from (the
  XPAR panel says when the estimate is unreliable). A poor model is no longer saved.
* A locked field of view of an estimated camera is learned again when it keeps failing, for the upscaler's
  motion vectors too.
* **Latency breakdown** in the XPAR panel and the log: how far behind the displayed camera is, the time from
  XPAR's present to the screen, and the age of the camera when it reaches the screen.
* Record detailed diagnostics also saves what the game hands its upscaler (`upscaler.csv`), and the DLSS
  inputs are logged whenever they change. Captures continue numbering instead of overwriting earlier ones.
* **More settings are remembered per game:** Keep still, Keep HUD still, Warp engine, Raw mouse drives
  rotation, Auto latency and Latency <-> smoothness, Rotation extrapolation, Orbit distance, Max
  extrapolation, Present lead, and Manual mouse gain with its values. Enable reprojection and the
  comparison and debug switches (Show original, Debug strip, Invert warp, Show the mask) are not
  remembered. As before, an update resets the remembered settings to the defaults.

## 1.8.5

* **The HUD stays still with FSR frame generation too.** Games that hand FSR frame generation a picture
  without the HUD (Resident Evil Requiem) get their HUD found exactly from it, every frame, as Cyberpunk
  2077's already was with its Streamline HUD-less picture.
* **FSR frame generation no longer stalls when the game stops sending its depth** (Cyberpunk 2077 with
  FSR frame generation and DLSS: "this kind of frame generation is not supported" until the DLSS quality
  mode was changed). XPAR's fallback for games without DLSS or FSR started then and took the frames frame
  generation's images belong to; it stays off while the game's frame generation is in use.
* Record detailed diagnostics also saves the add-on's call counters once a second (`hooks.csv`).
* **Controller support.** The right stick of a controller (XInput: Xbox pads, or others through Steam
  Input) now drives the reprojected camera, as the mouse does: XPAR learns from the game's own camera how
  fast the stick turns it and the game's response curve, and moves the camera with the held stick between
  the game's frames. On by default (**Controller right stick drives rotation**, remembered per game).
* The presenter starts by itself again in games started with frame generation already on (it had to be
  started from the XPAR panel).
* **Fewer missed refreshes with the mouse too.** The camera model's fit used to run on the presenter's
  render thread every 30 game frames and took long enough to miss a refresh at 120 Hz now and then; it
  runs on its own thread now, in half the time, with exactly the same results.
* **Games without a usable depth buffer work too: camera turns only.** Without DLSS or FSR, XPAR takes the
  depth from ReShade; some games have no depth buffer that belongs to the picture (every one reads empty
  or unrelated). XPAR used to wait for one, or used whatever ReShade had selected, which moved parts of
  the picture by distances that are not there. Now, while no depth buffer stands out, XPAR treats
  everything as far away: camera turns are reprojected exactly (they need no depth), and walking and
  strafing move at the game's frame rate. It keeps looking for a usable buffer and switches to it when
  one turns up, without pausing reprojection while it looks. A buffer ticked by hand in ReShade's list
  is still used as it is. Needs the updated XPAR.fx (the installer puts it in place).
* **New option: Force a borderless window** (off by default, remembered per game), for games that only
  run in exclusive fullscreen, where nothing can be shown over the game. The game stays in a window that
  covers the screen, and is told it is fullscreen (games that check would otherwise keep switching).
  Applies when the game is restarted.
* **Warning when a frame rate cap outside XPAR holds its output back.** XPAR's window is never the
  focused one (the game keeps the keyboard and mouse), so NVIDIA's "Background Application Max Frame
  Rate" caps it, and can cap the game it covers. The XPAR panel now warns when XPAR's output stays well
  below the refresh rate because its presents are held back from outside. The README's troubleshooting
  section explains the fix.
* The presenter GPU priority is remembered per game, like the other saved options.
* Installer: the update question shows the version again ("Update it to 1.8.0?"), and games without DLSS
  or FSR are no longer marked as not working in the game list.

## 1.8.0

* **The game's own frame generation works with XPAR** (DLSS Frame Generation, including multi frame
  generation, and FSR 3.1 frame generation; XPAR engine). It used to be one or the other: with frame
  generation on, XPAR paused. Now XPAR takes each image the game's frame generation makes, right where it
  is made, and over each game frame shows them in turn and then the frame itself - every one of them moved
  to the current camera, as with any frame. Moving objects move at the frame generation's rate, one game
  frame late as frame generation always is; the camera is not delayed. Nothing to switch on: it starts by
  itself when frame generation is on in the game. With NVIDIA Latewarp as the warp engine, XPAR still
  pauses while frame generation is on. The ReShade menu, drawn after these images are taken, shows the
  game's own picture while it is open. FSR 3.0 frame generation (the older FidelityFX SDK) is not taken
  yet: XPAR pauses as before. Multi frame generation up to 6x: with more than three generated images per
  frame, three are shown, spread evenly (6x: the first, third and fifth).
* Frame generation is also recognised from the game's calls to it (DLSS Frame Generation, FSR 3.x frame
  generation), not only from the number of images presented. Where ReShade sees only the rendered frames'
  presents (on some setups), 1.7.0 missed frame generation entirely: it neither paused nor
  warned, and ran on top of it.
* **No GPU work while nothing moves.** A refresh that would show the same picture as the one before (the camera
  has not moved since the game's frame) is no longer drawn or presented at all; the window keeps showing the
  picture. Since 1.5.0 such a refresh still went through the warp shader once per refresh (taking frames in on
  their own GPU queue made the plain copy a full-screen pass), so a still camera cost as much as a moving one.
  A refresh that does show a new frame as it is now copies it with a plain copy, not the warp shader: about
  half the GPU time at 4K (0.2 ms instead of 0.4 ms).
* **The refresh rate is taken from the display**, from its current mode (exactly: 119.88 Hz as such), instead of
  being measured from FrameWarp's own presents. The measurement could lock onto a wrong rate at the start and
  keep it until reprojection was switched off and on again. The presents now only tell when each refresh
  happens; the rate is read again when the game moves to another display or the mode changes.
* Less GPU time per game frame, with exactly the same results: the HUD detection's check against the camera
  motion classifies each pixel once instead of twice, and the final mask reads each pixel's HUD state once
  per tile instead of nine times. The log shows the work on each game frame pass by pass ("game frame GPU by
  pass").
* The XPAR panel warns when video memory is nearly full (Windows leaves FrameWarp too little of it): the game
  and FrameWarp then don't both fit and both stutter. Frame generation needs extra video memory too; once it
  has been off for a few seconds, the textures FrameWarp took its images into are released.
* Games that send their HUD-less picture without a UI layer (Cyberpunk 2077 with frame generation on): the HUD
  is found exactly, every frame, where the picture differs from the HUD-less one, which is also the scenery
  behind the HUD (it fell back to the learned HUD map, which let parts of the HUD move with the camera).
* With FSR in a game that sends its depth and motion vectors through Streamline, the HUD is found from
  the upscaler's output too, as with DLSS (it fell back to the learned HUD map).
* **Move objects at the display rate even without frame generation** (new option, experimental, off by
  default; XPAR engine, games with DLSS or FSR): XPAR's own frame generation for what moves on its own,
  for when the game's is off or not available. Cars, people and everything else the
  game's motion vectors show moving are shown one game frame late and move at the display rate, always
  between two real frames (never guessed ahead), instead of stepping at the game's frame rate; the camera
  is not delayed (the warp shows it at the displayed moment, as before). Each moving pixel goes in a
  straight line in 3D between its two positions (both frames' depth), so things that move with the camera,
  such as a car's interior while driving, stay where they belong; what an object uncovers shows the
  previous frame behind it. The character/weapon kept still by **Keep still** moves at the display rate
  too (its own on-screen motion, one frame late, still not warped with the camera); where the warp fills
  the area beside it from a moving object, the piece it copies moves with that object.
  Replaces an earlier attempt that never reached a release (it left halos around moving objects).
  Shadows (drawn on the ground, with the ground's motion vectors) stay where the game drew them.
* The add-on stays intact until the game's process is gone: the game's DLSS or FSR may still call into it while
  the game shuts down (FSR reconfigures its swapchain then).
* The motion analysis behind the masks, the motion vector scale and the camera check runs for every game
  frame with depth and motion vectors, whatever **Keep still** is set to. Since 1.4.0 it already did, by
  accident of a missing pair of braces; the camera check (1.6.1) relies on it, so it is now on purpose.
  The frame numbers in `motion.csv` were wrong whenever **Keep still** was *Off* (or *HUD only* in a game
  with HUD layers) and could drift by a frame or two otherwise; each fit now carries its own frame.
* The capture key (Ctrl+Shift+D) saves three consecutive game frames (`prev2_`, `prev_` and the current
  one), the HUD-less picture where the game has one, and each frame's camera (`_cameras.txt`). A capture
  is only taken while no newer frame is half taken in, so depth and motion vectors are always the same
  frame's.

## 1.7.0

* **Games without DLSS or FSR, or with both switched off (experimental, DirectX 12 and 11):** XPAR now works
  from ReShade's depth buffer alone. The motion between two game frames is estimated from the pictures themselves (in the presenter,
  on its own GPU queue: no motion vector shader is needed), and the camera from that and the depth. Against
  a game's real motion vectors the estimate is typically within a quarter of a pixel where the picture
  has detail; rain, water and flat areas are recognised as unreliable and left out of the camera
  estimate. The installer adds `XPAR.fx` to the game's `reshade-shaders\Shaders`; the add-on enables it
  itself, and sets up ReShade's depth by itself (below). ReShade has to be able to see the game's
  depth buffer, as for any depth effect. With DLSS or FSR switched on the game's own depth and motion
  vectors are used, exactly as before; switching them on or off mid-game hands over by itself.
* The game's depth buffer is found automatically for the ReShade path: each depth buffer of the
  picture's shape, whatever its size, is selected in turn and the one whose depth belongs to the picture
  is kept (ReShade's own pick skips a scene rendered well below the picture's size and can settle on the
  wrong buffer; ticking the right one in its list did not last beyond the session). A buffer ticked by
  hand is respected. Every step is written to ReShade.log.
* A game whose depth buffers read empty by the end of the frame (it clears them first) is recognised -
  only while a 3D scene is being drawn and the picture moves, so never in a menu or a loading screen -
  and ReShade's "Copy depth buffer before clear operations" is switched on; the panel asks for one restart.
* Which way round the game stores its depth is found out too: ReShade's "reversed" depth setting no
  longer has to match the game for XPAR.
* Games whose camera XPAR estimates (no camera data from the game): the nearby scenery could start
  jumping between refreshes while the distance stayed smooth, after a frame with next to no depth in it
  had thrown the estimated position millions of units away (its steps were then coarser than a frame's
  move). Such moves are refused, and the position starts again from zero before it gets far.
* The field of view XPAR learns (games that give no camera) is judged by the camera's turn and move
  together. Judged by the turn alone it came out far too wide under a high third-person camera (the ground
  a few metres away moves with the camera's orbit), and never settled: the nearby ground shook, and the
  motion vector scale kept being learned again.
* **DirectX 11 games** (experimental): the game's frames reach the presenter through textures and a
  fence shared with the game's DirectX 11 device (Windows 10 1703 or later). Without DLSS or FSR data,
  as in DirectX 12 games: ReShade's depth and XPAR's own motion estimation.
* The ReShade path (no DLSS or FSR): frames are timed at the game's steady pace instead of when they
  were presented (an uneven presentation made the warp's camera speed jump: shaking, e.g. RE2), and only
  the parts of the picture whose motion XPAR is sure of can count as moving on their own (under TAA or in
  dark areas, the guessed motion of the rest was mistaken for movement and warped along: wobble).
* The ReShade path (no DLSS or FSR): the camera put together from the picture's motion is kept level
  and eases back from drifting up or down (it tipped over within minutes: the view swung and the mouse
  calibration ran away), a frame whose motion cannot be worked out keeps the motion of the one before
  instead of stopping the camera dead, and "Auto" latency shows the view a whole game frame behind
  (the camera estimated from the picture cannot be carried ahead as well; RE2 went from shaking to steady).
* The capture key (Ctrl+Shift+D) saves two consecutive game frames. It and the marker key (Ctrl+Shift+M)
  now only work with **Record detailed diagnostics** ticked: pressed by accident, a capture froze the
  game for a second.

## 1.6.1

* **The game's camera is checked against its motion vectors.** Some games send camera data through
  Streamline that does not describe how the picture moves (Assassin's Creed Black Flag Resynced): the
  image warped wildly, or not at all. While the camera moves, XPAR now compares the game's camera with
  the game's own motion vectors; when they never agree, the camera is estimated from the motion vectors
  instead, as in games that send no camera. The verdict is remembered per game until the next update,
  and taken back if the estimate does no better. Nothing to set; the XPAR panel says when the camera is
  estimated.
* Games that send a Streamline camera but whose depth and motion vectors do not reach the Streamline
  hooks: they are taken at the DLSS call instead.

## 1.6.0

* After an update, the settings remembered per game are reset to the defaults on the first launch (the
  ReShade log says so), so every improvement to the defaults applies right after installing.
* **Keep still what the camera turns around (third-person)** (new option, on by default, remembered per
  game): over-the-shoulder cameras circle the character, so it barely moves on screen while the room
  swings around it, and the warp broke it up (Resident Evil Requiem, Expedition 33). What moves less than
  a fifth of what the camera's turn alone would move it is now kept still while turning.
* **Background memory for uncovered areas** (new option, on by default, remembered per game, XPAR
  engine): the scenery last seen beside the character or weapon is remembered for up to a second, so
  when the camera turns the area the warp uncovers shows what was there instead of stretched edge
  pixels. Costs about 80 MB of video memory at 4K and about 0.1 ms of GPU time per game frame, on the
  intake queue.
* **Stretch around character/weapon** (new setting, 0-32 render pixels, 1 by default, remembered per
  game, XPAR engine): the scenery beside the character or weapon follows the warp less and less towards
  it, so what the warp uncovers there when strafing or turning is covered by slightly stretched scenery
  instead of streaks, and the outline its motion vectors miss stays with it instead of smearing away.

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
