# Changelog

## Unreleased

* Installer: the game list shows which FrameWarp version is installed, and it asks before replacing an
  existing install (it says so for the same version or a downgrade; `-Force` skips the question).
* Installer: updates are safer. Staged copies of the add-on (RE9's `_storage_`) are replaced too, and
  a presenter left running from a crashed game session is stopped instead of blocking the update.
  Logs, calibration and the Latewarp DLL are kept.
* Fixed: on displays with Windows scaling (laptop screens at 125-150%) the overlay showed only the
  top-left part of the picture, enlarged.
* The FrameWarp panel warns when Windows' hardware-accelerated GPU scheduling is off on the game's
  GPU. Without it, the output may not reach the refresh rate.
* README: explains why an RTX GPU is required.
* New option **Find the HUD from the DLSS output** (off by default, remembered per game): the HUD is
  found in every frame by comparing the final picture with DLSS's own output, which has no HUD. No
  learning, and it follows HUD that fades in and out. Very clean in Resident Evil Requiem and in games
  that call DLSS directly; in some games parts of the scenery (neon, strong bloom) may be kept still,
  so check it with the mask view.
* New game frames are prepared between display refreshes instead of inside one, so taking in a frame
  no longer makes the output miss a refresh.
* Debug option to show the no-warp mask on screen.

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
