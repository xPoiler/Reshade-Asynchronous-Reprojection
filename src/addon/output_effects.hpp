#pragma once
// The game's ReShade effects on XPAR's own picture (Settings::reshade_on_output) - the add-on's half.
//
// The presenter shows its own picture over the game's window. Taken from the back buffer it carries the game's ReShade effects; taken
// from the game's HUD-less picture or from frame generation's images it does not (both are taken before ReShade draws). So the
// presenter runs the game's preset itself, on every picture it shows (src/presenter/output_effects.hpp: the game's ReShade, copied next
// to it, with an effect runtime of its own on its swap chain), and the game's own effects are switched off meanwhile - they would be
// drawn twice on the back buffer's path. XPAR's own effect (XPAR.addonfx, the depth feed) keeps running: ReShade draws add-on effects
// with its effects switched off.
//
// Here: the files the presenter's ReShade needs (the game's ReShade DLL copied as FrameWarp\ReShade64.dll, FrameWarp\ReShade.ini with
// the game's effect and texture folders), the preset as the game has it now (FrameWarp\ReShadePreset.ini, written again whenever it
// changes - Shared::reshade_preset_serial), whether any of the user's effects is on (Shared::reshade_fx_active), and the game's effects
// switched off while the presenter reports drawing them (PresenterStatus::reshade_state 3) - back on while the ReShade menu is open
// (with effects off it hides their list), when the presenter stops, or the setting is turned off.
#include "shared/protocol.hpp"
#include <reshade_api.hpp>
#include <string>

namespace fw::output_effects {
void register_events();
void unregister_events();
// Before the presenter starts (launch_presenter; framewarp_dir: the folder FrameWarpPresenter.exe is in). Nothing when the setting is off.
void prepare(Shared* sh, const std::wstring& framewarp_dir);
// The game's swap chain presented a picture (reshade_present, after ReShade's own effects): the preset written when it changed, whether
// any effect is on, the game's effects switched off or back on for the next picture. presenter_alive: the presenter process runs;
// in_process: the development presenter inside the game (none of this there).
void after_present(reshade::api::effect_runtime* runtime, Shared* sh, bool presenter_alive, bool in_process);
// For the panel: the game's effects are switched off right now (the presenter draws them).
bool game_effects_off();
// For the panel: the depth feed runs from the old XPAR.fx (before 2.4) - the effects stay the game's until XPAR.addonfx is installed.
bool old_feed_effect();
}  // namespace fw::output_effects
