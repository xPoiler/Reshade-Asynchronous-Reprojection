// UI probe (diagnostics): on Ctrl+Shift+D (with detailed diagnostics on), the next game frame's draws are recorded and
// written to FrameWarp\logs\ui_probe_<n>.txt - per command list in the order they were submitted: which render target each
// group of draws went to (size, format, the swapchain's back buffer or not, with or without a depth buffer), the
// dispatches, copies and clears between them, and where the upscaler ran. To find where a game starts drawing its HUD,
// so that a HUD-less copy of the picture could be taken there.
#pragma once
#include <cstdint>

namespace reshade::api { struct swapchain; }

namespace fw::ui_probe {
void init();
void shutdown();
// Every present of the game's swapchain: the key, the frame boundaries, the file.
void on_present(reshade::api::swapchain* chain, bool keys_on);
// A mark in a command list (the native one: the upscaler's hooks see that), e.g. "DLSS evaluate".
void mark(std::uint64_t native_cmd_list, const char* what);
}
