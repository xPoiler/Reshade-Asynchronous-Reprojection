// Upscaler passes compiled into the game itself (Resident Evil 4: FSR 2 built into the engine as its own compute
// passes - no FidelityFX DLL to hook). Step 1, finding out what the game's shaders carry: every compute shader the game
// creates is searched for the resource names FSR 2's passes use (AMD's own and the engine's), and what is found is
// logged once per shader - its pass, the names, and the container's parts (does the bytecode keep its reflection).
#pragma once

namespace fw::shader_scan {
void init();
void shutdown();
}
