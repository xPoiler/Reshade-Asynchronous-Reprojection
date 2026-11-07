#pragma once
// The development keys of ReShade.ini's [FrameWarp], each with the environment variable it reaches the presenter by. One table,
// read by the add-on (launch_presenter) and by dev/seq_replay, so a replay runs with what the game runs with: the replays were
// written by hand and fell behind the game's keys more than once, and a day of results went with them.
namespace fw {

struct DevKey {
    const char* ini;      // the key
    const char* env;      // its variable
    const wchar_t* wenv;  // ...for SetEnvironmentVariableW
    char highest;         // the values it takes: '1' to this digit, passed on as they are; anything else leaves the variable unset
    bool passes_zero;     // ...'0' passed on too, not left unset (a key whose variable tells on from off by its value)
};

inline constexpr DevKey kDevKeys[] = {
    {"ClaimsReuse", "FW_CLAIMS_REUSE", L"FW_CLAIMS_REUSE", '1', true},
    {"BesideMoved", "FW_BESIDE_MOVED", L"FW_BESIDE_MOVED", '4', false},
    {"MemoryTwoLayers", "FW_MEM_TWO", L"FW_MEM_TWO", '1', false},
    {"CarriedFree", "FW_CARRIED_FREE", L"FW_CARRIED_FREE", '1', false},
    {"MemoryPositions", "FW_MEM_POS", L"FW_MEM_POS", '1', false},
    {"AttachedEstimated", "FW_ATT_EST", L"FW_ATT_EST", '1', false},
    {"ClaimsBesideHeld", "FW_CLAIMS_BESIDE", L"FW_CLAIMS_BESIDE", '1', false},
    {"AttachedAbsolute", "FW_ATT_ABS", L"FW_ATT_ABS", '1', false},
    {"EdgeDitherNoBand", "FW_DITHER_NO_BAND", L"FW_DITHER_NO_BAND", '1', false},
    {"MaskSmooth", "FW_MASK_SMOOTH", L"FW_MASK_SMOOTH", '1', false},
    {"DitherHoles", "FW_DITHER_HOLES", L"FW_DITHER_HOLES", '1', false},
    {"CameraUse", "FW_CAMERA_USE", L"FW_CAMERA_USE", '1', true},
    {"IntakeBetweenWarps", "FW_INTAKE_GATE", L"FW_INTAKE_GATE", '1', true},  // (on by default: IntakeBetweenWarps=0 lets the intake run alongside the warps, as before - Renderer::set_intake_gate)  // (on by default since 2.2: CameraUse=0 turns it off)
    {"CameraDump", "FW_CAMERA_DUMP", L"FW_CAMERA_DUMP", '2', false},  // (2: object matrices too, every frame searched - see camera_search.cpp g_dump)
    {"StencilDump", "FW_STENCIL_DUMP", L"FW_STENCIL_DUMP", '1', false},
    {"MotionReference", "FW_MOTION_REF", L"FW_MOTION_REF", '1', false},  // (what moves with the screen by the game's vectors - Renderer::set_motion_reference)
    {"StencilMask", "FW_STENCIL_MASK", L"FW_STENCIL_MASK", '1', false},
    {"ContinuousWarp", "FW_TETHER", L"FW_TETHER", '1', true},  // (on by default since 2026-11-07: ContinuousWarp=0 brings back the old holding - Renderer::set_tether)  // (the character from the game's stencil - Renderer::stencil_character)  // (the depth's stencil, every 60th frame - see Producer::stencil_dump)
};

// The variable's value for a key's first character (0: not set).
inline char dev_key_value(const DevKey& k, char c) {
    if (c == '0' && k.passes_zero) return '0';
    return c >= '1' && c <= k.highest ? c : 0;
}

}  // namespace fw
