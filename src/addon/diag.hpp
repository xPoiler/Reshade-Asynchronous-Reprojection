#pragma once
// Detailed diagnostics on the add-on's side (Settings::record_diagnostics - 2026-11-10, user: a shipped build debugs nothing and
// writes nothing unless "Record detailed diagnostics" is on; [FrameWarp] DiagnosticsOn=1 turns it on in the test games): every
// line the add-on writes to ReShade.log goes through diag_log, written only while it is on. Crashes are logged always
// (fault_log.cpp). Set from the shared settings at load and on every present (addon.cpp).
#include <atomic>
#include <reshade.hpp>

namespace fw {
inline std::atomic<bool> g_diag{false};
inline bool diag() { return g_diag.load(std::memory_order_relaxed); }
inline void diag_log(reshade::log::level level, const char* text) {
    if (diag()) reshade::log::message(level, text);
}
}  // namespace fw
