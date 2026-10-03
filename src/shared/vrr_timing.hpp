#pragma once

#include <cmath>
#include <cstdint>

namespace fw {

enum class VrrTargetMode : std::uint32_t { disabled = 0, automatic = 1, manual = 2 };

// Issue #3's automatic cap, interpreting its trailing "-1%" as one percent of panel max Hz.
inline double automatic_vrr_target_hz(double panel_max_hz) noexcept {
    if (!std::isfinite(panel_max_hz) || panel_max_hz <= 1.0) return 0.0;
    const double target = panel_max_hz - panel_max_hz * panel_max_hz / 3600.0 - panel_max_hz * 0.01;
    return std::isfinite(target) && target > 1.0 ? target : 0.0;
}

inline double resolve_vrr_target_hz(VrrTargetMode mode, double display_mode_hz, double panel_max_override_hz,
                                    double manual_target_hz) noexcept {
    switch (mode) {
    case VrrTargetMode::automatic: {
        const double panel_max_hz = panel_max_override_hz > 1.0 ? panel_max_override_hz : display_mode_hz;
        return automatic_vrr_target_hz(panel_max_hz);
    }
    case VrrTargetMode::manual:
        return std::isfinite(manual_target_hz) && manual_target_hz > 1.0 && manual_target_hz <= 1000.0
                   ? manual_target_hz
                   : 0.0;
    default:
        return 0.0;
    }
}

struct VrrSchedule {
    double target_hz;
    double period_qpc;
};

inline VrrSchedule resolve_vrr_schedule(VrrTargetMode mode, double display_mode_hz, double panel_max_override_hz,
                                        double manual_target_hz, double qpc_frequency,
                                        double legacy_period_qpc) noexcept {
    const double target_hz = resolve_vrr_target_hz(mode, display_mode_hz, panel_max_override_hz, manual_target_hz);
    const double period_qpc = target_hz > 1.0 && std::isfinite(qpc_frequency) && qpc_frequency > 0.0
                                  ? qpc_frequency / target_hz
                                  : legacy_period_qpc;
    return {target_hz, period_qpc};
}

}  // namespace fw
