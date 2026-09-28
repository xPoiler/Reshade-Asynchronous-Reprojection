#pragma once
#include "addon/producer.hpp"

namespace fw {
// Hooks sl.interposer.dll (slSetConstants / slSetTag / slSetTagForFrame) and the PCL latency
// marker function. Safe to call repeatedly: installs whatever is missing once Streamline is loaded.
void install_streamline_hooks(Producer* producer);
bool streamline_loaded();
// The game sent Streamline camera data within the last second (switching to FSR stops it in some games).
bool streamline_camera_recent();
// ...and depth within the last second (some games tag depth and motion vectors only while DLSS is on).
bool streamline_depth_recent();
// The frame of the newest Streamline camera data.
std::uint64_t streamline_current_frame();
// Asks Streamline which features are loaded (results go to HookStats). Call from the present thread.
void probe_streamline_features();
// Names for the UI.
extern const char* const kCountedExports[];
extern const int kCountedExportCount;
extern const std::uint32_t kProbedFeatures[];
extern const int kProbedFeatureCount;
}
