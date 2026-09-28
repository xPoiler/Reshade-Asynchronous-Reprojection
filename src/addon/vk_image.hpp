#pragma once
#include <cstdint>

namespace fw {

// A shared texture imported into a Vulkan game's device (Vulkan handles kept as integers, so that code
// which never touches Vulkan does not need its headers).
struct VkSharedImage {
    std::uint64_t image = 0;         // VkImage
    std::uint64_t memory = 0;        // VkDeviceMemory
    std::uint64_t staging = 0;       // reshade::api::resource (buffer) for depth copies
    std::uint32_t depth_bytes = 0;   // > 0: filled from a depth image through the staging buffer
};

}  // namespace fw
