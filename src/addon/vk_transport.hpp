#pragma once
// Vulkan games: the add-on keeps creating its shared textures with D3D12 (a device of its own on the game's
// GPU), exactly what the presenter opens, and imports them into the game's Vulkan device. Copies are
// recorded into the game's command buffers; completion is a Vulkan timeline semaphore that a small thread
// turns into a CPU signal of the shared D3D12 fence (the presenter only takes frames whose fence completed).
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include "addon/vk_image.hpp"
#include <dxgiformat.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace reshade::api { struct device; struct command_queue; }

namespace fw {

// The DXGI format a Vulkan image is shared as (the bits are copied unchanged). DXGI_FORMAT_UNKNOWN when not
// supported. Depth formats are shared as colour holding their depth bits (depth_bytes: 4 or 2).
DXGI_FORMAT dxgi_format_of(VkFormat format, std::uint32_t* depth_bytes = nullptr);
// The Vulkan format of a shared texture of this DXGI format (for importing it).
VkFormat vk_format_of(DXGI_FORMAT format);
// Depth-stencil formats (their layout transitions must name both aspects).
bool has_stencil(VkFormat format);

class VkTransport {
public:
    ~VkTransport();
    bool init(reshade::api::device* device, std::string& error);
    VkDevice device() const { return device_; }

    // Imports a shared D3D12 texture (NT handle) as a Vulkan image of `format`. depth_bytes > 0: it is filled
    // from a depth image, through a staging buffer.
    bool import(HANDLE handle, VkFormat format, std::uint32_t width, std::uint32_t height, std::uint32_t depth_bytes, VkSharedImage& out,
                std::string& error);
    void destroy(VkSharedImage& image);
    // Records a copy of `source` (in `layout`; `aspect`, mip and layer of the view) into `target`.
    void copy(VkCommandBuffer cb, VkImage source, VkImageAspectFlags barrier_aspect, std::uint32_t mip, std::uint32_t layer,
              VkImageLayout layout, const VkSharedImage& target, std::uint32_t width, std::uint32_t height);

    // After the frame's copies: signals `value` on the game's queue; `on_complete(value)` runs on the
    // completion thread once the GPU has passed it.
    void signal(reshade::api::command_queue* queue, std::uint64_t value);
    void set_on_complete(std::function<void(std::uint64_t)> f) { on_complete_ = std::move(f); }

private:
    void completion_thread();

    reshade::api::device* rs_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    std::uint64_t fence_ = 0;  // reshade::api::fence (a timeline semaphore)
    PFN_vkCreateImage CreateImage = nullptr;
    PFN_vkDestroyImage DestroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements = nullptr;
    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkBindImageMemory BindImageMemory = nullptr;
    PFN_vkGetMemoryWin32HandlePropertiesKHR GetMemoryWin32HandleProperties = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyImage CmdCopyImage = nullptr;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer = nullptr;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage = nullptr;

    std::function<void(std::uint64_t)> on_complete_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::uint64_t> pending_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

}  // namespace fw
