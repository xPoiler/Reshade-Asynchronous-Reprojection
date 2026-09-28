#include "addon/vk_transport.hpp"
#include <reshade_api.hpp>
#include <windows.h>

namespace fw {
using namespace reshade::api;

DXGI_FORMAT dxgi_format_of(VkFormat f, std::uint32_t* depth_bytes) {
    std::uint32_t depth = 0;
    DXGI_FORMAT out = DXGI_FORMAT_UNKNOWN;
    switch (f) {
        case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: depth = 4; out = DXGI_FORMAT_R32_FLOAT; break;
        case VK_FORMAT_D16_UNORM: case VK_FORMAT_D16_UNORM_S8_UINT: depth = 2; out = DXGI_FORMAT_R16_UNORM; break;
        case VK_FORMAT_R32_SFLOAT: out = DXGI_FORMAT_R32_FLOAT; break;
        case VK_FORMAT_R16G16_SFLOAT: out = DXGI_FORMAT_R16G16_FLOAT; break;
        case VK_FORMAT_R32G32_SFLOAT: out = DXGI_FORMAT_R32G32_FLOAT; break;
        case VK_FORMAT_R16G16B16A16_SFLOAT: out = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: out = DXGI_FORMAT_R11G11B10_FLOAT; break;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: out = DXGI_FORMAT_R10G10B10A2_UNORM; break;
        // sRGB images are shared with their raw bits (as the D3D12 path reads them).
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: out = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: out = DXGI_FORMAT_B8G8R8A8_UNORM; break;
        default: break;
    }
    if (depth_bytes) *depth_bytes = depth;
    return out;
}

VkFormat vk_format_of(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case DXGI_FORMAT_R16_UNORM: return VK_FORMAT_R16_UNORM;
        case DXGI_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
        case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case DXGI_FORMAT_R11G11B10_FLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        default: return VK_FORMAT_UNDEFINED;
    }
}

bool has_stencil(VkFormat f) {
    return f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT;
}

VkTransport::~VkTransport() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (rs_ && fence_) rs_->destroy_fence(fence{fence_});
}

bool VkTransport::init(reshade::api::device* rs, std::string& error) {
    rs_ = rs;
    device_ = reinterpret_cast<VkDevice>(rs->get_native());
    HMODULE loader = GetModuleHandleW(L"vulkan-1.dll");
    auto get = loader ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(loader, "vkGetDeviceProcAddr")) : nullptr;
    if (!get || !device_) { error = "Vulkan: no vkGetDeviceProcAddr"; return false; }
    auto load = [&](auto& fn, const char* name) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(get(device_, name)); return fn != nullptr; };
    if (!load(CreateImage, "vkCreateImage") || !load(DestroyImage, "vkDestroyImage") ||
        !load(GetImageMemoryRequirements, "vkGetImageMemoryRequirements") || !load(AllocateMemory, "vkAllocateMemory") ||
        !load(FreeMemory, "vkFreeMemory") || !load(BindImageMemory, "vkBindImageMemory") || !load(CmdPipelineBarrier, "vkCmdPipelineBarrier") ||
        !load(CmdCopyImage, "vkCmdCopyImage") || !load(CmdCopyImageToBuffer, "vkCmdCopyImageToBuffer") ||
        !load(CmdCopyBufferToImage, "vkCmdCopyBufferToImage")) {
        error = "Vulkan: device functions missing"; return false;
    }
    // (enabled by ReShade on the game's device when the driver offers it)
    if (!load(GetMemoryWin32HandleProperties, "vkGetMemoryWin32HandlePropertiesKHR")) {
        error = "Vulkan: VK_KHR_external_memory_win32 is not enabled on the game's device"; return false;
    }
    fence f{};
    if (!rs->create_fence(0, fence_flags::none, &f)) { error = "Vulkan: timeline semaphore not available"; return false; }
    fence_ = f.handle;
    thread_ = std::thread(&VkTransport::completion_thread, this);
    return true;
}

bool VkTransport::import(HANDLE handle, VkFormat format, std::uint32_t width, std::uint32_t height, std::uint32_t depth_bytes,
                         VkSharedImage& out, std::string& error) {
    out = VkSharedImage{};
    VkExternalMemoryImageCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.pNext = &external;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    if (CreateImage(device_, &ci, nullptr, &image) != VK_SUCCESS) { error = "Vulkan: vkCreateImage (external) failed"; return false; }
    VkMemoryRequirements req{};
    GetImageMemoryRequirements(device_, image, &req);
    std::uint32_t types = req.memoryTypeBits;
    VkMemoryWin32HandlePropertiesKHR props{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
    if (GetMemoryWin32HandleProperties(device_, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, handle, &props) == VK_SUCCESS &&
        (types & props.memoryTypeBits))
        types &= props.memoryTypeBits;
    std::uint32_t type = 0;
    while (type < 32 && !(types & (1u << type))) ++type;
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.image = image;
    VkImportMemoryWin32HandleInfoKHR import_info{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    import_info.pNext = &dedicated;
    import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    import_info.handle = handle;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &import_info;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (type >= 32 || AllocateMemory(device_, &ai, nullptr, &memory) != VK_SUCCESS) {
        DestroyImage(device_, image, nullptr);
        error = "Vulkan: importing the shared texture failed"; return false;
    }
    if (BindImageMemory(device_, image, memory, 0) != VK_SUCCESS) {
        FreeMemory(device_, memory, nullptr); DestroyImage(device_, image, nullptr);
        error = "Vulkan: vkBindImageMemory failed"; return false;
    }
    out.image = reinterpret_cast<std::uint64_t>(image);
    out.memory = reinterpret_cast<std::uint64_t>(memory);
    if (depth_bytes) {
        // Depth reaches a colour image through a buffer (image copies cannot change the aspect).
        resource buffer{};
        const resource_desc bd(std::uint64_t(width) * height * depth_bytes, memory_heap::default_,
                               resource_usage::copy_source | resource_usage::copy_dest);
        if (!rs_->create_resource(bd, nullptr, resource_usage::copy_dest, &buffer)) {
            destroy(out);
            error = "Vulkan: staging buffer for depth failed"; return false;
        }
        out.staging = buffer.handle;
        out.depth_bytes = depth_bytes;
    }
    return true;
}

void VkTransport::destroy(VkSharedImage& image) {
    if (image.staging) rs_->destroy_resource(resource{image.staging});
    if (image.image) DestroyImage(device_, reinterpret_cast<VkImage>(image.image), nullptr);
    if (image.memory) FreeMemory(device_, reinterpret_cast<VkDeviceMemory>(image.memory), nullptr);
    image = VkSharedImage{};
}

void VkTransport::copy(VkCommandBuffer cb, VkImage source, VkImageAspectFlags barrier_aspect, std::uint32_t mip, std::uint32_t layer,
                       VkImageLayout layout, const VkSharedImage& target, std::uint32_t width, std::uint32_t height) {
    const auto dst = reinterpret_cast<VkImage>(target.image);
    const bool depth = target.depth_bytes > 0;
    const VkImageAspectFlags copy_aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    // The source stays in its layout when a copy can read it there.
    const bool transition = layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && layout != VK_IMAGE_LAYOUT_GENERAL;
    const VkImageLayout read_layout = transition ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : layout;

    VkImageMemoryBarrier before[2]{};
    before[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    before[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    before[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    before[0].oldLayout = layout; before[0].newLayout = read_layout;
    before[0].srcQueueFamilyIndex = before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before[0].image = source;
    before[0].subresourceRange = {barrier_aspect, mip, 1, layer, 1};
    before[1] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    before[1].srcAccessMask = 0;
    before[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    before[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // overwritten entirely
    before[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    before[1].srcQueueFamilyIndex = before[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before[1].image = dst;
    before[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, before);

    if (!depth) {
        VkImageCopy region{};
        region.srcSubresource = {copy_aspect, mip, layer, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {width, height, 1};
        CmdCopyImage(cb, source, read_layout, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    } else {
        const auto buffer = reinterpret_cast<VkBuffer>(target.staging);
        VkBufferImageCopy to_buffer{};
        to_buffer.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, mip, layer, 1};
        to_buffer.imageExtent = {width, height, 1};
        CmdCopyImageToBuffer(cb, source, read_layout, buffer, 1, &to_buffer);
        VkBufferMemoryBarrier written{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        written.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        written.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        written.srcQueueFamilyIndex = written.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        written.buffer = buffer;
        written.size = VK_WHOLE_SIZE;
        CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &written, 0, nullptr);
        VkBufferImageCopy to_image{};
        to_image.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        to_image.imageExtent = {width, height, 1};
        CmdCopyBufferToImage(cb, buffer, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &to_image);
    }

    VkImageMemoryBarrier after[2]{before[0], before[1]};
    after[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    after[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    after[0].oldLayout = read_layout; after[0].newLayout = layout;
    after[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    after[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    after[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;  // what the presenter's D3D12 device reads it as (COMMON)
    CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, after);
}

void VkTransport::signal(command_queue* queue, std::uint64_t value) {
    if (!queue || !fence_ || !queue->signal(fence{fence_}, value)) return;
    {
        std::lock_guard lock(mutex_);
        pending_.push_back(value);
    }
    wake_.notify_one();
}

void VkTransport::completion_thread() {
    for (;;) {
        std::uint64_t value = 0;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || !pending_.empty(); });
            if (stop_) return;
            value = pending_.front();
        }
        // (bounded waits, so that shutting down never hangs on a lost device)
        while (!stop_ && !rs_->wait(fence{fence_}, value, 100'000'000ull)) {}
        if (stop_) return;
        {
            std::lock_guard lock(mutex_);
            pending_.pop_front();
        }
        if (on_complete_) on_complete_(value);
    }
}

}  // namespace fw
