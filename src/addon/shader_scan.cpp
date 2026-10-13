#include "addon/shader_scan.hpp"
#include <reshade.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

using namespace reshade::api;

namespace fw::shader_scan {
namespace {
// FSR 2's resource names: AMD's SDK passes, and the names an engine that rewrote the passes keeps (RE Engine).
const char* const kNames[] = {"r_input_motion_vectors", "r_motion_vectors", "r_input_depth", "r_depth", "r_dilated_motion_vectors",
                              "rw_dilated_motion_vectors", "rw_reconstructed_previous_nearest_depth", "r_input_color_jittered",
                              "rw_upscaled_output", "rw_internal_upscaled_color", "inputVelocity", "outputVelocity", "outputDepth",
                              "r_exposure", "rw_lock_status"};
std::atomic<int> g_logged{0};
std::atomic<int> g_compute{0};

bool contains(const std::uint8_t* data, std::size_t size, const char* name) {
    const std::size_t n = std::strlen(name);
    if (size < n) return false;
    return std::search(data, data + size, name, name + n) != data + size;
}
// The container's parts ("DXBC" header, then the chunk table): DXIL, RDAT / STAT / RDEF (reflection), PSV0, ...
std::string parts(const std::uint8_t* data, std::size_t size) {
    if (size < 32 || std::memcmp(data, "DXBC", 4) != 0) return size >= 4 && *reinterpret_cast<const std::uint32_t*>(data) == 0x07230203u ? "SPIR-V" : "?";
    std::uint32_t count = 0;
    std::memcpy(&count, data + 28, 4);
    std::string out;
    for (std::uint32_t i = 0; i < count && i < 16 && 32 + 4 * (i + 1) <= size; ++i) {
        std::uint32_t offset = 0;
        std::memcpy(&offset, data + 32 + 4 * i, 4);
        if (offset + 8 > size) break;
        if (!out.empty()) out += ' ';
        out.append(reinterpret_cast<const char*>(data + offset), 4);
    }
    return out;
}

void on_init_pipeline(device*, pipeline_layout, uint32_t count, const pipeline_subobject* subobjects, pipeline p) {
    for (uint32_t i = 0; i < count; ++i) {
        if (subobjects[i].type != pipeline_subobject_type::compute_shader || !subobjects[i].data) continue;
        const auto* shader = static_cast<const shader_desc*>(subobjects[i].data);
        if (!shader->code || !shader->code_size) continue;
        ++g_compute;
        const auto* data = static_cast<const std::uint8_t*>(shader->code);
        std::string found;
        for (const char* name : kNames)
            if (contains(data, shader->code_size, name)) { if (!found.empty()) found += ", "; found += name; }
        if (found.empty() || g_logged.fetch_add(1) >= 60) continue;
        char text[700];
        std::snprintf(text, sizeof(text), "XPAR shader scan: compute pipeline %llx (%zu bytes, parts: %s, entry %s) uses %s",
                      static_cast<unsigned long long>(p.handle), shader->code_size, parts(data, shader->code_size).c_str(),
                      shader->entry_point ? shader->entry_point : "-", found.c_str());
        reshade::log::message(reshade::log::level::info, text);
    }
}
void on_destroy_device(device*) {
    if (g_compute.load()) {
        char text[160];
        std::snprintf(text, sizeof(text), "XPAR shader scan: %d compute shaders searched, %d with FSR 2 resource names", g_compute.load(), g_logged.load());
        reshade::log::message(reshade::log::level::info, text);
    }
}
}  // namespace

void init() {
    reshade::register_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
    reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
}
void shutdown() {
    reshade::unregister_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
    reshade::unregister_event<reshade::addon_event::destroy_device>(on_destroy_device);
}
}  // namespace fw::shader_scan
