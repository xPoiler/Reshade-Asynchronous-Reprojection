// shader_bake <pack file> [<cache folder>]: every presenter shader compiled into the pack the presenter carries built in
// (FW_SHADERS - see Renderer::bake_shaders). Run by the build before the presenter is linked; with a cache folder only the
// shaders that changed since the last build are compiled again.
#include "renderer.hpp"

using fw::Renderer;

#include <cstdio>
#include <string>

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) { std::printf("usage: shader_bake <pack file> [<cache folder>]\n"); return 2; }
    if (argc > 2) Renderer::set_shader_cache(argv[2]);
    std::string report;
    const bool ok = Renderer::bake_shaders(argv[1], report);
    std::printf("%s\n", report.c_str());
    if (!ok) std::printf("shader_bake: a shader did not compile, or the pack could not be written\n");
    return ok ? 0 : 1;
}
