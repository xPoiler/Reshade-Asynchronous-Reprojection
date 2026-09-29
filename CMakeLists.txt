cmake_minimum_required(VERSION 3.24)
project(FrameWarp VERSION 1.6.0 LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "FrameWarp requires Windows, MSVC and x64.")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LATEWARP_DLL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/latewarp/nvngx_latewarp.dll" CACHE FILEPATH
    "NVIDIA Reflex 2 Frame Warp DLL (not redistributable; copied next to the presenter)")

# NVIDIA NGX SDK (not included): a checkout of https://github.com/NVIDIA/DLSS, or a folder holding its
# headers and nvsdk_ngx_d.lib side by side.
set(FW_NGX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DLSS" CACHE PATH "NVIDIA DLSS SDK checkout (github.com/NVIDIA/DLSS)")
if(NOT EXISTS "${FW_NGX_DIR}/include/nvsdk_ngx.h" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ngx/nvsdk_ngx.h")
  set(FW_NGX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ngx")
endif()
find_path(FW_NGX_INCLUDE nvsdk_ngx.h PATHS "${FW_NGX_DIR}/include" "${FW_NGX_DIR}" NO_DEFAULT_PATH NO_CACHE)
find_file(FW_NGX_LIB nvsdk_ngx_d.lib PATHS "${FW_NGX_DIR}/lib/Windows_x86_64/x64" "${FW_NGX_DIR}" NO_DEFAULT_PATH NO_CACHE)
if(NOT FW_NGX_INCLUDE OR NOT FW_NGX_LIB)
  message(FATAL_ERROR "NVIDIA NGX SDK not found. Run\n  git clone --depth 1 https://github.com/NVIDIA/DLSS third_party/DLSS\n"
                      "or pass -DFW_NGX_DIR=<path to a DLSS SDK checkout>.")
endif()

add_library(fw_options INTERFACE)
target_compile_options(fw_options INTERFACE /W4 /permissive- /EHsc /utf-8 /MP)
target_compile_definitions(fw_options INTERFACE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE FW_VERSION="${PROJECT_VERSION}")
target_include_directories(fw_options INTERFACE src)

# In-game ReShade add-on (static CRT so it has no runtime dependency inside the game).
add_library(FrameWarpAddon SHARED src/addon/addon.cpp src/addon/producer.cpp src/addon/vk_transport.cpp src/addon/streamline_hooks.cpp
            src/addon/ngx_hooks.cpp src/addon/ffx_hooks.cpp src/addon/game_probe.cpp src/common/inline_hook.cpp)
target_link_libraries(FrameWarpAddon PRIVATE fw_options d3d12 dxgi)
# NGX headers only (no library); Khronos Vulkan headers for Vulkan games.
target_include_directories(FrameWarpAddon SYSTEM PRIVATE third_party/reshade/include third_party/imgui third_party/vulkan/include "${FW_NGX_INCLUDE}")
set_target_properties(FrameWarpAddon PROPERTIES OUTPUT_NAME "FrameWarp" PREFIX "" SUFFIX ".addon64"
                      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Out-of-process presenter (dynamic CRT to match NVIDIA's nvsdk_ngx_d.lib).
add_library(fw_presenter_core STATIC src/presenter/renderer.cpp src/presenter/latewarp12.cpp)
target_link_libraries(fw_presenter_core PUBLIC fw_options d3d12 dxgi dcomp d3dcompiler
                      "${FW_NGX_LIB}" advapi32 shlwapi shell32 ole32)
target_include_directories(fw_presenter_core SYSTEM PUBLIC "${FW_NGX_INCLUDE}")
set_target_properties(fw_presenter_core PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

add_executable(FrameWarpPresenter WIN32 src/presenter/main.cpp)
target_link_libraries(FrameWarpPresenter PRIVATE fw_presenter_core user32 pdh)
set_target_properties(FrameWarpPresenter PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/FrameWarp")
if(EXISTS "${LATEWARP_DLL}")
  add_custom_command(TARGET FrameWarpPresenter POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LATEWARP_DLL}" "$<TARGET_FILE_DIR:FrameWarpPresenter>")
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(pose_tests tests/pose_tests.cpp)
  target_link_libraries(pose_tests PRIVATE fw_options)
  add_test(NAME pose COMMAND pose_tests)

  add_executable(motion_tests tests/motion_tests.cpp)
  target_link_libraries(motion_tests PRIVATE fw_options)
  add_test(NAME camera_motion COMMAND motion_tests)

  add_executable(estimator_tests tests/estimator_tests.cpp)
  target_link_libraries(estimator_tests PRIVATE fw_options)
  add_test(NAME camera_estimator COMMAND estimator_tests)

  add_executable(hook_tests tests/hook_tests.cpp src/common/inline_hook.cpp)
  target_link_libraries(hook_tests PRIVATE fw_options)
  add_test(NAME inline_hook COMMAND hook_tests)

  # End-to-end on the real GPU: producer on one device, presenter renderer + Latewarp on another.
  add_executable(pipeline_tests tests/pipeline_tests.cpp src/addon/producer.cpp src/addon/vk_transport.cpp)
  target_link_libraries(pipeline_tests PRIVATE fw_presenter_core user32 dxgi)
  target_include_directories(pipeline_tests SYSTEM PRIVATE third_party/reshade/include third_party/vulkan/include)
  set_target_properties(pipeline_tests PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/FrameWarp")
  add_test(NAME pipeline COMMAND pipeline_tests WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/FrameWarp")
  set_tests_properties(pipeline PROPERTIES TIMEOUT 60)
endif()

install(TARGETS FrameWarpAddon RUNTIME DESTINATION .)
install(TARGETS FrameWarpPresenter RUNTIME DESTINATION FrameWarp)
if(EXISTS "${LATEWARP_DLL}")
  install(FILES "${LATEWARP_DLL}" DESTINATION FrameWarp)
endif()

# Optional local additions (not part of the repository).
include(dev/dev.cmake OPTIONAL)
