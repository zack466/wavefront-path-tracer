cmake_minimum_required(VERSION 3.20)
project(wavefront_path_tracer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /wd4244)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    add_compile_options("$<$<CONFIG:Release>:-O3>")
    add_compile_options("$<$<CONFIG:Debug>:-g -O0>")
endif()

set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendored/json/single_include
    ${CMAKE_SOURCE_DIR}/vendored/stb
)

# ── CPU target ─────────────────────────────────────────────────────────────────
set(CPU_SOURCES
    src/main_cpu.cpp
    src/scene/loader.cpp
    src/scene/mesh_loader.cpp
    src/bvh/bvh.cpp
    src/output/image.cpp
    src/cpu/render_cpu.cpp
    src/cpu/wavefront_cpu.cpp
)

add_executable(path_tracer_cpu ${CPU_SOURCES})
target_include_directories(path_tracer_cpu PRIVATE ${COMMON_INCLUDE_DIRS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(path_tracer_cpu PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — parallel CPU rendering enabled")
else()
    message(STATUS "OpenMP not found — single-threaded CPU rendering")
endif()

# ── GPU target (Phase 3) ──────────────────────────────────────────────────────
# The GPU wavefront backend lives on a separate branch.  Each wavefront kernel
# here (generate / intersect / miss / shade) maps to a CUDA __global__ function
# launched via cudaLaunch (CUDA BVH path) or optixLaunch (OptiX RT-core path).
# Queue compaction on the GPU uses warp-ballot (__ballot_sync + atomicAdd)
# instead of the sequential compact_field() memmoves used here on the CPU.
