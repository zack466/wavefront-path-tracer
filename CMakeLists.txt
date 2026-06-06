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

# ── GPU target ────────────────────────────────────────────────────────────────
# Build:  cmake -B build -DENABLE_CUDA=ON && cmake --build build -j
# OptiX:  cmake -B build -DENABLE_CUDA=ON -DENABLE_OPTIX=ON && cmake --build build -j
#
# ENABLE_CUDA  — CUDA software BVH wavefront (always available when ON)
# ENABLE_OPTIX — adds OptiX RT-core backend as an optional --optix flag
#                (requires OptiX 8.x headers in vendored/optix/ and driver ≥ 555)
option(ENABLE_CUDA  "Build GPU wavefront backend (CUDA software BVH)"    OFF)
option(ENABLE_OPTIX "Add OptiX RT-core intersection backend (requires ENABLE_CUDA)" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")

    set(GPU_SOURCES
        src/main_gpu.cu
        src/scene/loader.cpp
        src/scene/mesh_loader.cpp
        src/bvh/bvh.cpp
        src/output/image.cpp
        src/gpu/device_scene.cu
        src/gpu/kernels.cu
        src/gpu/render_gpu.cu
    )

    if(ENABLE_OPTIX)
        set(OPTIX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/vendored/optix")
        if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
            message(FATAL_ERROR
                "OptiX 8.x headers not found in ${OPTIX_INCLUDE_DIR}. "
                "Copy the SDK's include/ directory there.")
        endif()
        message(STATUS "OptiX headers: ${OPTIX_INCLUDE_DIR}")

        # Compile OptiX device programs to PTX (loaded at runtime)
        set(OPTIX_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/optix_programs.ptx")
        add_custom_command(
            OUTPUT "${OPTIX_PTX_FILE}"
            COMMAND ${CMAKE_CUDA_COMPILER}
                -ptx
                -arch sm_86
                --expt-relaxed-constexpr
                -I "${CMAKE_SOURCE_DIR}/src"
                -I "${OPTIX_INCLUDE_DIR}"
                -o "${OPTIX_PTX_FILE}"
                "${CMAKE_SOURCE_DIR}/src/gpu/optix_programs.cu"
            DEPENDS
                "${CMAKE_SOURCE_DIR}/src/gpu/optix_programs.cu"
                "${CMAKE_SOURCE_DIR}/src/gpu/optix_params.h"
                "${CMAKE_SOURCE_DIR}/src/common/intersect.h"
                "${CMAKE_SOURCE_DIR}/src/common/math.h"
            COMMENT "Compiling OptiX device programs → optix_programs.ptx"
        )
        add_custom_target(optix_ptx_target DEPENDS "${OPTIX_PTX_FILE}")

        list(APPEND GPU_SOURCES src/gpu/optix_state.cu)
    endif()

    add_executable(path_tracer_gpu ${GPU_SOURCES})
    target_include_directories(path_tracer_gpu PRIVATE ${COMMON_INCLUDE_DIRS})
    target_link_libraries(path_tracer_gpu PRIVATE CUDA::cudart)
    set_target_properties(path_tracer_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86"
    )
    target_compile_options(path_tracer_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -lineinfo
        >
    )

    if(ENABLE_OPTIX)
        target_include_directories(path_tracer_gpu PRIVATE "${OPTIX_INCLUDE_DIR}")
        target_compile_definitions(path_tracer_gpu PRIVATE
            ENABLE_OPTIX=1
            OPTIX_PTX_FILE="${OPTIX_PTX_FILE}"
        )
        target_link_libraries(path_tracer_gpu PRIVATE CUDA::cuda_driver)
        add_dependencies(path_tracer_gpu optix_ptx_target)
        message(STATUS "OptiX RT-core backend ENABLED")
    else()
        message(STATUS "OptiX RT-core backend disabled (ENABLE_OPTIX=OFF)")
    endif()
endif()
