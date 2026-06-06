#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "optix_params.h"
#include "device_scene.h"
#include "common/queues.h"

// Host-side OptiX state: context, pipeline, SBT, single triangle GAS.
// When the OptiX backend is active, all analytic shapes are tessellated into
// triangles and baked into the same triangle GAS as the scene OBJ meshes.
struct OptixState {
    OptixDeviceContext      context    = nullptr;
    OptixPipeline           pipeline   = nullptr;
    OptixShaderBindingTable sbt        = {};

    OptixTraversableHandle  traversable = 0;
    CUdeviceptr             d_gas_tri_buf = 0;  // single triangle GAS

    CUdeviceptr d_sbt_raygen        = 0;
    CUdeviceptr d_sbt_raygen_shadow = 0;
    CUdeviceptr d_sbt_miss          = 0;
    CUdeviceptr d_sbt_hitgrp        = 0;

    // Tessellated analytic shape triangles (device pointer to Triangle[])
    CUdeviceptr d_tess_triangles    = 0;
    int         num_tess_triangles  = 0;
    int         num_scene_triangles = 0;

    OptixShaderBindingTable sbt_shadow = {};  // shadow raygen SBT variant

    LaunchParams* d_params = nullptr;
};

// Build OptiX state from an already-uploaded DeviceScene.
OptixState build_optix_state(const DeviceScene& ds, const char* ptx_source);

void destroy_optix_state(OptixState& state);

// Primary intersection pass (replaces intersect_kernel).
// Misses are accumulated directly into image_buf by __miss__primary.
void optix_intersect(
    OptixState&        state,
    const DeviceScene& ds,
    RayQueue&          ray,
    HitQueue&          hit,   int* d_hc,
    float*             image_buf);

// Shadow pass (replaces gpu_shadow_blocked in the shade kernel).
// Reads shadow ray candidates from the ShadowQueue and, for each unblocked
// ray, adds the pre-scaled contribution to image_buf.
void optix_shadow(
    OptixState&  state,
    float* sh_ox, float* sh_oy, float* sh_oz,
    float* sh_dx, float* sh_dy, float* sh_dz,
    float* sh_tmax,
    float* sh_cr, float* sh_cg, float* sh_cb,
    int*   sh_pidx,
    int    sh_count,
    float* image_buf);
