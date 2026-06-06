#pragma once
#include "scene/scene.h"
#include "output/image.h"

// Run the GPU wavefront path tracer.
// use_optix=false (default): CUDA software BVH with ray sorting.
// use_optix=true:            OptiX RT-core intersection (requires ENABLE_OPTIX build).
void render_wavefront_gpu(const SceneData& scene, ImageBuffer& img,
                          bool use_optix = false);
