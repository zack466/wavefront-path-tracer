#pragma once
#include "scene/scene.h"
#include "output/image.h"

// Renders `scene` into `img` using the wavefront (queue-based) path tracing
// algorithm.  Each bounce is processed as a flat kernel over all active rays
// rather than recursively — the same structure that will be mapped to GPU
// microkernels (OptiX + CUDA) in Phase 3.
void render_wavefront_cpu(const SceneData& scene, ImageBuffer& img);
