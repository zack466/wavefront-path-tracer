#pragma once
#include "scene/scene.h"
#include "output/image.h"

// Renders `scene` into `img` using a naive recursive path tracer.
// This is the Phase 1 reference implementation — correctness over performance.
// The wavefront loop (Phase 2) will replace this with queue-based kernels.
void render_cpu(const SceneData& scene, ImageBuffer& img);
