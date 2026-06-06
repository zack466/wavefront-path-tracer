#pragma once
// DeviceScene: GPU-side mirror of SceneData.
// All pointer fields point into device memory and can be passed by value into
// CUDA kernels. No STL, no std::string — safe for nvcc device compilation.

#include "common/bvh_node.h"
#include "common/shape.h"
#include "common/triangle.h"
#include "common/material.h"
#include "common/light.h"
#include "common/camera.h"

// POD render config for device code (no std::string, no DenoiseConfig).
struct DevRenderConfig {
    int   width, height;
    int   spp, max_depth, rr_min_depth;
    float firefly_clamp;
    float bg_r, bg_g, bg_b;
    int   bg_mode;       // 0 = solid, 1 = sky gradient
    float aperture;
    float focus_dist;
};

struct DeviceScene {
    const Shape*      shapes;
    // Vertex-only layout (36 B each) for BVH intersection hot path.
    // The full Triangle is loaded only once, for the winning hit.
    const TriangleV*  tri_verts;
    const Triangle*   triangles;   // full data (normals + material)
    const Material*   materials;
    const Light*      lights;

    // Software BVH — used by CUDA BVH backend and CPU-shadow fallback.
    const BVHNode*    bvh_nodes;
    const int*        prim_indices;

    int num_shapes;
    int num_triangles;
    int num_materials;
    int num_lights;
    int num_bvh_nodes;
    int num_prim_indices;

    // Primitive index split: idx < bvh_num_shapes → shapes[], else → triangles[]
    int bvh_num_shapes;

    DevRenderConfig config;
    CameraFrame     cam;
};

struct SceneData;
DeviceScene upload_scene(const SceneData& scene, const CameraFrame& cam);
void        free_device_scene(DeviceScene& ds);
