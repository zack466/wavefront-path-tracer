#include "device_scene.h"
#include "scene/scene.h"
#include "common/camera.h"

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do {                                             \
    cudaError_t _e = (x);                                              \
    if (_e != cudaSuccess) {                                           \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                     __FILE__, __LINE__, cudaGetErrorString(_e));      \
        throw std::runtime_error(cudaGetErrorString(_e));              \
    }                                                                  \
} while (0)

template <typename T>
static const T* upload_array(const T* host, size_t count) {
    if (count == 0) return nullptr;
    T* dev;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

DeviceScene upload_scene(const SceneData& scene, const CameraFrame& cam) {
    DeviceScene ds{};

    ds.num_shapes       = (int)scene.shapes.size();
    ds.num_triangles    = (int)scene.triangles.size();
    ds.num_materials    = (int)scene.materials.size();
    ds.num_lights       = (int)scene.lights.size();
    ds.num_bvh_nodes    = (int)scene.accel.nodes.size();
    ds.num_prim_indices = (int)scene.accel.prim_indices.size();
    ds.bvh_num_shapes   = scene.accel.num_shapes;

    // Build compact vertex-only array for BVH traversal hot path.
    // 36 bytes/triangle keeps the dragon's vertex data (31 MB) within the
    // A5000's 40 MB L2 cache; the full Triangle (80 B) would spill to DRAM.
    std::vector<TriangleV> tv(ds.num_triangles);
    for (int i = 0; i < ds.num_triangles; ++i) {
        tv[i].v[0] = scene.triangles[i].v[0];
        tv[i].v[1] = scene.triangles[i].v[1];
        tv[i].v[2] = scene.triangles[i].v[2];
    }

    ds.shapes       = upload_array(scene.shapes.data(),             ds.num_shapes);
    ds.tri_verts    = upload_array(tv.data(),                        ds.num_triangles);
    ds.triangles    = upload_array(scene.triangles.data(),           ds.num_triangles);
    ds.materials    = upload_array(scene.materials.data(),           ds.num_materials);
    ds.lights       = upload_array(scene.lights.data(),              ds.num_lights);
    ds.bvh_nodes    = upload_array(scene.accel.nodes.data(),         ds.num_bvh_nodes);
    ds.prim_indices = upload_array(scene.accel.prim_indices.data(),  ds.num_prim_indices);

    const RenderConfig& cfg = scene.config;
    ds.config.width         = cfg.width;
    ds.config.height        = cfg.height;
    ds.config.spp           = cfg.spp;
    ds.config.max_depth     = cfg.max_depth;
    ds.config.rr_min_depth  = cfg.rr_min_depth;
    ds.config.firefly_clamp = cfg.firefly_clamp;
    ds.config.bg_r          = cfg.background.x;
    ds.config.bg_g          = cfg.background.y;
    ds.config.bg_b          = cfg.background.z;
    ds.config.bg_mode       = (cfg.bg_mode == BackgroundMode::Sky) ? 1 : 0;
    ds.config.aperture      = scene.camera.aperture;
    ds.config.focus_dist    = scene.camera.focus_dist;
    ds.cam = cam;

    return ds;
}

void free_device_scene(DeviceScene& ds) {
    cudaFree(const_cast<Shape*>    (ds.shapes));
    cudaFree(const_cast<TriangleV*>(ds.tri_verts));
    cudaFree(const_cast<Triangle*> (ds.triangles));
    cudaFree(const_cast<Material*> (ds.materials));
    cudaFree(const_cast<Light*>    (ds.lights));
    cudaFree(const_cast<BVHNode*>  (ds.bvh_nodes));
    cudaFree(const_cast<int*>      (ds.prim_indices));
    ds = {};
}
