#pragma once
#include <vector>
#include "common/bvh_node.h"    // BVHNode — STL-free, shared with GPU device code
#include "common/aabb.h"
#include "common/intersect.h"
#include "common/triangle.h"

// ── Scene acceleration structure ─────────────────────────────────────────────
// Built once at scene-load time, used by all CPU intersection calls.
// Primitive index encoding in prim_indices:
//   i  <  num_shapes   → scene.shapes[i]  (analytic shape)
//   i  >= num_shapes   → scene.triangles[i - num_shapes]
struct SceneAccel {
    std::vector<BVHNode> nodes;
    std::vector<int>     prim_indices;  // reordered primitive refs
    int                  num_shapes;
};

// ── Forward declarations ──────────────────────────────────────────────────────
struct SceneData;   // defined in scene/scene.h

// Build the BVH over all shapes + triangles in `scene`. Populates scene.accel.
void build_bvh(SceneData& scene);

// Closest-hit traversal. Returns true and fills `hit` on success.
bool bvh_intersect(const Ray& ray, const SceneAccel& accel,
                   const SceneData& scene, HitRecord& hit);

// Any-hit (shadow) traversal. Returns true as soon as any occluder is found
// within [ray.tmin, ray.tmax], without computing normals. shape_id of the
// first hit is written to *hit_shape_id (if non-null) so callers can exclude
// the light emitter shape from being counted as an occluder.
bool bvh_shadow_blocked(const Ray& ray, const SceneAccel& accel,
                        const SceneData& scene, int* hit_shape_id = nullptr);
