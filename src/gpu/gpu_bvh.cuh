#pragma once
// __device__ BVH traversal — closest-hit and any-hit (shadow) variants.
//
// Triangle tests use TriangleV (36 bytes) rather than the full Triangle struct
// (80 bytes). For the dragon mesh (871K triangles) the vertex data fits in 31 MB,
// within the A5000's 40 MB L2 cache. The full Triangle is loaded only once, for
// the winning hit, to fill shading normals. Shadow rays never need normals so
// they only ever read TriangleV.

#include "common/bvh_node.h"
#include "common/intersect.h"
#include "common/shape.h"
#include "common/triangle.h"
#include "common/hit.h"
#include "common/math.h"

static constexpr int BVH_STACK_SIZE = 32;

// ── Inline AABB slab test ─────────────────────────────────────────────────────
__device__ __forceinline__
bool node_aabb_hit(const BVHNode* __restrict__ nodes, int idx,
                   const Ray& ray, float t_max)
{
    float t_near = ray.tmin;
    float t_far  = t_max;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        float inv_d = 1.0f / ray.dir[k];
        float t0 = (__ldg(&nodes[idx].aabb_min[k]) - ray.origin[k]) * inv_d;
        float t1 = (__ldg(&nodes[idx].aabb_max[k]) - ray.origin[k]) * inv_d;
        if (inv_d < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_near = fmaxf(t_near, t0);
        t_far  = fminf(t_far,  t1);
    }
    return t_near <= t_far;
}

// ── Closest-hit traversal ─────────────────────────────────────────────────────
__device__ bool gpu_bvh_intersect(
    const Ray&               ray,
    const BVHNode* __restrict__   nodes,
    const int*     __restrict__   prim_indices,
    int                      bvh_num_shapes,
    const Shape*   __restrict__   shapes,
    const TriangleV* __restrict__ tri_verts,
    const Triangle*  __restrict__ triangles,
    HitRecord& hit)
{
    if (!nodes) return false;

    bool  found     = false;
    float t_closest = ray.tmax;

    int   win_tri = -1;
    float win_u   = 0.f, win_v = 0.f;

    int stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];

        if (!node_aabb_hit(nodes, ni, ray, t_closest)) continue;

        int prim_count = __ldg(&nodes[ni].prim_count);

        if (prim_count > 0) {
            int first = __ldg(&nodes[ni].left_first);
            for (int k = 0; k < prim_count; ++k) {
                int prim = __ldg(&prim_indices[first + k]);

                if (prim < bvh_num_shapes) {
                    HitRecord tmp{};
                    Shape s = shapes[prim];
                    if (intersect_shape(ray, s, t_closest, tmp)) {
                        t_closest       = tmp.t;
                        tmp.material_id = s.material_id;
                        tmp.shape_id    = prim;
                        hit     = tmp;
                        win_tri = -1;
                        found   = true;
                    }
                } else {
                    int ti = prim - bvh_num_shapes;
                    float t_hit, u_hit, v_hit;
                    const TriangleV& tv = tri_verts[ti];
                    if (intersect_triangle_v(ray, tv.v[0], tv.v[1], tv.v[2],
                                             t_closest, t_hit, u_hit, v_hit)) {
                        t_closest = t_hit;
                        win_tri   = ti;
                        win_u     = u_hit;
                        win_v     = v_hit;
                        hit.t        = t_hit;
                        hit.shape_id = prim;
                        found = true;
                    }
                }
            }
        } else {
            int left  = __ldg(&nodes[ni].left_first);
            int right = __ldg(&nodes[ni].right_child);
            int axis  = __ldg(&nodes[ni].split_axis);
            // Push far child first so near child is processed first
            bool left_first = (ray.dir[axis] >= 0.0f);
            if (left_first) {
                stack[sp++] = right;
                stack[sp++] = left;
            } else {
                stack[sp++] = left;
                stack[sp++] = right;
            }
        }
    }

    // Deferred: load full Triangle only for the winning hit
    if (win_tri >= 0) {
        Triangle tri = triangles[win_tri];
        triangle_fill_normals(tri, ray, hit.t, win_u, win_v, hit);
        hit.material_id = tri.material_id;
        hit.shape_id    = win_tri + bvh_num_shapes;
    }

    return found;
}

// ── Any-hit (shadow) traversal ────────────────────────────────────────────────
// Returns true as soon as any occluder is found. Never loads normals or material.
// Only reads TriangleV (36 bytes/triangle).
__device__ bool gpu_shadow_blocked(
    const Ray&               ray,
    const BVHNode* __restrict__   nodes,
    const int*     __restrict__   prim_indices,
    int                      bvh_num_shapes,
    const Shape*   __restrict__   shapes,
    const TriangleV* __restrict__ tri_verts)
{
    if (!nodes) return false;

    int stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];

        if (!node_aabb_hit(nodes, ni, ray, ray.tmax)) continue;

        int prim_count = __ldg(&nodes[ni].prim_count);

        if (prim_count > 0) {
            int first = __ldg(&nodes[ni].left_first);
            for (int k = 0; k < prim_count; ++k) {
                int prim = __ldg(&prim_indices[first + k]);
                if (prim < bvh_num_shapes) {
                    HitRecord tmp{};
                    Shape s = shapes[prim];
                    if (intersect_shape(ray, s, ray.tmax, tmp)) return true;
                } else {
                    int ti = prim - bvh_num_shapes;
                    float t_hit, u_hit, v_hit;
                    const TriangleV& tv = tri_verts[ti];
                    if (intersect_triangle_v(ray, tv.v[0], tv.v[1], tv.v[2],
                                             ray.tmax, t_hit, u_hit, v_hit))
                        return true;
                }
            }
        } else {
            int left  = __ldg(&nodes[ni].left_first);
            int right = __ldg(&nodes[ni].right_child);
            int axis  = __ldg(&nodes[ni].split_axis);
            bool left_first = (ray.dir[axis] >= 0.0f);
            if (left_first) {
                stack[sp++] = right;
                stack[sp++] = left;
            } else {
                stack[sp++] = left;
                stack[sp++] = right;
            }
        }
    }

    return false;
}
