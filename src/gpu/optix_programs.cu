// OptiX device programs for wavefront intersection stage.
//
// Every analytic shape (sphere/cylinder/disk) is tessellated into triangles
// at AS-build time, so the pipeline contains only one hit group and no custom
// IS programs. Hardware triangle intersection handles all geometry.
//
// Pipeline:
//   raygen → optixTrace → __closesthit__triangle | __miss__primary
//
// Both programs write to HitQueue / MissQueue using the same warp-ballot
// coalesced-atomic pattern as the original intersect_kernel.

#include <optix.h>
#include <optix_device.h>
#include "optix_params.h"
#include "gpu_warp.cuh"
#include "common/math.h"
#include "common/intersect.h"
#include "common/hit.h"
#include "common/types.h"

extern "C" {
__constant__ LaunchParams params;
}

// Background sample (mirrors gpu_background in kernels.cu).
__device__ __forceinline__
Vec3 optix_background(Vec3 dir) {
    if (params.bg_mode == 1) {
        float t = 0.5f * (normalize(dir).y + 1.0f);
        return lerp(Vec3(1.f), Vec3(0.5f, 0.7f, 1.0f), t);
    }
    return Vec3(params.bg_r, params.bg_g, params.bg_b);
}

__device__ __forceinline__
Vec3 optix_firefly_clamp(Vec3 c, float limit) {
    if (limit <= 0.f) return c;
    float l = luma(c);
    if (l > limit) c = c * (limit / l);
    return c;
}

// ── Write HitRecord + ray state to the HitQueue ───────────────────────────────
__device__ __forceinline__
void write_hit(int i, const HitRecord& rec) {
    int j = warp_compact_slot(params.d_hit_count, true);
    params.hit_px[j]   = rec.point.x;      params.hit_py[j]   = rec.point.y;      params.hit_pz[j]   = rec.point.z;
    params.hit_nx[j]   = rec.normal.x;     params.hit_ny[j]   = rec.normal.y;     params.hit_nz[j]   = rec.normal.z;
    params.hit_gnx[j]  = rec.geo_normal.x; params.hit_gny[j]  = rec.geo_normal.y; params.hit_gnz[j]  = rec.geo_normal.z;
    params.hit_wox[j]  = -params.ray_dx[i]; params.hit_woy[j]  = -params.ray_dy[i]; params.hit_woz[j]  = -params.ray_dz[i];
    params.hit_tr[j]   = params.ray_tr[i];  params.hit_tg[j]   = params.ray_tg[i];  params.hit_tb[j]   = params.ray_tb[i];
    params.hit_rr[j]   = params.ray_rr[i];  params.hit_rg[j]   = params.ray_rg[i];  params.hit_rb[j]   = params.ray_rb[i];
    params.hit_mid[j]  = rec.material_id;
    params.hit_ff[j]   = rec.front_face ? 1 : 0;
    params.hit_pidx[j] = params.ray_pidx[i];
    params.hit_depth[j]= params.ray_depth[i];
    params.hit_seed[j] = params.ray_seed[i];
    params.hit_cemit[j]= params.ray_cemit[i];
}

// ── Raygen ────────────────────────────────────────────────────────────────────
extern "C" __global__ void __raygen__primary() {
    const int i = (int)optixGetLaunchIndex().x;
    if (i >= params.ray_count) return;

    const float3 origin = { params.ray_ox[i], params.ray_oy[i], params.ray_oz[i] };
    const float3 dir    = { params.ray_dx[i], params.ray_dy[i], params.ray_dz[i] };

    uint32_t ray_idx = (uint32_t)i;
    optixTrace(
        params.traversable,
        origin, dir,
        PT_EPSILON, PT_INFINITY,
        0.f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,    // sbtOffset, sbtStride, missSBTIndex
        ray_idx);
}

// ── Miss ──────────────────────────────────────────────────────────────────────
// Writes the background contribution directly into image_buf — eliminates the
// MissQueue roundtrip + miss_kernel launch (each ~250-450us per bounce).
extern "C" __global__ void __miss__primary() {
    const int i = (int)optixGetPayload_0();
    Vec3 dir = { params.ray_dx[i], params.ray_dy[i], params.ray_dz[i] };
    Vec3 tp  = { params.ray_tr[i], params.ray_tg[i], params.ray_tb[i] };
    Vec3 rad = { params.ray_rr[i], params.ray_rg[i], params.ray_rb[i] };
    Vec3 out = optix_firefly_clamp(rad + tp * optix_background(dir),
                                   params.firefly_clamp);
    int pidx = params.ray_pidx[i];
    atomicAdd(&params.image_buf[pidx*3+0], out.x);
    atomicAdd(&params.image_buf[pidx*3+1], out.y);
    atomicAdd(&params.image_buf[pidx*3+2], out.z);
}

// ── Shadow raygen ─────────────────────────────────────────────────────────────
// Fires an any-hit ray for each entry in the shadow queue.
// TERMINATE_ON_FIRST_HIT: stops as soon as any occluder is found (no CH called).
// DISABLE_CLOSESTHIT:     never invokes the closest-hit program.
// If the ray reaches the miss program, the path is unblocked → add contribution.
extern "C" __global__ void __raygen__shadow() {
    const int i = (int)optixGetLaunchIndex().x;
    if (i >= params.sh_count) return;

    const float3 origin = { params.sh_ox[i], params.sh_oy[i], params.sh_oz[i] };
    const float3 dir    = { params.sh_dx[i], params.sh_dy[i], params.sh_dz[i] };
    const float  tmax   = params.sh_tmax[i];

    uint32_t idx = (uint32_t)i;
    optixTrace(
        params.traversable,
        origin, dir,
        PT_EPSILON, tmax,
        0.f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0, 1,
        1,      // missSBTIndex = 1 → __miss__shadow
        idx);
}

// ── Shadow miss (unblocked) ───────────────────────────────────────────────────
// Called only when the shadow ray hits nothing → add pre-scaled NEE contribution
// directly to the HDR image buffer.
extern "C" __global__ void __miss__shadow() {
    const int i = (int)optixGetPayload_0();
    const int pidx = params.sh_pidx[i];
    atomicAdd(&params.image_buf[pidx*3+0], params.sh_cr[i]);
    atomicAdd(&params.image_buf[pidx*3+1], params.sh_cg[i]);
    atomicAdd(&params.image_buf[pidx*3+2], params.sh_cb[i]);
}

// ── Triangle closest-hit (handles both scene OBJ tris and tessellated shapes) ─
// prim_idx  <  num_scene_triangles → OBJ mesh triangle
// prim_idx  >= num_scene_triangles → tessellated analytic shape triangle
extern "C" __global__ void __closesthit__triangle() {
    const int   i        = (int)optixGetPayload_0();
    const int   prim_idx = (int)optixGetPrimitiveIndex();
    const float t        = optixGetRayTmax();
    const float2 bary    = optixGetTriangleBarycentrics();
    const float  u = bary.x, v = bary.y;

    const float3 ray_o = optixGetWorldRayOrigin();
    const float3 ray_d = optixGetWorldRayDirection();
    Ray ray;
    ray.origin = { ray_o.x, ray_o.y, ray_o.z };
    ray.dir    = { ray_d.x, ray_d.y, ray_d.z };

    // Dispatch to the correct Triangle array for normal lookup
    const Triangle* tri_ptr = (prim_idx < params.num_scene_triangles)
        ? &params.triangles[prim_idx]
        : &params.tess_triangles[prim_idx - params.num_scene_triangles];

    HitRecord rec{};
    triangle_fill_normals(*tri_ptr, ray, t, u, v, rec);
    rec.material_id = tri_ptr->material_id;
    rec.shape_id    = prim_idx;   // unified index within the GAS

    write_hit(i, rec);
}
