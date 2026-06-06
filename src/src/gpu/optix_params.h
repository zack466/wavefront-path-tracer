#pragma once
// Shared data between OptiX device programs (optix_programs.cu) and the host
// (optix_state.cu). Must compile cleanly for both __device__ and host code.
// Avoids including queues.h (which pulls in cstdlib/malloc — host-only).

#include <optix.h>
#include "common/math.h"
#include "common/triangle.h"

// ── Payload register layout ───────────────────────────────────────────────────
// A single 32-bit payload register carries the ray index i ∈ [0, ray_count).
// Set by the raygen program, read by closesthit and miss.
#define OPTIX_PAYLOAD_RAY_INDEX 0

// ── Launch parameters ─────────────────────────────────────────────────────────
// When OptiX is active every analytic shape (sphere/cyl/disk) is tessellated
// into triangles and baked into the same triangle GAS as the scene triangles.
// The traversable is therefore a single flat triangle GAS — no IAS, no custom
// primitive GAS, no IS programs.
//
// The closesthit program maps prim_idx → normal data:
//   prim_idx  <  num_scene_triangles → params.triangles[prim_idx]  (scene OBJ tris)
//   prim_idx  >= num_scene_triangles → params.tess_triangles[prim_idx - num_scene_triangles]

struct LaunchParams {
    // ── Input: ray queue fields ───────────────────────────────────────────────
    const float*        ray_ox;   const float* ray_oy;   const float* ray_oz;
    const float*        ray_dx;   const float* ray_dy;   const float* ray_dz;
    const float*        ray_tr;   const float* ray_tg;   const float* ray_tb;
    const float*        ray_rr;   const float* ray_rg;   const float* ray_rb;
    const int*          ray_pidx;
    const int*          ray_depth;
    const unsigned int* ray_seed;
    const int*          ray_cemit;
    int                 ray_count;

    // ── Output: hit queue fields ──────────────────────────────────────────────
    float*        hit_px;   float* hit_py;   float* hit_pz;
    float*        hit_nx;   float* hit_ny;   float* hit_nz;
    float*        hit_gnx;  float* hit_gny;  float* hit_gnz;
    float*        hit_wox;  float* hit_woy;  float* hit_woz;
    float*        hit_tr;   float* hit_tg;   float* hit_tb;
    float*        hit_rr;   float* hit_rg;   float* hit_rb;
    int*          hit_mid;  int*   hit_ff;
    int*          hit_pidx; int*   hit_depth;
    unsigned int* hit_seed;
    int*          hit_cemit;
    int*          d_hit_count;

    // ── Background config (used by __miss__primary) ──────────────────────────
    float bg_r, bg_g, bg_b;
    int   bg_mode;            // 0 = solid, 1 = sky gradient
    float firefly_clamp;

    // ── Geometry data ─────────────────────────────────────────────────────────
    OptixTraversableHandle traversable;

    // Triangle GAS: scene OBJ triangles followed by tessellated analytic shapes.
    const Triangle*  triangles;            // scene OBJ triangle normals
    const Triangle*  tess_triangles;       // tessellated analytic shape normals
    int              num_scene_triangles;  // split point in prim_idx space

    // ── Shadow queue (read by __raygen__shadow) ───────────────────────────────
    const float* sh_ox;  const float* sh_oy;  const float* sh_oz;
    const float* sh_dx;  const float* sh_dy;  const float* sh_dz;
    const float* sh_tmax;
    const float* sh_cr;  const float* sh_cg;  const float* sh_cb;
    const int*   sh_pidx;
    int          sh_count;

    // ── Image buffer (written by __miss__shadow) ──────────────────────────────
    float*       image_buf;
};
