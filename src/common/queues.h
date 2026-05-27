#pragma once
#include "types.h"
#include <cstdlib>

// CPU allocator.  On the GPU these would be replaced with cudaMalloc / cudaFree,
// with each SoA array living in device-global memory for coalesced warp access.
inline void* queue_alloc(size_t bytes) { return std::malloc(bytes); }
inline void  queue_free(void* ptr)     { std::free(ptr); }

// ── RayQueue ──────────────────────────────────────────────────────────────────
// SoA: one slot per active ray, indexed 0..count-1.
// All pointer arrays are carved from a single allocation (base = origin_x).
// This layout maximises cache-line utilisation when GPU warps read one field
// each across 32 consecutive slots.
struct RayQueue {
    // Ray geometry
    float* origin_x;  float* origin_y;  float* origin_z;
    float* dir_x;     float* dir_y;     float* dir_z;
    // Path throughput: accumulated BSDF * cos / pdf product along the path
    float* throughput_r;  float* throughput_g;  float* throughput_b;
    // Radiance already gathered from earlier bounces (direct + emission contributions)
    float* radiance_r;    float* radiance_g;    float* radiance_b;
    // Metadata
    int*   pixel_idx;       // flat index py*width+px into the output buffer
    int*   depth;           // current bounce count (0 = primary ray)
    uint*  seed;            // per-ray xorshift32 RNG state
    int*   count_emission;  // 1 = ray should tally emissive-hit contribution;
                            // 0 = NEE already covered direct lighting this bounce
    int    count;
    int    capacity;
};

inline RayQueue alloc_ray_queue(int cap) {
    RayQueue q{};
    q.capacity = cap;
    size_t fb = size_t(cap) * sizeof(float);
    size_t ib = size_t(cap) * sizeof(int);
    size_t ub = size_t(cap) * sizeof(uint);
    // 12 float arrays + 3 int arrays + 1 uint array
    char* p = (char*)queue_alloc(12*fb + 3*ib + ub);
    q.origin_x      = (float*)p; p += fb;
    q.origin_y      = (float*)p; p += fb;
    q.origin_z      = (float*)p; p += fb;
    q.dir_x         = (float*)p; p += fb;
    q.dir_y         = (float*)p; p += fb;
    q.dir_z         = (float*)p; p += fb;
    q.throughput_r  = (float*)p; p += fb;
    q.throughput_g  = (float*)p; p += fb;
    q.throughput_b  = (float*)p; p += fb;
    q.radiance_r    = (float*)p; p += fb;
    q.radiance_g    = (float*)p; p += fb;
    q.radiance_b    = (float*)p; p += fb;
    q.pixel_idx      = (int*)p;  p += ib;
    q.depth          = (int*)p;  p += ib;
    q.count_emission = (int*)p;  p += ib;
    q.seed           = (uint*)p;
    return q;
}

inline void free_ray_queue(RayQueue& q) {
    queue_free(q.origin_x);  // base of single-block allocation
    q = {};
}

// ── HitQueue ──────────────────────────────────────────────────────────────────
// Populated by the Intersect stage; consumed by Shade.
// Carries the full HitRecord plus the incoming ray state so Shade never needs
// to reach back into the RayQueue.
struct HitQueue {
    // Intersection geometry
    float* point_x;   float* point_y;   float* point_z;
    float* normal_x;  float* normal_y;  float* normal_z;    // shading normal
    float* geo_nx;    float* geo_ny;    float* geo_nz;      // geometric normal
    float* t;
    int*   material_id;
    int*   shape_id;
    int*   front_face;  // stored as int (0 or 1) for alignment
    // Outgoing view direction wo = -ray.dir (in world space)
    float* wo_x;  float* wo_y;  float* wo_z;
    // Ray path state copied from RayQueue
    float* throughput_r;  float* throughput_g;  float* throughput_b;
    float* radiance_r;    float* radiance_g;    float* radiance_b;
    int*   pixel_idx;
    int*   depth;
    uint*  seed;
    int*   count_emission;
    int    count;
    int    capacity;
};

inline HitQueue alloc_hit_queue(int cap) {
    HitQueue q{};
    q.capacity = cap;
    size_t fb = size_t(cap) * sizeof(float);
    size_t ib = size_t(cap) * sizeof(int);
    size_t ub = size_t(cap) * sizeof(uint);
    // 19 float arrays: point(3) + normal(3) + geo_normal(3) + t(1) + wo(3) + throughput(3) + radiance(3)
    // 6  int  arrays:  material_id + shape_id + front_face + pixel_idx + depth + count_emission
    // 1  uint array:   seed
    char* p = (char*)queue_alloc(19*fb + 6*ib + ub);
    q.point_x      = (float*)p; p += fb;
    q.point_y      = (float*)p; p += fb;
    q.point_z      = (float*)p; p += fb;
    q.normal_x     = (float*)p; p += fb;
    q.normal_y     = (float*)p; p += fb;
    q.normal_z     = (float*)p; p += fb;
    q.geo_nx       = (float*)p; p += fb;
    q.geo_ny       = (float*)p; p += fb;
    q.geo_nz       = (float*)p; p += fb;
    q.t            = (float*)p; p += fb;
    q.wo_x         = (float*)p; p += fb;
    q.wo_y         = (float*)p; p += fb;
    q.wo_z         = (float*)p; p += fb;
    q.throughput_r = (float*)p; p += fb;
    q.throughput_g = (float*)p; p += fb;
    q.throughput_b = (float*)p; p += fb;
    q.radiance_r   = (float*)p; p += fb;
    q.radiance_g   = (float*)p; p += fb;
    q.radiance_b   = (float*)p; p += fb;
    q.material_id   = (int*)p;  p += ib;
    q.shape_id      = (int*)p;  p += ib;
    q.front_face    = (int*)p;  p += ib;
    q.pixel_idx     = (int*)p;  p += ib;
    q.depth         = (int*)p;  p += ib;
    q.count_emission= (int*)p;  p += ib;
    q.seed          = (uint*)p;
    return q;
}

inline void free_hit_queue(HitQueue& q) {
    queue_free(q.point_x);
    q = {};
}

// ── MissQueue ─────────────────────────────────────────────────────────────────
// Populated by Intersect for rays that hit nothing; consumed by the Miss stage
// to add background / environment radiance.
struct MissQueue {
    float* dir_x;  float* dir_y;  float* dir_z;  // ray direction (for sky sampling)
    float* throughput_r;  float* throughput_g;  float* throughput_b;
    float* radiance_r;    float* radiance_g;    float* radiance_b;
    int*   pixel_idx;
    uint*  seed;
    int    count;
    int    capacity;
};

inline MissQueue alloc_miss_queue(int cap) {
    MissQueue q{};
    q.capacity = cap;
    size_t fb = size_t(cap) * sizeof(float);
    size_t ib = size_t(cap) * sizeof(int);
    size_t ub = size_t(cap) * sizeof(uint);
    // 9 float arrays + 1 int array + 1 uint array
    char* p = (char*)queue_alloc(9*fb + ib + ub);
    q.dir_x        = (float*)p; p += fb;
    q.dir_y        = (float*)p; p += fb;
    q.dir_z        = (float*)p; p += fb;
    q.throughput_r = (float*)p; p += fb;
    q.throughput_g = (float*)p; p += fb;
    q.throughput_b = (float*)p; p += fb;
    q.radiance_r   = (float*)p; p += fb;
    q.radiance_g   = (float*)p; p += fb;
    q.radiance_b   = (float*)p; p += fb;
    q.pixel_idx    = (int*)p;   p += ib;
    q.seed         = (uint*)p;
    return q;
}

inline void free_miss_queue(MissQueue& q) {
    queue_free(q.dir_x);
    q = {};
}

// ── ShadowQueue (GPU only) ────────────────────────────────────────────────────
// On the GPU, the shade kernel writes one entry per non-specular hit into a
// ShadowQueue (origin, dir, tmax, contrib, pixel_idx).  A second GPU kernel
// (or OptiX any-hit launch) then fires shadow rays for all entries at once and
// adds unblocked contributions directly to the device frame buffer.
// The CPU path tests shadow rays inline inside sample_direct_lighting().
