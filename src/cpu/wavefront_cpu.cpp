#include "wavefront_cpu.h"
#include "cpu_shading.h"
#include "common/queues.h"
#include "common/camera.h"
#include "shading/bsdf.h"
#include "bvh/bvh.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ── Ray sort key ──────────────────────────────────────────────────────────────
// Produces a 32-bit key: 3-bit direction octant (MSBs) + 29-bit origin Morton
// code.  Rays with identical octant traverse the BVH in the same child-ordering;
// nearby origins tend to visit the same leaf nodes → better cache locality.
//
// GPU equivalent: a compute_ray_keys_kernel fills a key array in device memory,
// then cub::DeviceRadixSort::SortPairs() sorts ray indices by key in O(n) time.

// Build a sorted index array for the ray queue.
//
// Two-pass radix sort on the 32-bit key (octant + Morton).
// Pass 1 (O(n)): sort by the 3-bit direction octant using a counting sort.
//   This groups rays that traverse BVH children in the same order together,
//   reducing branch mispredictions and improving L3 cache reuse for BVH nodes.
// Pass 2 (O(n)): within each octant, stable-sort by 3 bits of quantised
//   origin (x-axis), giving coarser spatial grouping at negligible cost.
//
// Unlike a full O(n log n) std::sort, this runs in O(n) and costs only ~2
// passes over the ray data — fast enough to pay off even for small meshes.
static void sort_ray_indices(const RayQueue& rays, std::vector<int>& idx)
{
    const int n = rays.count;
    idx.resize(n);

    // ── Pass 1: 3-bit octant sort (8 buckets) ────────────────────────────────
    int cnt[8] = {};
    for (int i = 0; i < n; ++i) {
        uint32_t oct = (uint32_t)(rays.dir_x[i] < 0.f)
                     | ((uint32_t)(rays.dir_y[i] < 0.f) << 1u)
                     | ((uint32_t)(rays.dir_z[i] < 0.f) << 2u);
        ++cnt[oct];
    }
    int off[8]; off[0] = 0;
    for (int k = 1; k < 8; ++k) off[k] = off[k-1] + cnt[k-1];

    int pos[8]; std::copy(off, off+8, pos);
    for (int i = 0; i < n; ++i) {
        uint32_t oct = (uint32_t)(rays.dir_x[i] < 0.f)
                     | ((uint32_t)(rays.dir_y[i] < 0.f) << 1u)
                     | ((uint32_t)(rays.dir_z[i] < 0.f) << 2u);
        idx[pos[oct]++] = i;
    }

    // ── Pass 2: within each octant, bucket by 3 quantised origin-x bits ──────
    // Gives coarse spatial grouping: rays from nearby regions in x access
    // the same BVH subtrees, improving L3 hit rates for leaf nodes.
    static thread_local std::vector<int> tmp;
    tmp.resize(n);
    constexpr float kScale = 8.0f / 40.0f;  // 8 cells over [-20, 20]
    for (int k = 0; k < 8; ++k) {
        int lo = off[k], hi = lo + cnt[k];
        if (hi - lo <= 1) continue;
        int c2[8] = {};
        for (int i = lo; i < hi; ++i) {
            int ri = idx[i];
            uint32_t qx = std::min((uint32_t)std::max(0.f, (rays.origin_x[ri] + 20.f) * kScale), 7u);
            ++c2[qx];
        }
        int o2[8]; o2[0] = lo;
        for (int j = 1; j < 8; ++j) o2[j] = o2[j-1] + c2[j-1];
        int p2[8]; std::copy(o2, o2+8, p2);
        for (int i = lo; i < hi; ++i) {
            int ri = idx[i];
            uint32_t qx = std::min((uint32_t)std::max(0.f, (rays.origin_x[ri] + 20.f) * kScale), 7u);
            tmp[p2[qx]++] = ri;
        }
        std::copy(tmp.begin() + lo, tmp.begin() + hi, idx.begin() + lo);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Write final path radiance to the accumulation buffer, with optional firefly
// clamping applied at path termination (same as Phase 1's per-path clamp).
static void accumulate(ImageBuffer& img, int pixel_idx, int width,
                       Vec3 radiance, float firefly_clamp)
{
    if (firefly_clamp > 0.f) {
        float l = luma(radiance);
        if (l > firefly_clamp) radiance = radiance * (firefly_clamp / l);
    }
    img.add(pixel_idx % width, pixel_idx / width, radiance);
}

// ── SoA compaction ────────────────────────────────────────────────────────────
// After a parallel kernel runs with schedule(static, chunk), thread t's valid
// output occupies [t*chunk, t*chunk+t_count[t]) in each SoA field.  This
// function moves each field's thread-local segments into a contiguous block.
//
// Thread 0's data is already at [0, t_count[0]) — no move needed.
// Sequential, O(N) total data touched, fast (memmove is bandwidth-bound).
//
// GPU equivalent: warp-ballot compaction — each thread evaluates its predicate,
// then __ballot_sync() counts the hits in its warp.  One atomicAdd per warp
// reserves a contiguous output slot, and threads scatter their results there.
// This is essentially free (no extra passes over data) and is the primary source
// of GPU throughput advantage over this CPU implementation.

static void compact_field(void* arr, size_t esz,
                          const int* t_count, int chunk, int nt)
{
    char* a = (char*)arr;
    int dst = t_count[0];
    for (int t = 1; t < nt; ++t) {
        if (t_count[t] > 0) {
            int src = t * chunk;
            if (src != dst)
                std::memmove(a + size_t(dst)*esz,
                             a + size_t(src)*esz,
                             size_t(t_count[t])*esz);
        }
        dst += t_count[t];
    }
}

// ── Kernel: GenerateRays ──────────────────────────────────────────────────────
// Fills `out` with one primary camera ray per pixel for sample index `s`.
// Slot index = flat pixel index — no shared counter needed.
//
// GPU equivalent: __global__ void generate_kernel(DeviceScene, int s, RayQueue*)
// launched as <<<(W*H+255)/256, 256>>> — one thread per pixel, fully independent.

static void kernel_generate(const SceneData& scene, const CameraFrame& cam,
                             int s, int sqrt_spp, bool stratified,
                             RayQueue& out)
{
    const RenderConfig& cfg = scene.config;
    const int total_pixels  = cfg.width * cfg.height;
    out.count = total_pixels;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int py = 0; py < cfg.height; ++py) {
        for (int px = 0; px < cfg.width; ++px) {
            int i     = py * cfg.width + px;
            uint seed = wang_hash(uint(i) + uint(s) * uint(total_pixels));

            float ou, ov;
            if (stratified) {
                int sx = s % sqrt_spp, sy = s / sqrt_spp;
                ou = (float(sx) + rand_float(seed)) / float(sqrt_spp);
                ov = (float(sy) + rand_float(seed)) / float(sqrt_spp);
            } else {
                ou = rand_float(seed);
                ov = rand_float(seed);
            }

            float lu = 0.f, lv = 0.f;
            if (scene.camera.aperture > 0.f) {
                float r   = sqrtf(rand_float(seed));
                float phi = PT_TWO_PI * rand_float(seed);
                lu = r * cosf(phi);
                lv = r * sinf(phi);
            }

            Ray ray = camera_ray(cam, px, py, ou, ov,
                                 cfg.width, cfg.height,
                                 lu, lv,
                                 scene.camera.aperture, scene.camera.focus_dist);

            out.origin_x[i] = ray.origin.x;
            out.origin_y[i] = ray.origin.y;
            out.origin_z[i] = ray.origin.z;
            out.dir_x[i]    = ray.dir.x;
            out.dir_y[i]    = ray.dir.y;
            out.dir_z[i]    = ray.dir.z;

            out.throughput_r[i] = 1.f;  out.throughput_g[i] = 1.f;  out.throughput_b[i] = 1.f;
            out.radiance_r[i]   = 0.f;  out.radiance_g[i]   = 0.f;  out.radiance_b[i]   = 0.f;

            out.pixel_idx[i]      = i;
            out.depth[i]          = 0;
            out.seed[i]           = seed;
            out.count_emission[i] = 1;
        }
    }
}

// ── Kernel: Intersect ─────────────────────────────────────────────────────────
// Tests each ray against the BVH.
//
// Parallelism: schedule(static, chunk) gives thread t exclusive ownership of
// input range [t*chunk, (t+1)*chunk) and output segments
//   hits  [t*chunk .. t*chunk + t_nhits[t])
//   misses[t*chunk .. t*chunk + t_nmisses[t])
// No atomics needed.  After the parallel phase, compact() merges segments.
//
// GPU equivalent: __global__ void intersect_kernel(RayQueue, DeviceScene, HitQueue*, MissQueue*)
// Each thread processes one ray.  BVH traversal uses either the CUDA software
// BVH (iterative stack-based traversal in device memory) or OptiX RT cores via
// optixTrace().  Warp-ballot compaction (see compact_field note above) writes hits
// and misses to contiguous device-memory queues without sequential passes.

// sorted_idx: when non-null, thread i processes ray sorted_idx[i].
// Sorted rays traverse similar BVH paths, improving cache locality for
// BVH node and triangle data — mirrors the GPU's CUB-sorted intersect pass.
static void kernel_intersect(const RayQueue& rays, const SceneData& scene,
                              HitQueue& hits, MissQueue& misses,
                              const std::vector<int>* sorted_idx = nullptr)
{
    const int n  = rays.count;
#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif
    const int chunk = (n + nt - 1) / nt;

    std::vector<int> t_nhits(nt, 0), t_nmisses(nt, 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static, chunk)
#endif
    for (int i = 0; i < n; ++i) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        // Gather: use sorted order when available
        int ri = sorted_idx ? (*sorted_idx)[i] : i;

        Ray ray;
        ray.origin = { rays.origin_x[ri], rays.origin_y[ri], rays.origin_z[ri] };
        ray.dir    = { rays.dir_x[ri],    rays.dir_y[ri],    rays.dir_z[ri]    };
        ray.tmin   = PT_EPSILON;
        ray.tmax   = PT_INFINITY;

        HitRecord hit{};
        if (bvh_intersect(ray, scene.accel, scene, hit)) {
            int j = tid * chunk + t_nhits[tid]++;
            hits.point_x[j]      = hit.point.x;
            hits.point_y[j]      = hit.point.y;
            hits.point_z[j]      = hit.point.z;
            hits.normal_x[j]     = hit.normal.x;
            hits.normal_y[j]     = hit.normal.y;
            hits.normal_z[j]     = hit.normal.z;
            hits.geo_nx[j]       = hit.geo_normal.x;
            hits.geo_ny[j]       = hit.geo_normal.y;
            hits.geo_nz[j]       = hit.geo_normal.z;
            hits.t[j]            = hit.t;
            hits.material_id[j]  = hit.material_id;
            hits.shape_id[j]     = hit.shape_id;
            hits.front_face[j]   = hit.front_face ? 1 : 0;
            hits.wo_x[j]         = -rays.dir_x[ri];
            hits.wo_y[j]         = -rays.dir_y[ri];
            hits.wo_z[j]         = -rays.dir_z[ri];
            hits.throughput_r[j] = rays.throughput_r[ri];
            hits.throughput_g[j] = rays.throughput_g[ri];
            hits.throughput_b[j] = rays.throughput_b[ri];
            hits.radiance_r[j]   = rays.radiance_r[ri];
            hits.radiance_g[j]   = rays.radiance_g[ri];
            hits.radiance_b[j]   = rays.radiance_b[ri];
            hits.pixel_idx[j]    = rays.pixel_idx[ri];
            hits.depth[j]        = rays.depth[ri];
            hits.seed[j]         = rays.seed[ri];
            hits.count_emission[j]= rays.count_emission[ri];
        } else {
            int j = tid * chunk + t_nmisses[tid]++;
            misses.dir_x[j]        = rays.dir_x[ri];
            misses.dir_y[j]        = rays.dir_y[ri];
            misses.dir_z[j]        = rays.dir_z[ri];
            misses.throughput_r[j] = rays.throughput_r[ri];
            misses.throughput_g[j] = rays.throughput_g[ri];
            misses.throughput_b[j] = rays.throughput_b[ri];
            misses.radiance_r[j]   = rays.radiance_r[ri];
            misses.radiance_g[j]   = rays.radiance_g[ri];
            misses.radiance_b[j]   = rays.radiance_b[ri];
            misses.pixel_idx[j]    = rays.pixel_idx[ri];
            misses.seed[j]         = rays.seed[ri];
        }
    }

    // Compact hits: move each thread's segment to a contiguous block
    const int* nh = t_nhits.data();
    compact_field(hits.point_x,       sizeof(float), nh, chunk, nt);
    compact_field(hits.point_y,       sizeof(float), nh, chunk, nt);
    compact_field(hits.point_z,       sizeof(float), nh, chunk, nt);
    compact_field(hits.normal_x,      sizeof(float), nh, chunk, nt);
    compact_field(hits.normal_y,      sizeof(float), nh, chunk, nt);
    compact_field(hits.normal_z,      sizeof(float), nh, chunk, nt);
    compact_field(hits.geo_nx,        sizeof(float), nh, chunk, nt);
    compact_field(hits.geo_ny,        sizeof(float), nh, chunk, nt);
    compact_field(hits.geo_nz,        sizeof(float), nh, chunk, nt);
    compact_field(hits.t,             sizeof(float), nh, chunk, nt);
    compact_field(hits.wo_x,          sizeof(float), nh, chunk, nt);
    compact_field(hits.wo_y,          sizeof(float), nh, chunk, nt);
    compact_field(hits.wo_z,          sizeof(float), nh, chunk, nt);
    compact_field(hits.throughput_r,  sizeof(float), nh, chunk, nt);
    compact_field(hits.throughput_g,  sizeof(float), nh, chunk, nt);
    compact_field(hits.throughput_b,  sizeof(float), nh, chunk, nt);
    compact_field(hits.radiance_r,    sizeof(float), nh, chunk, nt);
    compact_field(hits.radiance_g,    sizeof(float), nh, chunk, nt);
    compact_field(hits.radiance_b,    sizeof(float), nh, chunk, nt);
    compact_field(hits.material_id,   sizeof(int),   nh, chunk, nt);
    compact_field(hits.shape_id,      sizeof(int),   nh, chunk, nt);
    compact_field(hits.front_face,    sizeof(int),   nh, chunk, nt);
    compact_field(hits.pixel_idx,     sizeof(int),   nh, chunk, nt);
    compact_field(hits.depth,         sizeof(int),   nh, chunk, nt);
    compact_field(hits.count_emission,sizeof(int),   nh, chunk, nt);
    compact_field(hits.seed,          sizeof(uint),  nh, chunk, nt);
    hits.count = 0;
    for (int t = 0; t < nt; ++t) hits.count += t_nhits[t];

    // Compact misses
    const int* nm = t_nmisses.data();
    compact_field(misses.dir_x,        sizeof(float), nm, chunk, nt);
    compact_field(misses.dir_y,        sizeof(float), nm, chunk, nt);
    compact_field(misses.dir_z,        sizeof(float), nm, chunk, nt);
    compact_field(misses.throughput_r, sizeof(float), nm, chunk, nt);
    compact_field(misses.throughput_g, sizeof(float), nm, chunk, nt);
    compact_field(misses.throughput_b, sizeof(float), nm, chunk, nt);
    compact_field(misses.radiance_r,   sizeof(float), nm, chunk, nt);
    compact_field(misses.radiance_g,   sizeof(float), nm, chunk, nt);
    compact_field(misses.radiance_b,   sizeof(float), nm, chunk, nt);
    compact_field(misses.pixel_idx,    sizeof(int),   nm, chunk, nt);
    compact_field(misses.seed,         sizeof(uint),  nm, chunk, nt);
    misses.count = 0;
    for (int t = 0; t < nt; ++t) misses.count += t_nmisses[t];
}

// ── Kernel: Miss ──────────────────────────────────────────────────────────────
// Each miss ray has a unique pixel_idx within one sample iteration — no
// shared writes, no atomics needed.
//
// GPU equivalent: __global__ void miss_kernel(MissQueue, DeviceScene, float* frame_buffer)
// img.add() becomes two atomicAdd calls into the device frame buffer (one per channel).

static void kernel_miss(const MissQueue& misses, const SceneData& scene,
                        ImageBuffer& img)
{
    const RenderConfig& cfg = scene.config;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < misses.count; ++i) {
        Ray r; r.dir = { misses.dir_x[i], misses.dir_y[i], misses.dir_z[i] };
        Vec3 tp  = { misses.throughput_r[i], misses.throughput_g[i], misses.throughput_b[i] };
        Vec3 rad = { misses.radiance_r[i],   misses.radiance_g[i],   misses.radiance_b[i]   };
        accumulate(img, misses.pixel_idx[i], cfg.width,
                   rad + tp * background_radiance(r, cfg), cfg.firefly_clamp);
    }
}

// ── Kernel: Shade ─────────────────────────────────────────────────────────────
// Evaluates BSDF at each hit, gathers direct lighting (NEE), terminates paths
// that can't continue, and emits surviving rays into `nxt`.
//
// Termination writes: img.add() → unique pixel per ray per sample → no atomics.
// Continuation writes: thread t owns nxt[t*chunk .. t*chunk+t_nnxt[t]) → no atomics.
// After the parallel phase, nxt is compacted.
//
// GPU equivalent: __global__ void shade_kernel(HitQueue, DeviceScene, RayQueue* nxt, float* fb)
// Direct lighting (NEE) shadow rays are either tested inline via gpu_shadow_blocked()
// (CUDA software BVH path) or written to a ShadowQueue for a second GPU kernel
// launch (OptiX path).  Warp-ballot compaction fills nxt without sequential passes.

static void kernel_shade(HitQueue& hits, const SceneData& scene,
                         RayQueue& nxt, ImageBuffer& img)
{
    const RenderConfig& cfg = scene.config;
    const int n = hits.count;
#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif
    const int chunk = (n + nt - 1) / nt;

    std::vector<int> t_nnxt(nt, 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static, chunk)
#endif
    for (int i = 0; i < n; ++i) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        HitRecord hit;
        hit.point      = { hits.point_x[i],  hits.point_y[i],  hits.point_z[i]  };
        hit.normal     = { hits.normal_x[i], hits.normal_y[i], hits.normal_z[i] };
        hit.geo_normal = { hits.geo_nx[i],   hits.geo_ny[i],   hits.geo_nz[i]   };
        hit.t          = hits.t[i];
        hit.material_id = hits.material_id[i];
        hit.shape_id   = hits.shape_id[i];
        hit.front_face = (hits.front_face[i] != 0);

        Vec3 throughput = { hits.throughput_r[i], hits.throughput_g[i], hits.throughput_b[i] };
        Vec3 radiance   = { hits.radiance_r[i],   hits.radiance_g[i],   hits.radiance_b[i]   };
        Vec3 wo         = { hits.wo_x[i],          hits.wo_y[i],          hits.wo_z[i]          };

        int  pidx        = hits.pixel_idx[i];
        int  depth       = hits.depth[i];
        uint seed        = hits.seed[i];
        bool count_emissn= (hits.count_emission[i] != 0);

        const Material& mat = scene.materials[hit.material_id];

        // ── Emissive hit ──────────────────────────────────────────────────────
        if (mat.type == MaterialType::Emissive) {
            if (count_emissn) radiance += throughput * mat.emission;
            accumulate(img, pidx, cfg.width, radiance, cfg.firefly_clamp);
            continue;
        }

        // ── Max depth ─────────────────────────────────────────────────────────
        if (depth >= cfg.max_depth) {
            accumulate(img, pidx, cfg.width, radiance, cfg.firefly_clamp);
            continue;
        }

        // ── BSDF + NEE ────────────────────────────────────────────────────────
        BSDFSample s = sample_bsdf(mat, hit, wo, seed);
        if (!s.is_specular)
            radiance += throughput * sample_direct_lighting(hit, scene, seed);

        Vec3 new_tp = throughput * s.weight;

        // ── Russian Roulette (uses local BSDF weight, matching Phase 1) ───────
        if (depth >= cfg.rr_min_depth) {
            float q = fmaxf(0.05f, 1.0f - max_component(s.weight));
            if (rand_float(seed) < q) {
                accumulate(img, pidx, cfg.width, radiance, cfg.firefly_clamp);
                continue;
            }
            new_tp = new_tp / (1.0f - q);
        }

        // ── Emit continuation ray ─────────────────────────────────────────────
        float n_sign = (dot(s.direction, hit.geo_normal) >= 0.f) ? 1.f : -1.f;
        Vec3  origin = hit.point + (n_sign * 4e-4f) * hit.geo_normal;
        Ray   new_ray = make_ray(origin, s.direction);

        int j = tid * chunk + t_nnxt[tid]++;
        nxt.origin_x[j]       = new_ray.origin.x;
        nxt.origin_y[j]       = new_ray.origin.y;
        nxt.origin_z[j]       = new_ray.origin.z;
        nxt.dir_x[j]          = new_ray.dir.x;
        nxt.dir_y[j]          = new_ray.dir.y;
        nxt.dir_z[j]          = new_ray.dir.z;
        nxt.throughput_r[j]   = new_tp.x;
        nxt.throughput_g[j]   = new_tp.y;
        nxt.throughput_b[j]   = new_tp.z;
        nxt.radiance_r[j]     = radiance.x;
        nxt.radiance_g[j]     = radiance.y;
        nxt.radiance_b[j]     = radiance.z;
        nxt.pixel_idx[j]      = pidx;
        nxt.depth[j]          = depth + 1;
        nxt.seed[j]           = seed;
        nxt.count_emission[j] = s.is_specular ? 1 : 0;
    }

    // Compact nxt continuation rays
    const int* nn = t_nnxt.data();
    compact_field(nxt.origin_x,      sizeof(float), nn, chunk, nt);
    compact_field(nxt.origin_y,      sizeof(float), nn, chunk, nt);
    compact_field(nxt.origin_z,      sizeof(float), nn, chunk, nt);
    compact_field(nxt.dir_x,         sizeof(float), nn, chunk, nt);
    compact_field(nxt.dir_y,         sizeof(float), nn, chunk, nt);
    compact_field(nxt.dir_z,         sizeof(float), nn, chunk, nt);
    compact_field(nxt.throughput_r,  sizeof(float), nn, chunk, nt);
    compact_field(nxt.throughput_g,  sizeof(float), nn, chunk, nt);
    compact_field(nxt.throughput_b,  sizeof(float), nn, chunk, nt);
    compact_field(nxt.radiance_r,    sizeof(float), nn, chunk, nt);
    compact_field(nxt.radiance_g,    sizeof(float), nn, chunk, nt);
    compact_field(nxt.radiance_b,    sizeof(float), nn, chunk, nt);
    compact_field(nxt.pixel_idx,     sizeof(int),   nn, chunk, nt);
    compact_field(nxt.depth,         sizeof(int),   nn, chunk, nt);
    compact_field(nxt.count_emission,sizeof(int),   nn, chunk, nt);
    compact_field(nxt.seed,          sizeof(uint),  nn, chunk, nt);
    nxt.count = 0;
    for (int t = 0; t < nt; ++t) nxt.count += t_nnxt[t];
}

// ── Wavefront render loop ─────────────────────────────────────────────────────

void render_wavefront_cpu(const SceneData& scene, ImageBuffer& img)
{
    const RenderConfig& cfg = scene.config;
    CameraFrame cam = make_camera_frame(scene.camera, cfg.width, cfg.height);

    const int cap         = cfg.width * cfg.height;
    const int sqrt_spp    = int(sqrtf(float(cfg.spp)));
    const bool stratified = (sqrt_spp * sqrt_spp == cfg.spp);

    // Ray sorting pays off only when the triangle BVH is too large for the CPU
    // L3 cache.  Below ~50 K triangles the BVH fits and the sort overhead hurts.
    // Threshold matches the GPU implementation.
    const bool do_sort = (scene.triangles.size() >= 50000);

    const int padded_cap = cap + 256;

    RayQueue  cur    = alloc_ray_queue(padded_cap);
    RayQueue  nxt    = alloc_ray_queue(padded_cap);
    HitQueue  hits   = alloc_hit_queue(padded_cap);
    MissQueue misses = alloc_miss_queue(padded_cap);

    std::vector<int> sorted_idx;  // reused each bounce to avoid allocation

    for (int s = 0; s < cfg.spp; ++s) {
        kernel_generate(scene, cam, s, sqrt_spp, stratified, cur);

        // GPU: each kernel_*() call below maps to a GPU kernel launch.
        // The while-loop runs on the host, checking cur.count via a small
        // cudaMemcpy each iteration to determine whether any rays remain active.
        while (cur.count > 0) {
            const std::vector<int>* idx_ptr = nullptr;
            if (do_sort) {
                sort_ray_indices(cur, sorted_idx);
                idx_ptr = &sorted_idx;
            }
            kernel_intersect(cur, scene, hits, misses, idx_ptr);
            kernel_miss(misses, scene, img);
            kernel_shade(hits, scene, nxt, img);
            std::swap(cur, nxt);
        }

        std::printf("\r  %3d%%", (s + 1) * 100 / cfg.spp);
        std::fflush(stdout);
    }
    std::printf("\n");

    img.scale(1.0f / float(cfg.spp));

    free_ray_queue(cur);
    free_ray_queue(nxt);
    free_hit_queue(hits);
    free_miss_queue(misses);
}
