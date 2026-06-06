// CUDA kernels for the GPU wavefront path tracer.
//
// shade_kernel          — CUDA BVH backend; inline gpu_shadow_blocked for NEE.
// shade_kernel_deferred — OptiX backend; emits shadow candidates to ShadowQueue.

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "gpu_bvh.cuh"
#include "gpu_warp.cuh"
#include "device_scene.h"
#include "common/queues.h"
#include "shading/bsdf.h"
#include "shading/sampling.h"
#include "common/camera.h"

// ── Constant memory ───────────────────────────────────────────────────────────
static constexpr int MAX_CONST_LIGHTS    = 64;
static constexpr int MAX_CONST_MATERIALS = 256;

__constant__ char c_lights_raw   [MAX_CONST_LIGHTS    * sizeof(Light)];
__constant__ char c_materials_raw[MAX_CONST_MATERIALS * sizeof(Material)];
__constant__ int  c_num_lights;
__constant__ int  c_num_materials;

__device__ __forceinline__ const Light& get_light(int i) {
    return *reinterpret_cast<const Light*>(c_lights_raw + i * sizeof(Light));
}
__device__ __forceinline__ const Material& get_material(int i) {
    return *reinterpret_cast<const Material*>(c_materials_raw + i * sizeof(Material));
}

void upload_constants(const Light* lights, int nl, const Material* mats, int nm) {
    if (nl > MAX_CONST_LIGHTS || nm > MAX_CONST_MATERIALS) return;
    cudaMemcpyToSymbol(c_lights_raw,    lights, nl * sizeof(Light));
    cudaMemcpyToSymbol(c_materials_raw, mats,   nm * sizeof(Material));
    cudaMemcpyToSymbol(c_num_lights,    &nl, sizeof(int));
    cudaMemcpyToSymbol(c_num_materials, &nm, sizeof(int));
}

// ── Helpers ───────────────────────────────────────────────────────────────────

__device__ __forceinline__ float halton(int i, int base) {
    float result = 0.f, f = 1.f;
    while (i > 0) { f /= base; result += f * (i % base); i /= base; }
    return result;
}

__device__ __forceinline__
Vec3 gpu_background(Vec3 dir, const DevRenderConfig& cfg) {
    if (cfg.bg_mode == 1) {
        float t = 0.5f * (normalize(dir).y + 1.0f);
        return lerp(Vec3(1.f), Vec3(0.5f, 0.7f, 1.0f), t);
    }
    return Vec3(cfg.bg_r, cfg.bg_g, cfg.bg_b);
}

__device__ __forceinline__
Vec3 apply_firefly_clamp(Vec3 c, float limit) {
    if (limit <= 0.f) return c;
    float l = luma(c);
    if (l > limit) c = c * (limit / l);
    return c;
}

// ── NEE helpers ───────────────────────────────────────────────────────────────

struct NeeCandidate {
    Vec3  shadow_o;
    Vec3  shadow_d;
    float shadow_t;
    Vec3  contrib;
    bool  valid;
};

__device__ NeeCandidate gpu_nee_sample(const HitRecord& hit,
                                        const DeviceScene& scene, uint& seed)
{
    NeeCandidate c{}; c.valid = false;
    if (c_num_lights == 0) return c;

    int li = (int)(rand_float(seed) * c_num_lights) % c_num_lights;
    const Light& light = get_light(li);
    float inv_n = 1.f / (float)c_num_lights;

    const Material& surf = get_material(hit.material_id);
    Vec3 brdf = surf.albedo * PT_INV_PI;

    auto spawn = [&](Vec3 wi) -> Vec3 {
        float sg = dot(wi, hit.geo_normal) >= 0.f ? 1.f : -1.f;
        return hit.point + (sg * 4e-4f) * hit.geo_normal;
    };

    if (light.type == LightType::Area) {
        int sid = light.area.shape_id;
        if (sid < 0 || sid >= scene.num_shapes) return c;
        const Shape&    ls = scene.shapes[sid];
        const Material& lm = get_material(ls.material_id);
        if (lm.type != MaterialType::Emissive) return c;

        if (ls.type == ShapeType::Sphere) {
            const Sphere& sph = ls.sphere;
            Vec3  sp   = uniform_sample_sphere(rand_float(seed), rand_float(seed));
            Vec3  lp   = sph.center + sph.radius * sp;
            Vec3  to   = lp - hit.point;
            float dist2 = length_sq(to), dist = sqrtf(dist2);
            if (dist < 1e-8f) return c;
            Vec3  wi   = to / dist;
            float cs   = dot(hit.normal, wi);  if (cs <= 0.f) return c;
            float cl   = dot(-wi, sp);          if (cl <= 0.f) return c;
            float p_omega = dist2 / (cl * 4.f * PT_PI * sph.radius * sph.radius);
            c.shadow_o = spawn(wi); c.shadow_d = wi; c.shadow_t = dist * 0.999f;
            c.contrib  = lm.emission * brdf * cs / (p_omega * inv_n);
            c.valid    = true; return c;

        } else if (ls.type == ShapeType::Disk) {
            const Disk& disk = ls.disk;
            Vec3 dn = normalize(disk.normal), dt, db; make_onb(dn, dt, db);
            float r   = disk.radius * sqrtf(rand_float(seed));
            float phi = PT_TWO_PI * rand_float(seed);
            Vec3 lp = disk.center + r * (cosf(phi)*dt + sinf(phi)*db);
            Vec3 to = lp - hit.point;
            float dist2 = length_sq(to), dist = sqrtf(dist2);
            if (dist < 1e-8f) return c;
            Vec3 wi = to / dist;
            float cs = dot(hit.normal, wi);  if (cs <= 0.f) return c;
            float cl = dot(-wi, dn);          if (cl <= 0.f) return c;
            float p_omega = dist2 / (cl * PT_PI * disk.radius * disk.radius);
            c.shadow_o = spawn(wi); c.shadow_d = wi; c.shadow_t = dist * 0.999f;
            c.contrib  = lm.emission * brdf * cs / (p_omega * inv_n);
            c.valid    = true; return c;
        }
        return c;

    } else if (light.type == LightType::Point) {
        const PointLight& pl = light.point;
        Vec3  to   = pl.position - hit.point;
        float dist2 = length_sq(to), dist = sqrtf(dist2);
        if (dist < 1e-8f) return c;
        Vec3  wi   = to / dist;
        float cs   = dot(hit.normal, wi); if (cs <= 0.f) return c;
        c.shadow_o = spawn(wi); c.shadow_d = wi; c.shadow_t = dist - 1e-3f;
        c.contrib  = pl.color * pl.intensity * brdf * cs / (dist2 * inv_n);
        c.valid    = true; return c;

    } else if (light.type == LightType::Directional) {
        const DirectionalLight& dl = light.directional;
        Vec3  wi  = normalize(-dl.direction);
        float cs  = dot(hit.normal, wi); if (cs <= 0.f) return c;
        c.shadow_o = spawn(wi); c.shadow_d = wi; c.shadow_t = PT_INFINITY;
        c.contrib  = dl.color * dl.intensity * brdf * cs / inv_n;
        c.valid    = true; return c;
    }
    return c;
}

// NEE with inline BVH shadow test (used by shade_kernel in the CUDA BVH path).
__device__ Vec3 gpu_nee(const HitRecord& hit, const DeviceScene& scene, uint& seed) {
    NeeCandidate c = gpu_nee_sample(hit, scene, seed);
    if (!c.valid) return Vec3(0.f);
    Ray shadow = { c.shadow_o, c.shadow_d, PT_EPSILON, c.shadow_t };
    if (gpu_shadow_blocked(shadow, scene.bvh_nodes, scene.prim_indices,
                           scene.bvh_num_shapes, scene.shapes, scene.tri_verts))
        return Vec3(0.f);
    return c.contrib;
}

// ── Morton helpers (ray sort key) ─────────────────────────────────────────────

__device__ __forceinline__ uint32_t morton_expand3(uint32_t v) {
    v &= 0x3ffu;
    v = (v | (v << 16u)) & 0x030000ffu;
    v = (v | (v <<  8u)) & 0x0300f00fu;
    v = (v | (v <<  4u)) & 0x030c30c3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3d(uint32_t x, uint32_t y, uint32_t z) {
    return morton_expand3(x) | (morton_expand3(y) << 1u) | (morton_expand3(z) << 2u);
}

// ── Ray key kernel ────────────────────────────────────────────────────────────
// key = (direction_octant << 29) | morton29(quantised_origin)
// Rays with matching keys traverse the same BVH subtrees, improving L2 reuse.
__global__
void compute_ray_keys_kernel(
    const float* __restrict__ origin_x,
    const float* __restrict__ origin_y,
    const float* __restrict__ origin_z,
    const float* __restrict__ dir_x,
    const float* __restrict__ dir_y,
    const float* __restrict__ dir_z,
    int count, uint32_t* __restrict__ keys, int* __restrict__ indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float ox = origin_x[i], oy = origin_y[i], oz = origin_z[i];
    float dx = dir_x[i],    dy = dir_y[i],    dz = dir_z[i];

    uint32_t octant = (uint32_t)(dx < 0.f)
                    | ((uint32_t)(dy < 0.f) << 1u)
                    | ((uint32_t)(dz < 0.f) << 2u);

    // Quantise origin into a 1024^3 grid spanning a [-kHalf, +kHalf] world cube.
    // Scenes outside this range still sort correctly (they clamp to grid edges)
    // but lose locality. All current scenes fit comfortably inside ±20.
    constexpr float kHalf   = 20.0f;
    constexpr uint32_t kRes = 1024;
    constexpr float kScale  = float(kRes) / (2.0f * kHalf);
    uint32_t qx = min((uint32_t)max(0.f, (ox + kHalf) * kScale), kRes - 1);
    uint32_t qy = min((uint32_t)max(0.f, (oy + kHalf) * kScale), kRes - 1);
    uint32_t qz = min((uint32_t)max(0.f, (oz + kHalf) * kScale), kRes - 1);

    keys[i]    = (octant << 29u) | (morton3d(qx, qy, qz) & 0x1fffffffu);
    indices[i] = i;
}

// ── GenerateRays kernel ───────────────────────────────────────────────────────
__global__
__launch_bounds__(256, 4)
void generate_kernel(DeviceScene scene, int s, int sqrt_spp, bool stratified,
                     RayQueue ray)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_px = scene.config.width * scene.config.height;
    if (i >= total_px) return;

    int px = i % scene.config.width;
    int py = i / scene.config.width;
    uint seed = wang_hash(uint(i) + uint(s) * uint(total_px) + 1u);

    float ou, ov;
    if (stratified) {
        int sx = s % sqrt_spp, sy = s / sqrt_spp;
        ou = (float(sx) + rand_float(seed)) / float(sqrt_spp);
        ov = (float(sy) + rand_float(seed)) / float(sqrt_spp);
    } else {
        ou = fmodf(halton(s + 1, 2) + rand_float(seed) * (1.f/32.f), 1.f);
        ov = fmodf(halton(s + 1, 3) + rand_float(seed) * (1.f/32.f), 1.f);
    }

    float lu = 0.f, lv = 0.f;
    if (scene.config.aperture > 0.f) {
        float r   = sqrtf(rand_float(seed));
        float phi = PT_TWO_PI * rand_float(seed);
        lu = r * cosf(phi); lv = r * sinf(phi);
    }

    Ray r = camera_ray(scene.cam, px, py, ou, ov,
                       scene.config.width, scene.config.height,
                       lu, lv, scene.config.aperture, scene.config.focus_dist);

    ray.origin_x[i] = r.origin.x; ray.origin_y[i] = r.origin.y; ray.origin_z[i] = r.origin.z;
    ray.dir_x[i]    = r.dir.x;    ray.dir_y[i]    = r.dir.y;    ray.dir_z[i]    = r.dir.z;
    ray.throughput_r[i] = 1.f; ray.throughput_g[i] = 1.f; ray.throughput_b[i] = 1.f;
    ray.radiance_r[i]   = 0.f; ray.radiance_g[i]   = 0.f; ray.radiance_b[i]   = 0.f;
    ray.pixel_idx[i]      = i;
    ray.depth[i]          = 0;
    ray.seed[i]           = seed;
    ray.count_emission[i] = 1;
}

// ── Intersect kernel (CUDA BVH path) ─────────────────────────────────────────
// sorted_idx: when non-null, thread i processes ray sorted_idx[i] (gather read).
// Misses are written directly into image_buf here, eliminating the MissQueue
// roundtrip (~250-450us miss_kernel per bounce).
__global__
__launch_bounds__(128, 6)
void intersect_kernel(
    RayQueue ray, DeviceScene scene,
    HitQueue hit, int* d_hit_count,
    float* image_buf,
    const int* sorted_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ray.count) return;

    int ri = sorted_idx ? sorted_idx[i] : i;

    Ray r;
    r.origin = { ray.origin_x[ri], ray.origin_y[ri], ray.origin_z[ri] };
    r.dir    = { ray.dir_x[ri],    ray.dir_y[ri],    ray.dir_z[ri]    };
    r.tmin   = PT_EPSILON;
    r.tmax   = PT_INFINITY;

    HitRecord rec{};
    bool did_hit = gpu_bvh_intersect(r, scene.bvh_nodes, scene.prim_indices,
                                     scene.bvh_num_shapes,
                                     scene.shapes, scene.tri_verts, scene.triangles, rec);

    int j = warp_compact_slot(d_hit_count, did_hit);
    if (did_hit) {
        hit.point_x[j]=rec.point.x;   hit.point_y[j]=rec.point.y;   hit.point_z[j]=rec.point.z;
        hit.normal_x[j]=rec.normal.x; hit.normal_y[j]=rec.normal.y; hit.normal_z[j]=rec.normal.z;
        hit.geo_nx[j]=rec.geo_normal.x; hit.geo_ny[j]=rec.geo_normal.y; hit.geo_nz[j]=rec.geo_normal.z;
        hit.material_id[j]=rec.material_id;
        hit.front_face[j]=rec.front_face ? 1 : 0;
        hit.wo_x[j]=-ray.dir_x[ri]; hit.wo_y[j]=-ray.dir_y[ri]; hit.wo_z[j]=-ray.dir_z[ri];
        hit.throughput_r[j]=ray.throughput_r[ri]; hit.throughput_g[j]=ray.throughput_g[ri]; hit.throughput_b[j]=ray.throughput_b[ri];
        hit.radiance_r[j]=ray.radiance_r[ri];     hit.radiance_g[j]=ray.radiance_g[ri];     hit.radiance_b[j]=ray.radiance_b[ri];
        hit.pixel_idx[j]=ray.pixel_idx[ri]; hit.depth[j]=ray.depth[ri];
        hit.seed[j]=ray.seed[ri]; hit.count_emission[j]=ray.count_emission[ri];
    } else {
        Vec3 dir = { ray.dir_x[ri], ray.dir_y[ri], ray.dir_z[ri] };
        Vec3 tp  = { ray.throughput_r[ri], ray.throughput_g[ri], ray.throughput_b[ri] };
        Vec3 rad = { ray.radiance_r[ri],   ray.radiance_g[ri],   ray.radiance_b[ri]   };
        Vec3 out = apply_firefly_clamp(rad + tp * gpu_background(dir, scene.config),
                                       scene.config.firefly_clamp);
        int pidx = ray.pixel_idx[ri];
        atomicAdd(&image_buf[pidx*3+0], out.x);
        atomicAdd(&image_buf[pidx*3+1], out.y);
        atomicAdd(&image_buf[pidx*3+2], out.z);
    }
}

// Shadow-output bundle.  Used only when DEFERRED == true; pass zero-filled
// when DEFERRED == false (dead code stripped by the optimiser).
struct ShadowOut {
    float *ox, *oy, *oz;
    float *dx, *dy, *dz;
    float *tmax;
    float *cr, *cg, *cb;
    int   *pidx;
    int   *d_count;
};

// ── Shade kernel ─────────────────────────────────────────────────────────────
// Single template body shared by the CUDA BVH and OptiX backends.
//   DEFERRED == false  →  inline software shadow ray via gpu_nee() (CUDA BVH)
//   DEFERRED == true   →  emit shadow candidate to ShadowOut for optixLaunch
template<bool DEFERRED>
__global__
__launch_bounds__(128, 6)
void shade_kernel_impl(
    HitQueue hit, DeviceScene scene,
    RayQueue nxt, int* d_nxt_count,
    float* image_buf,
    ShadowOut sh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hit.count) return;

    HitRecord rec;
    rec.point      = { hit.point_x[i],  hit.point_y[i],  hit.point_z[i]  };
    rec.normal     = { hit.normal_x[i], hit.normal_y[i], hit.normal_z[i] };
    rec.geo_normal = { hit.geo_nx[i],   hit.geo_ny[i],   hit.geo_nz[i]   };
    rec.material_id = hit.material_id[i];
    rec.front_face = (hit.front_face[i] != 0);

    Vec3 tp  = { hit.throughput_r[i], hit.throughput_g[i], hit.throughput_b[i] };
    Vec3 rad = { hit.radiance_r[i],   hit.radiance_g[i],   hit.radiance_b[i]   };
    Vec3 wo  = { hit.wo_x[i], hit.wo_y[i], hit.wo_z[i] };

    int  pidx      = hit.pixel_idx[i];
    int  depth     = hit.depth[i];
    uint seed      = hit.seed[i];
    bool cnt_emiss = (hit.count_emission[i] != 0);

    const Material& mat = get_material(rec.material_id);

    bool terminated = false;
    Vec3 term_rad   = rad;
    Vec3 cont_orig(0.f), cont_dir(0.f), cont_tp(0.f), cont_rad(0.f);
    int  cont_depth = 0, cont_emit = 0;
    uint cont_seed  = seed;

    bool has_shadow = false;
    NeeCandidate nee_c{};

    if (mat.type == MaterialType::Emissive) {
        if (cnt_emiss) term_rad = rad + tp * mat.emission;
        terminated = true;

    } else if (depth >= scene.config.max_depth) {
        terminated = true;

    } else {
        BSDFSample bsdf = sample_bsdf(mat, rec, wo, seed);
        Vec3 new_rad = rad;

        if (!bsdf.is_specular) {
            if constexpr (DEFERRED) {
                nee_c = gpu_nee_sample(rec, scene, seed);
                has_shadow = nee_c.valid;
            } else {
                new_rad = rad + tp * gpu_nee(rec, scene, seed);
            }
        }

        Vec3 new_tp = tp * bsdf.weight;
        bool rr_kill = false;
        if (depth >= scene.config.rr_min_depth) {
            float q = fmaxf(0.05f, 1.f - max_component(bsdf.weight));
            if (rand_float(seed) < q) {
                rr_kill = true;
            } else {
                new_tp = new_tp / (1.f - q);
            }
        }

        if (rr_kill) {
            term_rad = new_rad; terminated = true;
        } else {
            float sgn  = (dot(bsdf.direction, rec.geo_normal) >= 0.f) ? 1.f : -1.f;
            cont_orig  = rec.point + (sgn * 4e-4f) * rec.geo_normal;
            cont_dir   = bsdf.direction;
            cont_tp    = new_tp;
            cont_rad   = new_rad;
            cont_depth = depth + 1;
            cont_seed  = seed;
            cont_emit  = bsdf.is_specular ? 1 : 0;
        }
    }

    if (terminated) {
        Vec3 out = apply_firefly_clamp(term_rad, scene.config.firefly_clamp);
        atomicAdd(&image_buf[pidx*3+0], out.x);
        atomicAdd(&image_buf[pidx*3+1], out.y);
        atomicAdd(&image_buf[pidx*3+2], out.z);
    }

    // Continuation ray
    int j = warp_compact_slot(d_nxt_count, !terminated);
    if (!terminated) {
        nxt.origin_x[j]=cont_orig.x; nxt.origin_y[j]=cont_orig.y; nxt.origin_z[j]=cont_orig.z;
        nxt.dir_x[j]=cont_dir.x; nxt.dir_y[j]=cont_dir.y; nxt.dir_z[j]=cont_dir.z;
        nxt.throughput_r[j]=cont_tp.x; nxt.throughput_g[j]=cont_tp.y; nxt.throughput_b[j]=cont_tp.z;
        nxt.radiance_r[j]=cont_rad.x; nxt.radiance_g[j]=cont_rad.y; nxt.radiance_b[j]=cont_rad.z;
        nxt.pixel_idx[j]=pidx; nxt.depth[j]=cont_depth; nxt.seed[j]=cont_seed;
        nxt.count_emission[j]=cont_emit;
    }

    // Deferred-only: emit shadow candidate
    if constexpr (DEFERRED) {
        int k = warp_compact_slot(sh.d_count, has_shadow);
        if (has_shadow) {
            sh.ox[k]=nee_c.shadow_o.x; sh.oy[k]=nee_c.shadow_o.y; sh.oz[k]=nee_c.shadow_o.z;
            sh.dx[k]=nee_c.shadow_d.x; sh.dy[k]=nee_c.shadow_d.y; sh.dz[k]=nee_c.shadow_d.z;
            sh.tmax[k]=nee_c.shadow_t;
            Vec3 sc = apply_firefly_clamp(tp * nee_c.contrib, scene.config.firefly_clamp);
            sh.cr[k]=sc.x; sh.cg[k]=sc.y; sh.cb[k]=sc.z;
            sh.pidx[k]=pidx;
        }
    }
}

// ── Scale kernel ──────────────────────────────────────────────────────────────
__global__ void scale_kernel(float* image_buf, int n, float inv_spp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) image_buf[i] *= inv_spp;
}

// ── Launch wrappers ───────────────────────────────────────────────────────────

void launch_generate(DeviceScene scene, int s, int sqrt_spp, bool stratified, RayQueue ray) {
    int n = scene.config.width * scene.config.height;
    generate_kernel<<<(n+255)/256, 256>>>(scene, s, sqrt_spp, stratified, ray);
}

void launch_intersect(RayQueue ray, DeviceScene scene,
                      HitQueue hit, int* d_hit_count,
                      float* image_buf,
                      const int* sorted_idx)
{
    if (ray.count == 0) return;
    intersect_kernel<<<(ray.count+127)/128, 128>>>(
        ray, scene, hit, d_hit_count, image_buf, sorted_idx);
}

void launch_shade(HitQueue hit, DeviceScene scene, RayQueue nxt, int* d_nxt_count,
                  float* image_buf)
{
    if (hit.count == 0) return;
    ShadowOut sh{};  // unused when DEFERRED == false
    shade_kernel_impl<false><<<(hit.count+127)/128, 128>>>(
        hit, scene, nxt, d_nxt_count, image_buf, sh);
}

void launch_shade_deferred(
    HitQueue hit, DeviceScene scene,
    RayQueue nxt, int* d_nxt_count,
    float* image_buf,
    float* sh_ox, float* sh_oy, float* sh_oz,
    float* sh_dx, float* sh_dy, float* sh_dz,
    float* sh_tmax, float* sh_cr, float* sh_cg, float* sh_cb,
    int* sh_pidx, int* d_sh_count)
{
    if (hit.count == 0) return;
    ShadowOut sh{ sh_ox, sh_oy, sh_oz, sh_dx, sh_dy, sh_dz,
                  sh_tmax, sh_cr, sh_cg, sh_cb, sh_pidx, d_sh_count };
    shade_kernel_impl<true><<<(hit.count+127)/128, 128>>>(
        hit, scene, nxt, d_nxt_count, image_buf, sh);
}

void launch_scale(float* image_buf, int total_floats, float inv_spp) {
    scale_kernel<<<(total_floats+255)/256, 256>>>(image_buf, total_floats, inv_spp);
}

void launch_sort_rays(
    RayQueue ray, int count,
    uint32_t* d_keys_in,  uint32_t* d_keys_out,
    int*      d_idx_in,   int*      d_idx_out,
    void*     d_cub_tmp,  size_t    cub_tmp_bytes)
{
    if (count == 0) return;
    compute_ray_keys_kernel<<<(count+255)/256, 256>>>(
        ray.origin_x, ray.origin_y, ray.origin_z,
        ray.dir_x, ray.dir_y, ray.dir_z,
        count, d_keys_in, d_idx_in);
    cub::DeviceRadixSort::SortPairs(
        d_cub_tmp, cub_tmp_bytes,
        d_keys_in, d_keys_out,
        d_idx_in,  d_idx_out,
        count);
}

void setup_kernel_cache_config() {
    cudaFuncSetCacheConfig(intersect_kernel, cudaFuncCachePreferL1);
}
