// GPU wavefront render loop.
//
// Two intersection backends are supported:
//   CUDA BVH (default): software BVH traversal in intersect_kernel + CUB ray
//     sorting by direction octant + Morton origin key for L2 cache coherence.
//     NEE shadow rays are tested inline by shade_kernel (gpu_shadow_blocked).
//   OptiX RT cores (--optix): OptiX handles intersection and any-hit shadow
//     rays via RT cores. shade_kernel_deferred queues shadow candidates for
//     the subsequent optixLaunch shadow pass.
//
// Multi-sample batching (N_BATCH = 8): generate and process 8 samples per
// wavefront iteration. Gives the GPU warp scheduler more active blocks and
// hides DRAM latency better than single-sample batching.

#include "render_gpu.h"
#include "device_scene.h"
#include "common/queues.h"
#include "common/camera.h"
#include "output/image.h"
#include "scene/scene.h"

#ifdef ENABLE_OPTIX
#include "optix_state.h"
#include <fstream>
#include <sstream>
#endif

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                              \
    if (_e != cudaSuccess) {                                           \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                     __FILE__, __LINE__, cudaGetErrorString(_e));      \
        throw std::runtime_error(cudaGetErrorString(_e));              \
    }                                                                  \
} while (0)

static constexpr int N_BATCH = 8;

// Forward declarations — defined in kernels.cu
void upload_constants(const Light* lights, int nl, const Material* mats, int nm);
void setup_kernel_cache_config();
void launch_generate(DeviceScene scene, int s, int sqrt_spp, bool stratified, RayQueue ray);
void launch_intersect(RayQueue ray, DeviceScene scene,
                      HitQueue hit, int* d_hc,
                      float* image_buf,
                      const int* sorted_idx);
void launch_shade(HitQueue hit, DeviceScene scene, RayQueue nxt, int* d_nc, float* image_buf);
void launch_shade_deferred(
    HitQueue hit, DeviceScene scene, RayQueue nxt, int* d_nc, float* image_buf,
    float* sh_ox, float* sh_oy, float* sh_oz,
    float* sh_dx, float* sh_dy, float* sh_dz,
    float* sh_tmax, float* sh_cr, float* sh_cg, float* sh_cb,
    int* sh_pidx, int* d_sc);
void launch_scale(float* image_buf, int total_floats, float inv_spp);
void launch_sort_rays(RayQueue ray, int count,
                      uint32_t* d_keys_in,  uint32_t* d_keys_out,
                      int*      d_idx_in,   int*      d_idx_out,
                      void*     d_cub_tmp,  size_t    cub_tmp_bytes);

// ── Queue allocators ──────────────────────────────────────────────────────────

static RayQueue alloc_ray_queue_gpu(int cap) {
    RayQueue q{}; q.capacity = cap;
    size_t fb=size_t(cap)*sizeof(float), ib=size_t(cap)*sizeof(int), ub=size_t(cap)*sizeof(uint);
    char* p; CUDA_CHECK(cudaMalloc(&p, 12*fb+3*ib+ub));
    q.origin_x=(float*)p;p+=fb; q.origin_y=(float*)p;p+=fb; q.origin_z=(float*)p;p+=fb;
    q.dir_x=(float*)p;p+=fb;    q.dir_y=(float*)p;p+=fb;    q.dir_z=(float*)p;p+=fb;
    q.throughput_r=(float*)p;p+=fb; q.throughput_g=(float*)p;p+=fb; q.throughput_b=(float*)p;p+=fb;
    q.radiance_r=(float*)p;p+=fb;   q.radiance_g=(float*)p;p+=fb;   q.radiance_b=(float*)p;p+=fb;
    q.pixel_idx=(int*)p;p+=ib; q.depth=(int*)p;p+=ib; q.count_emission=(int*)p;p+=ib;
    q.seed=(uint*)p;
    return q;
}
static void free_ray_queue_gpu(RayQueue& q) { CUDA_CHECK(cudaFree(q.origin_x)); q={}; }

static HitQueue alloc_hit_queue_gpu(int cap) {
    HitQueue q{}; q.capacity = cap;
    size_t fb=size_t(cap)*sizeof(float), ib=size_t(cap)*sizeof(int), ub=size_t(cap)*sizeof(uint);
    char* p; CUDA_CHECK(cudaMalloc(&p, 18*fb+5*ib+ub));
    q.point_x=(float*)p;p+=fb;  q.point_y=(float*)p;p+=fb;  q.point_z=(float*)p;p+=fb;
    q.normal_x=(float*)p;p+=fb; q.normal_y=(float*)p;p+=fb; q.normal_z=(float*)p;p+=fb;
    q.geo_nx=(float*)p;p+=fb;   q.geo_ny=(float*)p;p+=fb;   q.geo_nz=(float*)p;p+=fb;
    q.wo_x=(float*)p;p+=fb; q.wo_y=(float*)p;p+=fb; q.wo_z=(float*)p;p+=fb;
    q.throughput_r=(float*)p;p+=fb; q.throughput_g=(float*)p;p+=fb; q.throughput_b=(float*)p;p+=fb;
    q.radiance_r=(float*)p;p+=fb;   q.radiance_g=(float*)p;p+=fb;   q.radiance_b=(float*)p;p+=fb;
    q.material_id=(int*)p;p+=ib; q.front_face=(int*)p;p+=ib;
    q.pixel_idx=(int*)p;p+=ib;   q.depth=(int*)p;p+=ib;    q.count_emission=(int*)p;p+=ib;
    q.seed=(uint*)p;
    return q;
}
static void free_hit_queue_gpu(HitQueue& q) { CUDA_CHECK(cudaFree(q.point_x)); q={}; }

// Offset SoA pointers so generate_kernel writes sample b into slot [b*px, …)
// without changing its interface.
static RayQueue ray_queue_slice(const RayQueue& q, int offset) {
    RayQueue s = q;
    s.origin_x += offset; s.origin_y += offset; s.origin_z += offset;
    s.dir_x    += offset; s.dir_y    += offset; s.dir_z    += offset;
    s.throughput_r += offset; s.throughput_g += offset; s.throughput_b += offset;
    s.radiance_r   += offset; s.radiance_g   += offset; s.radiance_b   += offset;
    s.pixel_idx += offset; s.depth += offset; s.count_emission += offset;
    s.seed      += offset; s.capacity -= offset;
    return s;
}

// ── Render loop ────────────────────────────────────────────────────────────────

void render_wavefront_gpu(const SceneData& scene, ImageBuffer& img, bool use_optix) {
    const RenderConfig& cfg = scene.config;
    const int W = cfg.width, H = cfg.height, total_px = W * H;

    CameraFrame cam = make_camera_frame(scene.camera, W, H);
    DeviceScene dscene = upload_scene(scene, cam);

    upload_constants(scene.lights.data(),    (int)scene.lights.size(),
                     scene.materials.data(), (int)scene.materials.size());
    setup_kernel_cache_config();

    float* d_image;
    CUDA_CHECK(cudaMalloc(&d_image, size_t(total_px) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_image, 0, size_t(total_px) * 3 * sizeof(float)));

    const int cap = N_BATCH * total_px + 256;
    RayQueue  cur  = alloc_ray_queue_gpu(cap);
    RayQueue  nxt  = alloc_ray_queue_gpu(cap);
    HitQueue  hits = alloc_hit_queue_gpu(cap);

    int *d_hc, *d_nc;
    CUDA_CHECK(cudaMalloc(&d_hc, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nc, sizeof(int)));

    int *h_hc, *h_nc;
    CUDA_CHECK(cudaMallocHost(&h_hc, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_nc, sizeof(int)));

    // ── CUDA BVH path: sort buffers ───────────────────────────────────────────
    // Ray sorting improves L2 cache reuse for large meshes. Preallocate once.
    uint32_t *d_keys_in  = nullptr, *d_keys_out = nullptr;
    int       *d_idx_in  = nullptr, *d_idx_out  = nullptr;
    void*     d_cub_tmp  = nullptr;
    size_t    cub_tmp_bytes = 0;

    // Ray sorting pays off when the triangle BVH is too large to stay in L2.
    const bool do_sort = !use_optix && (dscene.num_triangles >= 50000);

    if (!use_optix) {
        CUDA_CHECK(cudaMalloc(&d_keys_in,  size_t(cap) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_out, size_t(cap) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_idx_in,   size_t(cap) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_idx_out,  size_t(cap) * sizeof(int)));
        // Dry run to determine CUB temp-storage size.
        cub::DeviceRadixSort::SortPairs(
            d_cub_tmp, cub_tmp_bytes,
            d_keys_in, d_keys_out, d_idx_in, d_idx_out, cap);
        CUDA_CHECK(cudaMalloc(&d_cub_tmp, cub_tmp_bytes));
    }

    // ── OptiX path: shadow queue + state ─────────────────────────────────────
    float *sh_ox=nullptr, *sh_oy=nullptr, *sh_oz=nullptr;
    float *sh_dx=nullptr, *sh_dy=nullptr, *sh_dz=nullptr, *sh_tmax=nullptr;
    float *sh_cr=nullptr, *sh_cg=nullptr, *sh_cb=nullptr;
    int   *sh_pidx=nullptr, *d_sc=nullptr, *h_sc=nullptr;

#ifdef ENABLE_OPTIX
    OptixState optix_state{};
    if (use_optix) {
        size_t fb = size_t(cap) * sizeof(float);
        size_t ib = size_t(cap) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&sh_ox,   fb)); CUDA_CHECK(cudaMalloc(&sh_oy,   fb));
        CUDA_CHECK(cudaMalloc(&sh_oz,   fb)); CUDA_CHECK(cudaMalloc(&sh_dx,   fb));
        CUDA_CHECK(cudaMalloc(&sh_dy,   fb)); CUDA_CHECK(cudaMalloc(&sh_dz,   fb));
        CUDA_CHECK(cudaMalloc(&sh_tmax, fb));
        CUDA_CHECK(cudaMalloc(&sh_cr,   fb)); CUDA_CHECK(cudaMalloc(&sh_cg,   fb));
        CUDA_CHECK(cudaMalloc(&sh_cb,   fb));
        CUDA_CHECK(cudaMalloc(&sh_pidx, ib));
        CUDA_CHECK(cudaMalloc(&d_sc,    sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_sc, sizeof(int)));

        std::ifstream ptx_file(OPTIX_PTX_FILE);
        if (!ptx_file)
            throw std::runtime_error(std::string("Cannot open PTX: ") + OPTIX_PTX_FILE);
        std::string ptx_src((std::istreambuf_iterator<char>(ptx_file)),
                             std::istreambuf_iterator<char>());
        std::printf("  Building OptiX acceleration structures ... ");
        std::fflush(stdout);
        optix_state = build_optix_state(dscene, ptx_src.c_str());
        std::printf("done\n");
    }
#else
    if (use_optix) {
        std::fprintf(stderr, "OptiX backend requested but not compiled in. "
                             "Rebuild with -DENABLE_OPTIX=ON.\n");
        throw std::runtime_error("OptiX not available");
    }
#endif

    const int  sqrt_spp   = (int)sqrtf((float)cfg.spp);
    const bool stratified = (sqrt_spp * sqrt_spp == cfg.spp);

    // ── Sample loop ───────────────────────────────────────────────────────────
    for (int s = 0; s < cfg.spp; s += N_BATCH) {
        int batch = std::min(N_BATCH, cfg.spp - s);

        for (int b = 0; b < batch; ++b) {
            RayQueue slice = ray_queue_slice(cur, b * total_px);
            launch_generate(dscene, s + b, sqrt_spp, stratified, slice);
        }
        cur.count = batch * total_px;

        while (cur.count > 0) {
            CUDA_CHECK(cudaMemset(d_hc, 0, sizeof(int)));

#ifdef ENABLE_OPTIX
            if (use_optix) {
                optix_intersect(optix_state, dscene, cur, hits, d_hc, d_image);
            } else
#endif
            {
                const int* sorted_idx = nullptr;
                if (do_sort) {
                    launch_sort_rays(cur, cur.count,
                                     d_keys_in, d_keys_out,
                                     d_idx_in,  d_idx_out,
                                     d_cub_tmp, cub_tmp_bytes);
                    sorted_idx = d_idx_out;
                }
                launch_intersect(cur, dscene, hits, d_hc, d_image, sorted_idx);
            }

            CUDA_CHECK(cudaMemcpy(h_hc, d_hc, sizeof(int), cudaMemcpyDeviceToHost));
            hits.count = *h_hc;

            CUDA_CHECK(cudaMemset(d_nc, 0, sizeof(int)));

#ifdef ENABLE_OPTIX
            if (use_optix) {
                CUDA_CHECK(cudaMemset(d_sc, 0, sizeof(int)));
                launch_shade_deferred(hits, dscene, nxt, d_nc, d_image,
                                      sh_ox, sh_oy, sh_oz,
                                      sh_dx, sh_dy, sh_dz,
                                      sh_tmax, sh_cr, sh_cg, sh_cb,
                                      sh_pidx, d_sc);
                CUDA_CHECK(cudaMemcpy(h_nc, d_nc, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_sc, d_sc, sizeof(int), cudaMemcpyDeviceToHost));
                nxt.count = *h_nc;
                optix_shadow(optix_state,
                             sh_ox, sh_oy, sh_oz, sh_dx, sh_dy, sh_dz, sh_tmax,
                             sh_cr, sh_cg, sh_cb, sh_pidx, *h_sc, d_image);
            } else
#endif
            {
                launch_shade(hits, dscene, nxt, d_nc, d_image);
                CUDA_CHECK(cudaMemcpy(h_nc, d_nc, sizeof(int), cudaMemcpyDeviceToHost));
                nxt.count = *h_nc;
            }

            { RayQueue tmp = cur; cur = nxt; nxt = tmp; }
        }

        int done = std::min(s + batch, cfg.spp);
        if ((done * 20) % cfg.spp == 0 || done == cfg.spp) {
            std::printf("\r  GPU %3d%%", done * 100 / cfg.spp);
            std::fflush(stdout);
        }
    }
    std::printf("\n");

    launch_scale(d_image, total_px * 3, 1.f / float(cfg.spp));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host_buf(size_t(total_px) * 3);
    CUDA_CHECK(cudaMemcpy(host_buf.data(), d_image,
                          size_t(total_px) * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    img.data = std::move(host_buf);

    // ── Cleanup ───────────────────────────────────────────────────────────────
#ifdef ENABLE_OPTIX
    if (use_optix) {
        destroy_optix_state(optix_state);
        cudaFree(sh_ox); cudaFree(sh_oy); cudaFree(sh_oz);
        cudaFree(sh_dx); cudaFree(sh_dy); cudaFree(sh_dz); cudaFree(sh_tmax);
        cudaFree(sh_cr); cudaFree(sh_cg); cudaFree(sh_cb);
        cudaFree(sh_pidx); cudaFree(d_sc); cudaFreeHost(h_sc);
    }
#endif
    if (!use_optix) {
        cudaFree(d_keys_in); cudaFree(d_keys_out);
        cudaFree(d_idx_in);  cudaFree(d_idx_out);
        cudaFree(d_cub_tmp);
    }
    cudaFree(d_image);
    cudaFree(d_hc); cudaFree(d_nc);
    cudaFreeHost(h_hc); cudaFreeHost(h_nc);
    free_ray_queue_gpu(cur); free_ray_queue_gpu(nxt);
    free_hit_queue_gpu(hits);
    free_device_scene(dscene);
}
