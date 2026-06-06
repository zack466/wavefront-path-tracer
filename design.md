# Wavefront Path Tracer — Design Document

## 1. Scope and Goals

A physically-based Monte Carlo path tracer that:
- Produces photorealistic images via unbiased path tracing with NEE (next-event estimation)
- Runs on CPU (reference/debug) and GPU (production via OptiX RT cores) using shared shading logic
- Supports Lambertian diffuse, perfect mirror/glass, rough-glass (GGX), and emissive materials
- Supports sphere, cylinder, disk, and triangle-mesh (OBJ) shapes
- Supports point, directional, and area lights
- Uses a wavefront (queue-based) architecture on both CPU and GPU

**Four renderer implementations:**
1. **CPU recursive** (Phase 1) — straightforward recursive path tracer, OpenMP parallel
2. **CPU wavefront** (Phase 2) — same algorithm as the GPU wavefront, but running on CPU with OpenMP
3. **GPU wavefront — CUDA BVH** (Phase 3a) — wavefront renderer using a software SAH BVH on the GPU; CUB radix-sort ray coherence; NEE shadow rays inline via `gpu_shadow_blocked`
4. **GPU wavefront — OptiX** (Phase 3b) — same wavefront loop, but intersection and any-hit shadow queries delegated to OptiX RT cores via two `optixLaunch` calls per bounce

---

## 2. Repository Layout

```
final-project/
├── CMakeLists.txt
├── benchmark.py             # performance measurement script
├── scenes/                  JSON scene files
├── models/                  OBJ triangle meshes (bunny, dragon, …)
├── outputs/                 rendered images
├── vendored/
│   ├── json/                nlohmann/json (single-header)
│   ├── stb/                 stb_image_write.h (PNG output)
│   └── optix/               OptiX 8.1.0 headers (no libs needed)
└── src/
    ├── common/              HD-annotated headers shared by CPU and GPU
    │   ├── types.h          HD macro, PT_* constants
    │   ├── math.h           Vec3, Ray, helpers
    │   ├── hit.h            HitRecord
    │   ├── bvh_node.h       BVHNode (STL-free for CUDA device code)
    │   ├── material.h       MaterialType enum + Material struct
    │   ├── shape.h          ShapeType + analytic shape structs
    │   ├── triangle.h       Triangle + TriangleV structs
    │   ├── aabb.h           AABB struct + per-shape AABB helpers
    │   ├── intersect.h      Analytic + Möller-Trumbore intersection (HD)
    │   ├── light.h          LightType + Light struct
    │   ├── camera.h         CameraParams, CameraFrame, camera_ray()
    │   └── queues.h         RayQueue, HitQueue, MissQueue, ShadowQueue SoA structs
    ├── bvh/                 CPU SAH BVH builder + traversal (used by CPU renderers)
    ├── scene/               JSON scene loader + OBJ mesh loader
    ├── shading/             BSDF + sampling utilities (HD, shared by all renderers)
    ├── cpu/                 CPU path tracers
    │   ├── render_cpu.cpp   Phase 1: recursive reference renderer
    │   └── wavefront_cpu.cpp Phase 2: wavefront renderer
    ├── gpu/                 Phase 3: GPU OptiX wavefront
    │   ├── device_scene.h/.cu    Upload SceneData → device memory (DeviceScene)
    │   ├── optix_params.h        LaunchParams struct shared between host + device programs
    │   ├── optix_programs.cu     OptiX device programs (raygen, miss, closesthit, shadow)
    │   ├── optix_state.h/.cu     AS build, pipeline, SBT, optix_intersect/shadow
    │   ├── kernels.cu            generate, miss, shade_deferred, scale CUDA kernels
    │   ├── render_gpu.cu         Wavefront loop (always OptiX)
    │   └── render_gpu.h
    ├── output/              Float32 image buffer + tone mapping + PNG/PPM write
    ├── main_cpu.cpp         CPU entry point
    └── main_gpu.cu          GPU entry point
```

---

## 3. Core Data Structures

All structs carry the `HD` (`__host__ __device__ __forceinline__`) annotation so they compile on both CPU and GPU.

### 3.1 Math (`common/math.h`)
```cpp
struct Vec3 { float x, y, z; };
struct Ray  { Vec3 origin, dir; float tmin, tmax; };
```

### 3.2 HitRecord (`common/hit.h`)
```cpp
struct HitRecord {
    Vec3  point;
    Vec3  normal;       // shading normal, opposing ray direction
    Vec3  geo_normal;   // geometric face normal (for spawn offset on meshes)
    float t;
    int   material_id;
    int   shape_id;
    bool  front_face;
};
```

### 3.3 Material (`common/material.h`)
```cpp
enum class MaterialType : uint8_t {
    Diffuse, SpecularIdeal, DielectricIdeal, Emissive, GGXDielectric
};
struct Material {
    MaterialType type;
    Vec3  albedo;
    Vec3  emission;
    float ior;
    float roughness;   // GGX roughness [0,1]
    float metallic;    // reserved for future GGX conductor
};
```

### 3.4 Shape and Triangle (`common/shape.h`, `common/triangle.h`)
```cpp
// Analytic shapes for CPU BVH and NEE area-light sampling
struct Shape { ShapeType type; union{Sphere; Cylinder; Disk}; int material_id; };

// Full triangle for shading normals
struct Triangle { Vec3 v[3]; Vec3 n[3]; bool smooth_normals; int material_id; };

// Compact vertex-only layout for OptiX CH normal lookup (36 bytes)
struct TriangleV { Vec3 v[3]; };
```

### 3.5 Wavefront Queues (`common/queues.h`)

All queues use **Struct-of-Arrays (SoA)** layout for coalesced GPU memory access. A single `cudaMalloc` allocation backs all arrays in each queue.

| Queue | Purpose |
|---|---|
| `RayQueue` | Active rays: geometry, throughput, radiance, metadata |
| `HitQueue` | Intersection results from OptiX CH programs |
| `MissQueue` | Rays that hit nothing (background accumulation) |
| `ShadowQueue` | NEE shadow ray candidates (origin, dir, tmax, scaled contrib) |

---

## 4. BVH (`bvh/`)

Used only by the CPU renderers. The GPU uses OptiX to build and traverse the AS.

**SAH binned build** (`bvh.cpp`): 8-bin binned SAH, recursive, flattened to a compact 40-byte `BVHNode` array. Unified primitive index space: `[0, num_shapes)` → analytic shapes, `[num_shapes, …)` → triangles.

**CPU traversal:**
- `bvh_intersect()` — closest-hit, iterative stack (depth ≤ 64)
- `bvh_shadow_blocked()` — any-hit early-exit, never loads normals; used by NEE shadow queries

---

## 5. Triangle Mesh Loading (`scene/mesh_loader.cpp`)

Parses Wavefront OBJ (`v`, `v//vn`, `v/vt/vn`; quad fan-triangulation). Two-phase: parse positions + faces, then compute area-weighted smooth normals if OBJ lacks `vn` entries. Per-mesh transforms (translate, scale, rotate_y) applied at load time; BVH operates in world space.

---

## 6. Wavefront Pipeline

```
for each batch of N_BATCH=8 samples:
  GenerateRays  →  RayQueue (N_BATCH × W×H rays)
  while RayQueue not empty:
    OptiX intersect  →  HitQueue + MissQueue
    miss_kernel      →  background + image_buf
    shade_deferred   →  BSDF + NEE sampling → ShadowQueue + next RayQueue
    OptiX shadow     →  any-hit; unblocked rays add NEE contrib to image_buf
    swap queues
scale image by 1/spp
```

---

## 7. CPU Backend

### 7.1 Recursive renderer — Phase 1 (`cpu/render_cpu.cpp`)

Naive recursive `trace_path()`. `max_depth` safety cap + Russian Roulette unbiased termination. OpenMP parallelises over pixels (`schedule(dynamic, 4)`). Shadow queries use `bvh_shadow_blocked()` (any-hit early exit). NEE with `count_emission` flag prevents double-counting when NEE and BSDF sampling both see a light.

### 7.2 Wavefront renderer — Phase 2 (`cpu/wavefront_cpu.cpp`)

Queue-based iterative renderer mirroring the GPU wavefront structure. OpenMP parallelism via static-segment dispatch: thread `t` owns exclusive output segment `[t×chunk, …)` — no atomics. Sequential `compact_field()` memmoves segments after the parallel phase.

**Ray sorting (large mesh scenes):** before each intersect pass, rays are sorted by direction octant + quantised origin via a two-pass O(n) counting sort (enabled when `num_triangles ≥ 50 000`). Groups rays that traverse similar BVH subtrees, improving L3 cache locality.

---

## 8. GPU Backend

### 8.1 Device Scene (`gpu/device_scene.h/.cu`)

`DeviceScene` mirrors `SceneData` with device pointers. Key uploads: `shapes[]`, `triangles[]`, `tri_verts[]` (36-byte vertex-only layout for CH normal dispatch), `materials[]`, `lights[]`. Lights and materials are also uploaded to 64 KB constant memory (`c_lights_raw`, `c_materials_raw`) for hot-path access. Used by both GPU backends.

### 8.2 CUDA Software BVH backend (`gpu/gpu_bvh.cuh`, `gpu/kernels.cu`)

`gpu_bvh.cuh` provides two device-side traversal routines over the CPU-built SAH BVH uploaded to device memory:

- `gpu_bvh_intersect` — closest-hit; deferred full `Triangle` load (normals + material) only for the winning primitive; all intermediate tests use `TriangleV` (36 bytes, fits in L2).
- `gpu_shadow_blocked` — any-hit early exit; only reads `TriangleV`.

`intersect_kernel` calls `gpu_bvh_intersect` and uses warp-ballot to write hits and misses to their respective SoA queues without per-thread atomics. `shade_kernel` evaluates the BSDF, calls `gpu_nee()` (which calls `gpu_shadow_blocked` inline), and emits continuation rays via warp-ballot.

**Ray sorting (large mesh scenes):** `compute_ray_keys_kernel` computes a 32-bit key per ray — 3-bit direction octant (MSBs) plus a 29-bit Morton code of the quantised origin. `cub::DeviceRadixSort::SortPairs` reorders the ray queue so that rays with the same key traverse the same BVH subtrees, improving L2 node-cache reuse. Enabled when `num_triangles ≥ 50 000`.

### 8.3 OptiX RT-core backend (`gpu/optix_state.cu`, `gpu/optix_programs.cu`)

**Shape tessellation:** Before building the acceleration structure, all analytic shapes (sphere, cylinder, disk) are tessellated into triangle meshes. Smooth per-vertex normals match the exact analytic surface. Tessellation resolutions: sphere 16×8 (≈112 triangles), cylinder 16 slices (32 triangles), disk 16 slices (16 triangles). All tessellated triangles are concatenated with scene OBJ mesh triangles into **one flat triangle GAS** — no IAS, no custom-primitive GAS, no IS programs.

**OptiX programs** (compiled to PTX at build time, loaded from `OPTIX_PTX_FILE`):

| Program | Role |
|---|---|
| `__raygen__primary` | Reads ray from `RayQueue`, calls `optixTrace` (missSBTIndex=0) |
| `__miss__primary` | Writes to `MissQueue` via warp-ballot atomic |
| `__closesthit__triangle` | Gets OptiX barycentrics, dispatches to scene or tessellated `Triangle[]` for normals, writes to `HitQueue` via warp-ballot |
| `__raygen__shadow` | Reads from `ShadowQueue`, calls `optixTrace` with `TERMINATE_ON_FIRST_HIT \| DISABLE_CLOSESTHIT` (missSBTIndex=1) |
| `__miss__shadow` | Unblocked shadow ray → `atomicAdd` NEE contrib into `image_buf` |

**Warp-ballot note:** OptiX rejects `__shfl_sync(mask, ...)` when `mask` is a ballot result. Since only active threads reach the `__shfl_sync`, `__activemask() == mask` at that point — a semantically identical substitution that the OptiX PTX validator accepts.

### 8.4 CUDA kernels (`gpu/kernels.cu`)

Regular CUDA kernels for the non-intersection stages, shared by both GPU backends:

| Kernel | Threads/block | Used by |
|---|---|---|
| `generate_kernel` | 256 — Halton + stratified pixel sampling | Both |
| `miss_kernel` | 256 — background accumulation | Both |
| `shade_kernel` | 128 — BSDF + inline NEE shadow (gpu_shadow_blocked) | CUDA BVH |
| `shade_kernel_deferred` | 128 — BSDF + NEE sample, shadow deferred to OptiX | OptiX |
| `scale_kernel` | 256 — divide HDR buffer by spp | Both |

### 8.5 Render loop (`gpu/render_gpu.cu`)

`render_wavefront_gpu(scene, img, use_optix)` runs the same wavefront loop for both backends; the `use_optix` flag switches the intersect and shade calls.

**Multi-sample batching (N_BATCH=8):** processes 8 samples per wavefront iteration, giving the warp scheduler far more active blocks and dramatically improving DRAM latency hiding. `ray_queue_slice()` offsets SoA pointers so `generate_kernel` fills different slots without interface changes.

**OptiX shadow pass:** `shade_kernel_deferred` writes NEE candidates to `ShadowQueue` (origin, dir, tmax, tp×contrib, pixel_idx). A second `optixLaunch` with `__raygen__shadow` tests visibility using `TERMINATE_ON_FIRST_HIT` — if the ray reaches `__miss__shadow`, the contribution is added directly to `image_buf`. Both primary and shadow passes share one pipeline; shadow uses a second raygen SBT record (`sbt_shadow`).

---

## 9. Shading Logic (`shading/`)

### 9.1 BSDFs (`shading/bsdf.h`)

`sample_bsdf()` dispatches on `mat.type`:

| Type | Algorithm | Specular |
|---|---|---|
| Diffuse | Cosine-weighted hemisphere; weight = albedo | No |
| SpecularIdeal | Perfect mirror; weight = albedo | Yes |
| DielectricIdeal | Schlick Fresnel → stochastic reflect/refract; weight = albedo | Yes |
| GGXDielectric | VNDF sampling (Heitz 2018); full Fresnel; Smith G1 masking | Yes |
| Emissive | Degenerate — handled before `sample_bsdf` | Yes |

`is_specular = true` skips NEE (delta BSDFs gain nothing from explicit light sampling).

### 9.2 Direct lighting / NEE

`gpu_nee_sample()` samples one random light and returns a shadow ray candidate + unoccluded contribution. The actual shadow test is deferred to OptiX. The CPU path calls `bvh_shadow_blocked()` inline.

---

## 10. Scene File Format (JSON)

```json
{
  "camera":    { "position": […], "look_at": […], "vfov_deg": 45 },
  "render":    { "width": 960, "height": 540, "spp": 1024, "max_depth": 20,
                 "output": "out.png", "tonemapping": "aces",
                 "denoise": { "enabled": true, "sigma_r": 0.12, "atrous_passes": 5 } },
  "materials": [ { "type": "diffuse|specular_ideal|dielectric_ideal|rough_dielectric|emissive",
                   "albedo": […], "ior": 1.5, "roughness": 0.1, "emission": […] } ],
  "shapes":    [ { "type": "sphere|cylinder|disk|mesh", … } ],
  "lights":    [ { "type": "point|directional|area", … } ]
}
```

`"rough_dielectric"` maps to `GGXDielectric`. Mesh shapes load OBJ files and support `translate`, `scale`, `rotate_y`.

---

## 11. Output and Tone Mapping (`output/image.h`)

Internal: `float` buffer `W × H × 3` (linear RGB). Tone map modes: `"gamma"` (γ 2.2), `"reinhard"`, `"aces"` (Narkowicz 2016 filmic). Optional à-trous wavelet denoiser (Dammertz 2010) applied after tone mapping.

---

## 12. Build

```bash
# CPU only (recursive + wavefront)
cmake -B build && cmake --build build -j

# CPU + GPU CUDA software BVH (no OptiX)
cmake -B build -DENABLE_CUDA=ON && cmake --build build -j

# CPU + GPU CUDA BVH + OptiX RT-core backend
cmake -B build -DENABLE_CUDA=ON -DENABLE_OPTIX=ON && cmake --build build -j
```

**Requirements:** CUDA 12+, RTX GPU (sm_80+). OptiX: headers in `vendored/optix/` (driver ≥ 555 for OptiX 8.1.0 ABI).

**Usage:**
```bash
./path_tracer_cpu  <scene.json> [--wavefront] [--width W] [--height H] [--spp N]
./path_tracer_gpu  <scene.json> [--optix]     [--width W] [--height H] [--spp N]
python benchmark.py [--gpu-only] [--cpu-only] [--scenes a,b,c] [--gpu-res WxH] [--cpu-res WxH]
```

---

## 13. Performance

Mrays/s = W×H×spp/time — resolution- and spp-independent throughput, directly comparable across all renderers.

**CPU measurements** — Intel Xeon E5-2640 v3 (8 cores / 16 threads, 2.6 GHz), **1280×720, spp=128**.  
**GPU measurements** — RTX A5000 (sm_86, 84 SMs, 24 GB VRAM, 768 GB/s), **3840×2160, native scene spp** (1024 or 2048†).

GPU results below reflect the post-optimisation kernels: MissQueue eliminated
(background written directly from `intersect_kernel` / `__miss__primary`),
slimmer HitQueue (`t` and `shape_id` dropped), and `__launch_bounds__(128,6)`
on both intersect and shade.  Cumulative gain over the previous baseline is
~+6–14 % on CUDA BVH and ~+5–13 % on OptiX (largest on small/medium scenes
where the old miss kernel was a larger fraction of frame time).

### 13.1 Four-Renderer Comparison (Mrays/s)

| Scene | Mesh | CPU Recursive | CPU Wavefront | GPU CUDA BVH | GPU OptiX | CUDA/WF | OptiX/CUDA |
|---|---|---|---|---|---|---|---|
| spheres | 0 tris | 10.3 | 2.4 | 833 | 521 | **347×** | 0.63× |
| cornell\_box† | 0 tris | 2.8 | 1.4 | 200 | 138 | **143×** | 0.69× |
| materials | 0 tris | 14.5 | 2.6 | 966 | 670 | **371×** | 0.69× |
| mesh\_test | 12 tris | 12.3 | 2.3 | 795 | 480 | **346×** | 0.60× |
| bunny\_studio | 144K tris | 2.9 | 1.2 | 149 | 463 | **124×** | **3.1×** |
| dragon\_glass† | 871K tris | 1.4 | 0.7 | 64 | 274 | **92×** | **4.3×** |
| bunny\_diffuse ★ | 144K tris | 4.9 | 2.0 | 224 | 603 | **112×** | **2.7×** |
| dragon\_diffuse ★ | 871K tris | 3.2 | 1.4 | 141 | 479 | **101×** | **3.4×** |

† spp=2048 for GPU; CPU uses spp=128. ★ Diffuse-only variants — maximise shadow ray count.

### 13.2 GPU CUDA vs CPU Wavefront — raw hardware speedup

The **CUDA/WF** column isolates the GPU hardware benefit: both renderers run the identical wavefront algorithm (generate → BVH intersect → shade with NEE shadow), differing only in execution platform.

**Analytic-only scenes (0–12 triangles): 143–371× speedup.**
The A5000 has 84 SMs with 128 CUDA cores each vs 8 CPU cores — ~1300× more hardware threads, though SIMD and memory-bandwidth effects dominate.  The CPU wavefront also pays queue-compaction and SoA overhead that the recursive renderer avoids; on GPU, warp-ballot compaction is a handful of instructions and costs essentially nothing.

**Large mesh scenes (144K–871K triangles): 92–124× speedup.**
Both renderers traverse the same software BVH in the same code paths. The difference is memory bandwidth: GPU has 768 GB/s vs CPU's ~40 GB/s (19×). For the 871K-triangle dragon the BVH node array (~35 MB) approaches the A5000's 40 MB L2 cache but DRAM refills still dominate the CPU — whose 20 MB L3 is overrun entirely — reducing the speedup compared to analytic scenes.

### 13.3 GPU OptiX vs GPU CUDA BVH — RT-core benefit

**Analytic-only scenes: CUDA software BVH is ~1.4–1.7× faster than OptiX.**
Analytic shapes are tessellated before AS build (sphere → 112 triangles, cylinder → 32, disk → 16). The resulting tessellated GAS is a deeper hierarchy than the software BVH over a handful of raw primitives, adding traversal overhead per bounce.

**Large mesh scenes: OptiX is 2.7–4.3× faster than CUDA software BVH.**
`gpu_bvh_intersect` loads BVH nodes from VRAM every traversal step. For 871K-triangle scenes the node array thrashes DRAM even on the GPU. OptiX RT cores traverse in fixed-function hardware with dedicated on-chip storage, bypassing this bottleneck. Shadow rays at every diffuse bounce amplify the gap.

### 13.4 CPU Recursive vs CPU Wavefront

Recursive is **2–6× faster** than wavefront on CPU. The wavefront adds per-bounce queue-write, compaction (38 `memmove` calls across SoA fields per bounce), and queue-read overhead that the recursive renderer eliminates by keeping path state on the call stack. On GPU these overheads vanish: warp-ballot compaction costs a handful of instructions, and hiding global-memory latency across thousands of concurrent warps is free. The wavefront architecture is designed for GPU execution; its CPU incarnation exists for algorithmic parity and correctness validation.

---

## 14. Future Work

**OptiX native sphere primitives:** `OptixBuildInputSphereArray` was implemented but produced zero hits on driver 555.42.02 — sphere RT-core intersection likely requires driver ≥ 555.85. The API is retained in OptiX 8.1.0 headers. With a newer driver, tessellated spheres could be replaced with exact sphere intersection, eliminating polygon faceting and reducing GAS size for analytic-heavy scenes.

**Multiple importance sampling (MIS):** Balance BSDF sampling and NEE with the power heuristic for faster convergence on scenes with complex lighting.

**GGX metal conductor:** Add a `GGXMetal` material using conductor Fresnel (complex IOR) with the VNDF sampling already in place. The `metallic` field in `Material` is already parsed and reserved.

**Texture mapping:** `stb_image` for PNG textures; CUDA texture objects for GPU sampling.
