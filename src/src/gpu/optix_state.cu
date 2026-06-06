// Host-side OptiX setup: tessellate analytic shapes, build a single triangle
// GAS over all geometry, create pipeline + SBT, and launch intersection.
//
// Key design: every analytic shape (sphere/cyl/disk) is tessellated into
// triangles before the AS is built. The result is ONE flat triangle GAS —
// no IAS, no custom-primitive GAS, no IS programs. This eliminates the
// two-level-AS overhead that hurt small scenes and gives RT cores full
// utilisation for all geometry types.
//
// Shadow rays are handled by a second optixLaunch (any-hit) whose miss program
// accumulates unblocked NEE contributions directly into the image buffer.

#include "optix_state.h"
#include "optix_params.h"
#include "common/math.h"
#include "common/shape.h"
#include "common/triangle.h"

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ── Error macros ──────────────────────────────────────────────────────────────
#define CUDA_CHECK(x) do {                                                \
    cudaError_t _e = (x);                                                 \
    if (_e != cudaSuccess) {                                              \
        char buf[256];                                                    \
        snprintf(buf, sizeof(buf), "CUDA error %s:%d: %s",               \
                 __FILE__, __LINE__, cudaGetErrorString(_e));             \
        throw std::runtime_error(buf);                                    \
    }                                                                     \
} while(0)

#define OPTIX_CHECK(x) do {                                               \
    OptixResult _r = (x);                                                 \
    if (_r != OPTIX_SUCCESS) {                                            \
        char buf[256];                                                    \
        snprintf(buf, sizeof(buf), "OptiX error %d at %s:%d",            \
                 (int)_r, __FILE__, __LINE__);                            \
        throw std::runtime_error(buf);                                    \
    }                                                                     \
} while(0)

// ── SBT record ────────────────────────────────────────────────────────────────
template<typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
struct EmptySbtData {};
using RaygenRecord   = SbtRecord<EmptySbtData>;
using MissRecord     = SbtRecord<EmptySbtData>;
using HitGroupRecord = SbtRecord<EmptySbtData>;

// ── Upload helper ─────────────────────────────────────────────────────────────
template<typename T>
static CUdeviceptr upload_to_device(const T* host, size_t count) {
    CUdeviceptr d;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d), count * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d), host,
                          count * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// ── OptiX log callback ────────────────────────────────────────────────────────
static void optix_log_cb(unsigned int level, const char* tag,
                         const char* msg, void*) {
    if (level <= 2)
        fprintf(stderr, "[OptiX %u][%s] %s\n", level, tag, msg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape tessellation
// ─────────────────────────────────────────────────────────────────────────────
// OptiX 8.1.0 sphere primitives (OptixBuildInputSphereArray) produce zero hits
// on driver 555.42 — the RT core sphere intersection feature is not supported
// by this driver version. All analytic shapes (sphere, cylinder, disk) are
// therefore tessellated into triangle meshes with smooth vertex normals.

// Smooth-shaded triangle (used by sphere, cylinder).
static void add_tri(Vec3 p0, Vec3 p1, Vec3 p2,
                    Vec3 n0, Vec3 n1, Vec3 n2, int mat,
                    std::vector<TriangleV>& tv, std::vector<Triangle>& tf) {
    TriangleV v;
    v.v[0] = p0; v.v[1] = p1; v.v[2] = p2;
    tv.push_back(v);
    Triangle t;
    t.v[0] = p0; t.v[1] = p1; t.v[2] = p2;
    t.n[0] = n0; t.n[1] = n1; t.n[2] = n2;
    t.smooth_normals = true;
    t.material_id    = mat;
    tf.push_back(t);
}

// Flat-shaded triangle (used by cube, disk, plane).
static void add_tri_flat(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 n, int mat,
                         std::vector<TriangleV>& tv, std::vector<Triangle>& tf) {
    TriangleV v;
    v.v[0] = p0; v.v[1] = p1; v.v[2] = p2;
    tv.push_back(v);
    Triangle t;
    t.v[0] = p0; t.v[1] = p1; t.v[2] = p2;
    t.n[0] = t.n[1] = t.n[2] = n;
    t.smooth_normals = false;
    t.material_id    = mat;
    tf.push_back(t);
}

// UV sphere: 16 sectors × 8 stacks ≈ 112 triangles.
// Smooth per-vertex normals match the exact analytic sphere normal at each
// vertex (outward unit vector), so shading is smooth despite coarse geometry.
// Kept deliberately coarse: the sphere only needs to be accurate enough for
// BVH intersection; smooth shading is handled by vertex normal interpolation.
static void tess_sphere(const Shape& s,
                        std::vector<TriangleV>& tv, std::vector<Triangle>& tf) {
    const Sphere& sph = s.sphere;
    const int NS = 16, NT = 8;  // sectors, stacks

    // Pre-build vertex grid
    struct V { Vec3 pos, norm; };
    std::vector<V> verts((NT + 1) * (NS + 1));
    for (int t = 0; t <= NT; ++t) {
        float phi = float(M_PI) * t / NT;          // 0 (north) → π (south)
        float sp = sinf(phi), cp = cosf(phi);
        for (int sc = 0; sc <= NS; ++sc) {
            float theta = 2.f * float(M_PI) * sc / NS;
            Vec3 n = { sp * cosf(theta), cp, sp * sinf(theta) };
            verts[t * (NS + 1) + sc] = { sph.center + n * sph.radius, n };
        }
    }

    for (int t = 0; t < NT; ++t) {
        for (int sc = 0; sc < NS; ++sc) {
            const V& v00 = verts[ t      * (NS+1) + sc    ];
            const V& v10 = verts[(t + 1) * (NS+1) + sc    ];
            const V& v01 = verts[ t      * (NS+1) + sc + 1];
            const V& v11 = verts[(t + 1) * (NS+1) + sc + 1];
            // Upper triangle (skip degenerate at north pole)
            if (t > 0)
                add_tri(v00.pos, v01.pos, v10.pos,
                        v00.norm, v01.norm, v10.norm, s.material_id, tv, tf);
            // Lower triangle (skip degenerate at south pole)
            if (t < NT - 1)
                add_tri(v01.pos, v11.pos, v10.pos,
                        v01.norm, v11.norm, v10.norm, s.material_id, tv, tf);
        }
    }
}

// Open cylinder (no caps) tessellated with 16 slices = 32 triangles.
// Normals point radially outward from the cylinder axis.
static void tess_cylinder(const Shape& s,
                          std::vector<TriangleV>& tv, std::vector<Triangle>& tf) {
    const Cylinder& cyl = s.cylinder;
    const int NS = 16;
    Vec3 ax = normalize(cyl.axis);

    // Build a local frame perpendicular to the axis
    Vec3 up = (fabsf(ax.y) < 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 tang = normalize(cross(up, ax));
    Vec3 bitan = cross(ax, tang);

    // Top and bottom ring vertices
    std::vector<Vec3> bot(NS), top(NS), nrm(NS);
    for (int sc = 0; sc < NS; ++sc) {
        float theta = 2.f * float(M_PI) * sc / NS;
        Vec3 radial = cosf(theta) * tang + sinf(theta) * bitan;
        nrm[sc] = radial;
        bot[sc]  = cyl.center - ax * cyl.half_height + radial * cyl.radius;
        top[sc]  = cyl.center + ax * cyl.half_height + radial * cyl.radius;
    }

    for (int sc = 0; sc < NS; ++sc) {
        int sc1 = (sc + 1) % NS;
        add_tri(bot[sc], bot[sc1], top[sc],
                nrm[sc], nrm[sc1], nrm[sc], s.material_id, tv, tf);
        add_tri(bot[sc1], top[sc1], top[sc],
                nrm[sc1], nrm[sc1], nrm[sc], s.material_id, tv, tf);
    }
}

// Disk tessellated as a fan of 16 triangles from the centre.
// All normals equal the disk's outward-facing analytic normal (flat surface).
static void tess_disk(const Shape& s,
                      std::vector<TriangleV>& tv, std::vector<Triangle>& tf) {
    const Disk& disk = s.disk;
    const int NS = 16;
    Vec3 n = normalize(disk.normal);
    Vec3 up = (fabsf(n.y) < 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 tang = normalize(cross(up, n));
    Vec3 bitan = cross(n, tang);

    std::vector<Vec3> rim(NS);
    for (int sc = 0; sc < NS; ++sc) {
        float theta = 2.f * float(M_PI) * sc / NS;
        rim[sc] = disk.center + (cosf(theta) * tang + sinf(theta) * bitan) * disk.radius;
    }

    for (int sc = 0; sc < NS; ++sc) {
        int sc1 = (sc + 1) % NS;
        add_tri(disk.center, rim[sc], rim[sc1], n, n, n, s.material_id, tv, tf);
    }
}

// Axis-aligned cube tessellated as 12 flat-shaded triangles (6 faces × 2).
// Winding order chosen so cross(e1,e2) points outward for each face.
static void tess_cube(const Shape& s,
                      std::vector<TriangleV>& tv, std::vector<Triangle>& tf) {
    const Cube& box = s.cube;
    Vec3 c = box.center;
    float hx = box.half_extents.x, hy = box.half_extents.y, hz = box.half_extents.z;

    Vec3 v[8] = {
        {c.x-hx, c.y-hy, c.z-hz},  // 0 (---)
        {c.x+hx, c.y-hy, c.z-hz},  // 1 (+--)
        {c.x+hx, c.y+hy, c.z-hz},  // 2 (++-)
        {c.x-hx, c.y+hy, c.z-hz},  // 3 (-+-)
        {c.x-hx, c.y-hy, c.z+hz},  // 4 (--+)
        {c.x+hx, c.y-hy, c.z+hz},  // 5 (+-+)
        {c.x+hx, c.y+hy, c.z+hz},  // 6 (+++)
        {c.x-hx, c.y+hy, c.z+hz},  // 7 (-++)
    };
    int m = s.material_id;
    add_tri_flat(v[0],v[3],v[2], {0,0,-1}, m, tv, tf);  // -Z
    add_tri_flat(v[0],v[2],v[1], {0,0,-1}, m, tv, tf);
    add_tri_flat(v[4],v[5],v[7], {0,0,+1}, m, tv, tf);  // +Z
    add_tri_flat(v[5],v[6],v[7], {0,0,+1}, m, tv, tf);
    add_tri_flat(v[0],v[4],v[3], {-1,0,0}, m, tv, tf);  // -X
    add_tri_flat(v[4],v[7],v[3], {-1,0,0}, m, tv, tf);
    add_tri_flat(v[1],v[2],v[5], {+1,0,0}, m, tv, tf);  // +X
    add_tri_flat(v[2],v[6],v[5], {+1,0,0}, m, tv, tf);
    add_tri_flat(v[0],v[1],v[4], {0,-1,0}, m, tv, tf);  // -Y
    add_tri_flat(v[1],v[5],v[4], {0,-1,0}, m, tv, tf);
    add_tri_flat(v[2],v[3],v[6], {0,+1,0}, m, tv, tf);  // +Y
    add_tri_flat(v[3],v[7],v[6], {0,+1,0}, m, tv, tf);
}

// Infinite plane tessellated as 2 large triangles (half-size 10000 in each
// tangent direction). The plane is only approximately infinite but suffices
// for ground planes and walls in typical scenes.
static void tess_plane(const Shape& s,
                       std::vector<TriangleV>& tv, std::vector<Triangle>& tf) {
    const Plane& pl = s.plane;
    Vec3 n = normalize(pl.normal);
    Vec3 up    = (fabsf(n.y) < 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 tang  = normalize(cross(up, n));
    Vec3 bitan = cross(n, tang);

    const float hs = 10000.f;
    Vec3 p0 = pl.center - tang * hs - bitan * hs;
    Vec3 p1 = pl.center + tang * hs - bitan * hs;
    Vec3 p2 = pl.center + tang * hs + bitan * hs;
    Vec3 p3 = pl.center - tang * hs + bitan * hs;
    // cross(tang, bitan) = n, so winding (p0,p1,p3) and (p1,p2,p3) gives +n normal.
    add_tri_flat(p0, p1, p3, n, s.material_id, tv, tf);
    add_tri_flat(p1, p2, p3, n, s.material_id, tv, tf);
}

// Tessellate all analytic shapes into triangle lists.
static void tessellate_shapes(const std::vector<char>& shape_buf, int num_shapes,
                               std::vector<TriangleV>& out_verts,
                               std::vector<Triangle>&  out_tris) {
    const Shape* shapes = reinterpret_cast<const Shape*>(shape_buf.data());
    for (int i = 0; i < num_shapes; ++i) {
        switch (shapes[i].type) {
            case ShapeType::Sphere:   tess_sphere  (shapes[i], out_verts, out_tris); break;
            case ShapeType::Cylinder: tess_cylinder(shapes[i], out_verts, out_tris); break;
            case ShapeType::Disk:     tess_disk    (shapes[i], out_verts, out_tris); break;
            case ShapeType::Cube:     tess_cube    (shapes[i], out_verts, out_tris); break;
            case ShapeType::Plane:    tess_plane   (shapes[i], out_verts, out_tris); break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AS + pipeline
// ─────────────────────────────────────────────────────────────────────────────

// Build a single triangle GAS from a combined vertex buffer.
static OptixTraversableHandle build_triangle_gas(
    OptixDeviceContext ctx,
    CUdeviceptr        d_verts,      // device ptr to float3 array
    unsigned int       num_verts,    // total vertex count (= num_triangles * 3)
    CUdeviceptr&       d_gas_buf)
{
    assert(num_verts > 0);

    OptixBuildInputTriangleArray tri_in{};
    tri_in.vertexBuffers            = &d_verts;
    tri_in.numVertices              = num_verts;
    tri_in.vertexFormat             = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri_in.vertexStrideInBytes      = sizeof(float) * 3;
    tri_in.numSbtRecords            = 1;
    unsigned int tri_flags          = OPTIX_GEOMETRY_FLAG_NONE;
    tri_in.flags                    = &tri_flags;
    tri_in.sbtIndexOffsetBuffer     = 0;
    tri_in.sbtIndexOffsetSizeInBytes= 0;

    OptixBuildInput bi{};
    bi.type          = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray = tri_in;

    OptixAccelBuildOptions opts{};
    opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                      OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &opts, &bi, 1, &sizes));

    CUdeviceptr d_tmp;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp), sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_buf), sizes.outputSizeInBytes));

    OptixTraversableHandle handle;
    OPTIX_CHECK(optixAccelBuild(ctx, 0, &opts, &bi, 1,
                                d_tmp, sizes.tempSizeInBytes,
                                d_gas_buf, sizes.outputSizeInBytes,
                                &handle, nullptr, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp)));
    return handle;
}



// ── Public: build_optix_state ─────────────────────────────────────────────────
OptixState build_optix_state(const DeviceScene& ds, const char* ptx_source) {
    OptixState state{};

    // ── Context ───────────────────────────────────────────────────────────────
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions ctx_opts{};
    ctx_opts.logCallbackFunction = optix_log_cb;
    ctx_opts.logCallbackLevel    = 3;
    OPTIX_CHECK(optixDeviceContextCreate(0, &ctx_opts, &state.context));

    // ── Tessellate all analytic shapes ────────────────────────────────────────
    std::vector<TriangleV> tess_tv;
    std::vector<Triangle>  tess_tf;
    if (ds.num_shapes > 0) {
        std::vector<char> shape_buf(ds.num_shapes * sizeof(Shape));
        CUDA_CHECK(cudaMemcpy(shape_buf.data(), ds.shapes,
                              ds.num_shapes * sizeof(Shape), cudaMemcpyDeviceToHost));
        tessellate_shapes(shape_buf, ds.num_shapes, tess_tv, tess_tf);
        printf("  Tessellated %d shapes → %zu triangles\n", ds.num_shapes, tess_tf.size());
    }

    // ── Build combined vertex buffer: [scene tris | tessellated shape tris] ──
    std::vector<TriangleV> scene_tv(ds.num_triangles);
    if (ds.num_triangles > 0)
        CUDA_CHECK(cudaMemcpy(scene_tv.data(), ds.tri_verts,
                              ds.num_triangles * sizeof(TriangleV), cudaMemcpyDeviceToHost));
    std::vector<TriangleV> all_tv;
    all_tv.reserve(scene_tv.size() + tess_tv.size());
    all_tv.insert(all_tv.end(), scene_tv.begin(), scene_tv.end());
    all_tv.insert(all_tv.end(), tess_tv.begin(),  tess_tv.end());

    state.d_tess_triangles    = tess_tf.empty() ? 0 : upload_to_device(tess_tf.data(), tess_tf.size());
    state.num_tess_triangles  = (int)tess_tf.size();
    state.num_scene_triangles = ds.num_triangles;

    // ── Module ────────────────────────────────────────────────────────────────
    OptixModuleCompileOptions mod_opts{};
    mod_opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mod_opts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mod_opts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipe_opts{};
    pipe_opts.usesMotionBlur                   = 0;
    pipe_opts.numPayloadValues                 = 1;
    pipe_opts.numAttributeValues               = 2;
    pipe_opts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipe_opts.pipelineLaunchParamsVariableName = "params";
    pipe_opts.usesPrimitiveTypeFlags           = (unsigned)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipe_opts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

    char log[8192]; size_t log_size = sizeof(log);
    OptixModule mod;
    {
        OptixResult r = optixModuleCreate(
            state.context, &mod_opts, &pipe_opts,
            ptx_source, strlen(ptx_source),
            log, &log_size, &mod);
        if (r != OPTIX_SUCCESS) {
            fprintf(stderr, "optixModuleCreate failed (%d):\n%s\n", (int)r, log);
            throw std::runtime_error("OptiX module creation failed");
        }
    }

    // ── Program groups ────────────────────────────────────────────────────────
    OptixProgramGroupOptions pg_opts{};
    OptixProgramGroup pg_raygen, pg_raygen_shadow, pg_miss, pg_miss_shadow, pg_hit_tri;

    {
        OptixProgramGroupDesc d{};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module            = mod;
        d.raygen.entryFunctionName = "__raygen__primary";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context, &d, 1, &pg_opts, log, &log_size, &pg_raygen));
    }
    {
        OptixProgramGroupDesc d{};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module            = mod;
        d.raygen.entryFunctionName = "__raygen__shadow";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context, &d, 1, &pg_opts, log, &log_size, &pg_raygen_shadow));
    }
    {
        OptixProgramGroupDesc d{};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        d.miss.module            = mod;
        d.miss.entryFunctionName = "__miss__primary";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context, &d, 1, &pg_opts, log, &log_size, &pg_miss));
    }
    {
        OptixProgramGroupDesc d{};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        d.miss.module            = mod;
        d.miss.entryFunctionName = "__miss__shadow";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context, &d, 1, &pg_opts, log, &log_size, &pg_miss_shadow));
    }
    {
        OptixProgramGroupDesc d{};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        d.hitgroup.moduleCH            = mod;
        d.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.context, &d, 1, &pg_opts, log, &log_size, &pg_hit_tri));
    }

    // ── Pipeline ──────────────────────────────────────────────────────────────
    // All 5 program groups in one pipeline. Shadow rays use DISABLE_CLOSESTHIT
    // so the CH program is never called for shadow — same hit group works fine.
    OptixProgramGroup pgs[] = { pg_raygen, pg_raygen_shadow, pg_miss, pg_miss_shadow, pg_hit_tri };
    OptixPipelineLinkOptions link_opts{};
    link_opts.maxTraceDepth = 1;
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        state.context, &pipe_opts, &link_opts,
        pgs, 5, log, &log_size, &state.pipeline));

    HitGroupRecord hg_rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_hit_tri, &hg_rec));
    state.d_sbt_hitgrp = upload_to_device(&hg_rec, 1);

    OptixStackSizes ss{};
    for (auto pg : pgs) OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &ss, state.pipeline));
    uint32_t dc, cc, css;
    OPTIX_CHECK(optixUtilComputeStackSizes(&ss, 1, 0, 0, &dc, &cc, &css));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, dc, cc, css, 1));

    // ── SBT ───────────────────────────────────────────────────────────────────
    // Primary raygen SBT record
    RaygenRecord rg_rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_raygen, &rg_rec));
    state.d_sbt_raygen = upload_to_device(&rg_rec, 1);

    // Shadow raygen SBT record
    RaygenRecord rg_shadow_rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_raygen_shadow, &rg_shadow_rec));
    state.d_sbt_raygen_shadow = upload_to_device(&rg_shadow_rec, 1);

    // Miss SBT: two records — [0] primary miss, [1] shadow miss
    MissRecord ms_recs[2];
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_miss,        &ms_recs[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_miss_shadow, &ms_recs[1]));
    state.d_sbt_miss = upload_to_device(ms_recs, 2);

    // Primary SBT
    state.sbt.raygenRecord                = state.d_sbt_raygen;
    state.sbt.missRecordBase              = state.d_sbt_miss;
    state.sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    state.sbt.missRecordCount             = 2;
    state.sbt.hitgroupRecordBase          = state.d_sbt_hitgrp;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    state.sbt.hitgroupRecordCount         = 1;

    // Shadow SBT: identical to primary except uses shadow raygen record
    state.sbt_shadow                  = state.sbt;
    state.sbt_shadow.raygenRecord     = state.d_sbt_raygen_shadow;

    // ── Build single triangle GAS ─────────────────────────────────────────────
    assert(!all_tv.empty());
    CUdeviceptr d_all_verts = upload_to_device(all_tv.data(), all_tv.size());
    state.traversable = build_triangle_gas(
        state.context, d_all_verts, (unsigned)all_tv.size() * 3, state.d_gas_tri_buf);
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_all_verts)));

    // ── Per-launch params buffer ──────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&state.d_params, sizeof(LaunchParams)));

    optixProgramGroupDestroy(pg_raygen);
    optixProgramGroupDestroy(pg_raygen_shadow);
    optixProgramGroupDestroy(pg_miss);
    optixProgramGroupDestroy(pg_miss_shadow);
    optixProgramGroupDestroy(pg_hit_tri);
    optixModuleDestroy(mod);

    return state;
}

// ── Public: destroy_optix_state ───────────────────────────────────────────────
void destroy_optix_state(OptixState& state) {
    auto cf = [](CUdeviceptr p) { if (p) cudaFree(reinterpret_cast<void*>(p)); };
    if (state.d_params) cudaFree(state.d_params);
    cf(state.d_sbt_raygen); cf(state.d_sbt_raygen_shadow);
    cf(state.d_sbt_miss);   cf(state.d_sbt_hitgrp);
    cf(state.d_gas_tri_buf);
    cf(state.d_tess_triangles);
    if (state.pipeline) optixPipelineDestroy(state.pipeline);
    if (state.context)  optixDeviceContextDestroy(state.context);
    state = {};
}

// ── Public: optix_intersect ───────────────────────────────────────────────────
void optix_intersect(
    OptixState& state, const DeviceScene& ds,
    RayQueue& ray, HitQueue& hit, int* d_hc,
    float* image_buf)
{
    if (ray.count == 0) return;

    LaunchParams lp{};
    lp.ray_ox = ray.origin_x; lp.ray_oy = ray.origin_y; lp.ray_oz = ray.origin_z;
    lp.ray_dx = ray.dir_x;    lp.ray_dy = ray.dir_y;    lp.ray_dz = ray.dir_z;
    lp.ray_tr = ray.throughput_r; lp.ray_tg = ray.throughput_g; lp.ray_tb = ray.throughput_b;
    lp.ray_rr = ray.radiance_r;   lp.ray_rg = ray.radiance_g;   lp.ray_rb = ray.radiance_b;
    lp.ray_pidx  = ray.pixel_idx;  lp.ray_depth = ray.depth;
    lp.ray_seed  = ray.seed;       lp.ray_cemit = ray.count_emission;
    lp.ray_count = ray.count;

    lp.hit_px = hit.point_x;  lp.hit_py = hit.point_y;  lp.hit_pz = hit.point_z;
    lp.hit_nx = hit.normal_x; lp.hit_ny = hit.normal_y; lp.hit_nz = hit.normal_z;
    lp.hit_gnx= hit.geo_nx;   lp.hit_gny= hit.geo_ny;   lp.hit_gnz= hit.geo_nz;
    lp.hit_wox= hit.wo_x;     lp.hit_woy= hit.wo_y;     lp.hit_woz= hit.wo_z;
    lp.hit_tr = hit.throughput_r; lp.hit_tg = hit.throughput_g; lp.hit_tb = hit.throughput_b;
    lp.hit_rr = hit.radiance_r;   lp.hit_rg = hit.radiance_g;   lp.hit_rb = hit.radiance_b;
    lp.hit_mid = hit.material_id; lp.hit_ff  = hit.front_face;
    lp.hit_pidx = hit.pixel_idx;  lp.hit_depth= hit.depth;
    lp.hit_seed = hit.seed;       lp.hit_cemit= hit.count_emission;
    lp.d_hit_count = d_hc;

    lp.image_buf = image_buf;
    lp.bg_r = ds.config.bg_r; lp.bg_g = ds.config.bg_g; lp.bg_b = ds.config.bg_b;
    lp.bg_mode = ds.config.bg_mode;
    lp.firefly_clamp = ds.config.firefly_clamp;

    lp.traversable           = state.traversable;
    lp.triangles             = ds.triangles;
    lp.tess_triangles        = reinterpret_cast<const Triangle*>(state.d_tess_triangles);
    lp.num_scene_triangles   = state.num_scene_triangles;

    CUDA_CHECK(cudaMemcpy(state.d_params, &lp, sizeof(LaunchParams),
                          cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(
        state.pipeline, 0,
        reinterpret_cast<CUdeviceptr>(state.d_params), sizeof(LaunchParams),
        &state.sbt,
        (unsigned)ray.count, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── Public: optix_shadow ──────────────────────────────────────────────────────
void optix_shadow(
    OptixState&  state,
    float* sh_ox, float* sh_oy, float* sh_oz,
    float* sh_dx, float* sh_dy, float* sh_dz,
    float* sh_tmax,
    float* sh_cr, float* sh_cg, float* sh_cb,
    int*   sh_pidx,
    int    sh_count,
    float* image_buf)
{
    if (sh_count == 0) return;

    // Re-use the same d_params buffer (already allocated)
    LaunchParams lp{};
    lp.sh_ox   = sh_ox;  lp.sh_oy   = sh_oy;  lp.sh_oz   = sh_oz;
    lp.sh_dx   = sh_dx;  lp.sh_dy   = sh_dy;  lp.sh_dz   = sh_dz;
    lp.sh_tmax = sh_tmax;
    lp.sh_cr   = sh_cr;  lp.sh_cg   = sh_cg;  lp.sh_cb   = sh_cb;
    lp.sh_pidx = sh_pidx;
    lp.sh_count= sh_count;
    lp.image_buf   = image_buf;
    lp.traversable = state.traversable;
    // CH programs are disabled for shadow rays; no geometry data needed.

    CUDA_CHECK(cudaMemcpy(state.d_params, &lp, sizeof(LaunchParams),
                          cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(
        state.pipeline, 0,
        reinterpret_cast<CUdeviceptr>(state.d_params), sizeof(LaunchParams),
        &state.sbt_shadow,   // uses shadow raygen, same miss + hitgroup SBT
        (unsigned)sh_count, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}
