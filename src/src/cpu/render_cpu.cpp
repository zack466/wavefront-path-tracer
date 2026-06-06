#include "render_cpu.h"
#include "cpu_shading.h"
#include "shading/bsdf.h"

#include <cstdio>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ── Recursive path tracer ─────────────────────────────────────────────────────
// `count_emission`: whether this ray should count emissive surface contributions.
//
//  - Camera rays and rays after specular bounces:  count_emission = true
//    (mirrors and glass must show the lights they reflect/refract)
//  - Rays after diffuse bounces:                   count_emission = false
//    (direct lighting was already provided by sample_direct_lighting above;
//     counting emission here too would double-count it)
//
// This is the same logic the wavefront shade kernel will use, just expressed
// recursively instead of via queues.

static Vec3 trace_path(const Ray& ray, const SceneData& scene,
                       int bounce, uint& seed, bool count_emission)
{
    const RenderConfig& cfg = scene.config;

    if (bounce >= cfg.max_depth) return Vec3(0.f);

    HitRecord hit;
    if (!bvh_intersect(ray, scene.accel, scene, hit))
        return background_radiance(ray, cfg);

    const Material& mat = scene.materials[hit.material_id];

    if (mat.type == MaterialType::Emissive)
        return count_emission ? mat.emission : Vec3(0.f);

    Vec3 wo = -ray.dir;
    BSDFSample s = sample_bsdf(mat, hit, wo, seed);
    Vec3 throughput = s.weight;

    // Direct lighting via NEE for non-specular (diffuse) hits.
    // Specular/glass BSDFs are delta distributions — NEE contributes nothing.
    Vec3 direct(0.f);
    if (!s.is_specular)
        direct = sample_direct_lighting(hit, scene, seed);

    // Russian Roulette path termination.
    if (bounce >= cfg.rr_min_depth) {
        float q = fmaxf(0.05f, 1.0f - max_component(throughput));
        if (rand_float(seed) < q) return direct;
        throughput = throughput / (1.0f - q);
    }

    // Offset along geometric (face) normal — prevents self-intersection on
    // smooth meshes where the shading normal diverges from the triangle plane.
    float n_sign  = (dot(s.direction, hit.geo_normal) >= 0.f) ? 1.f : -1.f;
    Vec3  origin  = hit.point + (n_sign * 4e-4f) * hit.geo_normal;
    Ray   scattered = make_ray(origin, s.direction);

    // Continuation: specular bounces see emission; diffuse bounces do not (NEE covers it).
    Vec3 indirect = throughput * trace_path(scattered, scene, bounce + 1, seed, s.is_specular);
    return direct + indirect;
}

// ── Render loop ───────────────────────────────────────────────────────────────

void render_cpu(const SceneData& scene, ImageBuffer& img) {
    const RenderConfig& cfg = scene.config;
    CameraFrame cam = make_camera_frame(scene.camera, cfg.width, cfg.height);

    int  total_pixels = cfg.width * cfg.height;
    int  sqrt_spp     = int(sqrtf(float(cfg.spp)));
    bool stratified   = (sqrt_spp * sqrt_spp == cfg.spp);

#ifndef _OPENMP
    int reported = -1;
#endif

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4) collapse(2)
    #endif
    for (int py = 0; py < cfg.height; ++py) {
        for (int px = 0; px < cfg.width; ++px) {
            Vec3 color(0.f);

            for (int s = 0; s < cfg.spp; ++s) {
                uint seed = wang_hash(
                    uint(py) * uint(cfg.width) + uint(px) +
                    uint(s) * uint(total_pixels)
                );

                float ou, ov;
                if (stratified) {
                    int sx = s % sqrt_spp, sy = s / sqrt_spp;
                    ou = (float(sx) + rand_float(seed)) / float(sqrt_spp);
                    ov = (float(sy) + rand_float(seed)) / float(sqrt_spp);
                } else {
                    ou = rand_float(seed);
                    ov = rand_float(seed);
                }

                // Thin-lens DoF: sample a point on the circular lens aperture.
                float lu = 0.f, lv = 0.f;
                if (scene.camera.aperture > 0.f) {
                    float r   = sqrtf(rand_float(seed));
                    float phi = PT_TWO_PI * rand_float(seed);
                    lu = r * cosf(phi);
                    lv = r * sinf(phi);
                }

                Ray ray = camera_ray(cam, px, py, ou, ov, cfg.width, cfg.height,
                                     lu, lv, scene.camera.aperture, scene.camera.focus_dist);
                // Camera rays always see emissive surfaces (area lights are visible).
                Vec3 c = trace_path(ray, scene, 0, seed, /*count_emission=*/true);

                if (cfg.firefly_clamp > 0.f) {
                    float l = luma(c);
                    if (l > cfg.firefly_clamp) c = c * (cfg.firefly_clamp / l);
                }
                color += c;
            }

            color = color / float(cfg.spp);
            float* p = img.pixel(px, py);
            p[0] = color.x; p[1] = color.y; p[2] = color.z;
        }

        #ifndef _OPENMP
        {
            int pct = (py * 100) / (cfg.height - 1);
            if (pct != reported) {
                reported = pct;
                std::printf("\r  %3d%%", pct);
                std::fflush(stdout);
            }
        }
        #endif
    }

    #ifndef _OPENMP
    std::printf("\n");
    #endif
}
