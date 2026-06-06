#pragma once
// Shared CPU-only shading helpers used by both the recursive (Phase 1) and
// wavefront (Phase 2) renderers.  These are inline because both translation
// units include this header.
#include "common/math.h"
#include "common/hit.h"
#include "common/material.h"
#include "common/light.h"
#include "common/shape.h"
#include "scene/scene.h"
#include "bvh/bvh.h"
#include "shading/sampling.h"

// ── Background radiance ───────────────────────────────────────────────────────

inline Vec3 background_radiance(const Ray& ray, const RenderConfig& cfg) {
    if (cfg.bg_mode == BackgroundMode::Sky) {
        float t = 0.5f * (normalize(ray.dir).y + 1.0f);
        return lerp(Vec3(1.f, 1.f, 1.f), Vec3(0.5f, 0.7f, 1.0f), t);
    }
    return cfg.background;
}

// ── Next Event Estimation (direct lighting) ───────────────────────────────────
// Randomly selects one scene light, samples a point on it, and returns the
// unoccluded direct contribution weighted by the Lambertian BRDF.
// Only called for non-specular (diffuse) hits; delta BSDFs gain nothing from NEE.

inline Vec3 sample_direct_lighting(const HitRecord& hit,
                                    const SceneData& scene, uint& seed)
{
    if (scene.lights.empty()) return Vec3(0.f);

    const int   n_lights = int(scene.lights.size());
    const int   li       = int(rand_float(seed) * n_lights) % n_lights;
    const Light& light   = scene.lights[li];
    const float  inv_n   = 1.f / n_lights;

    const Material& surf = scene.materials[hit.material_id];
    const Vec3 brdf = surf.albedo * PT_INV_PI;

    auto is_unoccluded = [&](Vec3 wi, float tmax, int light_sid) -> bool {
        float n_sign = dot(wi, hit.geo_normal) >= 0.f ? 1.f : -1.f;
        Vec3  origin = hit.point + (n_sign * 4e-4f) * hit.geo_normal;
        Ray   shadow = { origin, wi, PT_EPSILON, tmax };
        // Use the any-hit shadow traversal: exits at the first occluder without
        // loading normals — mirrors gpu_shadow_blocked on the GPU path.
        int blocker_sid = -1;
        bool blocked = bvh_shadow_blocked(shadow, scene.accel, scene, &blocker_sid);
        return !blocked || (light_sid >= 0 && blocker_sid == light_sid);
    };

    if (light.type == LightType::Area) {
        int sid = light.area.shape_id;
        if (sid < 0 || sid >= int(scene.shapes.size())) return Vec3(0.f);
        const Shape&    ls = scene.shapes[sid];
        const Material& lm = scene.materials[ls.material_id];
        if (lm.type != MaterialType::Emissive) return Vec3(0.f);

        if (ls.type == ShapeType::Sphere) {
            const Sphere& sph = ls.sphere;
            Vec3  sp      = uniform_sample_sphere(rand_float(seed), rand_float(seed));
            Vec3  lp      = sph.center + sph.radius * sp;
            Vec3  to      = lp - hit.point;
            float dist2   = length_sq(to), dist = sqrtf(dist2);
            Vec3  wi      = to / dist;
            float cos_s   = dot(hit.normal, wi);    if (cos_s  <= 0.f) return Vec3(0.f);
            float cos_l   = dot(-wi, sp);            if (cos_l  <= 0.f) return Vec3(0.f);
            if (!is_unoccluded(wi, dist * 1.01f, sid)) return Vec3(0.f);
            float p_omega = dist2 / (cos_l * 4.f * PT_PI * sph.radius * sph.radius);
            return lm.emission * brdf * cos_s / (p_omega * inv_n);

        } else if (ls.type == ShapeType::Disk) {
            const Disk& disk = ls.disk;
            Vec3 dn = normalize(disk.normal);
            Vec3 dt, db; make_onb(dn, dt, db);
            float r   = disk.radius * sqrtf(rand_float(seed));
            float phi = PT_TWO_PI * rand_float(seed);
            Vec3  lp  = disk.center + r * (cosf(phi)*dt + sinf(phi)*db);
            Vec3  to  = lp - hit.point;
            float dist2 = length_sq(to), dist = sqrtf(dist2);
            Vec3  wi    = to / dist;
            float cos_s = dot(hit.normal, wi);  if (cos_s <= 0.f) return Vec3(0.f);
            float cos_l = dot(-wi, dn);          if (cos_l <= 0.f) return Vec3(0.f);
            if (!is_unoccluded(wi, dist * 1.01f, sid)) return Vec3(0.f);
            float p_omega = dist2 / (cos_l * PT_PI * disk.radius * disk.radius);
            return lm.emission * brdf * cos_s / (p_omega * inv_n);
        }
        return Vec3(0.f);

    } else if (light.type == LightType::Point) {
        const PointLight& pl = light.point;
        Vec3  to    = pl.position - hit.point;
        float dist2 = length_sq(to), dist = sqrtf(dist2);
        Vec3  wi    = to / dist;
        float cos_s = dot(hit.normal, wi);
        if (cos_s <= 0.f) return Vec3(0.f);
        if (!is_unoccluded(wi, dist - 1e-3f, -1)) return Vec3(0.f);
        return pl.color * pl.intensity * brdf * cos_s / (dist2 * inv_n);

    } else if (light.type == LightType::Directional) {
        const DirectionalLight& dl = light.directional;
        Vec3  wi    = normalize(-dl.direction);
        float cos_s = dot(hit.normal, wi);
        if (cos_s <= 0.f) return Vec3(0.f);
        if (!is_unoccluded(wi, PT_INFINITY, -1)) return Vec3(0.f);
        return dl.color * dl.intensity * brdf * cos_s / inv_n;
    }
    return Vec3(0.f);
}
