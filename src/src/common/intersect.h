#pragma once
#include "math.h"
#include "hit.h"
#include "shape.h"
#include "triangle.h"

// ── Per-primitive analytic intersection functions ─────────────────────────────
//
// All functions are HD so they can be called from:
//   • CPU BVH traversal (Phase 1/2)
//   • OptiX intersection programs (Phase 3)
//
// They fill the GEOMETRIC fields of rec (point, normal, t, front_face) but
// leave material_id and shape_id untouched — set those in the caller.
//
// Convention: rec.normal is the shading normal that OPPOSES the ray direction
// (always in the same hemisphere as -ray.dir).

// ── Sphere ────────────────────────────────────────────────────────────────────

HD bool intersect_sphere(const Ray& ray, const Sphere& sph,
                         float t_max, HitRecord& rec)
{
    Vec3  oc     = ray.origin - sph.center;
    float a      = length_sq(ray.dir);
    float half_b = dot(oc, ray.dir);
    float c      = length_sq(oc) - sph.radius * sph.radius;
    float disc   = half_b * half_b - a * c;
    if (disc < 0.f) return false;

    float sqrtd = sqrtf(disc);
    float t     = (-half_b - sqrtd) / a;
    if (t < ray.tmin || t > t_max) {
        t = (-half_b + sqrtd) / a;
        if (t < ray.tmin || t > t_max) return false;
    }

    Vec3 hit_pt        = ray_at(ray, t);
    Vec3 outward_n     = (hit_pt - sph.center) / sph.radius;
    bool front         = dot(ray.dir, outward_n) < 0.f;

    rec.t = t; rec.point = hit_pt; rec.front_face = front;
    rec.normal = rec.geo_normal = front ? outward_n : -outward_n;
    return true;
}

// ── Cylinder ──────────────────────────────────────────────────────────────────
// Finite open cylinder (no caps). Axis is `cyl.axis` (unit), extending
// ±half_height from center. Caps are not tested here — add Disk shapes at
// each end if you want closed caps.

HD bool intersect_cylinder(const Ray& ray, const Cylinder& cyl,
                           float t_max, HitRecord& rec)
{
    Vec3 axis   = normalize(cyl.axis);
    Vec3 oc     = ray.origin - cyl.center;

    // Project ray components perpendicular to cylinder axis
    Vec3 d_perp  = ray.dir - dot(ray.dir, axis) * axis;
    Vec3 oc_perp = oc      - dot(oc,      axis) * axis;

    float a = length_sq(d_perp);
    if (a < 1e-8f) return false;   // ray parallel to axis

    float half_b = dot(d_perp, oc_perp);
    float c      = length_sq(oc_perp) - cyl.radius * cyl.radius;
    float disc   = half_b * half_b - a * c;
    if (disc < 0.f) return false;

    float sqrtd = sqrtf(disc);
    float t     = (-half_b - sqrtd) / a;
    if (t < ray.tmin || t > t_max) {
        t = (-half_b + sqrtd) / a;
        if (t < ray.tmin || t > t_max) return false;
    }

    Vec3  hit_pt    = ray_at(ray, t);
    float along     = dot(hit_pt - cyl.center, axis);
    if (fabsf(along) > cyl.half_height) return false;  // outside height bounds

    Vec3 outward_n = normalize((hit_pt - cyl.center) - along * axis);
    bool front     = dot(ray.dir, outward_n) < 0.f;

    rec.t = t; rec.point = hit_pt; rec.front_face = front;
    rec.normal = rec.geo_normal = front ? outward_n : -outward_n;
    return true;
}

// ── Disk ──────────────────────────────────────────────────────────────────────

HD bool intersect_disk(const Ray& ray, const Disk& disk,
                       float t_max, HitRecord& rec)
{
    Vec3  n    = normalize(disk.normal);
    float denom = dot(ray.dir, n);
    if (fabsf(denom) < 1e-8f) return false;   // ray parallel to disk plane

    float t = dot(disk.center - ray.origin, n) / denom;
    if (t < ray.tmin || t > t_max) return false;

    Vec3 hit_pt = ray_at(ray, t);
    if (length_sq(hit_pt - disk.center) > disk.radius * disk.radius) return false;

    bool front = denom < 0.f;
    rec.t = t; rec.point = hit_pt; rec.front_face = front;
    rec.normal = rec.geo_normal = front ? n : -n;
    return true;
}

// ── Triangle: vertex-only Möller-Trumbore ─────────────────────────────────────
// Tests intersection using only the three vertex positions (no normals loaded).
// Returns barycentric (u, v) so the caller can compute normals for the winner.
// Used during BVH traversal so the hot loop only reads TriangleV (36 bytes),
// not the full 80-byte Triangle.

HD bool intersect_triangle_v(const Ray& ray,
                              const Vec3& v0, const Vec3& v1, const Vec3& v2,
                              float t_max, float& t_out, float& u_out, float& v_out)
{
    Vec3  e1 = v1 - v0;
    Vec3  e2 = v2 - v0;
    Vec3  h  = cross(ray.dir, e2);
    float a  = dot(e1, h);
    if (fabsf(a) < 1e-8f) return false;

    float f = 1.0f / a;
    Vec3  s = ray.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    Vec3  q = cross(s, e1);
    float v = f * dot(ray.dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * dot(e2, q);
    if (t < ray.tmin || t > t_max) return false;

    t_out = t; u_out = u; v_out = v;
    return true;
}

// Fill HitRecord normals from a Triangle whose vertex intersection was already
// confirmed with intersect_triangle_v (barycentric u, v are re-used here).
HD void triangle_fill_normals(const Triangle& tri, const Ray& ray,
                               float t, float u, float v, HitRecord& rec)
{
    float w = 1.0f - u - v;
    Vec3 e1 = tri.v[1] - tri.v[0];
    Vec3 e2 = tri.v[2] - tri.v[0];
    Vec3 face_n = normalize(cross(e1, e2));
    Vec3 shading_n = tri.smooth_normals
                   ? normalize(w * tri.n[0] + u * tri.n[1] + v * tri.n[2])
                   : face_n;
    bool front     = dot(ray.dir, face_n) < 0.0f;
    rec.t          = t;
    rec.point      = ray_at(ray, t);
    rec.normal     = front ?  shading_n : -shading_n;
    rec.geo_normal = front ?  face_n    : -face_n;
    rec.front_face = front;
}

// ── Triangle (Möller-Trumbore) ────────────────────────────────────────────────
// Fills rec with geometry + interpolated shading normal. Does NOT set
// material_id or shape_id — caller must fill those.

HD bool intersect_triangle(const Ray& ray, const Triangle& tri,
                           float t_max, HitRecord& rec)
{
    Vec3 e1 = tri.v[1] - tri.v[0];
    Vec3 e2 = tri.v[2] - tri.v[0];
    Vec3 h  = cross(ray.dir, e2);
    float a = dot(e1, h);
    if (fabsf(a) < 1e-8f) return false;           // ray parallel to triangle

    float f = 1.0f / a;
    Vec3  s = ray.origin - tri.v[0];
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    Vec3  q = cross(s, e1);
    float v = f * dot(ray.dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * dot(e2, q);
    if (t < ray.tmin || t > t_max) return false;

    float w = 1.0f - u - v;    // barycentric weight for v[0]
    Vec3 face_n   = normalize(cross(e1, e2));
    Vec3 shading_n = tri.smooth_normals
                   ? normalize(w * tri.n[0] + u * tri.n[1] + v * tri.n[2])
                   : face_n;

    // Use geometric (face) normal to determine front/back — the smooth shading
    // normal can deviate enough that it disagrees with the actual surface side.
    bool front     = dot(ray.dir, face_n) < 0.0f;
    Vec3 geo_n     = front ?  face_n : -face_n;
    Vec3 shade_n   = front ?  shading_n : -shading_n;

    rec.t          = t;
    rec.point      = ray_at(ray, t);
    rec.normal     = shade_n;
    rec.geo_normal = geo_n;    // ← actual face normal, for spawn-offset computation
    rec.front_face = front;
    return true;
}

// ── Cube (axis-aligned box) ───────────────────────────────────────────────────
// Handles both outside-entry and inside-exit (for dielectric glass cubes).

HD bool intersect_cube(const Ray& ray, const Cube& box, float t_max, HitRecord& rec)
{
    Vec3 lo = box.center - box.half_extents;
    Vec3 hi = box.center + box.half_extents;

    float t_near = ray.tmin, t_far = t_max;
    int   near_ax = -1, far_ax = -1;
    float near_sg = 0.f, far_sg = 0.f;

    for (int i = 0; i < 3; ++i) {
        float inv = 1.f / ray.dir[i];
        float ta = (lo[i] - ray.origin[i]) * inv;
        float tb = (hi[i] - ray.origin[i]) * inv;
        float sa = -1.f, sb = +1.f;  // lo-face outward = -axis, hi-face = +axis
        if (inv < 0.f) {
            float tmp = ta; ta = tb; tb = tmp;
            float stmp = sa; sa = sb; sb = stmp;
        }
        // ta: near (entry) t for this slab; tb: far (exit) t
        if (ta > t_near) { t_near = ta; near_ax = i; near_sg = sa; }
        if (tb < t_far)  { t_far  = tb; far_ax  = i; far_sg  = sb; }
        if (t_near > t_far) return false;
    }

    float t_hit; int ax; float sg;
    if (near_ax >= 0) {
        t_hit = t_near; ax = near_ax; sg = near_sg;
    } else if (far_ax >= 0) {
        // Ray started inside the box — report exit face
        t_hit = t_far; ax = far_ax; sg = far_sg;
    } else {
        return false;
    }

    Vec3 n{0, 0, 0};
    n[ax] = sg;
    bool front = dot(ray.dir, n) < 0.f;
    rec.t = t_hit;
    rec.point = ray_at(ray, t_hit);
    rec.normal = rec.geo_normal = front ? n : -n;
    rec.front_face = front;
    return true;
}

// ── Plane (infinite) ──────────────────────────────────────────────────────────

HD bool intersect_plane(const Ray& ray, const Plane& pl, float t_max, HitRecord& rec)
{
    Vec3  n     = normalize(pl.normal);
    float denom = dot(ray.dir, n);
    if (fabsf(denom) < 1e-8f) return false;
    float t = dot(pl.center - ray.origin, n) / denom;
    if (t < ray.tmin || t > t_max) return false;
    bool front = denom < 0.f;
    rec.t = t;
    rec.point = ray_at(ray, t);
    rec.normal = rec.geo_normal = front ? n : -n;
    rec.front_face = front;
    return true;
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
// Convenience function: intersect a single Shape (dispatches by type).
// Returns true and fills rec (geometry only) on a hit.

HD bool intersect_shape(const Ray& ray, const Shape& shape,
                        float t_max, HitRecord& rec)
{
    switch (shape.type) {
        case ShapeType::Sphere:
            return intersect_sphere(ray, shape.sphere, t_max, rec);
        case ShapeType::Cylinder:
            return intersect_cylinder(ray, shape.cylinder, t_max, rec);
        case ShapeType::Disk:
            return intersect_disk(ray, shape.disk, t_max, rec);
        case ShapeType::Cube:
            return intersect_cube(ray, shape.cube, t_max, rec);
        case ShapeType::Plane:
            return intersect_plane(ray, shape.plane, t_max, rec);
    }
    return false;
}
