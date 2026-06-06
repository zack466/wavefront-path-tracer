#pragma once
#include "math.h"

// Axis-aligned bounding box.  HD so it can be used in OptiX IS programs.
struct AABB {
    Vec3 min_pt;
    Vec3 max_pt;

    // Ray-AABB slab intersection. Returns true and sets t_near/t_far.
    // Caller should check t_near < ray.tmax and t_far > ray.tmin.
    HD bool intersect(const Ray& ray, float& t_near, float& t_far) const {
        t_near = ray.tmin;
        t_far  = ray.tmax;
        for (int i = 0; i < 3; ++i) {
            float inv_d = 1.0f / ray.dir[i];
            float t0 = (min_pt[i] - ray.origin[i]) * inv_d;
            float t1 = (max_pt[i] - ray.origin[i]) * inv_d;
            if (inv_d < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            t_near = fmaxf(t_near, t0);
            t_far  = fminf(t_far,  t1);
        }
        return t_near <= t_far;
    }

    HD Vec3  centroid()     const { return (min_pt + max_pt) * 0.5f; }
    HD float surface_area() const {
        Vec3 d = max_pt - min_pt;
        return 2.0f * (d.x*d.y + d.y*d.z + d.z*d.x);
    }
    HD bool  is_valid()     const {
        return min_pt.x <= max_pt.x &&
               min_pt.y <= max_pt.y &&
               min_pt.z <= max_pt.z;
    }
};

HD AABB aabb_empty() {
    return { Vec3( 1e30f),  Vec3(-1e30f) };
}
HD AABB aabb_union(AABB a, AABB b) {
    return { min_vec(a.min_pt, b.min_pt), max_vec(a.max_pt, b.max_pt) };
}
HD AABB aabb_point(Vec3 p) { return { p, p }; }
HD AABB aabb_expand(AABB a, Vec3 p) {
    return { min_vec(a.min_pt, p), max_vec(a.max_pt, p) };
}

// Per-analytic-shape AABB helpers
#include "shape.h"

HD AABB sphere_aabb(const Sphere& s) {
    Vec3 r(s.radius);
    return { s.center - r, s.center + r };
}
HD AABB cylinder_aabb(const Cylinder& c) {
    Vec3 ax = abs_vec(c.axis) * c.half_height;
    Vec3 rad(c.radius);
    // Loose bound: AABB of the bounding sphere for simplicity
    float half_diag = sqrtf(c.radius*c.radius + c.half_height*c.half_height);
    Vec3 d(half_diag);
    return { c.center - d, c.center + d };
}
HD AABB disk_aabb(const Disk& d) {
    // Tight axis-aligned bound: for each axis i the max extent is sqrt(1-ni^2)*r,
    // i.e. the projection of the disk radius onto that axis.
    Vec3 n    = normalize(d.normal);
    Vec3 half = { sqrtf(fmaxf(0.f, 1.f - n.x*n.x)) * d.radius + 1e-3f,
                  sqrtf(fmaxf(0.f, 1.f - n.y*n.y)) * d.radius + 1e-3f,
                  sqrtf(fmaxf(0.f, 1.f - n.z*n.z)) * d.radius + 1e-3f };
    return { d.center - half, d.center + half };
}

HD AABB cube_aabb(const Cube& c) {
    return { c.center - c.half_extents, c.center + c.half_extents };
}
HD AABB plane_aabb(const Plane& p) {
    // Infinite plane: approximate with a large slab for BVH.
    Vec3 n = normalize(p.normal);
    const float hs = 10000.f;
    Vec3 r = { sqrtf(fmaxf(0.f, 1.f - n.x*n.x)) * hs + 1e-3f,
               sqrtf(fmaxf(0.f, 1.f - n.y*n.y)) * hs + 1e-3f,
               sqrtf(fmaxf(0.f, 1.f - n.z*n.z)) * hs + 1e-3f };
    return { p.center - r, p.center + r };
}

HD AABB shape_aabb(const Shape& s) {
    switch (s.type) {
        case ShapeType::Sphere:   return sphere_aabb(s.sphere);
        case ShapeType::Cylinder: return cylinder_aabb(s.cylinder);
        case ShapeType::Disk:     return disk_aabb(s.disk);
        case ShapeType::Cube:     return cube_aabb(s.cube);
        case ShapeType::Plane:    return plane_aabb(s.plane);
    }
    return aabb_empty();
}
