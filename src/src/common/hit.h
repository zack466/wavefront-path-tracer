#pragma once
#include "math.h"

// Result of a ray-scene intersection.
//
// Convention: both `normal` and `geo_normal` oppose the incident ray direction.
//   normal     = shading normal (may be interpolated for smooth meshes)
//   geo_normal = flat face/surface normal (always exact geometric quantity)
//
// These agree for analytic shapes and flat-shaded triangles. They differ for
// smooth-shaded triangles, which is exactly why geo_normal exists: offsetting
// the bounce ray origin along geo_normal avoids self-intersection even when the
// shading normal deviates from the surface plane.
struct HitRecord {
    Vec3  point;
    Vec3  normal;      // shading normal  (opposes ray direction)
    Vec3  geo_normal;  // geometric normal (opposes ray direction)
    float t;
    int   material_id;
    int   shape_id;
    bool  front_face;
};
