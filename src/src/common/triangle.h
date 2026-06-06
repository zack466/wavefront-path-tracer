#pragma once
#include "math.h"

// Vertex-only layout used by GPU BVH intersection tests (36 bytes).
// At 871 K triangles this array is 31 MB — fits in the RTX A5000's 40 MB L2,
// so intersection tests stay cache-resident after the first bounce.
struct TriangleV {
    Vec3 v[3];
};

// Full triangle data (vertices + shading normals + material).
// Loaded only once per bounce for the *winning* hit, not for every candidate test.
struct Triangle {
    Vec3 v[3];            // vertex positions
    Vec3 n[3];            // per-vertex shading normals
    bool smooth_normals;  // true → interpolate n[]; false → use face normal
    int  material_id;
};
