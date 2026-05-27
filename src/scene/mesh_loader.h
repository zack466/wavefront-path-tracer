#pragma once
#include <string>
#include <vector>
#include "common/triangle.h"
#include "common/math.h"

// Transform applied to a mesh when loading.
struct MeshTransform {
    Vec3  translate = { 0.f, 0.f, 0.f };
    float scale     = 1.0f;
    float rotate_y  = 0.0f;   // degrees, around world Y axis
};

// Loads a Wavefront OBJ file and appends triangles to `out`.
// Smooth normals: uses per-vertex normals from OBJ if present, otherwise
// computes flat face normals.
// Returns the number of triangles added, or -1 on error.
int load_obj(const std::string& path, int material_id,
             const MeshTransform& xform, std::vector<Triangle>& out);
