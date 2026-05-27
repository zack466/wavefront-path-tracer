#pragma once
#include "math.h"

enum class LightType : uint8_t {
    Point,
    Directional,
    Area,        // emissive shape — shape_id references a Shape with Emissive material
    // Volumetric,  // reach goal
};

struct PointLight {
    Vec3  position;
    Vec3  color;
    float intensity;
};

struct DirectionalLight {
    Vec3  direction;   // unit vector pointing FROM the light TOWARD the scene
    Vec3  color;
    float intensity;
};

// Area lights delegate all geometry to the referenced shape. The shape must
// have an Emissive material; its emission field carries the spectral radiance.
struct AreaLight {
    int shape_id;
};

struct Light {
    LightType type;
    union {
        PointLight       point;
        DirectionalLight directional;
        AreaLight        area;
    };
};
