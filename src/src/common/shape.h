#pragma once
#include "math.h"

enum class ShapeType : uint8_t {
    Sphere,
    Cylinder,
    Disk,
    Cube,
    Plane,
};

struct Sphere {
    Vec3  center;
    float radius;
};

// Finite cylinder: axis-aligned segment of an infinite cylinder, capped by
// flat disks at ±half_height along `axis` from `center`.
struct Cylinder {
    Vec3  center;
    Vec3  axis;        // unit vector
    float radius;
    float half_height;
};

struct Disk {
    Vec3  center;
    Vec3  normal;      // unit vector (defines which side is "front")
    float radius;
};

// Axis-aligned box: center ± half_extents on each axis.
struct Cube {
    Vec3 center;
    Vec3 half_extents;
};

// Infinite plane through `center` with outward unit `normal`.
struct Plane {
    Vec3 center;
    Vec3 normal;
};

struct Shape {
    ShapeType type;
    union {
        Sphere   sphere;
        Cylinder cylinder;
        Disk     disk;
        Cube     cube;
        Plane    plane;
    };
    int material_id;
};
