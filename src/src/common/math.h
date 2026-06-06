#pragma once
#include "types.h"

// Pull in C math functions on CPU; CUDA provides them as device built-ins.
#ifndef __CUDACC__
  #include <cmath>
#endif

// ── Vec3 ──────────────────────────────────────────────────────────────────────

struct Vec3 {
    float x, y, z;

    HD Vec3()                     : x(0.f), y(0.f), z(0.f) {}
    HD Vec3(float v)              : x(v),   y(v),   z(v)   {}
    HD Vec3(float x, float y, float z) : x(x), y(y), z(z)  {}

    HD float  operator[](int i) const { return (&x)[i]; }
    HD float& operator[](int i)       { return (&x)[i]; }

    HD Vec3 operator+(Vec3 o)  const { return {x+o.x, y+o.y, z+o.z}; }
    HD Vec3 operator-(Vec3 o)  const { return {x-o.x, y-o.y, z-o.z}; }
    HD Vec3 operator*(Vec3 o)  const { return {x*o.x, y*o.y, z*o.z}; }
    HD Vec3 operator/(Vec3 o)  const { return {x/o.x, y/o.y, z/o.z}; }
    HD Vec3 operator*(float s) const { return {x*s,   y*s,   z*s};   }
    HD Vec3 operator/(float s) const { float r=1.f/s; return {x*r, y*r, z*r}; }
    HD Vec3 operator-()        const { return {-x, -y, -z}; }

    HD Vec3& operator+=(Vec3  o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    HD Vec3& operator-=(Vec3  o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
    HD Vec3& operator*=(Vec3  o) { x*=o.x; y*=o.y; z*=o.z; return *this; }
    HD Vec3& operator*=(float s) { x*=s;   y*=s;   z*=s;   return *this; }
    HD Vec3& operator/=(float s) { float r=1.f/s; x*=r; y*=r; z*=r; return *this; }
};

HD Vec3 operator*(float s, Vec3 v) { return v * s; }

HD float dot(Vec3 a, Vec3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
HD Vec3  cross(Vec3 a, Vec3 b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

HD float length_sq(Vec3 v)  { return dot(v, v); }
HD float length(Vec3 v)     { return sqrtf(length_sq(v)); }
HD Vec3  normalize(Vec3 v)  { return v / length(v); }

HD float max_component(Vec3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
HD float min_component(Vec3 v) { return fminf(v.x, fminf(v.y, v.z)); }

HD Vec3 abs_vec(Vec3 v) { return { fabsf(v.x), fabsf(v.y), fabsf(v.z) }; }
HD Vec3 min_vec(Vec3 a, Vec3 b) {
    return { fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z) };
}
HD Vec3 max_vec(Vec3 a, Vec3 b) {
    return { fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z) };
}
HD Vec3 clamp_vec(Vec3 v, float lo, float hi) {
    return { fmaxf(lo, fminf(hi, v.x)),
             fmaxf(lo, fminf(hi, v.y)),
             fmaxf(lo, fminf(hi, v.z)) };
}
HD Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + t * (b - a); }

// Perceptual luminance (BT.709 coefficients) — used for RR and firefly clamping
HD float luma(Vec3 c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

// ── Ray ───────────────────────────────────────────────────────────────────────

struct Ray {
    Vec3  origin;
    Vec3  dir;    // normalized
    float tmin;   // PT_EPSILON keeps self-intersection at bay
    float tmax;
};

// Always normalizes dir and sets canonical tmin/tmax.
HD Ray make_ray(Vec3 origin, Vec3 dir) {
    return { origin, normalize(dir), PT_EPSILON, PT_INFINITY };
}

HD Vec3 ray_at(const Ray& r, float t) { return r.origin + t * r.dir; }
