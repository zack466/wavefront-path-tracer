#pragma once
#include "common/math.h"

// ── RNG ───────────────────────────────────────────────────────────────────────
// xorshift32: stateless except for a 32-bit seed; fast on both CPU and GPU.

HD uint xorshift32(uint& state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

// Generates a uniform float in [0, 1).
HD float rand_float(uint& state) {
    return float(xorshift32(state)) * (1.0f / 4294967296.0f);
}

// Wang hash: good for seeding per-pixel RNG state from a flat index.
HD uint wang_hash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed ^= seed >> 4u;
    seed *= 0x27d4eb2du;
    seed ^= seed >> 15u;
    return seed != 0u ? seed : 1u;  // xorshift32 must not start at 0
}

// ── Orthonormal basis ─────────────────────────────────────────────────────────
// Build a tangent frame from a unit normal `n`.
// Avoids degeneracy by choosing a different reference axis when n is nearly
// aligned with x.

HD void make_onb(Vec3 n, Vec3& tangent, Vec3& bitangent) {
    Vec3 ref = (fabsf(n.x) < 0.9f) ? Vec3(1.f, 0.f, 0.f) : Vec3(0.f, 1.f, 0.f);
    tangent   = normalize(cross(n, ref));
    bitangent = cross(n, tangent);
}

// Transforms a local-frame direction (where z maps to `n`) into world space.
HD Vec3 local_to_world(Vec3 local, Vec3 n) {
    Vec3 t, b;
    make_onb(n, t, b);
    return local.x * t + local.y * b + local.z * n;
}

// ── Directional sampling ──────────────────────────────────────────────────────

// Cosine-weighted hemisphere sample in local frame (z = up = normal direction).
// PDF = cos(theta) / PI.
HD Vec3 cosine_sample_hemisphere(float u1, float u2) {
    float phi       = PT_TWO_PI * u1;
    float cos_theta = sqrtf(u2);
    float sin_theta = sqrtf(1.0f - u2);
    return { cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta };
}

// Uniform sphere sample. PDF = 1 / (4*PI).
HD Vec3 uniform_sample_sphere(float u1, float u2) {
    float cos_theta = 1.0f - 2.0f * u1;
    float sin_theta = sqrtf(fmaxf(0.f, 1.0f - cos_theta * cos_theta));
    float phi       = PT_TWO_PI * u2;
    return { cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta };
}

// ── Mirror / refraction ───────────────────────────────────────────────────────

// Mirror reflection: both wo and result are in the same hemisphere as n.
// wo = outgoing direction FROM the surface (equals -ray.dir at the hit point).
HD Vec3 reflect_dir(Vec3 wo, Vec3 n) {
    return -wo + 2.0f * dot(wo, n) * n;
}

// Snell's law refraction.
// wo: outgoing direction (same hemisphere as n).
// n:  shading normal (also same hemisphere as wo, by HitRecord convention).
// eta = ni / nt (ratio of the IOR on the incident side to the transmitted side).
// Returns false on total internal reflection.
HD bool refract_dir(Vec3 wo, Vec3 n, float eta, Vec3& refracted) {
    float cos_i  = dot(wo, n);           // >= 0 by convention
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f) return false;    // total internal reflection
    float cos_t  = sqrtf(1.0f - sin2_t);
    refracted = -eta * wo + (eta * cos_i - cos_t) * n;
    return true;
}

// Schlick Fresnel approximation (used by DielectricIdeal).
HD float schlick(float cos_i, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cos_i, 5.0f);
}

// ── GGX microfacet utilities (used by GGXDielectric) ─────────────────────────

// Full polarisation-averaged Fresnel reflectance for a dielectric interface.
// eta = ni/nt (ratio of incident-side IOR to transmitted-side IOR).
// Returns reflectance R in [0,1]; transmittance = 1 - R.
HD float fresnel_dielectric(float cos_i, float eta) {
    cos_i = fabsf(cos_i);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f) return 1.0f;                    // total internal reflection
    float cos_t = sqrtf(1.0f - sin2_t);
    float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (rs * rs + rp * rp);
}

// Smith masking function G1 for GGX (Trowbridge-Reitz).
// cos_theta: cosine of the angle between the direction and the macrosurface normal.
// alpha:     GGX roughness (NOT roughness^2; we square it inside).
HD float smith_g1_ggx(float cos_theta, float alpha) {
    if (cos_theta <= 0.0f) return 0.0f;
    float cos2  = cos_theta * cos_theta;
    float tan2  = (1.0f - cos2) / cos2;
    return 2.0f / (1.0f + sqrtf(1.0f + alpha * alpha * tan2));
}

// Sample the GGX Visible Normal Distribution Function (VNDF) in the local frame
// where the macrosurface normal is z=(0,0,1).
//
// wo_local: outgoing direction in local frame (wo_local.z must be > 0).
// alpha:    GGX roughness parameter.
// Returns the sampled microsurface normal (also in local frame, z > 0).
//
// Heitz 2018, "Sampling the GGX Distribution of Visible Normals", JCGT.
HD Vec3 sample_ggx_vndf(Vec3 wo_local, float alpha, float u1, float u2) {
    // Stretch wo into the GGX hemisphere configuration
    Vec3 Vh = normalize(Vec3(alpha * wo_local.x, alpha * wo_local.y, wo_local.z));

    // Orthonormal basis (T1, T2, Vh)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    Vec3 T1 = lensq > 1e-7f
                ? Vec3(-Vh.y, Vh.x, 0.f) / sqrtf(lensq)
                : Vec3(1.f, 0.f, 0.f);
    Vec3 T2 = cross(Vh, T1);

    // Sample point on the projected disk
    float r   = sqrtf(u1);
    float phi = PT_TWO_PI * u2;
    float t1  = r * cosf(phi);
    float t2  = r * sinf(phi);
    float s   = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    // Reproject onto hemisphere, then unstretch back to ellipsoid
    Vec3 Nh = t1 * T1 + t2 * T2
            + sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2)) * Vh;
    return normalize(Vec3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
}
