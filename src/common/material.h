#pragma once
#include "math.h"

enum class MaterialType : uint8_t {
    Diffuse,          // Lambertian — cosine-weighted hemisphere scatter
    SpecularIdeal,    // perfect mirror reflection (delta BRDF)
    DielectricIdeal,  // perfect glass — Schlick Fresnel + Snell (delta BSDF)
    Emissive,         // light source — returns emission, no scatter
    GGXDielectric,    // rough/smooth glass — GGX microfacet BSDF, full Fresnel
                      //   roughness=0 ≈ perfect glass; roughness=0.3 ≈ frosted
    // GGXMetal,      // future: GGX conductor (rough metal)
};

struct Material {
    MaterialType type;
    Vec3  albedo;      // reflectance tint (also used as filter color for glass)
    Vec3  emission;    // radiance emitted (non-zero only for Emissive type)
    float ior;         // index of refraction (DielectricIdeal only; default 1.5)

    // Reserved for future PBR parameters (zero-initialised):
    float roughness;   // GGX roughness [0,1]
    float metallic;    // metallic factor [0,1]
};
