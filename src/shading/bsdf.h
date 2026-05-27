#pragma once
#include "sampling.h"
#include "common/material.h"
#include "common/hit.h"

// ── BSDF sample result ────────────────────────────────────────────────────────

struct BSDFSample {
    Vec3 direction;    // scattered ray direction (world space, normalised)
    Vec3 weight;       // throughput contribution: BSDF * cos(theta) / pdf
    bool is_specular;  // true → skip NEE (delta BSDFs gain nothing from explicit light sampling)
};

// ── Main dispatch ─────────────────────────────────────────────────────────────
// Given a hit and the OUTGOING direction wo (-ray.dir at the hit point),
// sample the BSDF and return the new ray direction + weight.
//
// wo must be normalised and in the same hemisphere as hit.normal (guaranteed
// by the HitRecord convention of storing the shading normal opposing the ray).

HD BSDFSample sample_bsdf(const Material& mat, const HitRecord& hit,
                           Vec3 wo, uint& seed)
{
    switch (mat.type) {

    // ── Lambertian diffuse ────────────────────────────────────────────────────
    // PDF  = cos(theta) / PI
    // BRDF = albedo / PI
    // Weight = BRDF * cos(theta) / PDF = albedo   (PI and cos cancel)
    case MaterialType::Diffuse: {
        Vec3 local = cosine_sample_hemisphere(rand_float(seed), rand_float(seed));
        return { local_to_world(local, hit.normal), mat.albedo, false };
    }

    // ── Perfect mirror ────────────────────────────────────────────────────────
    // Delta BRDF: weight = albedo, direction is deterministic.
    case MaterialType::SpecularIdeal: {
        Vec3 dir = reflect_dir(wo, hit.normal);
        return { dir, mat.albedo, true };
    }

    // ── Perfect glass (Fresnel + Snell) ───────────────────────────────────────
    // Stochastically choose reflect vs. refract weighted by Schlick reflectance.
    // front_face tells us whether we are entering (ni=1, nt=ior) or exiting
    // (ni=ior, nt=1).
    case MaterialType::DielectricIdeal: {
        float eta     = hit.front_face ? (1.0f / mat.ior) : mat.ior;
        float cos_i   = dot(wo, hit.normal);   // >= 0 by HitRecord convention
        float reflect_prob = schlick(cos_i, mat.ior);

        Vec3 dir;
        Vec3 refracted;
        if (rand_float(seed) < reflect_prob || !refract_dir(wo, hit.normal, eta, refracted)) {
            dir = reflect_dir(wo, hit.normal);
        } else {
            dir = refracted;
        }
        return { dir, mat.albedo, true };
    }

    // ── GGX microfacet dielectric (rough/smooth glass) ────────────────────────
    // Uses the Visible Normal Distribution Function (VNDF) for importance
    // sampling, giving the unbiased weight:
    //   reflect:  F(m,wo) * G1(wi,m) * albedo
    //   refract:  (1-F(m,wo)) * G1(wi,m) * albedo
    // where m is the sampled microsurface normal and G1 is Smith masking.
    //
    // roughness=0.0 → essentially perfect glass (alpha clamped to ~0).
    // roughness=0.3 → clearly frosted/sandblasted glass.
    case MaterialType::GGXDielectric: {
        float alpha = fmaxf(mat.roughness * mat.roughness, 1e-4f);

        // Transform wo to local frame where hit.normal = z-axis
        Vec3 t, b;
        make_onb(hit.normal, t, b);
        Vec3 wo_l = { dot(wo, t), dot(wo, b), dot(wo, hit.normal) };
        if (wo_l.z <= 0.0f) return { hit.normal, Vec3(0.f), true };

        // Sample microsurface normal using VNDF
        Vec3 m_l = sample_ggx_vndf(wo_l, alpha, rand_float(seed), rand_float(seed));
        Vec3 m   = m_l.x * t + m_l.y * b + m_l.z * hit.normal;

        float cos_im = dot(wo, m);
        if (cos_im <= 0.0f) return { hit.normal, Vec3(0.f), true };

        // eta = ni / nt (incident / transmitted)
        float eta = hit.front_face ? (1.0f / mat.ior) : mat.ior;

        // Full Fresnel reflectance (more accurate than Schlick at grazing angles)
        float F = fresnel_dielectric(cos_im, eta);

        Vec3 new_dir;
        if (rand_float(seed) < F) {
            new_dir = reflect_dir(wo, m);
            if (dot(new_dir, hit.normal) <= 0.0f)
                return { hit.normal, Vec3(0.f), true };
        } else {
            Vec3 refracted;
            if (!refract_dir(wo, m, eta, refracted)) {
                new_dir = reflect_dir(wo, m);  // numerical TIR fallback
            } else {
                new_dir = refracted;
            }
        }

        // Smith G1 masking for the sampled outgoing direction.
        // For transmission, the outgoing direction points into the material
        // (dot < 0 w.r.t. hit.normal), so we use the absolute value.
        float cos_out = fabsf(dot(new_dir, hit.normal));
        float G1_wi   = smith_g1_ggx(cos_out, alpha);

        return { new_dir, mat.albedo * G1_wi, true };
    }

    // Emissive materials should be handled before calling sample_bsdf;
    // return a degenerate sample here so the path terminates.
    case MaterialType::Emissive:
    default:
        return { hit.normal, Vec3(0.f), true };
    }
}
