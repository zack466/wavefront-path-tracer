#pragma once
#include <cstdint>   // uint8_t — needed on both CPU and CUDA device paths

// ── Cross-platform host/device annotation ─────────────────────────────────────
// All math, BSDF, and intersection functions are marked HD so they compile on
// both CPU and CUDA device code without modification.
#ifdef __CUDACC__
  #define HD __host__ __device__ __forceinline__
#else
  #define HD inline
#endif

// ── Scalar constants ──────────────────────────────────────────────────────────
#define PT_PI        3.14159265358979323846f
#define PT_INV_PI    0.31830988618379067154f
#define PT_TWO_PI    6.28318530717958647692f
#define PT_HALF_PI   1.57079632679489661923f
#define PT_EPSILON   1e-4f   // shadow bias / tmin
#define PT_INFINITY  1e30f

// Unsigned int alias (used for RNG state)
using uint = unsigned int;
