#pragma once
#include <string>
#include <vector>
#include "common/material.h"
#include "common/shape.h"
#include "common/triangle.h"
#include "common/light.h"
#include "common/camera.h"
#include "bvh/bvh.h"

// ── Render configuration ──────────────────────────────────────────────────────

enum class TonemapMode  { Gamma, Reinhard, ACES };
enum class BackgroundMode { Color, Sky };   // Sky = gradient; Color = flat Vec3
struct DenoiseConfig {
    bool  enabled       = false;
    float sigma_r       = 0.12f;  // colour-distance threshold
    int   atrous_passes = 5;      // scales: 1, 2, 4, 8, 16 pixels
};

struct RenderConfig {
    int    width         = 800;
    int    height        = 600;
    int    spp           = 256;
    int    max_depth     = 20;
    int    rr_min_depth  = 3;
    float  firefly_clamp = 0.f;
    Vec3           background    = { 0.f, 0.f, 0.f };
    BackgroundMode bg_mode       = BackgroundMode::Color;
    std::string    output_path   = "render.png";
    TonemapMode    tonemap       = TonemapMode::ACES;
    DenoiseConfig  denoise;
};

// ── Scene aggregate ───────────────────────────────────────────────────────────
// CPU-side scene description. Holds STL containers so it is not directly
// uploadable to the GPU. Phase 3 will add a DeviceScene that mirrors these
// arrays in device memory.

struct SceneData {
    std::vector<Shape>    shapes;      // analytic primitives (sphere, cylinder, disk)
    std::vector<Triangle> triangles;   // flat triangle list from all loaded meshes
    std::vector<Material> materials;
    std::vector<Light>    lights;
    CameraParams          camera;
    RenderConfig          config;
    SceneAccel            accel;       // BVH built by load_scene(), covers shapes + triangles
};
