#pragma once
#include <string>
#include <vector>
#include "common/math.h"
#include "scene/scene.h"

// ── Float32 accumulation buffer ───────────────────────────────────────────────
// Stores linear-light RGB values (no gamma). Pixel (x, y) is at index
// (y * width + x) * 3. y=0 is the TOP row (image convention; camera_ray()
// already handles the y-flip so no flip is needed at write time).

struct ImageBuffer {
    std::vector<float> data;  // width * height * 3 floats, linear RGB
    int width;
    int height;

    ImageBuffer(int w, int h) : data(w * h * 3, 0.f), width(w), height(h) {}

    float* pixel(int x, int y) { return data.data() + (y * width + x) * 3; }
    const float* pixel(int x, int y) const { return data.data() + (y * width + x) * 3; }

    void add(int x, int y, Vec3 c) {
        float* p = pixel(x, y);
        p[0] += c.x;  p[1] += c.y;  p[2] += c.z;
    }
    void scale(float inv_spp) {
        for (float& v : data) v *= inv_spp;
    }
};

// ── Tone mapping ──────────────────────────────────────────────────────────────
// All functions operate in-place on a linear HDR buffer.

void tonemap(ImageBuffer& buf, TonemapMode mode);

// ── Writers ───────────────────────────────────────────────────────────────────
// Write the buffer (assumed already tone-mapped to [0,1]) to disk.

void write_ppm(const ImageBuffer& buf, const std::string& path);
void write_png(const ImageBuffer& buf, const std::string& path);

// Dispatch based on file extension (.ppm → PPM, otherwise PNG).
void write_image(const ImageBuffer& buf, const std::string& path);

// ── Denoising ─────────────────────────────────────────────────────────────────
// Apply AFTER tone mapping (values expected in ~[0,1]).
//
// À-trous ("with holes") edge-aware wavelet filter.
// Applies a 3×3 B-spline kernel at step sizes 1, 2, 4, … 2^(passes-1).
// Multi-scale: removes noise at all spatial frequencies while preserving edges.
// sigma_r: colour-distance threshold controlling edge sharpness.
ImageBuffer atrous_filter(const ImageBuffer& src, float sigma_r, int passes);
