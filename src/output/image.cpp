#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>

// ── Tone mapping ──────────────────────────────────────────────────────────────

static float gamma_encode(float v) {
    return std::pow(std::max(0.f, v), 1.0f / 2.2f);
}

// Reinhard per-channel
static Vec3 tonemap_reinhard_v(Vec3 c) {
    return { c.x/(c.x+1.f), c.y/(c.y+1.f), c.z/(c.z+1.f) };
}

// ACES filmic approximation (Krzysztof Narkowicz, 2016)
static Vec3 tonemap_aces_v(Vec3 x) {
    return clamp_vec((x*(2.51f*x + Vec3(0.03f))) / (x*(2.43f*x + Vec3(0.59f)) + Vec3(0.14f)),
                     0.f, 1.f);
}

void tonemap(ImageBuffer& buf, TonemapMode mode) {
    int n = buf.width * buf.height;
    for (int i = 0; i < n; ++i) {
        float* p = buf.data.data() + i * 3;
        Vec3 c { p[0], p[1], p[2] };

        switch (mode) {
            case TonemapMode::Gamma:
                // Just gamma encode (no HDR compression)
                c = clamp_vec(c, 0.f, 1.f);
                break;
            case TonemapMode::Reinhard:
                c = tonemap_reinhard_v(c);
                break;
            case TonemapMode::ACES:
            default:
                c = tonemap_aces_v(c);
                break;
        }

        p[0] = gamma_encode(c.x);
        p[1] = gamma_encode(c.y);
        p[2] = gamma_encode(c.z);
    }
}

// ── Writers ───────────────────────────────────────────────────────────────────

static uint8_t to_u8(float v) {
    return uint8_t(std::max(0.f, std::min(1.f, v)) * 255.f + 0.5f);
}

void write_ppm(const ImageBuffer& buf, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open output file: " + path);
    f << "P3\n" << buf.width << ' ' << buf.height << "\n255\n";
    for (int i = 0; i < buf.width * buf.height; ++i) {
        const float* p = buf.data.data() + i * 3;
        f << int(to_u8(p[0])) << ' '
          << int(to_u8(p[1])) << ' '
          << int(to_u8(p[2])) << '\n';
    }
}

void write_png(const ImageBuffer& buf, const std::string& path) {
    std::vector<uint8_t> pixels(buf.width * buf.height * 3);
    for (int i = 0; i < buf.width * buf.height * 3; ++i)
        pixels[i] = to_u8(buf.data[i]);

    int ok = stbi_write_png(path.c_str(), buf.width, buf.height, 3,
                            pixels.data(), buf.width * 3);
    if (!ok) throw std::runtime_error("Failed to write PNG: " + path);
}

void write_image(const ImageBuffer& buf, const std::string& path) {
    if (path.size() >= 4 && path.substr(path.size()-4) == ".ppm")
        write_ppm(buf, path);
    else
        write_png(buf, path);
}

// ── À-trous wavelet filter ────────────────────────────────────────────────────
// Applies a 3×3 B-spline kernel at step sizes 1, 2, 4, … 2^(passes-1).
// Each pass removes noise at its spatial scale while the colour-distance weight
// preserves edges. The multi-scale cascade removes noise at all frequencies,
// which is why it outperforms a single wide bilateral pass on path-tracing noise.
//
// Reference: Dammertz et al., "Edge-Avoiding À-Trous Wavelet Transform for
// fast Global Illumination Filtering", HPG 2010.

ImageBuffer atrous_filter(const ImageBuffer& src, float sigma_r, int passes) {
    // 3×3 B-spline kernel weights (sum = 1)
    static const float K[3][3] = {
        { 1.f/16, 2.f/16, 1.f/16 },
        { 2.f/16, 4.f/16, 2.f/16 },
        { 1.f/16, 2.f/16, 1.f/16 },
    };

    float inv_2sr = 1.0f / (2.0f * sigma_r * sigma_r);

    ImageBuffer cur = src;
    ImageBuffer nxt(src.width, src.height);

    for (int pass = 0; pass < passes; ++pass) {
        int step = 1 << pass;   // 1, 2, 4, 8, 16

        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                const float* cp = cur.pixel(x, y);
                float cr = cp[0], cg = cp[1], cb = cp[2];
                float sumr = 0.f, sumg = 0.f, sumb = 0.f, sumw = 0.f;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        // Clamp to border instead of skip — avoids dark edges
                        int nx = std::max(0, std::min(src.width  - 1, x + dx * step));
                        int ny = std::max(0, std::min(src.height - 1, y + dy * step));
                        const float* np = cur.pixel(nx, ny);
                        float dr = np[0]-cr, dg = np[1]-cg, db = np[2]-cb;
                        float w = K[dy+1][dx+1]
                                * expf(-(dr*dr + dg*dg + db*db) * inv_2sr);
                        sumr += w * np[0]; sumg += w * np[1]; sumb += w * np[2];
                        sumw += w;
                    }
                }

                float* dp = nxt.pixel(x, y);
                dp[0] = sumr/sumw; dp[1] = sumg/sumw; dp[2] = sumb/sumw;
            }
        }
        std::swap(cur.data, nxt.data);
    }
    return cur;
}
