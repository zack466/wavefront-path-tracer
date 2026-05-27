#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "scene/loader.h"
#include "output/image.h"
#include "cpu/render_cpu.h"
#include "cpu/wavefront_cpu.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <scene.json> [options]\n"
            "  --wavefront        Use queue-based wavefront renderer\n"
            "  --width  W         Override scene resolution width\n"
            "  --height H         Override scene resolution height\n"
            "  --spp    N         Override samples-per-pixel\n",
            argv[0]);
        return 1;
    }

    bool use_wavefront = false;
    std::string scene_path;
    int override_width  = 0;
    int override_height = 0;
    int override_spp    = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--wavefront") == 0) {
            use_wavefront = true;
        } else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            override_width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            override_height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--spp") == 0 && i + 1 < argc) {
            override_spp = std::atoi(argv[++i]);
        } else {
            scene_path = argv[i];
        }
    }
    if (scene_path.empty()) {
        std::fprintf(stderr, "Error: no scene file specified\n");
        return 1;
    }

    SceneData scene;
    try {
        scene = load_scene(scene_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error loading scene: %s\n", e.what());
        return 1;
    }

    if (override_width  > 0) scene.config.width  = override_width;
    if (override_height > 0) scene.config.height = override_height;
    if (override_spp    > 0) scene.config.spp    = override_spp;

    const RenderConfig& cfg = scene.config;

    const char* renderer_dir = use_wavefront ? "cpu_wavefront" : "cpu_recursive";
    std::filesystem::path out_dir = std::filesystem::path("outputs") / renderer_dir;
    std::filesystem::create_directories(out_dir);
    std::string output_path = (out_dir / std::filesystem::path(cfg.output_path).filename()).string();

    std::printf("Scene:    %s\n", scene_path.c_str());
    std::printf("Output:   %s\n", output_path.c_str());
    std::printf("Res:      %d x %d  spp=%d  max_depth=%d\n",
                cfg.width, cfg.height, cfg.spp, cfg.max_depth);
    std::printf("Shapes:   %zu  Materials: %zu  Lights: %zu\n",
                scene.shapes.size(), scene.materials.size(), scene.lights.size());
    std::printf("Renderer: %s\n", use_wavefront ? "wavefront" : "recursive");

    ImageBuffer img(cfg.width, cfg.height);

    auto t0 = std::chrono::steady_clock::now();
    if (use_wavefront)
        render_wavefront_cpu(scene, img);
    else
        render_cpu(scene, img);
    auto t1 = std::chrono::steady_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("Render: %.2f s  (%.1f Mrays/s)\n",
                secs,
                double(cfg.width) * cfg.height * cfg.spp / secs * 1e-6);

    tonemap(img, cfg.tonemap);

    if (cfg.denoise.enabled) {
        const DenoiseConfig& d = cfg.denoise;
        std::printf("Denoise: atrous  passes=%d  sigma_r=%.2f\n",
                    d.atrous_passes, d.sigma_r);
        img = atrous_filter(img, d.sigma_r, d.atrous_passes);
    }

    try {
        write_image(img, output_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error writing image: %s\n", e.what());
        return 1;
    }

    std::printf("Saved: %s\n", output_path.c_str());
    return 0;
}
