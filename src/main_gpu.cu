// GPU wavefront path tracer entry point.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "scene/loader.h"
#include "output/image.h"
#include "gpu/render_gpu.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <scene.json> [options]\n"
            "  --width  W    Override resolution width\n"
            "  --height H    Override resolution height\n"
            "  --spp    N    Override samples-per-pixel\n"
            "  --optix       Use OptiX RT-core intersection (requires ENABLE_OPTIX build)\n",
            argv[0]);
        return 1;
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }
    cudaSetDevice(0);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s  (sm_%d%d, %.0f MB, %.0f GB/s)\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1024.0 / 1024.0,
                prop.memoryClockRate * 2.0 * (prop.memoryBusWidth / 8) / 1e6);

    std::string scene_path;
    int  override_width  = 0;
    int  override_height = 0;
    int  override_spp    = 0;
    bool use_optix       = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            override_width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            override_height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--spp") == 0 && i + 1 < argc) {
            override_spp = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--optix") == 0) {
            use_optix = true;
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

    std::filesystem::path out_dir = std::filesystem::path("outputs") /
        (use_optix ? "gpu_optix" : "gpu_cuda_bvh");
    std::filesystem::create_directories(out_dir);
    std::string output_path =
        (out_dir / std::filesystem::path(cfg.output_path).filename()).string();

    std::printf("Scene:    %s\n", scene_path.c_str());
    std::printf("Output:   %s\n", output_path.c_str());
    std::printf("Res:      %d x %d  spp=%d  max_depth=%d\n",
                cfg.width, cfg.height, cfg.spp, cfg.max_depth);
    std::printf("Shapes:   %zu  Triangles: %zu  Materials: %zu  Lights: %zu\n",
                scene.shapes.size(), scene.triangles.size(),
                scene.materials.size(), scene.lights.size());
    std::printf("Renderer: GPU wavefront (%s)\n",
                use_optix ? "OptiX RT cores" : "CUDA software BVH");

    ImageBuffer img(cfg.width, cfg.height);

    auto t0 = std::chrono::steady_clock::now();
    try {
        render_wavefront_gpu(scene, img, use_optix);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Render error: %s\n", e.what());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("Render: %.2f s  (%.1f Mrays/s)\n",
                secs, double(cfg.width) * cfg.height * cfg.spp / secs * 1e-6);

    tonemap(img, cfg.tonemap);

    if (cfg.denoise.enabled) {
        std::printf("Denoise: atrous  passes=%d  sigma_r=%.2f\n",
                    cfg.denoise.atrous_passes, cfg.denoise.sigma_r);
        img = atrous_filter(img, cfg.denoise.sigma_r, cfg.denoise.atrous_passes);
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
