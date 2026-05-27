# CS 179 Final Project - Path Tracer, CPU Demo

Sample renderings can be found in `outputs/cpu_recursive`.

## Building and Running

The project follows normal cmake conventions.
You can build it using:

```bash
mkdir build
cd build && cmake .. && make
```

This will produce the executable `path_tracer_cpu` in the `build/` directory.
To run the program and render a scene, you can provide this executable with a scene description in the form of a json file.
The scene description provides default values for the rendering resolution and number of samples taken per pixel, but these can also be overriden by command-line settings.
By default, the recursive path tracing algorithm is used by the renderer, but you can also switch to the wavefront algorithm using a flag, which is intended to be very similar to how wavefront path tracing will be implemented on the actual GPU.

```bash
# render scene
./build/path_tracer_cpu scenes/cube_cornell.json

# override rendering systems
./build/path_tracer_cpu --width 4196 --height 2160 --spp 2048 scenes/cube_cornell.json

# use wavefront renderer
./build/path_tracer_cpu --wavefront scenes/cube_cornell.json
```

By default, the output renderings will appear in `outputs/cpu_recursive` if using the normal renderer, or `outputs/cpu_wavefront` if using the wavefront algorithm.
Both CPU implementations are naively parallelized using OpenMP.

## Description

As described in the proposal, I have implemented a path tracer that can render photo-realistic images by simulating the propagation of light in a scene.

This renderer includes all featured mentioned in the proposal, including:
- Shape primitives:
  - Sphere, cylinder, disk, cube, plane, triangle meshes (loaded as `.obj` files)
  - accelerated ray-scene intersection using a Bounded Volume Hierarchy (BVH)
- Materials:
  - Diffuse (matte)
  - Specular (reflective)
  - Dieletric (refractive)
  - Emissive (bright)
- Postprocessing:
  - ACES tonemapping
  - Atrous wavelet filter for denoising

Scenes can then be described using a simple description language in JSON, which allows specifying shapes, materials, camera settings, and other rendering options.
For example, this is the scene description for the "bunny_studio" scene.

```json
{
  "camera": {
    "position": [0.0, 2.5, 6.0],
    "look_at":  [0.0, 1.2, 0.0],
    "up":       [0.0, 1.0, 0.0],
    "vfov_deg": 36.0
  },
  "render": {
    "width":         960,
    "height":        540,
    "spp":           1024,
    "max_depth":     20,
    "rr_min_depth":  3,
    "firefly_clamp": 20.0,
    "background":    [0.04, 0.04, 0.06],
    "output":        "bunny_studio.png",
    "tonemapping":   "aces",
    "denoise": {
      "enabled":       true,
      "sigma_r":       0.06,
      "atrous_passes": 3
    }
  },
  "materials": [
    { "type": "diffuse",          "albedo": [0.70, 0.62, 0.48] },
    { "type": "rough_dielectric", "albedo": [0.97, 0.97, 0.95], "ior": 1.46, "roughness": 0.10 },
    { "type": "emissive",         "albedo": [0.0, 0.0, 0.0], "emission": [14.0, 11.0,  6.0] },
    { "type": "emissive",         "albedo": [0.0, 0.0, 0.0], "emission": [ 4.0,  7.0, 16.0] },
    { "type": "emissive",         "albedo": [0.0, 0.0, 0.0], "emission": [10.0, 11.0, 14.0] }
  ],
  "shapes": [
    { "type": "plane",  "center": [0.0,   0.0, 0.0], "normal": [0, 1, 0], "material": 0 },
    {
      "type":      "mesh",
      "file":      "../models/bunny.obj",
      "material":  1,
      "translate": [0.22, -0.03, -0.11],
      "scale":     1.5,
      "rotate_y":  20.0
    },
    { "type": "sphere", "center": [ 5.0, 6.0,  5.0], "radius": 1.5, "material": 2 },
    { "type": "sphere", "center": [-5.0, 5.0,  3.0], "radius": 1.0, "material": 3 },
    { "type": "sphere", "center": [-1.0, 4.0, -6.0], "radius": 0.8, "material": 4 }
  ],
  "lights": [
    { "type": "area", "shape_id": 1 },
    { "type": "area", "shape_id": 2 },
    { "type": "area", "shape_id": 3 }
  ]
}
```

## Test Outputs

You can automatically render a selection of scenes using both renders using the provided `benchmark.py` script.

```bash
# renders a selection of scenes using the reference renderer (CPU, recursive algorithm)
python3 benchmark.py --recursive-only
```

You can find sample renderings of all scenes in `output/cpu_recursive`.

Also, running the script using the `--test` flag will render each scene using both the recursive (default) and wavefront renderers, and then compare the pixel values to ensure that both implementations are effectively the same.
This is important since the GPU will simply accelerate the wavefront renderer by replacing a few of the function calls with GPU kernels.
So when the actual GPU implementation is finished, this script can easily check if the wavefront implementation matches up with the reference recursive renderer.
We do render the images at a lower resolution so tests finish within a reasonable amount of time.
Note that this script requires `numpy` and `Pillow` to be installed locally.

Running `python3 benchmark.py` on my M1 Macbook Pro gives the following result:

```
Path Tracer Test Suite
==========================================================================
  Resolution : 480×360   spp=512
  Tolerance  : RMSE < 0.0100  (2.6 / 255)
  Scenes     : spheres, cube_cornell, room, showcase, bunny_diffuse, bunny_studio, dragon_glass

  Both renderers share the same per-pixel seed, so their images are
  expected to be essentially pixel-identical.  A nonzero RMSE indicates
  floating-point differences introduced by tonemapping, not path divergence.

  [spheres]
    recursive  ...    2.8 s    31.1 Mrays/s  → spheres.png
    wavefront  ...   11.9 s     7.4 Mrays/s  → spheres.png
    compare    RMSE=0.00003  max=0.00392  → PASS

  [cube_cornell]
    recursive  ...   13.8 s     6.4 Mrays/s  → cube_cornell.png
    wavefront  ...   37.7 s     2.3 Mrays/s  → cube_cornell.png
    compare    RMSE=0.00001  max=0.00392  → PASS

  [room]
    recursive  ...    5.0 s    17.6 Mrays/s  → room.png
    wavefront  ...   18.0 s     4.9 Mrays/s  → room.png
    compare    RMSE=0.00099  max=0.10588  → PASS

  [showcase]
    recursive  ...    4.9 s    18.0 Mrays/s  → showcase.png
    wavefront  ...   15.3 s     5.8 Mrays/s  → showcase.png
    compare    RMSE=0.00049  max=0.21569  → PASS

  [bunny_diffuse]
    recursive  ...    6.1 s    14.4 Mrays/s  → bunny_diffuse.png
    wavefront  ...   21.2 s     4.2 Mrays/s  → bunny_diffuse.png
    compare    RMSE=0.00002  max=0.00392  → PASS

  [bunny_studio]
    recursive  ...   13.6 s     6.5 Mrays/s  → bunny_studio.png
    wavefront  ...   37.9 s     2.3 Mrays/s  → bunny_studio.png
    compare    RMSE=0.00154  max=0.13333  → PASS

  [dragon_glass]
    recursive  ...   31.2 s     2.8 Mrays/s  → dragon_glass.png
    wavefront  ...   73.2 s     1.2 Mrays/s  → dragon_glass.png
    compare    RMSE=0.00707  max=0.19216  → PASS

==========================================================================
Summary
==========================================================================
Scene             Rec Mrays/s  WF Mrays/s  WF/Rec      RMSE  MaxDiff  Status
----------------------------------------------------------------------------
spheres                  31.1         7.4    0.24×   0.00003  0.00392  PASS
cube_cornell              6.4         2.3    0.36×   0.00001  0.00392  PASS
room                     17.6         4.9    0.28×   0.00099  0.10588  PASS
showcase                 18.0         5.8    0.32×   0.00049  0.21569  PASS
bunny_diffuse            14.4         4.2    0.29×   0.00002  0.00392  PASS
bunny_studio              6.5         2.3    0.35×   0.00154  0.13333  PASS
dragon_glass              2.8         1.2    0.43×   0.00707  0.19216  PASS

  Passed  : 7 / 7
  Total wall-clock time: 5.0 min
```

Thus, we see that the wavefront renderer and the reference recursive renderer agree down to a small noise level, which is expected since path tracers are effectively running Monte Carlo simulations.

## Parallelization

We also see that the wavefront renderer is much slower than the recursive reference approach.
The standard recursive implementation is easy to understand and modify.
However, in order to achieve better parallelism, a wavefront architecture is more amenable to the GPU architecture.
Instead of simulating rays through deep recursion, a wavefront path tracer splits up the recursive ray-tracing steps that produce rays and place the output into queues for later steps.
Specifically, a wavefront path tracer first generates rays ("kernel_generate"), then intersects all of the rays with the scene in parallel ("kernel_intersect"), checks for rays that completely miss the scene ("kernel_miss"), and then accumulates a color at every valid ray-scene intersection, potentially producing a new ray ("kernel_shade").
It then repeats the last three kernels until no more rays exist in the scene.

We see that there is a lot of inherent parallelism using this method, since each kernel operates on rays independently.
However, this algorithm is runs slower on the CPU than the naive recursive implementation mainly due to memory bandwidth constraints.
Every ray has to be written to/from a queue in RAM between the kernels, which is slow even with caching due to the large number of rays.

But since GPUs have high memory bandwidth and can utilize both memory coalescing and latency hiding, they are much more amenable to the wavefront formulation.
Each kernel maps well onto the GPU since each thread can operate on an independent rays, and warp divergence is even reduced since rays that close in proximity in the original image are likely to hit the same type of shapes and be shaded using the same material.
Furthermore, techniques like stream compaction make it so that queues in GPU memory can be much more efficient than queues in RAM.
Thus, it is expected that a GPU wavefront renderer will be able to achieve a vastly higher throughput than both the recursive and wavefront CPU implementations.
