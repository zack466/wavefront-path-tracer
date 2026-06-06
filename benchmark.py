#!/usr/bin/env python3
"""
Path tracer benchmark.

Runs path_tracer_cpu and/or path_tracer_gpu at specified resolutions and
prints a performance table with Mrays/s and wall-clock time for each scene.

Usage examples
--------------
  # Full benchmark: GPU at 4K, CPU at 720p (default)
  python benchmark.py

  # GPU only (e.g. on a laptop without a discrete GPU flag for CPU)
  python benchmark.py --gpu-only

  # CPU only
  python benchmark.py --cpu-only

  # Specific scenes
  python benchmark.py --scenes spheres,dragon_glass

  # Override resolution
  python benchmark.py --gpu-res 1920x1080 --cpu-res 960x540

  # Override spp (useful for quick sanity checks)
  python benchmark.py --spp 64

  # Change build directory
  python benchmark.py --build-dir /path/to/build
"""

import argparse
import subprocess
import re
import sys
import os
import time
from pathlib import Path

# ── Default settings ──────────────────────────────────────────────────────────

ALL_SCENES = [
    "spheres",
    "cornell_box",
    "cube_cornell",
    "room",
    "materials",
    "mesh_test",
    "bunny_studio",
    "dragon_glass",
]

DEFAULT_GPU_RES = (3840, 2160)   # 4K — amortises per-kernel-launch overhead
DEFAULT_CPU_RES = (1280, 720)    # 720p — keeps CPU runs to a few minutes each

# ── Output parsing ─────────────────────────────────────────────────────────────

def parse_output(text):
    """Extract render time (s) and Mrays/s from binary stdout."""
    m = re.search(r"Render:\s+([\d.]+)\s+s\s+\(([\d.]+)\s+Mrays/s\)", text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))

def parse_res_str(s):
    """Parse 'WxH' string into (W, H) tuple."""
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must be WxH, got: {s!r}")
    return int(parts[0]), int(parts[1])

# ── Single run ─────────────────────────────────────────────────────────────────

def run_one(binary, scene_json, width, height, spp_override, extra_args=()):
    """Run the binary and return (wall_secs, mrays_s, stdout_text)."""
    cmd = [str(binary), str(scene_json),
           "--width",  str(width),
           "--height", str(height)]
    if spp_override:
        cmd += ["--spp", str(spp_override)]
    cmd += list(extra_args)

    wall_start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall_end   = time.monotonic()

    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)
        return None, None, result.stdout + result.stderr

    secs, mrays = parse_output(result.stdout)
    if secs is None:
        print(f"  WARNING: could not parse output", file=sys.stderr)
        print(result.stdout.strip(), file=sys.stderr)
        return None, None, result.stdout

    return secs, mrays, result.stdout

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark path_tracer_cpu / path_tracer_gpu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--gpu-only",  action="store_true", help="Run GPU benchmarks only")
    ap.add_argument("--cpu-only",  action="store_true", help="Run CPU benchmarks only")
    ap.add_argument("--scenes",    default=",".join(ALL_SCENES),
                    help=f"Comma-separated scene names (default: {','.join(ALL_SCENES)})")
    ap.add_argument("--gpu-res",   default="x".join(map(str, DEFAULT_GPU_RES)),
                    help=f"GPU resolution WxH (default: {DEFAULT_GPU_RES[0]}x{DEFAULT_GPU_RES[1]})")
    ap.add_argument("--cpu-res",   default="x".join(map(str, DEFAULT_CPU_RES)),
                    help=f"CPU resolution WxH (default: {DEFAULT_CPU_RES[0]}x{DEFAULT_CPU_RES[1]})")
    ap.add_argument("--spp",       type=int, default=0,
                    help="Override spp for all runs (0 = use scene default)")
    ap.add_argument("--build-dir", default="build",
                    help="Path to CMake build directory (default: build)")
    ap.add_argument("--wavefront", action="store_true",
                    help="Use CPU wavefront renderer instead of recursive")
    args = ap.parse_args()

    project_root = Path(__file__).parent
    build_dir    = project_root / args.build_dir
    scenes_dir   = project_root / "scenes"

    gpu_bin = build_dir / "path_tracer_gpu"
    cpu_bin = build_dir / "path_tracer_cpu"

    run_gpu = not args.cpu_only
    run_cpu = not args.gpu_only

    gpu_extra = []

    if run_gpu and not gpu_bin.exists():
        print(f"GPU binary not found: {gpu_bin}", file=sys.stderr)
        print("Build with ENABLE_CUDA=ON or pass --cpu-only", file=sys.stderr)
        sys.exit(1)
    if run_cpu and not cpu_bin.exists():
        print(f"CPU binary not found: {cpu_bin}", file=sys.stderr)
        sys.exit(1)

    scenes    = [s.strip() for s in args.scenes.split(",") if s.strip()]
    gpu_w, gpu_h = parse_res_str(args.gpu_res)
    cpu_w, cpu_h = parse_res_str(args.cpu_res)
    spp_override = args.spp or 0

    # Check all scene files exist before starting
    for name in scenes:
        p = scenes_dir / f"{name}.json"
        if not p.exists():
            print(f"Scene not found: {p}", file=sys.stderr)
            sys.exit(1)

    cpu_extra = ["--wavefront"] if args.wavefront else []
    cpu_label = "CPU wavefront" if args.wavefront else "CPU recursive"

    # ── Print header ──────────────────────────────────────────────────────────

    print()
    print("Path Tracer Benchmark")
    print("=" * 60)
    if run_gpu:
        print(f"  GPU resolution : {gpu_w}×{gpu_h}  (OptiX RT cores)")
    if run_cpu:
        print(f"  CPU resolution : {cpu_w}×{cpu_h}  ({cpu_label})")
    if spp_override:
        print(f"  SPP override   : {spp_override}")
    print(f"  Scenes         : {', '.join(scenes)}")
    print()

    # ── Collect results ───────────────────────────────────────────────────────

    results = {}   # scene -> {gpu: (secs, mrays), cpu: (secs, mrays)}

    total_start = time.monotonic()

    for name in scenes:
        scene_json = scenes_dir / f"{name}.json"
        results[name] = {}

        if run_gpu:
            print(f"  GPU  {name} @ {gpu_w}×{gpu_h} ... ", end="", flush=True)
            secs, mrays, out = run_one(gpu_bin, scene_json, gpu_w, gpu_h, spp_override,
                                       extra_args=gpu_extra)
            if secs is not None:
                print(f"{secs:7.1f} s  {mrays:7.1f} Mrays/s")
                results[name]["gpu"] = (secs, mrays)
            else:
                print("FAILED")

        if run_cpu:
            print(f"  CPU  {name} @ {cpu_w}×{cpu_h} ... ", end="", flush=True)
            secs, mrays, out = run_one(cpu_bin, scene_json, cpu_w, cpu_h, spp_override,
                                       extra_args=cpu_extra)
            if secs is not None:
                print(f"{secs:7.1f} s  {mrays:7.1f} Mrays/s")
                results[name]["cpu"] = (secs, mrays)
            else:
                print("FAILED")

    total_elapsed = time.monotonic() - total_start

    # ── Print summary table ───────────────────────────────────────────────────

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()

    # Header
    col_scene = 14
    if run_gpu and run_cpu:
        header = (f"{'Scene':<{col_scene}}  "
                  f"{'GPU Mrays/s':>11}  {'GPU time':>8}  "
                  f"{'CPU Mrays/s':>11}  {'CPU time':>8}  "
                  f"{'GPU/CPU':>7}")
        sep = "-" * len(header)
        print(header)
        print(sep)
        for name in scenes:
            r = results[name]
            gpu = r.get("gpu")
            cpu = r.get("cpu")
            gpu_m = f"{gpu[1]:>11.1f}" if gpu else f"{'—':>11}"
            gpu_t = f"{gpu[0]:>7.1f}s"  if gpu else f"{'—':>8}"
            cpu_m = f"{cpu[1]:>11.1f}" if cpu else f"{'—':>11}"
            cpu_t = f"{cpu[0]:>7.1f}s"  if cpu else f"{'—':>8}"
            ratio = f"{gpu[1]/cpu[1]:>7.0f}x" if (gpu and cpu) else f"{'—':>7}"
            print(f"{name:<{col_scene}}  {gpu_m}  {gpu_t}  {cpu_m}  {cpu_t}  {ratio}")
    elif run_gpu:
        header = f"{'Scene':<{col_scene}}  {'GPU Mrays/s':>11}  {'GPU time':>8}"
        print(header)
        print("-" * len(header))
        for name in scenes:
            gpu = results[name].get("gpu")
            gpu_m = f"{gpu[1]:>11.1f}" if gpu else f"{'—':>11}"
            gpu_t = f"{gpu[0]:>7.1f}s"  if gpu else f"{'—':>8}"
            print(f"{name:<{col_scene}}  {gpu_m}  {gpu_t}")
    else:
        header = f"{'Scene':<{col_scene}}  {'CPU Mrays/s':>11}  {'CPU time':>8}"
        print(header)
        print("-" * len(header))
        for name in scenes:
            cpu = results[name].get("cpu")
            cpu_m = f"{cpu[1]:>11.1f}" if cpu else f"{'—':>11}"
            cpu_t = f"{cpu[0]:>7.1f}s"  if cpu else f"{'—':>8}"
            print(f"{name:<{col_scene}}  {cpu_m}  {cpu_t}")

    print()
    print(f"Total wall-clock time: {total_elapsed/60:.1f} min")
    print()

if __name__ == "__main__":
    main()
