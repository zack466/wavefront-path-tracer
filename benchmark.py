#!/usr/bin/env python3
"""
Path tracer benchmark.

Runs path_tracer_cpu and/or path_tracer_gpu at specified resolutions and
prints a performance table with Mrays/s and wall-clock time for each scene.

Usage examples
--------------
  # Full benchmark: GPU CUDA BVH at 4K, CPU recursive at 720p (default)
  python benchmark.py

  # GPU OptiX backend instead of CUDA BVH
  python benchmark.py --optix

  # Both GPU backends side-by-side
  python benchmark.py --both-gpu

  # CPU wavefront instead of recursive
  python benchmark.py --wavefront

  # GPU only
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

# ── Table helpers ──────────────────────────────────────────────────────────────

def fmt_mrays(val):
    return f"{val:>11.1f}" if val is not None else f"{'—':>11}"

def fmt_time(val):
    return f"{val:>7.1f}s" if val is not None else f"{'—':>8}"

def fmt_ratio(a, b):
    return f"{a/b:>7.0f}x" if (a is not None and b is not None) else f"{'—':>7}"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark path_tracer_cpu / path_tracer_gpu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--gpu-only",  action="store_true", help="Run GPU benchmarks only")
    ap.add_argument("--cpu-only",  action="store_true", help="Run CPU benchmarks only")
    ap.add_argument("--optix",     action="store_true",
                    help="Use OptiX RT-core backend for GPU (default: CUDA BVH)")
    ap.add_argument("--both-gpu",  action="store_true",
                    help="Run both GPU backends (CUDA BVH and OptiX) and show both columns")
    ap.add_argument("--wavefront", action="store_true",
                    help="Use CPU wavefront renderer instead of recursive")
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
    args = ap.parse_args()

    if args.optix and args.both_gpu:
        ap.error("--optix and --both-gpu are mutually exclusive")

    project_root = Path(__file__).parent
    build_dir    = project_root / args.build_dir
    scenes_dir   = project_root / "scenes"

    gpu_bin = build_dir / "path_tracer_gpu"
    cpu_bin = build_dir / "path_tracer_cpu"

    run_gpu = not args.cpu_only
    run_cpu = not args.gpu_only

    if run_gpu and not gpu_bin.exists():
        print(f"GPU binary not found: {gpu_bin}", file=sys.stderr)
        print("Build with ENABLE_CUDA=ON or pass --cpu-only", file=sys.stderr)
        sys.exit(1)
    if run_cpu and not cpu_bin.exists():
        print(f"CPU binary not found: {cpu_bin}", file=sys.stderr)
        sys.exit(1)

    scenes       = [s.strip() for s in args.scenes.split(",") if s.strip()]
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

    # Decide which GPU backends to run
    run_cuda  = run_gpu and (not args.optix)   # default or both-gpu
    run_optix = run_gpu and (args.optix or args.both_gpu)

    # ── Print header ──────────────────────────────────────────────────────────

    print()
    print("Path Tracer Benchmark")
    print("=" * 60)
    if run_cuda:
        print(f"  GPU CUDA BVH   : {gpu_w}×{gpu_h}")
    if run_optix:
        print(f"  GPU OptiX      : {gpu_w}×{gpu_h}")
    if run_cpu:
        print(f"  CPU resolution : {cpu_w}×{cpu_h}  ({cpu_label})")
    if spp_override:
        print(f"  SPP override   : {spp_override}")
    print(f"  Scenes         : {', '.join(scenes)}")
    print()

    # ── Collect results ───────────────────────────────────────────────────────
    # results[scene] = { "cuda": (secs, mrays), "optix": ..., "cpu": ... }

    results = {name: {} for name in scenes}
    total_start = time.monotonic()

    for name in scenes:
        scene_json = scenes_dir / f"{name}.json"

        if run_cuda:
            print(f"  CUDA  {name} @ {gpu_w}×{gpu_h} ... ", end="", flush=True)
            secs, mrays, _ = run_one(gpu_bin, scene_json, gpu_w, gpu_h, spp_override)
            if secs is not None:
                print(f"{secs:7.1f} s  {mrays:7.1f} Mrays/s")
                results[name]["cuda"] = (secs, mrays)
            else:
                print("FAILED")

        if run_optix:
            print(f"  OptiX {name} @ {gpu_w}×{gpu_h} ... ", end="", flush=True)
            secs, mrays, _ = run_one(gpu_bin, scene_json, gpu_w, gpu_h, spp_override,
                                     extra_args=["--optix"])
            if secs is not None:
                print(f"{secs:7.1f} s  {mrays:7.1f} Mrays/s")
                results[name]["optix"] = (secs, mrays)
            else:
                print("FAILED")

        if run_cpu:
            print(f"  CPU   {name} @ {cpu_w}×{cpu_h} ... ", end="", flush=True)
            secs, mrays, _ = run_one(cpu_bin, scene_json, cpu_w, cpu_h, spp_override,
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

    col = 14  # scene column width

    # Build column list dynamically based on what was run
    cols = []
    if run_cuda:
        cols.append(("CUDA Mrays/s", "CUDA time", "cuda"))
    if run_optix:
        cols.append(("OptiX Mrays/s", "OptiX time", "optix"))
    if run_cpu:
        cols.append((f"{cpu_label[:11]} Mrays/s", f"{cpu_label[:4]} time", "cpu"))

    # Header
    header = f"{'Scene':<{col}}"
    for mrays_label, time_label, _ in cols:
        header += f"  {mrays_label:>13}  {time_label:>9}"
    # Ratio column: last GPU vs CPU, or CUDA vs OptiX if both-gpu
    if run_cuda and run_optix and not run_cpu:
        header += f"  {'OptiX/CUDA':>10}"
    elif run_gpu and run_cpu:
        header += f"  {'GPU/CPU':>7}"
    print(header)
    print("-" * len(header))

    for name in scenes:
        r = results[name]
        row = f"{name:<{col}}"
        for _, _, key in cols:
            entry = r.get(key)
            row += f"  {fmt_mrays(entry[1] if entry else None)}  {fmt_time(entry[0] if entry else None)}"
        # Ratio
        if run_cuda and run_optix and not run_cpu:
            cuda_m  = r["cuda"][1]  if "cuda"  in r else None
            optix_m = r["optix"][1] if "optix" in r else None
            row += f"  {fmt_ratio(optix_m, cuda_m):>10}"
        elif run_gpu and run_cpu:
            gpu_key = "optix" if (run_optix and not run_cuda) else "cuda"
            gpu_m = r[gpu_key][1] if gpu_key in r else None
            cpu_m = r["cpu"][1]   if "cpu"   in r else None
            row += f"  {fmt_ratio(gpu_m, cpu_m)}"
        print(row)

    print()
    print(f"Total wall-clock time: {total_elapsed/60:.1f} min")
    print()

if __name__ == "__main__":
    main()
