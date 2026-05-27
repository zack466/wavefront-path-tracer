#!/usr/bin/env python3
"""
Path tracer benchmark / test suite (CPU build).

Benchmark mode (default)
------------------------
  Runs path_tracer_cpu with both the recursive and wavefront renderers and
  prints a side-by-side performance table (Mrays/s, wall-clock time).

  python benchmark.py                            # both renderers, 720p
  python benchmark.py --recursive-only           # recursive only
  python benchmark.py --wavefront-only           # wavefront only
  python benchmark.py --scenes spheres,room      # subset of scenes
  python benchmark.py --res 1920x1080            # custom resolution
  python benchmark.py --spp 64                   # override spp

Test suite mode (--test)
------------------------
  Runs both renderers for every scene, then compares the output PNG images
  pixel-by-pixel.  Both renderers use the same per-pixel seed formula, so
  their images should be essentially identical; the tolerance exists only to
  absorb float arithmetic differences after tonemapping.

  python benchmark.py --test                     # default settings
  python benchmark.py --test --test-res 640x360 --test-spp 64
  python benchmark.py --test --tolerance 0.04   # tighter threshold
  python benchmark.py --test --scenes spheres   # single scene

  Exit code: 0 if all scenes pass, 1 if any fail (suitable for CI).

General options
---------------
  --build-dir DIR   CMake build directory (default: build)
"""

import argparse
import json
import subprocess
import re
import sys
import time
from pathlib import Path

# ── Scene list ─────────────────────────────────────────────────────────────────

ALL_SCENES = [
    "spheres",
    "cube_cornell",
    "room",
    "showcase",
    "bunny_diffuse",
    "bunny_studio",
    "dragon_glass",
]

# test at a low resolution and less samples so it doesn't take forever to run
DEFAULT_BENCH_RES = (1920, 1080)
DEFAULT_TEST_RES  = (480, 360)
DEFAULT_TEST_SPP  = 512
DEFAULT_TOLERANCE = 0.01

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_output(text):
    """Extract (secs, mrays) from binary stdout, or (None, None)."""
    m = re.search(r"Render:\s+([\d.]+)\s+s\s+\(([\d.]+)\s+Mrays/s\)", text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def parse_res(s):
    """Parse 'WxH' string into (W, H)."""
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must be WxH, got: {s!r}")
    return int(parts[0]), int(parts[1])


def scene_output_filename(scene_json):
    """Read the 'render.output' field from a scene JSON file."""
    with open(scene_json) as f:
        d = json.load(f)
    return d.get("render", {}).get("output", Path(scene_json).stem + ".png")


def run_one(binary, scene_json, width, height, spp_override, extra_args=()):
    """
    Run the binary with the given arguments.
    Returns (secs, mrays, stdout_text) on success, or (None, None, combined_output)
    on failure.
    """
    cmd = [str(binary), str(scene_json),
           "--width",  str(width),
           "--height", str(height)]
    if spp_override:
        cmd += ["--spp", str(spp_override)]
    cmd += list(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)
        return None, None, result.stdout + result.stderr

    secs, mrays = parse_output(result.stdout)
    if secs is None:
        print("  WARNING: could not parse timing output", file=sys.stderr)
        print(result.stdout.strip(), file=sys.stderr)
        return None, None, result.stdout

    return secs, mrays, result.stdout


# ── Image comparison (requires numpy + Pillow) ─────────────────────────────────

def compare_images(path_a, path_b):
    """
    Load two PNG files and compute RMSE and max absolute difference in [0, 1]
    float space across all pixels and channels.

    Raises ImportError if numpy/Pillow are missing, FileNotFoundError if an
    image is absent, or ValueError on size mismatch.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise ImportError(
            "numpy and Pillow are required for --test image comparison.\n"
            "Install with:  pip install numpy Pillow"
        )

    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32) / 255.0

    if a.shape != b.shape:
        raise ValueError(f"Image size mismatch: {a.shape} vs {b.shape}")

    diff = a - b
    rmse     = float(np.sqrt(np.mean(diff ** 2)))
    max_diff = float(np.abs(diff).max())
    return rmse, max_diff


# ── Benchmark mode ─────────────────────────────────────────────────────────────

def run_benchmark(args, cpu_bin, scenes_dir, scenes):
    run_rec = not args.wavefront_only
    run_wf  = not args.recursive_only
    res_w, res_h = parse_res(args.res)

    print()
    print("Path Tracer Benchmark")
    print("=" * 66)
    if run_rec:
        print(f"  Recursive renderer  @ {res_w}×{res_h}")
    if run_wf:
        print(f"  Wavefront renderer  @ {res_w}×{res_h}")
    if args.spp:
        print(f"  SPP override : {args.spp}")
    print(f"  Scenes : {', '.join(scenes)}")
    print()

    results = {}
    total_start = time.monotonic()

    for name in scenes:
        scene_json = scenes_dir / f"{name}.json"
        results[name] = {}

        if run_rec:
            print(f"  recursive  {name:<16} ... ", end="", flush=True)
            secs, mrays, _ = run_one(cpu_bin, scene_json, res_w, res_h, args.spp)
            if secs is not None:
                print(f"{secs:7.1f} s  {mrays:6.1f} Mrays/s")
                results[name]["rec"] = (secs, mrays)
            else:
                print("FAILED")

        if run_wf:
            print(f"  wavefront  {name:<16} ... ", end="", flush=True)
            secs, mrays, _ = run_one(cpu_bin, scene_json, res_w, res_h, args.spp,
                                     extra_args=["--wavefront"])
            if secs is not None:
                print(f"{secs:7.1f} s  {mrays:6.1f} Mrays/s")
                results[name]["wf"] = (secs, mrays)
            else:
                print("FAILED")

    total_elapsed = time.monotonic() - total_start

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("Results")
    print("=" * 66)
    col = 16

    if run_rec and run_wf:
        hdr = (f"{'Scene':<{col}}  {'Rec Mrays/s':>11}  {'Rec time':>8}"
               f"  {'WF Mrays/s':>10}  {'WF time':>8}  {'WF/Rec':>6}")
        print(hdr)
        print("-" * len(hdr))
        for name in scenes:
            r   = results[name]
            rec = r.get("rec")
            wf  = r.get("wf")
            rm  = f"{rec[1]:>11.1f}" if rec else f"{'—':>11}"
            rt  = f"{rec[0]:>7.1f}s"  if rec else f"{'—':>8}"
            wm  = f"{wf[1]:>10.1f}"  if wf  else f"{'—':>10}"
            wt  = f"{wf[0]:>7.1f}s"  if wf  else f"{'—':>8}"
            # WF/Rec < 1 means wavefront has CPU overhead (queue compaction cost)
            ratio = f"{wf[1]/rec[1]:>6.2f}×" if (rec and wf) else f"{'—':>7}"
            print(f"{name:<{col}}  {rm}  {rt}  {wm}  {wt}  {ratio}")
        print()
        print("  WF/Rec ratio < 1.0 reflects CPU queue-compaction overhead that is")
        print("  absent on the GPU (warp-ballot compaction is essentially free).")

    elif run_rec:
        hdr = f"{'Scene':<{col}}  {'Rec Mrays/s':>11}  {'Rec time':>8}"
        print(hdr)
        print("-" * len(hdr))
        for name in scenes:
            rec = results[name].get("rec")
            rm  = f"{rec[1]:>11.1f}" if rec else f"{'—':>11}"
            rt  = f"{rec[0]:>7.1f}s"  if rec else f"{'—':>8}"
            print(f"{name:<{col}}  {rm}  {rt}")

    else:
        hdr = f"{'Scene':<{col}}  {'WF Mrays/s':>10}  {'WF time':>8}"
        print(hdr)
        print("-" * len(hdr))
        for name in scenes:
            wf = results[name].get("wf")
            wm = f"{wf[1]:>10.1f}" if wf else f"{'—':>10}"
            wt = f"{wf[0]:>7.1f}s"  if wf else f"{'—':>8}"
            print(f"{name:<{col}}  {wm}  {wt}")

    print()
    print(f"Total wall-clock time: {total_elapsed / 60:.1f} min")
    print()


# ── Test suite mode ────────────────────────────────────────────────────────────

def run_test(args, cpu_bin, scenes_dir, scenes, project_root):
    test_w, test_h = parse_res(args.test_res)
    test_spp  = args.test_spp
    tolerance = args.tolerance

    print()
    print("Path Tracer Test Suite")
    print("=" * 74)
    print(f"  Resolution : {test_w}×{test_h}   spp={test_spp}")
    print(f"  Tolerance  : RMSE < {tolerance:.4f}  ({tolerance * 255:.1f} / 255)")
    print(f"  Scenes     : {', '.join(scenes)}")
    print()
    print("  Both renderers share the same per-pixel seed, so their images are")
    print("  expected to be essentially pixel-identical.  A nonzero RMSE indicates")
    print("  floating-point differences introduced by tonemapping, not path divergence.")
    print()

    results   = {}
    n_pass    = 0
    n_fail    = 0
    n_unknown = 0
    total_start = time.monotonic()

    for name in scenes:
        scene_json   = scenes_dir / f"{name}.json"
        out_filename = scene_output_filename(scene_json)
        rec_png = project_root / "outputs" / "cpu_recursive" / out_filename
        wf_png  = project_root / "outputs" / "cpu_wavefront" / out_filename

        results[name] = {}

        print(f"  [{name}]")

        # Recursive
        print(f"    recursive  ... ", end="", flush=True)
        secs, mrays, _ = run_one(cpu_bin, scene_json, test_w, test_h, test_spp)
        if secs is None:
            print("FAILED")
            results[name]["pass"] = False
            n_fail += 1
            continue
        print(f"{secs:6.1f} s  {mrays:6.1f} Mrays/s  → {rec_png.name}")
        results[name]["rec"] = (secs, mrays)

        # Wavefront
        print(f"    wavefront  ... ", end="", flush=True)
        secs, mrays, _ = run_one(cpu_bin, scene_json, test_w, test_h, test_spp,
                                 extra_args=["--wavefront"])
        if secs is None:
            print("FAILED")
            results[name]["pass"] = False
            n_fail += 1
            continue
        print(f"{secs:6.1f} s  {mrays:6.1f} Mrays/s  → {wf_png.name}")
        results[name]["wf"] = (secs, mrays)

        # Image comparison
        try:
            rmse, max_diff = compare_images(rec_png, wf_png)
            ok = rmse < tolerance
            results[name]["rmse"]     = rmse
            results[name]["max_diff"] = max_diff
            results[name]["pass"]     = ok
            status = "PASS" if ok else f"FAIL  (RMSE {rmse:.5f} > {tolerance:.4f})"
            print(f"    compare    RMSE={rmse:.5f}  max={max_diff:.5f}  → {status}")
            if ok:
                n_pass += 1
            else:
                n_fail += 1
        except FileNotFoundError as e:
            print(f"    compare    MISSING output: {e}", file=sys.stderr)
            results[name]["pass"] = None
            n_unknown += 1
        except (ImportError, ValueError) as e:
            print(f"    compare    ERROR: {e}", file=sys.stderr)
            results[name]["pass"] = None
            n_unknown += 1

        print()

    total_elapsed = time.monotonic() - total_start

    # ── Summary table ──────────────────────────────────────────────────────────
    print("=" * 74)
    print("Summary")
    print("=" * 74)
    col = 16
    hdr = (f"{'Scene':<{col}}  {'Rec Mrays/s':>11}  {'WF Mrays/s':>10}"
           f"  {'WF/Rec':>6}  {'RMSE':>8}  {'MaxDiff':>7}  Status")
    print(hdr)
    print("-" * len(hdr))

    for name in scenes:
        r   = results.get(name, {})
        rec = r.get("rec")
        wf  = r.get("wf")
        rm  = f"{rec[1]:>11.1f}" if rec else f"{'—':>11}"
        wm  = f"{wf[1]:>10.1f}"  if wf  else f"{'—':>10}"
        ratio_s   = f"{wf[1]/rec[1]:>6.2f}×" if (rec and wf) else f"{'—':>7}"
        rmse_s    = f"{r['rmse']:>8.5f}"      if "rmse"     in r else f"{'—':>8}"
        maxd_s    = f"{r['max_diff']:>7.5f}"  if "max_diff" in r else f"{'—':>7}"
        pass_val  = r.get("pass")
        if pass_val is True:
            status = "PASS"
        elif pass_val is False:
            status = "FAIL"
        else:
            status = "?"
        print(f"{name:<{col}}  {rm}  {wm}  {ratio_s}  {rmse_s}  {maxd_s}  {status}")

    total_scenes = n_pass + n_fail + n_unknown
    print()
    print(f"  Passed  : {n_pass} / {total_scenes}")
    if n_fail:
        print(f"  Failed  : {n_fail}")
    if n_unknown:
        print(f"  Unknown : {n_unknown}  (output missing or comparison error)")
    print(f"  Total wall-clock time: {total_elapsed / 60:.1f} min")
    print()

    return 0 if (n_fail == 0 and n_unknown == 0) else 1


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark / test path_tracer_cpu (CPU build)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common
    ap.add_argument("--scenes",    default=",".join(ALL_SCENES),
                    help="Comma-separated scene names (default: all)")
    ap.add_argument("--build-dir", default="build",
                    help="CMake build directory (default: build)")
    ap.add_argument("--spp",       type=int, default=0,
                    help="Override spp for benchmark mode (0 = scene default)")

    # Benchmark-specific
    bench = ap.add_argument_group("benchmark options")
    bench.add_argument("--res", default="x".join(map(str, DEFAULT_BENCH_RES)),
                       help=f"Resolution WxH (default: {DEFAULT_BENCH_RES[0]}x{DEFAULT_BENCH_RES[1]})")
    mode = bench.add_mutually_exclusive_group()
    mode.add_argument("--recursive-only", action="store_true",
                      help="Run only the recursive renderer")
    mode.add_argument("--wavefront-only", action="store_true",
                      help="Run only the wavefront renderer")

    # Test-specific
    test_grp = ap.add_argument_group("test suite options (--test)")
    test_grp.add_argument("--test", action="store_true",
                          help="Run test suite: both renderers + image comparison")
    test_grp.add_argument("--test-res",
                          default="x".join(map(str, DEFAULT_TEST_RES)),
                          help=f"Test resolution WxH (default: {DEFAULT_TEST_RES[0]}x{DEFAULT_TEST_RES[1]})")
    test_grp.add_argument("--test-spp", type=int, default=DEFAULT_TEST_SPP,
                          help=f"Test spp (default: {DEFAULT_TEST_SPP})")
    test_grp.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                          help=f"RMSE pass threshold in [0,1] (default: {DEFAULT_TOLERANCE})")

    args = ap.parse_args()

    project_root = Path(__file__).parent
    build_dir    = project_root / args.build_dir
    scenes_dir   = project_root / "scenes"
    cpu_bin      = build_dir / "path_tracer_cpu"

    if not cpu_bin.exists():
        print(f"CPU binary not found: {cpu_bin}", file=sys.stderr)
        print("Build first:  cmake -B build && cmake --build build -j", file=sys.stderr)
        sys.exit(1)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    for name in scenes:
        p = scenes_dir / f"{name}.json"
        if not p.exists():
            print(f"Scene not found: {p}", file=sys.stderr)
            sys.exit(1)

    if args.test:
        sys.exit(run_test(args, cpu_bin, scenes_dir, scenes, project_root))
    else:
        run_benchmark(args, cpu_bin, scenes_dir, scenes)


if __name__ == "__main__":
    main()
