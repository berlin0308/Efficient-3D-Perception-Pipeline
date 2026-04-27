"""
cuda_pp_metrics.py — Run CUDA-PointPillars (TensorRT) and collect latency + energy metrics.

Usage (from repo root):
    python cuda_pp_metrics.py \\
        --build-dir CUDA-PointPillars/build \\
        --data-dir  CUDA-PointPillars/data \\
        --model-dir CUDA-PointPillars/model \\
        --precision fp32 \\
        --warmup 5 \\
        --steps  10 \\
        --output-csv modal_outputs/cuda_pp/M5_FP32.csv

    python cuda_pp_metrics.py --precision fp16 --output-csv modal_outputs/cuda_pp/M5_FP16.csv

The binary (./demo) must already be built in --build-dir.
For FP16, rebuild with: cmake .. -DCMAKE_CXX_FLAGS="-DUSE_FP16" && make -j$(nproc)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

# ---------------------------------------------------------------------------
# RAPL helpers
# ---------------------------------------------------------------------------

_RAPL_PKG_PATH = '/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj'
_RAPL_MAX_PATH = '/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj'


def _rapl_available() -> bool:
    return os.path.isfile(_RAPL_PKG_PATH) and os.access(_RAPL_PKG_PATH, os.R_OK)


def _rapl_read_uj() -> int:
    with open(_RAPL_PKG_PATH) as f:
        return int(f.read().strip())


def _rapl_max_uj() -> int:
    try:
        with open(_RAPL_MAX_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return 2 ** 32


def _rapl_delta_j(start_uj: int, end_uj: int) -> float:
    """Handle counter wraparound and return delta in Joules."""
    delta = end_uj - start_uj
    if delta < 0:
        delta += _rapl_max_uj()
    return delta / 1e6


# ---------------------------------------------------------------------------
# NVML energy helpers
# ---------------------------------------------------------------------------

def _nvml_init(cuda_id: int = 0):
    if not _HAS_NVML:
        return None, None
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_id)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    return handle, name


def _sample_power(handle) -> float:
    """Return current GPU power in Watts."""
    if handle is None:
        return 0.0
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W


def _measure_energy(handle, fn, sample_interval: float = 0.05):
    """Run fn() while sampling GPU power; return (result, mean_power_W, total_J)."""
    if handle is None:
        result = fn()
        return result, 0.0, 0.0

    samples: list[float] = []
    stop_flag = [False]

    import threading

    def _sampler():
        while not stop_flag[0]:
            samples.append(_sample_power(handle))
            time.sleep(sample_interval)

    t = threading.Thread(target=_sampler, daemon=True)
    t0 = time.perf_counter()
    t.start()
    result = fn()
    stop_flag[0] = True
    t1 = time.perf_counter()
    t.join(timeout=2.0)

    wall = t1 - t0
    avg_w = mean(samples) if samples else 0.0
    total_j = avg_w * wall
    return result, avg_w, total_j


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_binary(
    src_dir: Path,
    build_dir: Path,
    precision: str,
    cuda_toolkit: str = "/usr/local/cuda-12.1",
) -> Path:
    """cmake + make inside build_dir. Returns path to ./demo binary."""
    build_dir.mkdir(parents=True, exist_ok=True)

    extra_flags = "-DUSE_FP16" if precision == "fp16" else ""
    cmake_cmd = [
        "cmake", str(src_dir),
        f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_toolkit}",
    ]
    if extra_flags:
        cmake_cmd += [f"-DCMAKE_CXX_FLAGS={extra_flags}"]

    print(f"[build] cmake {' '.join(cmake_cmd)}", flush=True)
    subprocess.run(cmake_cmd, cwd=str(build_dir), check=True)

    make_cmd = ["make", f"-j{os.cpu_count() or 4}"]
    print(f"[build] {' '.join(make_cmd)}", flush=True)
    subprocess.run(make_cmd, cwd=str(build_dir), check=True)

    demo = build_dir / "demo"
    if not demo.exists():
        raise RuntimeError(f"Build succeeded but ./demo not found in {build_dir}")
    print(f"[build] binary ready: {demo}", flush=True)
    return demo


# ---------------------------------------------------------------------------
# Run + parse
# ---------------------------------------------------------------------------

_TIME_RE           = re.compile(r"TIME:\s*pointpillar:\s*([\d.]+)\s*ms")
_TIME_READ_RE      = re.compile(r"TIME:\s*read_points:\s*([\d.]+)\s*ms")
_TIME_VOXELS_RE    = re.compile(r"TIME:\s*generateVoxels:\s*([\d.]+)\s*ms")
_TIME_FEATURES_RE  = re.compile(r"TIME:\s*generateFeatures:\s*([\d.]+)\s*ms")
_TIME_INFER_RE     = re.compile(r"TIME:\s*doinfer:\s*([\d.]+)\s*ms")
_TIME_POSTPROC_RE  = re.compile(r"TIME:\s*doPostprocessCuda:\s*([\d.]+)\s*ms")


# def _run_once(demo: Path) -> list[float]:
#     """Original: run ./demo once as a new subprocess (cold-start TRT each time)."""
#     result = subprocess.run(
#         [str(demo)],
#         cwd=str(demo.parent),
#         capture_output=True,
#         text=True,
#     )
#     if result.returncode != 0:
#         print(result.stderr, file=sys.stderr)
#         raise RuntimeError(f"./demo exited with code {result.returncode}")
#     times = [float(m.group(1)) for m in _TIME_RE.finditer(result.stdout)]
#     return times

def _run_persistent(
    demo: Path,
    data_dir: Path,
    warmup: int,
    steps: int,
    save_dir: Path | None = None,
    save_preds: bool = False,
) -> dict[str, list[float]]:
    """Single subprocess: engine loads once, warmup + measurement inside binary.
    Matches teammate's persistent Python process in profile_suite.py."""
    cmd = [
        str(demo),
        "--data-dir", str(data_dir),
        "--warmup",   str(warmup),
        "--repeat",   str(steps),
    ]
    if save_preds and save_dir is not None:
        cmd += ["--save-dir", str(save_dir), "--save-preds"]

    result = subprocess.run(cmd, cwd=str(demo.parent), capture_output=True, text=True)
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"./demo exited with code {result.returncode}")
    return {
        "full_frame":        [float(m.group(1)) for m in _TIME_RE.finditer(result.stdout)],
        "read_points":       [float(m.group(1)) for m in _TIME_READ_RE.finditer(result.stdout)],
        "generate_voxels":   [float(m.group(1)) for m in _TIME_VOXELS_RE.finditer(result.stdout)],
        "generate_features": [float(m.group(1)) for m in _TIME_FEATURES_RE.finditer(result.stdout)],
        "doinfer":           [float(m.group(1)) for m in _TIME_INFER_RE.finditer(result.stdout)],
        "postprocess":       [float(m.group(1)) for m in _TIME_POSTPROC_RE.finditer(result.stdout)],
    }


def _clear_trt_cache(model_dir: Path) -> None:
    """Remove TRT engine cache so a new precision build is applied."""
    cache = model_dir / "pointpillar.onnx.cache"
    if cache.exists():
        cache.unlink()
        print(f"[run] removed TRT cache: {cache}", flush=True)


def run_benchmark(
    demo: Path,
    data_dir: Path,
    model_dir: Path,
    warmup: int,
    steps: int,
    handle,
    save_preds: bool = False,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Single subprocess: TRT engine loads once, warmup + measurement inside binary.
    Matches teammate's profile_suite.py persistent process — mean is now comparable.
    """
    # Clear cache so the binary rebuilds TRT engine with current precision flags,
    # then pre-run once (warmup=1 step=1) to cache the engine BEFORE energy measurement.
    _clear_trt_cache(model_dir)
    print("[run] pre-run: building TRT engine cache (not measured for energy)...", flush=True)
    _run_persistent(demo, data_dir, warmup=1, steps=1)

    print(f"[run] energy+latency run: data_dir={data_dir}  warmup={warmup}  steps={steps}", flush=True)
    use_rapl = _rapl_available()
    if use_rapl:
        print("[run] RAPL available — measuring CPU package energy", flush=True)
    else:
        print("[run] RAPL not available — CPU energy will not be measured", flush=True)

    stage_times: dict[str, list[float]] = {}
    rapl_start_uj = _rapl_read_uj() if use_rapl else 0

    def _measured_runs():
        result = _run_persistent(demo, data_dir, warmup, steps,
                                 save_dir=save_dir, save_preds=save_preds)
        for k, v in result.items():
            stage_times.setdefault(k, []).extend(v)

    _, avg_power_w, total_j = _measure_energy(handle, _measured_runs)

    rapl_end_uj = _rapl_read_uj() if use_rapl else 0
    cpu_energy_j = _rapl_delta_j(rapl_start_uj, rapl_end_uj) if use_rapl else 0.0
    wall_s_rapl  = total_j / avg_power_w if avg_power_w > 0 else 0.0
    cpu_mean_power_w = cpu_energy_j / wall_s_rapl if wall_s_rapl > 0 else 0.0

    all_times = stage_times.get("full_frame", [])
    n = len(all_times)
    if n == 0:
        raise RuntimeError("No timing lines found in ./demo output")

    sorted_t = sorted(all_times)
    p50 = sorted_t[int(n * 0.50)]
    p95 = sorted_t[int(n * 0.95)]
    p99 = sorted_t[min(int(n * 0.99), n - 1)]
    avg = mean(all_times)
    wall_s = total_j / avg_power_w if avg_power_w > 0 else 0.0
    throughput = n / wall_s if wall_s > 0 else 0.0

    def _stage_mean(key: str) -> float:
        vals = stage_times.get(key, [])
        return round(mean(vals), 4) if vals else 0.0

    return {
        "full_frame_mean_ms":         round(avg, 4),
        "full_frame_p50_ms":          round(p50, 4),
        "full_frame_p95_ms":          round(p95, 4),
        "full_frame_p99_ms":          round(p99, 4),
        "throughput_sps":             round(throughput, 4),
        "energy_mean_power_W":        round(avg_power_w, 4),
        "energy_total_J":             round(total_j, 4),
        "cpu_energy_j":               round(cpu_energy_j, 4),
        "cpu_mean_power_w":           round(cpu_mean_power_w, 4),
        "rapl_measured":              use_rapl,
        "n_frames":                   n,
        "stage_read_points_ms":       _stage_mean("read_points"),
        "stage_generate_voxels_ms":   _stage_mean("generate_voxels"),
        "stage_generate_features_ms": _stage_mean("generate_features"),
        "stage_doinfer_ms":           _stage_mean("doinfer"),
        "stage_postprocess_ms":       _stage_mean("postprocess"),
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(row: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[csv] written → {output_csv}", flush=True)


# ---------------------------------------------------------------------------
# GPU name helper (without NVML)
# ---------------------------------------------------------------------------

def _gpu_name_from_smi() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        return out
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CUDA-PointPillars TRT benchmark")
    p.add_argument("--build-dir",  default="CUDA-PointPillars/build")
    p.add_argument("--src-dir",    default="CUDA-PointPillars",
                   help="Source root (contains CMakeLists.txt)")
    p.add_argument("--data-dir",   default="CUDA-PointPillars/data")
    p.add_argument("--model-dir",  default="CUDA-PointPillars/model")
    p.add_argument("--precision",  choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--warmup",     type=int, default=5)
    p.add_argument("--steps",      type=int, default=10,
                   help="Number of measured ./demo runs (each runs 10 frames)")
    p.add_argument("--cuda-id",    type=int, default=0)
    p.add_argument("--output-csv", default="modal_outputs/cuda_pp/runs.csv")
    p.add_argument("--rebuild",    action="store_true",
                   help="Force cmake+make even if binary already exists")
    p.add_argument("--cuda-toolkit", default="/usr/local/cuda-12.1")
    p.add_argument("--variant-name", default=None,
                   help="Override variant name (default: M5_FP32 / M5_FP16)")
    p.add_argument("--save-preds",  action="store_true",
                   help="Save bounding box predictions (for accuracy eval)")
    p.add_argument("--save-dir",    default=None,
                   help="Directory to save predictions (used with --save-preds)")
    return p.parse_args()


def main():
    args = parse_args()

    build_dir = Path(args.build_dir).resolve()
    src_dir   = Path(args.src_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_csv = Path(args.output_csv).resolve()

    precision_label = args.precision.upper()
    variant = args.variant_name or f"M5_{precision_label}"

    # --- Build ---
    demo = build_dir / "demo"
    if args.rebuild or not demo.exists():
        demo = build_binary(src_dir, build_dir, args.precision, args.cuda_toolkit)
    else:
        print(f"[build] reusing existing binary: {demo}", flush=True)

    # --- NVML ---
    handle, gpu_name = _nvml_init(args.cuda_id)
    if gpu_name is None:
        gpu_name = _gpu_name_from_smi()
    print(f"[info] GPU: {gpu_name}  precision: {precision_label}", flush=True)

    # --- Benchmark ---
    data_dir = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir).resolve() if args.save_dir else None
    if args.save_preds and save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    stats = run_benchmark(
        demo, data_dir, model_dir, args.warmup, args.steps, handle,
        save_preds=args.save_preds, save_dir=save_dir,
    )

    # --- Compose row (mirrors key columns from runs.csv) ---
    row: dict[str, Any] = {
        "run_id":            uuid.uuid4().hex[:12],
        "timestamp_iso":     dt.datetime.utcnow().isoformat() + "+00:00",
        "variant_name":      variant,
        "experiment_cell_id": variant,
        "model_variant":     "M5",
        "precision_mode":    precision_label,
        "experiment_status": "measured",
        "model_name":        "CUDA-PointPillars-TRT",
        "gpu_name":          gpu_name,
        "warmup_steps":      args.warmup,
        "measured_steps":    args.steps,
        "trt_precision":     args.precision,
        # latency — aligned to runs.csv column names used by plot scripts
        # prof_forward_mean_ms = TRT infer only (matches M0-M4 forward semantics)
        "prof_read_points_mean_ms":      stats["stage_read_points_ms"],
        "prof_forward_mean_ms":          stats["stage_doinfer_ms"],
        "prof_pre_processing_mean_ms":   round(stats["stage_generate_voxels_ms"] + stats["stage_generate_features_ms"], 4),
        "prof_h2d_mean_ms":              0.0,
        "prof_postprocess_mean_ms":      stats["stage_postprocess_ms"],
        # full frame (all stages) — for Pareto / reporting
        "prof_full_frame_mean_ms":       stats["full_frame_mean_ms"],
        "prof_full_frame_p50_ms":        stats["full_frame_p50_ms"],
        "prof_full_frame_p95_ms":        stats["full_frame_p95_ms"],
        "prof_full_frame_p99_ms":        stats["full_frame_p99_ms"],
        "prof_throughput_sps":           stats["throughput_sps"],
        # energy — GPU (NVML) + CPU (RAPL package)
        "energy_gpu_mean_power_W":       stats["energy_mean_power_W"],
        "energy_gpu_total_J":            stats["energy_total_J"],
        "energy_cpu_mean_power_W":       stats["cpu_mean_power_w"],
        "energy_cpu_total_J":            stats["cpu_energy_j"],
        "energy_rapl_measured":          stats["rapl_measured"],
        "measured_steps":                stats["n_frames"],
        "n_measured_frames":             stats["n_frames"],
    }

    write_csv(row, output_csv)

    print("\n[done] results:", flush=True)
    for k, v in row.items():
        print(f"  {k}: {v}", flush=True)

    if handle is not None:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
