#!/usr/bin/env python3
"""
ncu_intensity_sweep.py — Measure DRAM arithmetic intensity for all 15 variants.

Runs ncu on each variant (14 OpenPCDet + M5_FP32 CUDA-PP TRT), sums ld+st DRAM
bytes across all GPU kernels, computes intensity = THEORY_FLOPS / dram_bytes.

Output:
  - ncu_intensity.csv: measured intensity for all 15 variants

Usage (must run with sudo for ncu perf counters):
    sudo -E LD_PRELOAD=/home/chao/miniconda3/envs/mlsys/lib/libstdc++.so.6 \\
        /home/chao/miniconda3/envs/mlsys/bin/python ncu_intensity_sweep.py

    # Or skip specific variants:
    sudo -E ... python ncu_intensity_sweep.py --skip M5_FP32
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent
TOOLS_DIR   = REPO_ROOT / "OpenPCDet" / "tools"
CFG_FILE    = "cfgs/kitti_models/pointpillar.yaml"
CKPT_FILE   = "ckpt/pointpillar_7728.pth"

# Theoretical forward FLOPs (same as plot_roofline_forward.py)
# PillarVFE + scatter + BEV backbone + anchor head
THEORY_FLOPS = 1.5e9 + 0.05e9 + 18.0e9 + 1.2e9  # ~20.75 GFLOPs

NCU_BIN      = "/usr/local/cuda-12.1/bin/ncu"
PYTHON_BIN   = "/home/chao/miniconda3/envs/mls/bin/python"
LD_PRELOAD   = "/home/chao/miniconda3/envs/mls/lib/libstdc++.so.6"

# ncu DRAM metrics
METRIC_LD = "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
METRIC_ST = "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"

# ---------------------------------------------------------------------------
# Variant definitions — all 15 measured independently
# ---------------------------------------------------------------------------

ALL_VARIANTS = [
    # M0: baseline
    {"id": "M0_FP32",             "type": "openpcdet", "extra_flags": []},
    {"id": "M0_AMP",              "type": "openpcdet", "extra_flags": ["--amp"]},
    # M1: torch.compile
    {"id": "M1_FP32",             "type": "openpcdet", "extra_flags": ["--compile"]},
    {"id": "M1_AMP",              "type": "openpcdet", "extra_flags": ["--compile", "--amp"]},
    # M2: memory opts
    {"id": "M2_FP32_mem_scatter", "type": "openpcdet", "extra_flags": ["--memory_opt_scatter"]},
    {"id": "M2_FP32_mem_conv2d",  "type": "openpcdet", "extra_flags": ["--memory_opt_conv2d"]},
    {"id": "M2_FP32_mem_both",    "type": "openpcdet", "extra_flags": ["--memory_opt_scatter", "--memory_opt_conv2d"]},
    {"id": "M2_AMP_mem_scatter",  "type": "openpcdet", "extra_flags": ["--amp", "--memory_opt_scatter"]},
    {"id": "M2_AMP_mem_conv2d",   "type": "openpcdet", "extra_flags": ["--amp", "--memory_opt_conv2d"]},
    {"id": "M2_AMP_mem_both",     "type": "openpcdet", "extra_flags": ["--amp", "--memory_opt_scatter", "--memory_opt_conv2d"]},
    # M3: GPU preprocess
    {"id": "M3_FP32",             "type": "openpcdet", "extra_flags": ["--preprocess_gpu"]},
    {"id": "M3_AMP",              "type": "openpcdet", "extra_flags": ["--preprocess_gpu", "--amp"]},
    # M4: GPU preprocess + memory opts
    {"id": "M4_FP32",             "type": "openpcdet", "extra_flags": ["--preprocess_gpu", "--memory_opt_scatter", "--memory_opt_conv2d"]},
    {"id": "M4_AMP",              "type": "openpcdet", "extra_flags": ["--preprocess_gpu", "--memory_opt_scatter", "--memory_opt_conv2d", "--amp"]},
    # M5: CUDA-PointPillars TensorRT FP32
    {"id": "M5_FP32",             "type": "cuda_pp",   "extra_flags": []},
]

# ---------------------------------------------------------------------------
# ncu helpers
# ---------------------------------------------------------------------------

def _run_ncu_openpcdet(variant_flags: list[str], ncu_bin: str, python_bin: str,
                       ld_preload: str = LD_PRELOAD) -> str:
    """Run ncu on profile_suite.py, return stdout."""
    # Use /usr/bin/env to inject LD_PRELOAD directly into the ncu target command.
    # sudo -E would strip LD_PRELOAD; embedding it here bypasses that stripping.
    cmd = [
        "sudo", "-E",
        ncu_bin,
        "--csv",
        "--metrics", f"{METRIC_LD},{METRIC_ST}",
        "--target-processes", "all",
        "/usr/bin/env", f"LD_PRELOAD={ld_preload}",
        python_bin,
        "profile_suite.py",
        "--cfg_file", CFG_FILE,
        "--ckpt",     CKPT_FILE,
        "--warmup",   "0",
        "--steps",    "1",
    ] + variant_flags

    print(f"  Running: {' '.join(cmd[4:])}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(TOOLS_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  [warn] ncu exited {result.returncode}", flush=True)
        print(result.stderr[-2000:], flush=True)
    return result.stdout


def _run_ncu_cuda_pp(demo_path: Path, ncu_bin: str) -> str:
    """Run ncu on ./demo binary, return stdout."""
    cmd = [
        "sudo", "-E",
        ncu_bin,
        "--csv",
        "--metrics", f"{METRIC_LD},{METRIC_ST}",
        str(demo_path),
        "--warmup", "0",
        "--repeat", "1",
    ]

    print(f"  Running: ncu ./demo --warmup 0 --repeat 1", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(demo_path.parent),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  [warn] ncu exited {result.returncode}", flush=True)
        print(result.stderr[-2000:], flush=True)
    return result.stdout


def _parse_ncu_csv(ncu_stdout: str) -> dict[str, float]:
    """
    Parse ncu --csv output.
    Returns {metric_name: total_bytes} summed across all kernels.
    """
    totals: dict[str, float] = {METRIC_LD: 0.0, METRIC_ST: 0.0}

    # ncu --csv output has a CSV section after some header lines
    csv_lines = []
    in_csv = False
    for line in ncu_stdout.splitlines():
        if line.startswith('"ID"') or line.startswith("ID,"):
            in_csv = True
        if in_csv:
            csv_lines.append(line)

    if not csv_lines:
        print("  [warn] No CSV section found in ncu output", flush=True)
        return totals

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    _debug_printed = False
    for row in reader:
        if not _debug_printed:
            print(f"  [debug] CSV columns: {list(row.keys())[:8]}", flush=True)
            _debug_printed = True
        metric = row.get("Metric Name", "").strip().strip('"')
        if metric not in (METRIC_LD, METRIC_ST):
            continue
        val_str = row.get("Metric Value", "0").strip().strip('"').replace(",", "")
        try:
            totals[metric] += float(val_str)
        except ValueError:
            pass

    return totals


def _compute_intensity(ld_bytes: float, st_bytes: float) -> float:
    total = ld_bytes + st_bytes
    if total <= 0:
        return float("nan")
    return THEORY_FLOPS / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NCU DRAM intensity sweep for all 15 variants")
    parser.add_argument("--ncu-bin",     default=NCU_BIN)
    parser.add_argument("--python-bin",  default=PYTHON_BIN)
    parser.add_argument("--ld-preload",  default=LD_PRELOAD,
                        help="LD_PRELOAD path injected into ncu target command")
    parser.add_argument(
        "--demo",
        type=Path,
        default=REPO_ROOT / "CUDA-PointPillars" / "build-fp32" / "demo",
        help="Path to CUDA-PP ./demo binary",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "ncu_intensity.csv",
        help="Output CSV for all measured variants",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Variant IDs to skip (e.g. --skip M5_FP32 M1_FP32)",
    )
    args = parser.parse_args()

    measured: list[dict] = []

    for v in ALL_VARIANTS:
        vid = v["id"]
        if vid in args.skip:
            print(f"\n[skip] {vid}", flush=True)
            continue

        print(f"\n[{vid}] measuring...", flush=True)

        if v["type"] == "openpcdet":
            stdout = _run_ncu_openpcdet(v["extra_flags"], args.ncu_bin, args.python_bin,
                                        args.ld_preload)
        else:
            if not args.demo.exists():
                print(f"  [warn] demo binary not found: {args.demo} — skipping M5", flush=True)
                continue
            stdout = _run_ncu_cuda_pp(args.demo, args.ncu_bin)

        totals = _parse_ncu_csv(stdout)
        ld = totals[METRIC_LD]
        st = totals[METRIC_ST]
        intensity = _compute_intensity(ld, st)

        print(f"  ld={ld/1e9:.3f} GB  st={st/1e9:.3f} GB  intensity={intensity:.1f} FLOP/byte", flush=True)

        measured.append({
            "experiment_cell_id":              vid,
            "ncu_dram_ld_bytes":               round(ld),
            "ncu_dram_st_bytes":               round(st),
            "ncu_dram_total_bytes":            round(ld + st),
            "ncu_compute_intensity_flop_per_byte": round(intensity, 4),
            "theory_flops":                    THEORY_FLOPS,
        })

    if not measured:
        print("\n[warn] Nothing measured.", flush=True)
        return

    fieldnames = list(measured[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(measured)
    print(f"\n[done] {len(measured)} variants → {args.out}", flush=True)

    # Print summary table
    print("\n=== Intensity Summary ===")
    print(f"{'Variant':<30} {'Intensity (FLOP/B)':>20} {'Total DRAM (GB)':>18}")
    print("-" * 70)
    for r in measured:
        print(
            f"{r['experiment_cell_id']:<30}"
            f" {r['ncu_compute_intensity_flop_per_byte']:>20.1f}"
            f" {r['ncu_dram_total_bytes']/1e9:>18.3f}"
        )


if __name__ == "__main__":
    main()
