#!/usr/bin/env python3
"""Merge experiment_matrix_fp32_amp.csv with runs.csv (same logic as plot_research_results.ipynb)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LEGACY_TO_MATRIX_VARIANT = {
    "baseline_fp32": "M0_FP32",
    "fp16_amp": "M0_AMP",
    "torch_compile_fp32": "M1_FP32",
    "M1_baseline_fp32": "M0_FP32",
    "M1_fp16_amp": "M0_AMP",
    "M2_torch_compile_fp32": "M1_FP32",
}

DEFAULT_METRICS = (
    "map_car_r11",
    "kitti_car_3d_easy_r40",
    "kitti_car_3d_moderate_r40",
    "kitti_car_3d_hard_r40",
    "prof_full_frame_mean_ms",
    "energy_forward_mean_ms",
    "energy_total_J",
    "energy_samples_per_J",
    "prof_peak_gpu_memory_mb",
    "prof_peak_gpu_memory_steady_mb",
    "prof_t_rt_mean_ms",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory with experiment_matrix_fp32_amp.csv and runs.csv",
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to write merged summary CSV",
    )
    args = p.parse_args()
    data_dir = args.data_dir.resolve()
    exp_csv = data_dir / "experiment_matrix_fp32_amp.csv"
    runs_csv = data_dir / "runs.csv"
    if not exp_csv.is_file():
        raise SystemExit("Missing %s" % exp_csv)

    exp = pd.read_csv(exp_csv)
    exp["status"] = exp["status"].str.strip()
    summary = exp[["cell_id", "model_variant", "precision_mode", "variant_name", "status"]].copy()

    if runs_csv.is_file():
        runs_m = pd.read_csv(runs_csv).copy()
        runs_m["matrix_variant"] = (
            runs_m["variant_name"].map(LEGACY_TO_MATRIX_VARIANT).fillna(runs_m["variant_name"])
        )
        metric_cols = [c for c in DEFAULT_METRICS if c in runs_m.columns]
        rmini = runs_m[["matrix_variant"] + metric_cols]
        summary = summary.merge(rmini, left_on="variant_name", right_on="matrix_variant", how="left")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print("Wrote", args.output_csv.resolve())


if __name__ == "__main__":
    main()
