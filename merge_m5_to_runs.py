"""
Merge M5 (CUDA-PointPillars TRT) benchmark results into modal_v3_{gpu}/runs.csv.

Usage:
    conda run -n mlsys python merge_m5_to_runs.py [--dry-run]
"""

import argparse
import csv
import sys
from pathlib import Path

BASE = Path(__file__).parent
M5_CSVS = [
    BASE / "modal_outputs/cuda_pp/cuda_pp/M5_FP32.csv",
    BASE / "modal_outputs/cuda_pp/cuda_pp/M5_FP16.csv",
]

GPU_TO_DIR = {
    "NVIDIA A10G":           BASE / "modal_outputs/modal_v3_a10/runs.csv",
    "Tesla T4":              BASE / "modal_outputs/modal_v3_t4/runs.csv",
    "NVIDIA H100 80GB HBM3": BASE / "modal_outputs/modal_v3_h100/runs.csv",
}

# Column names for each M5 CSV schema variant
OLD_COLS = [
    "run_id", "timestamp_iso", "variant_name", "experiment_cell_id",
    "model_variant", "precision_mode", "experiment_status", "model_name",
    "gpu_name", "warmup_steps", "measured_steps", "trt_precision",
    "prof_full_frame_mean_ms", "prof_full_frame_p50_ms",
    "prof_full_frame_p95_ms", "prof_full_frame_p99_ms",
    "prof_throughput_sps", "energy_mean_power_W", "energy_total_J",
    "n_measured_frames",
]
NEW_COLS = [
    "run_id", "timestamp_iso", "variant_name", "experiment_cell_id",
    "model_variant", "precision_mode", "experiment_status", "model_name",
    "gpu_name", "warmup_steps", "measured_steps", "trt_precision",
    "prof_forward_mean_ms", "prof_pre_processing_mean_ms",
    "prof_h2d_mean_ms", "prof_postprocess_mean_ms",
    "prof_full_frame_mean_ms", "prof_full_frame_p50_ms",
    "prof_full_frame_p95_ms", "prof_full_frame_p99_ms",
    "prof_throughput_sps", "energy_mean_power_W", "energy_total_J",
    "n_measured_frames",
]


def parse_m5_csv(path: Path) -> list[dict]:
    """Parse M5 CSV, keep warmup=500 rows, return latest per (gpu_name, precision_mode)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip stale header
        for line in reader:
            n = len(line)
            if n == len(NEW_COLS):
                row = dict(zip(NEW_COLS, line))
            elif n == len(OLD_COLS):
                continue  # skip old schema (no stage breakdown)
            else:
                print(f"  [warn] skipping row with unexpected {n} cols: {line[:3]}")
                continue
            if int(row["warmup_steps"]) == 500:
                rows.append(row)

    # Keep latest timestamp per (gpu_name, precision_mode)
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["gpu_name"], r["precision_mode"])
        if key not in best or r["timestamp_iso"] > best[key]["timestamp_iso"]:
            best[key] = r
    return list(best.values())


def get_runs_columns(runs_csv: Path) -> list[str]:
    with open(runs_csv, newline="") as f:
        return next(csv.reader(f))


def existing_run_ids(runs_csv: Path) -> set[str]:
    ids = set()
    with open(runs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["run_id"])
    return ids


def build_runs_row(m5: dict, runs_cols: list[str]) -> dict:
    is_fp16 = m5.get("trt_precision", "fp32").lower() == "fp16"
    row = {col: "" for col in runs_cols}

    # Identity
    row["run_id"]             = m5["run_id"]
    row["timestamp_iso"]      = m5["timestamp_iso"]
    row["variant_name"]       = m5["variant_name"]
    row["experiment_cell_id"] = m5["experiment_cell_id"]
    row["model_variant"]      = m5["model_variant"]
    row["precision_mode"]     = m5["precision_mode"]
    row["experiment_status"]  = m5["experiment_status"]
    row["model_name"]         = m5["model_name"]
    row["gpu_name"]           = m5["gpu_name"]
    row["warmup_steps"]       = m5["warmup_steps"]
    row["measured_steps"]     = m5["measured_steps"]

    # Flags (TRT binary — no Python-level flags)
    for flag in ["flag_compile", "flag_amp", "flag_preprocess_gpu",
                 "flag_compile_voxelizer", "flag_nhwc", "flag_memory_opt_scatter",
                 "flag_int8"]:
        row[flag] = "false"
    row["flag_fp16_full"] = "true" if is_fp16 else "false"

    # Latency
    for col in ["prof_full_frame_mean_ms", "prof_full_frame_p50_ms",
                "prof_full_frame_p95_ms", "prof_full_frame_p99_ms",
                "prof_throughput_sps", "prof_h2d_mean_ms",
                "prof_forward_mean_ms", "prof_pre_processing_mean_ms",
                "prof_postprocess_mean_ms"]:
        if col in m5:
            row[col] = m5[col]

    # Energy
    row["energy_mean_power_W"] = m5.get("energy_mean_power_W", "")
    row["energy_total_J"]      = m5.get("energy_total_J", "")

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be appended without writing")
    args = parser.parse_args()

    all_rows = []
    for csv_path in M5_CSVS:
        rows = parse_m5_csv(csv_path)
        print(f"[{csv_path.name}] parsed {len(rows)} warmup=500 rows")
        all_rows.extend(rows)

    # Group by target runs.csv
    by_file: dict[Path, list[dict]] = {}
    for m5 in all_rows:
        target = GPU_TO_DIR.get(m5["gpu_name"])
        if target is None:
            print(f"  [warn] unknown gpu '{m5['gpu_name']}', skipping")
            continue
        by_file.setdefault(target, []).append(m5)

    for runs_csv, m5_rows in by_file.items():
        runs_cols = get_runs_columns(runs_csv)
        existing = existing_run_ids(runs_csv)
        new_rows = [r for r in m5_rows if r["run_id"] not in existing]
        skipped  = len(m5_rows) - len(new_rows)

        print(f"\n{runs_csv.relative_to(BASE)}")
        print(f"  {len(m5_rows)} candidate rows, {skipped} already present, "
              f"{len(new_rows)} to append")

        for r in new_rows:
            print(f"  + {r['run_id']}  {r['variant_name']}  "
                  f"gpu={r['gpu_name']}  "
                  f"full_frame={r.get('prof_full_frame_mean_ms','')} ms  "
                  f"forward={r.get('prof_forward_mean_ms','')} ms")

        if args.dry_run or not new_rows:
            continue

        with open(runs_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=runs_cols, extrasaction="ignore")
            for r in new_rows:
                writer.writerow(build_runs_row(r, runs_cols))

        print(f"  [done] appended {len(new_rows)} rows")

    if args.dry_run:
        print("\n[dry-run] nothing written")


if __name__ == "__main__":
    main()
