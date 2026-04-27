#!/usr/bin/env python3
"""
merge_ncu_intensity.py — Fill ncu_compute_intensity_flop_per_byte in all three runs.csv files.

Reads ncu_intensity.csv (measured locally with ncu_intensity_sweep.py) and updates
the corresponding rows in modal_v3_a10, modal_v3_h100, and modal_v3_t4 runs.csv.

Usage:
    python3 merge_ncu_intensity.py [--dry-run]
"""

import argparse
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
NCU_CSV   = REPO_ROOT / "ncu_intensity.csv"

RUNS_CSVS = [
    REPO_ROOT / "modal_outputs" / "modal_v3_a10"  / "runs.csv",
    REPO_ROOT / "modal_outputs" / "modal_v3_h100" / "runs.csv",
    REPO_ROOT / "modal_outputs" / "modal_v3_t4"   / "runs.csv",
]

COL = "ncu_compute_intensity_flop_per_byte"


def load_intensity(path: Path) -> dict[str, float]:
    with open(path) as f:
        return {
            row["experiment_cell_id"]: float(row[COL])
            for row in csv.DictReader(f)
            if row[COL] and row[COL] != "nan"
        }


def merge_one(runs_path: Path, intensity: dict[str, float], dry_run: bool) -> None:
    with open(runs_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    for row in rows:
        cell_id = row["experiment_cell_id"]
        if cell_id in intensity:
            old = row.get(COL, "")
            new = str(round(intensity[cell_id], 4))
            if old != new:
                row[COL] = new
                updated += 1

    print(f"  {runs_path.parent.name}/runs.csv — {updated} rows updated")

    if dry_run:
        return

    backup = runs_path.with_suffix(".csv.bak")
    shutil.copy2(runs_path, backup)

    with open(runs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    intensity = load_intensity(NCU_CSV)
    print(f"Loaded {len(intensity)} variants from {NCU_CSV.name}:")
    for k, v in sorted(intensity.items()):
        print(f"  {k:<30} {v:.4f} FLOP/byte")
    print()

    for runs_path in RUNS_CSVS:
        if not runs_path.exists():
            print(f"  [skip] {runs_path} not found")
            continue
        merge_one(runs_path, intensity, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] No files written.")
    else:
        print("\nDone. Backups saved as *.csv.bak")


if __name__ == "__main__":
    main()
