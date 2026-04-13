#!/usr/bin/env python3
"""
3D scatter for latency-energy-accuracy trade-off.

- X axis: latency (ms)
- Y axis: energy (J/sample)
- Z axis: accuracy
- Color: GPU set (A10 / H100 / T4)
- Marker size: fixed (all points identical)
- Label: experiment_cell_id shown above each point
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None


def _resolve_runs_csv(path_like: str) -> Path:
    p = Path(path_like).expanduser().resolve()
    if p.is_dir():
        p = p / "runs.csv"
    if not p.is_file():
        raise FileNotFoundError(f"runs.csv not found: {p}")
    return p


def _gpu_mask(series: pd.Series, gpu_hint: str) -> pd.Series:
    s = series.astype(str)
    if gpu_hint.upper() == "A10":
        return s.str.contains(r"A10(?!0)", case=False, regex=True, na=False)
    return s.str.contains(gpu_hint, case=False, regex=False, na=False)


def _load_latest(
    runs_csv: Path,
    *,
    gpu_hint: str,
    bundle_name: str,
    latency_col: str,
    accuracy_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(runs_csv)
    if "experiment_status" in df.columns:
        df = df[df["experiment_status"].astype(str) == "measured"]
    if "gpu_name" not in df.columns:
        raise ValueError(f"{runs_csv} missing gpu_name")
    df = df[_gpu_mask(df["gpu_name"], gpu_hint)].copy()
    if df.empty:
        raise ValueError(f"No rows match gpu={gpu_hint} in {runs_csv}")

    df["_ts"] = pd.to_datetime(df["timestamp_iso"], utc=True, errors="coerce")
    df = df.sort_values("_ts").drop_duplicates(subset=["experiment_cell_id"], keep="last")

    for col in [latency_col, "energy_total_J", "measured_steps", accuracy_col]:
        if col not in df.columns:
            raise ValueError(f"{runs_csv} missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame(
        {
            "experiment_cell_id": df["experiment_cell_id"].astype(str),
            "latency_ms": df[latency_col],
            "energy_j_per_sample": df["energy_total_J"] / df["measured_steps"].replace(0, pd.NA),
            "accuracy": df[accuracy_col],
            "bundle": bundle_name,
        }
    ).dropna(subset=["latency_ms", "energy_j_per_sample", "accuracy"])
    return out


def _pareto_mask_min_min_max(df: pd.DataFrame) -> pd.Series:
    """
    3D Pareto mask for objectives:
    - minimize energy_j_per_sample
    - minimize latency_ms
    - maximize accuracy
    """
    vals = df[["energy_j_per_sample", "latency_ms", "accuracy"]].to_numpy()
    n = len(vals)
    keep = [True] * n
    for i in range(n):
        ei, li, ai = vals[i]
        for j in range(n):
            if i == j:
                continue
            ej, lj, aj = vals[j]
            better_or_equal = (ej <= ei) and (lj <= li) and (aj >= ai)
            strictly_better = (ej < ei) or (lj < li) or (aj > ai)
            if better_or_equal and strictly_better:
                keep[i] = False
                break
    return pd.Series(keep, index=df.index)


def _add_frontier_mesh(
    ax,
    frontier: pd.DataFrame,
    *,
    color: str,
    alpha: float,
) -> None:
    """
    Add a shallow convex-hull mesh for the frontier cloud.
    Coordinates are (energy, latency, accuracy).
    """
    pts = frontier[["energy_j_per_sample", "latency_ms", "accuracy"]].to_numpy(dtype=float)
    if len(pts) < 3:
        return
    if len(pts) == 3:
        tri = Poly3DCollection([pts], facecolors=color, edgecolors="none", alpha=alpha)
        ax.add_collection3d(tri)
        return
    if ConvexHull is None:
        return
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    faces = [pts[simplex] for simplex in hull.simplices]
    poly = Poly3DCollection(faces, facecolors=color, edgecolors="none", alpha=alpha)
    ax.add_collection3d(poly)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D latency-energy-accuracy scatter with GPU colors and point labels."
    )
    parser.add_argument(
        "--a10-root",
        default="/home/nas/polin/cmu-berlin/MLS/modal_outputs/modal_v2_a10",
        help="A10 bundle root or runs.csv path",
    )
    parser.add_argument(
        "--h100-root",
        default="/home/nas/polin/cmu-berlin/MLS/modal_outputs/modal_v2_h100",
        help="H100 bundle root or runs.csv path",
    )
    parser.add_argument(
        "--t4-root",
        default="/home/nas/polin/cmu-berlin/MLS/modal_outputs/modal_v2_t4",
        help="T4 bundle root or runs.csv path",
    )
    parser.add_argument(
        "--latency-col",
        default="prof_forward_mean_ms",
        help="Latency column (default: prof_forward_mean_ms)",
    )
    parser.add_argument(
        "--accuracy-col",
        default="map_car_r11",
        help="Accuracy column (default: map_car_r11)",
    )
    parser.add_argument(
        "--out",
        default="/home/nas/polin/cmu-berlin/MLS/report/accuracy_latency_energy_3d.png",
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--elev", type=float, default=22.0, help="3D view elevation")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D view azimuth")
    parser.add_argument(
        "--frontier-mesh-alpha",
        type=float,
        default=0.14,
        help="Alpha for convex-hull mesh shading on each GPU frontier.",
    )
    args = parser.parse_args()

    bundles = [
        ("A10", "A10", _resolve_runs_csv(args.a10_root)),
        ("H100", "H100", _resolve_runs_csv(args.h100_root)),
        ("T4", "T4", _resolve_runs_csv(args.t4_root)),
    ]

    rows = []
    for bundle_name, gpu_hint, runs_csv in bundles:
        rows.append(
            _load_latest(
                runs_csv,
                gpu_hint=gpu_hint,
                bundle_name=bundle_name,
                latency_col=args.latency_col,
                accuracy_col=args.accuracy_col,
            )
        )
    df = pd.concat(rows, ignore_index=True)
    if df.empty:
        raise ValueError("No valid rows to plot.")

    colors = {"A10": "#1f77b4", "H100": "#d62728", "T4": "#2ca02c"}
    marker_size = 42

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for bundle in ["A10", "H100", "T4"]:
        d = df[df["bundle"] == bundle]
        if d.empty:
            continue
        ax.scatter(
            d["energy_j_per_sample"],
            d["latency_ms"],
            d["accuracy"],
            s=marker_size,
            c=colors[bundle],
            label=bundle,
            depthshade=False,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.4,
        )
        for _, r in d.iterrows():
            ax.text(
                float(r["energy_j_per_sample"]),
                float(r["latency_ms"]),
                float(r["accuracy"]) + 0.00035,
                str(r["experiment_cell_id"]),
                fontsize=7,
                color=colors[bundle],
                ha="center",
                va="bottom",
            )

        # 3D Pareto frontier (min energy, min latency, max accuracy), one line per GPU.
        frontier = d[_pareto_mask_min_min_max(d)].copy()
        if not frontier.empty:
            frontier = frontier.sort_values(
                by=["energy_j_per_sample", "latency_ms", "accuracy"],
                ascending=[True, True, False],
            )
            _add_frontier_mesh(
                ax,
                frontier,
                color=colors[bundle],
                alpha=float(args.frontier_mesh_alpha),
            )
            # Frontier is represented by mesh shading only (no explicit line).

    ax.set_xlabel("energy_total_J / measured_steps [J/sample]", labelpad=10)
    ax.set_ylabel(f"{args.latency_col} [ms]", labelpad=10)
    ax.set_zlabel(args.accuracy_col, labelpad=10)
    ax.set_zlim(84.4, 85.0)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    desired = ["T4", "A10", "H100"]
    hmap = {lb: h for h, lb in zip(handles, labels)}
    ordered_handles = [hmap[k] for k in desired if k in hmap]
    ordered_labels = [k for k in desired if k in hmap]
    ax.legend(ordered_handles, ordered_labels, title="GPU", loc="upper left")
    ax.set_title("3D trade-off: latency-energy-accuracy + per-GPU Pareto frontier")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path} ({len(df)} points)")


if __name__ == "__main__":
    main()

