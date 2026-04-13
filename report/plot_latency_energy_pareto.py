#!/usr/bin/env python3
"""
Scatter + Pareto frontiers: energy forward latency vs energy per sample [J] from runs.csv.
**FP32** and **AMP** cells each get their own Pareto set (minimize latency and J/sample within the family).
Objectives: minimize energy_forward_mean_ms and minimize J/sample (= 1 / energy_samples_per_J).

Uses the same runs.csv resolution and cell ordering as ``plot_latency_breakdown.py`` /
``plot_energy.py`` (``normalize_runs_csv_path``, ``sort_by_m_group``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from plot_latency_breakdown import (
    MLSYS_PLOT_RC,
    _slug_for_filename,
    list_distinct_gpus,
    load_filtered_latest,
    normalize_runs_csv_path,
    sort_by_m_group,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_FULL = _REPO_ROOT / "modal_mls_results" / "modal_v2_allcells_full" / "runs.csv"
_V2 = _REPO_ROOT / "modal_mls_results" / "modal_v2" / "runs.csv"
DEFAULT_CSV = _V2_FULL if _V2_FULL.is_file() else _V2
DEFAULT_GPU = "A10"


def _precision_kind(df: pd.DataFrame) -> pd.Series:
    """Per-row label: ``FP32``, ``AMP``, or ``other`` (same rules as ``sort_by_m_group``)."""
    cid = df["experiment_cell_id"].astype(str)
    pr = pd.Series("other", index=df.index, dtype=object)
    if "precision_mode" in df.columns:
        pm = df["precision_mode"].astype(str).str.strip().str.upper()
        pr[pm == "FP32"] = "FP32"
        pr[pm == "AMP"] = "AMP"
    low = cid.str.lower()
    pr[(pr == "other") & low.str.contains("_fp32", na=False)] = "FP32"
    pr[(pr == "other") & low.str.contains("_amp", na=False)] = "AMP"
    return pr


def _pareto_min_x_min_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True where point is Pareto-optimal for minimize x, minimize y.
    j dominates i iff x_j <= x_i, y_j <= y_i, and at least one strict inequality.
    """
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            mask[i] = False
            continue
        for j in range(n):
            if i == j or not np.isfinite(x[j]) or not np.isfinite(y[j]):
                continue
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                mask[i] = False
                break
    return mask


def _pareto_mask_subset(
    x_full: np.ndarray, y_full: np.ndarray, group_mask: np.ndarray
) -> np.ndarray:
    """Full-length boolean mask: True only at ``group_mask`` positions that are Pareto within that group."""
    out = np.zeros(len(x_full), dtype=bool)
    if not group_mask.any():
        return out
    idx = np.flatnonzero(group_mask)
    xs, ys = x_full[idx], y_full[idx]
    sub = _pareto_min_x_min_y(xs, ys)
    out[idx[sub]] = True
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pareto plot: energy forward latency vs J/sample with separate FP32 and AMP frontiers."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to runs.csv or matrix directory containing it (default: modal_v2_allcells_full if present)",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Matrix bundle directory (must contain runs.csv); overrides --csv when set",
    )
    p.add_argument(
        "--gpu",
        type=str,
        default=DEFAULT_GPU,
        help="gpu_name filter (same rules as plot_latency_breakdown.py --gpu). Default: %(default)s",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: pareto_energy_<gpu_slug>.png next to this script)",
    )
    p.add_argument(
        "--list-gpus",
        action="store_true",
        help="Print distinct gpu_name values and exit",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=176,
        help="PNG resolution (default aligned with other report plots)",
    )
    args = p.parse_args()
    try:
        if args.runs_root is not None:
            csv_path = normalize_runs_csv_path(args.runs_root)
        else:
            csv_path = normalize_runs_csv_path(args.csv)
    except FileNotFoundError as ex:
        print("%s" % ex, file=sys.stderr)
        sys.exit(1)

    if args.list_gpus:
        for n in list_distinct_gpus(csv_path):
            print(n)
        sys.exit(0)

    out_path = (
        args.out.resolve()
        if args.out
        else Path(__file__).resolve().parent
        / ("pareto_energy_%s.png" % _slug_for_filename(args.gpu))
    )

    df, gpu_resolved = load_filtered_latest(csv_path, args.gpu)
    df = sort_by_m_group(df)

    xcol = "energy_forward_mean_ms"
    spj_col = "energy_samples_per_J"
    if xcol not in df.columns or spj_col not in df.columns:
        raise ValueError("runs.csv missing %s or %s" % (xcol, spj_col))

    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    spj = pd.to_numeric(df[spj_col], errors="coerce").to_numpy(dtype=float)
    labels = df["experiment_cell_id"].astype(str).tolist()

    # J/sample = 1 / (samples/J); require positive samples/J.
    j_per_sample = np.divide(1.0, spj, out=np.full_like(spj, np.nan), where=(spj > 0))

    valid = np.isfinite(x) & np.isfinite(j_per_sample) & (x > 0) & (j_per_sample > 0)
    if not valid.any():
        raise ValueError(
            "No rows with finite positive %s and %s for this GPU filter." % (xcol, spj_col)
        )

    x_v, y_v, lab_v = x[valid], j_per_sample[valid], [labels[i] for i in range(len(labels)) if valid[i]]
    df_v = df.iloc[np.flatnonzero(valid)].copy()
    prec_v = _precision_kind(df_v).tolist()
    kinds = np.array(prec_v, dtype=object)
    fp32_m = kinds == "FP32"
    amp_m = kinds == "AMP"
    other_m = ~(fp32_m | amp_m)

    pareto_fp32 = _pareto_mask_subset(x_v, y_v, fp32_m)
    pareto_amp = _pareto_mask_subset(x_v, y_v, amp_m)

    # Okabe–Ito style: FP32 frontier blue, AMP frontier orange; dominated = lighter fills.
    c_fp32_dom, c_fp32_par = "#b3cde3", "#0072B2"
    c_amp_dom, c_amp_par = "#f4d7a8", "#E69F00"
    c_other, c_other_ed = "#c0c0c0", "#8a8a8a"

    def _plot_hull(ax, mask: np.ndarray, color: str, label: str) -> None:
        if not mask.any():
            return
        px, py = x_v[mask], y_v[mask]
        order = np.argsort(px)
        ax.plot(
            px[order],
            py[order],
            color=color,
            linestyle="--",
            linewidth=1.25,
            alpha=0.78,
            zorder=1,
            label=label,
        )

    with plt.rc_context(MLSYS_PLOT_RC):
        fig, ax = plt.subplots(figsize=(12.0, 7.0))

        if fp32_m.any():
            dom_fp = fp32_m & ~pareto_fp32
            if dom_fp.any():
                ax.scatter(
                    x_v[dom_fp],
                    y_v[dom_fp],
                    s=56,
                    c=c_fp32_dom,
                    edgecolors="#f5f5f5",
                    linewidths=0.65,
                    zorder=2,
                    label="FP32 dominated",
                )
            if pareto_fp32.any():
                ax.scatter(
                    x_v[pareto_fp32],
                    y_v[pareto_fp32],
                    s=92,
                    c=c_fp32_par,
                    edgecolors="#003d66",
                    linewidths=1.0,
                    zorder=4,
                    label="FP32 Pareto",
                )
                _plot_hull(ax, pareto_fp32, c_fp32_par, "FP32 frontier hull")

        if amp_m.any():
            dom_amp = amp_m & ~pareto_amp
            if dom_amp.any():
                ax.scatter(
                    x_v[dom_amp],
                    y_v[dom_amp],
                    s=56,
                    c=c_amp_dom,
                    edgecolors="#f5f5f5",
                    linewidths=0.65,
                    zorder=2,
                    label="AMP dominated",
                )
            if pareto_amp.any():
                ax.scatter(
                    x_v[pareto_amp],
                    y_v[pareto_amp],
                    s=92,
                    c=c_amp_par,
                    edgecolors="#8a5a00",
                    linewidths=1.0,
                    zorder=4,
                    label="AMP Pareto",
                )
                _plot_hull(ax, pareto_amp, c_amp_par, "AMP frontier hull")

        if other_m.any():
            ax.scatter(
                x_v[other_m],
                y_v[other_m],
                s=56,
                c=c_other,
                edgecolors="#f5f5f5",
                linewidths=0.65,
                zorder=2,
                label="Other precision",
            )

        for xi, yi, li in zip(x_v, y_v, lab_v):
            ax.annotate(
                li,
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.92,
            )

        ax.set_xlabel("Energy-window forward latency [ms] (minimize)")
        ax.set_ylabel("Energy per sample [J] (minimize)")
        ax.set_title(
            "Latency vs energy per sample — %s\n"
            "Two frontiers: Pareto within FP32 cells only, and within AMP cells only (lower-left is better)"
            % gpu_resolved
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.94)
        ax.grid(True, alpha=0.45)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
    print(
        "Wrote %s (%d points, FP32 Pareto=%d, AMP Pareto=%d, gpu=%s)"
        % (
            out_path,
            len(x_v),
            int(pareto_fp32.sum()),
            int(pareto_amp.sum()),
            gpu_resolved,
        )
    )


if __name__ == "__main__":
    main()
