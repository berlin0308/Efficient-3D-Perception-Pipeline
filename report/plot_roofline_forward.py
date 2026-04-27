#!/usr/bin/env python3
"""
Roofline-style plot for **forward (GPU) only**: all experiment cells on one chart.

- **Y**: attained GFLOP/s = (theoretical forward FLOPs) / (``prof_forward_mean_ms`` / 1e3).
- **X**: arithmetic intensity [FLOP / byte] = theoretical FLOPs / bytes moved (memory traffic).

Intensity source (``--intensity-source``):

- ``blend`` (default): use ``ncu_compute_intensity_flop_per_byte`` from ``runs.csv`` when finite;
  otherwise fall back to ``F_theory / forward_bytes_proxy``.
- ``ncu``: NCU column only (cells without it are skipped).
- ``proxy``: always ``F_theory / forward_bytes_proxy``.

If **no row** has a valid NCU intensity (common when ``ncu_*`` columns were never filled), ``blend`` and
``proxy`` put **every cell at the same x** — that is expected: you only have one global byte proxy, not
per-cell DRAM traffic. Fill NCU metrics in ``runs.csv`` (or add a per-cell byte model) to spread the x axis.

Theoretical FLOPs match the forward-only sum in ``report/plot_energy.py`` ``STAGE_GMACS``
(PillarVFE + scatter + BEV backbone + anchor; excludes ``post_processing``).

Roof curves use device **peak FP32** (TFLOP/s) and **DRAM bandwidth** (10⁹ byte/s style) as
straightforward ceilings — calibrate ``--peak-tflops`` and ``--dram-bandwidth-gbs`` to your SKU.

Usage (repo root):

    python3 report/plot_roofline_forward.py --gpu A10 --runs-root modal_mls_results/modal_v2_allcells_full
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

try:
    from plot_latency import (
        MLSYS_PLOT_RC,
        _slug_for_filename,
        list_distinct_gpus,
        load_filtered_latest,
        normalize_runs_csv_path,
        sort_by_m_group,
    )
except ImportError:
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

# Same forward-only FLOP budget as report/plot_energy.py STAGE_GMACS (no post_processing).
_FORWARD_THEORY_FLOPS = (
    1.5e9 + 0.05e9 + 18.0e9 + 1.2e9
)  # ≈ 20.75e9 FLOPs / inference forward pass (analytical order-of-magnitude)


def _precision_kind(df: pd.DataFrame) -> pd.Series:
    cid = df["experiment_cell_id"].astype(str)
    pr = pd.Series("other", index=df.index, dtype=object)
    if "precision_mode" in df.columns:
        pm = df["precision_mode"].astype(str).str.strip().str.upper()
        pr[pm == "FP32"] = "FP32"
        pr[pm == "AMP"] = "AMP"
    low = cid.str.lower()
    pr[(pr == "other") & low.str.contains("_fp32", na=False)] = "FP32"
    pr[(pr == "other") & low.str.contains("_amp", na=False)] = "AMP"
    pr[(pr == "other") & low.str.contains("_fp16", na=False)] = "FP16"
    return pr


def _roof_gflops(intensity_flop_per_b: np.ndarray, peak_tflops: float, dram_bw_gbs: float) -> np.ndarray:
    """
    Roof GFLOP/s = min(peak_TF * 1024, I * dram_bw) with I in [FLOP/byte], dram_bw in 10^9 byte/s,
    peak_TF in 10^12 FLOP/s → peak_GF = peak_TF * 1000 (TFLOPS ≈ 1000 GFLOPS for labeling consistency).

    Here ``peak_tflops`` is peak **10^12 FLOP/s**; GFLOP/s numerically = peak_tflops * 1000.
    """
    peak_gf = peak_tflops * 1e3
    bw = dram_bw_gbs * 1e9
    mem_roof_gf = intensity_flop_per_b * bw / 1e9
    return np.minimum(peak_gf, mem_roof_gf)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Roofline-style plot: forward theoretical FLOPs vs prof_forward_mean_ms for all cells."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="runs.csv or matrix directory (default: modal_v2_allcells_full if present)",
    )
    p.add_argument("--runs-root", type=Path, default=None, help="Bundle dir with runs.csv; overrides --csv")
    p.add_argument("--gpu", type=str, default=DEFAULT_GPU, help="gpu_name filter (same as plot_latency_breakdown)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: roofline_forward_<gpu_slug>.png next to this script)",
    )
    p.add_argument("--list-gpus", action="store_true", help="Print gpu_name values and exit")
    p.add_argument("--dpi", type=int, default=176)
    p.add_argument(
        "--theory-flops",
        type=float,
        default=None,
        help="Override theoretical forward FLOPs (default: %.3g)" % _FORWARD_THEORY_FLOPS,
    )
    p.add_argument(
        "--forward-bytes-proxy",
        type=float,
        default=3.2e8,
        help="DRAM bytes proxy for forward when NCU intensity missing [bytes]. Default ~320 MB order-of-mag.",
    )
    p.add_argument(
        "--intensity-source",
        choices=("blend", "ncu", "proxy"),
        default="blend",
        help="How to set x-axis intensity (NCU column vs analytical proxy).",
    )
    p.add_argument(
        "--peak-tflops",
        type=float,
        default=31.0,
        help="Device peak FP32 throughput (10^12 FLOP/s), for roof ceiling (calibrate to SKU).",
    )
    p.add_argument(
        "--dram-bandwidth-gbs",
        type=float,
        default=600.0,
        help="DRAM bandwidth in 10^9 bytes/s (marketing GB/s style), for memory roof slope.",
    )
    p.add_argument(
        "--imax",
        type=float,
        default=None,
        help="Max intensity for roof curve [FLOP/byte] (default: auto from data)",
    )
    args = p.parse_args()

    try:
        csv_path = normalize_runs_csv_path(args.runs_root) if args.runs_root is not None else normalize_runs_csv_path(args.csv)
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
        else _SCRIPT_DIR / ("roofline_forward_%s.png" % _slug_for_filename(args.gpu))
    )

    f_theory = float(args.theory_flops) if args.theory_flops is not None else _FORWARD_THEORY_FLOPS

    df, gpu_resolved = load_filtered_latest(csv_path, args.gpu)
    df = sort_by_m_group(df)
    tcol = "prof_forward_mean_ms"
    ncol = "ncu_compute_intensity_flop_per_byte"
    if tcol not in df.columns:
        raise ValueError("runs.csv missing %s" % tcol)

    t_ms = pd.to_numeric(df[tcol], errors="coerce")
    t_s = t_ms / 1000.0
    achieved_gflops = (f_theory / t_s) / 1e9

    ncu_i = None
    if ncol in df.columns:
        ncu_i = pd.to_numeric(df[ncol], errors="coerce")

    proxy_i = f_theory / float(args.forward_bytes_proxy)

    intensity = pd.Series(np.nan, index=df.index, dtype=float)
    used_ncu = pd.Series(False, index=df.index, dtype=bool)
    for idx in df.index:
        use_ncu = ncu_i is not None and np.isfinite(ncu_i.loc[idx]) and float(ncu_i.loc[idx]) > 0
        if args.intensity_source == "proxy":
            intensity.loc[idx] = proxy_i
        elif args.intensity_source == "ncu":
            intensity.loc[idx] = float(ncu_i.loc[idx]) if use_ncu else np.nan
            used_ncu.loc[idx] = bool(use_ncu)
        else:
            if use_ncu:
                intensity.loc[idx] = float(ncu_i.loc[idx])
                used_ncu.loc[idx] = True
            else:
                intensity.loc[idx] = proxy_i

    prec = _precision_kind(df)
    labels = df["experiment_cell_id"].astype(str)

    ok = np.isfinite(intensity.to_numpy()) & np.isfinite(achieved_gflops.to_numpy()) & (t_s.to_numpy() > 0)
    if not ok.any():
        raise ValueError("No valid points (check forward ms and intensity source).")

    x_plot = intensity.to_numpy(dtype=float)[ok]
    y_plot = achieved_gflops.to_numpy(dtype=float)[ok]
    lab_plot = labels.to_numpy()[ok]
    prec_plot = prec.to_numpy()[ok]

    n_ncu_pts = int(np.sum(used_ncu.to_numpy()[ok]))
    n_pts = int(ok.sum())
    x_std = float(np.std(x_plot)) if len(x_plot) > 1 else 0.0
    uniform_x = (len(x_plot) > 1) and (x_std <= 1e-12 * max(float(np.mean(x_plot)), 1.0))
    if uniform_x:
        print(
            "plot_roofline_forward: all %d points share the same arithmetic intensity (%.4g FLOP/B). "
            "runs.csv has no per-cell ncu_compute_intensity_flop_per_byte (using --forward-bytes-proxy). "
            "Fill NCU in runs.csv for a spread x-axis."
            % (n_pts, float(x_plot[0]) if len(x_plot) else 0.0),
            file=sys.stderr,
        )

    i_min = max(0.05, float(np.nanmin(x_plot)) * 0.35)
    i_ridge = (args.peak_tflops * 1e12) / (args.dram_bandwidth_gbs * 1e9)
    i_max = float(args.imax) if args.imax is not None else float(np.nanmax(x_plot)) * 2.5
    i_max = max(i_max, i_min * 10.0, 5.0, i_ridge * 2.0)
    ii = np.geomspace(i_min, i_max, 200)
    roof = _roof_gflops(ii, args.peak_tflops, args.dram_bandwidth_gbs)

    with plt.rc_context(MLSYS_PLOT_RC):
        fig, ax = plt.subplots(figsize=(12.0, 7.0))
        ax.loglog(ii, roof, color="#333333", linewidth=2.0, linestyle="-", label="Roof min(peak, I·BW)", zorder=1)
        ax.axhline(
            args.peak_tflops * 1e3,
            color="#666666",
            linewidth=1.2,
            linestyle=":",
            label="Peak %.1f TFLOP/s (FP32 ceiling)" % args.peak_tflops,
            zorder=1,
        )

        c_fp32, c_amp, c_fp16, c_other = "#0072B2", "#E69F00", "#009E73", "#7f7f7f"
        for kind, c, m, leg in (
            ("FP32", c_fp32, "o", "FP32 cells"),
            ("AMP", c_amp, "s", "AMP cells"),
            ("FP16", c_fp16, "D", "FP16 cells"),
            ("other", c_other, "^", "Other"),
        ):
            msk = prec_plot == kind
            if not msk.any():
                continue
            ax.scatter(
                x_plot[msk],
                y_plot[msk],
                s=72,
                c=c,
                marker=m,
                edgecolors="#f5f5f5",
                linewidths=0.85,
                zorder=3,
                label="%s (n=%d)" % (leg, int(msk.sum())),
            )

        for xi, yi, li in zip(x_plot, y_plot, lab_plot):
            ax.annotate(
                li,
                (xi, yi),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.9,
            )

        ax.set_xlabel("Arithmetic intensity [FLOP / byte] (forward; NCU or proxy)")
        ax.set_ylabel("Attained [GFLOP/s] = F_theory / t_forward")
        if args.intensity_source == "blend":
            n_proxy = n_pts - n_ncu_pts
            if n_proxy == 0:
                sub_i = "NCU intensity: %d / %d cells (all from NCU)" % (n_ncu_pts, n_pts)
            else:
                sub_i = "NCU intensity: %d / %d cells (%d use bytes proxy)" % (n_ncu_pts, n_pts, n_proxy)
        elif args.intensity_source == "ncu":
            sub_i = "Intensity = NCU only"
        else:
            sub_i = "Intensity = F_theory / bytes_proxy (same for all)"
        ax.set_title(
            "Forward roofline-style chart — %s\nF_theory = %.3g FLOP/run; ridge I* ≈ %.2g FLOP/B\n%s"
            % (gpu_resolved, f_theory, i_ridge, sub_i)
        )
        if uniform_x:
            ax.text(
                0.02,
                0.98,
                "All cells share one x (no NCU intensity in CSV).\nY still varies with prof_forward_mean_ms.",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.88),
            )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
        ax.grid(True, which="both", alpha=0.4)
        # Ensure compute ceiling line is always visible in the y-axis range.
        cur_ylim = ax.get_ylim()
        ax.set_ylim(top=max(cur_ylim[1], args.peak_tflops * 1e3 * 1.5))
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)

    print(
        "Wrote %s (%d cells, intensity=%s, NCU_pts=%d, F_theory=%.4g, gpu=%s)"
        % (out_path, int(ok.sum()), args.intensity_source, n_ncu_pts, f_theory, gpu_resolved)
    )


if __name__ == "__main__":
    main()
