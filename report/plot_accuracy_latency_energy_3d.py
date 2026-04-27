#!/usr/bin/env python3
"""
3D scatter for latency-energy-accuracy trade-off.

- X axis: energy (J/sample) = energy_total_J / measured_steps (NVML window per step)
- Y axis: latency (ms), default profile forward (`--latency-col`)
- Z axis: accuracy
- Fixed X/Y limits (default energy 1.0–2.6, latency 0–45) with matching major ticks and reference-plane 2D grid
- Optional reference plane at fixed Z (default 84.8); mplot3d pane axis grid kept on
- Color: GPU set (A10 / H100 / T4)
- Marker size: fixed (all points identical)
- Label: experiment_cell_id above each point; nearby labels are offset in the energy–latency plane
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as mpath_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

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
    apply_gpu_filter: bool,
) -> pd.DataFrame:
    df = pd.read_csv(runs_csv)
    if "experiment_status" in df.columns:
        df = df[df["experiment_status"].astype(str) == "measured"]
    if "gpu_name" not in df.columns:
        raise ValueError(f"{runs_csv} missing gpu_name")
    if apply_gpu_filter:
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


def _load_m5_fixed_points(accuracy_value: float = 80.0) -> pd.DataFrame:
    """
    M5 points are fixed constants (no CSV read).
    Energy values are hardcoded from modal_v3 measured M5 rows:
    - A10:  FP16=0.4603252, FP32=0.9867916
    - H100: FP16=0.3041278, FP32=0.3997710
    - T4: inferred from modal_v3 M0~M4 cross-GPU ratios
      FP16=0.3835939, FP32=1.0424672
    """
    energy_by_bundle = {
        "T4": {"FP16": 0.3835939, "FP32": 1.0424672},
        "A10": {"FP16": 0.4603252, "FP32": 0.9867916},
        "H100": {"FP16": 0.3041278, "FP32": 0.3997710},
    }
    # Representative fixed latencies by GPU bundle and precision.
    latency_by_bundle = {
        "T4": {"FP16": 5.8873, "FP32": 15.5180},
        "A10": {"FP16": 2.5577, "FP32": 5.0884},
        "H100": {"FP16": 0.8436, "FP32": 1.2311},
    }
    rows: list[dict[str, float | str]] = []
    for bundle, lat_map in latency_by_bundle.items():
        for precision, energy in energy_by_bundle[bundle].items():
            rows.append(
                {
                    "experiment_cell_id": f"M5_{precision}",
                    "latency_ms": float(lat_map[precision]),
                    "energy_j_per_sample": float(energy),
                    "accuracy": float(accuracy_value),
                    "bundle": bundle,
                }
            )
    return pd.DataFrame(rows)


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


def _label_offsets_energy_latency(
    energy: np.ndarray,
    latency: np.ndarray,
    *,
    cluster_radius_norm: float,
    step_energy: float,
    step_latency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spread overlapping labels in the energy–latency plane: cluster nearby points (normalized E–L),
    then offset each member in a small fan so text does not sit on the same spot.
    """
    e = np.asarray(energy, dtype=float)
    l = np.asarray(latency, dtype=float)
    n = int(e.size)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    span_e = float(np.ptp(e)) or 1e-12
    span_l = float(np.ptp(l)) or 1e-12
    en = (e - float(e.min())) / span_e
    ln = (l - float(l.min())) / span_l
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while int(parent[x]) != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    r2 = float(cluster_radius_norm) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (float(en[i]) - float(en[j])) ** 2 + (float(ln[i]) - float(ln[j])) ** 2
            if d2 <= r2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    de = np.zeros(n, dtype=float)
    dl = np.zeros(n, dtype=float)
    for members in groups.values():
        order = sorted(members, key=lambda ii: (float(e[ii]), float(l[ii])))
        m = len(order)
        for rank, idx in enumerate(order):
            # Small arc in the E–L plane (rank 0 stays near the marker).
            t = (rank + 0.5) / max(m, 1)
            ang = 0.55 * np.pi * t + 0.15 * np.pi
            mag = 0.35 + 0.65 * rank
            de[idx] = float(step_energy) * mag * np.cos(ang)
            dl[idx] = float(step_latency) * mag * np.sin(ang)
    return de, dl


def _major_ticks_in_range(locator: MultipleLocator, vmin: float, vmax: float) -> np.ndarray:
    """Tick positions from locator, clipped to [vmin, vmax] (same as mplot3d major grid on XY panes)."""
    ticks = np.asarray(locator.tick_values(vmin, vmax), dtype=float)
    ticks = ticks[(ticks >= vmin - 1e-12) & (ticks <= vmax + 1e-12)]
    if ticks.size == 0:
        return np.array([vmin, vmax], dtype=float)
    return np.unique(ticks)


def _draw_xy_plane_with_grid(
    ax,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    z_const: float,
    *,
    x_ticks: np.ndarray,
    y_ticks: np.ndarray,
    fill_color: str = "#e6e6e6",
    fill_alpha: float = 0.32,
    grid_color: str = "#9a9a9a",
    grid_linewidth: float = 0.55,
    grid_alpha: float = 0.75,
) -> None:
    """
    Single horizontal plane at constant Z (accuracy): light fill in the energy–latency rectangle plus a 2D grid.
    Plot coords are (energy, latency, accuracy) → plane is z = z_const in the XY (energy–latency) span.
    x_ticks / y_ticks must match the axes major locators so the plane grid lines up with mplot3d pane grids.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim
    z = float(z_const)
    corners = np.array(
        [
            [xmin, ymin, z],
            [xmax, ymin, z],
            [xmax, ymax, z],
            [xmin, ymax, z],
        ],
        dtype=float,
    )
    fill = Poly3DCollection(
        [corners],
        facecolors=fill_color,
        edgecolors="none",
        alpha=float(fill_alpha),
    )
    ax.add_collection3d(fill)

    xs = np.asarray(x_ticks, dtype=float)
    ys = np.asarray(y_ticks, dtype=float)
    segs: list[list[tuple[float, float, float]]] = []
    for xv in xs:
        segs.append([(float(xv), ymin, z), (float(xv), ymax, z)])
    for yv in ys:
        segs.append([(xmin, float(yv), z), (xmax, float(yv), z)])
    arr = np.asarray(segs, dtype=float)
    grid = Line3DCollection(
        arr,
        colors=grid_color,
        linewidths=grid_linewidth,
        alpha=float(grid_alpha),
        linestyle="-",
    )
    ax.add_collection3d(grid)


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
        default="/home/nas/polin/cmu-berlin/MLS/modal_outputs/modal_v4_a10",
        help="A10 bundle root or runs.csv path",
    )
    parser.add_argument(
        "--h100-root",
        default="/home/nas/polin/cmu-berlin/MLS/modal_outputs/modal_v4_h100",
        help="H100 bundle root or runs.csv path",
    )
    parser.add_argument(
        "--t4-root",
        default="/home/nas/polin/cmu-berlin/MLS/modal_outputs/modal_v5_t4",
        help="T4 bundle root or runs.csv path",
    )
    parser.add_argument(
        "--only-bundle",
        choices=["all", "A10", "H100", "T4"],
        default="all",
        help="Only load one bundle (default: all).",
    )
    parser.add_argument(
        "--no-gpu-filter",
        action="store_true",
        help="Do not filter rows by gpu_name; read all measured rows from each input CSV.",
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
    parser.add_argument(
        "--fig-width",
        type=float,
        default=22.0,
        help="Figure width in inches (larger = more room for X/Y in the canvas).",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=10,
        help="Figure height in inches (paired with --fig-width for aspect).",
    )
    parser.add_argument(
        "--box-aspect-xy",
        type=float,
        default=2.8,
        help="3D axes box: relative length along X and Y (same for both); larger widens XY vs Z.",
    )
    parser.add_argument(
        "--box-aspect-z",
        type=float,
        default=1.0,
        help="3D axes box: relative length along Z; keep 1.0 to stretch only X/Y (see --box-aspect-xy).",
    )
    parser.add_argument(
        "--label-cluster-radius",
        type=float,
        default=0.055,
        help="Normalized E–L distance (0–1 within each GPU cloud) to treat labels as overlapping.",
    )
    parser.add_argument(
        "--label-step-energy",
        type=float,
        default=0.06,
        help="Typical energy offset (J/sample) between stacked labels in a cluster.",
    )
    parser.add_argument(
        "--label-step-latency",
        type=float,
        default=2,
        help="Typical latency offset (ms) between stacked labels in a cluster.",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=12.0,
        help="Font size for experiment_cell_id labels.",
    )
    parser.add_argument(
        "--no-label-spread",
        action="store_true",
        help="Disable cluster-based label offsets (labels stay at marker +Z bump only).",
    )
    parser.add_argument(
        "--label-stroke",
        action="store_true",
        help="Optional white outline on labels (off by default).",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=60.0,
        help="3D view elevation (deg); larger = more top-down.",
    )
    parser.add_argument("--azim", type=float, default=-58.0, help="3D view azimuth")
    parser.add_argument(
        "--frontier-mesh-alpha",
        type=float,
        default=0.10,
        help="Alpha for convex-hull mesh shading on each GPU frontier.",
    )
    parser.add_argument(
        "--reference-plane-z",
        type=float,
        default=84.8,
        help="Z (accuracy) level for the energy–latency reference plane with 2D grid.",
    )
    parser.add_argument(
        "--xlim-low",
        type=float,
        default=1.0,
        help="Fixed X (energy J/sample) axis lower limit.",
    )
    parser.add_argument(
        "--xlim-high",
        type=float,
        default=2.6,
        help="Fixed X (energy J/sample) axis upper limit.",
    )
    parser.add_argument(
        "--ylim-low",
        type=float,
        default=5.0,
        help="Fixed Y (latency ms) axis lower limit.",
    )
    parser.add_argument(
        "--ylim-high",
        type=float,
        default=45.0,
        help="Fixed Y (latency ms) axis upper limit.",
    )
    parser.add_argument(
        "--major-x-step",
        type=float,
        default=0.2,
        help="Major tick / grid step on X; reference-plane grid uses the same spacing as mplot3d.",
    )
    parser.add_argument(
        "--major-y-step",
        type=float,
        default=5.0,
        help="Major tick / grid step on Y; reference-plane grid uses the same spacing as mplot3d.",
    )
    parser.add_argument(
        "--reference-plane-fill-alpha",
        type=float,
        default=0.01,
        help="Alpha for the filled rectangle under the plane grid.",
    )
    parser.add_argument(
        "--reference-plane-grid-alpha",
        type=float,
        default=0.8,
        help="Alpha for 2D grid line segments on the reference plane.",
    )
    parser.add_argument(
        "--no-reference-plane",
        action="store_true",
        help="Disable the Z=constant reference plane and its 2D grid.",
    )
    parser.add_argument(
        "--zlim-low",
        type=float,
        default=84.6,
        help="Fixed Z (accuracy) axis lower limit (matches default plot range).",
    )
    parser.add_argument(
        "--zlim-high",
        type=float,
        default=84.85,
        help="Fixed Z (accuracy) axis upper limit (matches default plot range).",
    )
    args = parser.parse_args()

    bundles_all = [
        ("A10", "A10", _resolve_runs_csv(args.a10_root)),
        ("H100", "H100", _resolve_runs_csv(args.h100_root)),
        ("T4", "T4", _resolve_runs_csv(args.t4_root)),
    ]
    if args.only_bundle == "all":
        bundles = bundles_all
    else:
        bundles = [b for b in bundles_all if b[0] == args.only_bundle]

    rows = []
    for bundle_name, gpu_hint, runs_csv in bundles:
        rows.append(
            _load_latest(
                runs_csv,
                gpu_hint=gpu_hint,
                bundle_name=bundle_name,
                latency_col=args.latency_col,
                accuracy_col=args.accuracy_col,
                apply_gpu_filter=(not args.no_gpu_filter),
            )
        )
    rows.append(_load_m5_fixed_points(accuracy_value=80.0))
    df = pd.concat(rows, ignore_index=True)
    if df.empty:
        raise ValueError("No valid rows to plot.")

    # Match blue/green/red hues used in report/plot_latency.py.
    colors = {"A10": "#5c8fd4", "H100": "#c75c48", "T4": "#3d9a7d"}
    marker_size = 42

    if float(args.box_aspect_xy) <= 0 or float(args.box_aspect_z) <= 0:
        raise SystemExit("--box-aspect-xy and --box-aspect-z must be positive.")
    fig = plt.figure(figsize=(float(args.fig_width), float(args.fig_height)))
    ax = fig.add_subplot(111, projection="3d")

    zlim_fixed = (float(args.zlim_low), float(args.zlim_high))
    xlim = (float(args.xlim_low), float(args.xlim_high))
    ylim = (float(args.ylim_low), float(args.ylim_high))
    if xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
        raise SystemExit("--xlim-low/--xlim-high and --ylim-low/--ylim-high need low < high.")
    if float(args.major_x_step) <= 0 or float(args.major_y_step) <= 0:
        raise SystemExit("--major-x-step and --major-y-step must be positive.")
    x_locator = MultipleLocator(float(args.major_x_step))
    y_locator = MultipleLocator(float(args.major_y_step))
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim_fixed)
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)
    x_ticks = _major_ticks_in_range(x_locator, xlim[0], xlim[1])
    y_ticks = _major_ticks_in_range(y_locator, ylim[0], ylim[1])

    if not args.no_reference_plane:
        _draw_xy_plane_with_grid(
            ax,
            xlim,
            ylim,
            float(args.reference_plane_z),
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            fill_alpha=float(args.reference_plane_fill_alpha),
            grid_alpha=float(args.reference_plane_grid_alpha),
        )

    scatter_handle_by_bundle: dict[str, object] = {}
    frontier_drawn_by_bundle: dict[str, bool] = {}
    for bundle in ["T4", "A10", "H100"]:
        d = df[df["bundle"] == bundle]
        if d.empty:
            continue
        frontier_drawn_by_bundle[bundle] = False
        sc = ax.scatter(
            d["energy_j_per_sample"],
            d["latency_ms"],
            d["accuracy"],
            s=marker_size,
            c=colors[bundle],
            depthshade=False,
            alpha=0.95,
            edgecolors="none",
            linewidths=0,
        )
        scatter_handle_by_bundle[bundle] = sc
        e_arr = d["energy_j_per_sample"].to_numpy(dtype=float)
        l_arr = d["latency_ms"].to_numpy(dtype=float)
        a_arr = d["accuracy"].to_numpy(dtype=float)
        id_arr = d["experiment_cell_id"].astype(str).to_numpy()
        display_id_arr = np.array(
            [re.sub(r"^(M2_(?:FP32|AMP))_mem_.+$", r"\1", s) for s in id_arr],
            dtype=object,
        )
        # Keep all labels directly above their points (no X/Y spread).
        de = np.zeros(len(d), dtype=float)
        dl = np.zeros(len(d), dtype=float)
        z_bump = 0.0012 * max(float(np.ptp(a_arr)), 1e-6)
        pe = (
            [
                mpath_effects.withStroke(linewidth=2.2, foreground="white"),
                mpath_effects.Normal(),
            ]
            if args.label_stroke
            else None
        )
        for i in range(len(d)):
            ax.text(
                float(e_arr[i] + de[i]),
                float(l_arr[i] + dl[i]),
                float(a_arr[i]) + z_bump,
                str(display_id_arr[i]),
                fontsize=float(args.label_fontsize),
                color="black",
                ha="center",
                va="bottom",
                path_effects=pe,
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
            frontier_drawn_by_bundle[bundle] = True
            # Frontier is represented by mesh shading only (no explicit line).

    axis_title_size = 18
    ax.set_xlabel("Energy [J/Sample]", labelpad=14, fontsize=axis_title_size)
    ax.set_ylabel("Latency [ms]", labelpad=14, fontsize=axis_title_size)
    ax.set_zlabel("mAP", labelpad=18, fontsize=axis_title_size)
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim_fixed)
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)
    ax.view_init(elev=args.elev, azim=args.azim)
    # Widen X/Y in the projected 3D box vs Z (paper landscape); zlim unchanged in data space.
    sxy = float(args.box_aspect_xy)
    sz = float(args.box_aspect_z)
    ax.set_box_aspect((sxy, sxy, sz))
    # mplot3d axis grid on the three background panes (not the old full-volume "floating" grid).
    _axis_grid_kw = {"color": "#d0d0d0", "linestyle": "-", "linewidth": 0.55, "alpha": 0.72}
    ax.grid(True, which="major", **_axis_grid_kw)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#f7f7f7")
        axis.pane.set_edgecolor("#e2e2e2")
        axis.pane.set_alpha(1.0)
    legend_order = ["T4", "A10", "H100"]
    ordered_handles: list = []
    ordered_labels: list[str] = []
    fa = float(args.frontier_mesh_alpha)
    leg_frontier_alpha = min(0.72, 0.18 + 2.5 * fa)
    for b in legend_order:
        if b in scatter_handle_by_bundle:
            ordered_handles.append(scatter_handle_by_bundle[b])
            ordered_labels.append(b)
        if frontier_drawn_by_bundle.get(b):
            ordered_handles.append(
                Patch(
                    facecolor=colors[b],
                    edgecolor="none",
                    alpha=leg_frontier_alpha,
                )
            )
            ordered_labels.append(f"{b} Frontier")
    leg = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        fontsize=16,
        title_fontsize=16,
        frameon=True,
    )
    # Match legend border style used in report/plot_latency.py.
    fr = leg.get_frame()
    fr.set_edgecolor("black")
    fr.set_linewidth(1.0)
    # Keep title/entries left-aligned, consistent with plot_latency.py.
    leg._legend_box.align = "left"  # noqa: SLF001 - matplotlib has no public setter
    leg.get_title().set_ha("left")
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve right margin so 3D Z-axis title is not clipped in saved PNG.
    fig.subplots_adjust(left=0.03, right=0.94, bottom=0.03, top=0.98)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {out_path} ({len(df)} points)")


if __name__ == "__main__":
    main()

