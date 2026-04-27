"""
PointPillar energy plots.

**runs.csv mode (default)** — same input as ``plot_latency_breakdown.py``:
  - Default ``runs.csv``: local modal v2 aggregate (``modal_v2_allcells_full/runs.csv`` if present,
    else ``modal_v2/runs.csv``). Pass ``--csv path/to/runs.csv`` **or** the matrix directory
    (``.../modal_v2_allcells_full`` → ``runs.csv`` inside it), or use ``--runs-root`` for the same.
  - ``--gpu``: filter ``runs.csv``, latest row per ``experiment_cell_id``.
  - Per-frame energy: ``energy_total_J / measured_steps`` (Joules per sample).
  - Pipeline stages: same profile buckets as the latency breakdown; each stage's
    energy share = (stage mean ms / sum of stage ms) * per-frame energy (mJ).
  - Forward substages (PillarVFE / scatter / BEV / anchor): **on by default** via
    ``artifacts/forward_nvtx_ms.json`` next to each cell (modal_v2 layout). Disable with
    ``--no-nest-forward-from-artifacts``. Use ``--forward-nvtx-json`` when every row should
    share one local JSON (same NVTX split for all bars).
  - Output: ``energy_stacked_<gpu_slug>.png`` under ``--output_dir``.

**Legacy mode** (``--legacy``): Figure 8–style Total / DRAM / MAC breakdown
  from integrated ``energy_samples.csv`` + nsys ``report.sqlite`` (FP32 vs FP16).

Usage (from MLS repo root):
    python report/plot_energy.py --gpu A10

    python report/plot_energy.py --gpu A10 --runs-root modal_mls_results/modal_v2_allcells_full

    python report/plot_energy.py --gpu A10 --no-nest-forward-from-artifacts

    python report/plot_energy.py --legacy \\
        --fp32_csv  profile_outputs/amp_benchmark/energy_fp32/energy_samples.csv \\
        --fp16_csv  profile_outputs/amp_benchmark/energy_fp16_amp/energy_samples.csv \\
        --nsys_db   profile_outputs/nsys_baseline/report.sqlite \\
        --fp32_steps 50 --fp16_steps 50 \\
        --output_dir profile_outputs
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from plot_latency import (
        COLOR_BY_STAGE,
        FORWARD_NVTX_COLORS,
        FORWARD_NVTX_HATCH,
        FORWARD_NVTX_HATCH_LINEWIDTH,
        FORWARD_NVTX_LEGEND_LABEL,
        LEGEND_ORDER,
        M5_FORWARD_HATCH,
        MLSYS_PLOT_RC,
        POINTPILLAR_FORWARD_NVTX,
        STACKED_BARH_FIG_WIDTH,
        _slug_for_filename,
        list_distinct_gpus,
        load_filtered_latest,
        normalize_runs_csv_path,
        load_forward_nvtx_fractions_ordered,
        resolve_forward_nvtx_json_path,
        sort_by_m_group,
        stage_matrix,
        stack_order_for_cell,
    )
except ImportError:
    from plot_latency_breakdown import (
        COLOR_BY_STAGE,
        FORWARD_NVTX_COLORS,
        FORWARD_NVTX_HATCH,
        FORWARD_NVTX_HATCH_LINEWIDTH,
        FORWARD_NVTX_LEGEND_LABEL,
        LEGEND_ORDER,
        M5_FORWARD_HATCH,
        MLSYS_PLOT_RC,
        POINTPILLAR_FORWARD_NVTX,
        STACKED_BARH_FIG_WIDTH,
        _slug_for_filename,
        list_distinct_gpus,
        load_filtered_latest,
        normalize_runs_csv_path,
        load_forward_nvtx_fractions_ordered,
        resolve_forward_nvtx_json_path,
        sort_by_m_group,
        stage_matrix,
        stack_order_for_cell,
    )

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_FULL = _REPO_ROOT / "modal_mls_results" / "modal_v2_allcells_full" / "runs.csv"
_V2 = _REPO_ROOT / "modal_mls_results" / "modal_v2" / "runs.csv"
_V4_A10 = _REPO_ROOT / "modal_outputs" / "modal_v4_a10"
_V3_A10 = _REPO_ROOT / "modal_outputs" / "modal_v3_a10"
DEFAULT_RUNS_CSV = _V4_A10 if (_V4_A10 / "runs.csv").is_file() else (_V2_FULL if _V2_FULL.is_file() else _V2)
DEFAULT_FORWARD_NVTX_ROOT = _V3_A10
DEFAULT_M5_SOURCE_CSV = _V3_A10 / "runs.csv"
DEFAULT_GPU = "A10"
M2_KEEP_ONLY = {"M2_FP32_mem_both", "M2_AMP_mem_both"}
ENERGY_AXIS_MAX_MJ = 3500.0

# Match report/plot_latency.py visual defaults
LEGEND_FONTSIZE = 22
LEGEND_TITLE_FONTSIZE = 22
LEGEND_FONTSIZE_FWD = 22
LEGEND_BORDER_COLOR = "black"
LEGEND_BORDER_WIDTH = 1.0
LEGEND_BOX_LEFT_X = 0.81


def _style_legend_border_black(leg) -> None:
    fr = leg.get_frame()
    fr.set_edgecolor(LEGEND_BORDER_COLOR)
    fr.set_linewidth(LEGEND_BORDER_WIDTH)


def _style_legend_left_aligned(leg) -> None:
    leg._legend_box.align = "left"  # noqa: SLF001
    leg.get_title().set_ha("left")

_TOOLS = _REPO_ROOT / "OpenPCDet" / "tools"
if _TOOLS.is_dir() and str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
try:
    from nsys_nvtx_breakdown import POINTPILLAR_NVTX_STAGES  # noqa: E402
except ImportError:
    POINTPILLAR_NVTX_STAGES = (
        "PillarVFE",
        "PointPillarScatter",
        "BaseBEVBackbone",
        "AnchorHeadSingle",
        "post_processing",
    )

# ── Hardware energy constants (RTX 3080 Ti / GDDR6X) ─────────────────────────
# GDDR6X: ~14 pJ/bit => 1.75 nJ/byte
GDDR6X_J_PER_BYTE = 1.75e-9
# TF32 Tensor Core: ~0.4 pJ/MAC
TF32_J_PER_MAC = 4e-13
# FP16 Tensor Core: ~0.2 pJ/MAC
FP16_J_PER_MAC = 2e-13

STAGE_ORDER = list(POINTPILLAR_NVTX_STAGES)

# Approximate parameter sizes per stage (FP32 bytes)
PARAM_BYTES_FP32 = {
    "PillarVFE":          0.3e6 * 4,
    "PointPillarScatter": 0,
    "BaseBEVBackbone":    4.9e6 * 4,
    "AnchorHeadSingle":   0.4e6 * 4,
    "post_processing":    0,
}

# Approximate FLOPs per stage
STAGE_GMACS = {
    "PillarVFE":          1.5e9,
    "PointPillarScatter": 0.05e9,
    "BaseBEVBackbone":   18.0e9,
    "AnchorHeadSingle":   1.2e9,
    "post_processing":    0.05e9,
}

COLORS = {
    "inputs":  "#4C72B0",
    "outputs": "#DD8452",
    "weights": "#55A868",
    "mac":     "#C4A000",
}

GROUPS = {
    "Total": ["inputs", "outputs", "weights", "mac"],
    "DRAM":  ["inputs", "outputs", "weights"],
    "MAC":   ["mac"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="PointPillar energy plots: runs.csv (default) or legacy nsys+energy_samples."
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use energy_samples CSV + nsys sqlite (FP32 vs FP16) instead of runs.csv",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RUNS_CSV,
        help="runs.csv path or directory containing it (runs mode; default: modal_outputs/modal_v4_a10 if present)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Matrix bundle directory (must contain runs.csv); overrides --csv when set",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=DEFAULT_GPU,
        help="runs mode: gpu_name filter (same rules as plot_latency_breakdown.py). Default: %(default)s",
    )
    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="runs mode: print distinct gpu_name values from --csv and exit",
    )
    parser.add_argument(
        "--nest-forward-from-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="runs mode: split forward bar by forward_nvtx_ms.json (default: true; modal_v2 per-cell artifacts)",
    )
    parser.add_argument(
        "--forward-nvtx-json",
        type=Path,
        default=None,
        help="runs mode: path to forward_nvtx_ms.json when CSV has no artifacts_forward_nvtx_json "
        "(e.g. from nsys_nvtx_breakdown.write_forward_nvtx_json after local nsys export)",
    )
    parser.add_argument(
        "--forward-nvtx-root",
        type=Path,
        default=DEFAULT_FORWARD_NVTX_ROOT,
        help="runs mode: root directory containing <cell>/artifacts/forward_nvtx_ms.json "
        "(default: modal_outputs/modal_v3_a10)",
    )
    parser.add_argument(
        "--m5-source-csv",
        type=Path,
        default=DEFAULT_M5_SOURCE_CSV,
        help="runs mode: use M5 rows from this runs.csv (default: modal_outputs/modal_v3_a10/runs.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="runs mode: output PNG path (default: energy_stacked_<gpu>.png under --output_dir)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=176,
        help="PNG resolution (runs mode; default tuned for hatch + outside legend)",
    )
    parser.add_argument(
        "--fp32_csv",
        default="profile_outputs/amp_benchmark/energy_fp32/energy_samples.csv",
        help="legacy: energy_samples.csv from FP32 energy_monitor.py run",
    )
    parser.add_argument(
        "--fp16_csv",
        default="profile_outputs/amp_benchmark/energy_fp16_amp/energy_samples.csv",
        help="legacy: energy_samples.csv from FP16 AMP energy_monitor.py run",
    )
    parser.add_argument(
        "--nsys_db",
        default="profile_outputs/nsys_baseline/report.sqlite",
        help="legacy: nsys report.sqlite for NVTX stage timings",
    )
    parser.add_argument(
        "--fp32_steps", type=int, default=50, help="legacy: measured FP32 inference steps"
    )
    parser.add_argument(
        "--fp16_steps", type=int, default=50, help="legacy: measured FP16 inference steps"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (runs default: this script's directory; legacy: profile_outputs)",
    )
    return parser.parse_args()


def _existing_json_path(candidate: Path | None, repo_root: Path) -> Path | None:
    if candidate is None:
        return None
    p = candidate.expanduser()
    if p.is_file():
        return p.resolve()
    q = (repo_root / p).resolve()
    return q if q.is_file() else None


def _existing_dir_path(candidate: Path | None, repo_root: Path) -> Path | None:
    if candidate is None:
        return None
    p = candidate.expanduser()
    if p.is_dir():
        return p.resolve()
    q = (repo_root / p).resolve()
    return q if q.is_dir() else None


def integrate_csv(csv_path):
    """Integrate power over time from energy_samples.csv -> total Joules."""
    timestamps, powers = [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp_s']))
            powers.append(float(row['power_W']))
    ts = np.array(timestamps)
    pw = np.array(powers)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz(pw, ts))


def load_nsys_data(db_path):
    """Query nsys sqlite for stage fractions and DRAM bytes. Returns (stage_frac, total_dram_bytes)."""
    con = sqlite3.connect(db_path)

    # NVTX stage durations (skip first iter as warmup)
    _names = ", ".join("'%s'" % s for s in STAGE_ORDER)
    raw = con.execute(
        """
        SELECT text, (end-start)/1e6
        FROM NVTX_EVENTS
        WHERE text IN (%s)
          AND end IS NOT NULL AND end > start
        ORDER BY start
        """
        % _names
    ).fetchall()

    per_iter = defaultdict(list)
    for name, ms in raw:
        per_iter[name].append(ms)
    for k in per_iter:
        per_iter[k] = per_iter[k][1:]  # drop first (JIT warmup)

    stage_ms = {s: np.mean(per_iter[s]) if per_iter[s] else 1.0 for s in STAGE_ORDER}
    total_ms = sum(stage_ms.values())
    stage_frac = {s: stage_ms[s] / total_ms for s in STAGE_ORDER}

    # DRAM bytes from memcpy (HtoD=1, DtoH=2, DtoD=8)
    memcpy_rows = con.execute(
        "SELECT copyKind, SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
    ).fetchall()
    con.close()

    bytes_by_kind = {k: b for k, b in memcpy_rows}
    n_profile_frames = 25
    total_dram_bytes = (
        bytes_by_kind.get(1, 0) +
        bytes_by_kind.get(8, 0) +
        bytes_by_kind.get(2, 0)
    ) / n_profile_frames

    return stage_frac, total_dram_bytes


def decompose_energy(total_E_J, precision, stage_frac, total_dram_bytes):
    """
    Returns dict: stage -> {inputs, outputs, weights, mac} in mJ.
    Distributes measured total energy across stages and operand types using
    FLOPs fractions (MAC) and memory traffic fractions (DRAM).
    """
    j_per_mac = FP16_J_PER_MAC if precision == "fp16" else TF32_J_PER_MAC
    pb = {k: v / 2 for k, v in PARAM_BYTES_FP32.items()} if precision == "fp16" else PARAM_BYTES_FP32

    raw_mac_J  = sum(STAGE_GMACS.values()) * j_per_mac
    raw_dram_J = total_dram_bytes * GDDR6X_J_PER_BYTE

    accounted = raw_mac_J + raw_dram_J
    scale = min(total_E_J / accounted, 4.0) if accounted > 0 else 1.0

    mac_J_total  = raw_mac_J  * scale
    dram_J_total = raw_dram_J * scale

    total_flops = sum(STAGE_GMACS.values())
    total_pb    = sum(pb.values()) + 1e-9

    result = {}
    for s in STAGE_ORDER:
        mac_frac     = STAGE_GMACS[s] / total_flops
        act_bytes    = total_dram_bytes * stage_frac[s]
        weight_bytes = pb[s]
        dram_frac    = (act_bytes + weight_bytes) / (total_dram_bytes + total_pb)

        mac_J  = mac_J_total  * mac_frac
        dram_J = dram_J_total * dram_frac

        w_share   = weight_bytes / (act_bytes + weight_bytes) if (act_bytes + weight_bytes) > 0 else 0.0
        act_share = 1.0 - w_share

        result[s] = {
            "inputs":  dram_J * act_share * 0.60 * 1e3,   # mJ
            "outputs": dram_J * act_share * 0.40 * 1e3,
            "weights": dram_J * w_share          * 1e3,
            "mac":     mac_J                     * 1e3,
        }
    return result


def agg(en_dict, cols):
    return sum(en_dict[s][c] for s in STAGE_ORDER for c in cols)


def _energy_mj_per_stage_row(
    e_j_per_sample: float,
    row_ms: pd.Series,
    order: list[str],
) -> dict[str, float]:
    """Split per-sample energy (J) across pipeline stages using profile mean-ms shares."""
    t = 0.0
    for s in order:
        v = float(row_ms[s]) if s in row_ms.index else 0.0
        if np.isfinite(v) and v > 0:
            t += v
    if t <= 0 or not np.isfinite(e_j_per_sample) or e_j_per_sample <= 0:
        return {s: 0.0 for s in order}
    scale = e_j_per_sample * 1000.0 / t
    out: dict[str, float] = {}
    for s in order:
        v = float(row_ms[s]) if s in row_ms.index else 0.0
        v = v if np.isfinite(v) else 0.0
        out[s] = max(v, 0.0) * scale
    return out


def plot_runs_energy_stacked(
    df: pd.DataFrame,
    gpu_resolved: str,
    out_path: Path,
    dpi: int,
    *,
    nest_forward_from_artifacts: bool = False,
    repo_root: Path | None = None,
    runs_csv: Path | None = None,
    forward_nvtx_json_fallback: Path | None = None,
    forward_nvtx_root: Path | None = None,
) -> None:
    """Horizontal stacked bars: mJ per sample per stage (same stages as plot_latency_breakdown)."""
    labels = df["experiment_cell_id"].astype(str).str.replace("_mem_both", "", regex=False).tolist()
    mat_ms = stage_matrix(df).reindex(df.index)

    etot = pd.to_numeric(df["energy_total_J"], errors="coerce")
    steps = pd.to_numeric(df["measured_steps"], errors="coerce").replace(0, np.nan)
    e_j = etot / steps

    fig_h = 1.8 * max(6.0, 0.35 * len(labels) + 2.0)
    root = repo_root or _REPO_ROOT
    fb_json = _existing_json_path(forward_nvtx_json_fallback, root)
    nvtx_root = _existing_dir_path(forward_nvtx_root, root)

    pipe_legend_handles = [
        mpatches.Patch(
            facecolor=COLOR_BY_STAGE[s],
            edgecolor="#5a5a5a",
            linewidth=0.35,
            label=s.replace("_", " "),
        )
        for s in LEGEND_ORDER
    ]

    drew_forward_nvtx = False
    tried_nvtx_paths: list[Path] = []

    with plt.rc_context(MLSYS_PLOT_RC):
        fig, ax = plt.subplots(figsize=(STACKED_BARH_FIG_WIDTH, fig_h))

        if nest_forward_from_artifacts and fb_json is not None and len(labels) > 1:
            print(
                "plot_energy: using --forward-nvtx-json for all rows (%d cells); "
                "ensure the same NVTX split applies to each bar."
                % len(labels),
                file=sys.stderr,
            )

        xmax = 0.0
        for yi, lab in enumerate(labels):
            order = stack_order_for_cell(lab)
            ej = float(e_j.iloc[yi]) if np.isfinite(e_j.iloc[yi]) else float("nan")
            stages = _energy_mj_per_stage_row(ej, mat_ms.iloc[yi], order)
            left = 0.0
            m5_row = str(lab).startswith("M5_")
            frac = None
            sub_order = list(POINTPILLAR_FORWARD_NVTX)
            nvtx_path: Path | None = None
            if nest_forward_from_artifacts and not m5_row:
                nvtx_path = resolve_forward_nvtx_json_path(df.iloc[yi], root, runs_csv=runs_csv)
                if nvtx_path is None and nvtx_root is not None:
                    cell = str(df.iloc[yi].get("experiment_cell_id", "") or "").strip()
                    if cell:
                        cand = nvtx_root / cell / "artifacts" / "forward_nvtx_ms.json"
                        if cand.is_file():
                            nvtx_path = cand.resolve()
                if nvtx_path is None:
                    nvtx_path = fb_json
                if nvtx_path is not None:
                    tried_nvtx_paths.append(nvtx_path)
                    frac, sub_order = load_forward_nvtx_fractions_ordered(nvtx_path)
            for name in order:
                if name == "Forward" and frac is not None:
                    fwd_e = stages["Forward"]
                    sub_total = sum(float(frac.get(sub, 0.0)) for sub in POINTPILLAR_FORWARD_NVTX)
                    if sub_total < 1e-9 and fwd_e > 0:
                        ax.barh(
                            lab,
                            fwd_e,
                            left=left,
                            color=COLOR_BY_STAGE["Forward"],
                            height=0.7,
                            edgecolor="white",
                            linewidth=0.65,
                        )
                        left += fwd_e
                    else:
                        for sub in sub_order:
                            w = fwd_e * float(frac.get(sub, 0.0))
                            if w <= 0:
                                continue
                            drew_forward_nvtx = True
                            bars = ax.barh(
                                lab,
                                w,
                                left=left,
                                facecolor=FORWARD_NVTX_COLORS.get(sub, COLOR_BY_STAGE["Forward"]),
                                hatch=FORWARD_NVTX_HATCH.get(sub, ""),
                                height=0.7,
                                edgecolor="#f5f5f0",
                                linewidth=1.0,
                            )
                            for patch in bars:
                                patch.set_hatch_linewidth(FORWARD_NVTX_HATCH_LINEWIDTH)
                            left += w
                    continue
                val = stages[name]
                if m5_row and name == "Forward":
                    bars = ax.barh(
                        lab,
                        val,
                        left=left,
                        facecolor=COLOR_BY_STAGE[name],
                        hatch=M5_FORWARD_HATCH,
                        height=0.7,
                        edgecolor="white",
                        linewidth=0.65,
                    )
                    for patch in bars.patches:
                        patch.set_hatch_linewidth(FORWARD_NVTX_HATCH_LINEWIDTH)
                else:
                    ax.barh(
                        lab,
                        val,
                        left=left,
                        color=COLOR_BY_STAGE[name],
                        height=0.7,
                        edgecolor="white",
                        linewidth=0.65,
                    )
                left += val
            xmax = max(xmax, left)

        if nest_forward_from_artifacts and drew_forward_nvtx:
            fwd_legend_handles = []
            for sub in POINTPILLAR_FORWARD_NVTX:
                leg_patch = mpatches.Patch(
                    facecolor=FORWARD_NVTX_COLORS.get(sub, COLOR_BY_STAGE["Forward"]),
                    edgecolor="#f5f5f0",
                    linewidth=1.0,
                    hatch=FORWARD_NVTX_HATCH.get(sub, ""),
                    label=FORWARD_NVTX_LEGEND_LABEL.get(sub, sub),
                )
                leg_patch.set_hatch_linewidth(FORWARD_NVTX_HATCH_LINEWIDTH)
                fwd_legend_handles.append(leg_patch)
            leg_forward = ax.legend(
                handles=fwd_legend_handles,
                loc="lower left",
                bbox_to_anchor=(LEGEND_BOX_LEFT_X, 0.02),
                borderaxespad=0.35,
                fontsize=LEGEND_FONTSIZE_FWD,
                framealpha=1.0,
                title="Forward Substage",
                title_fontsize=LEGEND_FONTSIZE_FWD,
            )
            _style_legend_border_black(leg_forward)
            _style_legend_left_aligned(leg_forward)
            ax.add_artist(leg_forward)
            leg_pipe = ax.legend(
                handles=pipe_legend_handles,
                loc="lower left",
                bbox_to_anchor=(LEGEND_BOX_LEFT_X, 0.31),
                borderaxespad=0.35,
                fontsize=LEGEND_FONTSIZE,
                framealpha=1.0,
                title="Stage",
                title_fontsize=LEGEND_TITLE_FONTSIZE,
            )
            _style_legend_border_black(leg_pipe)
            _style_legend_left_aligned(leg_pipe)
            ax.add_artist(leg_pipe)
            tight_rect = [0.0, 0.0, 1.0, 1.0]
        else:
            if nest_forward_from_artifacts:
                paths_hint = (
                    ", ".join(sorted({str(p) for p in tried_nvtx_paths}))
                    if tried_nvtx_paths
                    else "(no JSON path resolved; check --forward-nvtx-json or CSV paths)"
                )
                print(
                    "plot_energy: --nest-forward-from-artifacts but no forward NVTX sub-bars were drawn. "
                    "Typical causes: JSON missing; fraction_of_forward_sum / mean_ms all zero; or keys not in "
                    "%s. Paths used: %s"
                    % (list(POINTPILLAR_FORWARD_NVTX), paths_hint),
                    file=sys.stderr,
                )
            leg_stage = ax.legend(
                handles=pipe_legend_handles,
                loc="lower right",
                bbox_to_anchor=(0.99, 0.02),
                borderaxespad=0.35,
                fontsize=LEGEND_FONTSIZE,
                framealpha=0.92,
                title="Stage",
                title_fontsize=LEGEND_TITLE_FONTSIZE,
            )
            _style_legend_border_black(leg_stage)
            _style_legend_left_aligned(leg_stage)
            tight_rect = [0.0, 0.0, 1.0, 1.0]

        ax.set_xlabel("Energy per Sample [mJ]")
        ax.set_xlim(0, ENERGY_AXIS_MAX_MJ)
        ax.invert_yaxis()
        fig.tight_layout(rect=tight_rect)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
    print("Saved → %s (%d cells)" % (out_path, len(labels)))


def plot_precision_bars(en_fp32, en_fp16, E_fp32_J, E_fp16_J, output_dir):
    """Figure 1: stacked bars by precision (FP32 vs FP16), grouped by Total/DRAM/MAC."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    fig.suptitle("PointPillar Energy Breakdown  ·  FP32 vs FP16 AMP",
                 fontsize=13, fontweight="bold", y=1.02)

    bar_w = 0.55
    x = np.array([0, 1])
    datasets = [en_fp32, en_fp16]
    prec_labels = ["FP32", "FP16 AMP"]

    for ax, (group_name, components) in zip(axes, GROUPS.items()):
        bottoms = np.zeros(2)
        for comp in components:
            vals = np.array([agg(d, [comp]) for d in datasets])
            ax.bar(x, vals, bar_w, bottom=bottoms,
                   color=COLORS[comp], edgecolor="white", linewidth=0.6)
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 0.003:
                    ax.text(xi, b + v / 2, f"{v:.3f}",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(prec_labels, fontsize=11)
        ax.set_xlabel("Precision", fontsize=10)
        ax.set_title(group_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy (mJ)" if group_name == "Total" else "")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=COLORS[c], label=c.capitalize())
               for c in ["inputs", "outputs", "weights", "mac"]]
    fig.legend(handles=handles, title="Operand", loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=False)

    annot = (f"Measured:  FP32 = {E_fp32_J*1e3:.1f} mJ/frame"
             f"  |  FP16 AMP = {E_fp16_J*1e3:.1f} mJ/frame\n"
             f"Energy reduction: {(1 - E_fp16_J/E_fp32_J)*100:.1f}%  "
             f"({E_fp32_J/E_fp16_J:.2f}× less energy per frame)")
    fig.text(0.5, -0.14, annot, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.85))

    plt.tight_layout()
    out = str(Path(output_dir) / "pointpillar_energy_breakdown.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def plot_stage_bars(en_fp32, en_fp16, output_dir):
    """Figure 2: stacked bars by pipeline stage, FP32 vs FP16 side-by-side."""
    short = {
        "PillarVFE":          "PillarVFE",
        "PointPillarScatter": "PPScatter",
        "BaseBEVBackbone":    "BEVBackbone",
        "AnchorHeadSingle":   "AnchorHead",
        "post_processing":    "PostProc",
    }

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
    fig2.suptitle("PointPillar Energy by Pipeline Stage  ·  FP32 vs FP16 AMP",
                  fontsize=13, fontweight="bold", y=1.02)

    xs = np.arange(len(STAGE_ORDER))
    w2 = 0.35

    for ax2, (group_name, components) in zip(axes2, GROUPS.items()):
        for di, (en_dict, offset) in enumerate([(en_fp32, -w2/2), (en_fp16, w2/2)]):
            bottoms = np.zeros(len(STAGE_ORDER))
            for comp in components:
                vals = np.array([en_dict[s][comp] for s in STAGE_ORDER])
                hatch = "" if di == 0 else "//"
                ax2.bar(xs + offset, vals, w2, bottom=bottoms,
                        color=COLORS[comp], alpha=0.85 if di == 0 else 0.6,
                        hatch=hatch, edgecolor="white", linewidth=0.5)
                bottoms += vals

        ax2.set_xticks(xs)
        ax2.set_xticklabels([short[s] for s in STAGE_ORDER],
                            rotation=25, ha="right", fontsize=8.5)
        ax2.set_title(group_name, fontsize=12, fontweight="bold")
        ax2.set_ylabel("Energy (mJ)" if group_name == "Total" else "")
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)

    op_handles = [mpatches.Patch(color=COLORS[c], label=c.capitalize())
                  for c in ["inputs", "outputs", "weights", "mac"]]
    prec_handles = [mpatches.Patch(facecolor="gray", label="FP32 (solid)"),
                    mpatches.Patch(facecolor="gray", hatch="//", alpha=0.6, label="FP16 AMP (hatch)")]
    fig2.legend(handles=op_handles + prec_handles, loc="lower center",
                ncol=6, bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)

    plt.tight_layout()
    out2 = str(Path(output_dir) / "pointpillar_energy_per_stage.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
    plt.close()


def main_runs(args: argparse.Namespace) -> None:
    try:
        if args.runs_root is not None:
            csv_path = normalize_runs_csv_path(args.runs_root)
        else:
            csv_path = normalize_runs_csv_path(args.csv)
    except FileNotFoundError as ex:
        print("%s" % ex, file=sys.stderr)
        sys.exit(1)

    if args.list_gpus:
        try:
            names = list_distinct_gpus(csv_path)
        except Exception as ex:
            print("Failed to read %s: %s" % (csv_path, ex), file=sys.stderr)
            sys.exit(1)
        for n in names:
            print(n)
        return

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else Path(__file__).resolve().parent
    )
    out_path = (
        args.out.resolve()
        if args.out
        else out_dir / ("energy_stacked_%s.png" % _slug_for_filename(args.gpu))
    )

    df, gpu_resolved = load_filtered_latest(csv_path, args.gpu)
    keep_mask = df["experiment_cell_id"].astype(str).apply(
        lambda c: (not c.startswith("M2_")) or (c in M2_KEEP_ONLY)
    )
    df = df[keep_mask].copy()

    m5_source = Path(args.m5_source_csv).expanduser()
    if not m5_source.is_absolute():
        m5_source = (_REPO_ROOT / m5_source).resolve()
    else:
        m5_source = m5_source.resolve()
    if m5_source.is_file():
        try:
            m5_df, _ = load_filtered_latest(m5_source, args.gpu)
            m5_df = m5_df[m5_df["experiment_cell_id"].astype(str).str.startswith("M5_")].copy()
            if not m5_df.empty:
                base = df[~df["experiment_cell_id"].astype(str).str.startswith("M5_")].copy()
                df = pd.concat([base, m5_df], ignore_index=True)
        except Exception as ex:
            print("plot_energy: failed to load M5 source %s: %s" % (m5_source, ex), file=sys.stderr)

    if "energy_total_J" not in df.columns or "measured_steps" not in df.columns:
        raise ValueError("runs.csv must include energy_total_J and measured_steps columns")
    df = sort_by_m_group(df)
    plot_runs_energy_stacked(
        df,
        gpu_resolved,
        out_path,
        args.dpi,
        nest_forward_from_artifacts=bool(args.nest_forward_from_artifacts),
        repo_root=_REPO_ROOT,
        runs_csv=csv_path,
        forward_nvtx_json_fallback=getattr(args, "forward_nvtx_json", None),
        forward_nvtx_root=getattr(args, "forward_nvtx_root", None),
    )


def main_legacy(args: argparse.Namespace) -> None:
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else Path("profile_outputs").resolve()
    )

    print("Reading FP32 energy from %s" % args.fp32_csv)
    total_energy_fp32 = integrate_csv(args.fp32_csv)
    E_fp32_J = total_energy_fp32 / args.fp32_steps
    print(
        "  Total: %.2f J over %d steps -> %.2f mJ/frame"
        % (total_energy_fp32, args.fp32_steps, E_fp32_J * 1e3)
    )

    print("Reading FP16 energy from %s" % args.fp16_csv)
    total_energy_fp16 = integrate_csv(args.fp16_csv)
    E_fp16_J = total_energy_fp16 / args.fp16_steps
    print(
        "  Total: %.2f J over %d steps -> %.2f mJ/frame"
        % (total_energy_fp16, args.fp16_steps, E_fp16_J * 1e3)
    )

    print("Reading nsys data from %s" % args.nsys_db)
    stage_frac, total_dram_bytes = load_nsys_data(args.nsys_db)
    print("  Stage fractions: %s" % {k: "%.3f" % v for k, v in stage_frac.items()})
    print("  Total DRAM bytes/frame: %.2f MB" % (total_dram_bytes / 1e6))

    en_fp32 = decompose_energy(E_fp32_J, "fp32", stage_frac, total_dram_bytes)
    en_fp16 = decompose_energy(E_fp16_J, "fp16", stage_frac, total_dram_bytes)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_precision_bars(en_fp32, en_fp16, E_fp32_J, E_fp16_J, str(output_dir))
    plot_stage_bars(en_fp32, en_fp16, str(output_dir))


def main():
    args = parse_args()
    if args.legacy:
        main_legacy(args)
    else:
        try:
            main_runs(args)
        except Exception as ex:
            print("%s" % ex, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
