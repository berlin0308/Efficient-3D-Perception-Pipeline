#!/usr/bin/env python3
"""
Stacked horizontal bar chart: per-stage mean latency from research_matrix runs.csv.
Filters by GPU (--gpu substring on gpu_name), one row per experiment_cell_id (latest timestamp_iso).
No dataloader segment. M3/M4 stack: H2D then Preprocess; M0–M2: Preprocess then H2D.
CPU-voxel runs (M0–M2): profile_summary prints load_data_to_gpu as "data_to_gpu" → CSV prof_data_to_gpu_pts_* (not prof_H2D); that belongs in the H2D segment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_FULL_CSV = _REPO_ROOT / "modal_mls_results" / "modal_v2_allcells_full" / "runs.csv"
_V2_CSV = _REPO_ROOT / "modal_mls_results" / "modal_v2" / "runs.csv"
# Defaults match common workflow: v4 runs + v3 NVTX artifacts fallback + standard report name.
_DEFAULT_INPUT_DIR_V4 = _REPO_ROOT / "modal_outputs" / "modal_v4_a10"
_V2_DEFAULT = _V2_FULL_CSV if _V2_FULL_CSV.is_file() else _V2_CSV
DEFAULT_CSV = (
    _DEFAULT_INPUT_DIR_V4
    if (_DEFAULT_INPUT_DIR_V4 / "runs.csv").is_file()
    else _V2_DEFAULT
)
DEFAULT_NVTX_FALLBACK_DIR = _REPO_ROOT / "modal_outputs" / "modal_v3_a10"
DEFAULT_OUT_PNG = _REPO_ROOT / "report" / "latency_a10_v4.png"

_TOOLS = _REPO_ROOT / "OpenPCDet" / "tools"
if _TOOLS.is_dir() and str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
try:
    from nsys_nvtx_breakdown import POINTPILLAR_FORWARD_NVTX  # noqa: E402
except ImportError:
    POINTPILLAR_FORWARD_NVTX = (
        "PillarVFE",
        "PointPillarScatter",
        "BaseBEVBackbone",
        "AnchorHeadSingle",
    )

FORWARD_NVTX_LEGEND_LABEL = {
    "PillarVFE": "VFE",
    "PointPillarScatter": "Scatter",
    "BaseBEVBackbone": "Backbone",
    "AnchorHeadSingle": "Anchor Head",
}
# Distinct hatches so substages are separable; fill matches pipeline forward (see FORWARD_NVTX_COLORS below).
FORWARD_NVTX_HATCH = {
    # Swapped with M5_FORWARD_HATCH (M5 block uses ///; M0–M4 VFE uses \)
    "PillarVFE": "\\\\\\\\",
    # Not '-': can fail to show on narrow bars. '**' = star/cross hatching; distinct from +/|/\\.
    "PointPillarScatter": "**",
    "BaseBEVBackbone": "|||",
    "AnchorHeadSingle": "+++",
}
# Default hatch linewidth is thin at typical PNG dpi; bump for visible substage texture.
FORWARD_NVTX_HATCH_LINEWIDTH = 2.5
# Matches Modal A10 instances in runs.csv whether the driver reports "NVIDIA A10" or "NVIDIA A10G" (not A100).
DEFAULT_GPU = "A10"
# Stacked horizontal bar charts (latency + energy): wide canvas for bar length + NVTX legend.
STACKED_BARH_FIG_WIDTH = 20.0

# legend text (both pipeline and forward/NVTX legends)
LEGEND_FONTSIZE = 22
LEGEND_TITLE_FONTSIZE = 22
LEGEND_FONTSIZE_FWD = 22
LEGEND_BORDER_COLOR = "black"
LEGEND_BORDER_WIDTH = 1.0
LEGEND_BOX_LEFT_X = 0.81
# Fixed time axis upper bound for horizontal bars (ms).
TIME_AXIS_MAX_MS = 52.0
# Keep only these M2 cells in the plot.
M2_KEEP_ONLY = {"M2_FP32_mem_both", "M2_AMP_mem_both"}

# CSV column groups (no dataloader). Must match runs.csv column names (snake_case).
# M5 (TRT) rows only set prof_pre_processing_mean_ms / prof_h2d_mean_ms (no read_points split).
STAGE_PARTS = {
    "Preprocess": [
        "prof_read_points_mean_ms",
        "prof_cpu_prepare_mean_ms",
        "prof_pre_processing_mean_ms",
    ],
    # Includes prof_data_to_gpu_pts: CPU-voxel path stores load_data_to_gpu there (see profile_suite write_summary).
    "H2D": [
        "prof_h2d_mean_ms",
        "prof_h2d_voxel_tail_mean_ms",
        "prof_data_to_gpu_pts_mean_ms",
    ],
    "Forward": ["prof_forward_mean_ms"],
    "Postprocess": ["prof_postprocess_mean_ms"],
}

# Pipeline stages: mid-tone fills (readable on screen / projector; still separable from NVTX hues).
COLOR_BY_STAGE = {
    "Preprocess": "#5c8fd4",
    "H2D": "#c4923a",
    "Forward": "#3d9a7d",
    "Postprocess": "#c75c48",
}

# NVTX forward substages: same face color as aggregate "Forward"; only hatches differ.
FORWARD_NVTX_COLORS = {k: COLOR_BY_STAGE["Forward"] for k in POINTPILLAR_FORWARD_NVTX}

# Shared rc tweaks for latency + energy runs (readable grid, not heavy).
MLSYS_PLOT_RC = {
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "grid.color": "#c8c8c8",
    "grid.linestyle": "-",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.28,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "figure.facecolor": "white",
    "axes.facecolor": "#f0f0f0",
}

# M0–M2: pipeline order in profile (prep then load rest to GPU).
STAGE_ORDER_M012 = ["Preprocess", "H2D", "Forward", "Postprocess"]
# M3/M4: GPU voxel path — show H2D segment before Preprocess segment (visual order).
STAGE_ORDER_M34 = ["H2D", "Preprocess", "Forward", "Postprocess"]

LEGEND_ORDER = ["Preprocess", "H2D", "Forward", "Postprocess"]

# M5 (TRT) rows: no OpenPCDet NVTX; draw forward as a single block (forward green); hatch swapped with VFE NVTX.
M5_FORWARD_HATCH = "///"


def _style_legend_border_black(leg) -> None:
    fr = leg.get_frame()
    fr.set_edgecolor(LEGEND_BORDER_COLOR)
    fr.set_linewidth(LEGEND_BORDER_WIDTH)


def _style_legend_left_aligned(leg) -> None:
    # Keep title and entries left-aligned consistently across legends.
    leg._legend_box.align = "left"  # noqa: SLF001 - matplotlib has no public setter
    leg.get_title().set_ha("left")


def resolve_artifact_path(cell_value: str, repo_root: Path) -> Path | None:
    s = (cell_value or "").strip()
    if not s:
        return None
    p = Path(s)
    if p.is_file():
        return p
    q = repo_root / s
    return q if q.is_file() else None


def normalize_runs_csv_path(p: Path) -> Path:
    """
    Accept ``runs.csv`` or a directory that contains it (e.g. modal_v2_allcells_full bundle root).
    Relative paths are resolved from the current working directory.
    """
    q = p.expanduser()
    if not q.is_absolute():
        q = (Path.cwd() / q).resolve()
    else:
        q = q.resolve()
    if q.is_dir():
        q = q / "runs.csv"
    if not q.is_file():
        raise FileNotFoundError(
            "Expected runs.csv or a directory containing runs.csv; not found: %s" % (q,)
        )
    return q


def resolve_forward_nvtx_json_path(
    row: pd.Series,
    repo_root: Path,
    *,
    runs_csv: Path | None = None,
    fallback_artifact_root: Path | None = None,
) -> Path | None:
    """
    Path to forward_nvtx_ms.json for --nest-forward-from-artifacts.

    Order: explicit artifacts_forward_nvtx_json if that file exists;
    then ``<parent(runs_csv)>/<experiment_cell_id>/artifacts/forward_nvtx_ms.json`` (local bundle);
    then sibling of resolved run_manifest / metadata_json_path;
    then if ``fallback_artifact_root`` is set: that directory + cell + ``artifacts/`` or ``profile/`` + forward_nvtx_ms.json
    (useful when the primary runs.csv is from a newer run without downloaded NVTX artifacts, e.g. v4 + v3 fallback).
    """
    cell = str(row.get("experiment_cell_id", "") or "").strip()
    raw = str(row.get("artifacts_forward_nvtx_json", "") or "").strip()
    jp = resolve_artifact_path(raw, repo_root)
    if jp is not None:
        return jp
    if runs_csv is not None and cell:
        cand = runs_csv.resolve().parent / cell / "artifacts" / "forward_nvtx_ms.json"
        if cand.is_file():
            return cand
    for key in ("artifacts_run_manifest_json", "metadata_json_path"):
        raw_m = str(row.get(key, "") or "").strip()
        mp = resolve_artifact_path(raw_m, repo_root)
        if mp is not None and mp.is_file():
            cand = mp.parent / "forward_nvtx_ms.json"
            if cand.is_file():
                return cand
    if fallback_artifact_root is not None and cell:
        root = fallback_artifact_root.resolve()
        for sub in ("artifacts", "profile"):
            cand = root / cell / sub / "forward_nvtx_ms.json"
            if cand.is_file():
                return cand
    return None


def _parse_forward_nvtx_json(json_path: Path) -> tuple[dict[str, float] | None, list[str]]:
    """
    Read forward_nvtx_ms.json once: NVTX time fractions (sum ~1) and draw order
    (modal_v2 ``nvtx_stage_order`` when present).
    """
    fallback_order = list(POINTPILLAR_FORWARD_NVTX)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None, fallback_order

    def _normalize_from_counts(counts: dict[str, float]) -> dict[str, float] | None:
        t = sum(counts.get(k, 0.0) for k in POINTPILLAR_FORWARD_NVTX)
        if t < 1e-12:
            return None
        return {k: max(0.0, counts.get(k, 0.0)) / t for k in POINTPILLAR_FORWARD_NVTX}

    frac: dict[str, float] | None = None
    frac_in = data.get("fraction_of_forward_sum")
    if isinstance(frac_in, dict) and frac_in:
        counts: dict[str, float] = {}
        for k in POINTPILLAR_FORWARD_NVTX:
            v = frac_in.get(k)
            if v is None:
                counts[k] = 0.0
            else:
                try:
                    counts[k] = float(v)
                except (TypeError, ValueError):
                    counts[k] = 0.0
        frac = _normalize_from_counts(counts)

    if frac is None:
        mean_ms = data.get("mean_ms")
        if isinstance(mean_ms, dict) and mean_ms:
            counts = {}
            for k in POINTPILLAR_FORWARD_NVTX:
                v = mean_ms.get(k)
                if v is None:
                    counts[k] = 0.0
                else:
                    try:
                        counts[k] = float(v)
                    except (TypeError, ValueError):
                        counts[k] = 0.0
            frac = _normalize_from_counts(counts)

    raw_order = data.get("nvtx_stage_order")
    if isinstance(raw_order, list) and raw_order:
        stage_order = [str(x) for x in raw_order if str(x) in POINTPILLAR_FORWARD_NVTX]
        for k in POINTPILLAR_FORWARD_NVTX:
            if k not in stage_order:
                stage_order.append(k)
    else:
        stage_order = fallback_order

    if frac is None:
        return None, stage_order
    return frac, stage_order


def load_forward_nvtx_fractions_ordered(
    json_path: Path,
) -> tuple[dict[str, float] | None, list[str]]:
    """Same as ``load_forward_nvtx_fractions`` plus substage bar order from JSON."""
    return _parse_forward_nvtx_json(json_path)


def load_forward_nvtx_fractions(json_path: Path) -> dict[str, float] | None:
    """
    Load per-substage shares of forward time (sum ~1 over POINTPILLAR_FORWARD_NVTX).
    Uses fraction_of_forward_sum when usable; otherwise derives from mean_ms.
    """
    frac, _ = _parse_forward_nvtx_json(json_path)
    return frac


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _slug_for_filename(gpu_arg: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", gpu_arg.strip()).strip("_").lower()
    return s or "gpu"


def _gpu_filter_mask(gpu_series: pd.Series, needle: str) -> pd.Series:
    """Case-insensitive match; --gpu A10 matches A10/A10G rows but not A100."""
    n = needle.strip()
    s = gpu_series.astype(str)
    if n.upper() == "A10":
        return s.str.contains(r"A10(?!0)", case=False, na=False, regex=True)
    return s.str.contains(re.escape(n), case=False, na=False, regex=True)


def list_distinct_gpus(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    if "gpu_name" not in df.columns:
        return []
    return sorted(df["gpu_name"].dropna().astype(str).unique().tolist())


def load_filtered_latest(csv_path: Path, gpu_substring: str) -> tuple[pd.DataFrame, str]:
    """
    Rows where gpu_name matches gpu_substring (default: literal substring; A10 uses a regex so A100 is excluded).
    Returns (dataframe, resolved_gpu_name) where resolved is the single gpu_name if unique.
    """
    df = pd.read_csv(csv_path)
    if "gpu_name" not in df.columns:
        raise ValueError("runs.csv missing gpu_name")
    needle = gpu_substring.strip()
    if not needle:
        raise ValueError("--gpu must be non-empty (e.g. A10, A10G, T4, L4). Modal provisioning uses gpu=A10.")

    mask = _gpu_filter_mask(df["gpu_name"], needle)
    df = df[mask]
    if "experiment_status" in df.columns:
        df = df[df["experiment_status"].astype(str) == "measured"]
    if df.empty:
        raise ValueError(
            "No measured rows for gpu filter %r in %s" % (needle, csv_path)
        )

    uniq = sorted(df["gpu_name"].dropna().astype(str).unique().tolist())
    if len(uniq) > 1:
        # Modal may report "NVIDIA A10" or "NVIDIA A10G" for the same SKU family; --gpu A10 matches both.
        if needle.upper() == "A10":
            resolved = "NVIDIA A10 / A10G"
        else:
            raise ValueError(
                "GPU filter %r matches multiple gpu_name values: %s. "
                "Use a more specific --gpu substring." % (needle, uniq)
            )
    else:
        resolved = uniq[0]
    ts = pd.to_datetime(df["timestamp_iso"], utc=True, errors="coerce")
    df = df.assign(_ts=ts).sort_values("_ts")
    df = df.drop_duplicates(subset=["experiment_cell_id"], keep="last")
    return df.drop(columns=["_ts"], errors="ignore"), resolved


def sort_by_m_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order rows by M0, M1, … then FP32 before AMP, then ``experiment_cell_id``.

    Uses ``precision_mode`` when present (FP32 / AMP); otherwise infers from cell id
    (``_FP32`` / ``_AMP`` substrings) for robustness.
    """
    cid = df["experiment_cell_id"].astype(str)
    m = cid.str.extract(r"^M(\d+)_", expand=False)
    mnum = pd.to_numeric(m, errors="coerce").fillna(999).astype(int)
    pr = pd.Series(2, index=df.index, dtype=int)
    if "precision_mode" in df.columns:
        pm = df["precision_mode"].astype(str).str.strip().str.upper()
        pr[pm == "FP32"] = 0
        pr[pm == "AMP"] = 1
    low = cid.str.lower()
    pending = pr == 2
    pr.loc[pending & low.str.contains("_fp32", na=False)] = 0
    pending = pr == 2
    pr.loc[pending & low.str.contains("_amp", na=False)] = 1
    out = df.assign(_mnum=mnum, _prank=pr).sort_values(
        by=["_mnum", "_prank", "experiment_cell_id"]
    )
    return out.drop(columns=["_mnum", "_prank"])


def stage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for stage_name, parts in STAGE_PARTS.items():
        acc = None
        for c in parts:
            if c not in df.columns:
                continue
            v = _num(df[c])
            acc = v if acc is None else acc + v
        cols[stage_name] = acc if acc is not None else pd.Series(0.0, index=df.index)
    return pd.DataFrame(cols, index=df.index)


def stack_order_for_cell(cell_id: str) -> list[str]:
    if re.match(r"^M[34]_", cell_id):
        return STAGE_ORDER_M34
    return STAGE_ORDER_M012


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot stacked per-stage latency from runs.csv for one GPU (substring match)."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to runs.csv or a directory that contains it (default: modal_outputs/modal_v4_a10 if present, else modal_v2* bundle)",
    )
    p.add_argument(
        "--gpu",
        type=str,
        default=DEFAULT_GPU,
        help=(
            "Filter runs.csv gpu_name (case-insensitive). Use A10 for Modal A10/A10G rows (excludes A100). "
            "Examples: T4, L4, H100, A10G. Default: %(default)s"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PNG,
        help="Output PNG (default: report/latency_stacked_a10_v4_nvtx_from_v3.png)",
    )
    p.add_argument(
        "--list-gpus",
        action="store_true",
        help="Print distinct gpu_name values from the CSV and exit",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--nest-forward-from-artifacts",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Split forward using NVTX forward_nvtx_ms.json (default: on; use --no-nest-forward-from-artifacts to off)",
    )
    p.add_argument(
        "--forward-nvtx-fallback-artifacts-dir",
        type=Path,
        default=DEFAULT_NVTX_FALLBACK_DIR,
        help=(
            "If NVTX JSON is missing next to the primary runs.csv, try "
            "DIR/<experiment_cell_id>/{artifacts,profile}/forward_nvtx_ms.json (default: modal_outputs/modal_v3_a10)."
        ),
    )
    args = p.parse_args()
    try:
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
        print("Distinct gpu_name in %s:" % csv_path)
        for n in names:
            print(" ", n)
        sys.exit(0)

    nvtx_fb: Path | None = None
    fbd = args.forward_nvtx_fallback_artifacts_dir.expanduser()
    if not fbd.is_absolute():
        fbd = (Path.cwd() / fbd).resolve()
    else:
        fbd = fbd.resolve()
    if fbd.is_dir():
        nvtx_fb = fbd
    else:
        print(
            "plot_latency: forward-nvtx-fallback not a directory: %s (NVTX fallback disabled)"
            % (fbd,),
            file=sys.stderr,
        )

    out_path = args.out.resolve()

    df, gpu_resolved = load_filtered_latest(csv_path, args.gpu)
    # For M2, only keep mem_both variants; keep all non-M2 cells.
    keep_mask = df["experiment_cell_id"].astype(str).apply(
        lambda c: (not c.startswith("M2_")) or (c in M2_KEEP_ONLY)
    )
    df = df[keep_mask].copy()
    df = sort_by_m_group(df)
    labels = df["experiment_cell_id"].astype(str).str.replace("_mem_both", "", regex=False).tolist()
    mat = stage_matrix(df)
    mat = mat.reindex(df.index)

    fig_h = 1.8 * max(6.0, 0.35 * len(labels) + 2.0)
    drew_forward_nvtx = False
    tried_nvtx_paths: list[Path] = []

    pipe_legend_handles = [
        Patch(
            facecolor=COLOR_BY_STAGE[s],
            edgecolor="#5a5a5a",
            linewidth=0.35,
            label=s.replace("_", " "),
        )
        for s in LEGEND_ORDER
    ]

    with plt.rc_context(MLSYS_PLOT_RC):
        fig, ax = plt.subplots(figsize=(STACKED_BARH_FIG_WIDTH, fig_h))
        xmax = 0.0
        for yi, lab in enumerate(labels):
            row = mat.iloc[yi]
            order = stack_order_for_cell(lab)
            left = 0.0
            m5_row = str(lab).startswith("M5_")
            frac = None
            sub_order = list(POINTPILLAR_FORWARD_NVTX)
            if args.nest_forward_from_artifacts and not m5_row:
                jp = resolve_forward_nvtx_json_path(
                    df.iloc[yi],
                    _REPO_ROOT,
                    runs_csv=csv_path,
                    fallback_artifact_root=nvtx_fb,
                )
                if jp is not None:
                    tried_nvtx_paths.append(jp)
                    frac, sub_order = load_forward_nvtx_fractions_ordered(jp)
            for name in order:
                if name == "Forward" and frac is not None:
                    fms = float(row["Forward"])
                    sub_total = sum(float(frac.get(sub, 0.0)) for sub in POINTPILLAR_FORWARD_NVTX)
                    if sub_total < 1e-9 and fms > 0:
                        ax.barh(
                            lab,
                            fms,
                            left=left,
                            color=COLOR_BY_STAGE["Forward"],
                            height=0.7,
                            edgecolor="white",
                            linewidth=0.65,
                        )
                        left += fms
                    else:
                        for sub in sub_order:
                            w = fms * float(frac.get(sub, 0.0))
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
                val = float(row[name])
                if m5_row and name == "Forward":
                    b = ax.barh(
                        lab,
                        val,
                        left=left,
                        facecolor=COLOR_BY_STAGE[name],
                        hatch=M5_FORWARD_HATCH,
                        height=0.7,
                        edgecolor="white",
                        linewidth=0.65,
                    )
                    for p in b.patches:
                        p.set_hatch_linewidth(FORWARD_NVTX_HATCH_LINEWIDTH)
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

        if args.nest_forward_from_artifacts and drew_forward_nvtx:
            fwd_legend_handles = []
            for sub in POINTPILLAR_FORWARD_NVTX:
                leg = Patch(
                    facecolor=FORWARD_NVTX_COLORS.get(sub, COLOR_BY_STAGE["Forward"]),
                    edgecolor="#f5f5f0",
                    linewidth=1.0,
                    hatch=FORWARD_NVTX_HATCH.get(sub, ""),
                    label=FORWARD_NVTX_LEGEND_LABEL.get(sub, sub),
                )
                leg.set_hatch_linewidth(FORWARD_NVTX_HATCH_LINEWIDTH)
                fwd_legend_handles.append(leg)
            # Bottom of axes: "Forward substage" (NVTX); just above: "Stage" (pipeline).
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
            leg_stage = ax.legend(
                handles=pipe_legend_handles,
                loc="lower left",
                bbox_to_anchor=(LEGEND_BOX_LEFT_X, 0.31),
                borderaxespad=0.35,
                fontsize=LEGEND_FONTSIZE,
                framealpha=1.0,
                title="Stage",
                title_fontsize=LEGEND_TITLE_FONTSIZE,
            )
            _style_legend_border_black(leg_stage)
            _style_legend_left_aligned(leg_stage)
            tight_rect = [0.0, 0.0, 1.0, 1.0]
        else:
            if args.nest_forward_from_artifacts:
                paths_hint = (
                    ", ".join(sorted({str(p) for p in tried_nvtx_paths}))
                    if tried_nvtx_paths
                    else "(no JSON path resolved)"
                )
                print(
                    "plot_latency_breakdown: --nest-forward-from-artifacts but no forward NVTX sub-bars. "
                    "Check forward_nvtx_ms.json (zeros or wrong keys). Expected keys: %s. Paths: %s"
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

        ax.set_xlabel("Time [ms]")
        ax.set_xlim(0, TIME_AXIS_MAX_MS)
        ax.invert_yaxis()
        fig.tight_layout(rect=tight_rect)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
    print(
        "Wrote %s (%d cells, gpu=%s, latest run per experiment_cell_id)"
        % (out_path, len(labels), gpu_resolved)
    )


if __name__ == "__main__":
    main()
