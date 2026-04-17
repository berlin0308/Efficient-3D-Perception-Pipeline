#!/usr/bin/env python3
"""
Stacked horizontal bar chart: per-stage mean latency from research_matrix runs.csv.
Filters by GPU (--gpu substring on gpu_name), one row per experiment_cell_id (latest timestamp_iso).
No dataloader segment. M3/M4 stack: h2d then pre_processing; M0–M2: pre_processing then h2d.
CPU-voxel runs (M0–M2): profile_summary prints load_data_to_gpu as "data_to_gpu" → CSV prof_data_to_gpu_pts_* (not prof_h2d); that belongs in the h2d segment.
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
DEFAULT_CSV = _V2_FULL_CSV if _V2_FULL_CSV.is_file() else _V2_CSV

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

# Okabe–Ito–style palette (colorblind-friendly). Forward substages stay fully saturated so they
# read clearly inside the forward pipeline segment (still distinct from the aggregate forward hue).
FORWARD_NVTX_COLORS = {
    "PillarVFE": "#005a92",
    "PointPillarScatter": "#e68900",
    "BaseBEVBackbone": "#007a5c",
    "AnchorHeadSingle": "#a8558f",
}
FORWARD_NVTX_LEGEND_LABEL = {
    "PillarVFE": "VFE",
    "PointPillarScatter": "Scatter",
    "BaseBEVBackbone": "BEV",
    "AnchorHeadSingle": "Anchor head",
}
# Hatch = redundant encoding for print / grayscale; distinct shapes per substage.
FORWARD_NVTX_HATCH = {
    "PillarVFE": "///",
    "PointPillarScatter": "...",
    "BaseBEVBackbone": "|||",
    "AnchorHeadSingle": "+++",
}
# Default hatch linewidth is thin at typical PNG dpi; bump for visible substage texture.
FORWARD_NVTX_HATCH_LINEWIDTH = 2.25
# Matches Modal A10 instances in runs.csv whether the driver reports "NVIDIA A10" or "NVIDIA A10G" (not A100).
DEFAULT_GPU = "A10"
# Stacked horizontal bar charts (latency + energy): wider canvas so bar length reads clearly.
STACKED_BARH_FIG_WIDTH = 16.0

# CSV column groups (no dataloader).
STAGE_PARTS = {
    "pre_processing": [
        "prof_read_points_mean_ms",
        "prof_cpu_prepare_mean_ms",
        "prof_pre_processing_mean_ms",
    ],
    # Includes prof_data_to_gpu_pts: CPU-voxel path stores load_data_to_gpu there (see profile_suite write_summary).
    "h2d": [
        "prof_h2d_mean_ms",
        "prof_h2d_voxel_tail_mean_ms",
        "prof_data_to_gpu_pts_mean_ms",
    ],
    "forward": ["prof_forward_mean_ms"],
    "postprocess": ["prof_postprocess_mean_ms"],
}

# Pipeline stages: mid-tone fills (readable on screen / projector; still separable from NVTX hues).
COLOR_BY_STAGE = {
    "pre_processing": "#5c8fd4",
    "h2d": "#c4923a",
    "forward": "#3d9a7d",
    "postprocess": "#c75c48",
}

# Shared rc tweaks for latency + energy runs (readable grid, not heavy).
MLSYS_PLOT_RC = {
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.edgecolor": "#b0b0b0",
    "axes.linewidth": 0.9,
    "grid.color": "#c8c8c8",
    "grid.linestyle": "-",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.45,
    "figure.facecolor": "white",
    "axes.facecolor": "#f0f0f0",
}

# M0–M2: pipeline order in profile (prep then load rest to GPU).
STAGE_ORDER_M012 = ["pre_processing", "h2d", "forward", "postprocess"]
# M3/M4: GPU voxel path — show h2d segment before pre_processing segment (visual order).
STAGE_ORDER_M34 = ["h2d", "pre_processing", "forward", "postprocess"]

LEGEND_ORDER = ["pre_processing", "h2d", "forward", "postprocess"]


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
) -> Path | None:
    """
    Path to forward_nvtx_ms.json for --nest-forward-from-artifacts.

    Order: explicit artifacts_forward_nvtx_json if that file exists;
    then ``<parent(runs_csv)>/<experiment_cell_id>/artifacts/forward_nvtx_ms.json`` (local bundle);
    then sibling of resolved run_manifest / metadata_json_path.
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
        help="Path to runs.csv or matrix directory containing it (default: modal_v2_allcells_full if present)",
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
        default=None,
        help="Output PNG path (default: latency_stacked_<gpu_slug>.png next to this script)",
    )
    p.add_argument(
        "--list-gpus",
        action="store_true",
        help="Print distinct gpu_name values from the CSV and exit",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--nest-forward-from-artifacts",
        action="store_true",
        help="Split forward using artifacts_forward_nvtx_json (fraction_of_forward_sum) when file exists",
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

    out_path = (
        args.out.resolve()
        if args.out
        else Path(__file__).resolve().parent
        / ("latency_stacked_%s.png" % _slug_for_filename(args.gpu))
    )

    df, gpu_resolved = load_filtered_latest(csv_path, args.gpu)
    df = sort_by_m_group(df)
    labels = df["experiment_cell_id"].astype(str).tolist()
    mat = stage_matrix(df)
    mat = mat.reindex(df.index)

    fig_h = max(6.0, 0.35 * len(labels) + 2.0)
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
            frac = None
            sub_order = list(POINTPILLAR_FORWARD_NVTX)
            if args.nest_forward_from_artifacts:
                jp = resolve_forward_nvtx_json_path(df.iloc[yi], _REPO_ROOT, runs_csv=csv_path)
                if jp is not None:
                    tried_nvtx_paths.append(jp)
                    frac, sub_order = load_forward_nvtx_fractions_ordered(jp)
            for name in order:
                if name == "forward" and frac is not None:
                    fms = float(row["forward"])
                    sub_total = sum(float(frac.get(sub, 0.0)) for sub in POINTPILLAR_FORWARD_NVTX)
                    if sub_total < 1e-9 and fms > 0:
                        ax.barh(
                            lab,
                            fms,
                            left=left,
                            color=COLOR_BY_STAGE["forward"],
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
                                facecolor=FORWARD_NVTX_COLORS.get(sub, COLOR_BY_STAGE["forward"]),
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
                    facecolor=FORWARD_NVTX_COLORS.get(sub, "#333333"),
                    edgecolor="#f5f5f0",
                    linewidth=1.0,
                    hatch=FORWARD_NVTX_HATCH.get(sub, ""),
                    label=FORWARD_NVTX_LEGEND_LABEL.get(sub, sub),
                )
                leg.set_hatch_linewidth(FORWARD_NVTX_HATCH_LINEWIDTH)
                fwd_legend_handles.append(leg)
            leg_pipe = ax.legend(
                handles=pipe_legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.35,
                fontsize=8,
                framealpha=0.96,
                title="Pipeline",
                title_fontsize=8,
            )
            ax.add_artist(leg_pipe)
            ax.legend(
                handles=fwd_legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 0.38),
                borderaxespad=0.35,
                fontsize=7,
                framealpha=0.96,
                title="Forward (NVTX share)",
                title_fontsize=7,
            )
            tight_rect = [0.0, 0.0, 0.70, 1.0]
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
            ax.legend(
                handles=pipe_legend_handles,
                loc="upper right",
                fontsize=8,
                framealpha=0.92,
            )
            tight_rect = [0.0, 0.0, 0.92, 1.0]

        ax.set_xlabel("Mean time [ms]")
        ax.set_ylabel("experiment_cell_id")
        ax.set_title(
            "Per-stage latency breakdown (profile_suite means; %s; no dataloader)" % gpu_resolved
        )
        ax.set_xlim(0, xmax * 1.02 if xmax > 0 else 1.0)
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
