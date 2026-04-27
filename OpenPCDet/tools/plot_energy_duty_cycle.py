"""
plot_energy_duty_cycle.py — Holistic 10 Hz duty-cycle energy budget (edge only).

Full pipeline breakdown in chronological order, color-coded by CPU vs GPU.

CPU segments (warm tones, diagonal hatch):
  platform_overhead, read_points, pre_processing, data_to_gpu

GPU segments (cool tones, no hatch):
  gpu_forward_vfe, gpu_forward_scatter, gpu_forward_bev,
  gpu_forward_head, gpu_postprocess (NMS)

Idle (pale tones):
  gpu_idle, cpu_idle

Rows (top→bottom):
  RTX 3080 Ti: M0_FP32, M0_AMP, M1_AMP, M3_AMP, M4_AMP

All values measured locally 2026-04-23 (profile_suite + energy_monitor).
GPU forward substage fractions from NVTX profile (A10G used as proxy for
local since NVTX not re-run locally; local forward total is measured).

Run from OpenPCDet/tools/:
    python plot_energy_duty_cycle.py \
        --output_dir ../../profile_outputs/amp_benchmark/report_figures
"""

import argparse
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Style (aligned with report/plot_latency.py) ─────────────────────────────
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

DUTY_MS = 100.0

# ── Stage colors/hatches (aligned with plot_latency.py) ─────────────────────
COLOR_BY_STAGE = {
    "Preprocess": "#5c8fd4",
    "H2D": "#c4923a",
    "Forward": "#3d9a7d",
    "Postprocess": "#c75c48",
}
FORWARD_NVTX_HATCH = {
    "gpu_vfe": "\\\\\\\\",
    "gpu_scatter": "**",
    "gpu_bev": "|||",
    "gpu_head": "+++",
}
FORWARD_NVTX_HATCH_LINEWIDTH = 2.5

C_PLATFORM = "#9e9e9e"                 # not in latency chart; keep neutral
C_READ = COLOR_BY_STAGE["Preprocess"]  # read_points grouped under preprocess family
C_PREPROC = COLOR_BY_STAGE["Preprocess"]
C_H2D = COLOR_BY_STAGE["H2D"]
C_VFE = COLOR_BY_STAGE["Forward"]
C_SCATTER = COLOR_BY_STAGE["Forward"]
C_BEV = COLOR_BY_STAGE["Forward"]
C_HEAD = COLOR_BY_STAGE["Forward"]
C_NMS = COLOR_BY_STAGE["Postprocess"]
# Idle (pale)
C_GPU_IDLE  = "#A89BBF"   # requested GPU idle color
C_CPU_IDLE  = "#F6C054"   # requested CPU idle color

LEGEND_FONTSIZE = 22
LEGEND_TITLE_FONTSIZE = 22
LEGEND_BORDER_COLOR = "black"
LEGEND_BORDER_WIDTH = 1.0
CPU_ACTIVE_HATCH = "|||"
GPU_ACTIVE_HATCH = "///"
ACTIVE_HATCH_LINEWIDTH = 2.2
ACTIVE_HATCH_COLOR = "#666666"
ACTIVE_BOX_FONTSIZE = 11

SEG_ORDER_M01 = [
    "platform_overhead",
    "cpu_idle",
    "gpu_idle",
    "pre_processing",
    "data_to_gpu",
    "gpu_forward",
    "gpu_nms",
]

SEG_ORDER_M34 = [
    "platform_overhead",
    "cpu_idle",
    "gpu_idle",
    "data_to_gpu",
    "pre_processing",
    "gpu_forward",
    "gpu_nms",
]

SEG_ALL = [
    "platform_overhead",
    "pre_processing",
    "data_to_gpu",
    "cpu_idle",
    "gpu_forward",
    "gpu_nms",
    "gpu_idle",
]

SEG_META = {
    # (color, hatch, label, cpu_or_gpu)
    "platform_overhead": (C_PLATFORM, "", "Platform Overhead\n(DRAM/SSD/Fan)", "cpu"),
    "pre_processing":    (C_PREPROC, "", "Preprocess", "cpu"),
    "data_to_gpu":       (C_H2D, "", "H2D", "cpu"),
    "gpu_forward":       (COLOR_BY_STAGE["Forward"], "", "Forward", "gpu"),
    "gpu_nms":           (C_NMS, "", "Postprocess", "gpu"),
    "gpu_idle":          (C_GPU_IDLE, "", "GPU Idle  (18.2 W)", "gpu"),
    "cpu_idle":          (C_CPU_IDLE, "", "CPU Idle  (11.6 W)", "cpu"),
}

# ── Measured constants ────────────────────────────────────────────────────────
GPU_IDLE_W        = 18.2
CPU_IDLE_PKG_W    = 11.6
PLATFORM_ACTIVE_W = 74.5   # psys − pkg − GPU during active window
PLATFORM_IDLE_W   = 30.0   # estimated at rest

# ── Edge variants — measured locally 2026-04-23 ───────────────────────────────
# latency breakdown from profile_suite; active powers from energy_monitor
LOCAL_VARIANTS = [
    dict(
        label="M0_AMP",
        gpu_W=40.3, cpu_W=65.74,
        read_ms=3.45, preproc_ms=9.78, h2d_ms=0.61,
        forward_ms=9.88, nms_ms=0.38,
        gpu_voxel=False,
    ),
    dict(
        label="M1_AMP",
        gpu_W=33.0, cpu_W=47.89,
        read_ms=3.43, preproc_ms=8.97, h2d_ms=0.64,
        forward_ms=9.20, nms_ms=0.38,
        gpu_voxel=False,
    ),
    dict(
        label="M3_AMP",
        gpu_W=32.2, cpu_W=53.61,
        read_ms=3.66, preproc_ms=1.28, h2d_ms=0.27,   # h2d = data_to_gpu + h2d_rest
        forward_ms=9.19, nms_ms=0.39,
        gpu_voxel=True,
    ),
]

# ── A10G cloud reference ──────────────────────────────────────────────────────
A10G = dict(
    label="A10G  M0_FP32\n(cloud, GPU measured\nCPU est.)",
    gpu_W=124.2, cpu_active_W=4.0, cpu_idle_W=2.0,
    read_ms=8.56, preproc_ms=17.06, h2d_ms=0.90,
    forward_ms=12.96, nms_ms=0.60,
)

# M5 values are fixed constants (no file I/O). Source baseline:
# modal_v3_a10/runs.csv latest measured rows (energy_analysis data path).
M5_FIXED_VARIANTS = [
    dict(
        label="M5",
        energy_per_frame_j=230.1626 / 500.0,
        preproc_ms=0.1721,
        h2d_ms=0.0,
        forward_ms=3.1841,
        # Estimated to align with OpenPCDet: median(M0/M1/M3 nms/forward) * M5 forward.
        nms_ms=0.1315,
        gpu_voxel=False,
    ),
]


def _to_float(v, default=0.0):
    try:
        if v is None or str(v).strip() == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def m5_segments(v):
    """
    Build M5 segment energies from measured per-frame total energy.
    Idle/platform follow existing constants; active budget is split by stage time share.
    """
    preproc_ms = _to_float(v.get("preproc_ms"))
    h2d_ms = _to_float(v.get("h2d_ms"))
    forward_ms = _to_float(v.get("forward_ms"))
    nms_ms = _to_float(v.get("nms_ms"))
    lat = preproc_ms + h2d_ms + forward_ms + nms_ms
    idle = max(0.0, DUTY_MS - lat)
    total_mj = max(0.0, _to_float(v.get("energy_per_frame_j")) * 1000.0)
    # For M5, measured energy is treated as active-window energy.
    # Add duty-cycle idle/platform estimates on top so the plot includes
    # platform_overhead/cpu_idle/gpu_idle like other rows.
    platform_mj = PLATFORM_ACTIVE_W * lat + PLATFORM_IDLE_W * idle
    cpu_idle_mj = CPU_IDLE_PKG_W * idle
    gpu_idle_mj = GPU_IDLE_W * idle
    active_budget_mj = total_mj
    active_ms = preproc_ms + h2d_ms + forward_ms + nms_ms
    scale = (active_budget_mj / active_ms) if active_ms > 1e-9 else 0.0
    segs = {
        "platform_overhead": platform_mj,
        "pre_processing": preproc_ms * scale,
        "data_to_gpu": h2d_ms * scale,
        "gpu_forward": forward_ms * scale,
        "gpu_nms": nms_ms * scale,
        "gpu_idle": gpu_idle_mj,
        "cpu_idle": cpu_idle_mj,
    }
    return segs


def edge_segments(v):
    if "energy_per_frame_j" in v:
        return m5_segments(v)
    lat  = v['read_ms'] + v['preproc_ms'] + v['h2d_ms'] + v['forward_ms'] + v['nms_ms']
    idle = DUTY_MS - lat
    fwd_mj = v['gpu_W'] * v['forward_ms']
    segs = {
        "platform_overhead": PLATFORM_ACTIVE_W * lat + PLATFORM_IDLE_W * idle,
        "pre_processing":    v['cpu_W'] * (v['read_ms'] + v['preproc_ms']),
        "data_to_gpu":       v['cpu_W'] * v['h2d_ms'],
        "gpu_forward":       fwd_mj,
        "gpu_nms":           v['gpu_W'] * v['nms_ms'],
        "gpu_idle":          GPU_IDLE_W * idle,
        "cpu_idle":          CPU_IDLE_PKG_W * idle,
    }
    return segs


def cloud_segments(c):
    lat  = c['read_ms'] + c['preproc_ms'] + c['h2d_ms'] + c['forward_ms'] + c['nms_ms']
    idle = DUTY_MS - lat
    fwd_mj = c['gpu_W'] * c['forward_ms']
    segs = {
        "platform_overhead": 0.0,
        "pre_processing":    c['cpu_active_W'] * (c['read_ms'] + c['preproc_ms']),
        "data_to_gpu":       c['cpu_active_W'] * c['h2d_ms'],
        "gpu_forward":       fwd_mj,
        "gpu_nms":           c['gpu_W'] * c['nms_ms'],
        "gpu_idle":          GPU_IDLE_W * idle,
        "cpu_idle":          c['cpu_idle_W'] * idle,
    }
    return segs


def seg_order_for_label(label: str) -> list[str]:
    if label.startswith("M3_") or label.startswith("M4_"):
        return SEG_ORDER_M34
    return SEG_ORDER_M01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='/home/nas/polin/cmu-berlin/MLS/report')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Assemble rows ────────────────────────────────────────────────────────
    rows = []
    for v in LOCAL_VARIANTS:
        rows.append(dict(label=v['label'].replace('_AMP', ''), segs=edge_segments(v),
                         cloud=False))
    for v in M5_FIXED_VARIANTS:
        rows.append(dict(label=v["label"], segs=edge_segments(v), cloud=False))
    print("Loaded fixed M5 variants: %s" % ", ".join(v["label"] for v in M5_FIXED_VARIANTS))
    for r in rows:
        r['total'] = sum(r['segs'].get(s, 0) for s in SEG_ALL)

    n     = len(rows)
    bar_h = 0.65
    fig_h = max(6.5, n * 1.15 + 2.2)
    y     = np.arange(n, dtype=float)

    XLIM  = 5000.0
    scale = XLIM / max(r['total'] for r in rows)

    with plt.rc_context(MLSYS_PLOT_RC):
        fig, (ax, ax_leg) = plt.subplots(
            1, 2, figsize=(20.0, fig_h),
            gridspec_kw=dict(width_ratios=[5, 1]))
        ax_leg.axis('off')

        # ── Draw bars ────────────────────────────────────────────────────────
        for ri, row in enumerate(rows):
            left = 0.0
            row_order = seg_order_for_label(row['label'])
            seg_pos: dict[str, tuple[float, float]] = {}
            for seg in row_order:
                real = row['segs'].get(seg, 0)
                disp = real * scale
                if real <= 0:
                    continue
                color, hatch_base, _, kind = SEG_META[seg]

                # Cloud: skip platform (unknown); hatch CPU (estimated)
                if row['cloud'] and seg == 'platform_overhead':
                    continue
                if row['cloud'] and kind == 'cpu':
                    hatch = '//'
                else:
                    hatch = hatch_base

                bars = ax.barh(
                    y[ri], disp, height=bar_h, left=left,
                    color=color, edgecolor='white',
                    linewidth=0.5, hatch=hatch, alpha=0.92
                )
                if hatch:
                    for p in bars.patches:
                        p.set_hatch_linewidth(FORWARD_NVTX_HATCH_LINEWIDTH)
                seg_pos[seg] = (left, left + disp)
                left += disp

            # Overlay active-role hatches directly on bars (not legend).
            # Rules:
            # - M0/M1: preprocess=CPU active; after preprocess (H2D+forward+postprocess)=GPU active
            # - M3/M4: from preprocess onward = GPU active; H2D marked as CPU active
            label = row['label']
            cpu_span: tuple[float, float] | None = None
            gpu_span: tuple[float, float] | None = None
            if label.startswith("M0") or label.startswith("M1"):
                cpu_span = seg_pos.get("pre_processing")
                h2d = seg_pos.get("data_to_gpu")
                post = seg_pos.get("gpu_nms")
                if h2d and post:
                    gpu_span = (h2d[0], post[1])
            elif label.startswith("M5"):
                # TRT M5 follows CPU-preprocess then GPU active pipeline like M0/M1.
                cpu_span = seg_pos.get("pre_processing")
                h2d = seg_pos.get("data_to_gpu")
                fwd = seg_pos.get("gpu_forward")
                post = seg_pos.get("gpu_nms")
                if h2d and post:
                    gpu_span = (h2d[0], post[1])
                elif fwd and post:
                    # M5 fixed data may have zero H2D width; then start GPU-active
                    # overlay at forward so hatch remains visible.
                    gpu_span = (fwd[0], post[1])
            elif label.startswith("M3") or label.startswith("M4"):
                h2d = seg_pos.get("data_to_gpu")
                if h2d:
                    cpu_span = h2d
                pre = seg_pos.get("pre_processing")
                post = seg_pos.get("gpu_nms")
                if pre and post:
                    gpu_span = (pre[0], post[1])

            def _overlay(span: tuple[float, float] | None, hatch: str) -> None:
                if not span:
                    return
                s, e = span
                w = e - s
                if w <= 0:
                    return
                overlay = ax.barh(
                    y[ri],
                    w,
                    left=s,
                    height=bar_h,
                    facecolor="none",
                    edgecolor=ACTIVE_HATCH_COLOR,
                    linewidth=0.0,
                    hatch=hatch,
                    zorder=5,
                )
                for p in overlay.patches:
                    p.set_hatch_linewidth(ACTIVE_HATCH_LINEWIDTH)

            _overlay(cpu_span, CPU_ACTIVE_HATCH)
            _overlay(gpu_span, GPU_ACTIVE_HATCH)

        # ── Axes ─────────────────────────────────────────────────────────────
        ax.set_yticks(y)
        ax.set_yticklabels([r['label'] for r in rows], fontsize=22)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(1000 * scale))
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: f'{x/scale/1000:.0f}'))
        ax.set_xlabel('Energy per 10Hz Cycle [J]', fontsize=22)
        ax.set_xlim(0, 8.3 * 1000.0 * scale)
        # Keep plot style consistent with report/plot_latency.py: no figure title.

        # ── Segment legend (top) ──────────────────────────────────────────────
        def seg_patch(key):
            color, hatch, label, _ = SEG_META[key]
            ec = '#888' if hatch else '#ccc'
            return mpatches.Patch(facecolor=color, hatch=hatch,
                                  edgecolor=ec, linewidth=0.5, label=label)

        # Align main stage order with report/plot_latency.py:
        # Preprocess -> H2D -> Forward -> Postprocess
        seg_patches = (
            [mpatches.Patch(visible=False, label='── Idle / Platform ──')] +
            [seg_patch('platform_overhead'), seg_patch('cpu_idle'), seg_patch('gpu_idle')] +
            [mpatches.Patch(visible=False, label='── Active ──')] +
            [
                mpatches.Patch(facecolor='white', edgecolor=ACTIVE_HATCH_COLOR, hatch=CPU_ACTIVE_HATCH, linewidth=0.8, label='CPU Active'),
                mpatches.Patch(facecolor='white', edgecolor=ACTIVE_HATCH_COLOR, hatch=GPU_ACTIVE_HATCH, linewidth=0.8, label='GPU Active'),
            ] +
            [mpatches.Patch(visible=False, label='── Stage ──')] +
            [seg_patch(s) for s in ["pre_processing", "data_to_gpu", "gpu_forward", "gpu_nms"]]
        )
        leg = ax_leg.legend(
            handles=seg_patches,
            title='',
            title_fontsize=LEGEND_TITLE_FONTSIZE,
            fontsize=LEGEND_FONTSIZE,
            loc='upper left',
            framealpha=1.0,
            edgecolor=LEGEND_BORDER_COLOR,
            borderaxespad=0,
        )
        leg._legend_box.align = "left"  # noqa: SLF001
        leg.get_title().set_ha("left")
        fr = leg.get_frame()
        fr.set_edgecolor(LEGEND_BORDER_COLOR)
        fr.set_linewidth(LEGEND_BORDER_WIDTH)

        plt.tight_layout(rect=[0, 0, 1, 1])
        path = out / 'energy_duty_cycle.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', path)


if __name__ == '__main__':
    main()
