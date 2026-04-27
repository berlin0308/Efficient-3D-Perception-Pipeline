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

# ── Style ────────────────────────────────────────────────────────────────────
MLSYS_PLOT_RC = {
    "axes.grid":        True,
    "axes.axisbelow":   True,
    "axes.edgecolor":   "#b0b0b0",
    "axes.linewidth":   0.9,
    "grid.color":       "#c8c8c8",
    "grid.linestyle":   "-",
    "grid.linewidth":   0.55,
    "grid.alpha":       0.45,
    "figure.facecolor": "white",
    "axes.facecolor":   "#f0f0f0",
}

DUTY_MS = 100.0

# ── Colors — warm = CPU, cool = GPU ──────────────────────────────────────────
# CPU (warm, hatched with \\)
C_PLATFORM  = "#9e9e9e"   # gray
C_READ      = "#d4a96a"   # light amber   — read_points
C_PREPROC   = "#c4923a"   # amber         — voxelization
C_H2D       = "#a0522d"   # brown         — H2D transfer
# GPU forward substages (cool, no hatch)
C_VFE       = "#005a92"   # deep blue
C_SCATTER   = "#e68900"   # orange        — distinct from CPU warm
C_BEV       = "#3d9a7d"   # teal
C_HEAD      = "#a8558f"   # purple
C_NMS       = "#c75c48"   # coral
# Idle (pale)
C_GPU_IDLE  = "#b2dfdb"   # pale teal
C_CPU_IDLE  = "#ffe0b2"   # pale amber

SEG_ORDER = [
    "platform_overhead",   # system (not CPU/GPU)
    "read_points",         # CPU active
    "pre_processing",      # CPU active
    "data_to_gpu",         # CPU active
    "cpu_idle",            # CPU idle  ← grouped with CPU active
    "gpu_vfe",             # GPU active
    "gpu_scatter",         # GPU active
    "gpu_bev",             # GPU active
    "gpu_head",            # GPU active
    "gpu_nms",             # GPU active
    "gpu_idle",            # GPU idle
]

SEG_META = {
    # (color, hatch, label, cpu_or_gpu)
    "platform_overhead": (C_PLATFORM, "xx",   "Platform overhead\n(DRAM/SSD/display/fan)", "cpu"),
    "read_points":       (C_READ,     "\\\\", "read_points",                               "cpu"),
    "pre_processing":    (C_PREPROC,  "\\\\", "pre_processing\n(voxelization)",            "cpu"),
    "data_to_gpu":       (C_H2D,      "\\\\", "CPU→GPU: H2D transfer",                     "cpu"),
    "gpu_vfe":           (C_VFE,      "",     "VFE",                                       "gpu"),
    "gpu_scatter":       (C_SCATTER,  "",     "PointPillarScatter",                        "gpu"),
    "gpu_bev":           (C_BEV,      "",     "BEV backbone",                              "gpu"),
    "gpu_head":          (C_HEAD,     "",     "anchor head",                               "gpu"),
    "gpu_nms":           (C_NMS,      "",     "NMS (postprocess)",                         "gpu"),
    "gpu_idle":          (C_GPU_IDLE, "..",   "GPU idle  (18.2 W measured)",               "gpu"),
    "cpu_idle":          (C_CPU_IDLE, "--",   "CPU idle  (11.6 W measured)",               "cpu"),
}

# ── Measured constants ────────────────────────────────────────────────────────
GPU_IDLE_W        = 18.2
CPU_IDLE_PKG_W    = 11.6
PLATFORM_ACTIVE_W = 74.5   # psys − pkg − GPU during active window
PLATFORM_IDLE_W   = 30.0   # estimated at rest

# ── NVTX forward substage fractions ──────────────────────────────────────────
# "default": A10G modal_v3 nsys proxy — used for M0–M4 PyTorch variants.
# "trt":     measured locally on RTX 3080 Ti via nsys nvtx_gpu_proj_sum (FP16, 50 frames).
#   gpu_voxelize range (VFE+scatter): 128,133 ns avg
#   trt_forward range: 5,709,300 ns avg
#     PPScatter inside TRT: 176,393 ns  → gpu_scatter
#     MatMul_161 + myelinGraphExecute:  684,396 ns → gpu_head
#     remainder BEV convs:            4,848,511 ns → gpu_bev
#   Voxelize folded into gpu_vfe bucket (combined stage total = trt_fwd + voxelize)
NVTX_FRACS = {
    "default": {"gpu_vfe": 0.18,   "gpu_scatter": 0.08,   "gpu_bev": 0.55,   "gpu_head": 0.19},
    "trt":     {"gpu_vfe": 0.0220, "gpu_scatter": 0.0302, "gpu_bev": 0.8306, "gpu_head": 0.1172},
}

def split_forward(forward_mj, variant="default"):
    f = NVTX_FRACS[variant]
    return {k: forward_mj * v for k, v in f.items()}


# ── Edge variants — measured locally 2026-04-23 ───────────────────────────────
# latency breakdown from profile_suite; active powers from energy_monitor
LOCAL_VARIANTS = [
    dict(
        label="M0_FP32",
        gpu_W=40.2, cpu_W=63.18,
        read_ms=3.44, preproc_ms=8.85, h2d_ms=0.63,
        forward_ms=13.10, nms_ms=0.38,
        gpu_voxel=False,
    ),
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
    dict(
        label="M4_AMP",
        gpu_W=53.8, cpu_W=46.74,
        read_ms=3.92, preproc_ms=1.56, h2d_ms=0.25,
        forward_ms=8.23, nms_ms=0.40,
        gpu_voxel=True,
    ),
]

# ── M5: CUDA-PointPillars TRT — measured locally 2026-04-27 ──────────────────
# All measured locally on RTX 3080 Ti, 2026-04-27, 500 warmup / 500 steps.
# gpu_W, cpu_W: NVML + RAPL package (energy_rapl_measured=True).
# read_ms, voxelize_ms, forward_ms, nms_ms: cuda_pp_metrics 500-frame means.
TRT_VARIANTS = [
    dict(label="M5_FP32\n(TRT)", gpu_W=75.74, cpu_W=35.89, read_ms=0.18,
         voxelize_ms=0.10, forward_ms=6.05, nms_ms=0.65),
    dict(label="M5_FP16\n(TRT)", gpu_W=75.45, cpu_W=38.87, read_ms=0.18,
         voxelize_ms=0.10, forward_ms=5.75, nms_ms=0.65),
]


def trt_segments(v):
    # active = read + voxelize (GPU) + trt_forward + nms
    lat  = v['read_ms'] + v['voxelize_ms'] + v['forward_ms'] + v['nms_ms']
    idle = DUTY_MS - lat
    fwd_mj      = v['gpu_W'] * v['forward_ms']
    voxelize_mj = v['gpu_W'] * v['voxelize_ms']
    segs = {
        "platform_overhead": PLATFORM_ACTIVE_W * lat + PLATFORM_IDLE_W * idle,
        "read_points":       v['cpu_W'] * v['read_ms'],
        "pre_processing":    0.0,   # voxelization on GPU — counted in gpu_vfe
        "data_to_gpu":       0.0,
        "gpu_nms":           v['gpu_W'] * v['nms_ms'],
        "gpu_idle":          GPU_IDLE_W * idle,
        "cpu_idle":          CPU_IDLE_PKG_W * idle,
    }
    # forward substages from real TRT nsys fracs; voxelize folded into gpu_vfe
    total_mj = fwd_mj + voxelize_mj
    segs.update({k: total_mj * f for k, f in NVTX_FRACS['trt'].items()})
    return segs


# ── A10G cloud reference ──────────────────────────────────────────────────────
A10G = dict(
    label="A10G  M0_FP32\n(cloud, GPU measured\nCPU est.)",
    gpu_W=124.2, cpu_active_W=4.0, cpu_idle_W=2.0,
    read_ms=8.56, preproc_ms=17.06, h2d_ms=0.90,
    forward_ms=12.96, nms_ms=0.60,
)


def edge_segments(v):
    lat  = v['read_ms'] + v['preproc_ms'] + v['h2d_ms'] + v['forward_ms'] + v['nms_ms']
    idle = DUTY_MS - lat
    fwd_mj = v['gpu_W'] * v['forward_ms']
    segs = {
        "platform_overhead": PLATFORM_ACTIVE_W * lat + PLATFORM_IDLE_W * idle,
        "read_points":       v['cpu_W'] * v['read_ms'],
        "pre_processing":    v['cpu_W'] * v['preproc_ms'],
        "data_to_gpu":       v['cpu_W'] * v['h2d_ms'],
        "gpu_nms":           v['gpu_W'] * v['nms_ms'],
        "gpu_idle":          GPU_IDLE_W * idle,
        "cpu_idle":          CPU_IDLE_PKG_W * idle,
    }
    segs.update(split_forward(fwd_mj))
    return segs


def cloud_segments(c):
    lat  = c['read_ms'] + c['preproc_ms'] + c['h2d_ms'] + c['forward_ms'] + c['nms_ms']
    idle = DUTY_MS - lat
    fwd_mj = c['gpu_W'] * c['forward_ms']
    segs = {
        "platform_overhead": 0.0,
        "read_points":       c['cpu_active_W'] * c['read_ms'],
        "pre_processing":    c['cpu_active_W'] * c['preproc_ms'],
        "data_to_gpu":       c['cpu_active_W'] * c['h2d_ms'],
        "gpu_nms":           c['gpu_W'] * c['nms_ms'],
        "gpu_idle":          GPU_IDLE_W * idle,
        "cpu_idle":          c['cpu_idle_W'] * idle,
    }
    segs.update(split_forward(fwd_mj))
    return segs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='../../profile_outputs/amp_benchmark/report_figures')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Assemble rows ────────────────────────────────────────────────────────
    rows = []
    for v in LOCAL_VARIANTS:
        rows.append(dict(label=v['label'], segs=edge_segments(v),
                         cloud=False, trt=False))
    for v in TRT_VARIANTS:
        rows.append(dict(label=v['label'], segs=trt_segments(v),
                         cloud=False, trt=True))
    for r in rows:
        r['total'] = sum(r['segs'].get(s, 0) for s in SEG_ORDER)

    n_pytorch = len(LOCAL_VARIANTS)   # index where TRT rows start

    n     = len(rows)
    bar_h = 0.65
    fig_h = max(6.5, n * 1.15 + 2.2)
    y     = np.arange(n, dtype=float)

    XLIM  = 5000.0
    scale = XLIM / max(r['total'] for r in rows)

    with plt.rc_context(MLSYS_PLOT_RC):
        fig, (ax, ax_leg) = plt.subplots(
            1, 2, figsize=(16.0, fig_h),
            gridspec_kw=dict(width_ratios=[5, 1]))
        ax_leg.axis('off')

        # ── Draw bars ────────────────────────────────────────────────────────
        for ri, row in enumerate(rows):
            left = 0.0
            for seg in SEG_ORDER:
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

                ax.barh(y[ri], disp, height=bar_h, left=left,
                        color=color, edgecolor='white',
                        linewidth=0.5, hatch=hatch, alpha=0.92)

                if real > 280:
                    ax.text(left + disp / 2, y[ri],
                            f'{real/1000:.2f}J',
                            ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold',
                            clip_on=True)
                left += disp

            # End label
            base  = rows[0]['total']
            total = row['total']
            stot  = total * scale
            if ri == 0:
                lbl  = f"{total/1000:.2f} J  (baseline)"
                lcol = '#333'
            else:
                pct  = (base - total) / base * 100
                sign = '−' if pct >= 0 else '+'
                lbl  = f"{total/1000:.2f} J  {sign}{abs(pct):.0f}%"
                lcol = '#1b5e20' if pct > 0 else '#b71c1c'
            ax.text(stot + 55, y[ri], lbl,
                    ha='left', va='center', fontsize=9,
                    fontweight='bold', color=lcol)


        # ── PyTorch / TRT divider ─────────────────────────────────────────────
        divider_y = n_pytorch - 0.5
        ax.axhline(divider_y, color='#777', lw=1.1, ls='--', alpha=0.7)
        ax.text(XLIM * 0.35, divider_y,
                'PyTorch (M0–M4) ↑     ↓ TensorRT C++ (M5)',
                ha='center', va='center', fontsize=8, color='#555',
                style='italic',
                bbox=dict(boxstyle='round,pad=0.2', fc='#f5f5f5',
                          ec='none', alpha=0.9))

        # ── Annotations ──────────────────────────────────────────────────────
        # NMS on M0_FP32
        m0   = rows[0]
        nms_center = sum(m0['segs'].get(s, 0) for s in
                         ["platform_overhead","read_points","pre_processing",
                          "data_to_gpu","gpu_vfe","gpu_scatter","gpu_bev",
                          "gpu_head"]) * scale + \
                     m0['segs']['gpu_nms'] * scale / 2
        ax.annotate('NMS ≈70%\nof GPU time',
                    xy=(nms_center, y[0]),
                    xytext=(nms_center - XLIM * 0.05, y[0] - 0.35),
                    fontsize=7.5, color='#b71c1c', fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#b71c1c',
                                    lw=1.0, connectionstyle='arc3,rad=0.2'))

        # compile ↓ arrow M0_AMP → M1_AMP
        # ax.annotate('',
        #             xy=(rows[2]['total'] * scale, y[2]),
        #             xytext=(rows[1]['total'] * scale, y[1]),
        #             arrowprops=dict(arrowstyle='->', color='#1565C0',
        #                             lw=1.5, connectionstyle='arc3,rad=0.0'))

        # Calculate a vertical offset based on your font size/axis scale
        y_offset = 0.4 

        ax.annotate('',
            xy=(rows[2]['total'] * scale, y[2]),
            xytext=((rows[1]['total'] + rows[2]['total']) / 2 * scale + XLIM * 0.03, ((y[1] + y[2]) / 2) - y_offset),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5)
        )

        ax.text((rows[1]['total'] + rows[2]['total']) / 2 * scale + XLIM * 0.03,
                (y[1] + y[2]) / 2,
                'compile ↓', ha='left', va='center',
                fontsize=8, color='#1565C0', fontweight='bold')


        # ── CPU / GPU band labels on y-axis side ─────────────────────────────
        ax.text(-XLIM * 0.01, -0.5, 'CPU', ha='right', va='center',
                fontsize=8, color='#a0522d', fontweight='bold',
                transform=ax.transData)

        # ── Axes ─────────────────────────────────────────────────────────────
        ax.set_yticks(y)
        ax.set_yticklabels([r['label'] for r in rows], fontsize=10)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(1000 * scale))
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: f'{x/scale/1000:.0f}'))
        ax.set_xlabel('Energy per 10 Hz duty cycle  (J / 100 ms frame)',
                      fontsize=11)
        ax.set_xlim(0, XLIM * 1.32)
        ax.set_title(
            'Holistic 10 Hz LiDAR inference energy budget  '
            '(active inference + idle per duty cycle)\n'
            'RAPL pkg (CPU)  ·  NVML (GPU)  ·  all values measured  '
            '|  hatch pattern shows phase type (platform / CPU active / CPU idle / GPU active / GPU idle)',
            fontsize=10.5, fontweight='bold', pad=8, loc='left')

        # ── Segment legend (top) ──────────────────────────────────────────────
        def seg_patch(key):
            color, hatch, label, _ = SEG_META[key]
            ec = '#888' if hatch else '#ccc'
            return mpatches.Patch(facecolor=color, hatch=hatch,
                                  edgecolor=ec, linewidth=0.5, label=label)

        seg_patches = (
            [mpatches.Patch(visible=False, label='── Platform ──')] +
            [seg_patch('platform_overhead')] +
            [mpatches.Patch(visible=False, label='── CPU active ──')] +
            [seg_patch(s) for s in ["read_points", "pre_processing", "data_to_gpu"]] +
            [mpatches.Patch(visible=False, label='── CPU idle ──')] +
            [seg_patch('cpu_idle')] +
            [mpatches.Patch(visible=False, label='── GPU active ──')] +
            [seg_patch(s) for s in ["gpu_vfe", "gpu_scatter", "gpu_bev", "gpu_head", "gpu_nms"]] +
            [mpatches.Patch(visible=False, label='── GPU idle ──')] +
            [seg_patch('gpu_idle')]
        )
        ax_leg.legend(handles=seg_patches,
                             title='Pipeline segment', title_fontsize=9,
                             fontsize=7.5, loc='upper left',
                      framealpha=0.0, edgecolor='none', borderaxespad=0)

        # ROI box
        base_total  = rows[0]['total']
        gpu_frac    = (sum(rows[0]['segs'].get(s, 0)
                           for s in ["gpu_vfe","gpu_scatter","gpu_bev",
                                     "gpu_head","gpu_nms"])
                       / base_total * 100)
        best_saving = (base_total - min(r['total'] for r in rows)
                       ) / base_total * 100
        roi = (f"GPU compute = {gpu_frac:.0f}%\n"
               f"of total system energy\n"
               f"(M0_FP32 baseline)\n\n"
               f"Best SW saving:\n"
               f"~{best_saving:.0f}% of total\n\n"
               f"→ Idle + platform\n"
               f"dominate; SW opt.\n"
               f"alone limited ROI.")
        ax_leg.text(0.02, 0.5, roi,
                    transform=ax_leg.transAxes,
                    ha='left', va='top', fontsize=7.5,
                    color='#1b5e20', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.45', fc='#f1f8e9',
                              ec='#2e7d32', alpha=0.92))

        # ── Footer ───────────────────────────────────────────────────────────
        fig.text(
            0.5, 0.005,
            'M0–M4: RTX 3080 Ti, 2026-04-23; RAPL pkg (CPU) + NVML (GPU); '
            'forward substage fracs from A10G nsys proxy. '
            'M5: CUDA-PointPillars TRT, 2026-04-27; GPU power NVML; forward/voxelize/NMS from local nsys; '
            'CPU power from RAPL pkg; read_points from C++ binary timing. '
            'GPU idle = 18.2 W, CPU idle = 11.6 W. Platform ≈ 74.5 W active / 30 W idle (est.).',
            ha='center', fontsize=7.5, color='#666', style='italic')

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        path = out / 'energy_duty_cycle.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', path)


if __name__ == '__main__':
    main()
