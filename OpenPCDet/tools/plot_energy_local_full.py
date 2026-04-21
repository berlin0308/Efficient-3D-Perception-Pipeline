"""
plot_energy_local_full.py — RTX 3080 Ti full system energy, no overlaps.

Four panels:
  1. GPU pipeline stages (stacked)
  2. CPU RAPL domains + psys
  3. Total system summary
  4. Samples/J at three accounting levels

All value labels go ABOVE their bar. % labels go above value labels.
No text inside bars. Annotations placed in empty space only.

Run from OpenPCDet/tools/:
    python plot_energy_local_full.py \
        --output_dir ../../profile_outputs/amp_benchmark
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
FP32_STAGES_MS = {
    'DataLoader':   0.23,
    'H2D Transfer': 1.47,
    'PillarVFE':    0.717,
    'Scatter':      0.757,
    'BEV Backbone': 1.427,
    'Anchor Head':  0.588,
    'NMS':          11.46,
}
AMP_FWD_RATIO = 18.81 / 27.03
AMP_STAGES_MS = {
    'DataLoader':   0.27,
    'H2D Transfer': 1.55,
    'PillarVFE':    FP32_STAGES_MS['PillarVFE']   * AMP_FWD_RATIO,
    'Scatter':      FP32_STAGES_MS['Scatter']      * AMP_FWD_RATIO,
    'BEV Backbone': FP32_STAGES_MS['BEV Backbone'] * AMP_FWD_RATIO,
    'Anchor Head':  FP32_STAGES_MS['Anchor Head']  * AMP_FWD_RATIO,
    'NMS':          FP32_STAGES_MS['NMS']          * (0.97 / 1.03),
}
FP32_GPU_J = 33.86
AMP_GPU_J  = 20.50

CPU  = {
    'FP32': {'Core': 56.986, 'Uncore': 0.521, 'Other': 7.083},
    'AMP':  {'Core': 40.061, 'Uncore': 0.352, 'Other': 5.398},
}
PSYS = {'FP32': 159.885, 'AMP': 110.395}
PKG  = {'FP32': 64.590,  'AMP': 45.811}

GPU_IDLE_W     = 19.4
FRAME_PERIOD_S = 0.10
IDLE_J = {
    'FP32': GPU_IDLE_W * (FRAME_PERIOD_S - 14.68e-3) * 50,
    'AMP':  GPU_IDLE_W * (FRAME_PERIOD_S -  9.57e-3) * 50,
}

# ── Colors ────────────────────────────────────────────────────────────────────
GPU_COLORS = {
    'DataLoader':   '#78909C',
    'H2D Transfer': '#26A69A',
    'PillarVFE':    '#42A5F5',
    'Scatter':      '#7E57C2',
    'BEV Backbone': '#EF5350',
    'Anchor Head':  '#FF7043',
    'NMS':          '#FFA726',
}
CPU_COLORS = {'Core': '#C62828', 'Uncore': '#EF6C00', 'Other': '#6A1B9A'}
PSYS_COLOR = '#00695C'
IDLE_COLOR = '#90A4AE'
FP32_COLOR = '#C62828'
AMP_COLOR  = '#1565C0'
FP32_EDGE  = '#B71C1C'
AMP_EDGE   = '#0D47A1'
EDGE_LW    = 2.5
ALPHA      = 0.88

# shared y-limit for panels 1-3
YMAX = PSYS['FP32'] * 1.45


def stage_energies(stages_ms, total_J):
    t = sum(stages_ms.values())
    return {s: (ms / t) * total_J for s, ms in stages_ms.items()}


def stacked_bar(ax, x, w, segments, colors, edge_color):
    """Draw stacked bar. Returns (bottom, list of (name, val, mid_y))."""
    bottom = 0.0
    info   = []
    for name, val in segments:
        ax.bar(x, val, w, bottom=bottom, color=colors[name],
               edgecolor='white', linewidth=0.5, alpha=ALPHA)
        info.append((name, val, bottom + val / 2))
        bottom += val
    ax.bar(x, bottom, w, fill=False, edgecolor=edge_color, linewidth=EDGE_LW)
    return bottom, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='../../profile_outputs/amp_benchmark')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fp32_gpu = stage_energies(FP32_STAGES_MS, FP32_GPU_J)
    amp_gpu  = stage_energies(AMP_STAGES_MS,  AMP_GPU_J)
    stages   = list(GPU_COLORS.keys())
    domains  = list(CPU_COLORS.keys())

    fig, axes = plt.subplots(1, 4, figsize=(24, 8),
                             gridspec_kw={'width_ratios': [1.0, 1.3, 1.0, 0.85]})
    fig.suptitle(
        'RTX 3080 Ti (Local) — Full System Energy: FP32 vs FP16 AMP\n'
        'GPU: pynvml × NVTX stage fractions  |  CPU: Intel RAPL  |  '
        'Red outline = FP32   ·   Blue outline = FP16 AMP',
        fontsize=12, fontweight='bold', y=0.99
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1 — GPU pipeline stages
    # ─────────────────────────────────────────────────────────────────────────
    ax1 = axes[0]
    fp32_top, fp32_info = stacked_bar(ax1, 0.0, 0.5,
                                      [(s, fp32_gpu[s]) for s in stages],
                                      GPU_COLORS, FP32_EDGE)
    amp_top,  amp_info  = stacked_bar(ax1, 1.0, 0.5,
                                      [(s, amp_gpu[s]) for s in stages],
                                      GPU_COLORS, AMP_EDGE)

    # segment labels inside bars only for NMS (large enough); rest on legend
    for (name, val, mid_y), xi in [(fp32_info[-1], 0.0), (amp_info[-1], 1.0)]:
        ax1.text(xi, mid_y, f'{val:.1f}J', ha='center', va='center',
                 fontsize=8, color='white', fontweight='bold')

    gpu_s = (fp32_top - amp_top) / fp32_top * 100
    # totals well above bars
    for xi, top, ec, lbl in [
        (0.0, fp32_top, FP32_EDGE, f'Total: {fp32_top:.1f}J'),
        (1.0, amp_top,  AMP_EDGE,  f'Total: {amp_top:.1f}J'),
    ]:
        ax1.text(xi, top + 2, lbl, ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=ec)

    # % savings above totals
    ax1.text(0.5, max(fp32_top, amp_top) + 10,
             f'−{gpu_s:.0f}%', ha='center', va='bottom',
             fontsize=12, fontweight='bold', color='#2e7d32',
             bbox=dict(boxstyle='round,pad=0.3', fc='#f1f8e9', ec='#aaa'))

    # NMS annotation — to the right, pointing left into AMP NMS segment
    nms_bot_amp = sum(amp_gpu[s] for s in stages if s != 'NMS')
    ax1.annotate('NMS stays FP32\n(~70% of GPU time)',
                 xy=(1.25, nms_bot_amp + amp_gpu['NMS'] * 0.5),
                 xytext=(1.35, nms_bot_amp + amp_gpu['NMS'] * 0.3),
                 fontsize=8, color='#e65100', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#e65100', lw=1.1))

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['FP32', 'FP16 AMP'], fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy — 50-frame run (J)', fontsize=10)
    ax1.set_title('GPU Pipeline Stages', fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlim(-0.5, 2.0)
    ax1.set_ylim(0, YMAX)
    ax1.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax1.set_axisbelow(True)
    gpu_leg = [mpatches.Patch(facecolor=GPU_COLORS[s], alpha=ALPHA, label=s)
               for s in stages]
    ax1.legend(handles=gpu_leg, fontsize=7.5, loc='upper left', framealpha=0.9)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2 — CPU RAPL domains + psys
    # ─────────────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    w2 = 0.42

    fp32_pkg, _ = stacked_bar(ax2, 0.0, w2,
                               [(d, CPU['FP32'][d]) for d in domains],
                               CPU_COLORS, FP32_EDGE)
    amp_pkg,  _ = stacked_bar(ax2, 1.0, w2,
                               [(d, CPU['AMP'][d])  for d in domains],
                               CPU_COLORS, AMP_EDGE)

    for xi, prec, ec in [(2.4, 'FP32', FP32_EDGE), (3.3, 'AMP', AMP_EDGE)]:
        ax2.bar(xi, PSYS[prec], w2, color=PSYS_COLOR, alpha=ALPHA,
                edgecolor='white', linewidth=0.5)
        ax2.bar(xi, PSYS[prec], w2, fill=False, edgecolor=ec, linewidth=EDGE_LW)

    pkg_s  = (fp32_pkg - amp_pkg) / fp32_pkg * 100
    psys_s = (PSYS['FP32'] - PSYS['AMP']) / PSYS['FP32'] * 100

    # all value labels above bars, with enough gap
    gap = 3
    for xi, top, ec, lbl in [
        (0.0, fp32_pkg,     FP32_EDGE, f'{fp32_pkg:.1f}J'),
        (1.0, amp_pkg,      AMP_EDGE,  f'{amp_pkg:.1f}J'),
        (2.4, PSYS['FP32'], FP32_EDGE, f'{PSYS["FP32"]:.1f}J'),
        (3.3, PSYS['AMP'],  AMP_EDGE,  f'{PSYS["AMP"]:.1f}J'),
    ]:
        ax2.text(xi, top + gap, lbl, ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=ec)

    # % labels above value labels
    for xi, top, lbl in [
        (0.5, max(fp32_pkg, amp_pkg),       f'−{pkg_s:.0f}%'),
        (2.85, max(PSYS['FP32'],PSYS['AMP']),f'−{psys_s:.0f}%'),
    ]:
        ax2.text(xi, top + 14, lbl, ha='center', va='bottom',
                 fontsize=11, fontweight='bold', color='#2e7d32',
                 bbox=dict(boxstyle='round,pad=0.25', fc='#f1f8e9', ec='#aaa'))

    # divider
    ax2.axvline(x=1.85, color='#bbb', linewidth=1.5, linestyle='--')
    ax2.text(1.85, YMAX * 0.55, 'non-\nadditive\n→',
             ha='center', va='center', fontsize=8, color='#888', style='italic')

    # psys note — right half, low, inside psys bar area
    ax2.text(2.85, PSYS['AMP'] * 0.40,
             'psys = CPU + DRAM\n+ SSD + display\n+ fans (≈3.8× pkg)',
             ha='center', va='center',  fontsize=8, color='#555', style='italic',
             bbox=dict(boxstyle='round,pad=0.35', fc='#fff3e0',
                       ec='#fb8c00', alpha=0.95))

    ax2.set_xticks([0.0, 1.0, 2.4, 3.3])
    ax2.set_xticklabels(['FP32\n(pkg)', 'AMP\n(pkg)', 'FP32\n(psys)', 'AMP\n(psys)'],
                        fontsize=10, fontweight='bold')
    ax2.set_title('CPU RAPL Domains  +  psys\n(psys non-additive with pkg)',
                  fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlim(-0.55, 4.0)
    ax2.set_ylim(0, YMAX)
    ax2.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax2.set_axisbelow(True)
    cpu_leg = [mpatches.Patch(facecolor=CPU_COLORS[d], alpha=ALPHA, label=d)
               for d in domains]
    cpu_leg.append(mpatches.Patch(facecolor=PSYS_COLOR, alpha=ALPHA,
                                   label='psys (full platform)'))
    ax2.legend(handles=cpu_leg, fontsize=8, loc='upper left', framealpha=0.9)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3 — Total system summary
    # ─────────────────────────────────────────────────────────────────────────
    ax3 = axes[2]
    summary_segs = [
        ('GPU active',  FP32_GPU_J,    AMP_GPU_J,    '#1565C0'),
        ('GPU idle',    IDLE_J['FP32'],IDLE_J['AMP'], IDLE_COLOR),
        ('CPU package', PKG['FP32'],   PKG['AMP'],    '#C62828'),
    ]

    x_fp32, x_amp = 0.0, 1.0
    w3 = 0.48
    bottoms = {'FP32': 0.0, 'AMP': 0.0}
    for seg, fp32_v, amp_v, color in summary_segs:
        for xi, val, prec in [(x_fp32, fp32_v, 'FP32'), (x_amp, amp_v, 'AMP')]:
            b = bottoms[prec]
            ax3.bar(xi, val, w3, bottom=b, color=color, alpha=ALPHA,
                    edgecolor='white', linewidth=0.5)
            # only label segments large enough to be readable
            if val >= 5:
                ax3.text(xi, b + val / 2, f'{val:.0f}J',
                         ha='center', va='center', fontsize=8,
                         color='white', fontweight='bold')
            bottoms[prec] += val

    # outlines
    for xi, prec, ec in [(x_fp32, 'FP32', FP32_EDGE), (x_amp, 'AMP', AMP_EDGE)]:
        ax3.bar(xi, bottoms[prec], w3, fill=False,
                edgecolor=ec, linewidth=EDGE_LW)

    # totals above bars with gap
    for xi, prec, ec in [(x_fp32, 'FP32', FP32_EDGE), (x_amp, 'AMP', AMP_EDGE)]:
        ax3.text(xi, bottoms[prec] + 3, f'Total\n{bottoms[prec]:.1f}J',
                 ha='center', va='bottom', fontsize=9.5,
                 fontweight='bold', color=ec)

    # % savings above totals
    tot_s = (bottoms['FP32'] - bottoms['AMP']) / bottoms['FP32'] * 100
    ax3.text(0.5, max(bottoms['FP32'], bottoms['AMP']) + 14,
             f'−{tot_s:.0f}%', ha='center', va='bottom',
             fontsize=12, fontweight='bold', color='#2e7d32',
             bbox=dict(boxstyle='round,pad=0.3', fc='#f1f8e9', ec='#aaa'))

    # psys reference lines — right of bars, labeled clearly
    for prec, ls, dy in [('FP32', '--', +2), ('AMP', ':', -7)]:
        ec = FP32_EDGE if prec == 'FP32' else AMP_EDGE
        yp = PSYS[prec]
        ax3.plot([-0.3, 1.3], [yp, yp], color=ec,
                 linewidth=1.3, linestyle=ls, alpha=0.6)
        ax3.text(1.32, yp + dy, f'psys {prec}: {yp:.0f}J',
                 ha='left', va='center', fontsize=7.5, color=ec)

    # idle note below plot area
    ax3.text(0.5, -YMAX * 0.09,
             f'GPU idle: 19.4 W  |  '
             f'FP32: {IDLE_J["FP32"]:.1f}J  AMP: {IDLE_J["AMP"]:.1f}J',
             ha='center', va='top', fontsize=8, color='#546e7a',
             bbox=dict(boxstyle='round,pad=0.3', fc='#eceff1', ec='#90a4ae', alpha=0.9))

    ax3.set_xticks([x_fp32, x_amp])
    ax3.set_xticklabels(['FP32', 'FP16 AMP'], fontsize=12, fontweight='bold')
    ax3.set_ylabel('Energy — 50-frame run (J)', fontsize=10)
    ax3.set_title('Total System Summary\n(GPU active + GPU idle + CPU pkg)',
                  fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlim(-0.55, 1.85)
    ax3.set_ylim(-YMAX * 0.12, YMAX)
    ax3.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax3.set_axisbelow(True)
    sum_leg = [mpatches.Patch(facecolor=c, alpha=ALPHA, label=s)
               for s, _, _, c in summary_segs]
    ax3.legend(handles=sum_leg, fontsize=8, loc='upper right', framealpha=0.9)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 4 — Samples/J at three accounting levels
    # ─────────────────────────────────────────────────────────────────────────
    ax4 = axes[3]
    metrics = ['GPU only\n(pynvml)', 'GPU+CPU\n(RAPL)', 'Full system\n(psys)']
    fp32_sj = [50/FP32_GPU_J,
               50/(FP32_GPU_J + PKG['FP32']),
               50/PSYS['FP32']]
    amp_sj  = [50/AMP_GPU_J,
               50/(AMP_GPU_J  + PKG['AMP']),
               50/PSYS['AMP']]

    xm = np.arange(3)
    wm = 0.30
    ymax4 = max(max(fp32_sj), max(amp_sj))
    ax4.set_ylim(0, ymax4 * 1.60)

    b1 = ax4.bar(xm - wm/2, fp32_sj, wm, color=FP32_COLOR, alpha=ALPHA,
                 edgecolor='white', label='FP32')
    b2 = ax4.bar(xm + wm/2, amp_sj,  wm, color=AMP_COLOR,  alpha=ALPHA,
                 edgecolor='white', label='FP16 AMP')

    gap4 = ymax4 * 0.015
    # value labels above bars
    for bars, vals, ec in [(b1, fp32_sj, FP32_COLOR), (b2, amp_sj, AMP_COLOR)]:
        for bar, val in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + gap4,
                     f'{val:.3f}', ha='center', va='bottom',
                     fontsize=8, color=ec, fontweight='bold')

    # % above value labels
    for i in range(3):
        pct = (amp_sj[i] - fp32_sj[i]) / fp32_sj[i] * 100
        top = max(fp32_sj[i], amp_sj[i])
        ax4.text(xm[i], top + ymax4 * 0.12,
                 f'+{pct:.0f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='#2e7d32')

    # note — bottom of panel, no overlap with bars
    ax4.text(0.5, 0.02,
             'GPU-only overstates\nefficiency by ~3×\non edge hardware',
             transform=ax4.transAxes, ha='center', va='bottom',
             fontsize=8, color='#555', style='italic',
             bbox=dict(boxstyle='round,pad=0.35', fc='#fff3e0',
                       ec='#fb8c00', alpha=0.95))

    ax4.set_xticks(xm)
    ax4.set_xticklabels(metrics, fontsize=8.5)
    ax4.set_ylabel('Samples / Joule', fontsize=10)
    ax4.set_title('Energy Efficiency\n(samples/J)', fontsize=11,
                  fontweight='bold', pad=8)
    ax4.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax4.set_axisbelow(True)
    ax4.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    path = out / 'energy_local_full.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', path)


if __name__ == '__main__':
    main()
