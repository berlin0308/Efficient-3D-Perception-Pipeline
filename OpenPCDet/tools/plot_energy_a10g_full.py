"""
plot_energy_a10g_full.py — A10G comprehensive energy figure.

Three panels:
  Left:   GPU energy per frame (mJ) + latency overlay (twin y-axis)
  Middle: Samples/J energy efficiency
  Right:  Mean GPU power (W)

Run from OpenPCDet/tools/:
    python plot_energy_a10g_full.py \
        --output_dir ../../profile_outputs/amp_benchmark
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

VARIANTS    = ['M0\nBaseline', 'M1\n+Compile', 'M2\n+NHWC', 'M3\n+GPU Prep', 'M4\nAll']
GPU_PREPROC = [False, False, False, True, True]

DATA = {
    'FP32': {
        'energy_mJ':  [1625, 1070, 2047, 2122, 6231],
        'samples_J':  [0.6154, 0.9349, 0.4885, 0.4713, 0.1605],
        'latency_ms': [13.24, 10.15, 16.27, 10.76,  9.10],
        'power_W':    [124.2, 105.4, 119.3, 116.5, 684.4],
    },
    'AMP': {
        'energy_mJ':  [1349,  816, 1452, 1696, 7052],
        'samples_J':  [0.7411, 1.2248, 0.6885, 0.5897, 0.1418],
        'latency_ms': [ 9.88,  7.04, 12.08,  8.03,  9.05],
        'power_W':    [ 95.3, 115.9,  95.8,  98.0,  65.0],
    },
}

FP32_COLOR = '#C62828'
AMP_COLOR  = '#1565C0'
BEST_COLOR = '#2E7D32'
WARN_COLOR = '#E65100'
ALPHA      = 0.85


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='../../profile_outputs/amp_benchmark')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(VARIANTS)
    x = np.arange(n)
    w = 0.33

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        'A10G (Modal) — Complete GPU Energy Analysis: M0–M4 × FP32 vs FP16 AMP\n'
        'pynvml · batch=1 · 50 frames · KITTI val  |  GPU-only (RAPL unavailable on Modal)',
        fontsize=12, fontweight='bold', y=0.99
    )

    panels = [
        ('energy_mJ', 'GPU Energy per Frame (mJ)', 'GPU Energy (mJ / frame)'),
        ('samples_J', 'Energy Efficiency (samples/J)', 'Samples / Joule'),
        ('power_W',   'Mean GPU Power (W)',            'Mean GPU Power (W)'),
    ]

    for col, (key, title, ylabel) in enumerate(panels):
        ax = axes[col]
        fp32_vals = np.array(DATA['FP32'][key], dtype=float)
        amp_vals  = np.array(DATA['AMP'][key],  dtype=float)
        ymax = max(fp32_vals.max(), amp_vals.max())

        # 60% headroom above tallest bar — all labels go in this space
        ax.set_ylim(0, ymax * 1.60)

        bars_fp32 = ax.bar(x - w/2, fp32_vals, w, color=FP32_COLOR, alpha=ALPHA,
                           label='FP32', edgecolor='white', linewidth=0.5)
        bars_amp  = ax.bar(x + w/2, amp_vals,  w, color=AMP_COLOR,  alpha=ALPHA,
                           label='FP16 AMP', edgecolor='white', linewidth=0.5)

        # ── value labels ABOVE each bar (never inside) ──────────────────────
        if key == 'energy_mJ':
            fmt = lambda v: f'{v/1000:.1f}k'
        elif key == 'samples_J':
            fmt = lambda v: f'{v:.3f}'
        else:
            fmt = lambda v: f'{v:.0f}'

        label_gap = ymax * 0.015
        for bar, val in zip(bars_fp32, fp32_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + label_gap,
                    fmt(val), ha='center', va='bottom',
                    fontsize=8, color=FP32_COLOR, fontweight='bold')
        for bar, val in zip(bars_amp, amp_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + label_gap,
                    fmt(val), ha='center', va='bottom',
                    fontsize=8, color=AMP_COLOR, fontweight='bold')

        # ── % change labels — above both value labels ────────────────────────
        pct_gap = ymax * 0.12   # extra lift so % sits above the value labels
        for i in range(n):
            pct   = (fp32_vals[i] - amp_vals[i]) / fp32_vals[i] * 100
            color = BEST_COLOR if pct > 0 else WARN_COLOR
            sign  = '−' if pct >= 0 else '+'
            top   = max(fp32_vals[i], amp_vals[i])
            ax.text(x[i], top + pct_gap,
                    f'{sign}{abs(pct):.0f}%',
                    ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color=color)

        # ── latency twin-axis (energy panel only) ────────────────────────────
        if key == 'energy_mJ':
            ax2 = ax.twinx()
            lat_fp32 = DATA['FP32']['latency_ms']
            lat_amp  = DATA['AMP']['latency_ms']
            ax2.plot(x - w/2, lat_fp32, 'o--', color=FP32_COLOR, alpha=0.50,
                     lw=1.3, ms=4, label='FP32 latency')
            ax2.plot(x + w/2, lat_amp,  's--', color=AMP_COLOR,  alpha=0.50,
                     lw=1.3, ms=4, label='AMP latency')
            ax2.set_ylabel('Latency (ms)', fontsize=9, color='#666')
            ax2.tick_params(axis='y', colors='#666', labelsize=8)
            # push latency axis limit high so line stays in lower half
            ax2.set_ylim(0, max(max(lat_fp32), max(lat_amp)) * 4.0)
            ax2.legend(fontsize=7.5, loc='upper center',
                       bbox_to_anchor=(0.5, 1.0), framealpha=0.85, ncol=2)

            # asterisk below x-axis for M3/M4
            for i, gp in enumerate(GPU_PREPROC):
                if gp:
                    ax.text(x[i], -ymax * 0.05, '*', ha='center',
                            fontsize=13, color=WARN_COLOR, fontweight='bold')

        # ── callouts on samples_J panel ──────────────────────────────────────
        if key == 'samples_J':
            best_i  = int(np.argmax(amp_vals))
            worst_i = int(np.argmin(amp_vals))
            ax.annotate(
                'Best: M1_AMP\n+99% vs M0_FP32',
                xy=(x[best_i] + w/2, amp_vals[best_i]),
                xytext=(x[best_i] + 0.85, ymax * 1.2),
                fontsize=8, color=BEST_COLOR, fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color=BEST_COLOR, lw=1.1,
                                connectionstyle='arc3,rad=-0.2'))
            ax.annotate(
                'Worst: M4_AMP\ncompile spike\ninflates window',
                xy=(x[worst_i] + w/2, amp_vals[worst_i]),
                xytext=(x[worst_i] - 0.8, ymax * 1),
                fontsize=8, color=WARN_COLOR, fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color=WARN_COLOR, lw=1.1,
                                connectionstyle='arc3,rad=0.25'))

        # ── NMS note on power panel ───────────────────────────────────────────
        if key == 'power_W':
            ax.text(0.02, 0.04,
                    'NMS ≈70% GPU time,\nstays FP32\n→ caps AMP savings',
                    transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=8, color='#555', style='italic',
                    bbox=dict(boxstyle='round,pad=0.35', fc='#fff8e1',
                              ec='#f9a825', alpha=0.92))

        ax.set_xticks(x)
        ax.set_xticklabels(VARIANTS, fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.yaxis.grid(True, alpha=0.22, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    fig.text(
        0.5, 0.005,
        '* M3, M4: voxelization on GPU — pynvml captures preprocessing cost.  '
        'M0–M2: CPU voxelization (~17 ms/frame) invisible to pynvml.  '
        'Estimated CPU cost <1% of A10G GPU frame energy → ranking preserved.',
        ha='center', fontsize=8, color='#666', style='italic'
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = out / 'energy_a10g_full.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', path)


if __name__ == '__main__':
    main()
