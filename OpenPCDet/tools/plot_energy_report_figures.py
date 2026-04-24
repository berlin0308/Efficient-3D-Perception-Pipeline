"""
plot_energy_report_figures.py — Individual panel figures for MLSys report.

Saves each panel as a separate PNG to:
    profile_outputs/amp_benchmark/report_figures/

A10G panels:
  a10g_energy_per_frame.png   — GPU energy/frame (mJ) + latency overlay
  a10g_samples_per_joule.png  — Energy efficiency (samples/J)
  a10g_mean_power.png         — Mean GPU power (W)

Local RTX 3080 Ti panels:
  local_gpu_stages.png        — GPU pipeline stage breakdown
  local_cpu_rapl.png          — CPU RAPL domains + psys
  local_total_system.png      — Total system: GPU active + idle + CPU
  local_samples_per_joule.png — Samples/J at three accounting levels

Run from OpenPCDet/tools/:
    python plot_energy_report_figures.py \
        --output_dir ../../profile_outputs/amp_benchmark/report_figures
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Shared style ──────────────────────────────────────────────────────────────
FP32_COLOR = '#C62828'
AMP_COLOR  = '#1565C0'
FP32_EDGE  = '#B71C1C'
AMP_EDGE   = '#0D47A1'
BEST_COLOR = '#2E7D32'
WARN_COLOR = '#E65100'
EDGE_LW    = 2.5
ALPHA      = 0.88

# ── A10G data ─────────────────────────────────────────────────────────────────
VARIANTS    = ['M0\nBaseline', 'M1\n+Compile', 'M2\n+NHWC', 'M3\n+GPU Prep', 'M4\nAll']
GPU_PREPROC = [False, False, False, True, True]

A10G = {
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

# ── Local RTX 3080 Ti data ────────────────────────────────────────────────────
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
YMAX_LOCAL = PSYS['FP32'] * 1.45


def save(fig, path, name):
    p = path / name
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', p)


def stage_energies(stages_ms, total_J):
    t = sum(stages_ms.values())
    return {s: (ms / t) * total_J for s, ms in stages_ms.items()}


def stacked_bar(ax, x, w, segments, colors, edge_color):
    bottom = 0.0
    info   = []
    for name, val in segments:
        ax.bar(x, val, w, bottom=bottom, color=colors[name],
               edgecolor='white', linewidth=0.5, alpha=ALPHA)
        info.append((name, val, bottom + val / 2))
        bottom += val
    ax.bar(x, bottom, w, fill=False, edgecolor=edge_color, linewidth=EDGE_LW)
    return bottom, info


# ═════════════════════════════════════════════════════════════════════════════
# A10G panels
# ═════════════════════════════════════════════════════════════════════════════

def plot_a10g_energy(out):
    n = len(VARIANTS)
    x = np.arange(n)
    w = 0.33
    fp32_vals = np.array(A10G['FP32']['energy_mJ'], dtype=float)
    amp_vals  = np.array(A10G['AMP']['energy_mJ'],  dtype=float)
    ymax = max(fp32_vals.max(), amp_vals.max())

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        'A10G (Modal) — GPU Energy per Frame\npynvml · batch=1 · 50 frames · KITTI val',
        fontsize=11, fontweight='bold'
    )
    ax.set_ylim(0, ymax * 1.60)

    bars_fp32 = ax.bar(x - w/2, fp32_vals, w, color=FP32_COLOR, alpha=ALPHA,
                       label='FP32', edgecolor='white', linewidth=0.5)
    bars_amp  = ax.bar(x + w/2, amp_vals,  w, color=AMP_COLOR,  alpha=ALPHA,
                       label='FP16 AMP', edgecolor='white', linewidth=0.5)

    label_gap = ymax * 0.015
    for bar, val in zip(bars_fp32, fp32_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_gap,
                f'{val/1000:.1f}k', ha='center', va='bottom',
                fontsize=8, color=FP32_COLOR, fontweight='bold')
    for bar, val in zip(bars_amp, amp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_gap,
                f'{val/1000:.1f}k', ha='center', va='bottom',
                fontsize=8, color=AMP_COLOR, fontweight='bold')

    pct_gap = ymax * 0.12
    for i in range(n):
        pct   = (fp32_vals[i] - amp_vals[i]) / fp32_vals[i] * 100
        color = BEST_COLOR if pct > 0 else WARN_COLOR
        sign  = '−' if pct >= 0 else '+'
        top   = max(fp32_vals[i], amp_vals[i])
        ax.text(x[i], top + pct_gap, f'{sign}{abs(pct):.0f}%',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=color)

    # latency twin-axis
    ax2 = ax.twinx()
    lat_fp32 = A10G['FP32']['latency_ms']
    lat_amp  = A10G['AMP']['latency_ms']
    ax2.plot(x - w/2, lat_fp32, 'o--', color=FP32_COLOR, alpha=0.50, lw=1.3, ms=4, label='FP32 latency')
    ax2.plot(x + w/2, lat_amp,  's--', color=AMP_COLOR,  alpha=0.50, lw=1.3, ms=4, label='AMP latency')
    ax2.set_ylabel('Latency (ms)', fontsize=9, color='#666')
    ax2.tick_params(axis='y', colors='#666', labelsize=8)
    ax2.set_ylim(0, max(max(lat_fp32), max(lat_amp)) * 4.0)
    ax2.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, 1.0), framealpha=0.85, ncol=2)

    for i, gp in enumerate(GPU_PREPROC):
        if gp:
            ax.text(x[i], -ymax * 0.05, '*', ha='center',
                    fontsize=13, color=WARN_COLOR, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, fontsize=9.5)
    ax.set_ylabel('GPU Energy (mJ / frame)', fontsize=10)
    ax.set_title('GPU Energy per Frame (mJ) + Latency Overlay', fontsize=11, fontweight='bold', pad=8)
    ax.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    fig.text(0.5, 0.01,
             '* M3, M4: GPU voxelization counted in pynvml.  M0–M2: CPU voxelization (~17 ms/frame) invisible to pynvml.  <1% of A10G frame energy.',
             ha='center', fontsize=7.5, color='#666', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    save(fig, out, 'a10g_energy_per_frame.png')


def plot_a10g_samples_j(out):
    n = len(VARIANTS)
    x = np.arange(n)
    w = 0.33
    fp32_vals = np.array(A10G['FP32']['samples_J'], dtype=float)
    amp_vals  = np.array(A10G['AMP']['samples_J'],  dtype=float)
    ymax = max(fp32_vals.max(), amp_vals.max())

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        'A10G (Modal) — Energy Efficiency (samples/J)\npynvml · batch=1 · 50 frames · KITTI val  |  Higher = better',
        fontsize=11, fontweight='bold'
    )
    ax.set_ylim(0, ymax * 1.60)

    bars_fp32 = ax.bar(x - w/2, fp32_vals, w, color=FP32_COLOR, alpha=ALPHA,
                       label='FP32', edgecolor='white', linewidth=0.5)
    bars_amp  = ax.bar(x + w/2, amp_vals,  w, color=AMP_COLOR,  alpha=ALPHA,
                       label='FP16 AMP', edgecolor='white', linewidth=0.5)

    label_gap = ymax * 0.015
    for bar, val in zip(bars_fp32, fp32_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_gap,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=8, color=FP32_COLOR, fontweight='bold')
    for bar, val in zip(bars_amp, amp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_gap,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=8, color=AMP_COLOR, fontweight='bold')

    pct_gap = ymax * 0.12
    for i in range(n):
        pct   = (fp32_vals[i] - amp_vals[i]) / fp32_vals[i] * 100
        color = BEST_COLOR if pct > 0 else WARN_COLOR
        sign  = '−' if pct >= 0 else '+'
        top   = max(fp32_vals[i], amp_vals[i])
        ax.text(x[i], top + pct_gap, f'{sign}{abs(pct):.0f}%',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=color)

    # callouts
    best_i  = int(np.argmax(amp_vals))
    worst_i = int(np.argmin(amp_vals))
    ax.annotate(
        'Best: M1_AMP\n+99% vs M0_FP32',
        xy=(x[best_i] + w/2, amp_vals[best_i]),
        xytext=(x[best_i] + 0.85, ymax * 1.2),
        fontsize=8, color=BEST_COLOR, fontweight='bold', ha='center',
        arrowprops=dict(arrowstyle='->', color=BEST_COLOR, lw=1.1,
                        connectionstyle='arc3,rad=-0.2'))
    ax.annotate(
        'Worst: M4_AMP\ncompile spike\ninflates window',
        xy=(x[worst_i] + w/2, amp_vals[worst_i]),
        xytext=(x[worst_i] - 0.8, ymax * 1.0),
        fontsize=8, color=WARN_COLOR, fontweight='bold', ha='center',
        arrowprops=dict(arrowstyle='->', color=WARN_COLOR, lw=1.1,
                        connectionstyle='arc3,rad=0.25'))

    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, fontsize=9.5)
    ax.set_ylabel('Samples / Joule', fontsize=10)
    ax.set_title('Energy Efficiency (samples/J)', fontsize=11, fontweight='bold', pad=8)
    ax.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    save(fig, out, 'a10g_samples_per_joule.png')


def plot_a10g_power(out):
    n = len(VARIANTS)
    x = np.arange(n)
    w = 0.33
    fp32_vals = np.array(A10G['FP32']['power_W'], dtype=float)
    amp_vals  = np.array(A10G['AMP']['power_W'],  dtype=float)

    # Cap axis at 220W so M0-M3 are readable; M4_FP32=684W is an outlier
    CAP = 220
    ymax = CAP
    plot_fp32 = np.minimum(fp32_vals, CAP)
    plot_amp  = np.minimum(amp_vals,  CAP)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        'A10G (Modal) — Mean GPU Power (W)\npynvml · batch=1 · 50 frames · KITTI val',
        fontsize=11, fontweight='bold'
    )
    ax.set_ylim(0, CAP * 1.55)

    bars_fp32 = ax.bar(x - w/2, plot_fp32, w, color=FP32_COLOR, alpha=ALPHA,
                       label='FP32', edgecolor='white', linewidth=0.5)
    bars_amp  = ax.bar(x + w/2, plot_amp,  w, color=AMP_COLOR,  alpha=ALPHA,
                       label='FP16 AMP', edgecolor='white', linewidth=0.5)

    label_gap = CAP * 0.02
    for i, (bar, val, pval) in enumerate(zip(bars_fp32, fp32_vals, plot_fp32)):
        if val > CAP:
            # hatching to show truncation
            ax.bar(x[i] - w/2, CAP, w, color=FP32_COLOR, alpha=0.4,
                   edgecolor=FP32_COLOR, linewidth=1.2, hatch='//', fill=False)
            ax.text(bar.get_x() + bar.get_width()/2, CAP + label_gap,
                    f'{val:.0f}W ↑\n(compile spike)', ha='center', va='bottom',
                    fontsize=7.5, color=FP32_COLOR, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, pval + label_gap,
                    f'{val:.0f}', ha='center', va='bottom',
                    fontsize=8, color=FP32_COLOR, fontweight='bold')
    for bar, val, pval in zip(bars_amp, amp_vals, plot_amp):
        ax.text(bar.get_x() + bar.get_width()/2, pval + label_gap,
                f'{val:.0f}', ha='center', va='bottom',
                fontsize=8, color=AMP_COLOR, fontweight='bold')

    pct_gap = CAP * 0.14
    for i in range(n):
        pct   = (fp32_vals[i] - amp_vals[i]) / fp32_vals[i] * 100
        color = BEST_COLOR if pct > 0 else WARN_COLOR
        sign  = '−' if pct >= 0 else '+'
        top   = min(max(plot_fp32[i], plot_amp[i]) + label_gap * 4, CAP * 1.05)
        if fp32_vals[i] > CAP:
            top = CAP * 1.28
        ax.text(x[i], top + pct_gap * 0.3, f'{sign}{abs(pct):.0f}%',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=color)

    ax.text(0.02, 0.04,
            'NMS ≈70% GPU time,\nstays FP32\n→ caps AMP savings\nfor M0–M3',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8.5, color='#555', style='italic',
            bbox=dict(boxstyle='round,pad=0.35', fc='#fff8e1', ec='#f9a825', alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, fontsize=9.5)
    ax.set_ylabel('Mean GPU Power (W)', fontsize=10)
    ax.set_title('Mean GPU Power (W)  [y-axis capped at 220W; M4 FP32 = 684W spike]',
                 fontsize=10, fontweight='bold', pad=8)
    ax.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    save(fig, out, 'a10g_mean_power.png')


# ═════════════════════════════════════════════════════════════════════════════
# Local RTX 3080 Ti panels
# ═════════════════════════════════════════════════════════════════════════════

def plot_local_gpu_stages(out):
    fp32_gpu = stage_energies(FP32_STAGES_MS, FP32_GPU_J)
    amp_gpu  = stage_energies(AMP_STAGES_MS,  AMP_GPU_J)
    stages   = list(GPU_COLORS.keys())

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        'RTX 3080 Ti (Local) — GPU Pipeline Stage Energy\npynvml × NVTX stage fractions  |  50 frames',
        fontsize=11, fontweight='bold'
    )

    fp32_top, fp32_info = stacked_bar(ax, 0.0, 0.5,
                                      [(s, fp32_gpu[s]) for s in stages],
                                      GPU_COLORS, FP32_EDGE)
    amp_top,  amp_info  = stacked_bar(ax, 1.0, 0.5,
                                      [(s, amp_gpu[s]) for s in stages],
                                      GPU_COLORS, AMP_EDGE)

    # label only NMS inside (large enough)
    for (name, val, mid_y), xi in [(fp32_info[-1], 0.0), (amp_info[-1], 1.0)]:
        ax.text(xi, mid_y, f'{val:.1f}J', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    gpu_s = (fp32_top - amp_top) / fp32_top * 100
    for xi, top, ec, lbl in [
        (0.0, fp32_top, FP32_EDGE, f'Total: {fp32_top:.1f}J'),
        (1.0, amp_top,  AMP_EDGE,  f'Total: {amp_top:.1f}J'),
    ]:
        ax.text(xi, top + 1, lbl, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=ec)

    ax.text(0.5, max(fp32_top, amp_top) + 6,
            f'−{gpu_s:.0f}%', ha='center', va='bottom',
            fontsize=13, fontweight='bold', color=BEST_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc='#f1f8e9', ec='#aaa'))

    nms_bot_amp = sum(amp_gpu[s] for s in stages if s != 'NMS')
    ax.annotate('NMS stays FP32\n(~70% of GPU time)',
                xy=(1.25, nms_bot_amp + amp_gpu['NMS'] * 0.5),
                xytext=(1.55, nms_bot_amp + amp_gpu['NMS'] * 0.3),
                fontsize=8, color=WARN_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=WARN_COLOR, lw=1.1))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['FP32', 'FP16 AMP'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy — 50-frame run (J)', fontsize=10)
    ax.set_title('GPU Pipeline Stages', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(-0.5, 2.2)
    ax.set_ylim(0, fp32_top * 1.75)
    ax.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax.set_axisbelow(True)
    gpu_leg = [mpatches.Patch(facecolor=GPU_COLORS[s], alpha=ALPHA, label=s) for s in stages]
    ax.legend(handles=gpu_leg, fontsize=8, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save(fig, out, 'local_gpu_stages.png')


def plot_local_cpu_rapl(out):
    domains = list(CPU_COLORS.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        'RTX 3080 Ti (Local) — CPU RAPL Domains + psys\nIntel RAPL  |  50 frames  |  psys non-additive with pkg',
        fontsize=11, fontweight='bold'
    )
    w2 = 0.42

    fp32_pkg, _ = stacked_bar(ax, 0.0, w2,
                               [(d, CPU['FP32'][d]) for d in domains],
                               CPU_COLORS, FP32_EDGE)
    amp_pkg,  _ = stacked_bar(ax, 1.0, w2,
                               [(d, CPU['AMP'][d])  for d in domains],
                               CPU_COLORS, AMP_EDGE)

    for xi, prec, ec in [(2.4, 'FP32', FP32_EDGE), (3.3, 'AMP', AMP_EDGE)]:
        ax.bar(xi, PSYS[prec], w2, color=PSYS_COLOR, alpha=ALPHA,
               edgecolor='white', linewidth=0.5)
        ax.bar(xi, PSYS[prec], w2, fill=False, edgecolor=ec, linewidth=EDGE_LW)

    pkg_s  = (fp32_pkg - amp_pkg) / fp32_pkg * 100
    psys_s = (PSYS['FP32'] - PSYS['AMP']) / PSYS['FP32'] * 100

    gap = 3
    for xi, top, ec, lbl in [
        (0.0, fp32_pkg,     FP32_EDGE, f'{fp32_pkg:.1f}J'),
        (1.0, amp_pkg,      AMP_EDGE,  f'{amp_pkg:.1f}J'),
        (2.4, PSYS['FP32'], FP32_EDGE, f'{PSYS["FP32"]:.1f}J'),
        (3.3, PSYS['AMP'],  AMP_EDGE,  f'{PSYS["AMP"]:.1f}J'),
    ]:
        ax.text(xi, top + gap, lbl, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=ec)

    for xi, top, lbl in [
        (0.5,  max(fp32_pkg, amp_pkg),        f'−{pkg_s:.0f}%'),
        (2.85, max(PSYS['FP32'], PSYS['AMP']), f'−{psys_s:.0f}%'),
    ]:
        ax.text(xi, top + 14, lbl, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=BEST_COLOR,
                bbox=dict(boxstyle='round,pad=0.25', fc='#f1f8e9', ec='#aaa'))

    ax.axvline(x=1.85, color='#bbb', linewidth=1.5, linestyle='--')
    ax.text(1.85, YMAX_LOCAL * 0.55, 'non-\nadditive\n→',
            ha='center', va='center', fontsize=8, color='#888', style='italic')

    ax.text(2.85, PSYS['AMP'] * 0.40,
            'psys = CPU pkg + DRAM\n+ SSD + display + fans\n(≈3.8× CPU pkg)',
            ha='center', va='center', fontsize=8.5, color='#555', style='italic',
            bbox=dict(boxstyle='round,pad=0.35', fc='#fff3e0', ec='#fb8c00', alpha=0.95))

    ax.set_xticks([0.0, 1.0, 2.4, 3.3])
    ax.set_xticklabels(['FP32\n(pkg)', 'AMP\n(pkg)', 'FP32\n(psys)', 'AMP\n(psys)'],
                       fontsize=10, fontweight='bold')
    ax.set_title('CPU RAPL Domains  +  psys', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(-0.55, 4.0)
    ax.set_ylim(0, YMAX_LOCAL)
    ax.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax.set_axisbelow(True)
    cpu_leg = [mpatches.Patch(facecolor=CPU_COLORS[d], alpha=ALPHA, label=d) for d in domains]
    cpu_leg.append(mpatches.Patch(facecolor=PSYS_COLOR, alpha=ALPHA, label='psys (full platform)'))
    ax.legend(handles=cpu_leg, fontsize=8.5, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save(fig, out, 'local_cpu_rapl.png')


def plot_local_total_system(out):
    summary_segs = [
        ('GPU active',  FP32_GPU_J,    AMP_GPU_J,    '#1565C0'),
        ('GPU idle',    IDLE_J['FP32'],IDLE_J['AMP'], IDLE_COLOR),
        ('CPU package', PKG['FP32'],   PKG['AMP'],    '#C62828'),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        'RTX 3080 Ti (Local) — Total System Energy\nGPU active + GPU idle + CPU package  |  50 frames',
        fontsize=11, fontweight='bold'
    )

    x_fp32, x_amp = 0.0, 1.0
    w3 = 0.48
    bottoms = {'FP32': 0.0, 'AMP': 0.0}
    for seg, fp32_v, amp_v, color in summary_segs:
        for xi, val, prec in [(x_fp32, fp32_v, 'FP32'), (x_amp, amp_v, 'AMP')]:
            b = bottoms[prec]
            ax.bar(xi, val, w3, bottom=b, color=color, alpha=ALPHA,
                   edgecolor='white', linewidth=0.5)
            if val >= 5:
                ax.text(xi, b + val / 2, f'{val:.0f}J',
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold')
            bottoms[prec] += val

    for xi, prec, ec in [(x_fp32, 'FP32', FP32_EDGE), (x_amp, 'AMP', AMP_EDGE)]:
        ax.bar(xi, bottoms[prec], w3, fill=False, edgecolor=ec, linewidth=EDGE_LW)

    for xi, prec, ec in [(x_fp32, 'FP32', FP32_EDGE), (x_amp, 'AMP', AMP_EDGE)]:
        ax.text(xi, bottoms[prec] + 3, f'Total\n{bottoms[prec]:.1f}J',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=ec)

    tot_s = (bottoms['FP32'] - bottoms['AMP']) / bottoms['FP32'] * 100
    ax.text(0.5, max(bottoms['FP32'], bottoms['AMP']) + 14,
            f'−{tot_s:.0f}%', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color=BEST_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc='#f1f8e9', ec='#aaa'))

    for prec, ls, dy in [('FP32', '--', +2), ('AMP', ':', -7)]:
        ec = FP32_EDGE if prec == 'FP32' else AMP_EDGE
        yp = PSYS[prec]
        ax.plot([-0.35, 1.35], [yp, yp], color=ec, linewidth=1.3, linestyle=ls, alpha=0.6)
        ax.text(1.37, yp + dy, f'psys {prec}: {yp:.0f}J',
                ha='left', va='center', fontsize=7.5, color=ec)

    ax.text(0.5, -YMAX_LOCAL * 0.09,
            f'GPU idle: {GPU_IDLE_W}W  |  FP32: {IDLE_J["FP32"]:.1f}J  AMP: {IDLE_J["AMP"]:.1f}J  (50 frames @ 10 Hz)',
            ha='center', va='top', fontsize=8, color='#546e7a',
            bbox=dict(boxstyle='round,pad=0.3', fc='#eceff1', ec='#90a4ae', alpha=0.9))

    ax.set_xticks([x_fp32, x_amp])
    ax.set_xticklabels(['FP32', 'FP16 AMP'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy — 50-frame run (J)', fontsize=10)
    ax.set_title('Total System: GPU active + GPU idle + CPU pkg', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(-0.55, 1.85)
    ax.set_ylim(-YMAX_LOCAL * 0.12, YMAX_LOCAL)
    ax.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax.set_axisbelow(True)
    sum_leg = [mpatches.Patch(facecolor=c, alpha=ALPHA, label=s) for s, _, _, c in summary_segs]
    ax.legend(handles=sum_leg, fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout(rect=[0, 0.04, 1, 1.0])
    save(fig, out, 'local_total_system.png')


def plot_local_samples_j(out):
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

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        'RTX 3080 Ti (Local) — Energy Efficiency at Three Accounting Levels\nHigher = better  |  50 frames',
        fontsize=11, fontweight='bold'
    )
    ax.set_ylim(0, ymax4 * 1.65)

    b1 = ax.bar(xm - wm/2, fp32_sj, wm, color=FP32_COLOR, alpha=ALPHA,
                edgecolor='white', label='FP32')
    b2 = ax.bar(xm + wm/2, amp_sj,  wm, color=AMP_COLOR,  alpha=ALPHA,
                edgecolor='white', label='FP16 AMP')

    gap4 = ymax4 * 0.015
    for bars, vals, ec in [(b1, fp32_sj, FP32_COLOR), (b2, amp_sj, AMP_COLOR)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + gap4,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=8.5, color=ec, fontweight='bold')

    for i in range(3):
        pct = (amp_sj[i] - fp32_sj[i]) / fp32_sj[i] * 100
        top = max(fp32_sj[i], amp_sj[i])
        ax.text(xm[i], top + ymax4 * 0.12,
                f'+{pct:.0f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=BEST_COLOR)

    ax.text(0.5, 0.835,
            'GPU-only overstates efficiency by ~3×\non edge hardware (CPU pkg = 1.9× GPU energy at batch=1)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=8.5, color='#555', style='italic',
            bbox=dict(boxstyle='round,pad=0.35', fc='#fff3e0', ec='#fb8c00', alpha=0.95))

    ax.set_xticks(xm)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('Samples / Joule', fontsize=10)
    ax.set_title('Samples/J at Three Accounting Levels', fontsize=11, fontweight='bold', pad=8)
    ax.yaxis.grid(True, alpha=0.22, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    save(fig, out, 'local_samples_per_joule.png')


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='../../profile_outputs/amp_benchmark/report_figures')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('=== A10G panels ===')
    plot_a10g_energy(out)
    plot_a10g_samples_j(out)
    plot_a10g_power(out)

    print('=== Local RTX 3080 Ti panels ===')
    plot_local_gpu_stages(out)
    plot_local_cpu_rapl(out)
    plot_local_total_system(out)
    plot_local_samples_j(out)

    print(f'\nAll 7 figures saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
