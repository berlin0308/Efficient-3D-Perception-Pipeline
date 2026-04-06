"""
End-to-End PointPillar Pipeline Comparison: Baseline (FP32) vs AMP (FP16)

Data sources:
  Baseline E2E stage latencies: nsight_stats_20260302_103231.txt  (NVTX avg, ns)
  AMP improvement:              energy_monitor runs (FP32 13.75ms, FP16 9.50ms)
  AMP stage-level speedup:      AMP only affects GPU stages (forward/post_proc);
                                 CPU stages (read, preproc, voxelize, data_to_gpu)
                                 are unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Baseline E2E stage latencies (from nsight_stats, avg_ns → ms) ─────────────
# Excludes wait_10hz (idle time to simulate 10Hz rate, not pipeline cost)
baseline_stages = {
    "read_points":       2.478,   # avg ms
    "pre_processing":    3.591,
    "voxelize_compiled": 3.398,
    "data_to_gpu":       0.657,
    "forward":          69.786,   # includes JIT outlier iterations; med=10.32ms
    "post_processing":   3.958,
}

# forward avg is skewed by first-iteration JIT (3982ms outlier in 90 instances).
# Use median for a fair comparison: 10.318 ms
baseline_stages["forward"] = 10.318

# ── AMP (FP16) stage latencies ────────────────────────────────────────────────
# CPU-bound stages unchanged; GPU stages scale by measured speedup ratio.
# energy_monitor: FP32 mean=13.75ms, FP16 mean=9.50ms  => GPU speedup = 1.447×
# forward and post_processing are GPU-bound; others are CPU-bound.
gpu_speedup = 13.75 / 9.50  # = 1.447

amp_stages = {
    "read_points":       baseline_stages["read_points"],       # CPU, unchanged
    "pre_processing":    baseline_stages["pre_processing"],    # CPU, unchanged
    "voxelize_compiled": baseline_stages["voxelize_compiled"], # CPU, unchanged
    "data_to_gpu":       baseline_stages["data_to_gpu"],       # PCIe, unchanged
    "forward":           baseline_stages["forward"] / gpu_speedup,
    "post_processing":   baseline_stages["post_processing"] / gpu_speedup,
}

stage_labels = {
    "read_points":       "Read\nPoints",
    "pre_processing":    "Pre-\nProcess",
    "voxelize_compiled": "Voxelize\n(compiled)",
    "data_to_gpu":       "Data\nto GPU",
    "forward":           "Forward\n(GPU)",
    "post_processing":   "Post-\nProcess",
}

stages = list(baseline_stages.keys())
base_vals = np.array([baseline_stages[s] for s in stages])
amp_vals  = np.array([amp_stages[s]      for s in stages])

# Stage colors: CPU stages warm, GPU stages cool
STAGE_COLORS = {
    "read_points":       "#E07B39",
    "pre_processing":    "#E0A239",
    "voxelize_compiled": "#D4C244",
    "data_to_gpu":       "#8B8B8B",
    "forward":           "#4C72B0",
    "post_processing":   "#5F9E6E",
}
colors = [STAGE_COLORS[s] for s in stages]

base_total = base_vals.sum()
amp_total  = amp_vals.sum()
speedup    = base_total / amp_total

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs  = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)

ax1 = fig.add_subplot(gs[0, :])   # grouped bar: stage breakdown
ax2 = fig.add_subplot(gs[1, 0])   # stacked total bar
ax3 = fig.add_subplot(gs[1, 1])   # speedup per stage
ax4 = fig.add_subplot(gs[1, 2])   # throughput & power metrics

fig.suptitle("PointPillar E2E Latency: Baseline (FP32) vs AMP (FP16)\nRTX 3080 Ti Laptop GPU",
             fontsize=13, fontweight="bold", y=0.98)

# ── Plot 1: Grouped bar per stage ─────────────────────────────────────────────
x  = np.arange(len(stages))
w  = 0.35

bars1 = ax1.bar(x - w/2, base_vals, w, color=colors, alpha=0.95,
                edgecolor="white", linewidth=0.8, label="Baseline (FP32)")
bars2 = ax1.bar(x + w/2, amp_vals,  w, color=colors, alpha=0.55,
                edgecolor="white", linewidth=0.8, hatch="//", label="AMP (FP16)")

ax1.set_xticks(x)
ax1.set_xticklabels([stage_labels[s] for s in stages], fontsize=10)
ax1.set_ylabel("Latency (ms)")
ax1.set_title("Pipeline Stage Latency Breakdown", fontsize=11, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)
ax1.spines[["top", "right"]].set_visible(False)

# value labels
for bar, val in zip(bars1, base_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1,
             f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
for bar, val in zip(bars2, amp_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1,
             f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333")

# Legend + total annotation
ax1.legend(fontsize=10, loc="upper right")
ax1.text(0.01, 0.95,
         f"E2E total — Baseline: {base_total:.2f} ms  |  AMP: {amp_total:.2f} ms  "
         f"|  Speedup: {speedup:.2f}×",
         transform=ax1.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.85))

# ── Plot 2: Stacked total bar ─────────────────────────────────────────────────
variants = ["Baseline\n(FP32)", "AMP\n(FP16)"]
data_sets = [base_vals, amp_vals]

for xi, (label, vals) in enumerate(zip(variants, data_sets)):
    bottom = 0
    for s, v, c in zip(stages, vals, colors):
        ax2.bar(xi, v, 0.5, bottom=bottom, color=c,
                alpha=0.9 if xi == 0 else 0.6,
                edgecolor="white", linewidth=0.5)
        if v > 0.3:
            ax2.text(xi, bottom + v/2, f"{v:.1f}",
                     ha="center", va="center", fontsize=7.5,
                     color="white", fontweight="bold")
        bottom += v
    ax2.text(xi, bottom + 0.3, f"{vals.sum():.1f} ms",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2.set_xticks([0, 1])
ax2.set_xticklabels(variants, fontsize=10)
ax2.set_ylabel("Total E2E Latency (ms)")
ax2.set_title("Total E2E Latency", fontsize=11, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)

# Stage legend
handles = [mpatches.Patch(color=STAGE_COLORS[s], label=stage_labels[s].replace("\n", " "))
           for s in stages]
ax2.legend(handles=handles, fontsize=7, loc="upper right",
           title="Stage", title_fontsize=8)

# ── Plot 3: Per-stage speedup ─────────────────────────────────────────────────
stage_speedups = base_vals / amp_vals
bar_colors_sp  = ["#C44E52" if sp < 1.05 else "#55A868" for sp in stage_speedups]

bars3 = ax3.bar(x, stage_speedups, 0.55, color=bar_colors_sp,
                edgecolor="white", linewidth=0.8)
ax3.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5, label="No change")
ax3.set_xticks(x)
ax3.set_xticklabels([stage_labels[s] for s in stages], fontsize=8.5)
ax3.set_ylabel("Speedup (×)")
ax3.set_title("Per-Stage Speedup (FP32→FP16)", fontsize=11, fontweight="bold")
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)

for bar, sp in zip(bars3, stage_speedups):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{sp:.2f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

green_patch = mpatches.Patch(color="#55A868", label="GPU-accelerated")
red_patch   = mpatches.Patch(color="#C44E52", label="CPU-bound (unchanged)")
ax3.legend(handles=[green_patch, red_patch], fontsize=8, loc="upper left")

# ── Plot 4: System metrics comparison ────────────────────────────────────────
metrics = ["Latency\n(ms)", "Throughput\n(fps)", "Power\n(W)", "Perf/Watt\n(fps/W)"]
base_m  = [13.75,  70.03, 38.0, 1.843]
amp_m   = [ 9.50,  89.73, 31.3, 2.868]

# Normalize to baseline for visual comparison
norm_base = [1.0] * 4
norm_amp  = [amp_m[i] / base_m[i] for i in range(4)]
# Latency: lower is better → invert
norm_amp[0] = base_m[0] / amp_m[0]

xm = np.arange(len(metrics))
wm = 0.35
bm1 = ax4.bar(xm - wm/2, [1.0]*4, wm, color="#4C72B0", alpha=0.8,
              edgecolor="white", label="Baseline (FP32)")
bm2 = ax4.bar(xm + wm/2, norm_amp,  wm, color="#55A868", alpha=0.8,
              edgecolor="white", label="AMP (FP16)")
ax4.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.4)

ax4.set_xticks(xm)
ax4.set_xticklabels(metrics, fontsize=9)
ax4.set_ylabel("Relative to Baseline")
ax4.set_title("System Metrics (normalized)", fontsize=11, fontweight="bold")
ax4.grid(axis="y", alpha=0.3)
ax4.spines[["top", "right"]].set_visible(False)
ax4.legend(fontsize=9)

for bar, bval, aval, nm in zip(bm2, base_m, amp_m, norm_amp):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{nm:.2f}×", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
             color="#2a7a2a")

# raw values below x-axis
for i, (bv, av, metric) in enumerate(zip(base_m, amp_m, metrics)):
    unit = ["ms","fps","W","fps/W"][i]
    ax4.text(i, -0.12, f"B:{bv} / A:{av:.1f} {unit}",
             ha="center", va="top", fontsize=7.5, color="#555",
             transform=ax4.get_xaxis_transform())

# ── save ──────────────────────────────────────────────────────────────────────
out = "profile_outputs/pointpillar_e2e_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
