"""
PointPillar Pipeline Analysis Plots
Generated from nsys profile: profile_outputs/nsys_baseline/report.sqlite
RTX 3080 Ti Laptop GPU
"""
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

DB = "profile_outputs/nsys_baseline/report.sqlite"

# ── helpers ──────────────────────────────────────────────────────────────────
def query(sql):
    con = sqlite3.connect(DB)
    cur = con.execute(sql)
    rows = cur.fetchall()
    con.close()
    return rows

# ── data extraction ───────────────────────────────────────────────────────────
stage_order = ["PillarVFE", "PointPillarScatter", "BaseBEVBackbone",
               "AnchorHeadSingle", "post_processing"]

# Per-iteration timings (skip first row – JIT warm-up outlier)
raw = query("""
    SELECT text, (end-start)/1e6
    FROM NVTX_EVENTS
    WHERE text IN ('PillarVFE','PointPillarScatter','BaseBEVBackbone',
                   'AnchorHeadSingle','post_processing')
      AND end IS NOT NULL AND end > start
    ORDER BY start
""")

from collections import defaultdict
per_iter = defaultdict(list)
for name, ms in raw:
    per_iter[name].append(ms)

# Drop first sample (warm-up) from each stage
for k in per_iter:
    per_iter[k] = per_iter[k][1:]

stage_means = {s: np.mean(per_iter[s]) for s in stage_order}
stage_stds  = {s: np.std(per_iter[s])  for s in stage_order}

# Kernel type totals (categorised)
kernel_rows = query("""
    SELECT s.value, SUM((k.end-k.start)/1e6)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    GROUP BY s.value
    ORDER BY SUM((k.end-k.start)/1e6) DESC
""")

def classify(name):
    n = name.lower()
    if "conv" in n or "fprop" in n or "winograd" in n or "gemm" in n or "xmma" in n:
        return "Conv/GEMM"
    if "bn_fw" in n or "batch_norm" in n:
        return "BatchNorm"
    if "nms" in n:
        return "NMS"
    if "elementwise" in n or "clamp" in n or "add" in n or "mul" in n or "fill" in n:
        return "Elementwise"
    if "nchw" in n or "nhwc" in n or "copy" in n or "cat" in n or "reduce" in n:
        return "Layout/Copy"
    if "dgrad" in n:
        return "Conv/GEMM"
    return "Other"

cat_totals = defaultdict(float)
for name, ms in kernel_rows:
    cat_totals[classify(name)] += ms

# Memory transfers  (copyKind: 1=HtoD, 2=DtoH, 8=DtoD)
memcpy_rows = query("""
    SELECT copyKind, SUM((end-start)/1e6), SUM(bytes)/1e6
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    GROUP BY copyKind
""")
kind_label = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
mem_data = {kind_label.get(k, str(k)): (ms, mb) for k, ms, mb in memcpy_rows}

# ── colours ───────────────────────────────────────────────────────────────────
STAGE_COLORS = {
    "PillarVFE":        "#4C72B0",
    "PointPillarScatter":"#DD8452",
    "BaseBEVBackbone":  "#55A868",
    "AnchorHeadSingle": "#C44E52",
    "post_processing":  "#8172B3",
}
CAT_COLORS = ["#4C72B0","#55A868","#DD8452","#C44E52","#8172B3","#937860"]

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs  = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.42)

ax1 = fig.add_subplot(gs[0, :2])   # stage latency bar (wide)
ax2 = fig.add_subplot(gs[0, 2])    # stage latency pie
ax3 = fig.add_subplot(gs[1, 0])    # kernel category breakdown
ax4 = fig.add_subplot(gs[1, 1])    # per-iter timeline (box)
ax5 = fig.add_subplot(gs[1, 2])    # memory transfer

fig.suptitle("PointPillar Pipeline Analysis  ·  RTX 3080 Ti Laptop GPU",
             fontsize=14, fontweight="bold", y=0.98)

# ── Plot 1: Stage mean latency bar ───────────────────────────────────────────
labels  = [s.replace("post_processing","post_proc") for s in stage_order]
means   = [stage_means[s] for s in stage_order]
stds    = [stage_stds[s]  for s in stage_order]
colors  = [STAGE_COLORS[s] for s in stage_order]

bars = ax1.bar(labels, means, yerr=stds, color=colors,
               capsize=5, edgecolor="white", linewidth=0.8, zorder=3)
ax1.set_ylabel("Latency (ms)")
ax1.set_title("Pipeline Stage Mean Latency (±1σ, warm iterations)")
ax1.set_ylim(0, max(means) * 1.3)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.tick_params(axis="x", labelsize=9)

for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2, mean + std + 0.1,
             f"{mean:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

total_ms = sum(means)
ax1.text(0.98, 0.95, f"Total: {total_ms:.2f} ms\n({1000/total_ms:.1f} FPS est.)",
         transform=ax1.transAxes, ha="right", va="top",
         fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

# ── Plot 2: Stage share pie ───────────────────────────────────────────────────
pie_labels = [f"{s.replace('post_processing','post_proc')}\n{means[i]:.1f}ms"
              for i, s in enumerate(stage_order)]
ax2.pie(means, labels=pie_labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 7.5}, pctdistance=0.75)
ax2.set_title("Stage Time Share")

# ── Plot 3: Kernel category pie ───────────────────────────────────────────────
cats   = list(cat_totals.keys())
c_vals = [cat_totals[c] for c in cats]
c_pct  = np.array(c_vals) / sum(c_vals) * 100
c_colors = CAT_COLORS[:len(cats)]

wedges, texts, autotexts = ax3.pie(
    c_vals, labels=cats, colors=c_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 8}, pctdistance=0.78)
ax3.set_title("GPU Kernel Time by Category")

# ── Plot 4: Per-iteration latency box ────────────────────────────────────────
box_data  = [per_iter[s] for s in stage_order]
bp = ax4.boxplot(box_data, patch_artist=True,
                 medianprops=dict(color="black", linewidth=1.5),
                 flierprops=dict(marker=".", markersize=3, alpha=0.5))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax4.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
ax4.set_ylabel("Latency (ms)")
ax4.set_title("Per-Iteration Latency Distribution")
ax4.grid(axis="y", alpha=0.3)
ax4.set_yscale("log")
ax4.set_ylabel("Latency (ms, log scale)")

# ── Plot 5: Memory transfer – two subplots stacked (volume + bandwidth) ───────
mem_kinds   = list(mem_data.keys())
mem_ms_vals = [mem_data[k][0] for k in mem_kinds]
mem_mb_vals = [mem_data[k][1] for k in mem_kinds]
mem_bw      = [mb / ms * 1000 if ms > 0 else 0
               for mb, ms in zip(mem_mb_vals, mem_ms_vals)]  # GB/s

mem_colors = ["#4C72B0", "#DD8452", "#55A868"]
x = np.arange(len(mem_kinds))

# Volume bars (log scale to handle HtoD 99 MB vs DtoD 1598 MB)
ax5.bar(x, mem_mb_vals, color=mem_colors[:len(mem_kinds)],
        alpha=0.85, edgecolor="white")
ax5.set_yscale("log")
ax5.set_xticks(x)
ax5.set_xticklabels(mem_kinds, fontsize=10)
ax5.set_ylabel("Volume (MB, log)")
ax5.set_title("Memory Transfers\n(HtoD / DtoH / DtoD)")
ax5.grid(axis="y", alpha=0.3, which="both")
for xi, val in zip(x, mem_mb_vals):
    ax5.text(xi, val * 1.15, f"{val:.0f} MB", ha="center", va="bottom", fontsize=8)

# Annotate BW inside bars
for xi, bw in zip(x, mem_bw):
    if bw > 0:
        ax5.text(xi, mem_mb_vals[xi] * 0.4,
                 f"{bw:.1f} GB/s", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")

# ── save ──────────────────────────────────────────────────────────────────────
out = "profile_outputs/pointpillar_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
