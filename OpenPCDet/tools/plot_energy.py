"""
PointPillar Energy Breakdown Plot
Mirrors Figure 8 structure: Total / DRAM / MAC energy broken down by pipeline
stage, across FP32 and FP16 precisions.

Method:
  - Total energy per frame: from pynvml (energy_monitor.py runs)
  - Stage time fractions: from nsys NVTX timings (report.sqlite)
  - DRAM vs MAC split: from nsys memcpy bytes + RTX 3080 Ti hardware constants
      DRAM energy:  bytes_transferred * GDDR6X_energy_per_bit
      MAC energy:   flops_estimated   * TF32_energy_per_flop
  - Stage operand breakdown (Inputs/Outputs/Weights/MAC):
      Inputs  ~ activations read  (proportional to HtoD + DtoD read share per stage)
      Outputs ~ activations write (DtoD write share)
      Weights ~ parameter bytes   (known per-layer from model config)
      MAC     ~ residual compute energy
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Hardware energy constants (RTX 3080 Ti / GDDR6X) ─────────────────────────
# GDDR6X: ~14 pJ/bit => 14/8 pJ/byte = 1.75 nJ/byte = 1.75e-9 J/byte
GDDR6X_J_PER_BYTE = 1.75e-9
# TF32 Tensor Core: ~0.4 pJ/MAC => 4e-13 J/MAC
TF32_J_PER_MAC = 4e-13
# FP16 Tensor Core: ~0.2 pJ/MAC => 2e-13 J/MAC
FP16_J_PER_MAC = 2e-13

# ── Measured total energy per frame (from energy_monitor.py) ─────────────────
# FP32: 27.13 J over 50 frames
# FP16: 17.08 J over 50 frames
E_fp32_J = 27.13 / 50  # J/frame
E_fp16_J = 17.08 / 50  # J/frame

# ── nsys data ─────────────────────────────────────────────────────────────────
DB = "profile_outputs/nsys_baseline/report.sqlite"

def query(sql):
    con = sqlite3.connect(DB)
    rows = con.execute(sql).fetchall()
    con.close()
    return rows

stage_order = ["PillarVFE", "PointPillarScatter", "BaseBEVBackbone",
               "AnchorHeadSingle", "post_processing"]

# Average stage durations (warm iters, skip first)
raw = query("""
    SELECT text, (end-start)/1e6
    FROM NVTX_EVENTS
    WHERE text IN ('PillarVFE','PointPillarScatter','BaseBEVBackbone',
                   'AnchorHeadSingle','post_processing')
      AND end IS NOT NULL AND end > start
    ORDER BY start
""")
per_iter = defaultdict(list)
for name, ms in raw:
    per_iter[name].append(ms)
for k in per_iter:
    per_iter[k] = per_iter[k][1:]
stage_ms   = {s: np.mean(per_iter[s]) for s in stage_order}
total_ms   = sum(stage_ms.values())
stage_frac = {s: stage_ms[s] / total_ms for s in stage_order}

# Total DRAM bytes from memcpy (HtoD=1, DtoH=2, DtoD=8) – per 25 frames in profile
memcpy_rows = query("SELECT copyKind, SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind")
bytes_by_kind = {k: b for k, b in memcpy_rows}
# HtoD (kind=1): input point cloud to GPU
# DtoD (kind=8): internal tensor copies (activations + weights reads)
# DtoH (kind=2): tiny (results back to CPU)
n_profile_frames = 25  # nsys captured 25 warmup + measurement iters (first dropped)
htod_bytes_per_frame = bytes_by_kind.get(1, 0) / n_profile_frames
dtod_bytes_per_frame = bytes_by_kind.get(8, 0) / n_profile_frames
dtoh_bytes_per_frame = bytes_by_kind.get(2, 0) / n_profile_frames
total_dram_bytes = htod_bytes_per_frame + dtod_bytes_per_frame + dtoh_bytes_per_frame

# ── PointPillar parameter sizes (approximate, FP32 bytes) ────────────────────
# PillarVFE:          ~0.3M params  (linear layers on pillar features)
# PointPillarScatter: 0 params      (scatter op only)
# BaseBEVBackbone:    ~4.9M params  (2D conv backbone)
# AnchorHeadSingle:   ~0.4M params  (cls + reg heads)
# post_processing:    0 params
param_bytes_fp32 = {
    "PillarVFE":         0.3e6 * 4,
    "PointPillarScatter":0,
    "BaseBEVBackbone":   4.9e6 * 4,
    "AnchorHeadSingle":  0.4e6 * 4,
    "post_processing":   0,
}
param_bytes_fp16 = {k: v / 2 for k, v in param_bytes_fp32.items()}

# ── Approximate FLOPs per stage (from model config / standard estimates) ──────
# PillarVFE:           ~1.5 GMACs  (linear on ~12k pillars × 64 features × 4 layers)
# PointPillarScatter:  ~0.05 GMACs (scatter = index-add, minimal MAC)
# BaseBEVBackbone:     ~18 GMACs   (VGG-style 2D convs on 496×432 BEV)
# AnchorHeadSingle:    ~1.2 GMACs  (3×3 conv heads)
# post_processing:     ~0.05 GMACs (NMS – mostly comparisons)
stage_gmacs = {
    "PillarVFE":          1.5e9,
    "PointPillarScatter": 0.05e9,
    "BaseBEVBackbone":   18.0e9,
    "AnchorHeadSingle":   1.2e9,
    "post_processing":    0.05e9,
}

# ── Energy decomposition function ─────────────────────────────────────────────
def decompose_energy(total_E_J, precision):
    """
    Returns dict: stage -> {inputs, outputs, weights, mac} in mJ
    Strategy:
      1. Total MAC energy  = FLOPs × J/MAC  (hardware constant)
      2. Total DRAM energy = total_dram_bytes × J/byte
      3. Scale both to sum to measured total_E_J (overhead accounts for rest)
      4. Distribute DRAM per stage by (param_bytes + activation_proxy)
      5. Distribute MAC per stage by FLOPs
      6. Split DRAM into inputs/outputs/weights proportionally
    """
    j_per_mac   = FP16_J_PER_MAC if precision == "fp16" else TF32_J_PER_MAC
    pb          = param_bytes_fp16 if precision == "fp16" else param_bytes_fp32

    raw_mac_J   = sum(stage_gmacs.values()) * j_per_mac
    raw_dram_J  = total_dram_bytes * GDDR6X_J_PER_BYTE

    # Scale to measured total (rest = idle + host overhead, absorbed proportionally)
    accounted   = raw_mac_J + raw_dram_J
    if accounted > 0:
        scale = total_E_J / accounted
    else:
        scale = 1.0
    # Cap scale to avoid extreme stretching (overhead dominates at low utilisation)
    scale = min(scale, 4.0)

    mac_J_total  = raw_mac_J  * scale
    dram_J_total = raw_dram_J * scale

    total_flops = sum(stage_gmacs.values())
    total_pb    = sum(pb.values()) + 1e-9

    result = {}
    for s in stage_order:
        mac_frac   = stage_gmacs[s] / total_flops
        # Activation bytes proxy: proportional to stage time fraction (correlated w/ data volume)
        act_bytes  = total_dram_bytes * stage_frac[s]
        weight_bytes = pb[s]
        dram_frac  = (act_bytes + weight_bytes) / (total_dram_bytes + total_pb)

        mac_J    = mac_J_total  * mac_frac
        dram_J   = dram_J_total * dram_frac

        # Split DRAM into inputs / outputs / weights
        # Weights are read once; inputs+outputs are activations (split 60/40)
        if (act_bytes + weight_bytes) > 0:
            w_share = weight_bytes / (act_bytes + weight_bytes)
        else:
            w_share = 0.0
        act_share    = 1.0 - w_share
        inputs_J     = dram_J * act_share * 0.60
        outputs_J    = dram_J * act_share * 0.40
        weights_J    = dram_J * w_share

        result[s] = {
            "inputs":  inputs_J  * 1e3,   # → mJ
            "outputs": outputs_J * 1e3,
            "weights": weights_J * 1e3,
            "mac":     mac_J     * 1e3,
        }
    return result

en_fp32 = decompose_energy(E_fp32_J, "fp32")
en_fp16 = decompose_energy(E_fp16_J, "fp16")

# ── Aggregate to match Figure 8's column structure ───────────────────────────
# Figure 8 shows bars per precision with stacked operand types.
# We'll show: Total | DRAM | MAC  ×  FP32 | FP16
# with stacked: Inputs (blue) / Outputs (pink) / Weights (green) / MAC (yellow)

COLORS = {
    "inputs":  "#4C72B0",
    "outputs": "#DD8452",
    "weights": "#55A868",
    "mac":     "#C4A000",
}

def agg(en_dict, cols):
    """Sum selected component(s) across all stages."""
    return sum(en_dict[s][c] for s in stage_order for c in cols)

prec_labels = ["FP32", "FP16"]
datasets = [en_fp32, en_fp16]

groups = {
    "Total": ["inputs", "outputs", "weights", "mac"],
    "DRAM":  ["inputs", "outputs", "weights"],
    "MAC":   ["mac"],
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
fig.suptitle("PointPillar Energy Breakdown  ·  RTX 3080 Ti Laptop GPU",
             fontsize=13, fontweight="bold", y=1.02)

bar_w = 0.55
x = np.array([0, 1])  # FP32, FP16

for ax, (group_name, components) in zip(axes, groups.items()):
    bottoms = np.zeros(2)
    for comp in components:
        vals = np.array([agg(d, [comp]) for d in datasets])
        bars = ax.bar(x, vals, bar_w, bottom=bottoms,
                      color=COLORS[comp], label=comp.capitalize(),
                      edgecolor="white", linewidth=0.6)
        # Label segments if large enough
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

# Shared legend
handles = [mpatches.Patch(color=COLORS[c], label=c.capitalize())
           for c in ["inputs", "outputs", "weights", "mac"]]
fig.legend(handles=handles, title="Operand", loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=False)

# Annotation box: measured totals
annot = (f"Measured:  FP32 = {E_fp32_J*1e3:.1f} mJ/frame  "
         f"|  FP16 = {E_fp16_J*1e3:.1f} mJ/frame\n"
         f"Speedup FP16 vs FP32: {E_fp32_J/E_fp16_J:.2f}×  "
         f"(latency: FP32 13.75 ms → FP16 9.50 ms)")
fig.text(0.5, -0.14, annot, ha="center", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.85))

plt.tight_layout()
out = "profile_outputs/pointpillar_energy_breakdown.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()

# ── Also: per-stage stacked bar (like Figure 8 but with stages on x-axis) ────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle("PointPillar Energy by Pipeline Stage  ·  FP32 vs FP16",
              fontsize=13, fontweight="bold", y=1.02)

short = {
    "PillarVFE":          "PillarVFE",
    "PointPillarScatter": "PPScatter",
    "BaseBEVBackbone":    "BEVBackbone",
    "AnchorHeadSingle":   "AnchorHead",
    "post_processing":    "PostProc",
}
xs = np.arange(len(stage_order))
w2 = 0.35

for ax2, (group_name, components) in zip(axes2, groups.items()):
    for di, (en_dict, label, offset) in enumerate(
            [(en_fp32, "FP32", -w2/2), (en_fp16, "FP16", w2/2)]):
        bottoms = np.zeros(len(stage_order))
        for ci, comp in enumerate(components):
            vals = np.array([en_dict[s][comp] for s in stage_order])
            hatch = "" if di == 0 else "//"
            ax2.bar(xs + offset, vals, w2, bottom=bottoms,
                    color=COLORS[comp], alpha=0.85 if di == 0 else 0.6,
                    hatch=hatch, edgecolor="white", linewidth=0.5)
            bottoms += vals

    ax2.set_xticks(xs)
    ax2.set_xticklabels([short[s] for s in stage_order],
                        rotation=25, ha="right", fontsize=8.5)
    ax2.set_title(group_name, fontsize=12, fontweight="bold")
    ax2.set_ylabel("Energy (mJ)" if group_name == "Total" else "")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

# legend: operand type + precision
op_handles  = [mpatches.Patch(color=COLORS[c], label=c.capitalize()) for c in ["inputs","outputs","weights","mac"]]
prec_handles= [mpatches.Patch(facecolor="gray", label="FP32 (solid)"),
               mpatches.Patch(facecolor="gray", hatch="//", alpha=0.6, label="FP16 (hatch)")]
fig2.legend(handles=op_handles+prec_handles, loc="lower center",
            ncol=6, bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)

plt.tight_layout()
out2 = "profile_outputs/pointpillar_energy_per_stage.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
plt.close()
