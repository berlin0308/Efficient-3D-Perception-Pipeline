"""
PointPillar Energy Breakdown Plot
Mirrors Figure 8 structure: Total / DRAM / MAC energy broken down by pipeline
stage, across FP32 and FP16 precisions.

Method:
  - Total energy per frame: integrated from energy_samples.csv (energy_monitor.py output)
  - Stage time fractions: from nsys NVTX timings (report.sqlite)
  - DRAM vs MAC split: from nsys memcpy bytes + hardware constants
      DRAM energy:  bytes_transferred * GDDR6X_energy_per_bit
      MAC energy:   flops_estimated   * J_per_MAC
  - Stage operand breakdown (Inputs/Outputs/Weights/MAC):
      Inputs  ~ activations read  (proportional to HtoD + DtoD read share per stage)
      Outputs ~ activations write (DtoD write share)
      Weights ~ parameter bytes   (known per-layer from model config)
      MAC     ~ residual compute energy

Usage:
    python plot_energy.py \
        --fp32_csv  profile_outputs/amp_benchmark/energy_fp32/energy_samples.csv \
        --fp16_csv  profile_outputs/amp_benchmark/energy_fp16_amp/energy_samples.csv \
        --nsys_db   profile_outputs/nsys_baseline/report.sqlite \
        --fp32_steps 50 --fp16_steps 50 \
        --output_dir profile_outputs
"""

import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Hardware energy constants (RTX 3080 Ti / GDDR6X) ─────────────────────────
# GDDR6X: ~14 pJ/bit => 1.75 nJ/byte
GDDR6X_J_PER_BYTE = 1.75e-9
# TF32 Tensor Core: ~0.4 pJ/MAC
TF32_J_PER_MAC = 4e-13
# FP16 Tensor Core: ~0.2 pJ/MAC
FP16_J_PER_MAC = 2e-13

STAGE_ORDER = ["PillarVFE", "PointPillarScatter", "BaseBEVBackbone",
               "AnchorHeadSingle", "post_processing"]

# Approximate parameter sizes per stage (FP32 bytes)
PARAM_BYTES_FP32 = {
    "PillarVFE":          0.3e6 * 4,
    "PointPillarScatter": 0,
    "BaseBEVBackbone":    4.9e6 * 4,
    "AnchorHeadSingle":   0.4e6 * 4,
    "post_processing":    0,
}

# Approximate FLOPs per stage
STAGE_GMACS = {
    "PillarVFE":          1.5e9,
    "PointPillarScatter": 0.05e9,
    "BaseBEVBackbone":   18.0e9,
    "AnchorHeadSingle":   1.2e9,
    "post_processing":    0.05e9,
}

COLORS = {
    "inputs":  "#4C72B0",
    "outputs": "#DD8452",
    "weights": "#55A868",
    "mac":     "#C4A000",
}

GROUPS = {
    "Total": ["inputs", "outputs", "weights", "mac"],
    "DRAM":  ["inputs", "outputs", "weights"],
    "MAC":   ["mac"],
}


def parse_args():
    parser = argparse.ArgumentParser(description='PointPillar energy breakdown plot')
    parser.add_argument('--fp32_csv',
                        default='profile_outputs/amp_benchmark/energy_fp32/energy_samples.csv',
                        help='energy_samples.csv from FP32 energy_monitor.py run')
    parser.add_argument('--fp16_csv',
                        default='profile_outputs/amp_benchmark/energy_fp16_amp/energy_samples.csv',
                        help='energy_samples.csv from FP16 AMP energy_monitor.py run')
    parser.add_argument('--nsys_db',
                        default='profile_outputs/nsys_baseline/report.sqlite',
                        help='nsys report.sqlite for NVTX stage timings')
    parser.add_argument('--fp32_steps', type=int, default=50,
                        help='number of measured inference steps in the FP32 run')
    parser.add_argument('--fp16_steps', type=int, default=50,
                        help='number of measured inference steps in the FP16 run')
    parser.add_argument('--output_dir', default='profile_outputs',
                        help='directory to save output PNG files')
    return parser.parse_args()


def integrate_csv(csv_path):
    """Integrate power over time from energy_samples.csv -> total Joules."""
    timestamps, powers = [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp_s']))
            powers.append(float(row['power_W']))
    ts = np.array(timestamps)
    pw = np.array(powers)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz(pw, ts))


def load_nsys_data(db_path):
    """Query nsys sqlite for stage fractions and DRAM bytes. Returns (stage_frac, total_dram_bytes)."""
    con = sqlite3.connect(db_path)

    # NVTX stage durations (skip first iter as warmup)
    raw = con.execute("""
        SELECT text, (end-start)/1e6
        FROM NVTX_EVENTS
        WHERE text IN ('PillarVFE','PointPillarScatter','BaseBEVBackbone',
                       'AnchorHeadSingle','post_processing')
          AND end IS NOT NULL AND end > start
        ORDER BY start
    """).fetchall()

    per_iter = defaultdict(list)
    for name, ms in raw:
        per_iter[name].append(ms)
    for k in per_iter:
        per_iter[k] = per_iter[k][1:]  # drop first (JIT warmup)

    stage_ms = {s: np.mean(per_iter[s]) if per_iter[s] else 1.0 for s in STAGE_ORDER}
    total_ms = sum(stage_ms.values())
    stage_frac = {s: stage_ms[s] / total_ms for s in STAGE_ORDER}

    # DRAM bytes from memcpy (HtoD=1, DtoH=2, DtoD=8)
    memcpy_rows = con.execute(
        "SELECT copyKind, SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
    ).fetchall()
    con.close()

    bytes_by_kind = {k: b for k, b in memcpy_rows}
    n_profile_frames = 25
    total_dram_bytes = (
        bytes_by_kind.get(1, 0) +
        bytes_by_kind.get(8, 0) +
        bytes_by_kind.get(2, 0)
    ) / n_profile_frames

    return stage_frac, total_dram_bytes


def decompose_energy(total_E_J, precision, stage_frac, total_dram_bytes):
    """
    Returns dict: stage -> {inputs, outputs, weights, mac} in mJ.
    Distributes measured total energy across stages and operand types using
    FLOPs fractions (MAC) and memory traffic fractions (DRAM).
    """
    j_per_mac = FP16_J_PER_MAC if precision == "fp16" else TF32_J_PER_MAC
    pb = {k: v / 2 for k, v in PARAM_BYTES_FP32.items()} if precision == "fp16" else PARAM_BYTES_FP32

    raw_mac_J  = sum(STAGE_GMACS.values()) * j_per_mac
    raw_dram_J = total_dram_bytes * GDDR6X_J_PER_BYTE

    accounted = raw_mac_J + raw_dram_J
    scale = min(total_E_J / accounted, 4.0) if accounted > 0 else 1.0

    mac_J_total  = raw_mac_J  * scale
    dram_J_total = raw_dram_J * scale

    total_flops = sum(STAGE_GMACS.values())
    total_pb    = sum(pb.values()) + 1e-9

    result = {}
    for s in STAGE_ORDER:
        mac_frac     = STAGE_GMACS[s] / total_flops
        act_bytes    = total_dram_bytes * stage_frac[s]
        weight_bytes = pb[s]
        dram_frac    = (act_bytes + weight_bytes) / (total_dram_bytes + total_pb)

        mac_J  = mac_J_total  * mac_frac
        dram_J = dram_J_total * dram_frac

        w_share   = weight_bytes / (act_bytes + weight_bytes) if (act_bytes + weight_bytes) > 0 else 0.0
        act_share = 1.0 - w_share

        result[s] = {
            "inputs":  dram_J * act_share * 0.60 * 1e3,   # mJ
            "outputs": dram_J * act_share * 0.40 * 1e3,
            "weights": dram_J * w_share          * 1e3,
            "mac":     mac_J                     * 1e3,
        }
    return result


def agg(en_dict, cols):
    return sum(en_dict[s][c] for s in STAGE_ORDER for c in cols)


def plot_precision_bars(en_fp32, en_fp16, E_fp32_J, E_fp16_J, output_dir):
    """Figure 1: stacked bars by precision (FP32 vs FP16), grouped by Total/DRAM/MAC."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)
    fig.suptitle("PointPillar Energy Breakdown  ·  FP32 vs FP16 AMP",
                 fontsize=13, fontweight="bold", y=1.02)

    bar_w = 0.55
    x = np.array([0, 1])
    datasets = [en_fp32, en_fp16]
    prec_labels = ["FP32", "FP16 AMP"]

    for ax, (group_name, components) in zip(axes, GROUPS.items()):
        bottoms = np.zeros(2)
        for comp in components:
            vals = np.array([agg(d, [comp]) for d in datasets])
            ax.bar(x, vals, bar_w, bottom=bottoms,
                   color=COLORS[comp], edgecolor="white", linewidth=0.6)
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

    handles = [mpatches.Patch(color=COLORS[c], label=c.capitalize())
               for c in ["inputs", "outputs", "weights", "mac"]]
    fig.legend(handles=handles, title="Operand", loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=False)

    annot = (f"Measured:  FP32 = {E_fp32_J*1e3:.1f} mJ/frame"
             f"  |  FP16 AMP = {E_fp16_J*1e3:.1f} mJ/frame\n"
             f"Energy reduction: {(1 - E_fp16_J/E_fp32_J)*100:.1f}%  "
             f"({E_fp32_J/E_fp16_J:.2f}× less energy per frame)")
    fig.text(0.5, -0.14, annot, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.85))

    plt.tight_layout()
    out = str(Path(output_dir) / "pointpillar_energy_breakdown.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def plot_stage_bars(en_fp32, en_fp16, output_dir):
    """Figure 2: stacked bars by pipeline stage, FP32 vs FP16 side-by-side."""
    short = {
        "PillarVFE":          "PillarVFE",
        "PointPillarScatter": "PPScatter",
        "BaseBEVBackbone":    "BEVBackbone",
        "AnchorHeadSingle":   "AnchorHead",
        "post_processing":    "PostProc",
    }

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
    fig2.suptitle("PointPillar Energy by Pipeline Stage  ·  FP32 vs FP16 AMP",
                  fontsize=13, fontweight="bold", y=1.02)

    xs = np.arange(len(STAGE_ORDER))
    w2 = 0.35

    for ax2, (group_name, components) in zip(axes2, GROUPS.items()):
        for di, (en_dict, offset) in enumerate([(en_fp32, -w2/2), (en_fp16, w2/2)]):
            bottoms = np.zeros(len(STAGE_ORDER))
            for comp in components:
                vals = np.array([en_dict[s][comp] for s in STAGE_ORDER])
                hatch = "" if di == 0 else "//"
                ax2.bar(xs + offset, vals, w2, bottom=bottoms,
                        color=COLORS[comp], alpha=0.85 if di == 0 else 0.6,
                        hatch=hatch, edgecolor="white", linewidth=0.5)
                bottoms += vals

        ax2.set_xticks(xs)
        ax2.set_xticklabels([short[s] for s in STAGE_ORDER],
                            rotation=25, ha="right", fontsize=8.5)
        ax2.set_title(group_name, fontsize=12, fontweight="bold")
        ax2.set_ylabel("Energy (mJ)" if group_name == "Total" else "")
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)

    op_handles = [mpatches.Patch(color=COLORS[c], label=c.capitalize())
                  for c in ["inputs", "outputs", "weights", "mac"]]
    prec_handles = [mpatches.Patch(facecolor="gray", label="FP32 (solid)"),
                    mpatches.Patch(facecolor="gray", hatch="//", alpha=0.6, label="FP16 AMP (hatch)")]
    fig2.legend(handles=op_handles + prec_handles, loc="lower center",
                ncol=6, bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)

    plt.tight_layout()
    out2 = str(Path(output_dir) / "pointpillar_energy_per_stage.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out2}")
    plt.close()


def main():
    args = parse_args()

    # ── Load energy from CSV (no hardcoded values) ─────────────────────────
    print(f"Reading FP32 energy from {args.fp32_csv}")
    total_energy_fp32 = integrate_csv(args.fp32_csv)
    E_fp32_J = total_energy_fp32 / args.fp32_steps
    print(f"  Total: {total_energy_fp32:.2f} J over {args.fp32_steps} steps -> {E_fp32_J*1e3:.2f} mJ/frame")

    print(f"Reading FP16 energy from {args.fp16_csv}")
    total_energy_fp16 = integrate_csv(args.fp16_csv)
    E_fp16_J = total_energy_fp16 / args.fp16_steps
    print(f"  Total: {total_energy_fp16:.2f} J over {args.fp16_steps} steps -> {E_fp16_J*1e3:.2f} mJ/frame")

    # ── Load nsys stage timings and DRAM bytes ─────────────────────────────
    print(f"Reading nsys data from {args.nsys_db}")
    stage_frac, total_dram_bytes = load_nsys_data(args.nsys_db)
    print(f"  Stage fractions: { {k: f'{v:.3f}' for k, v in stage_frac.items()} }")
    print(f"  Total DRAM bytes/frame: {total_dram_bytes/1e6:.2f} MB")

    # ── Decompose energy per stage and operand ─────────────────────────────
    en_fp32 = decompose_energy(E_fp32_J, "fp32", stage_frac, total_dram_bytes)
    en_fp16 = decompose_energy(E_fp16_J, "fp16", stage_frac, total_dram_bytes)

    # ── Plot ───────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_precision_bars(en_fp32, en_fp16, E_fp32_J, E_fp16_J, args.output_dir)
    plot_stage_bars(en_fp32, en_fp16, args.output_dir)


if __name__ == '__main__':
    main()
