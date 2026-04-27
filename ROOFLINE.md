# Roofline Analysis

Roofline-style plot for the PointPillars forward pass across all experiment variants (M0–M5) on A10G, H100, and T4.

## Prerequisites

- GPU machine with CUDA 12.1 and `ncu` (Nsight Compute)
- CUDA-PointPillars FP32 binary: `build/demo`
- CUDA-PointPillars FP16 binary: `build-fp16/demo` (requires separate build)

## Step 1 — Measure Arithmetic Intensity

Run `ncu_intensity_sweep.py` to profile all 16 variants. It uses `ncu` to measure DRAM load/store bytes per variant and computes intensity = theoretical FLOPs / DRAM bytes.

```bash
sudo -E python ncu_intensity_sweep.py \
    --demo /path/to/CUDA-PointPillars/build/demo \
    --demo-fp16 /path/to/CUDA-PointPillars/build-fp16/demo
```

Output: `ncu_intensity.csv` — intensity values for all 16 variants.

## Step 2 — Merge Intensity into runs.csv

```bash
python3 merge_ncu_intensity.py
```

This fills the `ncu_compute_intensity_flop_per_byte` column in the three `modal_outputs/modal_v3_{a10,h100,t4}/runs.csv` files.

## Step 3 — Plot

```bash
# A10G
python report/plot_roofline_forward.py \
  --runs-root modal_outputs/modal_v3_a10 --gpu "NVIDIA A10G" \
  --peak-tflops 31.2 --dram-bandwidth-gbs 600

# H100
python report/plot_roofline_forward.py \
  --runs-root modal_outputs/modal_v3_h100 --gpu "NVIDIA H100 80GB HBM3" \
  --peak-tflops 67.0 --dram-bandwidth-gbs 3350

# T4
python report/plot_roofline_forward.py \
  --runs-root modal_outputs/modal_v3_t4 --gpu "Tesla T4" \
  --peak-tflops 8.1 --dram-bandwidth-gbs 300
```

Output PNGs are saved under `report/`, or specify a custom path with `--out`.

## Reading the Plot

- **X-axis**: Arithmetic Intensity (FLOP/byte) — compute operations per byte of DRAM traffic
- **Y-axis**: Attained GFLOP/s — measured throughput
- **Diagonal line**: memory bandwidth ceiling
- **Horizontal dashed line**: compute ceiling (peak TFLOP/s)
- **Ridge point**: where the two ceilings meet — A10G ≈ 52, H100 ≈ 20, T4 ≈ 27 FLOP/byte
- **Takeaway**: all 16 variants fall below the diagonal → all **memory-bound**; the bottleneck is DRAM bandwidth, not compute

## Results

| Variant | Intensity (FLOP/byte) |
|---|---|
| M0_FP32 | 3.32 |
| M0_AMP | 5.94 |
| M1_FP32 | 2.88 |
| M1_AMP | 5.04 |
| M2_FP32_mem_scatter | 3.08 |
| M2_FP32_mem_conv2d | 2.42 |
| M2_FP32_mem_both | 2.29 |
| M2_AMP_mem_scatter | 5.25 |
| M2_AMP_mem_conv2d | 3.85 |
| M2_AMP_mem_both | 3.55 |
| M3_FP32 | 3.30 |
| M3_AMP | 5.87 |
| M4_FP32 | 2.28 |
| M4_AMP | 3.53 |
| M5_FP32 | 3.40 |
| M5_FP16 | 3.40 |

All variants are memory-bound (A10G ridge point ≈ 52 FLOP/byte).
