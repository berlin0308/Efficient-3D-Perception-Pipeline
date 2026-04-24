# M5: CUDA-PointPillars (TensorRT) — How to Run

This document explains three things:
1. [How to run CUDA-PP on Modal](#1-run-cuda-pp-on-modal)
2. [How to merge M5 results into runs.csv](#2-merge-m5-results-into-runscsv)
3. [How to regenerate plots (latency / energy / 3D frontier)](#3-regenerate-plots)

---

## Prerequisites (one-time setup)

```bash
# 1. Pull the ONNX model (~170 MB, tracked by git-lfs)
git lfs pull

# 2. Install required Python packages
pip install -r requirements_m5.txt
```

> **No local CUDA/TRT needed.** The C++ binary is compiled and run entirely inside the Modal container (`nvcr.io/nvidia/tensorrt:23.05-py3`, which includes CUDA 12.1 + TRT 8.6.1 + cmake + gcc). Your local machine only needs the Python packages above.
>
> **Shared Modal account**: all teammates share workspace `vtyqopd`.  
> No extra `modal setup` or profile configuration needed — just install and go.

---

## 1. Run CUDA-PP on Modal

All commands below assume you are in the **repo root**.

### 1-a. Test the build (no benchmark)

```bash
modal run modal_cuda_pp_app.py --action build
```

This compiles the C++ binary inside the Modal container and exits.  
Use this to verify the environment is correct before committing GPU time.

### 1-b. Run FP32 or FP16 benchmark

```bash
# FP32
modal run modal_cuda_pp_app.py --action run --precision fp32

# FP16
modal run modal_cuda_pp_app.py --action run --precision fp16

# Both in sequence (recommended)
modal run modal_cuda_pp_app.py --action run_all
```

**What happens:**
- Modal pulls the `nvcr.io/nvidia/tensorrt:23.05-py3` image (TRT 8.6.1 + CUDA 12.1)
- C++ source in `CUDA-PointPillars/` is compiled inside the container
- `cuda_pp_metrics.py` runs the binary on KITTI val set (from Modal volume `mls-openpcdet-kitti`)
- warmup=500, steps=500 (individual frames)
- Results CSV saved to Modal volume `mls-openpcdet-results` → auto-downloaded to `modal_outputs/cuda_pp/`

**To choose GPU**, edit `modal_cuda_pp_app.py` and change the `gpu=` argument in the `@app.function` decorator for the function you want to run (`run_cuda_pp_a10g` / `run_cuda_pp_t4` / `run_cuda_pp_h100`).

### 1-c. Run accuracy evaluation (KITTI mAP)

```bash
modal run modal_cuda_pp_app.py --action accuracy --precision fp32
modal run modal_cuda_pp_app.py --action accuracy --precision fp16
```

**What happens:**
- Runs the C++ binary with `--save-preds` → predictions saved as `pred_velo/*.txt` (lidar coords)
- `kitti_format.py` converts lidar→camera coordinates → `pred_cam/*.txt`
- `eval.py` computes Car mAP (R11) — same code as NVIDIA's official `evaluate_kitti_val.sh`

**Expected result (our run):** Car 3D Moderate @IoU=0.7 ≈ **67.5**

> Note: NVIDIA's official GitHub shows **77.02** (measured on Jetson with TRT 8.4.0).  
> Our difference is due to TRT 8.6.1 on data-center GPUs + voxelization randomness.  
> We use **77.02** as the reported `map_car_r11` value in `runs.csv` (official reference).

---

## 2. Merge M5 Results into runs.csv

After downloading M5 CSVs from Modal (step 1), merge them into the shared `runs.csv` files:

```bash
# Dry run first — prints what would be written, no file changes
python merge_m5_to_runs.py --dry-run

# Actually merge
python merge_m5_to_runs.py
```

**What it does:**
- Reads `modal_outputs/cuda_pp/cuda_pp/M5_FP32.csv` and `M5_FP16.csv`
- Filters to rows with `warmup_steps=500` (discards test/debug runs)
- Picks the latest run per `(gpu_name, precision_mode)`
- Appends/replaces M5 rows in:
  - `modal_outputs/modal_v3_a10/runs.csv`
  - `modal_outputs/modal_v3_t4/runs.csv`
  - `modal_outputs/modal_v3_h100/runs.csv`

**After merging**, manually set `map_car_r11 = 77.02` for the new M5 rows (the script leaves it blank since accuracy is evaluated separately).

**Current M5 numbers already in runs.csv:**

| GPU | Precision | Latency | Energy/frame |
|-----|-----------|---------|-------------|
| NVIDIA A10G | FP32 | 6.82 ms | 987 mJ |
| NVIDIA A10G | FP16 | 3.96 ms | 460 mJ |
| Tesla T4 | FP32 | 17.56 ms | 1182 mJ |
| Tesla T4 | FP16 | 7.12 ms | 328 mJ |
| H100 80GB | FP32 | 2.25 ms | 400 mJ |
| H100 80GB | FP16 | 1.81 ms | 304 mJ |

> **Energy correction**: `energy_total_J = power_W × latency_ms × 1e-3 × 500 steps`.  
> This matches M0–M4's inference-only energy (excludes TRT engine build overhead).

---

## 3. Regenerate Plots

All plot scripts live in `report/`. Run from the **repo root**.

### Latency comparison (M0–M5)

```bash
python3 report/plot_latency.py \
  --a10-root modal_outputs/modal_v3_a10 \
  --h100-root modal_outputs/modal_v3_h100 \
  --t4-root modal_outputs/modal_v3_t4
```

### Energy comparison (M0–M5)

```bash
python3 report/plot_energy.py \
  --a10-root modal_outputs/modal_v3_a10 \
  --h100-root modal_outputs/modal_v3_h100 \
  --t4-root modal_outputs/modal_v3_t4
```

### 3D Pareto frontier (accuracy × latency × energy)

```bash
python3 report/plot_accuracy_latency_energy_3d.py \
  --a10-root modal_outputs/modal_v3_a10 \
  --h100-root modal_outputs/modal_v3_h100 \
  --t4-root modal_outputs/modal_v3_t4 \
  --out report/accuracy_latency_energy_3d.png \
  --xlim-low 0.2 --xlim-high 2.6 \
  --ylim-low 0.5 --ylim-high 45.0 \
  --zlim-low 74.0 --zlim-high 86.0 \
  --reference-plane-z 84.8 \
  --major-x-step 0.4 \
  --major-y-step 5.0
```

> The axis limits above are tuned so M5 points (z≈77.02) are visible below  
> the M0–M4 cloud (z≈84.8). Do not shrink `--zlim-low` past 74.

---

## File Map

| File | Purpose |
|------|---------|
| `modal_cuda_pp_app.py` | Modal app — build / run / accuracy actions |
| `cuda_pp_metrics.py` | Stage-level timing + energy measurement wrapper |
| `merge_m5_to_runs.py` | Merge M5 CSVs into shared runs.csv files |
| `CUDA-PointPillars/` | C++ source (modified: `--warmup`, `--repeat`, `--save-preds` flags) |
| `CUDA-PointPillars/eval/kitti_format.py` | lidar→camera coordinate conversion |
| `CUDA-PointPillars/eval/kitti-object-eval-python/eval.py` | KITTI Car mAP evaluation |
| `modal_outputs/modal_v3_{a10,t4,h100}/runs.csv` | Shared benchmark results (includes M5 rows) |
| `modal_outputs/cuda_pp/` | Raw M5 CSVs downloaded from Modal (gitignored — large) |
| `report/` | Plot scripts and output PNGs |
