# MLS Quick Commands

This README is a quick command cheat sheet for:
- Modal v5 runs (T4 / A10 / H100)
- CUDA-PointPillars commands
- Plot regeneration commands

---

## 1) Modal v5 (M0-M4) commands

Common settings:
- `warmup=1000`
- `steps=2000`
- full KITTI eval (`--kitti-full-val`)
- detached run (`--detach`)
- clean `runs.csv` (`--extra-args "--fresh_runs"`)

### T4

```bash
MLS_MODAL_RESEARCH_GPU=T4 modal run --detach modal_mls_app.py \
  --action run \
  --warmup 1000 \
  --steps 2000 \
  --matrix fp32_amp \
  --kitti-full-val \
  --output-root /mnt/results/modal_v5_t4 \
  --runs-csv /mnt/results/modal_v5_t4/runs.csv \
  --extra-args "--fresh_runs" \
  --skip-download
```

### A10

```bash
MLS_MODAL_RESEARCH_GPU=A10 modal run --detach modal_mls_app.py \
  --action run \
  --warmup 1000 \
  --steps 2000 \
  --matrix fp32_amp \
  --kitti-full-val \
  --output-root /mnt/results/modal_v5_a10 \
  --runs-csv /mnt/results/modal_v5_a10/runs.csv \
  --extra-args "--fresh_runs" \
  --skip-download
```

### H100

```bash
MLS_MODAL_RESEARCH_GPU=H100 modal run --detach modal_mls_app.py \
  --action run \
  --warmup 1000 \
  --steps 2000 \
  --matrix fp32_amp \
  --kitti-full-val \
  --output-root /mnt/results/modal_v5_h100 \
  --runs-csv /mnt/results/modal_v5_h100/runs.csv \
  --extra-args "--fresh_runs" \
  --skip-download
```

### Download back to local (after run finishes)

```bash
modal volume get mls-openpcdet-results /modal_v5_t4   ./modal_outputs --force
modal volume get mls-openpcdet-results /modal_v5_a10  ./modal_outputs --force
modal volume get mls-openpcdet-results /modal_v5_h100 ./modal_outputs --force
```

---

## 2) CUDA-PointPillars commands

See full details in `CUDA_PP_README.md`. Common commands:

```bash
# Build only
modal run modal_cuda_pp_app.py --action build

# Run FP32 + FP16 sequence
modal run modal_cuda_pp_app.py --action run_all

# Accuracy eval
modal run modal_cuda_pp_app.py --action accuracy --precision fp32
modal run modal_cuda_pp_app.py --action accuracy --precision fp16

# Merge M5 into runs.csv
python merge_m5_to_runs.py --dry-run
python merge_m5_to_runs.py
```

---

## 3) Plot commands (use v5 outputs)

Run from repo root:

```bash
python3 report/plot_latency.py \
  --a10-root modal_outputs/modal_v5_a10 \
  --h100-root modal_outputs/modal_v5_h100 \
  --t4-root modal_outputs/modal_v5_t4
```

```bash
python3 report/plot_energy.py \
  --a10-root modal_outputs/modal_v5_a10 \
  --h100-root modal_outputs/modal_v5_h100 \
  --t4-root modal_outputs/modal_v5_t4
```

```bash
python3 report/plot_accuracy_latency_energy_3d.py \
  --a10-root modal_outputs/modal_v5_a10 \
  --h100-root modal_outputs/modal_v5_h100 \
  --t4-root modal_outputs/modal_v5_t4 \
  --out report/accuracy_latency_energy_3d_v5.png
```
