# AMP (FP16) Benchmark — PointPillars Inference

Benchmarks comparing FP32 baseline vs FP16 Automatic Mixed Precision (AMP)
for PointPillars inference on KITTI data.

**GPU:** NVIDIA GeForce RTX 3080 Ti Laptop GPU  
**Model:** PointPillars, `pointpillar_7728.pth` (Car@R11=77.28)  
**Config:** `cfgs/kitti_models/pointpillar.yaml`  
**Settings:** batch=1, 100 warmup steps, 50 measured steps

---

## Results

### Latency & Memory (`profile_suite.py`)

| Stage | FP32 | FP16 AMP | Δ |
|---|---|---|---|
| Forward (mean) | 27.03 ms | 18.81 ms | −30% |
| Forward (p50) | 27.06 ms | 17.97 ms | −34% |
| Full frame (mean) | 29.81 ms | 21.60 ms | −28% |
| Throughput | 33.5 samples/s | 46.3 samples/s | +38% |
| Peak GPU memory | 292.7 MB | 170.5 MB | −42% |

### Energy & Power (`energy_monitor.py`)

| Metric | FP32 | FP16 AMP | Δ |
|---|---|---|---|
| Mean latency | 14.68 ms | 9.57 ms | −35% |
| Throughput | 57.09 samples/s | 70.92 samples/s | +24% |
| Mean power | 43.4 W | 36.9 W | −15% |
| Peak power | 66.7 W | 42.3 W | −37% |
| Total energy | 37.57 J | 25.97 J | −31% |
| Samples/J | 1.33 | 1.93 | +45% |
| Samples/s/W | 1.32 | 1.92 | +46% |

---

## Output Files

```
amp_benchmark/
├── fp32/
│   ├── profile_summary.txt          # latency/memory baseline (incl. p95 in newer runs)
│   ├── latency_per_step.csv         # per-step stage latencies (for distributions / p95–p99)
│   └── torch_profile_trace.json     # Chrome trace (chrome://tracing)
├── fp16_amp/
│   ├── profile_summary.txt
│   ├── latency_per_step.csv
│   └── torch_profile_trace.json
├── energy_fp32/
│   ├── energy_summary.txt           # power/energy baseline (incl. p95 forward latency)
│   ├── energy_samples.csv           # raw (timestamp_s, power_W) samples
│   └── energy_latency_per_step.csv  # per-step forward wall time (ms)
├── energy_fp16_amp/
│   ├── energy_summary.txt
│   ├── energy_samples.csv
│   └── energy_latency_per_step.csv
└── comparison.md                    # detailed notes
```

---

## Energy methodology (DRAM vs on-chip memory)

`energy_monitor.py` reports **whole-GPU** power via NVML and integrates it to Joules. That value **does not** separate DRAM energy from SRAM / on-chip cache energy.

For claims such as “data movement dominates” or “DRAM energy exceeds SRAM,” combine NVML with at least one of:

1. **Nsight Compute (NCU)** — memory throughput, L2 hit rate, DRAM bytes moved (compute vs memory bound).
2. **An energy model** — multiply measured bytes (from NCU or a simulator) by published J/byte assumptions, and cite the model.
3. **Platform-specific counters** — if available on your hardware.

In aggregated CSV rows, set `energy_method` to `nvml_integrated` for monitor runs, or document a composite method (e.g. `ncu_bytes_times_energy_model`) when you add NCU-based analysis. The frozen column list and longer text live in `OpenPCDet/tools/research_metrics_schema.py` (`ENERGY_BREAKDOWN_METHODOLOGY`).

---

## Research CSV aggregation (`collect_research_metrics.py`)

To merge profile + energy directories into one **`runs.csv`** for plotting (latency, energy, optional accuracy / NCU paths), use the collector from `OpenPCDet/tools/`:

```bash
# From OpenPCDet/tools/: set checkpoint (required if not under ckpt/pointpillar_7728.pth)
# Optional: only if ckpt is not at OpenPCDet/tools/ckpt/pointpillar_7728.pth
export OPENPCDET_CKPT=/your/path/pointpillar_7728.pth

# Defaults: cfg cfgs/kitti_models/pointpillar.yaml, ckpt tools/ckpt/pointpillar_7728.pth, output MLS/profile_outputs/research_matrix
python collect_research_metrics.py run --cuda_id 0 --warmup 100 --steps 50

# Or use the wrapper (sets OPENPCDET_CKPT from repo ckpt/ or /media/emma/... when present)
bash ../scripts/run_collect_research_metrics.sh --cuda_id 0

# Merge existing result folders (manifest = JSON array)
python collect_research_metrics.py merge \
  --manifest manifest.json \
  --runs_csv ../../profile_outputs/research_matrix/runs.csv

# 5×3 = 15 experiment design (export status table for the report; see tools/RESEARCH_EXPERIMENT_MATRIX.md)
python collect_research_metrics.py matrix --output_csv ../../profile_outputs/research_matrix/experiment_matrix_15.csv

# Run only *runnable* cells from that design (same three configs as legacy when tooling unchanged)
python collect_research_metrics.py run --matrix 15 --cuda_id 0 --warmup 100 --steps 50
```

`manifest.json` example:

```json
[
  {
    "variant_name": "baseline_fp32",
    "profile_dir": "/path/to/fp32",
    "energy_dir": "/path/to/energy_fp32",
    "map_car_r11": "77.28",
    "ncu_csv": "/path/to/ncu_export.csv"
  }
]
```

Column order is defined once in `research_metrics_schema.py` (`RUNS_CSV_COLUMNS`). Optional NCU kernel detail is appended to `ncu_kernels.csv` next to `runs.csv` when `ncu_csv` is set and the export is parseable.

---

## How to Reproduce

Run from `OpenPCDet/tools/`. Adjust `--ckpt` and `--cuda_id` for your machine.

### Latency & Memory

```bash
# FP32 baseline
python profile_suite.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --warmup 100 --steps 50 --batch_size 1 --workers 2 \
  --output_dir profile_outputs/amp_benchmark/fp32

# FP16 AMP
python profile_suite.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --warmup 100 --steps 50 --batch_size 1 --workers 2 \
  --amp \
  --output_dir profile_outputs/amp_benchmark/fp16_amp
```

### Energy & Power

```bash
# FP32 baseline
python energy_monitor.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --warmup 100 --steps 50 --batch_size 1 --workers 2 \
  --output_dir profile_outputs/amp_benchmark/energy_fp32

# FP16 AMP
python energy_monitor.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --warmup 100 --steps 50 --batch_size 1 --workers 2 \
  --amp \
  --output_dir profile_outputs/amp_benchmark/energy_fp16_amp
```

### Prerequisites

- Run from `OpenPCDet/tools/`
- Python env with `torch`, `pcdet` installed
- `pip install nvidia-ml-py` (required by `energy_monitor.py`)
- KITTI velodyne + calib + label data, and `pointpillar_7728.pth` checkpoint
- Use `--cuda_id 0` for single-GPU machines (check `nvidia-smi` if unsure)

---

## How AMP Works

`torch.autocast(device_type='cuda', dtype=torch.float16)` wraps the forward pass.
PyTorch automatically runs matmul-heavy ops (conv, linear) in FP16 using Tensor Cores,
while keeping numerically sensitive ops (batch norm, softmax) in FP32.
Model weights are unchanged — the FP32 checkpoint works as-is.

The RTX 3080 Ti Tensor Cores run FP16 matrix multiplications at roughly 2× the
throughput of FP32, which reduces both latency and power draw.

> Note: `--amp` and `--compile` cannot be used together — they trigger a
> torch.dynamo guard bug on numpy-backed tensors. Use one or the other.
