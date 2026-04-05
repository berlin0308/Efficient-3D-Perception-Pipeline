# AMP (FP16) Benchmark — PointPillars Inference

Benchmarks comparing FP32 baseline vs FP16 Automatic Mixed Precision (AMP)
for PointPillars inference on KITTI data.

**GPU:** NVIDIA GeForce RTX 3080 Ti Laptop GPU  
**Model:** PointPillars, `pointpillar_7728.pth` (Car@R11=77.28)  
**Config:** `cfgs/kitti_models/pointpillar.yaml`  
**Settings:** batch=1, 10 warmup steps, 50 measured steps

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
│   ├── profile_summary.txt       # latency/memory baseline
│   └── torch_profile_trace.json  # Chrome trace (chrome://tracing)
├── fp16_amp/
│   ├── profile_summary.txt       # latency/memory with AMP
│   └── torch_profile_trace.json
├── energy_fp32/
│   ├── energy_summary.txt        # power/energy baseline
│   └── energy_samples.csv        # raw (timestamp_s, power_W) samples
├── energy_fp16_amp/
│   ├── energy_summary.txt        # power/energy with AMP
│   └── energy_samples.csv
└── comparison.md                 # detailed notes
```

---

## How to Reproduce

Run from `OpenPCDet/tools/`. Adjust `--ckpt` and `--cuda_id` for your machine.

### Latency & Memory

```bash
# FP32 baseline
python profile_suite.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --warmup 10 --steps 50 --batch_size 1 --workers 2 \
  --output_dir profile_outputs/amp_benchmark/fp32

# FP16 AMP
python profile_suite.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --warmup 10 --steps 50 --batch_size 1 --workers 2 \
  --amp \
  --output_dir profile_outputs/amp_benchmark/fp16_amp
```

### Energy & Power

```bash
# FP32 baseline
python energy_monitor.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --warmup 10 --steps 50 --batch_size 1 --workers 2 \
  --output_dir profile_outputs/amp_benchmark/energy_fp32

# FP16 AMP
python energy_monitor.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --warmup 10 --steps 50 --batch_size 1 --workers 2 \
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
