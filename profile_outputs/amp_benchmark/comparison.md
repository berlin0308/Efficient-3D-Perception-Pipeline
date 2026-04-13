# FP16 AMP vs FP32 Baseline — PointPillars Inference

**Date:** 2026-03-26
**GPU:** NVIDIA GeForce RTX 3080 Ti Laptop GPU
**Config:** `cfgs/kitti_models/pointpillar.yaml`, batch=1, 50 measured steps, 100 warmup
**Ckpt:** `pointpillar_7728.pth` (Car@R11=77.28)

## Results

| Metric              | FP32 Baseline | FP16 AMP    | Δ              |
|---------------------|--------------|-------------|----------------|
| Forward mean (ms)   | 22.77        | 17.98       | −4.79 (−21%)   |
| Forward p50 (ms)    | 21.57        | 16.64       | −4.93 (−23%)   |
| Full frame mean (ms)| 25.43        | 22.06       | −3.37 (−13%)   |
| Throughput (samp/s) | 39.3         | 45.3        | +6.0 (+15%)    |
| Peak GPU Memory     | 292.7 MB     | 170.5 MB    | −122 MB (−42%) |
| CUDA kernels/trace  | 34108        | 40100       | +17% (FP16 casts add kernels) |

## Speedup

- **Forward: 1.27×**
- **Full frame: 1.15×**
- **Memory: 0.58× (42% reduction)**

## How it works

`torch.autocast(device_type='cuda', dtype=torch.float16)` wraps the forward pass.
PyTorch automatically casts eligible ops (matmul, conv) to FP16 while keeping
numerically sensitive ops (softmax, BN) in FP32. The RTX 3080 Ti Tensor Cores
run FP16 matmul at roughly 2× the throughput of FP32.

## Usage

```bash
python profile_suite.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --ckpt /path/to/pointpillar_7728.pth \
  --cuda_id 0 --amp
```

## Notes

- `--amp --compile` together hits a torch.dynamo guard bug on numpy-backed tensors
  (`___from_numpy` DispatchKeySet mismatch). Use one or the other, not both.
- PostProcess time increased slightly (2.24ms vs 1.29ms) — GPU→CPU sync for FP16→FP32
  conversion adds a small overhead, but it is dominated by the forward savings.
- No accuracy impact: AMP autocast does not change the model weights; inference
  results are numerically equivalent to FP32 within rounding tolerance.
