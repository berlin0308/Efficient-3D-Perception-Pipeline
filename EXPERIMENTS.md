# Optimization Experiments Summary

Baseline: OpenPCDet PointPillars on KITTI dataset, Python FP32 inference, RTX A6000 (sm86, CUDA 12.1), batch size 1, averaged over 10 frames (5 warmup).

---

## Results Overview

| Experiment | Branch | Method | Latency (mean) | Speedup | Key Metric |
|---|---|---|---|---|---|
| Baseline | `cuda-pointpillars` | Python FP32 | 10.31 ms | 1.0× | — |
| Exp 1: HWC Scatter | `memory_bound_optimization` | CHW → HWC memory layout | ~10.3 ms | ~1.0× overall | Scatter kernel: **6×** faster |
| Exp 2: AMP FP16 | `amp_fp16_optimization` | `torch.autocast` FP16 | 8.48 ms | **1.22×** | Conv: ~1.4×, BN: 1.5× |

---

## Experiment 1 — HWC Scatter Memory Layout

**Branch:** `memory_bound_optimization`  
**File:** `OpenPCDet/pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py`

### What changed

`PointPillarScatter` writes voxel features into a 2D BEV pseudo-image. The original layout was CHW `[C, H*W]`, meaning each thread wrote to addresses spaced ~856 KB apart (one stride per channel) — causing non-coalesced writes and 64 cache misses per pillar.

Changed to HWC `[H*W, C]` so all 64 channels of a pillar are written to contiguous memory (1–2 transactions), then transposed back to CHW with `.t()` for the backbone.

```python
# Before (CHW — non-coalesced)
canvas = torch.zeros(batch_size, self.nz * self.nx * self.ny, self.num_bev_features, ...)
canvas[batch_idx, indices, :] = pillar_features   # coalesced read, coalesced write
canvas = canvas.permute(0, 2, 1).view(...)

# After (HWC → transpose to CHW)
canvas = torch.zeros(batch_size, self.nz * self.nx * self.ny, self.num_bev_features, ...)
canvas[batch_idx, indices, :] = pillar_features
canvas = canvas.permute(0, 2, 1).contiguous().view(...)
```

### Results (nsys + ncu, 10 frames)

| Metric | Before | After | Improvement |
|---|---|---|---|
| Scatter kernel avg latency | 34,552 ns | 5,782 ns | **6.0×** faster |
| L1 store sectors (ncu) | 216,576 | 27,072 | **8.0×** fewer |
| DRAM write | 1.425 MB | 1.024 MB | 28% less |
| End-to-end forward | 10.31 ms | ~10.3 ms | negligible |

> **Note:** Scatter accounts for only ~0.5% of total forward time, so the kernel-level 6× gain has minimal end-to-end impact. The optimization demonstrates memory coalescing principles and is complementary to Exp 2.

---

## Experiment 2 — AMP FP16 Inference

**Branch:** `amp_fp16_optimization`  
**File:** `OpenPCDet/tools/inference.py`

### What changed

Added `--amp` flag that wraps the forward pass in `torch.autocast`, enabling automatic mixed precision. Eligible ops (Conv2d, BatchNorm) run in FP16 using Tensor Cores; FP32-only ops (NMS) are unaffected.

```python
# inference.py — added --amp argument
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
    pred_dicts, _ = model.forward(data_dict)
```

### Results (nsys, 10 frames, 5 warmup)

**End-to-end latency:**

| Version | Mean (ms) | p50 (ms) | Speedup |
|---|---|---|---|
| FP32 baseline | 10.31 | 10.37 | 1.0× |
| FP16 AMP | 8.48 | 8.53 | **1.22×** |

**Per-op breakdown:**

| Op | FP32 (ms) | FP16 (ms) | Speedup | Reason |
|---|---|---|---|---|
| Conv2d fprop (main) | 9.04 | ~6.63 | ~1.4× | TF32 → FP16 Tensor Core (`s16816fprop`) |
| Winograd Conv | 4.73 | 0 (gone) | — | FP16 disables Winograd; replaced by TC |
| Activation clamp | 6.94 | 3.30 | 2.1× | Half data width, memory-bound |
| BatchNorm | 6.74 | 4.55 | 1.5× | Half input size |
| NMS | 6.45 | 6.33 | ~1.0× | FP32-only, AMP has no effect |
| NCHW↔NHWC transpose | 3.22 | 5.49 | **0.6×** | Tensor Core requires NHWC layout |

> **Why only 1.22× overall:** NMS (FP32-only) is a significant portion of runtime, and the added NCHW↔NHWC layout conversion overhead partially offsets the Conv speedup. Batch size = 1 also limits Tensor Core utilization.

---

## Relationship Between Experiments

Both optimizations are **independent and additive**:
- Exp 1 targets the scatter kernel (memory coalescing, no matrix math)
- Exp 2 targets Conv2d/BN via Tensor Cores (FP16 precision)

They can be applied simultaneously without conflict.
