# Nsight Comparison Plan: PointPillar OpenPCDet (Compiled vs Baseline)

Concise comparison framework and interpretation for Nsight Systems stats.

---

## Data Sources

| Run       | Stats file |
|----------|------------|
| Baseline | `pointpillars/openpcdet/nsight_stats_20260226_123607_baseline.txt` |
| Compiled | `pointpillars/openpcdet/nsight_stats_20260226_123310_compiled.txt` |

(Compiled = TorchScript traced `.pt` forward; post_processing unchanged.)

---

## Metrics Snapshot

| Metric | Baseline | Compiled |
|--------|----------|----------|
| **NVTX forward** (avg) | ~72.4 ms | ~27.4 ms |
| **cudaLaunchKernel** | 16,752 calls | 10,014 calls |
| **cuLaunchKernel** | ~2 calls | 1,583 calls |
| **cudaMemcpyAsync** | 1,976 | 1,952 |

---

## What Conforms to Expectations

- **Faster forward (compiled)**  
  Lower NVTX forward time for compiled run → traced/JIT path is doing useful work.

- **Fewer small clamp kernels (compiled)**  
  Baseline: `launch_clamp_scalar` is a major kernel (6.6% time, 1,540 instances).  
  Compiled: same kernel ~1.5% time, 229 instances → **fusion or inlining** of clamp into larger kernels.

- **cuLaunchKernel vs cudaLaunchKernel**  
  - **cudaLaunchKernel**: Runtime API; used by PyTorch for pre-built (e.g. ATen/cuDNN) kernels.  
  - **cuLaunchKernel**: Driver API; used when launching from JIT-compiled modules (e.g. Inductor/Triton).  
  Compiled run has many more `cuLaunchKernel` calls → part of the workload is running via JIT-compiled code.

- **Fewer total kernel launches (compiled)**  
  Fewer `cudaLaunchKernel` calls with similar or better forward time → fewer small kernels (fusion) and/or more work per launch.

- **D2D / memcpy**  
  Similar or slightly lower `cudaMemcpyAsync` in compiled run → no unexpected extra copies.

---

## What Does Not Conform / Open Points

- **JIT coverage**  
  Many kernels still launched via `cudaLaunchKernel` (e.g. cuDNN, NMS, custom CUDA). Only the traceable forward path can be turned into `cuLaunchKernel`-backed kernels; the rest stays on Runtime API.

- **cudaLaunchKernel “not also JIT”**  
  JIT (Inductor/Triton) emits PTX/cubin and uses the Driver API (`cuLaunchKernel`). Legacy and third-party code (cuDNN, custom ops) use the Runtime API (`cudaLaunchKernel`). So “why doesn’t cudaLaunchKernel also go JIT?” → those call sites are not part of the compiled graph; they remain pre-compiled libraries or custom kernels.

- **Variance**  
  Forward time variance (e.g. StdDev) may still be high; worth checking warmup and fixed input size when comparing.

---

## cuLaunchKernel vs cudaLaunchKernel (Summary)

| | cudaLaunchKernel | cuLaunchKernel |
|-|------------------|----------------|
| **API** | CUDA Runtime | CUDA Driver |
| **Typical use** | Pre-built libraries (ATen, cuDNN), custom `.cu` | JIT-compiled modules (Inductor/Triton) |
| **In this profile** | Most kernels in baseline; many in compiled | Dominant “extra” launches in compiled run |

---

## Optimization Directions and Expected Outcomes

1. **Increase fusion**  
   - Reduce small elementwise (e.g. clamp, fill) as separate kernels.  
   - **Expected**: Fewer `cudaLaunchKernel`/kernel instances, lower launch overhead, possibly better forward time.

2. **Keep post_processing and NMS out of “compiled” metric**  
   - NMS and bbox/post logic are not in the traced `.pt`; they stay eager.  
   - **Expected**: NVTX “forward” reflects only the traced backbone; post_processing time stays similar unless optimized separately.

3. **Stabilize and re-profile**  
   - Fixed batch size, more warmup, same data.  
   - **Expected**: Clearer comparison and variance explanation (e.g. cache, first-run JIT).

4. **Optional: try `torch.compile` again**  
   - With graph-break fixes and `--compile_debug` to see recompiles/breaks.  
   - **Expected**: Either more kernels via `cuLaunchKernel` and faster forward, or clear evidence of graph breaks limiting gain.

---

## How to Re-run and Regenerate Stats

- Baseline: run `test.py` without `--traced_model` and without `--compile`; use `--nsight --nsight_steps N --warmup N`.  
- Compiled: run `test.py` with `--traced_model path/to/pointpillar_traced_compiled.pt` and same `--nsight*` options.  
- Export: `nsys export --type sqlite report.nsys-rep` then `nsys stats report.nsys-rep > nsight_stats_*.txt`.
