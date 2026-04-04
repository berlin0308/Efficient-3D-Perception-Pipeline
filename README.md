# 3D Perception Pipeline Optimization

---

## 0. Profile PointPillars (Baseline)

**Script:** `OpenPCDet/scripts/profile_pointpillar_baseline.sh`

**Results** (`profile_outputs/pointpillars/openpcdet/`):

| Output | Path | How to view |
|--------|------|-------------|
| Torch Profiler (Chrome trace) | [torch_profile_trace.json](profile_outputs/pointpillars/openpcdet/torch_profile_trace.json) | Open in Chrome: `chrome://tracing/` → Load |
| TorchScript model | [pointpillar_traced.pt](profile_outputs/pointpillars/openpcdet/pointpillar_traced.pt) | [Netron](https://netron.app) |
| Nsight Systems report | `profile_outputs/pointpillars/openpcdet/nsight_report_*_baseline.nsys-rep` | Nsight Systems GUI (open `.nsys-rep` file) |
| Nsight stats | [nsight_stats_20260226_123607_baseline.txt](profile_outputs/pointpillars/openpcdet/nsight_stats_20260226_123607_baseline.txt) | Text summary from `nsys stats` |

---

## 1. Profile PointPillars (Compiled)

**Script:** `OpenPCDet/scripts/profile_pointpillar_compiled.sh`

**Results:**

| Output | Path | How to view |
|--------|------|-------------|
| TorchScript (compiled) | [pointpillar_traced_compiled.pt](profile_outputs/pointpillars/openpcdet/pointpillar_traced_compiled.pt) | [Netron](https://netron.app) |
| Nsight Systems report | `profile_outputs/pointpillars/openpcdet_compiled/nsight_report_*_compiled.nsys-rep` | Nsight Systems GUI |
| Nsight stats | [nsight_stats_20260226_123310_compiled.txt](profile_outputs/pointpillars/openpcdet_compiled/nsight_stats_20260226_123310_compiled.txt) | Text summary from `nsys stats` |

**Comparison (Baseline vs. Compiled):** [profile_outputs/nsight_comparison_plan.md](profile_outputs/nsight_comparison_plan.md)

---

## 2. Efficiency Runtime Switches (Energy + Memory)

The inference/eval/profiling scripts now support `--amp` (FP16 autocast) and run with `torch.inference_mode()` for lower runtime overhead.

Also, `load_data_to_gpu` now preserves integer/bool numpy dtypes during H2D copy (instead of forcing everything to float), reducing unnecessary memory expansion in the end-to-end path.

Examples:

- Offline eval: `python OpenPCDet/tools/test.py ... --amp`
- Real-time path: `python OpenPCDet/tools/inference.py ... --amp`
- Profile suite: `python OpenPCDet/tools/profile_suite.py ... --amp`
- Energy monitor: `python OpenPCDet/tools/energy_monitor.py ... --amp`
- Memory trace: `python OpenPCDet/tools/memory_trace.py ... --amp`

---

## Test pipeline vs real-time onboard inference (discrepancies)

The current eval pipeline (`OpenPCDet/tools/test.py` + `eval_utils.py`) is **offline**: load dataset from disk → CPU preprocessing (DataLoader workers) → CPU→GPU → inference → GPU→CPU → evaluation. This differs from **real-time onboard** inference in the following ways; keep these in mind when comparing latency or designing a deployment path.

| Aspect | Current (test.py) | Real-time onboard |
|--------|-------------------|-------------------|
| **Lidar rate** | No fixed Hz; run as fast as possible (throughput) | Fixed period (e.g. **10 Hz** = 100 ms/frame); inference must fit within budget |
| **Data source** | Disk (`velodyne/*.bin`) | Live lidar stream (no disk I/O) |
| **DataLoader** | Multi-worker prefetch from disk, CPU preprocessing in workers | Typically single-frame, sensor-driven; no DataLoader |
| **Preprocessing** | CPU in dataset workers (voxelization etc.) | Often GPU or overlapped with previous frame |
| **Metric** | Throughput, sec_per_example, optional infer_time (forward only) | **End-to-end latency** (scan ready → detections ready); cold start matters |
| **Batch** | Configurable batch_size | Effectively batch=1 per scan |
| **Ordering** | Dataset index or shuffle | Strict time order |
| **GPU→CPU** | Full results for evaluation (recall, mAP) | Minimal copy for planning/control |

See plan: `test_vs_real-time_inference_discrepancies` for full detail.
