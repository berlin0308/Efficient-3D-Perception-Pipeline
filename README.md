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
