# 3D Perception Pipeline Optimization

---

## Run `modal_mls_app.py`

Run from repo root:

Default behavior in `modal_mls_app.py`:
- GPU: `A10` (override with `MLS_MODAL_RESEARCH_GPU`)
- Output root on Modal volume: `/mnt/results/modal_v3_a10`
- Auto download to local: `modal_outputs/modal_v3_a10`
- Extra args auto-enabled: `--nsight --nsys-profile` (for `forward_nvtx_ms.json` + `.nsys-rep`)

### Run all cells

```bash
MLS_MODAL_RESEARCH_GPU=A10 modal run -d modal_mls_app.py --action run --matrix fp32_amp --all-cells --download-to modal_outputs/modal_v3_a10
```

### Run one cell

```bash
# m: 0..4, p: 1=FP32, 2=AMP
MLS_MODAL_RESEARCH_GPU=A10 modal run -d modal_mls_app.py --action run --matrix fp32_amp --model-m 1 --precision-p 2 --download-to modal_outputs/modal_v3_a10
```

---

## Plot Modal Outputs

Run from repo root (`/home/nas/polin/cmu-berlin/MLS`):

### `report/plot_accuracy_latency_energy_3d.py`

```bash
python3 report/plot_accuracy_latency_energy_3d.py --a10-root modal_outputs/modal_v2_a10 --h100-root modal_outputs/modal_v2_h100 --t4-root modal_outputs/modal_v2_t4
```

### `report/plot_latency.py`

```bash
python3 report/plot_latency.py --csv modal_outputs/modal_v2_a10 --gpu A10 --nest-forward-from-artifacts
python3 report/plot_latency.py --csv modal_outputs/modal_v2_h100 --gpu H100 --nest-forward-from-artifacts
python3 report/plot_latency.py --csv modal_outputs/modal_v2_t4 --gpu T4 --nest-forward-from-artifacts
```

### `report/plot_energy.py`

```bash
python3 report/plot_energy.py --csv modal_outputs/modal_v2_a10 --gpu A10 --nest-forward-from-artifacts --forward-nvtx-root modal_outputs/modal_v2_a10
python3 report/plot_energy.py --csv modal_outputs/modal_v2_h100 --gpu H100 --nest-forward-from-artifacts --forward-nvtx-root modal_outputs/modal_v2_h100
python3 report/plot_energy.py --csv modal_outputs/modal_v2_t4 --gpu T4 --nest-forward-from-artifacts --forward-nvtx-root modal_outputs/modal_v2_t4
```

### `report/plot_latency_energy_pareto.py`

```bash
python3 report/plot_latency_energy_pareto.py --csv modal_outputs/modal_v2_a10 --gpu A10
python3 report/plot_latency_energy_pareto.py --csv modal_outputs/modal_v2_h100 --gpu H100
python3 report/plot_latency_energy_pareto.py --csv modal_outputs/modal_v2_t4 --gpu T4
```

### `report/plot_roofline_forward.py`

```bash
python3 report/plot_roofline_forward.py --csv modal_outputs/modal_v2_a10 --gpu A10
python3 report/plot_roofline_forward.py --csv modal_outputs/modal_v2_h100 --gpu H100
python3 report/plot_roofline_forward.py --csv modal_outputs/modal_v2_t4 --gpu T4
```

---


---

## Test pipeline vs real-time onboard inference

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
