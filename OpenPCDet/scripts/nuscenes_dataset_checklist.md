# NuScenes Dataset Checklist for BEVFusion Inference

## Current Status (MSCV-Capstone/OpenPCDet/data/nuscenes)

| Item | Status | Note |
|------|--------|------|
| **Raw data layout** | OK | `v1.0-trainval/` with `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/` (metadata JSONs) |
| **Metadata JSONs** | OK | `scene.json`, `sample.json`, `sample_annotation.json`, etc. under `v1.0-trainval/v1.0-trainval/` |
| **Info pkl files** | **MISSING** | Must be generated (see below) |
| **Data path for MLS script** | **MISMATCH** | Inference runs from **MLS**/OpenPCDet; config expects **MLS**/OpenPCDet/data/nuscenes. Your data is under **MSCV-Capstone**/OpenPCDet/data/nuscenes. |

---

## What You Are Missing

### 1. Generate info pkl files (required for test.py)

Config expects these under `DATA_PATH/v1.0-trainval/` (i.e. `../data/nuscenes/v1.0-trainval/` when run from `tools/`):

- **`nuscenes_infos_10sweeps_val.pkl`** — required for **inference/eval**
- `nuscenes_infos_10sweeps_train.pkl` — for training only
- `nuscenes_dbinfos_10sweeps_withvelo.pkl` — for training with gt_sampling only

**How to generate** (run from the OpenPCDet repo that **contains** your data, e.g. MSCV-Capstone):

```bash
cd /home/nas/polin/cmu-berlin/MSCV-Capstone/OpenPCDet/tools

# Multi-modal (with camera) — required for BEVFusion
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval \
    --with_cam
```

This writes `nuscenes_infos_10sweeps_train.pkl` and `nuscenes_infos_10sweeps_val.pkl` under  
`MSCV-Capstone/OpenPCDet/data/nuscenes/v1.0-trainval/`.

(Optional, for training only) Then create the ground-truth database:

- The same script’s default flow also calls `create_groundtruth_database`, which produces `nuscenes_dbinfos_10sweeps_withvelo.pkl`. If you only need inference, the two infos pkl above are enough.

### 2. Point MLS inference to your data (DATA_PATH)

Inference script runs **MLS**/OpenPCDet/tools, so `DATA_PATH: '../data/nuscenes'` resolves to **MLS**/OpenPCDet/data/nuscenes. Your data and pkl are under **MSCV-Capstone**/OpenPCDet/data/nuscenes.

**Option A — Symlink (recommended):**

```bash
mkdir -p /home/nas/polin/cmu-berlin/MLS/OpenPCDet/data
ln -snf /home/nas/polin/cmu-berlin/MSCV-Capstone/OpenPCDet/data/nuscenes /home/nas/polin/cmu-berlin/MLS/OpenPCDet/data/nuscenes
```

**Option B — Override in the inference script:**  
Add `--set DATA_CONFIG.DATA_PATH /home/nas/polin/cmu-berlin/MSCV-Capstone/OpenPCDet/data/nuscenes` (or the correct absolute path) when calling `test.py` so the dataset loads from MSCV-Capstone.

---

## Expected directory structure (after preparation)

```
data/nuscenes/
└── v1.0-trainval/
    ├── samples/           # lidar + camera frames
    ├── sweeps/
    ├── maps/
    ├── v1.0-trainval/     # metadata *.json
    ├── nuscenes_infos_10sweeps_train.pkl   # generated
    └── nuscenes_infos_10sweeps_val.pkl    # generated (needed for inference)
```

---

## Quick check commands

```bash
# 1. Check pkl exists (after generation)
ls -la /home/nas/polin/cmu-berlin/MSCV-Capstone/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl

# 2. If using symlink, check MLS sees the same data
ls -la /home/nas/polin/cmu-berlin/MLS/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl
```
