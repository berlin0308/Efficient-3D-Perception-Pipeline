# PointPillar quick start scripts

Run in order (from repo root or from this directory):

| Step | Script | Description |
|------|--------|-------------|
| 1 | `./01_install.sh` | Install pcdet (needs PyTorch + `pip install -e . --no-build-isolation`) |
| 2 | `./02_download_kitti.sh` | Print KITTI download instructions; optional: set `KITTI_ZIP_DIR` to auto-unpack zips |
| 3 | `./03_create_kitti_infos.sh` | Generate `kitti_infos_*.pkl` and gt database (requires **full** KITTI: training/{velodyne,calib,label_2,image_2}) |
| 4 | `./04_download_pointpillar_ckpt.sh` | Download pretrained PointPillar (needs `pip install gdown`) |
| 5 | `./05_demo.sh` | Run demo on one frame or a folder of `.bin` / `.npy` (default: no GUI, use `NO_VIZ=0` for display) |
| 6 | `./06_test.sh` | Run evaluation on KITTI val set (requires 03 done and `kitti_infos_val.pkl`) |

**Minimal run (demo only, no full KITTI):** 01 → 04 → 05. One sample `000008.bin` is under `data/kitti/training/velodyne/` for a quick demo.

**Getting non-zero mAP on 06_test:** You need **label_2** and **calib** (and velodyne) for val samples. We unpacked `data_object_label_2.zip` and `data_object_calib.zip` from AWS (`scripts/pointpillar_quickstart/download_kitti_aws.sh`). With only one velodyne file (e.g. 000008.bin), 06 runs but AP can be 0 on that single frame. For meaningful mAP, download and unpack full `data_object_velodyne.zip` so val has many samples, then re-run 03 and 06.

**Optional env vars**

- `05_demo.sh`: `CKPT`, `DATA_PATH`, `EXT`, `NO_VIZ` (default `NO_VIZ=1` skips GUI for headless; set `NO_VIZ=0` to show Open3D window)
- `06_test.sh`: `CKPT`, `BATCH_SIZE`
- `04_*` / `05_*` / `06_*`: `CKPT_DIR` (default: `tools/ckpt`)

**Make executable (once):**

```bash
chmod +x scripts/pointpillar_quickstart/*.sh
```
