"""
modal_cuda_pp_app.py — Run CUDA-PointPillars (TensorRT FP32 & FP16) on Modal.

## Quick start

1) Dry run — just build the C++ binary, no benchmark:
     modal run modal_cuda_pp_app.py --action build

2) Run FP32 benchmark:
     modal run modal_cuda_pp_app.py --action run --precision fp32

3) Run FP16 benchmark:
     modal run modal_cuda_pp_app.py --action run --precision fp16

4) Run both FP32 and FP16 in sequence:
     modal run modal_cuda_pp_app.py --action run_all

Results are saved to Modal volume and auto-downloaded to ./modal_outputs/cuda_pp/

## Switching to teammate's workspace
When you have access to the teammate's Modal workspace, just change RESULTS_VOLUME_NAME
to match their volume name, and you're done.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Paths (relative to this file, which lives at repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CUDA_PP_ROOT = REPO_ROOT / "CUDA-PointPillars"

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
RESULTS_VOLUME_NAME = "mls-openpcdet-results"
KITTI_VOLUME_NAME   = "mls-openpcdet-kitti"

results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
kitti_volume   = modal.Volume.from_name(KITTI_VOLUME_NAME,   create_if_missing=False)

MNT_RESULTS = "/mnt/results"
MNT_KITTI   = "/mnt/kitti"

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App("cuda-pointpillars")

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# Base: nvcr.io/nvidia/tensorrt:23.05-py3
# Already includes: TRT 8.6.1, CUDA 12.1, cuDNN 8.9, cmake, python3, gcc/g++/make
# No need to install anything — bypasses the Modal network restriction on archive.ubuntu.com
cuda_pp_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt:23.05-py3",
    )
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
    .pip_install("pynvml", "numpy", "numba", "fire", "scikit-image", "pillow", "matplotlib",
                 "opencv-python-headless")
    # Copy CUDA-PointPillars source (excluding build artifacts and cache)
    .add_local_dir(
        str(CUDA_PP_ROOT),
        "/opt/cuda-pp",
        copy=True,
        ignore=[
            "build/**",
            "**/*.cache",
        ],
    )
    # Copy the metrics wrapper
    .add_local_file(
        str(REPO_ROOT / "cuda_pp_metrics.py"),
        "/opt/cuda_pp_metrics.py",
    )
)


# ---------------------------------------------------------------------------
# Helper: download results volume -> local
# ---------------------------------------------------------------------------
def _download_results(local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="modal_vol_get_") as td:
        stage = Path(td)
        subprocess.run(
            ["modal", "volume", "get", RESULTS_VOLUME_NAME, "/", str(stage), "--force"],
            check=True,
        )
        # Merge into local_dir
        for item in stage.iterdir():
            target = local_dir / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))
    print(f"[local] results downloaded → {local_dir}", flush=True)


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

def _cuda_pp_benchmark_body(precision: str, warmup: int, steps: int, rebuild: bool) -> dict:
    """Shared benchmark logic — called by each per-GPU Modal function."""
    import subprocess as sp
    import sys

    src_dir    = "/opt/cuda-pp"
    build_dir  = f"/opt/cuda-pp/build-{precision}"
    output_csv = f"{MNT_RESULTS}/cuda_pp/M5_{precision.upper()}.csv"

    print(f"[modal] precision={precision}  warmup={warmup}  steps={steps}", flush=True)

    argv = [
        sys.executable, "/opt/cuda_pp_metrics.py",
        "--src-dir",      src_dir,
        "--build-dir",    build_dir,
        "--data-dir",     f"{MNT_KITTI}/training/velodyne",
        "--model-dir",    f"{src_dir}/model",
        "--precision",    precision,
        "--warmup",       str(warmup),
        "--steps",        str(steps),
        "--output-csv",   output_csv,
        "--cuda-toolkit", "/usr/local/cuda-12.1",
    ]
    if rebuild:
        argv.append("--rebuild")

    result = sp.run(argv, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cuda_pp_metrics.py failed (rc={result.returncode})")

    results_volume.commit()
    print(f"[modal] results committed → {output_csv}", flush=True)
    return {"status": "ok", "precision": precision, "output_csv": output_csv}


@app.function(
    image=cuda_pp_image, gpu="A10G",
    volumes={MNT_RESULTS: results_volume, MNT_KITTI: kitti_volume},
    timeout=3600, cpu=4,
)
def run_cuda_pp(precision="fp32", warmup=500, steps=500, rebuild=True) -> dict:
    return _cuda_pp_benchmark_body(precision, warmup, steps, rebuild)


@app.function(
    image=cuda_pp_image, gpu="T4",
    volumes={MNT_RESULTS: results_volume, MNT_KITTI: kitti_volume},
    timeout=3600, cpu=4,
)
def run_cuda_pp_t4(precision="fp32", warmup=500, steps=500, rebuild=True) -> dict:
    return _cuda_pp_benchmark_body(precision, warmup, steps, rebuild)


@app.function(
    image=cuda_pp_image, gpu="H100",
    volumes={MNT_RESULTS: results_volume, MNT_KITTI: kitti_volume},
    timeout=3600, cpu=4,
)
def run_cuda_pp_h100(precision="fp32", warmup=500, steps=500, rebuild=True) -> dict:
    return _cuda_pp_benchmark_body(precision, warmup, steps, rebuild)


@app.function(
    image=cuda_pp_image,
    gpu="A10G",
    volumes={MNT_RESULTS: results_volume, MNT_KITTI: kitti_volume},
    timeout=7200,
    cpu=4,
)
def run_accuracy(
    precision: str = "fp32",
    rebuild: bool = True,
) -> dict:
    """
    Run CUDA-PointPillars on full KITTI val set and save predictions for mAP eval.
    Uses mls-openpcdet-kitti volume for velodyne .bin files.
    precision: "fp32" or "fp16"
    """
    import subprocess as sp
    import sys
    import os

    src_dir   = "/opt/cuda-pp"
    build_dir = f"/opt/cuda-pp/build-{precision}"
    data_dir  = f"{MNT_KITTI}/training/velodyne"
    save_dir  = f"{MNT_RESULTS}/cuda_pp/preds_{precision}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"[modal] accuracy eval  precision={precision}  data_dir={data_dir}", flush=True)

    argv = [
        sys.executable, "/opt/cuda_pp_metrics.py",
        "--src-dir",    src_dir,
        "--build-dir",  build_dir,
        "--data-dir",   data_dir,
        "--model-dir",  f"{src_dir}/model",
        "--precision",  precision,
        "--warmup",     "0",
        "--steps",      "7481",  # all KITTI training frames (val set is subset)
        "--output-csv", f"{MNT_RESULTS}/cuda_pp/M5_{precision.upper()}_accuracy.csv",
        "--cuda-toolkit", "/usr/local/cuda-12.1",
        "--save-preds",
        "--save-dir",   save_dir,
    ]
    if rebuild:
        argv.append("--rebuild")

    result = sp.run(argv, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cuda_pp_metrics.py failed (rc={result.returncode})")

    results_volume.commit()
    print(f"[modal] predictions saved → {save_dir}", flush=True)
    return {"status": "ok", "precision": precision, "preds_dir": save_dir}


@app.function(
    image=cuda_pp_image,
    gpu="T4",
    cpu=4,
    volumes={MNT_RESULTS: results_volume, MNT_KITTI: kitti_volume},
    timeout=3600,
)
def run_kitti_eval(precision: str = "fp32") -> dict:
    """
    Convert CUDA-PP lidar predictions → KITTI camera format, then run evaluate.py.
    Reads pred_velo_{precision}/ from results volume.
    Writes pred_cam_{precision}/ and kitti_eval_{precision}.txt to results volume.
    """
    import subprocess as sp
    import sys
    import os
    import shutil
    from pathlib import Path

    eval_dir    = Path("/opt/cuda-pp/eval")
    pred_velo   = Path(f"{MNT_RESULTS}/cuda_pp/pred_velo_{precision}")
    pred_cam    = Path(f"{MNT_RESULTS}/cuda_pp/pred_cam_{precision}")
    pred_velo.mkdir(parents=True, exist_ok=True)
    pred_cam.mkdir(parents=True, exist_ok=True)

    calib_dir  = Path(f"{MNT_KITTI}/training/calib")
    image_dir  = Path(f"{MNT_KITTI}/training/image_2")
    label_dir  = Path(f"{MNT_KITTI}/training/label_2")

    # Rename preds_fp32000000.txt → pred_velo_fp32/000000.txt (if not done yet)
    prefix = f"preds_{precision}"
    raw_preds = list(Path(f"{MNT_RESULTS}/cuda_pp").glob(f"{prefix}*.txt"))
    if raw_preds:
        print(f"[eval] renaming {len(raw_preds)} raw pred files...", flush=True)
        for f in raw_preds:
            stem = f.stem[len(prefix):]   # "preds_fp32000000" → "000000"
            f.rename(pred_velo / f"{stem}.txt")

    print(f"[eval] precision={precision}  pred_velo={pred_velo}", flush=True)
    n = len(list(pred_velo.glob("*.txt")))
    print(f"[eval] found {n} prediction files", flush=True)

    kitti_obj = eval_dir / "kitti" / "object"

    # Skip kitti_format.py if pred_cam already has results
    n_pred_velo = len(list(pred_velo.glob("*.txt")))
    n_pred_cam  = len(list(pred_cam.glob("*.txt")))
    if n_pred_cam >= n_pred_velo > 0:
        print(f"[eval] pred_cam already has {n_pred_cam} files, skipping kitti_format.py", flush=True)
    else:
        # Set up directory structure expected by kitti_format.py
        (kitti_obj / "training").mkdir(parents=True, exist_ok=True)
        (kitti_obj / "pred_velo").mkdir(parents=True, exist_ok=True)
        (kitti_obj / "pred").mkdir(parents=True, exist_ok=True)

        for d, src in [
            (kitti_obj / "training" / "calib",   calib_dir),
            (kitti_obj / "training" / "image_2", image_dir),
            (kitti_obj / "training" / "label_2", label_dir),
        ]:
            if not d.exists():
                os.symlink(src, d)

        for f in pred_velo.glob("*.txt"):
            shutil.copy2(f, kitti_obj / "pred_velo" / f.name)

        print("[eval] running kitti_format.py ...", flush=True)
        sp.run([sys.executable, "kitti_format.py"], cwd=str(eval_dir), check=True)

        for f in (kitti_obj / "pred").glob("*.txt"):
            shutil.copy2(f, pred_cam / f.name)

    # evaluate.py reads pred from pred_cam (copied back into kitti_obj/pred)
    (kitti_obj / "pred").mkdir(parents=True, exist_ok=True)
    for f in pred_cam.glob("*.txt"):
        shutil.copy2(f, kitti_obj / "pred" / f.name)

    # Set up symlinks for evaluate.py (label_2 needed)
    (kitti_obj / "training").mkdir(parents=True, exist_ok=True)
    for d, src in [
        (kitti_obj / "training" / "label_2", label_dir),
    ]:
        if not d.exists():
            os.symlink(src, d)

    # Run evaluate.py
    eval_py   = eval_dir / "kitti-object-eval-python" / "evaluate.py"
    label_path = kitti_obj / "training" / "label_2"
    result_path = kitti_obj / "pred"
    val_txt     = eval_dir / "val.txt"
    out_txt     = Path(f"{MNT_RESULTS}/cuda_pp/kitti_eval_{precision}.txt")

    print("[eval] running evaluate.py ...", flush=True)
    res = sp.run(
        [
            sys.executable, str(eval_py), "evaluate",
            f"--label_path={label_path}",
            f"--result_path={result_path}",
            f"--label_split_file={val_txt}",
            "--current_class=0",
            "--coco=False",
        ],
        capture_output=True, text=True,
    )
    output = res.stdout + res.stderr
    print(output, flush=True)
    out_txt.write_text(output)

    results_volume.commit()
    print(f"[eval] done → {out_txt}", flush=True)
    return {"status": "ok", "precision": precision, "eval_output": out_txt.name}


@app.function(
    image=cuda_pp_image,
    gpu="A10G",
    timeout=600,
)
def build_only() -> dict:
    """Dry run: just cmake + make to verify the build works."""
    import subprocess as sp
    import os

    build_dir = "/opt/cuda-pp/build-fp32"
    os.makedirs(build_dir, exist_ok=True)

    print("[build] cmake...", flush=True)
    sp.run(["cmake", "/opt/cuda-pp", "-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1"],
           cwd=build_dir, check=True)

    print("[build] make...", flush=True)
    sp.run(["make", f"-j{os.cpu_count() or 4}"], cwd=build_dir, check=True)

    demo = f"{build_dir}/demo"
    exists = os.path.exists(demo)
    print(f"[build] binary exists: {exists}  path: {demo}", flush=True)
    return {"status": "ok" if exists else "failed", "binary": demo}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

# Modal GPU name → display label
_GPU_LABELS = {
    "T4":   "t4",
    "A10G": "a10g",
    "H100": "h100",
}
_ALL_GPUS = list(_GPU_LABELS.keys())


@app.local_entrypoint()
def main(
    action: str = "run",
    precision: str = "fp32",
    warmup: int = 5,
    steps: int = 10,
    gpu: str = "A10G",
    download_to: str = "./modal_outputs/cuda_pp",
):
    """
    action:    build | run | run_all
    precision: fp32 | fp16           (used when action=run/run_all)
    gpu:       T4 | A10G | H100 | all
    """
    gpu_list = _ALL_GPUS if gpu.lower() == "all" else [gpu.upper()]

    if action == "build":
        print("=== Dry run: build only ===")
        result = build_only.remote()
        print(f"Result: {result}")

    elif action == "run":
        _gpu_fn = {"A10G": run_cuda_pp, "T4": run_cuda_pp_t4, "H100": run_cuda_pp_h100}
        for g in gpu_list:
            fn = _gpu_fn[g]
            print(f"=== Running M5_{precision.upper()} on {g} ===")
            result = fn.remote(precision=precision, warmup=warmup, steps=steps, rebuild=True)
            print(f"Result: {result}")
        print(f"\nDownloading results to {download_to}...")
        _download_results(Path(download_to))

    elif action == "run_all":
        _gpu_fn = {"A10G": run_cuda_pp, "T4": run_cuda_pp_t4, "H100": run_cuda_pp_h100}
        for g in gpu_list:
            fn = _gpu_fn[g]
            print(f"\n=== GPU: {g} ===")
            for p in ["fp32", "fp16"]:
                print(f"\n--- {p.upper()} ---")
                result = fn.remote(precision=p, warmup=warmup, steps=steps, rebuild=True)
                print(f"Result: {result}")
        print(f"\nDownloading results to {download_to}...")
        _download_results(Path(download_to))

    elif action == "accuracy":
        print(f"=== Accuracy eval M5_{precision.upper()} on KITTI val set ===")
        result = run_accuracy.remote(precision=precision, rebuild=True)
        print(f"Result: {result}")
        print(f"\nDownloading results to {download_to}...")
        _download_results(Path(download_to))

    elif action == "eval":
        print(f"=== KITTI eval M5_{precision.upper()} ===")
        result = run_kitti_eval.remote(precision=precision)
        print(f"Result: {result}")
        print(f"\nDownloading results to {download_to}...")
        _download_results(Path(download_to))

    else:
        print(f"Unknown action: {action}. Use: build | run | run_all | accuracy")
