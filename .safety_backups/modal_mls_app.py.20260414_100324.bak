"""
Modal image matching local conda env `mls` (see OpenPCDet/INSTALL_MLS.md and create_mls_env.sh).

## Reproduce research metrics on Modal (real KITTI)

1) One-time KITTI on the volume (pick one):

   **A — Download on Modal (recommended; same source as OpenPCDet quickstart `download_kitti_aws.sh`):**
   Public AWS mirror `s3://avg-kitti` — no KITTI website login. Fetches the four object-detection zips,
   unpacks into volume `mls-openpcdet-kitti`, then runs `create_kitti_infos` (see GETTING_STARTED.md).

     modal run modal_mls_app.py --action download_kitti

   Large download (~40GB+ zips); first run can take hours. Re-runs skip if `kitti_infos_val.pkl` exists
   unless `--force-kitti-download true`.

   **B — Upload from your laptop** (slow on large datasets):

     modal run modal_mls_app.py --action upload_kitti --kitti-path /abs/path/to/OpenPCDet/data/kitti

     modal volume put --force mls-openpcdet-kitti /abs/path/to/data/kitti /

2) Upload checkpoints (same as local `tools/ckpt/`):

     modal run modal_mls_app.py --action upload
     # or: modal volume put --force mls-openpcdet-ckpts ... (see below)

3) Run (same knobs as collect_research_metrics.py `run`; CSV under volume `mls-openpcdet-results`):

   **KITTI eval sample cap (per cell):** by default Modal passes `--max_eval_samples 200` to
   `collect_research_metrics` (first 200 val samples for AP, same as local `--accuracy-false-samples 200`).
   Full val split: add `--kitti-full-val`. Override count: `--kitti-max-eval-samples N`.

   - All **runnable** fp32_amp cells (default):

       modal run modal_mls_app.py --action run --matrix fp32_amp --warmup 100 --steps 50

   - One cell → single CSV (aligned with --cell / --output-csv):

       modal run modal_mls_app.py --action run --cell M0_FP32 \\
         --output-csv /mnt/results/research_matrix/cell_runs/M0_FP32.csv

   - Shorthand --m / --p:

       modal run modal_mls_app.py --action run --model-m 1 --precision-p 1 \\
         --output-csv /mnt/results/research_matrix/cell_runs/M1_FP32.csv

   - All design cells (incl. placeholders): --all-cells true

   Successful runs **auto-download** the whole `--output-root` tree to `./modal_mls_results` (override with `--download-to`).

   **Logs:** `run_collect_research_metrics` streams `collect_research_metrics.py` stdout to the Modal function log
   (Dashboard → run → Logs) in real time. **GPU:** defaults to Modal's `A10` (`MLS_MODAL_RESEARCH_GPU` to override).
   Modal's `gpu=` argument uses `A10` (not `A10G`); `nvidia-smi` may still report **NVIDIA A10G** on the instance.

4) Manual download (if you used --skip-download):

     modal volume get mls-openpcdet-results /research_matrix ./out --force

**First-time Modal image:** building the image runs `python setup.py develop` (CUDA extension compile) and can take
many minutes. Keep the machine online until the build finishes; if the client disconnects mid-build you may see
`Image build ... terminated due to external shut-down`. After the image exists, re-runs reuse cached layers.
Validate locally first: `conda activate mls` then `bash OpenPCDet/tools/run_m4_amp_mls.sh` (see INSTALL_MLS.md).

By default `--action run` uses **real KITTI** (`require_real_kitti=True`). For a tiny synthetic smoke test only:

     modal run modal_mls_app.py --action run --require-real-kitti false --warmup 1 --steps 1 ...

## Checkpoints (volume mls-openpcdet-ckpts)

  - tools/ckpt/pointpillar_7728.pth, pointpillar_traced.pt, pointpillar_traced_compiled.pt

  modal run modal_mls_app.py --action upload

  modal volume put mls-openpcdet-ckpts /path/to/pointpillar_7728.pth pointpillar_7728.pth
  ...

Optional HTTPS inside the container (Secrets / env):

  MLS_URL_POINTPILLAR_7728_PTH
  MLS_URL_POINTPILLAR_TRACED_PT
  MLS_URL_POINTPILLAR_TRACED_COMPILED_PT
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import sys
import urllib.request
from pathlib import Path

import modal


def _modal_cli_argv() -> list[str]:
    if shutil.which("modal"):
        return ["modal"]
    return [sys.executable, "-m", "modal"]


# OpenPCDet tree (this file lives in MLS/)
OPENPCDET_ROOT = (Path(__file__).resolve().parent / "OpenPCDet").resolve()

# Volume stores flat filenames; we copy into the OpenPCDet layout inside the container.
CKPT_VOLUME_NAME = "mls-openpcdet-ckpts"
# Full OpenPCDet KITTI root (directory containing kitti_infos_val.pkl); mounted at /mnt/kitti.
KITTI_VOLUME_NAME = "mls-openpcdet-kitti"
# Persists runs.csv, profile/, energy/ trees from collect_research_metrics (default --output_root).
RESULTS_VOLUME_NAME = "mls-openpcdet-results"

# (filename on volume) -> list of absolute destinations under /opt/OpenPCDet
ARTIFACT_LAYOUT: dict[str, list[str]] = {
    "pointpillar_7728.pth": [
        "/opt/OpenPCDet/tools/ckpt/pointpillar_7728.pth",
        "/opt/OpenPCDet/ckpt/pointpillar_7728.pth",
    ],
    "pointpillar_traced.pt": [
        "/opt/OpenPCDet/tools/pointpillar_traced.pt",
    ],
    "pointpillar_traced_compiled.pt": [
        "/opt/OpenPCDet/tools/pointpillar_traced_compiled.pt",
    ],
}

URL_ENV_FOR_ARTIFACT: dict[str, str] = {
    "pointpillar_7728.pth": "MLS_URL_POINTPILLAR_7728_PTH",
    "pointpillar_traced.pt": "MLS_URL_POINTPILLAR_TRACED_PT",
    "pointpillar_traced_compiled.pt": "MLS_URL_POINTPILLAR_TRACED_COMPILED_PT",
}

app = modal.App("mls-openpcdet")

ckpt_volume = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)
kitti_volume = modal.Volume.from_name(KITTI_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

MNT_RESULTS = "/mnt/results"

# GPU for profile / energy / KITTI eval (Modal docs: gpu="A10", "T4", "H100", … — not "A10G").
RESEARCH_GPU = os.environ.get("MLS_MODAL_RESEARCH_GPU", "A10")


def _volume_prefix_under_results(container_path: str) -> str:
    """Path under the results volume mount (strip /mnt/results/). Used for modal volume get."""
    p = Path(container_path).as_posix().rstrip("/")
    if p == MNT_RESULTS:
        return ""
    if p.startswith(MNT_RESULTS + "/"):
        return p[len(MNT_RESULTS) + 1 :].lstrip("/")
    return "research_matrix"


def _merge_download_tree(src: Path, dst: Path) -> None:
    """Move src children into dst, replacing same-named files/dirs (avoids modal volume get ENOTEMPTY)."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))


def _download_results_tree_local(volume_prefix: str, local_dir: Path) -> None:
    local_dir = local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote = "/" + volume_prefix if volume_prefix else "/"
    target = local_dir / volume_prefix if volume_prefix else local_dir
    with tempfile.TemporaryDirectory(prefix="modal_vol_get_") as td:
        stage = Path(td)
        subprocess.run(
            _modal_cli_argv() + ["volume", "get", RESULTS_VOLUME_NAME, remote, str(stage), "--force"],
            check=True,
        )
        inner = stage / volume_prefix if volume_prefix else stage
        if volume_prefix and not inner.is_dir():
            inner = stage
        if not inner.is_dir():
            raise RuntimeError(
                "Unexpected layout after modal volume get %r -> %s (expected a directory)"
                % (remote, stage)
            )
        _merge_download_tree(inner, target)


def _stream_subprocess_to_logs(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    banner: str = "[mls]",
) -> tuple[int, str]:
    """
    Run a subprocess and copy merged stdout/stderr to this process stdout line-by-line
    so Modal's function log stream updates during long runs (not only at the end).
    """
    merged = dict(env)
    merged.setdefault("PYTHONUNBUFFERED", "1")
    print(f"{banner} subprocess cwd={cwd}", flush=True)
    print(f"{banner} subprocess {shlex.join(str(x) for x in argv)}", flush=True)
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            chunks.append(line)
            print(line, end="", flush=True)
    rc = proc.wait()
    print(f"{banner} subprocess finished returncode={rc}", flush=True)
    return rc, "".join(chunks)


def _argv_collect_research_run(
    *,
    tools_dir: Path,
    cuda_id: int,
    warmup: int,
    steps: int,
    batch_size: int,
    workers: int,
    output_root: str,
    matrix: str,
    cell: str | None,
    all_cells_15: bool,
    model_m: int | None,
    precision_p: int | None,
    runs_csv: str | None,
    output_csv: str | None,
    kitti_full_val: bool,
    kitti_max_eval_samples: int | None,
    extra_args: list[str] | None,
) -> list[str]:
    argv: list[str] = [
        sys.executable,
        "-u",
        str(tools_dir / "collect_research_metrics.py"),
        "run",
        "--cuda_id",
        str(cuda_id),
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
        "--batch_size",
        str(batch_size),
        "--workers",
        str(workers),
        "--output_root",
        output_root,
        "--matrix",
        matrix,
    ]
    if all_cells_15:
        argv.append("--all-cells-15")
    elif cell:
        argv.extend(["--cell", cell])
    elif model_m is not None and precision_p is not None:
        argv.extend(["--m", str(model_m), "--p", str(precision_p)])
    if runs_csv:
        argv.extend(["--runs_csv", runs_csv])
    if output_csv:
        argv.extend(["--output-csv", output_csv])
    if kitti_full_val:
        argv.append("--accuracy")
    elif kitti_max_eval_samples is not None:
        argv.extend(["--max_eval_samples", str(kitti_max_eval_samples)])
    if extra_args:
        argv.extend(extra_args)
    return argv

# Match create_mls_env.sh: PyTorch 2.1.2 + cu118, spconv-cu118, numpy<2, setuptools<70, setup.py develop
# Do not bake *.pth / traced *.pt into the image (use Volume instead).
mls_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "build-essential",
        "clang",
        "ca-certificates",
        # OpenCV (opencv-python) runtime deps
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgl1",
        "unzip",
    )
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install("spconv-cu118")
    .add_local_dir(
        str(OPENPCDET_ROOT),
        "/opt/OpenPCDet",
        copy=True,
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/data/**",
            "**/output/**",
            "**/tools/ckpt/**",
            "**/tools/pointpillar_traced*.pt",
        ],
    )
    .run_commands(
        "cd /opt/OpenPCDet && CC=gcc CXX=g++ pip install -r requirements.txt",
        "cd /opt/OpenPCDet && pip install 'numpy<2' 'setuptools>=58,<70'",
        'cd /opt/OpenPCDet && TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" python setup.py develop',
        "pip install 'awscli>=1.32,<2'",
    )
)

# Official OpenPCDet mirror (see scripts/pointpillar_quickstart/download_kitti_aws.sh).
KITTI_AWS_BUCKET = "s3://avg-kitti"
KITTI_AWS_ZIPS = (
    "data_object_calib.zip",
    "data_object_label_2.zip",
    "data_object_image_2.zip",
    "data_object_velodyne.zip",
)


def _kitti_training_layout_ready(kitti_root: Path) -> bool:
    """True if OpenPCDet-style training split dirs exist and are non-empty."""
    tr = kitti_root / "training"
    for sub in ("velodyne", "label_2", "calib", "image_2"):
        d = tr / sub
        if not d.is_dir():
            return False
        if not any(d.iterdir()):
            return False
    return True


def _download_url_to_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "modal-mls-openpcdet/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 — user-provided artifact URLs
        dest.write_bytes(resp.read())


def sync_research_artifacts_from_volume(vol_root: Path) -> dict[str, str]:
    """
    Copy artifacts from the mounted volume into /opt/OpenPCDet paths.
    Then fill any missing file using MLS_URL_* environment variables.
    Returns ok / missing per artifact (TorchScript files are optional for most runs).
    """
    status: dict[str, str] = {}
    for name, dests in ARTIFACT_LAYOUT.items():
        src = vol_root / name
        if src.is_file():
            for d in dests:
                p = Path(d)
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, p)

        primary = Path(dests[0])
        if not primary.is_file():
            env_key = URL_ENV_FOR_ARTIFACT.get(name)
            url = os.environ.get(env_key, "").strip() if env_key else ""
            if url:
                _download_url_to_file(url, primary)
                if name == "pointpillar_7728.pth" and len(dests) > 1:
                    Path(dests[1]).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(primary, Path(dests[1]))

        status[name] = "ok" if primary.is_file() else "missing"

    return status


def _ensure_kitti_data_layout(*, require_real: bool = False) -> str:
    """
    profile_suite needs a non-empty KittiDataset (cfg DATA_PATH -> ../data/kitti from tools/).

    - If /mnt/kitti/kitti_infos_val.pkl exists: symlink /opt/OpenPCDet/data/kitti -> /mnt/kitti (real KITTI).
    - Else if require_real: raise with upload instructions.
    - Else: synthetic 1-frame bootstrap (smoke tests only).
    """
    kitti_volume.reload()
    data_parent = Path("/opt/OpenPCDet/data")
    data_kitti = data_parent / "kitti"
    mnt = Path("/mnt/kitti")
    mnt_pkl = mnt / "kitti_infos_val.pkl"

    data_parent.mkdir(parents=True, exist_ok=True)

    if mnt_pkl.is_file():
        _symlink_openpcdet_data_kitti_to(mnt)
        return "kitti_volume"

    if require_real:
        raise RuntimeError(
            "Real KITTI required but /mnt/kitti/kitti_infos_val.pkl is missing.\n"
            "Populate the KITTI volume (pick one):\n"
            f"  modal run modal_mls_app.py --action download_kitti\n"
            "Or upload from your machine:\n"
            f"  modal run modal_mls_app.py --action upload_kitti --kitti-path /path/to/OpenPCDet/data/kitti\n"
            f"  modal volume put --force {KITTI_VOLUME_NAME} /path/to/data/kitti /\n"
            "Or pass require_real_kitti=False only for synthetic smoke tests."
        )

    if data_kitti.exists() or data_kitti.is_symlink():
        if data_kitti.is_symlink():
            data_kitti.unlink()
        elif data_kitti.is_dir():
            shutil.rmtree(data_kitti)
        else:
            data_kitti.unlink()

    subprocess.run(
        [sys.executable, "/opt/OpenPCDet/tools/modal_bootstrap_kitti.py", str(data_kitti)],
        check=True,
        env=os.environ.copy(),
    )
    return "minimal_bootstrap"


def _require_base_ckpt(status: dict[str, str]) -> None:
    if status.get("pointpillar_7728.pth") != "missing":
        return
    raise RuntimeError(
        "Base checkpoint missing: pointpillar_7728.pth\n"
        f"Upload to volume {CKPT_VOLUME_NAME} (modal run ... --action upload) or set "
        f"{URL_ENV_FOR_ARTIFACT['pointpillar_7728.pth']} to an HTTPS URL."
    )


def _symlink_openpcdet_data_kitti_to(target: Path) -> None:
    """Point OpenPCDet data/kitti at target (volume mount)."""
    data_parent = Path("/opt/OpenPCDet/data")
    data_kitti = data_parent / "kitti"
    data_parent.mkdir(parents=True, exist_ok=True)
    if data_kitti.exists() or data_kitti.is_symlink():
        if data_kitti.is_symlink():
            data_kitti.unlink()
        elif data_kitti.is_dir():
            shutil.rmtree(data_kitti)
        else:
            data_kitti.unlink()
    data_kitti.symlink_to(target.resolve(), target_is_directory=True)


@app.function(
    image=mls_image,
    volumes={"/mnt/kitti": kitti_volume},
    timeout=86400,
    cpu=8.0,
)
def download_kitti_aws_openpcdet(
    force: bool = False,
    remove_zips_after_unpack: bool = True,
) -> dict:
    """
    Download KITTI 3D object detection archives from AWS Open Data (same as
    OpenPCDet/scripts/pointpillar_quickstart/download_kitti_aws.sh), unpack to the
    kitti volume, run create_kitti_infos per GETTING_STARTED.md.
    """
    kitti_volume.reload()
    root = Path("/mnt/kitti")
    root.mkdir(parents=True, exist_ok=True)
    pkl = root / "kitti_infos_val.pkl"

    if not force and pkl.is_file():
        kitti_volume.commit()
        return {
            "status": "skipped",
            "reason": "kitti_infos_val.pkl already present (pass force=True to regenerate infos)",
            "kitti_infos_val": str(pkl),
        }

    layout_ok = _kitti_training_layout_ready(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/OpenPCDet/tools" + os.pathsep + env.get("PYTHONPATH", "")

    if not layout_ok:
        zip_dir = root / "_kitti_zips_modal"
        zip_dir.mkdir(parents=True, exist_ok=True)

        for name in KITTI_AWS_ZIPS:
            dest_zip = zip_dir / name
            if dest_zip.is_file() and dest_zip.stat().st_size > 10_000:
                print(f"[kitti] reuse zip: {name}", flush=True)
            else:
                print(f"[kitti] aws s3 cp {KITTI_AWS_BUCKET}/{name}", flush=True)
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "cp",
                        "--no-sign-request",
                        f"{KITTI_AWS_BUCKET}/{name}",
                        str(dest_zip),
                    ],
                    check=True,
                    env=env,
                )

        print("[kitti] unzip into volume (layout matches OpenPCDet data/kitti)", flush=True)
        for name in KITTI_AWS_ZIPS:
            subprocess.run(["unzip", "-o", "-q", str(zip_dir / name), "-d", str(root)], check=True)

        if remove_zips_after_unpack:
            for name in KITTI_AWS_ZIPS:
                zp = zip_dir / name
                if zp.is_file():
                    zp.unlink()
            try:
                zip_dir.rmdir()
            except OSError:
                pass
    else:
        print("[kitti] training/ layout already present; skip S3 download and unzip", flush=True)

    _symlink_openpcdet_data_kitti_to(root)

    print("[kitti] create_kitti_infos (cwd=OpenPCDet/tools)", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pcdet.datasets.kitti.kitti_dataset",
            "create_kitti_infos",
            "cfgs/dataset_configs/kitti_dataset.yaml",
        ],
        cwd="/opt/OpenPCDet/tools",
        env=env,
        check=True,
    )

    if not pkl.is_file():
        raise RuntimeError("create_kitti_infos finished but kitti_infos_val.pkl missing under /mnt/kitti")

    kitti_volume.commit()
    return {
        "status": "ok",
        "kitti_infos_val": str(pkl),
        "source": KITTI_AWS_BUCKET,
        "note": "Same zips as OpenPCDet scripts/pointpillar_quickstart/download_kitti_aws.sh",
    }


@app.function(image=mls_image, gpu="T4", timeout=60 * 45)
def verify_mls_environment() -> dict:
    """Smoke-test MLS stack (no checkpoints required)."""
    out: dict = {}

    def run_py(code: str) -> str:
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        return (p.stdout or "").strip()

    out["torch_cuda"] = run_py(
        "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda or '')"
    )
    out["pcdet_import"] = run_py(
        "from pcdet.config import cfg; print('pcdet_ok', cfg.__class__.__name__)"
    )

    compile_smoke = r"""
import torch
m = torch.nn.Linear(32, 32).cuda()
x = torch.randn(4, 32, device='cuda')
y = torch.compile(m)(x)
print('compile_ok', float(y.abs().mean()))
"""
    out["torch_compile"] = run_py(compile_smoke)

    matrix_csv = "/tmp/modal_matrix_smoke.csv"
    subprocess.run(
        [
            sys.executable,
            "/opt/OpenPCDet/tools/collect_research_metrics.py",
            "matrix",
            "--output_csv",
            matrix_csv,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    p = Path(matrix_csv)
    out["matrix_csv_lines"] = len(p.read_text(encoding="utf-8").splitlines()) if p.is_file() else 0

    return out


@app.function(
    image=mls_image,
    gpu=RESEARCH_GPU,
    timeout=86400,
    volumes={
        "/mnt/ckpts": ckpt_volume,
        "/mnt/kitti": kitti_volume,
        "/mnt/results": results_volume,
    },
)
def run_collect_research_metrics(
    warmup: int = 100,
    steps: int = 50,
    cuda_id: int = 0,
    batch_size: int = 1,
    workers: int = 2,
    output_root: str = "/mnt/results/research_matrix",
    matrix: str = "fp32_amp",
    cell: str = "",
    all_cells_15: bool = False,
    model_m: int = -1,
    precision_p: int = -1,
    runs_csv: str = "",
    output_csv: str = "",
    extra_args: list[str] | None = None,
    require_real_kitti: bool = True,
    kitti_full_val: bool = False,
    kitti_max_eval_samples: int = 200,
) -> dict:
    """
    Same CLI surface as collect_research_metrics.py run (matrix / cell / all-cells / m+p / runs_csv / output_csv).
    KITTI AP: by default caps val eval at kitti_max_eval_samples (200); set kitti_full_val for full split (--accuracy).
    Results persist under output_root on volume mls-openpcdet-results; local_entrypoint downloads that tree.
    Child process stdout is streamed to Modal logs (see dashboard) in real time.
    """
    print("[mls] run_collect_research_metrics: start", flush=True)
    ckpt_volume.reload()
    results_volume.reload()
    print("[mls] volumes reloaded (ckpts + results)", flush=True)

    tools_dir = Path("/opt/OpenPCDet/tools")
    status = sync_research_artifacts_from_volume(Path("/mnt/ckpts"))
    _require_base_ckpt(status)
    kitti_mode = _ensure_kitti_data_layout(require_real=require_real_kitti)
    print(f"[mls] artifact_status={status}", flush=True)
    print(f"[mls] kitti_data={kitti_mode!r} require_real_kitti={require_real_kitti}", flush=True)

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENPCDET_CKPT"] = "/opt/OpenPCDet/tools/ckpt/pointpillar_7728.pth"
    env["PYTHONPATH"] = str(tools_dir) + os.pathsep + env.get("PYTHONPATH", "")

    cell_n = cell.strip() or None
    runs_n = runs_csv.strip() or None
    out_csv_n = output_csv.strip() or None
    mm = None if model_m < 0 else model_m
    pp = None if precision_p < 0 else precision_p

    argv = _argv_collect_research_run(
        tools_dir=tools_dir,
        cuda_id=cuda_id,
        warmup=warmup,
        steps=steps,
        batch_size=batch_size,
        workers=workers,
        output_root=str(out_root),
        matrix=matrix,
        cell=cell_n,
        all_cells_15=all_cells_15,
        model_m=mm,
        precision_p=pp,
        runs_csv=runs_n,
        output_csv=out_csv_n,
        kitti_full_val=kitti_full_val,
        kitti_max_eval_samples=None if kitti_full_val else kitti_max_eval_samples,
        extra_args=extra_args,
    )

    print("[mls] streaming collect_research_metrics to Modal logs (next lines = child output)...", flush=True)
    rc, log_txt = _stream_subprocess_to_logs(argv, cwd=str(tools_dir), env=env, banner="[mls]")
    results_volume.commit()
    print("[mls] results_volume.commit() done", flush=True)
    vol_prefix = _volume_prefix_under_results(str(out_root))
    return {
        "artifact_status": status,
        "kitti_data": kitti_mode,
        "returncode": rc,
        "output_root": str(out_root),
        "volume_download_prefix": vol_prefix,
        "results_volume": RESULTS_VOLUME_NAME,
        "log_tail": log_txt[-12000:] if len(log_txt) > 12000 else log_txt,
    }


def upload_local_artifacts_to_volume() -> None:
    """Push known filenames from this repo's OpenPCDet/tools into the Modal volume."""
    pairs: list[tuple[Path, str]] = [
        (OPENPCDET_ROOT / "tools" / "ckpt" / "pointpillar_7728.pth", "pointpillar_7728.pth"),
        (OPENPCDET_ROOT / "tools" / "pointpillar_traced.pt", "pointpillar_traced.pt"),
        (OPENPCDET_ROOT / "tools" / "pointpillar_traced_compiled.pt", "pointpillar_traced_compiled.pt"),
    ]
    for local, remote_name in pairs:
        if not local.is_file():
            print(f"[upload] skip (not found): {local}")
            continue
        print(f"[upload] {local} -> {CKPT_VOLUME_NAME}:/ {remote_name}")
        subprocess.run(
            _modal_cli_argv() + ["volume", "put", "--force", CKPT_VOLUME_NAME, str(local), remote_name],
            check=True,
        )
    print("[upload] Done. Run verify or run_collect_research_metrics next.")


def upload_kitti_dir_to_volume(kitti_dir: Path) -> None:
    """Upload a local OpenPCDet data/kitti tree to the Modal KITTI volume (volume root = DATA_PATH)."""
    kitti_dir = kitti_dir.resolve()
    if not kitti_dir.is_dir():
        raise SystemExit(f"Not a directory: {kitti_dir}")
    pkl = kitti_dir / "kitti_infos_val.pkl"
    if not pkl.is_file():
        print(f"WARNING: missing {pkl} — generate with OpenPCDet KITTI setup / create_kitti_infos.")
    print(f"[upload_kitti] {kitti_dir} -> {KITTI_VOLUME_NAME}:/ (this can take a long time)")
    subprocess.run(
        _modal_cli_argv() + ["volume", "put", "--force", KITTI_VOLUME_NAME, str(kitti_dir), "/"],
        check=True,
    )
    print("[upload_kitti] Done. Use --action run to reproduce metrics (require_real_kitti defaults true).")


@app.local_entrypoint()
def main(
    action: str = "verify",
    warmup: int = 100,
    steps: int = 50,
    cuda_id: int = 0,
    batch_size: int = 1,
    workers: int = 2,
    extra_args: str = "",
    kitti_path: str = "",
    require_real_kitti: bool = True,
    output_root: str = "/mnt/results/research_matrix",
    matrix: str = "fp32_amp",
    cell: str = "",
    all_cells: bool = False,
    model_m: int = -1,
    precision_p: int = -1,
    runs_csv: str = "",
    output_csv: str = "",
    download_to: str = "modal_mls_results",
    skip_download: bool = False,
    force_kitti_download: bool = False,
    keep_kitti_zips: bool = False,
    kitti_full_val: bool = False,
    kitti_max_eval_samples: int = 200,
):
    """
    action=verify | upload | upload_kitti | download_kitti | run
    Run knobs mirror collect_research_metrics.py: --matrix, --cell, --all-cells-15 (--all-cells),
    --m/--p (--model-m / --precision-p), --runs_csv, --output-csv (--output-csv).
    KITTI: default --kitti-max-eval-samples 200 per cell (val subsample); --kitti-full-val for full val AP.
    Pass-through: --extra-args '...' (avoid duplicating flags set above).
    On success, downloads the whole output_root subtree from the results volume to --download-to.
    """
    print("OPENPCDET_ROOT:", OPENPCDET_ROOT)
    if not OPENPCDET_ROOT.is_dir():
        raise SystemExit(f"Missing OpenPCDet at {OPENPCDET_ROOT}")

    if action == "verify":
        result = verify_mls_environment.remote()
        for k, v in result.items():
            print(f"{k}: {v}")
        print("verify_mls_environment: OK")
        return

    if action == "upload":
        upload_local_artifacts_to_volume()
        return

    if action == "upload_kitti":
        if not kitti_path.strip():
            raise SystemExit(
                "upload_kitti requires --kitti-path /abs/path/to/OpenPCDet/data/kitti "
                "(folder that contains kitti_infos_val.pkl)"
            )
        upload_kitti_dir_to_volume(Path(kitti_path))
        return

    if action == "download_kitti":
        out = download_kitti_aws_openpcdet.remote(
            force=force_kitti_download,
            remove_zips_after_unpack=not keep_kitti_zips,
        )
        print(out)
        return

    if action == "run":
        extra_list = extra_args.split() if extra_args.strip() else None
        out = run_collect_research_metrics.remote(
            warmup=warmup,
            steps=steps,
            cuda_id=cuda_id,
            batch_size=batch_size,
            workers=workers,
            output_root=output_root,
            matrix=matrix,
            cell=cell,
            all_cells_15=all_cells,
            model_m=model_m,
            precision_p=precision_p,
            runs_csv=runs_csv,
            output_csv=output_csv,
            extra_args=extra_list,
            require_real_kitti=require_real_kitti,
            kitti_full_val=kitti_full_val,
            kitti_max_eval_samples=kitti_max_eval_samples,
        )
        print("artifact_status:", out.get("artifact_status"))
        print("kitti_data:", out.get("kitti_data"))
        print("results_volume:", out.get("results_volume"))
        print("returncode:", out.get("returncode"))
        print("output_root:", out.get("output_root"))
        print("volume_download_prefix:", out.get("volume_download_prefix"))
        print(out.get("log_tail", ""))
        if out.get("returncode") != 0:
            raise SystemExit(out.get("returncode", 1))
        if not skip_download:
            prefix = out.get("volume_download_prefix") or _volume_prefix_under_results(output_root)
            dest = Path(download_to)
            print(f"[download] modal volume get {RESULTS_VOLUME_NAME} /{prefix} -> {dest.resolve()}")
            _download_results_tree_local(prefix, dest)
            print("[download] Done.")
        return

    raise SystemExit(
        f"Unknown action: {action!r} (use verify, upload, upload_kitti, download_kitti, run)"
    )
