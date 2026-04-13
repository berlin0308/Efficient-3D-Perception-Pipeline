"""
Collect research metrics into runs.csv (and optional ncu_kernels.csv).

Run from OpenPCDet/tools/ (use a conda env where OpenPCDet CUDA extensions were built, e.g. conda activate mls):

  # Defaults: cfg under tools/, ckpt from OPENPCDET_CKPT or ckpt/pointpillar_7728.pth, output under MLS/profile_outputs/
  python collect_research_metrics.py run --cuda_id 0 --warmup 100 --steps 50

  # Or explicit paths:
  python collect_research_metrics.py run \\
      --cfg_file cfgs/kitti_models/pointpillar.yaml \\
      --ckpt /path/to/pointpillar_7728.pth \\
      --output_root ../../profile_outputs/research_matrix

  # Export M0–M4 × (FP32, AMP) design for reports
  python collect_research_metrics.py matrix --output_csv ../../profile_outputs/research_matrix/experiment_matrix_fp32_amp.csv

  # Run only runnable cells (see research_experiment_matrix.py); --matrix 15 is an alias for fp32_amp
  python collect_research_metrics.py run --matrix fp32_amp --cuda_id 0 --warmup 100 --steps 50

  # Single cell, one CSV (overwrite); cell id M{0-4}_FP32 or M{0-4}_AMP:
  python collect_research_metrics.py run --matrix fp32_amp --cell M0_FP32 \\
      --output-csv ../../profile_outputs/research_matrix/cell_runs/run_M0_FP32.csv

  # Shorthand: --m 0–4, --p 1=FP32 2=AMP
  python collect_research_metrics.py run --matrix fp32_amp --m 1 --p 1 --output-csv ./run_M1_FP32.csv

  # Full KITTI val for AP (slow):
  python collect_research_metrics.py run --matrix fp32_amp --cell M0_FP32 --accuracy --output-csv ./run_M0_FP32_full.csv

  # Merge existing profile + energy directories into runs.csv
  python collect_research_metrics.py merge \\
      --manifest my_manifest.json \\
      --runs_csv ../../profile_outputs/research_matrix/runs.csv

Manifest JSON (list of objects), example:
[
  {
    "variant_name": "baseline_fp32",
    "profile_dir": "/path/to/fp32",
    "energy_dir": "/path/to/energy_fp32",
    "map_car_r11": "77.28",
    "ncu_csv": "/path/to/ncu_export.csv"
  }
]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import research_metrics_schema as schema  # noqa: E402
import research_experiment_matrix as exp15  # noqa: E402


def _tools_dir() -> Path:
    return Path(__file__).resolve().parent


def _openpcdet_root() -> Path:
    return _tools_dir().parent


def _default_output_root(openpcdet_root: Path) -> Path:
    """Prefer MLS/profile_outputs/research_matrix when repo layout matches MLS/OpenPCDet."""
    mls_profile = openpcdet_root.parent / 'profile_outputs' / 'research_matrix'
    if (openpcdet_root.parent / 'profile_outputs').is_dir():
        return mls_profile
    return openpcdet_root / 'profile_outputs' / 'research_matrix'


def _resolve_cfg_file(cli_cfg: str, tools_dir: Path, openpcdet_root: Path) -> Path:
    p = Path(cli_cfg).expanduser()
    if p.is_file():
        return p.resolve()
    rel_tools = (tools_dir / cli_cfg).resolve()
    if rel_tools.is_file():
        return rel_tools
    rel_root = (openpcdet_root / cli_cfg).resolve()
    if rel_root.is_file():
        return rel_root
    raise SystemExit(
        'Config file not found: %r\nTried: %s, %s, %s\n(Run from OpenPCDet/tools/ or pass an absolute path.)'
        % (cli_cfg, p, rel_tools, rel_root)
    )


def _resolve_ckpt_path(cli_ckpt: str | None, tools_dir: Path, openpcdet_root: Path) -> Path:
    """
    Resolve checkpoint: explicit --ckpt, then OPENPCDET_CKPT / POINTPILLAR_CKPT,
    then tools/ckpt/pointpillar_7728.pth (same as inference_pointpillar_baseline.sh from tools/),
    then OpenPCDet/ckpt/pointpillar_7728.pth.
    """
    tried: list[str] = []
    ordered: list[Path] = []
    if cli_ckpt:
        ordered.append(Path(cli_ckpt).expanduser())
    for env_key in ('OPENPCDET_CKPT', 'POINTPILLAR_CKPT'):
        v = os.environ.get(env_key, '').strip()
        if v:
            ordered.append(Path(v).expanduser())
    ordered.append(tools_dir / 'ckpt' / 'pointpillar_7728.pth')
    ordered.append(openpcdet_root / 'ckpt' / 'pointpillar_7728.pth')
    # Historical lab path from prior profile logs (used only if present)
    ordered.append(Path('/media/emma/data/ML_sys/Kitti/pointpillar_7728.pth'))

    seen: set[str] = set()
    for p in ordered:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        tried.append(str(p))
        if p.is_file():
            return p.resolve()

    raise SystemExit(
        'Checkpoint not found. Tried (in order):\n  %s\n\n'
        'Set a real path, e.g.:\n'
        '  export OPENPCDET_CKPT=/path/to/pointpillar_7728.pth\n'
        'or place the file at:\n  %s\n'
        % ('\n  '.join(tried), tools_dir / 'ckpt' / 'pointpillar_7728.pth')
    )


def _prepare_run_paths(args: argparse.Namespace) -> None:
    """Set args.cfg_file and args.ckpt to absolute resolved strings; default output_root if needed."""
    tools = _tools_dir()
    root = _openpcdet_root()
    args.cfg_file = str(_resolve_cfg_file(args.cfg_file, tools, root))
    args.ckpt = str(_resolve_ckpt_path(args.ckpt, tools, root))
    if getattr(args, 'output_root', None) in (None, ''):
        args.output_root = str(_default_output_root(root))
    print('[collect_research_metrics] cfg_file:', args.cfg_file, flush=True)
    print('[collect_research_metrics] ckpt:', args.ckpt, flush=True)
    print('[collect_research_metrics] output_root:', args.output_root, flush=True)


def _empty_row() -> dict:
    return {c: '' for c in schema.RUNS_CSV_COLUMNS}


def _format_ap_cell(v) -> str:
    if v is None or v == '':
        return ''
    try:
        return f'{float(v):.4f}'
    except (TypeError, ValueError):
        return str(v)


def _apply_kitti_eval_json(
    row: dict,
    json_path: Path | None,
    cli_map_car: str,
    user_eval_notes: str,
) -> None:
    """
    Fill map_car_r11 and kitti_car_3d_* from kitti_eval_export.py output.
    If JSON is missing, keep cli_map_car / user_eval_notes on the row.
    """
    default_note = (
        'KITTI val split, official Python eval (R40). '
        'Column map_car_r11 stores Car 3D AP moderate (legacy headline name).'
    )
    if not json_path or not json_path.is_file():
        if cli_map_car:
            row['map_car_r11'] = cli_map_car
        if user_eval_notes:
            row['eval_protocol_notes'] = user_eval_notes
        return
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        if cli_map_car:
            row['map_car_r11'] = cli_map_car
        if user_eval_notes:
            row['eval_protocol_notes'] = user_eval_notes
        return
    metrics = data.get('metrics', data)
    easy_k = 'Car_3d/easy_R40'
    mod_k = 'Car_3d/moderate_R40'
    hard_k = 'Car_3d/hard_R40'
    row['kitti_car_3d_easy_r40'] = _format_ap_cell(metrics.get(easy_k))
    row['kitti_car_3d_moderate_r40'] = _format_ap_cell(metrics.get(mod_k))
    row['kitti_car_3d_hard_r40'] = _format_ap_cell(metrics.get(hard_k))
    row['map_car_r11'] = row['kitti_car_3d_moderate_r40'] or cli_map_car
    parts = [p for p in (user_eval_notes.strip(), default_note) if p]
    row['eval_protocol_notes'] = ' '.join(parts)


def _run_cmd(argv: list[str], cwd: Path, env: dict | None = None) -> None:
    print('[collect_research_metrics] exec:', ' '.join(argv), flush=True)
    r = subprocess.run(argv, cwd=str(cwd), env=env)
    if r.returncode != 0:
        raise SystemExit('Command failed (%d): %s' % (r.returncode, argv))


def _git_commit(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ''


def _nvidia_driver_version() -> str:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip().split('\n')[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ''


def _torch_cuda_versions() -> tuple[str, str]:
    try:
        import torch
        return str(torch.__version__), (torch.version.cuda or '')
    except Exception:
        return '', ''


def _parse_profile_summary(text: str) -> dict:
    """Parse profile_summary.txt into flat prof_* keys."""
    out = {}
    stage_map = [
        ('── T_rt (onboard) ──', 't_rt'),
        ('── Onboard frame ──', 'full_frame'),
        ('read_points (excl.T_rt)', 'read_points'),
        ('cpu_prepare', 'cpu_prepare'),
        ('data_to_gpu', 'data_to_gpu_pts'),
        ('pre_processing', 'pre_processing'),
        ('h2d (load rest)', 'h2d_voxel_tail'),
        ('DataLoader (CPU)', 'dataloader'),
        ('H2D Transfer', 'h2d'),
        ('Forward (GPU)', 'forward'),
        ('Forward (CPU+GPU)', 'forward'),
        ('PostProcess', 'postprocess'),
        ('── Full Frame ──', 'full_frame'),
        ('Full Frame', 'full_frame'),
    ]

    for line in text.splitlines():
        line_stripped = line.strip()
        for label, key in stage_map:
            if label in line and re.search(r'[\d.]+\s+[\d.]+', line):
                parts = line.split()
                floats = []
                i = len(parts) - 1
                while i >= 0:
                    try:
                        floats.append(float(parts[i]))
                        if len(floats) == 5:
                            break
                    except ValueError:
                        break
                    i -= 1
                floats.reverse()
                if len(floats) == 5:
                    mean, p50, p95, p99, std = floats
                elif len(floats) == 4:
                    mean, p50, p99, std = floats
                    p95 = ''
                else:
                    continue
                prefix = 'prof_%s' % key
                out['%s_mean_ms' % prefix] = str(mean)
                out['%s_p50_ms' % prefix] = str(p50)
                out['%s_p95_ms' % prefix] = str(p95) if p95 != '' else ''
                out['%s_p99_ms' % prefix] = str(p99)
                out['%s_std_ms' % prefix] = str(std)
                break

    m = re.search(r'Throughput\s*:\s*([\d.]+)\s*samples/s', text)
    if m:
        out['prof_throughput_sps'] = m.group(1)
    m = re.search(r'Peak GPU Memory\s*:\s*([\d.]+)\s*MB', text)
    if m:
        out['prof_peak_gpu_memory_mb'] = m.group(1)
    m = re.search(r'Peak GPU Mem \(steady\):\s*([\d.]+)\s*MB', text)
    if m:
        out['prof_peak_gpu_memory_steady_mb'] = m.group(1)
    m = re.search(r'T_rt mean \(onboard, excl\. read_points\):\s*([\d.]+)\s*ms', text)
    if m:
        out['prof_t_rt_mean_ms'] = m.group(1)
    m = re.search(r'Mean peak per step\s*:\s*([\d.]+)\s*MB', text)
    if m:
        out['prof_mean_peak_gpu_memory_mb'] = m.group(1)
    m = re.search(r'CUDA kernel events\s*:\s*(\d+|n/a[^\n]*)', text)
    if m:
        out['prof_cuda_kernel_events'] = m.group(1).split()[0] if m.group(1) else ''

    m = re.search(r'Config\s*:\s*(.+)', text)
    if m:
        out['_parsed_config'] = m.group(1).strip()
    m = re.search(r'Ckpt\s*:\s*(.+)', text)
    if m:
        out['_parsed_ckpt'] = m.group(1).strip()
    m = re.search(r'Compile\s*:\s*(\w+)', text)
    if m:
        out['_parsed_compile'] = m.group(1).strip()
    m = re.search(r'AMP \(fp16\):\s*(\w+)', text)
    if m:
        out['_parsed_amp'] = m.group(1).strip()
    m = re.search(r'INT8\s*:\s*(\w+)', text)
    if m:
        out['_parsed_int8'] = m.group(1).strip()
    m = re.search(r'Memory opt scatter \(HWC write\):\s*(\w+)', text)
    if m:
        out['_parsed_memory_opt_scatter'] = m.group(1).strip()
    m = re.search(r'Memory opt conv2d \(channels_last\):\s*(\w+)', text)
    if m:
        out['_parsed_memory_opt_conv2d'] = m.group(1).strip()
    m = re.search(r'Preprocess GPU \(voxelize on GPU\):\s*(\w+)', text)
    if m:
        out['_parsed_preprocess_gpu'] = m.group(1).strip()
    m = re.search(r'Compile voxelizer:\s*(\w+)', text)
    if m:
        out['_parsed_compile_voxelizer'] = m.group(1).strip()
    wm = re.search(r'Warmup\s*:\s*(\d+)\s*steps\s*Measured:\s*(\d+)\s*steps\s*Batch:\s*(\d+)', text)
    if wm:
        out['_parsed_warmup'] = wm.group(1)
        out['_parsed_steps'] = wm.group(2)
        out['_parsed_batch'] = wm.group(3)
    return out


def _parse_energy_summary(text: str) -> dict:
    out = {}
    m = re.search(
        r'Mean\s*:\s*([\d.]+)\s+p50\s*:\s*([\d.]+)\s+p95\s*:\s*([\d.]+)\s+p99\s*:\s*([\d.]+)',
        text,
        re.I,
    )
    if m:
        out['energy_forward_mean_ms'] = m.group(1)
        out['energy_forward_p50_ms'] = m.group(2)
        out['energy_forward_p95_ms'] = m.group(3)
        out['energy_forward_p99_ms'] = m.group(4)
    else:
        m = re.search(r'Mean\s*:\s*([\d.]+)\s+p50\s*:\s*([\d.]+)\s+p99\s*:\s*([\d.]+)', text, re.I)
        if m:
            out['energy_forward_mean_ms'] = m.group(1)
            out['energy_forward_p50_ms'] = m.group(2)
            out['energy_forward_p99_ms'] = m.group(3)
    m = re.search(r'([\d.]+)\s+samples/s', text)
    if m:
        out['energy_throughput_sps'] = m.group(1)
    m = re.search(r'total_samples=(\d+),\s*wall=([\d.]+)\s*s', text)
    if m:
        out['measured_steps'] = m.group(1)
        out['energy_wall_time_s'] = m.group(2)
    m = re.search(r'Mean power\s*:\s*([\d.]+)\s*W', text)
    if m:
        out['energy_mean_power_W'] = m.group(1)
    m = re.search(r'Peak power\s*:\s*([\d.]+)\s*W', text)
    if m:
        out['energy_peak_power_W'] = m.group(1)
    m = re.search(r'Total energy\s*:\s*([\d.]+)\s*J', text)
    if m:
        out['energy_total_J'] = m.group(1)
    m = re.search(r'([\d.]+)\s+samples/J', text)
    if m:
        out['energy_samples_per_J'] = m.group(1)
    m = re.search(r'([\d.]+)\s+samples/s/W', text)
    if m:
        out['energy_samples_per_s_per_W'] = m.group(1)
    m = re.search(r'GPU\s*:\s*(.+)', text)
    if m:
        out['gpu_name'] = m.group(1).strip()
    m = re.search(r'Config\s*:\s*(.+)', text)
    if m:
        out['_energy_config'] = m.group(1).strip()
    m = re.search(r'Compile\s*:\s*(\w+)', text)
    if m:
        out['_energy_compile'] = m.group(1).strip()
    m = re.search(r'AMP \(fp16\)\s*:\s*(\w+)', text)
    if m:
        out['_energy_amp'] = m.group(1).strip()
    m = re.search(r'Warmup\s*:\s*(\d+)\s+Steps:\s*(\d+)\s+Batch:\s*(\d+)', text)
    if m:
        out['_energy_warmup'] = m.group(1)
        out['_energy_steps'] = m.group(2)
        out['_energy_batch'] = m.group(3)
    return out


def _norm_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def _parse_ncu_csv(path: Path) -> tuple[dict, list[dict]]:
    """
    Best-effort parse of an Nsight Compute CSV export.
    Returns (summary_for_runs_row, kernel_rows_for_ncu_kernels_csv).
    """
    summary = {
        'ncu_report_path': str(path),
        'ncu_top_kernel_name': '',
        'ncu_roofline_bound': '',
        'ncu_compute_intensity_flop_per_byte': '',
        'ncu_dram_throughput_gbps': '',
        'ncu_mem_throughput_pct_of_peak': '',
    }
    kernel_rows: list[dict] = []
    if not path.is_file():
        return summary, kernel_rows

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return summary, kernel_rows
        fnorm = {_norm_key(h): h for h in reader.fieldnames if h}

        def pick(row, *candidates):
            for c in candidates:
                k = _norm_key(c)
                if k in fnorm:
                    v = row.get(fnorm[k], '').strip()
                    if v:
                        return v
            return ''

        best = None
        best_dur = -1.0
        for row in reader:
            if not any(v.strip() for v in row.values() if v):
                continue
            dur_s = pick(row, 'Duration', 'GPU Time Duration', 'Section Name')
            try:
                dur = float(re.sub(r'[^\d.]', '', dur_s) or 0)
            except ValueError:
                dur = 0.0
            kname = pick(row, 'Kernel Name', 'Name', 'Metric Name')
            if dur > best_dur and kname:
                best_dur = dur
                best = (kname, row)

            kr = {
                'kernel_name': pick(row, 'Kernel Name', 'Name') or '',
                'section_name': pick(row, 'Section Name', '') or '',
                'duration_us': pick(row, 'Duration', 'GPU Time Duration') or '',
                'dram_throughput_gbps': pick(
                    row, 'DRAM Throughput', 'Memory Throughput', 'L1/TEX Throughput',
                ) or '',
                'sm_throughput_pct': pick(row, 'Compute (SM) Throughput', 'SM Throughput') or '',
                'mem_throughput_pct': pick(row, 'Memory Throughput', 'DRAM Throughput') or '',
                'compute_throughput_pct': pick(row, 'Compute Throughput') or '',
                'l2_hit_rate_pct': pick(row, 'L2 Hit Rate', 'L2 Cache Hit Rate') or '',
                'dram_bytes_read': pick(row, 'DRAM Bytes Read', 'Bytes Read') or '',
                'dram_bytes_write': pick(row, 'DRAM Bytes Write', 'Bytes Write') or '',
                'roofline_bound_note': pick(row, 'Workload Analysis', 'Bound') or '',
            }
            if kr['kernel_name']:
                kernel_rows.append(kr)

        if best:
            summary['ncu_top_kernel_name'] = best[0]
            row = best[1]
            summary['ncu_dram_throughput_gbps'] = pick(
                row, 'DRAM Throughput', 'Memory Throughput',
            )
            summary['ncu_mem_throughput_pct_of_peak'] = pick(
                row, 'Memory Throughput', 'DRAM Throughput',
            )
            b = pick(row, 'Workload Analysis', 'Bound', 'Pipe')
            if b:
                summary['ncu_roofline_bound'] = b

    return summary, kernel_rows


def build_row(
    variant_name: str,
    profile_dir: Path,
    energy_dir: Path,
    cfg_file: str,
    ckpt: str,
    cuda_id: int | None,
    workers: int | None,
    ncu_csv: Path | None = None,
    map_car_r11: str = '',
    eval_notes: str = '',
    notes_amp_compile: str = '',
    experiment_meta: dict | None = None,
    kitti_eval_json: Path | None = None,
) -> dict:
    row = _empty_row()
    run_id = uuid.uuid4().hex[:12]
    row['run_id'] = run_id
    row['timestamp_iso'] = dt.datetime.now(dt.timezone.utc).isoformat()
    row['variant_name'] = variant_name
    row['model_name'] = 'PointPillars'
    if experiment_meta:
        row['experiment_cell_id'] = str(experiment_meta.get('experiment_cell_id', ''))
        row['model_variant'] = str(experiment_meta.get('model_variant', ''))
        row['precision_mode'] = str(experiment_meta.get('precision_mode', ''))
        row['experiment_status'] = str(experiment_meta.get('experiment_status', ''))
    row['config_path'] = cfg_file
    row['checkpoint_path'] = ckpt
    if cuda_id is not None:
        row['cuda_id'] = str(cuda_id)
    if workers is not None:
        row['num_workers'] = str(workers)
    row['energy_method'] = 'nvml_integrated'
    row['notes_amp_compile_mutually_exclusive'] = notes_amp_compile
    row['map_car_r11'] = map_car_r11
    row['eval_protocol_notes'] = eval_notes
    row['kitti_car_3d_easy_r40'] = ''
    row['kitti_car_3d_moderate_r40'] = ''
    row['kitti_car_3d_hard_r40'] = ''
    row['commit_hash'] = _git_commit(_openpcdet_root())
    row['driver_version'] = _nvidia_driver_version()
    tv, cv = _torch_cuda_versions()
    row['pytorch_version'] = tv
    row['cuda_version'] = cv

    prof_path = profile_dir / 'profile_summary.txt'
    if prof_path.is_file():
        parsed = _parse_profile_summary(prof_path.read_text())
        for k, v in parsed.items():
            if not k.startswith('_') and k in row:
                row[k] = v
        if parsed.get('_parsed_config'):
            row['config_path'] = parsed['_parsed_config']
        if parsed.get('_parsed_ckpt'):
            row['checkpoint_path'] = parsed['_parsed_ckpt']
        row['flag_compile'] = (
            'true' if str(parsed.get('_parsed_compile', '')).lower() == 'true' else 'false'
        )
        row['flag_amp'] = (
            'true' if str(parsed.get('_parsed_amp', '')).lower() == 'true' else 'false'
        )
        row['flag_int8'] = (
            'true' if str(parsed.get('_parsed_int8', '')).lower() == 'true' else 'false'
        )
        row['flag_nhwc'] = (
            'true' if str(parsed.get('_parsed_memory_opt_conv2d', '')).lower() == 'true' else 'false'
        )
        row['flag_memory_opt_scatter'] = (
            'true' if str(parsed.get('_parsed_memory_opt_scatter', '')).lower() == 'true' else 'false'
        )
        row['flag_preprocess_gpu'] = (
            'true' if str(parsed.get('_parsed_preprocess_gpu', '')).lower() == 'true' else 'false'
        )
        row['flag_compile_voxelizer'] = (
            'true' if str(parsed.get('_parsed_compile_voxelizer', '')).lower() == 'true' else 'false'
        )
        if parsed.get('_parsed_warmup'):
            row['warmup_steps'] = parsed['_parsed_warmup']
        if parsed.get('_parsed_steps'):
            row['measured_steps'] = parsed['_parsed_steps']
        if parsed.get('_parsed_batch'):
            row['batch_size'] = parsed['_parsed_batch']
    row['profile_output_dir'] = str(profile_dir)
    row['profile_latency_per_step_csv'] = str(profile_dir / 'latency_per_step.csv')

    eng_path = energy_dir / 'energy_summary.txt'
    if eng_path.is_file():
        ep = _parse_energy_summary(eng_path.read_text())
        for k, v in ep.items():
            if not k.startswith('_') and k in row:
                row[k] = v
        if ep.get('gpu_name'):
            row['gpu_name'] = ep['gpu_name']
        if ep.get('_energy_warmup'):
            row['warmup_steps'] = ep['_energy_warmup']
        if ep.get('_energy_steps'):
            row['measured_steps'] = ep['_energy_steps']
        if ep.get('_energy_batch'):
            row['batch_size'] = ep['_energy_batch']
    row['energy_output_dir'] = str(energy_dir)
    row['energy_samples_csv'] = str(energy_dir / 'energy_samples.csv')
    row['energy_latency_per_step_csv'] = str(energy_dir / 'energy_latency_per_step.csv')

    if ncu_csv:
        ncu_sum, _ = _parse_ncu_csv(Path(ncu_csv))
        for k, v in ncu_sum.items():
            if k in row:
                row[k] = v

    # normalize booleans to string true/false for CSV consistency
    for fk in (
        'flag_compile', 'flag_amp', 'flag_preprocess_gpu', 'flag_compile_voxelizer',
        'flag_nhwc', 'flag_memory_opt_scatter', 'flag_int8', 'flag_fp16_full',
    ):
        v = row[fk]
        if v in ('', None):
            row[fk] = 'false'
        elif isinstance(v, bool):
            row[fk] = 'true' if v else 'false'
        elif str(v).lower() in ('true', 'false'):
            row[fk] = str(v).lower()

    _apply_kitti_eval_json(row, kitti_eval_json, map_car_r11, eval_notes)

    return row


def _legacy_experiment_meta(variant_name: str) -> dict:
    """Map legacy 3-run variant names to M0/M1 experiment cell IDs."""
    table = {
        'baseline_fp32': {
            'experiment_cell_id': 'M0_FP32',
            'model_variant': 'M0',
            'precision_mode': 'FP32',
            'experiment_status': 'measured',
        },
        'fp16_amp': {
            'experiment_cell_id': 'M0_AMP',
            'model_variant': 'M0',
            'precision_mode': 'AMP',
            'experiment_status': 'measured',
        },
        'torch_compile_fp32': {
            'experiment_cell_id': 'M1_FP32',
            'model_variant': 'M1',
            'precision_mode': 'FP32',
            'experiment_status': 'measured',
        },
    }
    return table.get(variant_name, {})


def _cell_to_experiment_meta(cell: dict) -> dict:
    st = cell['status']
    exp_status = 'measured' if st == 'runnable' else ('na' if st == 'blocked' else 'future')
    return {
        'experiment_cell_id': cell['cell_id'],
        'model_variant': cell['model_variant'],
        'precision_mode': cell['precision_mode'],
        'experiment_status': exp_status,
    }


# Old M×P₃ ids (pre FP32/AMP rename) → new cell_id
_LEGACY_M_P_TO_CELL: dict[str, str] = {
    'M1_P1': 'M0_FP32', 'M1_P2': 'M0_AMP',
    'M2_P1': 'M1_FP32', 'M2_P2': 'M1_AMP',
    'M3_P1': 'M2_FP32',  # expands to M2_FP32_mem_* triple in _build_run_matrix
    'M3_P2': 'M2_AMP',  # expands to M2_AMP_mem_* triple in _build_run_matrix
    'M4_P1': 'M3_FP32', 'M4_P2': 'M3_AMP',
    'M5_P1': 'M4_FP32', 'M5_P2': 'M4_AMP',
}


def _normalize_cell_id(raw: str) -> str:
    s0 = raw.strip()
    for c in exp15.experiment_matrix_fp32_amp():
        if c['cell_id'] == s0:
            return c['cell_id']
    s = s0.upper()
    m = re.match(r'^M(\d+)_(FP32|AMP)$', s)
    if m:
        return 'M%d_%s' % (int(m.group(1)), m.group(2))
    m_old = re.match(r'^M(\d+)_P(\d+)$', s)
    if m_old:
        key = 'M%d_P%d' % (int(m_old.group(1)), int(m_old.group(2)))
        if key in _LEGACY_M_P_TO_CELL:
            return _LEGACY_M_P_TO_CELL[key]
        raise SystemExit(
            'Invalid legacy --cell %r (INT8 column removed; use Mx_FP32 / Mx_AMP, e.g. M0_FP32).' % raw
        )
    raise SystemExit(
        'Invalid --cell %r (expected design cell_id, M0_FP32, M1_AMP, M2_FP32_mem_scatter, …, or legacy M1_P1)' % raw
    )


def _matrix_uses_design(args: argparse.Namespace) -> bool:
    return getattr(args, 'matrix', 'legacy') in ('15', 'fp32_amp')


def _parse_cell_selection(args: argparse.Namespace) -> list[str] | None:
    """
    None = default design-matrix behavior (only runnable cells).
    Non-empty list = explicit cell_ids in order (may include future/blocked).
    """
    if getattr(args, 'all_cells_15', False):
        if getattr(args, 'cell', None) or getattr(args, 'm', None) is not None or getattr(args, 'p', None) is not None:
            raise SystemExit('Do not combine --all-cells-15 with --cell or --m/--p.')
        return [c['cell_id'] for c in exp15.experiment_matrix_fp32_amp()]
    ids: list[str] = []
    if getattr(args, 'm', None) is not None or getattr(args, 'p', None) is not None:
        if args.m is None or args.p is None:
            raise SystemExit(
                'With --matrix fp32_amp, use both --m (0–4) and --p (1=FP32, 2=AMP), e.g. --m 0 --p 1.'
            )
        prec = {1: 'FP32', 2: 'AMP'}.get(int(args.p))
        if prec is None:
            raise SystemExit('--p must be 1 (FP32) or 2 (AMP).')
        mi = int(args.m)
        if mi < 0 or mi > 4:
            raise SystemExit('--m must be between 0 and 4 (baseline=M0, compile=M1, …).')
        ids.append('M%d_%s' % (mi, prec))
    if getattr(args, 'cell', None):
        for part in str(args.cell).split(','):
            part = part.strip()
            if part:
                ids.append(_normalize_cell_id(part))
    if not ids:
        return None
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def _gpu_can_run_cell(cell: dict) -> bool:
    """True if profile_suite/energy should be attempted for this design cell."""
    return cell['status'] == 'runnable'


def _spec_from_cell_15(cell: dict) -> dict:
    return {
        'cell': cell,
        'variant_name': cell['variant_name'],
        'compile': bool(cell['compile']),
        'amp': bool(cell['amp']),
        'experiment_meta': _cell_to_experiment_meta(cell),
        'gpu_runnable': _gpu_can_run_cell(cell),
        'skip_reason': str(cell.get('skip_reason', '') or ''),
    }


def _expand_design_aliases(cell_ids: list[str], by_id: dict[str, dict]) -> list[str]:
    """M2_FP32 / M2_AMP expand to three memory-layout cells when not present as a single row."""
    m2_fp32 = (
        'M2_FP32_mem_scatter',
        'M2_FP32_mem_conv2d',
        'M2_FP32_mem_both',
    )
    m2_amp = (
        'M2_AMP_mem_scatter',
        'M2_AMP_mem_conv2d',
        'M2_AMP_mem_both',
    )
    out: list[str] = []
    for cid in cell_ids:
        if cid == 'M2_FP32' and cid not in by_id:
            for sub in m2_fp32:
                if sub in by_id:
                    out.append(sub)
        elif cid == 'M2_AMP' and cid not in by_id:
            for sub in m2_amp:
                if sub in by_id:
                    out.append(sub)
        else:
            out.append(cid)
    return out


def _memory_opts_for_run(spec: dict, args: argparse.Namespace) -> tuple[bool, bool]:
    cell = spec.get('cell')
    if cell is not None:
        return (
            bool(cell.get('memory_opt_scatter', False)),
            bool(cell.get('memory_opt_conv2d', False)),
        )
    return (
        bool(getattr(args, 'memory_opt_scatter', False)),
        bool(getattr(args, 'memory_opt_conv2d', False)),
    )


def _preprocess_opts_for_run(spec: dict, args: argparse.Namespace) -> tuple[bool, bool]:
    cell = spec.get('cell')
    if cell is not None:
        return (
            bool(cell.get('preprocess_gpu', False)),
            bool(cell.get('compile_voxelizer', False)),
        )
    return (
        bool(getattr(args, 'preprocess_gpu', False)),
        bool(getattr(args, 'compile_voxelizer', False)),
    )


def _build_run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    use_design = _matrix_uses_design(args)
    requested = _parse_cell_selection(args)

    if not use_design:
        if requested is not None:
            raise SystemExit('--cell / --m / --p / --all-cells-15 require --matrix fp32_amp (or legacy alias 15).')
        matrix = schema.default_variant_matrix()
        if args.variants.strip().lower() != 'all':
            wanted = {v.strip() for v in args.variants.split(',') if v.strip()}
            matrix = [m for m in matrix if m['variant_name'] in wanted]
            if not matrix:
                raise SystemExit('No variants matched: %s' % args.variants)
        out = []
        for m in matrix:
            out.append({
                'cell': None,
                'variant_name': m['variant_name'],
                'compile': m['compile'],
                'amp': m['amp'],
                'experiment_meta': _legacy_experiment_meta(m['variant_name']),
                'gpu_runnable': True,
                'skip_reason': '',
            })
        return out

    cells_all = exp15.experiment_matrix_fp32_amp()
    by_id = {c['cell_id']: c for c in cells_all}

    if requested is None:
        out = []
        for cell in cells_all:
            if cell['status'] != 'runnable':
                print(
                    '[skip] %s (%s): %s — %s'
                    % (
                        cell['cell_id'],
                        cell['variant_name'],
                        cell['status'],
                        cell.get('skip_reason', ''),
                    ),
                    flush=True,
                )
                continue
            spec = _spec_from_cell_15(cell)
            if not spec['gpu_runnable']:
                print(
                    '[skip] %s (%s): compile+amp not supported in this stack'
                    % (cell['cell_id'], cell['variant_name']),
                    flush=True,
                )
                continue
            out.append(spec)
        if not out:
            raise SystemExit('No runnable cells in fp32_amp matrix (check tooling).')
        return out

    out = []
    for cid in _expand_design_aliases(requested, by_id):
        if cid not in by_id:
            raise SystemExit(
                'Unknown cell_id %r. Valid ids: %s'
                % (cid, ', '.join(sorted(by_id.keys())))
            )
        out.append(_spec_from_cell_15(by_id[cid]))
    return out


def _eval_notes_with_subsample(base_notes: str, max_samples: int | None, ran_kitti: bool) -> str:
    if not ran_kitti or max_samples is None:
        return base_notes
    extra = 'KITTI AP from first %d val samples (subsampled; not full split).' % max_samples
    if base_notes.strip():
        return base_notes.strip() + ' ' + extra
    return extra


def write_runs_csv_overwrite(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(schema.RUNS_CSV_COLUMNS), extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in schema.RUNS_CSV_COLUMNS})
    print('[collect_research_metrics] wrote %d row(s) to %s' % (len(rows), path), flush=True)


def append_runs_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(schema.RUNS_CSV_COLUMNS), extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in schema.RUNS_CSV_COLUMNS})


def append_ncu_kernels_csv(path: Path, run_id: str, variant_name: str, kernels: list[dict]) -> None:
    if not kernels:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(schema.NCU_KERNELS_CSV_COLUMNS), extrasaction='ignore')
        if write_header:
            w.writeheader()
        for kr in kernels:
            out = {c: '' for c in schema.NCU_KERNELS_CSV_COLUMNS}
            out['run_id'] = run_id
            out['variant_name'] = variant_name
            out['kernel_name'] = kr.get('kernel_name', '')
            out['section_name'] = kr.get('section_name', '')
            out['duration_us'] = kr.get('duration_us', '')
            out['dram_throughput_gbps'] = kr.get('dram_throughput_gbps', '')
            out['sm_throughput_pct'] = kr.get('sm_throughput_pct', '')
            out['mem_throughput_pct'] = kr.get('mem_throughput_pct', '')
            out['compute_throughput_pct'] = kr.get('compute_throughput_pct', '')
            out['l2_hit_rate_pct'] = kr.get('l2_hit_rate_pct', '')
            out['dram_bytes_read'] = kr.get('dram_bytes_read', '')
            out['dram_bytes_write'] = kr.get('dram_bytes_write', '')
            out['roofline_bound_note'] = kr.get('roofline_bound_note', '')
            w.writerow(out)


def cmd_run(args: argparse.Namespace) -> None:
    tools = _tools_dir()
    matrix = _build_run_matrix(args)

    out_root = Path(args.output_root).resolve()
    py = sys.executable
    base_env = os.environ.copy()

    runs_path = Path(args.runs_csv).resolve() if getattr(args, 'runs_csv', None) else None
    if runs_path and getattr(args, 'fresh_runs', False) and runs_path.is_file():
        runs_path.unlink()
        print('[collect_research_metrics] removed existing %s (--fresh_runs)' % runs_path, flush=True)

    run_kitti = not getattr(args, 'skip_kitti_eval', False)
    kitti_cap: int | None = None
    if run_kitti:
        if getattr(args, 'max_eval_samples', None) is not None:
            kitti_cap = int(args.max_eval_samples)
        elif getattr(args, 'accuracy', False):
            kitti_cap = None
        else:
            kitti_cap = int(getattr(args, 'accuracy_false_samples', 200))

    output_csv_path = Path(args.output_csv).resolve() if getattr(args, 'output_csv', None) else None
    buffered_rows: list[dict] = []

    for spec in matrix:
        vname = spec['variant_name']
        use_compile = spec['compile']
        use_amp = spec['amp']
        exp_meta = spec['experiment_meta']
        cell = spec.get('cell')
        use_int8 = bool(cell and cell.get('int8') and not use_compile and not use_amp)
        run_subdir = cell['cell_id'] if cell is not None else vname

        vdir = out_root / run_subdir
        prof_dir = vdir / 'profile'
        eng_dir = vdir / 'energy'
        prof_dir.mkdir(parents=True, exist_ok=True)
        eng_dir.mkdir(parents=True, exist_ok=True)

        mem_s, mem_c = _memory_opts_for_run(spec, args)
        prep_g, prep_cv = _preprocess_opts_for_run(spec, args)
        batch_size_eff = 1 if prep_g else int(args.batch_size)

        note = ''
        cell_id = (cell or {}).get('cell_id', '')
        if use_compile and use_amp:
            note = (
                'M1_AMP: torch.compile + autocast(fp16); use generous --warmup for steady latency/energy '
                '(dynamo may recompile; short runs can skew energy p99).'
            )
        elif cell_id == 'M4_FP32':
            note = (
                'M4_FP32: torch.compile(model) + memory_opt_scatter/conv2d + GPU voxel preprocess + '
                'compile_voxelizer; use generous --warmup (dynamo + compiled voxelizer).'
            )
        elif cell_id == 'M4_AMP':
            note = (
                'M4_AMP: autocast(fp16) + memory_opt_scatter/conv2d + GPU voxel preprocess + '
                'compile_voxelizer; no torch.compile on model (per matrix).'
            )
        elif prep_g and use_amp and not use_compile:
            note = (
                'M3_AMP: GPU voxel preprocessing + autocast(fp16); no torch.compile on model or voxelizer '
                '(default matrix).'
            )
        elif use_compile or use_amp:
            note = 'Single mode: compile or AMP only (not both).'

        if not spec['gpu_runnable']:
            reason = spec['skip_reason'] or (cell and cell.get('status')) or 'not_runnable'
            en = 'No GPU measurement run: %s.' % reason
            if args.eval_notes:
                en = '%s %s' % (args.eval_notes.strip(), en)
            row = build_row(
                variant_name=vname,
                profile_dir=prof_dir,
                energy_dir=eng_dir,
                cfg_file=args.cfg_file,
                ckpt=args.ckpt,
                cuda_id=args.cuda_id,
                workers=args.workers,
                ncu_csv=None,
                map_car_r11=args.map_car_r11,
                eval_notes=en,
                notes_amp_compile=note,
                experiment_meta=exp_meta,
                kitti_eval_json=None,
            )
            if output_csv_path:
                buffered_rows.append(row)
            elif runs_path:
                append_runs_csv(runs_path, row)
            continue

        mb = int(getattr(args, 'measurement_burnin_steps', 0) or 0)

        ps_argv = [
            py, str(tools / 'profile_suite.py'),
            '--cfg_file', args.cfg_file,
            '--ckpt', args.ckpt,
            '--output_dir', str(prof_dir),
            '--warmup', str(args.warmup),
            '--steps', str(args.steps),
            '--batch_size', str(batch_size_eff),
            '--workers', str(args.workers),
            '--cuda_id', str(args.cuda_id),
        ]
        if use_compile:
            ps_argv.append('--compile')
        if use_amp:
            ps_argv.append('--amp')
        if use_int8:
            ps_argv.append('--int8')
        if mem_s:
            ps_argv.append('--memory_opt_scatter')
        if mem_c:
            ps_argv.append('--memory_opt_conv2d')
        if prep_g:
            ps_argv.append('--preprocess_gpu')
        if prep_cv:
            ps_argv.append('--compile_voxelizer')
        if mb > 0:
            ps_argv.extend(['--measurement_burnin_steps', str(mb)])
        pss = getattr(args, 'profile_steady_spike_ms', None)
        if pss is not None:
            ps_argv.extend(['--profile_steady_spike_ms', str(float(pss))])
        _run_cmd(ps_argv, cwd=tools, env=base_env)

        em_argv = [
            py, str(tools / 'energy_monitor.py'),
            '--cfg_file', args.cfg_file,
            '--ckpt', args.ckpt,
            '--output_dir', str(eng_dir),
            '--warmup', str(args.warmup),
            '--steps', str(args.steps),
            '--batch_size', str(batch_size_eff),
            '--workers', str(args.workers),
            '--cuda_id', str(args.cuda_id),
        ]
        if use_compile:
            em_argv.append('--compile')
        ex_spike_always = bool(getattr(args, 'energy_exclude_spikes_always', False))
        if ex_spike_always:
            em_argv.append('--energy_exclude_spikes_always')
        if use_compile or prep_cv or ex_spike_always:
            em_argv.extend([
                '--energy_exclude_compile_over_ms',
                str(float(getattr(args, 'energy_exclude_compile_over_ms', 200.0))),
            ])
        if use_amp:
            em_argv.append('--amp')
        if use_int8:
            em_argv.append('--int8')
        if mem_s:
            em_argv.append('--memory_opt_scatter')
        if mem_c:
            em_argv.append('--memory_opt_conv2d')
        if prep_g:
            em_argv.append('--preprocess_gpu')
        if prep_cv:
            em_argv.append('--compile_voxelizer')
        if mb > 0:
            em_argv.extend(['--measurement_burnin_steps', str(mb)])
        _run_cmd(em_argv, cwd=tools, env=base_env)

        kitti_eval_path: Path | None = None
        eval_notes_merged = _eval_notes_with_subsample(args.eval_notes, kitti_cap, run_kitti)
        if run_kitti:
            kitti_eval_path = vdir / 'kitti_eval_metrics.json'
            ke_argv = [
                py, str(tools / 'kitti_eval_export.py'),
                '--cfg_file', args.cfg_file,
                '--ckpt', args.ckpt,
                '--output_json', str(kitti_eval_path),
                '--eval_result_dir', str(vdir / 'kitti_eval_run'),
                '--cuda_id', str(args.cuda_id),
                '--batch_size', str(batch_size_eff),
                '--workers', str(args.workers),
                '--warmup', str(getattr(args, 'eval_warmup', 20)),
            ]
            if kitti_cap is not None:
                ke_argv.extend(['--max_samples', str(kitti_cap)])
            if use_compile:
                ke_argv.append('--compile')
            if use_amp:
                ke_argv.append('--amp')
            if use_int8:
                ke_argv.append('--int8')
            if mem_s:
                ke_argv.append('--memory_opt_scatter')
            if mem_c:
                ke_argv.append('--memory_opt_conv2d')
            if prep_g:
                ke_argv.append('--preprocess_gpu')
            if prep_cv:
                ke_argv.append('--compile_voxelizer')
            _run_cmd(ke_argv, cwd=tools, env=base_env)

        row = build_row(
            variant_name=vname,
            profile_dir=prof_dir,
            energy_dir=eng_dir,
            cfg_file=args.cfg_file,
            ckpt=args.ckpt,
            cuda_id=args.cuda_id,
            workers=args.workers,
            ncu_csv=None,
            map_car_r11=args.map_car_r11,
            eval_notes=eval_notes_merged,
            notes_amp_compile=note,
            experiment_meta=exp_meta,
            kitti_eval_json=kitti_eval_path if run_kitti else None,
        )
        if output_csv_path:
            buffered_rows.append(row)
        elif runs_path:
            append_runs_csv(runs_path, row)

    if output_csv_path:
        write_runs_csv_overwrite(output_csv_path, buffered_rows)
    elif not runs_path:
        raise SystemExit('Set --output_csv or let runs_csv default to <output_root>/runs.csv.')


def cmd_matrix(args: argparse.Namespace) -> None:
    """Export the full M0–M4 × (FP32, AMP) table to CSV for reports."""
    out = Path(args.output_csv).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'cell_id', 'model_variant', 'precision_mode', 'variant_name', 'status',
        'compile', 'amp', 'int8', 'nhwc', 'preprocess_gpu', 'compile_voxelizer',
        'memory_opt_scatter', 'memory_opt_conv2d',
        'skip_reason',
    ]
    cells = exp15.experiment_matrix_fp32_amp()
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for cell in cells:
            row = {k: cell.get(k, '') for k in fieldnames}
            row['compile'] = str(cell.get('compile', False)).lower()
            row['amp'] = str(cell.get('amp', False)).lower()
            row['int8'] = str(cell.get('int8', False)).lower()
            row['nhwc'] = str(cell.get('nhwc', False)).lower()
            row['preprocess_gpu'] = str(cell.get('preprocess_gpu', False)).lower()
            row['compile_voxelizer'] = str(cell.get('compile_voxelizer', False)).lower()
            row['memory_opt_scatter'] = str(cell.get('memory_opt_scatter', False)).lower()
            row['memory_opt_conv2d'] = str(cell.get('memory_opt_conv2d', False)).lower()
            w.writerow(row)
    print('[collect_research_metrics] Wrote %s (%d rows)' % (out, len(cells)), flush=True)
    print('M4 definition:', exp15.M4_ALL_APPLIED_DEFINITION, flush=True)


def cmd_merge(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text())
    if not isinstance(data, list):
        raise SystemExit('Manifest must be a JSON array')
    for entry in data:
        vname = entry['variant_name']
        prof_dir = Path(entry['profile_dir'])
        eng_dir = Path(entry['energy_dir'])
        ncu = Path(entry['ncu_csv']) if entry.get('ncu_csv') else None
        cid = entry.get('cuda_id')
        wid = entry.get('num_workers')
        kjp = entry.get('kitti_eval_json')
        kitti_path = Path(kjp) if kjp else None
        row = build_row(
            variant_name=vname,
            profile_dir=prof_dir,
            energy_dir=eng_dir,
            cfg_file=entry.get('config_path', ''),
            ckpt=entry.get('checkpoint_path', ''),
            cuda_id=int(cid) if cid is not None else None,
            workers=int(wid) if wid is not None else None,
            ncu_csv=ncu,
            map_car_r11=str(entry.get('map_car_r11', '')),
            eval_notes=str(entry.get('eval_protocol_notes', '')),
            notes_amp_compile=str(entry.get('notes_amp_compile_mutually_exclusive', '')),
            kitti_eval_json=kitti_path,
        )
        if entry.get('batch_size'):
            row['batch_size'] = str(entry['batch_size'])
        append_runs_csv(Path(args.runs_csv).resolve(), row)
        if ncu:
            _, krows = _parse_ncu_csv(ncu)
            append_ncu_kernels_csv(
                Path(args.ncu_kernels_csv).resolve(),
                row['run_id'],
                vname,
                krows,
            )


def main():
    parser = argparse.ArgumentParser(description='Collect PointPillars research metrics into CSV')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='Run profile_suite + energy_monitor for each variant')
    p_run.add_argument(
        '--cfg_file',
        default='cfgs/kitti_models/pointpillar.yaml',
        help='YAML config (relative to OpenPCDet/tools/ or absolute)',
    )
    p_run.add_argument(
        '--ckpt',
        default=None,
        help='Checkpoint .pth; if omitted, uses env vars then tools/ckpt/pointpillar_7728.pth',
    )
    p_run.add_argument(
        '--output_root',
        default=None,
        help='Default: MLS/profile_outputs/research_matrix or OpenPCDet/profile_outputs/research_matrix',
    )
    p_run.add_argument('--runs_csv', default=None,
                       help='Append rows here (default: <output_root>/runs.csv)')
    p_run.add_argument('--cuda_id', type=int, default=0)
    p_run.add_argument('--warmup', type=int, default=100)
    p_run.add_argument('--steps', type=int, default=50)
    p_run.add_argument('--batch_size', type=int, default=1)
    p_run.add_argument('--workers', type=int, default=2)
    p_run.add_argument('--variants', default='all',
                       help='Comma-separated variant names or "all"')
    p_run.add_argument('--map_car_r11', default='', help='Optional: fill accuracy column')
    p_run.add_argument('--eval_notes', default='', help='Optional eval protocol string')
    p_run.add_argument(
        '--matrix',
        choices=('legacy', 'fp32_amp', '15'),
        default='legacy',
        help='legacy = 3 runs; fp32_amp = M0–M4 × (FP32, AMP) (15 is a deprecated alias)',
    )
    p_run.add_argument(
        '--cell',
        default=None,
        help='With --matrix fp32_amp: cell id(s), e.g. M0_FP32,M1_AMP (legacy M1_P1-style also accepted)',
    )
    p_run.add_argument(
        '--m',
        type=int,
        default=None,
        help='With --matrix fp32_amp: model index 0–4 (M0=baseline; use with --p)',
    )
    p_run.add_argument(
        '--p',
        type=int,
        default=None,
        help='With --matrix fp32_amp: 1=FP32, 2=AMP (use with --m)',
    )
    p_run.add_argument(
        '--all-cells-15',
        action='store_true',
        dest='all_cells_15',
        help='With --matrix fp32_amp: include all 10 design cells (non-GPU cells get placeholder rows)',
    )
    p_run.add_argument(
        '--output-csv', '--output_csv',
        dest='output_csv',
        default=None,
        help='Write only this invocation’s row(s) to this path (overwrite). Does not append to runs.csv.',
    )
    p_run.add_argument(
        '--memory_opt_scatter',
        action='store_true',
        default=False,
        help='HWC scatter; forwarded to profile_suite / energy_monitor / kitti_eval_export',
    )
    p_run.add_argument(
        '--memory_opt_conv2d',
        action='store_true',
        default=False,
        help='channels_last BEV path; forwarded to profile_suite / energy_monitor / kitti_eval_export',
    )
    p_run.add_argument(
        '--preprocess_gpu',
        action='store_true',
        default=False,
        help='GPU voxelization (inference.py path); forwarded to profile / energy / kitti_eval',
    )
    p_run.add_argument(
        '--compile_voxelizer',
        action='store_true',
        default=False,
        help='torch.compile voxelizer with --preprocess_gpu',
    )
    p_run.add_argument(
        '--accuracy',
        action='store_true',
        default=False,
        help='KITTI: evaluate full val split for AP. Default (off): subsample (see --accuracy-false-samples).',
    )
    p_run.add_argument(
        '--accuracy-false-samples', '--accuracy_false_samples',
        type=int,
        default=200,
        dest='accuracy_false_samples',
        metavar='N',
        help='When --accuracy is not set: limit KITTI eval to N val samples (default: 200)',
    )
    p_run.add_argument(
        '--skip_kitti_eval',
        action='store_true',
        help='Skip KITTI eval entirely (accuracy columns empty unless --map_car_r11)',
    )
    p_run.add_argument(
        '--max_eval_samples',
        type=int,
        default=None,
        help='KITTI: cap val samples (overrides --accuracy and --accuracy-false-samples when set)',
    )
    p_run.add_argument(
        '--eval_warmup',
        type=int,
        default=20,
        help='Warmup batches inside KITTI eval loop before infer_time metering (default: 20)',
    )
    p_run.add_argument(
        '--fresh_runs',
        action='store_true',
        help='Delete runs.csv before appending new rows (clean re-run)',
    )
    p_run.add_argument(
        '--energy-exclude-compile-over-ms',
        type=float,
        default=200.0,
        dest='energy_exclude_compile_over_ms',
        metavar='MS',
        help='With --compile / --compile_voxelizer / --energy-exclude-spikes-always: drop forwards slower '
        'than MS from NVML energy and samples/J (default: 200; tune via k× steady p99).',
    )
    p_run.add_argument(
        '--measurement-burnin-steps',
        type=int,
        default=0,
        dest='measurement_burnin_steps',
        metavar='N',
        help='After warmup, N extra forwards excluded from NVML window (energy) and before timed '
        'profile loop (profile_suite); parity between tools.',
    )
    p_run.add_argument(
        '--energy-exclude-spikes-always',
        action='store_true',
        dest='energy_exclude_spikes_always',
        help='Pass through to energy_monitor: spike exclusion even without torch.compile',
    )
    p_run.add_argument(
        '--profile-steady-spike-ms',
        type=float,
        default=None,
        dest='profile_steady_spike_ms',
        metavar='MS',
        help='Forwarded to profile_suite: forward ms above MS drops step from Peak GPU Mem (steady)',
    )

    p_matrix = sub.add_parser('matrix', help='Export M0–M4 × (FP32, AMP) experiment matrix CSV')
    p_matrix.add_argument(
        '--output_csv',
        required=True,
        help='Path to write experiment_matrix_fp32_amp.csv (or any name)',
    )

    p_merge = sub.add_parser('merge', help='Merge existing dirs using a JSON manifest')
    p_merge.add_argument('--manifest', required=True)
    p_merge.add_argument('--runs_csv', required=True)
    p_merge.add_argument('--ncu_kernels_csv', default=None,
                         help='Append kernel rows (default: sibling of runs_csv named ncu_kernels.csv)')

    args = parser.parse_args()
    if args.cmd == 'run':
        _prepare_run_paths(args)
        if getattr(args, 'output_csv', None):
            args.runs_csv = None
        elif args.runs_csv is None:
            args.runs_csv = str(Path(args.output_root) / 'runs.csv')
        cmd_run(args)
    elif args.cmd == 'matrix':
        cmd_matrix(args)
    else:
        if args.ncu_kernels_csv is None:
            rp = Path(args.runs_csv)
            args.ncu_kernels_csv = str(rp.parent / 'ncu_kernels.csv')
        cmd_merge(args)


if __name__ == '__main__':
    main()
