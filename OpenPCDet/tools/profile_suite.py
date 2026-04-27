"""
profile_suite.py — comprehensive profiling for PointPillars / OpenPCDet.

Captures per-stage latency (dataloader, H2D, VFE+scatter, backbone+head, postprocess),
throughput (samples/s), peak GPU memory, kernel launch count, and exports a
Chrome-trace JSON and a plain-text summary.

Warmup vs measurement:
  --warmup: steps before measurement; excluded from summary stats and from peak-memory window
    (peak stats reset after warmup). Matches energy_monitor.py: energy NVML window starts only
    after warmup (and optional measurement burn-in there).
  --measurement_burnin_steps: extra forwards after reset_peak_memory, still before the timed
    profiler loop; aligns with energy_monitor --measurement_burnin_steps (excludes that window
    from NVML integration).

Run from OpenPCDet/tools/:
    python profile_suite.py \
        --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt /path/to/checkpoint.pth \
        --output_dir /path/to/profile_outputs/suite_run \
        [--warmup 100] [--steps 30] [--batch_size 1] [--workers 4] \
        [--traced_model /path/to/model.pt] \
        [--compile] [--amp] [--cuda_id 1]

Outputs (all in --output_dir):
    profile_summary.txt        — human-readable stage table + throughput + memory
    latency_per_step.csv       — per-step stage latencies (for p95/p99 / distribution plots)
    torch_profile_trace.json   — Chrome trace (open in chrome://tracing or Perfetto)
"""

import _init_path  # noqa: F401  adds pcdet to sys.path
import argparse
import csv
import datetime
import os
import time
from contextlib import nullcontext
from pathlib import Path

# ── CUDA device selection must happen before any torch import ──────────────
def _early_cuda_id():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--cuda_id', type=int, default=1)
    a, _ = p.parse_known_args()
    return a.cuda_id

os.environ['CUDA_VISIBLE_DEVICES'] = str(_early_cuda_id())

import numpy as np
import torch
import torch.profiler

from eval_utils.eval_utils import _nvtx_range
from model_loader import load_model_for_inference
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from preprocess_gpu_loop import (
    build_compiled_voxelize_fn,
    dataset_supports_gpu_voxel,
    gpu_voxelize_and_build_batch_dict,
    numpy_points_to_cuda,
    resolve_voxel_params,
)
from int8_utils import load_batch_to_device
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='PointPillars profile suite')
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='directory for outputs; default: profile_outputs/suite_<timestamp>')
    parser.add_argument('--warmup', type=int, default=100,
                        help='warmup steps before measurement (default: 100)')
    parser.add_argument('--steps', type=int, default=30,
                        help='measured steps (default: 30)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--traced_model', type=str, default=None,
                        help='path to TorchScript .pt (profile_utils/export.py)')
    parser.add_argument('--trt_engine', type=str, default=None,
                        help='path to TRT .engine file (profile_utils/export_onnx.py + trtexec)')
    parser.add_argument('--compile', action='store_true',
                        help='wrap model with torch.compile() before profiling')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable mixed-precision inference with fp16 autocast')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='CPU dynamic PTQ (nn.Linear int8); Conv stays float CPU; no GPU forward')
    parser.add_argument(
        '--memory_opt_scatter',
        action='store_true',
        default=False,
        help='HWC coalesced PointPillar scatter write (default off = legacy C×HW scatter)',
    )
    parser.add_argument(
        '--memory_opt_conv2d',
        action='store_true',
        default=False,
        help='channels_last on model + BEV 2D path (default off)',
    )
    parser.add_argument(
        '--preprocess_gpu',
        action='store_true',
        default=False,
        help='GPU voxelization path aligned with inference.py (requires KITTI; batch_size=1)',
    )
    parser.add_argument(
        '--compile_voxelizer',
        action='store_true',
        default=False,
        help='with --preprocess_gpu, torch.compile(dynamic=True) the voxelizer (forces num_workers=0)',
    )
    parser.add_argument('--cuda_id', type=int, default=1)
    parser.add_argument(
        '--measurement_burnin_steps',
        type=int,
        default=0,
        help='After warmup and peak-memory reset, run this many extra forwards before the '
        'measured profiler loop (parity with energy_monitor --measurement_burnin_steps).',
    )
    parser.add_argument(
        '--profile_steady_spike_ms',
        type=float,
        default=None,
        help='Per-step forward latency above this (ms) excludes that step from Peak GPU Memory '
        '(steady). Default: 200 if --compile or --compile_voxelizer else no filtering.',
    )
    parser.add_argument(
        '--nsight',
        action='store_true',
        default=False,
        help='Emit NVTX ranges (same names as inference.py); use with nsys profile ...',
    )
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


# ── Timing helpers ─────────────────────────────────────────────────────────

class CudaTimer:
    """Measures GPU-side elapsed time with CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self._elapsed_ms = None

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *_):
        self.end_event.record()
        torch.cuda.synchronize()
        self._elapsed_ms = self.start_event.elapsed_time(self.end_event)

    @property
    def ms(self):
        return self._elapsed_ms


class WallTimer:
    """Wall-clock stage timer (for CPU INT8 path)."""

    def __init__(self):
        self._elapsed_ms = 0.0
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._elapsed_ms = (time.perf_counter() - self._t0) * 1e3

    @property
    def ms(self):
        return self._elapsed_ms


class StageAccumulator:
    """Accumulates per-stage timings across steps."""

    def __init__(self, names):
        self.names = names
        self._data = {n: [] for n in names}

    def record(self, name, ms):
        self._data[name].append(ms)

    def stats(self, name):
        vals = np.array(self._data[name])
        if len(vals) == 0:
            return dict(mean=0.0, p50=0.0, p95=0.0, p99=0.0, std=0.0, n=0)
        return dict(
            mean=float(np.mean(vals)),
            p50=float(np.percentile(vals, 50)),
            p95=float(np.percentile(vals, 95)),
            p99=float(np.percentile(vals, 99)),
            std=float(np.std(vals)),
            n=len(vals),
        )

    def all_stats(self):
        return {n: self.stats(n) for n in self.names}


def _steady_forward_threshold_ms(args) -> float:
    """Forward duration above this is treated as a compile spike for steady peak memory."""
    v = getattr(args, 'profile_steady_spike_ms', None)
    if v is not None:
        return float(v)
    if getattr(args, 'compile', False) or getattr(args, 'compile_voxelizer', False):
        return 200.0
    return float('inf')


def _maybe_nvtx(args, name: str):
    if getattr(args, 'nsight', False):
        return _nvtx_range(name)
    return nullcontext()


def _peak_steady_mb(peak_mem_mb_steps: list, forward_ms_steps: list, thr_ms: float) -> float:
    if not peak_mem_mb_steps:
        return 0.0
    if not forward_ms_steps or len(forward_ms_steps) != len(peak_mem_mb_steps):
        return float(max(peak_mem_mb_steps))
    steady = [pm for pm, fw in zip(peak_mem_mb_steps, forward_ms_steps) if fw <= thr_ms]
    return float(max(steady)) if steady else float(max(peak_mem_mb_steps))


# ── Model forward wrapper ──────────────────────────────────────────────────

def forward_model(model, batch_dict, args):
    if getattr(args, 'int8', False):
        with torch.inference_mode():
            return model(batch_dict)
    amp_enabled = bool(getattr(args, 'amp', False)) and torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled) \
        if amp_enabled else nullcontext()
    if not getattr(args, 'compile', False):
        with amp_ctx:
            return model(batch_dict)
    safe = {k: v for k, v in batch_dict.items() if isinstance(v, (torch.Tensor, int, float))}
    with amp_ctx:
        pred_dicts, ret_dict = model(safe)
    batch_dict.update(safe)
    return pred_dicts, ret_dict


# ── Core profiling loop ────────────────────────────────────────────────────

STAGE_NAMES = ['dataloader', 'h2d', 'forward', 'postprocess', 'full_frame']

RT_STAGES_KITTI_CPU = [
    'read_points', 'pre_processing', 'h2d', 'forward', 'postprocess', 'full_frame', 'T_rt',
]
RT_STAGES_PREPROCESS_GPU = [
    'read_points', 'cpu_prepare', 'data_to_gpu', 'pre_processing', 'h2d',
    'forward', 'postprocess', 'full_frame', 'T_rt',
]


def _dataloader_next_cyclic(dataloader_iter, dataloader):
    try:
        return next(dataloader_iter), dataloader_iter
    except StopIteration:
        dataloader_iter = iter(dataloader)
        return next(dataloader_iter), dataloader_iter


def run_profile_loop(model, dataloader, dataset, class_names, args, logger, result_dir):
    model.eval()
    accum = StageAccumulator(STAGE_NAMES)
    dataloader_iter = iter(dataloader)
    cpu_int8 = bool(getattr(args, 'int8', False))
    TimerCls = WallTimer if cpu_int8 else CudaTimer

    def _to_compute_device(batch_dict):
        if cpu_int8:
            load_batch_to_device(batch_dict, torch.device('cpu'))
        else:
            load_data_to_gpu(batch_dict)

    if args.warmup > 0:
        logger.info('Running %d warmup steps...', args.warmup)
    for _ in range(args.warmup):
        batch_dict, dataloader_iter = _dataloader_next_cyclic(dataloader_iter, dataloader)
        _to_compute_device(batch_dict)
        with torch.inference_mode():
            forward_model(model, batch_dict, args)
    if not cpu_int8:
        torch.cuda.synchronize()
    dataloader_iter = iter(dataloader)

    peak_mem_mb_steps = []
    forward_ms_steps = []
    torch.cuda.reset_peak_memory_stats()

    burn = int(getattr(args, 'measurement_burnin_steps', 0) or 0)
    if burn > 0:
        logger.info('Measurement burn-in: %d steps (before profiler loop)', burn)
        for _ in range(burn):
            batch_dict, dataloader_iter = _dataloader_next_cyclic(dataloader_iter, dataloader)
            _to_compute_device(batch_dict)
            with torch.inference_mode():
                forward_model(model, batch_dict, args)
        if not cpu_int8:
            torch.cuda.synchronize()

    steps = min(args.steps, len(dataloader))
    logger.info('Profiling %d steps...', steps)

    trace_path = str(result_dir / 'torch_profile_trace.json')
    activities = [torch.profiler.ProfilerActivity.CPU]
    if not cpu_int8:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        on_trace_ready=None,
    ) as prof:
        for step_idx in range(steps):
            t_dl_start = time.perf_counter()
            with torch.profiler.record_function('dataloader'):
                batch_dict, dataloader_iter = _dataloader_next_cyclic(dataloader_iter, dataloader)
            dl_ms = (time.perf_counter() - t_dl_start) * 1e3

            frame_start = time.perf_counter()

            with TimerCls() as h2d_timer:
                with torch.profiler.record_function('h2d'):
                    _to_compute_device(batch_dict)

            with TimerCls() as fwd_timer:
                with torch.profiler.record_function('forward'):
                    with torch.inference_mode():
                        pred_dicts, ret_dict = forward_model(model, batch_dict, args)

            t_post_start = time.perf_counter()
            with torch.profiler.record_function('postprocess'):
                _ = dataset.generate_prediction_dicts(
                    batch_dict, pred_dicts, class_names, output_path=None
                )
            post_ms = (time.perf_counter() - t_post_start) * 1e3

            if not cpu_int8:
                torch.cuda.synchronize()
            frame_ms = (time.perf_counter() - frame_start) * 1e3

            accum.record('dataloader', dl_ms)
            accum.record('h2d', h2d_timer.ms)
            accum.record('forward', fwd_timer.ms)
            accum.record('postprocess', post_ms)
            accum.record('full_frame', frame_ms)

            if cpu_int8:
                torch.cuda.synchronize()
            peak_mem_mb_steps.append(torch.cuda.max_memory_allocated() / 1024**2)
            forward_ms_steps.append(float(fwd_timer.ms))
            prof.step()

    prof.export_chrome_trace(trace_path)
    logger.info('Chrome trace saved → %s', trace_path)

    _write_latency_per_step_csv(result_dir, accum)
    return accum, peak_mem_mb_steps, trace_path, forward_ms_steps


def run_profile_loop_preprocess_gpu(model, dataset, class_names, args, logger, result_dir, voxel_params, compiled_voxelize):
    """
    GPU voxel path with NVTX-aligned stages (inference.py): read_points, cpu_prepare,
    data_to_gpu (points), pre_processing (voxel+build), h2d (load_data_to_gpu), forward, postprocess.
    T_rt = onboard sum excluding read_points.
    """
    model.eval()
    accum = StageAccumulator(RT_STAGES_PREPROCESS_GPU)
    n_ds = len(dataset)
    if n_ds == 0:
        raise ValueError('empty dataset')
    TimerCls = CudaTimer

    def _one_frame(idx: int):
        input_dict, img_shape = dataset.build_input_dict(idx)
        cpu_dict = dataset.prepare_data_for_gpu_voxelization(input_dict)
        cpu_dict['image_shape'] = img_shape
        work = dict(cpu_dict)
        points_np = work.pop('points', None)
        points_gpu = numpy_points_to_cuda(points_np)
        batch_dict = gpu_voxelize_and_build_batch_dict(
            points_gpu, work, voxel_params, compiled_voxelize, batch_size=1,
        )
        load_data_to_gpu(batch_dict)
        with torch.inference_mode():
            forward_model(model, batch_dict, args)

    if args.warmup > 0:
        logger.info('Running %d warmup steps...', args.warmup)
    for i in range(args.warmup):
        _one_frame(i % n_ds)
    torch.cuda.synchronize()

    peak_mem_mb_steps = []
    forward_ms_steps = []
    torch.cuda.reset_peak_memory_stats()

    burn = int(getattr(args, 'measurement_burnin_steps', 0) or 0)
    if burn > 0:
        logger.info('Measurement burn-in: %d steps (before profiler loop)', burn)
        for i in range(burn):
            _one_frame(i % n_ds)
        torch.cuda.synchronize()

    steps = min(args.steps, n_ds)
    logger.info('Profiling %d steps...', steps)

    trace_path = str(result_dir / 'torch_profile_trace.json')
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        on_trace_ready=None,
    ) as prof:
        for step_idx in range(steps):
            idx = step_idx % n_ds

            t_rp0 = time.perf_counter()
            with torch.profiler.record_function('read_points'):
                with _maybe_nvtx(args, 'read_points'):
                    input_dict, img_shape = dataset.build_input_dict(idx)
            read_ms = (time.perf_counter() - t_rp0) * 1e3

            t_cpu0 = time.perf_counter()
            with torch.profiler.record_function('cpu_prepare'):
                cpu_dict = dataset.prepare_data_for_gpu_voxelization(input_dict)
                cpu_dict['image_shape'] = img_shape
            cpu_prep_ms = (time.perf_counter() - t_cpu0) * 1e3

            work = dict(cpu_dict)
            points_np = work.pop('points', None)

            t_d2g0 = time.perf_counter()
            with torch.profiler.record_function('data_to_gpu'):
                with _maybe_nvtx(args, 'data_to_gpu'):
                    points_gpu = numpy_points_to_cuda(points_np)
            d2g_ms = (time.perf_counter() - t_d2g0) * 1e3

            t_vox0 = time.perf_counter()
            with torch.profiler.record_function('pre_processing'):
                with _maybe_nvtx(args, 'pre_processing'):
                    batch_dict = gpu_voxelize_and_build_batch_dict(
                        points_gpu, work, voxel_params, compiled_voxelize, batch_size=1,
                    )
            vox_ms = (time.perf_counter() - t_vox0) * 1e3

            frame_start = time.perf_counter()

            with TimerCls() as h2d_timer:
                with torch.profiler.record_function('h2d'):
                    load_data_to_gpu(batch_dict)

            with TimerCls() as fwd_timer:
                with torch.profiler.record_function('forward'):
                    with _maybe_nvtx(args, 'forward'):
                        with torch.inference_mode():
                            pred_dicts, ret_dict = forward_model(model, batch_dict, args)

            t_post_start = time.perf_counter()
            with torch.profiler.record_function('postprocess'):
                _ = dataset.generate_prediction_dicts(
                    batch_dict, pred_dicts, class_names, output_path=None
                )
            post_ms = (time.perf_counter() - t_post_start) * 1e3

            torch.cuda.synchronize()
            frame_ms = (time.perf_counter() - frame_start) * 1e3

            t_rt = cpu_prep_ms + d2g_ms + vox_ms + h2d_timer.ms + fwd_timer.ms + post_ms

            accum.record('read_points', read_ms)
            accum.record('cpu_prepare', cpu_prep_ms)
            accum.record('data_to_gpu', d2g_ms)
            accum.record('pre_processing', vox_ms)
            accum.record('h2d', h2d_timer.ms)
            accum.record('forward', fwd_timer.ms)
            accum.record('postprocess', post_ms)
            accum.record('full_frame', frame_ms)
            accum.record('T_rt', t_rt)

            peak_mem_mb_steps.append(torch.cuda.max_memory_allocated() / 1024**2)
            forward_ms_steps.append(float(fwd_timer.ms))
            prof.step()

    prof.export_chrome_trace(trace_path)
    logger.info('Chrome trace saved → %s', trace_path)

    _write_latency_per_step_csv(result_dir, accum)
    return accum, peak_mem_mb_steps, trace_path, forward_ms_steps


def run_profile_loop_kitti_rt_cpu(model, dataset, class_names, args, logger, result_dir):
    """KITTI CPU-voxel path: read_points vs pre_processing vs h2d (inference.py NVTX names)."""
    model.eval()
    accum = StageAccumulator(RT_STAGES_KITTI_CPU)
    n_ds = len(dataset)
    if n_ds == 0:
        raise ValueError('empty dataset')
    TimerCls = CudaTimer

    def _one_frame(idx: int):
        input_dict, img_shape = dataset.build_input_dict(idx)
        data_dict = dataset.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = img_shape
        batch_dict = dataset.collate_batch([data_dict])
        load_data_to_gpu(batch_dict)
        with torch.inference_mode():
            forward_model(model, batch_dict, args)

    if args.warmup > 0:
        logger.info('Running %d warmup steps...', args.warmup)
    for i in range(args.warmup):
        _one_frame(i % n_ds)
    torch.cuda.synchronize()

    peak_mem_mb_steps = []
    forward_ms_steps = []
    torch.cuda.reset_peak_memory_stats()

    burn = int(getattr(args, 'measurement_burnin_steps', 0) or 0)
    if burn > 0:
        logger.info('Measurement burn-in: %d steps (before profiler loop)', burn)
        for i in range(burn):
            _one_frame(i % n_ds)
        torch.cuda.synchronize()

    steps = min(args.steps, n_ds)
    logger.info('Profiling %d steps (KITTI RT CPU stages)...', steps)

    trace_path = str(result_dir / 'torch_profile_trace.json')
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        on_trace_ready=None,
    ) as prof:
        for step_idx in range(steps):
            idx = step_idx % n_ds

            t_rp0 = time.perf_counter()
            with torch.profiler.record_function('read_points'):
                with _maybe_nvtx(args, 'read_points'):
                    input_dict, img_shape = dataset.build_input_dict(idx)
            read_ms = (time.perf_counter() - t_rp0) * 1e3

            t_prep0 = time.perf_counter()
            with torch.profiler.record_function('pre_processing'):
                with _maybe_nvtx(args, 'pre_processing'):
                    data_dict = dataset.prepare_data(data_dict=input_dict)
                    data_dict['image_shape'] = img_shape
                    batch_dict = dataset.collate_batch([data_dict])
            prep_ms = (time.perf_counter() - t_prep0) * 1e3

            frame_start = time.perf_counter()

            with TimerCls() as h2d_timer:
                with torch.profiler.record_function('h2d'):
                    with _maybe_nvtx(args, 'data_to_gpu'):
                        load_data_to_gpu(batch_dict)

            with TimerCls() as fwd_timer:
                with torch.profiler.record_function('forward'):
                    with _maybe_nvtx(args, 'forward'):
                        with torch.inference_mode():
                            pred_dicts, ret_dict = forward_model(model, batch_dict, args)

            t_post_start = time.perf_counter()
            with torch.profiler.record_function('postprocess'):
                _ = dataset.generate_prediction_dicts(
                    batch_dict, pred_dicts, class_names, output_path=None
                )
            post_ms = (time.perf_counter() - t_post_start) * 1e3

            torch.cuda.synchronize()
            frame_ms = (time.perf_counter() - frame_start) * 1e3

            t_rt = prep_ms + h2d_timer.ms + fwd_timer.ms + post_ms

            accum.record('read_points', read_ms)
            accum.record('pre_processing', prep_ms)
            accum.record('h2d', h2d_timer.ms)
            accum.record('forward', fwd_timer.ms)
            accum.record('postprocess', post_ms)
            accum.record('full_frame', frame_ms)
            accum.record('T_rt', t_rt)

            peak_mem_mb_steps.append(torch.cuda.max_memory_allocated() / 1024**2)
            forward_ms_steps.append(float(fwd_timer.ms))
            prof.step()

    prof.export_chrome_trace(trace_path)
    logger.info('Chrome trace saved → %s', trace_path)

    _write_latency_per_step_csv(result_dir, accum)
    return accum, peak_mem_mb_steps, trace_path, forward_ms_steps


def _write_latency_per_step_csv(result_dir, accum):
    """Export raw per-step latencies for distribution plots (p95/p99 from full sample)."""
    path = result_dir / 'latency_per_step.csv'
    names = list(accum.names)
    n = len(accum._data[names[0]])
    rows = []
    for i in range(n):
        rows.append([i] + [accum._data[s][i] for s in names])
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step_idx'] + names)
        w.writerows(rows)


# ── Kernel launch count from Chrome trace ─────────────────────────────────

def count_kernel_launches(trace_path):
    """Count CUDA kernel launch events in the exported Chrome trace JSON."""
    import json
    try:
        with open(trace_path) as f:
            data = json.load(f)
        events = data.get('traceEvents', [])
        cuda_launches = sum(
            1 for e in events
            if e.get('cat', '') in ('cuda_runtime', 'kernel')
            and e.get('name', '').lower() not in ('', 'process_name', 'thread_name')
        )
        return cuda_launches
    except Exception:
        return -1


# ── Summary writer ─────────────────────────────────────────────────────────

def write_summary(accum, peak_mem_mb_steps, batch_size, steps, trace_path,
                  result_dir, logger, args, forward_ms_steps=None):
    stats = accum.all_stats()
    total_samples = steps * batch_size

    rt_mode = 'T_rt' in accum.names
    if rt_mode:
        mean_frame_ms = stats['T_rt']['mean']
    else:
        mean_frame_ms = stats['full_frame']['mean']
    throughput = (batch_size / mean_frame_ms * 1e3) if mean_frame_ms > 0 else 0.0

    peak_mem_overall = max(peak_mem_mb_steps) if peak_mem_mb_steps else 0.0
    mean_peak_mem = np.mean(peak_mem_mb_steps) if peak_mem_mb_steps else 0.0
    thr_steady = _steady_forward_threshold_ms(args)
    peak_mem_steady = _peak_steady_mb(peak_mem_mb_steps, forward_ms_steps or [], thr_steady)

    kernel_count = count_kernel_launches(trace_path)

    lines = []
    lines.append('=' * 70)
    lines.append('  PointPillars Profile Suite — %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('=' * 70)
    lines.append('  Config  : %s' % args.cfg_file)
    lines.append('  Ckpt    : %s' % args.ckpt)
    lines.append('  Compile : %s' % getattr(args, 'compile', False))
    lines.append('  AMP (fp16): %s' % bool(getattr(args, 'amp', False)))
    lines.append('  Memory opt scatter (HWC write): %s' % bool(getattr(args, 'memory_opt_scatter', False)))
    lines.append('  Memory opt conv2d (channels_last): %s' % bool(getattr(args, 'memory_opt_conv2d', False)))
    lines.append('  Preprocess GPU (voxelize on GPU): %s' % bool(getattr(args, 'preprocess_gpu', False)))
    lines.append('  Compile voxelizer: %s' % bool(getattr(args, 'compile_voxelizer', False)))
    lines.append('  INT8    : %s  (CPU dynamic Linear PTQ if true)' % bool(getattr(args, 'int8', False)))
    lines.append('  Traced  : %s' % (args.traced_model or 'none'))
    lines.append('  Nsight NVTX: %s' % bool(getattr(args, 'nsight', False)))
    lines.append(
        '  Warmup  : %d steps   Measured: %d steps   Batch: %d   Meas. burn-in: %d' % (
            args.warmup, steps, batch_size, int(getattr(args, 'measurement_burnin_steps', 0) or 0),
        )
    )
    lines.append('')
    lines.append('── Per-stage latency (mean / p50 / p95 / p99 / std) [ms] ─────────────')

    col_w = 12
    header = ('%-16s' % 'Stage') + ''.join(('%*s' % (col_w, h)) for h in ['mean', 'p50', 'p95', 'p99', 'std'])
    lines.append(header)
    lines.append('-' * (16 + col_w * 5))

    fwd_lbl = 'Forward (CPU+GPU)' if getattr(args, 'int8', False) else 'Forward (GPU)'
    if rt_mode and 'cpu_prepare' in accum.names:
        stage_display = [
            ('read_points', 'read_points (excl.T_rt)'),
            ('cpu_prepare', 'cpu_prepare'),
            ('data_to_gpu', 'data_to_gpu'),
            ('pre_processing', 'pre_processing'),
            ('h2d', 'h2d (load rest)'),
            ('forward', fwd_lbl),
            ('postprocess', 'PostProcess'),
            ('full_frame', '── Onboard frame ──'),
            ('T_rt', '── T_rt (onboard) ──'),
        ]
    elif rt_mode:
        stage_display = [
            ('read_points', 'read_points (excl.T_rt)'),
            ('pre_processing', 'pre_processing'),
            ('h2d', 'data_to_gpu'),
            ('forward', fwd_lbl),
            ('postprocess', 'PostProcess'),
            ('full_frame', '── Onboard frame ──'),
            ('T_rt', '── T_rt (onboard) ──'),
        ]
    else:
        stage_display = [
            ('dataloader', 'DataLoader (CPU)'),
            ('h2d', 'H2D Transfer'),
            ('forward', fwd_lbl),
            ('postprocess', 'PostProcess'),
            ('full_frame', '── Full Frame ──'),
        ]
    for key, label in stage_display:
        s = stats[key]
        row = ('%-16s' % label) + ''.join(
            ('%*.2f' % (col_w, s[f])) for f in ['mean', 'p50', 'p95', 'p99', 'std']
        )
        lines.append(row)

    lines.append('')
    lines.append('── Throughput & Memory ─────────────────────────────────────────')
    tp_note = 'mean T_rt' if rt_mode else 'mean full_frame'
    lines.append('  Throughput         : %.1f samples/s  (batch=%d, %s=%.2f ms)' % (
        throughput, batch_size, tp_note, mean_frame_ms))
    lines.append(
        '  Peak GPU Memory    : %.1f MB  (NMS/recall on GPU; backbone mostly CPU when INT8)'
        % peak_mem_overall
    )
    if thr_steady < float('inf'):
        lines.append(
            '  Peak GPU Mem (steady): %.1f MB  (steps with forward <= %.1f ms)' % (
                peak_mem_steady, thr_steady,
            )
        )
    else:
        lines.append('  Peak GPU Mem (steady): %.1f MB  (no forward spike filter)' % peak_mem_steady)
    lines.append('  Mean peak per step : %.1f MB' % mean_peak_mem)
    lines.append('  CUDA kernel events : %s' % (str(kernel_count) if kernel_count >= 0 else 'n/a (parse error)'))
    if rt_mode:
        lines.append(
            '  T_rt mean (onboard, excl. read_points): %.2f ms' % stats['T_rt']['mean']
        )
    lines.append('')
    lines.append('── Outputs ──────────────────────────────────────────────────────')
    lines.append('  Chrome trace       : %s' % trace_path)
    lines.append('  Summary            : %s' % str(result_dir / 'profile_summary.txt'))
    lines.append('=' * 70)

    summary_text = '\n'.join(lines)
    summary_path = result_dir / 'profile_summary.txt'
    summary_path.write_text(summary_text)

    for line in lines:
        logger.info(line)

    return summary_text


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs, cfg)

    np.random.seed(1024)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir is None:
        result_dir = (
            Path(__file__).parent.parent.parent
            / 'profile_outputs'
            / ('suite_%s' % ts)
        )
    else:
        result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    log_file = result_dir / ('log_%s.txt' % ts)
    logger = common_utils.create_logger(log_file, rank=0)
    logger.info('Profile suite — output dir: %s', result_dir)

    if getattr(args, 'nsight', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    else:
        os.environ.pop('CUDA_LAUNCH_BLOCKING', None)

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    use_preprocess_gpu = getattr(args, 'preprocess_gpu', False)
    vp = None
    compiled_vox = None
    if use_preprocess_gpu:
        if getattr(args, 'int8', False):
            raise SystemExit('--preprocess_gpu is incompatible with --int8.')
        if int(args.batch_size) != 1:
            raise SystemExit('--preprocess_gpu requires --batch_size 1.')
        if not dataset_supports_gpu_voxel(test_set):
            raise SystemExit('--preprocess_gpu requires KITTI (dataset with fetch_sample_for_gpu_voxel).')
        vp = resolve_voxel_params(cfg)
        if getattr(args, 'compile_voxelizer', False):
            compiled_vox = build_compiled_voxelize_fn(
                vp['point_cloud_range'], vp['voxel_size'], vp['max_points_per_voxel'],
                vp['max_num_voxels'], vp['use_lead_xyz'], logger,
            )
        logger.info('Using GPU voxelization profiling path (inference.py-aligned); see --preprocess_gpu')

    model = load_model_for_inference(cfg, args, logger, test_set, to_cpu=False)

    use_rt_kitti_cpu = (
        not use_preprocess_gpu
        and not getattr(args, 'int8', False)
        and hasattr(test_set, 'build_input_dict')
    )

    if use_preprocess_gpu:
        accum, peak_mem_steps, trace_path, fwd_ms = run_profile_loop_preprocess_gpu(
            model, test_set, cfg.CLASS_NAMES, args, logger, result_dir, vp, compiled_vox,
        )
        steps_run = min(args.steps, len(test_set))
    elif use_rt_kitti_cpu:
        accum, peak_mem_steps, trace_path, fwd_ms = run_profile_loop_kitti_rt_cpu(
            model, test_set, cfg.CLASS_NAMES, args, logger, result_dir,
        )
        steps_run = min(args.steps, len(test_set))
    else:
        accum, peak_mem_steps, trace_path, fwd_ms = run_profile_loop(
            model, test_loader, test_set, cfg.CLASS_NAMES, args, logger, result_dir
        )
        steps_run = min(args.steps, len(test_loader))
    if getattr(args, 'nsight', False):
        logger.info('Nsight: run e.g. nsys profile -o trace python profile_suite.py ... --nsight (same args)')
    write_summary(
        accum, peak_mem_steps, args.batch_size, steps_run,
        trace_path, result_dir, logger, args, forward_ms_steps=fwd_ms,
    )


if __name__ == '__main__':
    main()
