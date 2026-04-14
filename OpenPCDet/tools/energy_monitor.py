"""
energy_monitor.py — GPU power / energy / performance-per-watt profiling.

Uses pynvml to sample GPU power draw at a configurable interval while running
inference, then computes:
  - Total energy (Joules) = integral of power over time
  - With --compile: optional exclusion of Dynamo/torch.compile spike forwards from energy & samples/J
  - Mean / peak power draw (Watts)
  - Throughput (samples/s)
  - Performance-per-watt (samples/J) — the key metric for edge deployment

Run from OpenPCDet/tools/:
    python energy_monitor.py \
        --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt /path/to/checkpoint.pth \
        [--warmup 100] [--steps 50] [--batch_size 1] [--workers 4] \
        [--sample_interval_ms 50] \
        [--traced_model /path/to/model.pt] \
        [--compile] [--amp] [--cuda_id 1] \
        [--output_dir /path/to/out]

Dependencies:
    pip install pynvml
    # pynvml ships with nvidia-ml-py; check: python -c "import pynvml; pynvml.nvmlInit()"

Outputs (in --output_dir):
    energy_summary.txt           — human-readable table
    energy_samples.csv           — raw (timestamp_s, power_W) rows for plotting
    energy_latency_per_step.csv  — per-step forward latency (ms) during measurement
"""

import _init_path  # noqa: F401
import argparse
import csv
import datetime
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path

# ── CUDA device selection before torch import ──────────────────────────────
def _early_cuda_id():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--cuda_id', type=int, default=1)
    a, _ = p.parse_known_args()
    return a.cuda_id

os.environ['CUDA_VISIBLE_DEVICES'] = str(_early_cuda_id())

import numpy as np
import torch

from int8_utils import load_batch_to_device
from model_loader import load_model_for_inference
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from preprocess_gpu_loop import (
    build_compiled_voxelize_fn,
    build_preprocess_gpu_dataloader,
    dataset_supports_gpu_voxel,
    resolve_voxel_params,
)
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


# ── pynvml availability check ──────────────────────────────────────────────

def _check_pynvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except ImportError:
        raise SystemExit(
            '[energy_monitor] pynvml not found.\n'
            'Install with:  pip install nvidia-ml-py\n'
            'Then retry.'
        )
    except Exception as e:
        raise SystemExit('[energy_monitor] pynvml.nvmlInit() failed: %s' % e)


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='GPU energy / perf-per-watt profiling')
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sample_interval_ms', type=int, default=50,
                        help='power sampling interval in ms (default: 50 ms)')
    parser.add_argument('--traced_model', type=str, default=None)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument(
        '--energy_exclude_compile_over_ms',
        type=float,
        default=200.0,
        help='With --compile / --compile_voxelizer / --energy_exclude_spikes_always: treat forwards '
        'slower than this (ms) as spikes; exclude their NVML time windows from energy & samples/J. '
        'Pick using k× steady forward p99 from a non-compile baseline (typical k=2–4; default 200 ms).',
    )
    parser.add_argument(
        '--measurement_burnin_steps',
        type=int,
        default=0,
        help='After warmup, run this many extra forwards while sampling power but exclude them from '
        'the NVML integration window (steady-state energy after dynamo/voxelizer settle).',
    )
    parser.add_argument(
        '--energy_exclude_spikes_always',
        action='store_true',
        default=False,
        help='Apply --energy_exclude_compile_over_ms spike exclusion even without --compile '
        '(e.g. odd latency outliers).',
    )
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable mixed-precision inference with fp16 autocast')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='CPU dynamic PTQ (nn.Linear int8); forward on CPU')
    parser.add_argument(
        '--memory_opt_scatter',
        action='store_true',
        default=False,
        help='HWC coalesced scatter (same as profile_suite)',
    )
    parser.add_argument(
        '--memory_opt_conv2d',
        action='store_true',
        default=False,
        help='channels_last BEV path (same as profile_suite)',
    )
    parser.add_argument(
        '--preprocess_gpu',
        action='store_true',
        default=False,
        help='GPU voxelization (same as profile_suite / inference.py; batch_size=1)',
    )
    parser.add_argument(
        '--compile_voxelizer',
        action='store_true',
        default=False,
        help='with --preprocess_gpu, torch.compile voxelizer (forces num_workers=0)',
    )
    parser.add_argument('--cuda_id', type=int, default=1)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


# ── Background power sampler ───────────────────────────────────────────────

class PowerSampler:
    """
    Samples GPU power in a background thread at a fixed interval.

    Usage:
        sampler = PowerSampler(handle, interval_ms=50)
        sampler.start()
        # ... run inference ...
        sampler.stop()
        timestamps, powers = sampler.get_samples()
    """

    def __init__(self, nvml_handle, interval_ms=50):
        self._handle = nvml_handle
        self._interval_s = interval_ms / 1e3
        self._timestamps = []
        self._powers_mw = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def start(self):
        self._t0 = time.perf_counter()
        self._thread.start()

    @property
    def t0(self):
        """perf_counter() origin used for timestamps from get_samples()."""
        return self._t0

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _sample_loop(self):
        import pynvml
        while not self._stop_event.is_set():
            t = time.perf_counter() - self._t0
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            except Exception:
                power_mw = 0
            self._timestamps.append(t)
            self._powers_mw.append(power_mw)
            time.sleep(self._interval_s)

    def get_samples(self):
        """Returns (timestamps_s, powers_W) as numpy arrays."""
        ts = np.array(self._timestamps, dtype=np.float64)
        pw = np.array(self._powers_mw, dtype=np.float64) / 1e3  # mW -> W
        return ts, pw


# ── Energy integration ─────────────────────────────────────────────────────

def integrate_energy(timestamps_s, powers_W):
    """Trapezoidal integration of power over time -> Joules."""
    if len(timestamps_s) < 2:
        return 0.0
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz(powers_W, timestamps_s))


def _midpoint_inside_intervals(t_mid, intervals):
    for a, b in intervals:
        if a <= t_mid < b:
            return True
    return False


def integrate_energy_excluding_intervals(timestamps_s, powers_W, exclude_intervals):
    """
    Trapezoidal integral of power, omitting sample segments whose midpoint falls
    inside exclude_intervals (each [a, b) in the same time base as timestamps_s).
    Used to drop torch.compile / Dynamo spikes from NVML energy totals.
    """
    if len(timestamps_s) < 2:
        return 0.0
    if not exclude_intervals:
        return integrate_energy(timestamps_s, powers_W)
    total = 0.0
    for k in range(len(timestamps_s) - 1):
        t0s, t1s = float(timestamps_s[k]), float(timestamps_s[k + 1])
        mid = 0.5 * (t0s + t1s)
        if _midpoint_inside_intervals(mid, exclude_intervals):
            continue
        p0, p1 = float(powers_W[k]), float(powers_W[k + 1])
        total += 0.5 * (t1s - t0s) * (p0 + p1)
    return float(total)


# ── Forward wrapper ────────────────────────────────────────────────────────

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


# ── Main inference + energy loop ───────────────────────────────────────────

def _next_batch_cyclic(dataloader_iter, dataloader):
    """Advance iterator; reset on exhaustion so burn-in + steps can exceed len(dataloader)."""
    try:
        return next(dataloader_iter), dataloader_iter
    except StopIteration:
        dataloader_iter = iter(dataloader)
        return next(dataloader_iter), dataloader_iter


def run_energy_profile(model, dataloader, args, logger, nvml_handle, interval_ms):
    model.eval()
    dataloader_iter = iter(dataloader)
    cpu_int8 = bool(getattr(args, 'int8', False))

    def _to_compute_device(batch_dict):
        if cpu_int8:
            load_batch_to_device(batch_dict, torch.device('cpu'))
        else:
            load_data_to_gpu(batch_dict)

    # warmup (no power measurement)
    if args.warmup > 0:
        logger.info('Running %d warmup steps (no measurement)...', args.warmup)
    for _ in range(args.warmup):
        batch_dict, dataloader_iter = _next_batch_cyclic(dataloader_iter, dataloader)
        _to_compute_device(batch_dict)
        with torch.inference_mode():
            forward_model(model, batch_dict, args)
    if not cpu_int8:
        torch.cuda.synchronize()
    dataloader_iter = iter(dataloader)

    steps = args.steps
    burnin = int(getattr(args, 'measurement_burnin_steps', 0) or 0)
    logger.info('Measuring energy over %d steps...', steps)

    # Start power sampling background thread
    sampler = PowerSampler(nvml_handle, interval_ms=interval_ms)
    sampler.start()
    t_origin = sampler.t0

    # Optional burn-in: power is sampled but excluded from integration window below
    if burnin > 0:
        logger.info('Measurement burn-in: %d steps (power sampled, excluded from reported energy)', burnin)
        with torch.inference_mode():
            for _ in range(burnin):
                batch_dict, dataloader_iter = _next_batch_cyclic(dataloader_iter, dataloader)
                _to_compute_device(batch_dict)
                forward_model(model, batch_dict, args)
                if not cpu_int8:
                    torch.cuda.synchronize()

    t_inference_start = time.perf_counter()
    latencies_ms = []
    forward_segments = []  # (t_rel0, t_rel1, lat_ms) relative to sampler t0

    with torch.inference_mode():
        for _ in range(steps):
            batch_dict, dataloader_iter = _next_batch_cyclic(dataloader_iter, dataloader)
            _to_compute_device(batch_dict)
            t0 = time.perf_counter()
            forward_model(model, batch_dict, args)
            if not cpu_int8:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1e3
            latencies_ms.append(lat_ms)
            forward_segments.append((t0 - t_origin, t1 - t_origin, lat_ms))

    t_inference_end = time.perf_counter()
    sampler.stop()

    timestamps_s, powers_W = sampler.get_samples()

    # Clip samples to the measurement window (timestamps are relative to sampler.t0)
    t_clip_lo = t_inference_start - t_origin
    t_clip_hi = t_inference_end - t_origin
    mask = (timestamps_s >= t_clip_lo) & (timestamps_s <= t_clip_hi)
    ts_clip = timestamps_s[mask]
    pw_clip = powers_W[mask]

    compile_exclude_intervals = []
    latencies_for_stats = latencies_ms
    use_spike_exclude = (
        bool(getattr(args, 'compile', False))
        or bool(getattr(args, 'compile_voxelizer', False))
        or bool(getattr(args, 'energy_exclude_spikes_always', False))
    )
    if use_spike_exclude:
        thr = float(getattr(args, 'energy_exclude_compile_over_ms', 200.0))
        compile_exclude_intervals = [(a, b) for (a, b, lat) in forward_segments if lat > thr]
        latencies_for_stats = [lat for (_, _, lat) in forward_segments if lat <= thr]
        if not latencies_for_stats:
            logger.warning(
                'All %d measured steps exceed spike threshold (%.1f ms); '
                'not excluding any intervals from energy.',
                steps,
                thr,
            )
            latencies_for_stats = latencies_ms
            compile_exclude_intervals = []
        elif compile_exclude_intervals:
            logger.info(
                'Energy spike exclusion: %d step(s) over %.1f ms (dropped from energy & samples/J)',
                len(compile_exclude_intervals),
                thr,
            )

    return latencies_ms, ts_clip, pw_clip, steps, latencies_for_stats, compile_exclude_intervals


# ── Summary writer ─────────────────────────────────────────────────────────

def write_summary(latencies_ms, timestamps_s, powers_W, steps, batch_size,
                  result_dir, logger, args, gpu_name,
                  latencies_for_stats=None, compile_exclude_intervals=None):
    """
    latencies_for_stats: if set (e.g. steady-state forwards under torch.compile), used for
    latency percentiles, throughput wall time, and samples/J numerator.
    compile_exclude_intervals: list of [t0,t1) in the same frame as timestamps_s; those
    regions are omitted from the trapezoidal energy integral.
    """
    if latencies_for_stats is None:
        latencies_for_stats = latencies_ms
    if compile_exclude_intervals is None:
        compile_exclude_intervals = []

    energy_J = integrate_energy_excluding_intervals(
        timestamps_s, powers_W, compile_exclude_intervals,
    )
    if len(powers_W):
        peak_power_W = float(np.max(powers_W))
    else:
        peak_power_W = 0.0

    steady_compile_report = bool(compile_exclude_intervals)
    if steady_compile_report:
        total_samples = len(latencies_for_stats) * batch_size
        wall_time_s = (
            sum(ms / 1e3 for ms in latencies_for_stats)
            if latencies_for_stats
            else 1e-9
        )
        throughput = total_samples / wall_time_s if wall_time_s > 0 else 0.0
        mean_power_W = energy_J / wall_time_s if wall_time_s > 0 and energy_J > 0 else 0.0
        samples_per_joule = total_samples / energy_J if energy_J > 0 else 0.0
    else:
        total_samples = steps * batch_size
        wall_time_s = (
            timestamps_s[-1] - timestamps_s[0] if len(timestamps_s) > 1 else 1e-9
        )
        throughput = total_samples / wall_time_s if wall_time_s > 0 else 0.0
        mean_power_W = float(np.mean(powers_W)) if len(powers_W) else 0.0
        samples_per_joule = total_samples / energy_J if energy_J > 0 else 0.0

    perf_per_watt = throughput / mean_power_W if mean_power_W > 0 else 0.0

    lat_arr = np.array(latencies_for_stats)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))

    lines = []
    lines.append('=' * 65)
    lines.append('  PointPillars Energy Monitor — %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('=' * 65)
    lines.append('  GPU            : %s' % gpu_name)
    lines.append('  Config         : %s' % args.cfg_file)
    lines.append('  Compile        : %s  Traced: %s' % (
        getattr(args, 'compile', False), args.traced_model or 'none'))
    lines.append('  AMP (fp16)     : %s' % bool(getattr(args, 'amp', False)))
    lines.append('  Memory opt scatter (HWC write): %s' % bool(getattr(args, 'memory_opt_scatter', False)))
    lines.append('  Memory opt conv2d (channels_last): %s' % bool(getattr(args, 'memory_opt_conv2d', False)))
    lines.append('  Preprocess GPU (voxelize on GPU): %s' % bool(getattr(args, 'preprocess_gpu', False)))
    lines.append('  Compile voxelizer: %s' % bool(getattr(args, 'compile_voxelizer', False)))
    lines.append('  INT8 (CPU PTQ) : %s' % bool(getattr(args, 'int8', False)))
    lines.append('  Warmup         : %d   Steps: %d   Batch: %d' % (
        args.warmup, steps, batch_size))
    lines.append('  Meas. burn-in  : %d  (power sampled; excluded from NVML J window)' % int(
        getattr(args, 'measurement_burnin_steps', 0) or 0,
    ))
    lines.append('')
    lines.append('── Latency (ms) ─────────────────────────────────────────')
    lines.append('  Mean : %.2f   p50 : %.2f   p95 : %.2f   p99 : %.2f' % (
        mean_lat, p50_lat, p95_lat, p99_lat))
    lines.append('')
    lines.append('── Throughput ───────────────────────────────────────────')
    lines.append('  %.2f samples/s  (total_samples=%d, wall=%.2f s)' % (
        throughput, total_samples, wall_time_s))
    lines.append('')
    lines.append('── Power & Energy ───────────────────────────────────────')
    lines.append('  Mean power     : %.1f W' % mean_power_W)
    lines.append('  Peak power     : %.1f W' % peak_power_W)
    n_ex = len(compile_exclude_intervals)
    energy_note = (
        '  (spike windows excluded: %d forward interval(s))' % n_ex if n_ex else ''
    )
    wall_note = 'steady forward wall' if steady_compile_report else 'NVML window'
    lines.append('  Total energy   : %.2f J  (%s=%.2f s, %d power samples)%s' % (
        energy_J, wall_note, wall_time_s, len(powers_W), energy_note))
    lines.append('')
    lines.append('── Performance-per-Watt ─────────────────────────────────')
    lines.append('  %.4f samples/s/W  (throughput / mean_power)' % perf_per_watt)
    lines.append(
        '  %.4f samples/J   (steady_samples / steady_energy; spike intervals omitted when enabled)'
        % samples_per_joule
    )
    lines.append('  Interpretation: higher = more efficient per unit energy')
    lines.append('')
    lines.append('── Outputs ──────────────────────────────────────────────')
    lines.append('  Summary  : %s' % str(result_dir / 'energy_summary.txt'))
    lines.append('  Raw CSV  : %s' % str(result_dir / 'energy_samples.csv'))
    lines.append('=' * 65)

    summary_text = '\n'.join(lines)
    (result_dir / 'energy_summary.txt').write_text(summary_text)
    for line in lines:
        logger.info(line)

    # Save raw power samples for plotting
    csv_path = result_dir / 'energy_samples.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_s', 'power_W'])
        for t, p in zip(timestamps_s, powers_W):
            writer.writerow([f'{t:.4f}', f'{p:.2f}'])
    logger.info('Raw power samples → %s', csv_path)

    lat_csv = result_dir / 'energy_latency_per_step.csv'
    with open(lat_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step_idx', 'forward_wall_ms'])
        for i, ms in enumerate(latencies_ms):
            w.writerow([i, f'{ms:.4f}'])
    logger.info('Per-step forward latency → %s', lat_csv)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    pynvml = _check_pynvml()

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
            / ('energy_%s' % ts)
        )
    else:
        result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    log_file = result_dir / ('log_%s.txt' % ts)
    logger = common_utils.create_logger(log_file, rank=0)

    # Resolve physical GPU index from CUDA_VISIBLE_DEVICES
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', str(args.cuda_id))
    try:
        phys_idx = int(visible.split(',')[0])
    except ValueError:
        phys_idx = args.cuda_id

    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(phys_idx)
    gpu_name = pynvml.nvmlDeviceGetName(nvml_handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    logger.info('GPU [%d]: %s', phys_idx, gpu_name)

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    if getattr(args, 'preprocess_gpu', False):
        if getattr(args, 'int8', False):
            raise SystemExit('--preprocess_gpu is incompatible with --int8.')
        if int(args.batch_size) != 1:
            raise SystemExit('--preprocess_gpu requires --batch_size 1.')
        if not dataset_supports_gpu_voxel(test_set):
            raise SystemExit('--preprocess_gpu requires KITTI (dataset with fetch_sample_for_gpu_voxel).')
        vp = resolve_voxel_params(cfg)
        compiled_vox = None
        if getattr(args, 'compile_voxelizer', False):
            compiled_vox = build_compiled_voxelize_fn(
                vp['point_cloud_range'], vp['voxel_size'], vp['max_points_per_voxel'],
                vp['max_num_voxels'], vp['use_lead_xyz'], logger,
            )
        test_loader = build_preprocess_gpu_dataloader(test_set, cfg, args, logger, compiled_vox)
        logger.info('Energy monitor: GPU voxelization DataLoader (inference.py-aligned)')

    model = load_model_for_inference(cfg, args, logger, test_set, to_cpu=False)

    latencies_ms, timestamps_s, powers_W, steps_run, lat_stats, compile_excl = run_energy_profile(
        model, test_loader, args, logger, nvml_handle, args.sample_interval_ms
    )

    write_summary(
        latencies_ms, timestamps_s, powers_W, steps_run, args.batch_size,
        result_dir, logger, args, gpu_name,
        latencies_for_stats=lat_stats,
        compile_exclude_intervals=compile_excl,
    )

    pynvml.nvmlShutdown()


if __name__ == '__main__':
    main()
