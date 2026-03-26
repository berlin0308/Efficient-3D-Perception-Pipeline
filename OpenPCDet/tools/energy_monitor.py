"""
energy_monitor.py — GPU power / energy / performance-per-watt profiling.

Uses pynvml to sample GPU power draw at a configurable interval while running
inference, then computes:
  - Total energy (Joules) = integral of power over time
  - Mean / peak power draw (Watts)
  - Throughput (samples/s)
  - Performance-per-watt (samples/J) — the key metric for edge deployment

Run from OpenPCDet/tools/:
    python energy_monitor.py \
        --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt /path/to/checkpoint.pth \
        [--warmup 10] [--steps 50] [--batch_size 1] [--workers 4] \
        [--sample_interval_ms 50] \
        [--traced_model /path/to/model.pt] \
        [--compile] [--amp] [--cuda_id 1] \
        [--output_dir /path/to/out]

Dependencies:
    pip install pynvml
    # pynvml ships with nvidia-ml-py; check: python -c "import pynvml; pynvml.nvmlInit()"

Outputs (in --output_dir):
    energy_summary.txt    — human-readable table
    energy_samples.csv    — raw (timestamp_s, power_W) rows for plotting
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

from model_loader import load_model_for_inference
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
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
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--sample_interval_ms', type=int, default=50,
                        help='power sampling interval in ms (default: 50 ms)')
    parser.add_argument('--traced_model', type=str, default=None)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable mixed-precision inference with fp16 autocast')
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


# ── Forward wrapper ────────────────────────────────────────────────────────

def forward_model(model, batch_dict, args):
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

def run_energy_profile(model, dataloader, args, logger, nvml_handle, interval_ms):
    model.eval()
    dataloader_iter = iter(dataloader)

    # warmup (no power measurement)
    if args.warmup > 0:
        logger.info('Running %d warmup steps (no measurement)...', args.warmup)
    for _ in range(min(args.warmup, len(dataloader))):
        batch_dict = next(dataloader_iter)
        load_data_to_gpu(batch_dict)
        with torch.inference_mode():
            forward_model(model, batch_dict, args)
    torch.cuda.synchronize()
    dataloader_iter = iter(dataloader)

    steps = min(args.steps, len(dataloader))
    logger.info('Measuring energy over %d steps...', steps)

    # Start power sampling background thread
    sampler = PowerSampler(nvml_handle, interval_ms=interval_ms)
    sampler.start()

    t_inference_start = time.perf_counter()
    latencies_ms = []

    with torch.inference_mode():
        for _ in range(steps):
            batch_dict = next(dataloader_iter)
            load_data_to_gpu(batch_dict)
            t0 = time.perf_counter()
            forward_model(model, batch_dict, args)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1e3)

    t_inference_end = time.perf_counter()
    sampler.stop()

    timestamps_s, powers_W = sampler.get_samples()

    # Clip samples to the measurement window only
    t_start_rel = 0.0  # sampler starts at warmup completion
    t_end_rel = t_inference_end - t_inference_start
    mask = (timestamps_s >= t_start_rel) & (timestamps_s <= t_end_rel)
    ts_clip = timestamps_s[mask]
    pw_clip = powers_W[mask]

    return latencies_ms, ts_clip, pw_clip, steps


# ── Summary writer ─────────────────────────────────────────────────────────

def write_summary(latencies_ms, timestamps_s, powers_W, steps, batch_size,
                  result_dir, logger, args, gpu_name):
    total_samples = steps * batch_size
    wall_time_s = timestamps_s[-1] - timestamps_s[0] if len(timestamps_s) > 1 else 1e-9
    throughput = total_samples / wall_time_s if wall_time_s > 0 else 0.0

    energy_J = integrate_energy(timestamps_s, powers_W)
    mean_power_W = float(np.mean(powers_W)) if len(powers_W) else 0.0
    peak_power_W = float(np.max(powers_W)) if len(powers_W) else 0.0
    perf_per_watt = throughput / mean_power_W if mean_power_W > 0 else 0.0
    samples_per_joule = total_samples / energy_J if energy_J > 0 else 0.0

    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
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
    lines.append('  Warmup         : %d   Steps: %d   Batch: %d' % (
        args.warmup, steps, batch_size))
    lines.append('')
    lines.append('── Latency (ms) ─────────────────────────────────────────')
    lines.append('  Mean : %.2f   p50 : %.2f   p99 : %.2f' % (mean_lat, p50_lat, p99_lat))
    lines.append('')
    lines.append('── Throughput ───────────────────────────────────────────')
    lines.append('  %.2f samples/s  (total_samples=%d, wall=%.2f s)' % (
        throughput, total_samples, wall_time_s))
    lines.append('')
    lines.append('── Power & Energy ───────────────────────────────────────')
    lines.append('  Mean power     : %.1f W' % mean_power_W)
    lines.append('  Peak power     : %.1f W' % peak_power_W)
    lines.append('  Total energy   : %.2f J  (over %.2f s, %d samples)' % (
        energy_J, wall_time_s, len(powers_W)))
    lines.append('')
    lines.append('── Performance-per-Watt ─────────────────────────────────')
    lines.append('  %.4f samples/s/W  (throughput / mean_power)' % perf_per_watt)
    lines.append('  %.4f samples/J   (total_samples / total_energy)' % samples_per_joule)
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

    model = load_model_for_inference(cfg, args, logger, test_set, to_cpu=False)

    latencies_ms, timestamps_s, powers_W, steps_run = run_energy_profile(
        model, test_loader, args, logger, nvml_handle, args.sample_interval_ms
    )

    write_summary(
        latencies_ms, timestamps_s, powers_W, steps_run, args.batch_size,
        result_dir, logger, args, gpu_name
    )

    pynvml.nvmlShutdown()


if __name__ == '__main__':
    main()
