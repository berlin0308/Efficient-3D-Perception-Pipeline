"""
profile_suite.py — comprehensive profiling for PointPillars / OpenPCDet.

Captures per-stage latency (dataloader, H2D, VFE+scatter, backbone+head, postprocess),
throughput (samples/s), peak GPU memory, kernel launch count, and exports a
Chrome-trace JSON and a plain-text summary.

Run from OpenPCDet/tools/:
    python profile_suite.py \
        --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt /path/to/checkpoint.pth \
        --output_dir /path/to/profile_outputs/suite_run \
        [--warmup 10] [--steps 30] [--batch_size 1] [--workers 4] \
        [--traced_model /path/to/model.pt] \
        [--compile] [--amp] [--cuda_id 1]

Outputs (all in --output_dir):
    profile_summary.txt        — human-readable stage table + throughput + memory
    torch_profile_trace.json   — Chrome trace (open in chrome://tracing or Perfetto)
"""

import _init_path  # noqa: F401  adds pcdet to sys.path
import argparse
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
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='PointPillars profile suite')
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='directory for outputs; default: profile_outputs/suite_<timestamp>')
    parser.add_argument('--warmup', type=int, default=10,
                        help='warmup steps before measurement (default: 10)')
    parser.add_argument('--steps', type=int, default=30,
                        help='measured steps (default: 30)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--traced_model', type=str, default=None,
                        help='path to TorchScript .pt (profile_utils/export.py)')
    parser.add_argument('--compile', action='store_true',
                        help='wrap model with torch.compile() before profiling')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable mixed-precision inference with fp16 autocast')
    parser.add_argument('--cuda_id', type=int, default=1)
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
            return dict(mean=0.0, p50=0.0, p99=0.0, std=0.0, n=0)
        return dict(
            mean=float(np.mean(vals)),
            p50=float(np.percentile(vals, 50)),
            p99=float(np.percentile(vals, 99)),
            std=float(np.std(vals)),
            n=len(vals),
        )

    def all_stats(self):
        return {n: self.stats(n) for n in self.names}


# ── Model forward wrapper ──────────────────────────────────────────────────

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


# ── Core profiling loop ────────────────────────────────────────────────────

STAGE_NAMES = ['dataloader', 'h2d', 'forward', 'postprocess', 'full_frame']


def run_profile_loop(model, dataloader, dataset, class_names, args, logger, result_dir):
    model.eval()
    accum = StageAccumulator(STAGE_NAMES)
    dataloader_iter = iter(dataloader)
    final_output_dir = result_dir / 'final_result' / 'data'

    # ── warmup ────────────────────────────────────────────────────────────
    if args.warmup > 0:
        logger.info('Running %d warmup steps...', args.warmup)
    for _ in range(min(args.warmup, len(dataloader))):
        batch_dict = next(dataloader_iter)
        load_data_to_gpu(batch_dict)
        with torch.inference_mode():
            forward_model(model, batch_dict, args)
    torch.cuda.synchronize()
    dataloader_iter = iter(dataloader)

    # ── reset memory stats after warmup ───────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    peak_mem_mb_steps = []

    steps = min(args.steps, len(dataloader))
    logger.info('Profiling %d steps...', steps)

    # ── torch.profiler context ─────────────────────────────────────────────
    trace_path = str(result_dir / 'torch_profile_trace.json')
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        on_trace_ready=None,         # export manually below
    ) as prof:
        for step_idx in range(steps):
            # ── dataloader stage ──────────────────────────────────────────
            t_dl_start = time.perf_counter()
            with torch.profiler.record_function('dataloader'):
                batch_dict = next(dataloader_iter)
            dl_ms = (time.perf_counter() - t_dl_start) * 1e3

            frame_start = time.perf_counter()

            # ── H2D transfer ──────────────────────────────────────────────
            with CudaTimer() as h2d_timer:
                with torch.profiler.record_function('h2d'):
                    load_data_to_gpu(batch_dict)

            # ── forward ───────────────────────────────────────────────────
            with CudaTimer() as fwd_timer:
                with torch.profiler.record_function('forward'):
                    with torch.inference_mode():
                        pred_dicts, ret_dict = forward_model(model, batch_dict, args)

            # ── postprocess (GPU->CPU, generate annos) ────────────────────
            t_post_start = time.perf_counter()
            with torch.profiler.record_function('postprocess'):
                _ = dataset.generate_prediction_dicts(
                    batch_dict, pred_dicts, class_names, output_path=None
                )
            post_ms = (time.perf_counter() - t_post_start) * 1e3

            torch.cuda.synchronize()
            frame_ms = (time.perf_counter() - frame_start) * 1e3

            # ── record ────────────────────────────────────────────────────
            accum.record('dataloader', dl_ms)
            accum.record('h2d', h2d_timer.ms)
            accum.record('forward', fwd_timer.ms)
            accum.record('postprocess', post_ms)
            accum.record('full_frame', frame_ms)

            peak_mem_mb_steps.append(torch.cuda.max_memory_allocated() / 1024**2)
            prof.step()

    prof.export_chrome_trace(trace_path)
    logger.info('Chrome trace saved → %s', trace_path)

    return accum, peak_mem_mb_steps, trace_path


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
                  result_dir, logger, args):
    stats = accum.all_stats()
    total_samples = steps * batch_size

    # throughput from mean full_frame
    mean_frame_ms = stats['full_frame']['mean']
    throughput = (batch_size / mean_frame_ms * 1e3) if mean_frame_ms > 0 else 0.0

    peak_mem_overall = max(peak_mem_mb_steps) if peak_mem_mb_steps else 0.0
    mean_peak_mem = np.mean(peak_mem_mb_steps) if peak_mem_mb_steps else 0.0

    kernel_count = count_kernel_launches(trace_path)

    lines = []
    lines.append('=' * 70)
    lines.append('  PointPillars Profile Suite — %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('=' * 70)
    lines.append('  Config  : %s' % args.cfg_file)
    lines.append('  Ckpt    : %s' % args.ckpt)
    lines.append('  Compile : %s' % getattr(args, 'compile', False))
    lines.append('  AMP (fp16): %s' % bool(getattr(args, 'amp', False)))
    lines.append('  Traced  : %s' % (args.traced_model or 'none'))
    lines.append('  Warmup  : %d steps   Measured: %d steps   Batch: %d' % (
        args.warmup, steps, batch_size))
    lines.append('')
    lines.append('── Per-stage latency (mean / p50 / p99 / std) [ms] ─────────────')

    col_w = 14
    header = ('%-16s' % 'Stage') + ''.join(('%*s' % (col_w, h)) for h in ['mean', 'p50', 'p99', 'std'])
    lines.append(header)
    lines.append('-' * (16 + col_w * 4))

    stage_display = [
        ('dataloader', 'DataLoader (CPU)'),
        ('h2d',        'H2D Transfer'),
        ('forward',    'Forward (GPU)'),
        ('postprocess','PostProcess'),
        ('full_frame', '── Full Frame ──'),
    ]
    for key, label in stage_display:
        s = stats[key]
        row = ('%-16s' % label) + ''.join(
            ('%*.2f' % (col_w, s[f])) for f in ['mean', 'p50', 'p99', 'std']
        )
        lines.append(row)

    lines.append('')
    lines.append('── Throughput & Memory ─────────────────────────────────────────')
    lines.append('  Throughput         : %.1f samples/s  (batch=%d, mean_frame=%.2f ms)' % (
        throughput, batch_size, mean_frame_ms))
    lines.append('  Peak GPU Memory    : %.1f MB  (max across steps)' % peak_mem_overall)
    lines.append('  Mean peak per step : %.1f MB' % mean_peak_mem)
    lines.append('  CUDA kernel events : %s' % (str(kernel_count) if kernel_count >= 0 else 'n/a (parse error)'))
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

    # Disable CUDA_LAUNCH_BLOCKING; we use CUDA events for accurate GPU timing
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

    model = load_model_for_inference(cfg, args, logger, test_set, to_cpu=False)

    accum, peak_mem_steps, trace_path = run_profile_loop(
        model, test_loader, test_set, cfg.CLASS_NAMES, args, logger, result_dir
    )

    steps_run = min(args.steps, len(test_loader))
    write_summary(accum, peak_mem_steps, args.batch_size, steps_run,
                  trace_path, result_dir, logger, args)


if __name__ == '__main__':
    main()
