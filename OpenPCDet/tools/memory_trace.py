"""
memory_trace.py — end-to-end GPU memory profiler for PointPillars.

Instruments each pipeline stage with per-stage snapshots:
  DataLoader → H2D transfer → VFE+Scatter → 2D Backbone → Head → PostProcess

Reports:
  - Peak allocated & reserved memory per stage
  - H2D transfer size (bytes moved from CPU to GPU per frame)
  - Memory fragmentation estimate (reserved - allocated)
  - Full-frame peak memory
  - Optional: torch.cuda.memory_snapshot() dump for the Memory Viz tool
    (https://pytorch.org/memory_viz)

Run from OpenPCDet/tools/:
    python memory_trace.py \
        --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt /path/to/checkpoint.pth \
        [--warmup 5] [--steps 20] [--batch_size 1] [--workers 4] \
        [--snapshot]   # enables torch memory snapshot (large file, opens in memory_viz)
        [--traced_model /path/to/model.pt] \
        [--amp] \
        [--cuda_id 1] \
        [--output_dir /path/to/out]

Outputs (in --output_dir):
    memory_summary.txt          — stage table + fragmentation
    memory_snapshot.pickle      — (if --snapshot) for pytorch.org/memory_viz
"""

import _init_path  # noqa: F401
import argparse
import datetime
import os
import pickle
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


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='E2E GPU memory profiler')
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--warmup', type=int, default=5,
                        help='warmup steps (default: 5)')
    parser.add_argument('--steps', type=int, default=20,
                        help='measured steps (default: 20)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--snapshot', action='store_true',
                        help='export torch memory snapshot for pytorch.org/memory_viz')
    parser.add_argument('--traced_model', type=str, default=None)
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable mixed-precision inference with fp16 autocast')
    parser.add_argument('--cuda_id', type=int, default=1)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


# ── Memory snapshot helpers ────────────────────────────────────────────────

class MemSnapshot:
    """Records allocated and reserved memory at a checkpoint."""

    def __init__(self):
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        self.allocated_mb = stats.get('allocated_bytes.all.current', 0) / 1024**2
        self.reserved_mb = stats.get('reserved_bytes.all.current', 0) / 1024**2
        self.peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
        self.peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    @property
    def fragmentation_mb(self):
        return self.reserved_mb - self.allocated_mb


def _snap():
    return MemSnapshot()


def forward_model(model, batch_dict, args):
    amp_enabled = bool(getattr(args, 'amp', False)) and torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled) \
        if amp_enabled else nullcontext()
    with amp_ctx:
        return model(batch_dict)


# ── H2D transfer size estimator ────────────────────────────────────────────

def h2d_bytes_in_batch(batch_dict):
    """Estimate bytes transferred host→device by summing CPU tensor sizes."""
    total = 0
    for v in batch_dict.values():
        if isinstance(v, torch.Tensor) and v.device.type == 'cpu':
            total += v.numel() * v.element_size()
        elif isinstance(v, np.ndarray):
            total += v.nbytes
    return total


# ── Instrumented model runner ──────────────────────────────────────────────

# We hook into the model's sub-modules to capture mid-forward memory.
# Hooks are attached before the first measured step and removed after.

_stage_snaps = {}  # filled by hooks during forward


def _make_post_hook(stage_name):
    def hook(module, inputs, output):
        _stage_snaps[stage_name] = _snap()
    return hook


def attach_hooks(model):
    """
    Attach forward hooks to key sub-modules.
    Returns a list of hook handles for removal.

    Targeted modules (OpenPCDet PointPillars):
        vfe       — PillarVFE  (pcdet.models.backbones_3d.vfe)
        map_to_bev— PointPillarScatter
        backbone  — BaseBEVBackbone  (2D backbone)
        dense_head— AnchorHeadSingle
    """
    handles = []
    target_types = {
        'vfe':        ('pcdet.models.backbones_3d.vfe', 'PillarVFE'),
        'scatter':    ('pcdet.models.backbones_3d.vfe', 'PointPillarScatter'),
        'backbone':   ('pcdet.models.backbones_2d', 'BaseBEVBackbone'),
        'dense_head': ('pcdet.models.dense_heads', 'AnchorHeadSingle'),
    }

    # Walk the model and attach by class name (type-name match, not isinstance,
    # so we don't need to import every sub-module here).
    for name, module in model.named_modules():
        type_name = type(module).__name__
        for stage, (_, cls_name) in target_types.items():
            if type_name == cls_name:
                h = module.register_forward_hook(_make_post_hook(stage))
                handles.append(h)
                break

    # Fallback: if the model is a TorchScript/traced wrapper, hooks won't fire.
    # We rely on pre/post snapshots in that case.
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── Per-stage accumulator ──────────────────────────────────────────────────

STAGES = ['before_h2d', 'after_h2d', 'after_vfe', 'after_scatter',
          'after_backbone', 'after_head', 'after_postprocess']


class StageMemAccum:
    def __init__(self):
        self._alloc = {s: [] for s in STAGES}
        self._peak = {s: [] for s in STAGES}
        self._frag = {s: [] for s in STAGES}

    def record(self, stage, snap):
        self._alloc[stage].append(snap.allocated_mb)
        self._peak[stage].append(snap.peak_allocated_mb)
        self._frag[stage].append(snap.fragmentation_mb)

    def stats(self, stage):
        alloc = np.array(self._alloc[stage])
        peak = np.array(self._peak[stage])
        frag = np.array(self._frag[stage])
        def _s(arr):
            if len(arr) == 0:
                return 0.0, 0.0
            return float(np.mean(arr)), float(np.max(arr))
        return {
            'alloc_mean': _s(alloc)[0], 'alloc_max': _s(alloc)[1],
            'peak_mean':  _s(peak)[0],  'peak_max':  _s(peak)[1],
            'frag_mean':  _s(frag)[0],  'frag_max':  _s(frag)[1],
        }


# ── Core measurement loop ──────────────────────────────────────────────────

def run_memory_trace(model, dataloader, dataset, class_names, args, logger):
    model.eval()
    dataloader_iter = iter(dataloader)
    accum = StageMemAccum()
    h2d_sizes_mb = []

    # warmup
    if args.warmup > 0:
        logger.info('Running %d warmup steps...', args.warmup)
    for _ in range(min(args.warmup, len(dataloader))):
        batch_dict = next(dataloader_iter)
        load_data_to_gpu(batch_dict)
        with torch.inference_mode():
            forward_model(model, batch_dict, args)
    torch.cuda.synchronize()
    dataloader_iter = iter(dataloader)

    handles = attach_hooks(model)
    steps = min(args.steps, len(dataloader))
    logger.info('Measuring memory over %d steps...', steps)

    snapshot_segments = []  # for memory_viz if --snapshot

    with torch.inference_mode():
        for step_idx in range(steps):
            torch.cuda.reset_peak_memory_stats()
            _stage_snaps.clear()

            # --- before H2D ---
            batch_dict = next(dataloader_iter)
            h2d_bytes = h2d_bytes_in_batch(batch_dict)
            h2d_sizes_mb.append(h2d_bytes / 1024**2)
            accum.record('before_h2d', _snap())

            # --- H2D ---
            load_data_to_gpu(batch_dict)
            torch.cuda.synchronize()
            accum.record('after_h2d', _snap())

            # --- forward (hooks fire inside) ---
            if args.snapshot and step_idx == 0:
                # Record one detailed snapshot on the first measured step
                torch.cuda.memory._record_memory_history(max_entries=100_000)

            pred_dicts, _ = forward_model(model, batch_dict, args)
            torch.cuda.synchronize()

            if args.snapshot and step_idx == 0:
                snapshot_segments = torch.cuda.memory._snapshot()
                torch.cuda.memory._record_memory_history(enabled=None)

            # Map hook-captured stages to accumulator keys
            for hook_key, accum_key in [
                ('vfe',        'after_vfe'),
                ('scatter',    'after_scatter'),
                ('backbone',   'after_backbone'),
                ('dense_head', 'after_head'),
            ]:
                if hook_key in _stage_snaps:
                    accum.record(accum_key, _stage_snaps[hook_key])

            # --- postprocess ---
            _ = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names, output_path=None
            )
            torch.cuda.synchronize()
            accum.record('after_postprocess', _snap())

    remove_hooks(handles)
    return accum, h2d_sizes_mb, snapshot_segments


# ── Summary writer ─────────────────────────────────────────────────────────

def write_summary(accum, h2d_sizes_mb, snapshot_segments, steps, batch_size,
                  result_dir, logger, args):
    lines = []
    lines.append('=' * 72)
    lines.append('  PointPillars Memory Trace — %s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append('=' * 72)
    lines.append('  Config  : %s' % args.cfg_file)
    lines.append('  Compile : %s   Traced : %s' % (
        getattr(args, 'compile', False), args.traced_model or 'none'))
    lines.append('  AMP (fp16): %s' % bool(getattr(args, 'amp', False)))
    lines.append('  Warmup  : %d   Steps : %d   Batch : %d' % (
        args.warmup, steps, batch_size))
    lines.append('')
    lines.append('── H2D Transfer ────────────────────────────────────────────────')
    if h2d_sizes_mb:
        lines.append('  Mean per frame : %.2f MB   Max : %.2f MB' % (
            np.mean(h2d_sizes_mb), np.max(h2d_sizes_mb)))
    lines.append('  Note: actual DMA bytes may differ; this estimates CPU tensor sizes.')
    lines.append('')
    lines.append('── Per-stage GPU Memory (MB) ────────────────────────────────────')

    stage_labels = {
        'before_h2d':       'Before H2D',
        'after_h2d':        'After H2D',
        'after_vfe':        'After VFE (hook)',
        'after_scatter':    'After Scatter (hook)',
        'after_backbone':   'After Backbone (hook)',
        'after_head':       'After Head (hook)',
        'after_postprocess':'After Postprocess',
    }

    col = 12
    hdr = ('%-22s' % 'Stage') + ''.join('%*s' % (col, h) for h in [
        'alloc_mean', 'alloc_max', 'peak_mean', 'peak_max', 'frag_mean'])
    lines.append(hdr)
    lines.append('-' * (22 + col * 5))

    for stage in STAGES:
        s = accum.stats(stage)
        # Check if we have data (hooks may not fire for TorchScript models)
        label = stage_labels.get(stage, stage)
        if s['alloc_max'] == 0.0 and 'hook' in label:
            label += ' *'
        row = ('%-22s' % label) + ''.join(
            '%*.1f' % (col, s[k]) for k in ['alloc_mean', 'alloc_max', 'peak_mean', 'peak_max', 'frag_mean']
        )
        lines.append(row)

    lines.append('')
    lines.append('  * Hook stages show 0 when model is TorchScript/traced (hooks do not fire).')
    lines.append('    In that case, use after_h2d / after_postprocess as pre/post bounds.')
    lines.append('')
    lines.append('── Fragmentation Interpretation ─────────────────────────────────')
    lines.append('  fragmentation = reserved - allocated (memory held by allocator but not used)')
    lines.append('  High fragmentation (> 200 MB) can cause OOM even with apparent free memory.')
    lines.append('  Mitigation: torch.cuda.empty_cache() between runs, or PYTORCH_NO_CUDA_MEMORY_CACHING=1')
    lines.append('')

    snap_path = ''
    if snapshot_segments:
        snap_path = str(result_dir / 'memory_snapshot.pickle')
        with open(snap_path, 'wb') as f:
            pickle.dump(snapshot_segments, f)
        lines.append('── Memory Snapshot ──────────────────────────────────────────────')
        lines.append('  Saved → %s' % snap_path)
        lines.append('  View  : python -m torch.cuda.memory_viz %s' % snap_path)
        lines.append('  Or    : upload at https://pytorch.org/memory_viz')
        lines.append('')

    lines.append('── Outputs ──────────────────────────────────────────────────────')
    lines.append('  Summary : %s' % str(result_dir / 'memory_summary.txt'))
    if snap_path:
        lines.append('  Snapshot: %s' % snap_path)
    lines.append('=' * 72)

    text = '\n'.join(lines)
    (result_dir / 'memory_summary.txt').write_text(text)
    for line in lines:
        logger.info(line)


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
            / ('memory_%s' % ts)
        )
    else:
        result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    log_file = result_dir / ('log_%s.txt' % ts)
    logger = common_utils.create_logger(log_file, rank=0)
    logger.info('Memory trace — output dir: %s', result_dir)

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

    accum, h2d_sizes_mb, snapshot_segments = run_memory_trace(
        model, test_loader, test_set, cfg.CLASS_NAMES, args, logger
    )

    steps_run = min(args.steps, len(test_loader))
    write_summary(accum, h2d_sizes_mb, snapshot_segments, steps_run,
                  args.batch_size, result_dir, logger, args)


if __name__ == '__main__':
    main()
