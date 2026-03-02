"""
Shared model loading for inference (offline test and online real-time).
Provides load_model_for_inference() so test.py and test_online.py use the same
ckpt / traced_model / torch.compile logic.
"""
from pathlib import Path

import torch
import torch.nn as nn

from pcdet.models import build_network


class TracedModelWrapper(nn.Module):
    """Uses a TorchScript-traced forward (voxels, voxel_num_points, voxel_coords) -> (batch_cls_preds, batch_box_preds)
    and delegates post_processing to the full Python model. Compatible with export.py output.
    NVTX ranges match baseline: forward (traced NN), post_processing."""

    def __init__(self, traced_module, full_model):
        super().__init__()
        self.traced = traced_module
        self.full_model = full_model
        self.full_model.eval()

    def forward(self, batch_dict):
        nvtx = getattr(torch.cuda, 'nvtx', None)
        voxels = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        voxel_coords = batch_dict['voxel_coords']
        if nvtx is not None:
            nvtx.range_push('forward')
        batch_cls_preds, batch_box_preds = self.traced(voxels, voxel_num_points, voxel_coords)
        if nvtx is not None:
            nvtx.range_pop()
        batch_dict = dict(batch_dict)
        batch_dict['batch_cls_preds'] = batch_cls_preds
        batch_dict['batch_box_preds'] = batch_box_preds
        batch_dict['cls_preds_normalized'] = False
        if nvtx is not None:
            nvtx.range_push('post_processing')
        out = self.full_model.post_processing(batch_dict)
        if nvtx is not None:
            nvtx.range_pop()
        return out


def load_model_for_inference(cfg, args, logger, dataset, to_cpu=False):
    """
    Build network, load checkpoint, optionally wrap with TracedModelWrapper or torch.compile.
    Returns model on GPU in eval mode.

    Args:
        cfg: pcdet config object (e.g. from cfg_from_yaml_file).
        args: parsed args with .ckpt, .pretrained_model, optional .traced_model, .compile.
        logger: logger instance.
        dataset: dataset instance (for num_class, etc.); used by build_network.
        to_cpu: if True, load_params_from_file loads to CPU (e.g. for distributed).

    Returns:
        model: torch.nn.Module on CUDA, eval mode.
    """
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(
        filename=args.ckpt, logger=logger, to_cpu=to_cpu,
        pre_trained_path=getattr(args, 'pretrained_model', None)
    )
    model.cuda()
    model.eval()

    traced_path = getattr(args, 'traced_model', None)
    if traced_path:
        path = Path(traced_path)
        if not path.exists():
            raise FileNotFoundError('--traced_model file not found: %s' % traced_path)
        logger.info('Loading traced model from %s (forward only; post_processing from --ckpt model)', path)
        traced = torch.jit.load(str(path), map_location='cuda')
        traced.eval()
        model = TracedModelWrapper(traced, model)
        model.cuda()
    elif getattr(args, 'compile', False):
        if hasattr(torch, 'compile'):
            logger.info('Wrapping model with torch.compile()')
            model = torch.compile(model, mode='default')
        else:
            logger.warning('--compile set but torch.compile not available; skipping')

    return model
