"""
Shared model loading for inference (offline test and real-time).
Provides load_model_for_inference() so test.py and inference.py use the same
ckpt / traced_model / torch.compile logic.
"""
from pathlib import Path

import torch
import torch.nn as nn

from pcdet.models import build_network

_SCATTER_MODULE_NAMES = ('PointPillarScatter', 'PointPillarScatter3d')
_BEV_BACKBONE_NAMES = ('BaseBEVBackbone', 'BaseBEVResBackbone')


def _configure_memory_opts(model: nn.Module, args, logger) -> None:
    """
    Optional memory optimizations (all default off via argparse):
    - memory_opt_scatter: HWC coalesced scatter in PointPillarScatter (see pointpillar_scatter.py).
    - memory_opt_conv2d: torch.channels_last on model + BEV backbone input layout.
    """
    scatter_on = bool(getattr(args, 'memory_opt_scatter', False))
    conv2d_on = bool(getattr(args, 'memory_opt_conv2d', False))
    traced_path = getattr(args, 'traced_model', None)
    if conv2d_on and traced_path:
        logger.warning('--memory_opt_conv2d is ignored when --traced_model is set')
        conv2d_on = False

    for m in model.modules():
        if m.__class__.__name__ in _SCATTER_MODULE_NAMES:
            m.memory_opt_scatter = scatter_on
            m.memory_opt_conv2d = conv2d_on
        if m.__class__.__name__ in _BEV_BACKBONE_NAMES:
            m.memory_opt_conv2d = conv2d_on

    if conv2d_on:
        if not torch.cuda.is_available():
            logger.warning('memory_opt_conv2d: CUDA not available; skipping channels_last')
        else:
            logger.info('memory_opt_conv2d: model.to(memory_format=torch.channels_last)')
            model.to(memory_format=torch.channels_last)


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
    Returns model on GPU in eval mode, unless --int8 (CPU dynamic Linear PTQ).

    Args:
        args: optional memory_opt_scatter, memory_opt_conv2d (default False).
    """
    use_int8 = bool(getattr(args, 'int8', False))
    if use_int8 and getattr(args, 'compile', False):
        raise ValueError('Do not combine --int8 with --compile (CPU PTQ path only).')
    if use_int8 and getattr(args, 'amp', False):
        raise ValueError('Do not combine --int8 with --amp.')

    load_cpu = bool(to_cpu or use_int8)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(
        filename=args.ckpt, logger=logger, to_cpu=load_cpu,
        pre_trained_path=getattr(args, 'pretrained_model', None)
    )
    model.eval()

    if use_int8:
        from int8_utils import apply_dynamic_int8_linear_ptq
        logger.info('Applying dynamic INT8 PTQ (nn.Linear on CPU; Conv2d float CPU).')
        model = apply_dynamic_int8_linear_ptq(model)
        return model

    model.cuda()

    if bool(getattr(args, 'memory_opt_scatter', False)) or bool(getattr(args, 'memory_opt_conv2d', False)):
        _configure_memory_opts(model, args, logger)

    traced_path = getattr(args, 'traced_model', None)
    trt_path    = getattr(args, 'trt_engine', None)
    if traced_path and trt_path:
        raise ValueError('Specify only one of --traced_model or --trt_engine.')

    if traced_path:
        path = Path(traced_path)
        if not path.exists():
            raise FileNotFoundError('--traced_model file not found: %s' % traced_path)
        logger.info('Loading traced model from %s (forward only; post_processing from --ckpt model)', path)
        traced = torch.jit.load(str(path), map_location='cuda')
        traced.eval()
        model = TracedModelWrapper(traced, model)
        model.cuda()
    elif trt_path:
        path = Path(trt_path)
        if not path.exists():
            raise FileNotFoundError('--trt_engine file not found: %s' % trt_path)
        from profile_utils.trt_runner import TRTModelWrapper
        model = TRTModelWrapper.from_engine_path(str(path), model, logger=logger)
        model.cuda()
    elif getattr(args, 'compile', False):
        if hasattr(torch, 'compile'):
            logger.info('Wrapping model with torch.compile()')
            # suppress_errors skips frames that trigger guard assertion bugs
            # (spconv x_offset numpy tensor ADInplaceOrView key mismatch on PyTorch 2.x)
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode='default')
        else:
            logger.warning('--compile set but torch.compile not available; skipping')

    return model
