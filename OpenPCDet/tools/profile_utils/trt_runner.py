"""
trt_runner.py — TensorRT engine runner for PointPillars.

Provides TRTModelWrapper: drop-in replacement for TracedModelWrapper.
  - TRT engine runs forward(voxels, voxel_num_points, voxel_coords)
  - full_model.post_processing(batch_dict) handles NMS in CUDA (unchanged)

Requires: tensorrt, cuda-python (pip install tensorrt cuda-python)

Usage (from model_loader.py via --trt_engine):
    wrapper = TRTModelWrapper.from_engine_path(engine_path, full_model, logger)

Standalone engine build (alternative to trtexec):
    python -c "
    from profile_utils.trt_runner import build_engine
    build_engine('pointpillar.onnx', 'pointpillar_fp16.engine', fp16=True)
    "
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger_default = logging.getLogger(__name__)


# ── TRT availability check ─────────────────────────────────────────────────

def _import_trt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        raise ImportError(
            'tensorrt not found. Install with:\n'
            '  pip install tensorrt\n'
            'or follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/'
        )


def _import_cuda():
    try:
        import cuda.cudart as cudart
        return cudart
    except ImportError:
        raise ImportError(
            'cuda-python not found. Install with:\n'
            '  pip install cuda-python'
        )


# ── Engine build helper ────────────────────────────────────────────────────

def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    min_voxels: int = 1,
    opt_voxels: int = 10000,
    max_voxels: int = 20000,
    max_voxel_points: int = 32,
    logger=None,
) -> None:
    """Build a TRT engine from an ONNX file and save it to engine_path."""
    log = logger or logger_default
    trt = _import_trt()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    log.info('Parsing ONNX: %s', onnx_path)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError('ONNX parse failed:\n' + '\n'.join(str(e) for e in errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
    if fp16:
        if not builder.platform_has_fast_fp16:
            log.warning('GPU does not report fast FP16; building FP16 engine anyway')
        config.set_flag(trt.BuilderFlag.FP16)
        log.info('FP16 mode enabled')

    # Dynamic shape profile for the num_voxels dimension
    profile = builder.create_optimization_profile()
    # voxels: [num_voxels, max_points_per_voxel, num_features]
    profile.set_shape('voxels',
                      (min_voxels, max_voxel_points, 4),
                      (opt_voxels, max_voxel_points, 4),
                      (max_voxels, max_voxel_points, 4))
    # voxel_num_points: [num_voxels]
    profile.set_shape('voxel_num_points', (min_voxels,), (opt_voxels,), (max_voxels,))
    # voxel_coords: [num_voxels, 4]  (batch_idx, z, y, x)
    profile.set_shape('voxel_coords', (min_voxels, 4), (opt_voxels, 4), (max_voxels, 4))
    config.add_optimization_profile(profile)

    log.info('Building TRT engine (this may take a few minutes)…')
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TRT engine build failed (returned None)')

    out = Path(engine_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        f.write(serialized)
    log.info('Saved TRT engine → %s', out)


# ── TRT inference context ──────────────────────────────────────────────────

class TRTContext:
    """Loads a serialized TRT engine and runs inference."""

    def __init__(self, engine_path: str, device: int = 0):
        trt = _import_trt()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError('Failed to deserialize TRT engine: %s' % engine_path)

        self._context = self._engine.create_execution_context()
        self._device = device

        # Identify input/output tensor names
        self._input_names  = []
        self._output_names = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def infer(
        self,
        voxels: torch.Tensor,
        voxel_num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
    ):
        """
        Run TRT forward. Inputs must be CUDA tensors (contiguous).
        Returns (batch_cls_preds, batch_box_preds) as CUDA float32 tensors.
        """
        # Ensure contiguous CUDA tensors with correct dtypes
        voxels           = voxels.contiguous().float()
        voxel_num_points = voxel_num_points.contiguous().int()
        voxel_coords     = voxel_coords.contiguous().int()

        num_voxels = voxels.shape[0]

        # Set dynamic input shapes
        self._context.set_input_shape('voxels',           tuple(voxels.shape))
        self._context.set_input_shape('voxel_num_points', (num_voxels,))
        self._context.set_input_shape('voxel_coords',     tuple(voxel_coords.shape))

        # Bind input tensor addresses
        self._context.set_tensor_address('voxels',           voxels.data_ptr())
        self._context.set_tensor_address('voxel_num_points', voxel_num_points.data_ptr())
        self._context.set_tensor_address('voxel_coords',     voxel_coords.data_ptr())

        # Allocate outputs based on inferred shapes
        outputs = {}
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            out   = torch.empty(shape, dtype=torch.float32, device='cuda')
            self._context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out

        stream = torch.cuda.current_stream().cuda_stream
        self._context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        return outputs['batch_cls_preds'], outputs['batch_box_preds']


# ── Drop-in wrapper (matches TracedModelWrapper interface) ─────────────────

class TRTModelWrapper(nn.Module):
    """
    Uses a TRT engine for forward(voxels, voxel_num_points, voxel_coords) and
    delegates post_processing to the full Python model.
    Interface identical to TracedModelWrapper.
    """

    def __init__(self, trt_context: TRTContext, full_model: nn.Module):
        super().__init__()
        self.trt   = trt_context
        self.full_model = full_model
        self.full_model.eval()

    @classmethod
    def from_engine_path(
        cls,
        engine_path: str,
        full_model: nn.Module,
        logger=None,
    ) -> 'TRTModelWrapper':
        log = logger or logger_default
        log.info('Loading TRT engine from %s', engine_path)
        ctx = TRTContext(engine_path)
        log.info('TRT engine loaded (forward only; post_processing from --ckpt model)')
        return cls(ctx, full_model)

    def forward(self, batch_dict):
        nvtx = getattr(torch.cuda, 'nvtx', None)

        voxels           = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        voxel_coords     = batch_dict['voxel_coords']

        if nvtx is not None:
            nvtx.range_push('forward')
        batch_cls_preds, batch_box_preds = self.trt.infer(
            voxels, voxel_num_points, voxel_coords
        )
        if nvtx is not None:
            nvtx.range_pop()

        batch_dict = dict(batch_dict)
        batch_dict['batch_cls_preds']      = batch_cls_preds
        batch_dict['batch_box_preds']      = batch_box_preds
        batch_dict['cls_preds_normalized'] = False

        if nvtx is not None:
            nvtx.range_push('post_processing')
        out = self.full_model.post_processing(batch_dict)
        if nvtx is not None:
            nvtx.range_pop()
        return out
