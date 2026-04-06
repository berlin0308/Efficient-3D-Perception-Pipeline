"""
Post-training dynamic INT8 (Linear only) helpers for PointPillars research harness.

PyTorch dynamic quantized Linear runs on CPU only; Conv2d stays float on CPU.
Anchor tensors in AnchorHeadTemplate are stored as a plain Python list of CUDA tensors;
they must be moved to CPU for CPU inference (see move_anchor_tensors_to_device).

3D NMS in OpenPCDet is CUDA-only; after CPU forward we move detection tensors to GPU
and keep post_processing on CUDA (see patch_detector_int8_cpu_forward_gpu_nms).
"""
from __future__ import annotations

import types

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic


def move_anchor_tensors_to_device(model: nn.Module, device: torch.device | str) -> None:
    """Move anchor list tensors (not registered as buffers) to the given device."""
    dev = torch.device(device)
    for m in model.modules():
        anchors = getattr(m, 'anchors', None)
        if anchors is not None and isinstance(anchors, list) and len(anchors) > 0:
            if isinstance(anchors[0], torch.Tensor):
                m.anchors = [a.to(dev) for a in anchors]


def apply_dynamic_int8_linear_ptq(model: nn.Module) -> nn.Module:
    """
    CPU dynamic PTQ: quantize all nn.Linear submodules to int8 (weights only at runtime).
    Model must already hold float weights on CPU.
    """
    model = model.cpu()
    model.eval()
    q = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    move_anchor_tensors_to_device(q, 'cpu')
    patch_detector_int8_cpu_forward_gpu_nms(q)
    return q


def move_batch_dict_tensors_to_cuda(batch_dict: dict) -> None:
    """
    Move every torch.Tensor in batch_dict (and list-of-tensor values) to CUDA.
    Needed so iou3d_nms and generate_recall_record see GPU tensors (gt_boxes, preds, etc.).
    """
    for k, v in list(batch_dict.items()):
        if isinstance(v, torch.Tensor):
            batch_dict[k] = v.cuda(non_blocking=False)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            batch_dict[k] = [t.cuda(non_blocking=False) for t in v]


def _int8_cpu_forward_gpu_nms_impl(self, batch_dict):
    """Inference-only: module_list on CPU, NMS post_process on CUDA."""
    self.eval()
    nvtx = getattr(torch.cuda, 'nvtx', None)
    if nvtx is not None:
        nvtx.range_push('forward')
    for cur_module in self.module_list:
        batch_dict = cur_module(batch_dict)
    if nvtx is not None:
        nvtx.range_pop()
    if nvtx is not None:
        nvtx.range_push('post_processing')
    move_batch_dict_tensors_to_cuda(batch_dict)
    pred_dicts, recall_dicts = self.post_processing(batch_dict)
    if nvtx is not None:
        nvtx.range_pop()
    return pred_dicts, recall_dicts


def patch_detector_int8_cpu_forward_gpu_nms(model: nn.Module) -> None:
    """
    Replace forward with CPU backbone + GPU NMS (OpenPCDet iou3d_nms is CUDA-only).
    Training is not supported on this path.
    """
    if not hasattr(model, 'module_list') or not hasattr(model, 'post_processing'):
        return
    model.forward = types.MethodType(_int8_cpu_forward_gpu_nms_impl, model)


def load_batch_to_device(batch_dict: dict, device: torch.device) -> None:
    """
    Populate batch_dict with torch tensors on `device` (mirrors load_data_to_gpu for CPU).
    Skips non-array keys same as pcdet load_data_to_gpu.
    """
    # Local copy of dtype inference to avoid importing private helpers
    def _infer_dtype(np_array, key):
        if key == 'image_shape':
            return torch.int32
        kind = np_array.dtype.kind
        if kind in ('i', 'u'):
            return torch.int32 if np_array.dtype.itemsize <= 4 else torch.int64
        if kind == 'b':
            return torch.bool
        return torch.float32

    for key, val in list(batch_dict.items()):
        if key == 'camera_imgs':
            batch_dict[key] = val.to(device, non_blocking=False)
        elif not isinstance(val, np.ndarray):
            continue
        elif key in (
            'frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos',
        ):
            continue
        elif key in ('images',):
            try:
                import kornia
                batch_dict[key] = kornia.image_to_tensor(val).float().to(device).contiguous()
            except ImportError:
                batch_dict[key] = torch.from_numpy(val).float().to(device)
        else:
            dtype = _infer_dtype(val, key)
            batch_dict[key] = torch.from_numpy(val).to(dtype=dtype, device=device)


def inference_uses_cpu_int8(args) -> bool:
    return bool(getattr(args, 'int8', False))
