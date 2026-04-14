from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def _infer_torch_dtype(np_array, key):
    if key == 'image_shape':
        return torch.int32
    kind = np_array.dtype.kind
    if kind in ('i', 'u'):
        return torch.int32 if np_array.dtype.itemsize <= 4 else torch.int64
    if kind == 'b':
        return torch.bool
    return torch.float32


def load_data_to_gpu(batch_dict, stream=None):
    """Copy numpy arrays in batch_dict to GPU.

    If *stream* is a ``torch.cuda.Stream``, all copies are issued non-blocking
    on that stream (host tensors are pinned first so DMA can proceed without
    stalling the CPU).  The caller is responsible for synchronising the stream
    before the GPU kernels that consume the data are launched.

    Without a stream the function falls back to the original synchronous path
    so existing call-sites are unaffected.
    """
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.cuda(non_blocking=True)
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda(non_blocking=True).contiguous()
        elif stream is not None:
            dtype = _infer_torch_dtype(val, key)
            pinned = torch.from_numpy(val).to(dtype).pin_memory()
            with torch.cuda.stream(stream):
                batch_dict[key] = pinned.to('cuda', non_blocking=True)
        else:
            dtype = _infer_torch_dtype(val, key)
            batch_dict[key] = torch.from_numpy(val).to(dtype=dtype).cuda(non_blocking=True)


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
