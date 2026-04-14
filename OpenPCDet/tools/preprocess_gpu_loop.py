"""
Shared helpers for --preprocess_gpu in profile_suite / energy_monitor / kitti_eval_export.
Aligns voxel params and GPU voxelization with inference.py (points_to_voxels_gpu / voxelize_tensor).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from gpu_voxelizer import build_batch_dict_from_gpu_voxels, points_to_voxels_gpu, voxelize_tensor


def resolve_voxel_params(cfg) -> dict[str, Any]:
    """Read voxelization settings from cfg.DATA_CONFIG (same as inference.py)."""
    data_cfg = cfg.DATA_CONFIG
    point_cloud_range = list(data_cfg.POINT_CLOUD_RANGE)
    voxel_cfg = None
    for p in data_cfg.DATA_PROCESSOR:
        if getattr(p, 'NAME', None) == 'transform_points_to_voxels':
            voxel_cfg = p
            break
    if voxel_cfg is None:
        raise ValueError('DATA_CONFIG has no transform_points_to_voxels; cannot use --preprocess_gpu')
    voxel_size = list(voxel_cfg.VOXEL_SIZE)
    max_pts = int(voxel_cfg.MAX_POINTS_PER_VOXEL)
    mode = 'test'
    max_voxels = int(voxel_cfg.MAX_NUMBER_OF_VOXELS.get(mode, voxel_cfg.MAX_NUMBER_OF_VOXELS.get('train', 40000)))
    use_lead_xyz = getattr(getattr(cfg.MODEL, 'VFE', None), 'USE_ABSLOTE_XYZ', True)
    return {
        'point_cloud_range': point_cloud_range,
        'voxel_size': voxel_size,
        'max_points_per_voxel': max_pts,
        'max_num_voxels': max_voxels,
        'use_lead_xyz': use_lead_xyz,
    }


def build_compiled_voxelize_fn(
    point_cloud_range: list,
    voxel_size: list,
    max_pts: int,
    max_voxels: int,
    use_lead_xyz: bool,
    logger,
):
    """Optional torch.compile(dynamic=True) wrapper around voxelize_tensor (inference.py parity)."""
    if not hasattr(torch, 'compile'):
        logger.warning('--compile_voxelizer set but torch.compile not available; using plain voxelizer')
        return None
    range_min_t = torch.tensor(point_cloud_range[:3], dtype=torch.float32, device='cuda')
    range_max_t = torch.tensor(point_cloud_range[3:], dtype=torch.float32, device='cuda')
    voxel_size_t = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
    grid_size = np.round(
        (np.array(point_cloud_range[3:]) - np.array(point_cloud_range[:3])) / np.array(voxel_size)
    ).astype(np.int64)
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    logger.info('Voxelizer wrapped with torch.compile(dynamic=True)')

    def _voxelize_fn(pts):
        return voxelize_tensor(
            pts, range_min_t, range_max_t, voxel_size_t,
            nx, ny, nz, max_pts, max_voxels, use_lead_xyz,
        )

    return torch.compile(_voxelize_fn, mode='default', dynamic=True)


def numpy_points_to_cuda(points_np: np.ndarray) -> torch.Tensor:
    """H2D for LiDAR points only (inference.py NVTX data_to_gpu for points)."""
    if not isinstance(points_np, np.ndarray):
        raise TypeError('points must be numpy before GPU transfer')
    return torch.from_numpy(points_np).float().cuda()


def gpu_voxelize_and_build_batch_dict(
    points_gpu: torch.Tensor,
    work: dict,
    voxel_params: dict,
    compiled_voxelize,
    batch_size: int,
) -> dict:
    """
    GPU voxelization + build_batch_dict_from_gpu_voxels + KITTI batch metadata.
    Matches inference.py NVTX pre_processing (voxelize + build batch).
    """
    if batch_size != 1:
        raise ValueError('--preprocess_gpu currently requires batch_size=1 (same as inference.py single-frame path).')
    pc_range = voxel_params['point_cloud_range']
    vs = voxel_params['voxel_size']
    max_pts = voxel_params['max_points_per_voxel']
    max_vox = voxel_params['max_num_voxels']
    use_lead = voxel_params['use_lead_xyz']

    if compiled_voxelize is not None:
        voxels, coords, num_pts = compiled_voxelize(points_gpu)
    else:
        voxels, coords, num_pts = points_to_voxels_gpu(
            points_gpu,
            point_cloud_range=pc_range,
            voxel_size=vs,
            max_points_per_voxel=max_pts,
            max_num_voxels=max_vox,
            use_lead_xyz=use_lead,
            device=torch.device('cuda'),
        )

    batch_dict = build_batch_dict_from_gpu_voxels(
        voxels, coords, num_pts, frame_id=0, batch_size=batch_size,
    )
    batch_dict['frame_id'] = np.array([work['frame_id']], dtype=object)

    for k, v in work.items():
        if k in batch_dict:
            continue
        batch_dict[k] = v

    gb = batch_dict.get('gt_boxes')
    if isinstance(gb, np.ndarray) and gb.ndim == 2:
        batch_dict['gt_boxes'] = np.expand_dims(gb, axis=0)
    g2 = batch_dict.get('gt_boxes2d')
    if isinstance(g2, np.ndarray) and g2.ndim == 2:
        batch_dict['gt_boxes2d'] = np.expand_dims(g2, axis=0)

    calib = batch_dict.get('calib')
    if calib is not None and not isinstance(calib, list):
        batch_dict['calib'] = [calib]

    imsh = batch_dict.get('image_shape')
    if imsh is not None:
        if isinstance(imsh, np.ndarray) and imsh.ndim == 1:
            batch_dict['image_shape'] = np.expand_dims(imsh, axis=0)
        elif isinstance(imsh, (tuple, list)) and not isinstance(imsh[0], (list, tuple, np.ndarray)):
            batch_dict['image_shape'] = np.array([imsh], dtype=np.int64)

    return batch_dict


def cpu_prepared_dict_to_gpu_batch(
    cpu_dict: dict,
    voxel_params: dict,
    compiled_voxelize,
    batch_size: int,
) -> dict:
    """
    Consume numpy `points` from prepare_data_for_gpu_voxelization output; produce batch_dict
    with CUDA voxels + original metadata for load_data_to_gpu.
    """
    work = dict(cpu_dict)
    points_np = work.pop('points', None)
    if points_np is None:
        raise KeyError('cpu_dict must contain points after CPU preprocessing')
    points_gpu = numpy_points_to_cuda(points_np)
    return gpu_voxelize_and_build_batch_dict(
        points_gpu, work, voxel_params, compiled_voxelize, batch_size,
    )


def dataset_supports_gpu_voxel(dataset) -> bool:
    return hasattr(dataset, 'fetch_sample_for_gpu_voxel')


class GpuVoxelDatasetProxy:
    """Wrap a dataset so DataLoader indexing uses fetch_sample_for_gpu_voxel (CPU prep, numpy points)."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, index):
        return self._base.fetch_sample_for_gpu_voxel(index)


def build_preprocess_gpu_dataloader(dataset, cfg, args, logger, compiled_voxelize):
    """
    DataLoader yielding batch_dict with CUDA voxels (batch_size=1). See inference.py --preprocess_gpu.
    """
    if not dataset_supports_gpu_voxel(dataset):
        raise ValueError('Dataset has no fetch_sample_for_gpu_voxel (KITTI required for this path).')
    if int(getattr(args, 'batch_size', 1)) != 1:
        raise ValueError('--preprocess_gpu requires batch_size=1.')
    voxel_params = resolve_voxel_params(cfg)
    collate = collate_gpu_voxel_batch(voxel_params, compiled_voxelize)
    # Collate runs GPU voxelization; cannot use forked DataLoader workers with CUDA.
    num_workers = 0
    if int(getattr(args, 'workers', 0)) > 0:
        logger.warning(
            'preprocess_gpu: forcing num_workers=0 (GPU voxelization runs in collate; incompatible with forked workers).'
        )
    if getattr(args, 'compile_voxelizer', False):
        logger.info('compile_voxelizer: torch.compile voxelizer active (still num_workers=0).')
    return DataLoader(
        GpuVoxelDatasetProxy(dataset),
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate,
        pin_memory=False,
        drop_last=False,
    )


def collate_gpu_voxel_batch(voxel_params: dict, compiled_voxelize):
    """
    Build a DataLoader collate_fn for batch_size=1: CPU-prepared dict -> GPU voxels.
    """

    def _collate(batch_list: list) -> dict:
        if len(batch_list) != 1:
            raise ValueError('GPU voxel collate supports batch_size=1 only')
        return cpu_prepared_dict_to_gpu_batch(batch_list[0], voxel_params, compiled_voxelize, batch_size=1)

    return _collate
