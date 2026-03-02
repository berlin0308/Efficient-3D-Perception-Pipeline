"""
GPU voxelization for PointPillar-style preprocessing.
Matches DATA_CONFIG: point_cloud_range, voxel_size, max_points_per_voxel, max_num_voxels.
Used when inference.py is run with --preprocess_gpu.
"""
import numpy as np
import torch


def points_to_voxels_gpu(
    points_input,
    point_cloud_range,
    voxel_size,
    max_points_per_voxel=32,
    max_num_voxels=40000,
    use_lead_xyz=True,
    device=None,
):
    """
    Voxelize points on GPU. PointPillar config: range [0,-39.68,-3, 69.12,39.68,1], size [0.16,0.16,4].

    Args:
        points_input: (N, 4) float32 numpy or torch.Tensor (xyz + intensity). If Tensor, must be on target device.
        point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size: [vx, vy, vz].
        max_points_per_voxel: pad/truncate to this many points per voxel.
        max_num_voxels: cap number of voxels (take first M).
        use_lead_xyz: if False, drop xyz from voxel features (keep intensity only).
        device: torch device for output tensors (used when points_input is numpy).

    Returns:
        voxels: (M, max_points_per_voxel, C) C=4 if use_lead_xyz else 1.
        coordinates: (M, 3) int64 (iz, iy, ix) grid indices.
        num_points: (M,) int32.
    """
    if device is None:
        device = torch.device("cuda")
    range_min = np.array(point_cloud_range[:3], dtype=np.float32)
    range_max = np.array(point_cloud_range[3:], dtype=np.float32)
    voxel_size_np = np.array(voxel_size, dtype=np.float32)
    grid_size = np.round((range_max - range_min) / voxel_size_np).astype(np.int64)
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

    if isinstance(points_input, np.ndarray):
        # CPU path: mask in numpy then move to GPU
        mask = (
            (points_input[:, 0] >= range_min[0]) & (points_input[:, 0] < range_max[0]) &
            (points_input[:, 1] >= range_min[1]) & (points_input[:, 1] < range_max[1]) &
            (points_input[:, 2] >= range_min[2]) & (points_input[:, 2] < range_max[2])
        )
        points_np = points_input[mask]
        if points_np.shape[0] == 0:
            M, C = 0, (4 if use_lead_xyz else 1)
            return (
                torch.zeros((M, max_points_per_voxel, C), dtype=torch.float32, device=device),
                torch.zeros((M, 3), dtype=torch.int64, device=device),
                torch.zeros((M,), dtype=torch.int32, device=device),
            )
        points = torch.from_numpy(points_np).float().to(device)
    else:
        # points_input is already on GPU (caller did data_to_gpu first)
        range_min_t = torch.tensor(range_min, device=points_input.device, dtype=points_input.dtype)
        range_max_t = torch.tensor(range_max, device=points_input.device, dtype=points_input.dtype)
        mask = (
            (points_input[:, 0] >= range_min_t[0]) & (points_input[:, 0] < range_max_t[0]) &
            (points_input[:, 1] >= range_min_t[1]) & (points_input[:, 1] < range_max_t[1]) &
            (points_input[:, 2] >= range_min_t[2]) & (points_input[:, 2] < range_max_t[2])
        )
        points = points_input[mask]
        if points.shape[0] == 0:
            M, C = 0, (4 if use_lead_xyz else 1)
            return (
                torch.zeros((M, max_points_per_voxel, C), dtype=torch.float32, device=points.device),
                torch.zeros((M, 3), dtype=torch.int64, device=points.device),
                torch.zeros((M,), dtype=torch.int32, device=points.device),
            )
    N = points.shape[0]
    device = points.device

    # Voxel grid indices: (ix, iy, iz)
    grid_float = (points[:, :3] - torch.tensor(range_min, device=device, dtype=torch.float32)) / torch.tensor(
        voxel_size_np, device=device, dtype=torch.float32
    )
    grid_int = grid_float.long().clamp(0)
    grid_int[:, 0] = torch.clamp(grid_int[:, 0], max=nx - 1)
    grid_int[:, 1] = torch.clamp(grid_int[:, 1], max=ny - 1)
    grid_int[:, 2] = torch.clamp(grid_int[:, 2], max=nz - 1)
    # Linear voxel id: iz * (nx*ny) + iy * nx + ix
    lid = grid_int[:, 2] * (nx * ny) + grid_int[:, 1] * nx + grid_int[:, 0]

    # Sort by lid
    sorted_idx = torch.argsort(lid)
    sorted_points = points[sorted_idx]
    sorted_lid = lid[sorted_idx]

    # Unique consecutive to get segment boundaries
    unique_lid, counts = torch.unique_consecutive(sorted_lid, return_counts=True)
    M = min(int(unique_lid.shape[0]), max_num_voxels)
    if M == 0:
        C = 4 if use_lead_xyz else 1
        voxels = torch.zeros((0, max_points_per_voxel, C), dtype=torch.float32, device=device)
        coordinates = torch.zeros((0, 3), dtype=torch.int64, device=device)
        num_points = torch.zeros((0,), dtype=torch.int32, device=device)
        return voxels, coordinates, num_points

    cumsum = torch.cat([torch.tensor([0], device=device, dtype=torch.long), counts.cumsum(0)])
    C = 4 if use_lead_xyz else 1
    counts_m = counts[:M]
    num_points = counts_m.clamp(max=max_points_per_voxel).to(torch.int32)

    # Coordinates (M, 3) in (z, y, x): first point of each voxel
    first_idx = cumsum[:M]
    coords_idx = sorted_idx[first_idx]
    coordinates = grid_int[coords_idx][:, [2, 1, 0]]

    # Voxels (M, max_points_per_voxel, C): vectorized gather (no Python loop)
    row_offsets = cumsum[:M]
    slot = torch.arange(max_points_per_voxel, device=device, dtype=torch.long).unsqueeze(0)
    point_index = (row_offsets.unsqueeze(1) + slot).clamp(max=N - 1)
    valid = slot < num_points.unsqueeze(1)
    point_index = torch.where(valid, point_index, torch.zeros_like(point_index))
    voxels = sorted_points[point_index, :C] * valid.unsqueeze(2).float()

    if not use_lead_xyz:
        voxels = voxels[..., 3:]  # (M, 32, 1)
    return voxels, coordinates, num_points


def build_batch_dict_from_gpu_voxels(voxels, coordinates, num_points, frame_id=0, batch_size=1):
    """
    Build a batch_dict compatible with PointPillar forward when voxels are already on GPU.
    Adds batch index to voxel_coords so shape is (M, 4) with first column batch_idx.
    """
    # voxel_coords for batch: pad with batch index (batch_idx, z, y, x)
    batch_idx = torch.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype, device=coordinates.device)
    voxel_coords_batch = torch.cat([batch_idx, coordinates], dim=1)
    batch_dict = {
        "voxels": voxels,
        "voxel_coords": voxel_coords_batch,
        "voxel_num_points": num_points,
        "batch_size": batch_size,
        "frame_id": frame_id,
    }
    return batch_dict
