"""
Online (real-time) inference: single-frame path, no DataLoader, latency-focused.
Use this to measure end-to-end latency and optionally simulate a fixed lidar rate (e.g. 10 Hz).
Run from OpenPCDet/tools: python test_online.py --cfg_file ... --ckpt ... --data_path ...
"""
import _init_path
import argparse
import time
from pathlib import Path

# Set CUDA device before importing torch
def _parse_cuda_id():
    p = argparse.ArgumentParser()
    p.add_argument('--cuda_id', type=int, default=0, help='CUDA device ID (default: 0)')
    args, _ = p.parse_known_args()
    return args.cuda_id

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(_parse_cuda_id())

import numpy as np
import torch

from demo import DemoDataset
from eval_utils.eval_utils import _nvtx_range
from gpu_voxelizer import points_to_voxels_gpu, build_batch_dict_from_gpu_voxels, voxelize_tensor
from model_loader import load_model_for_inference
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Online (real-time) inference: single-frame, latency, optional rate')
    parser.add_argument('--cfg_file', type=str, required=True, help='config file (e.g. cfgs/kitti_models/pointpillar.yaml)')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--data_path', type=str, required=True,
                        help='single point cloud file or directory of .bin/.npy files')
    parser.add_argument('--ext', type=str, default='.bin', help='point cloud extension (default: .bin)')
    parser.add_argument('--traced_model', type=str, default=None,
                        help='path to TorchScript .pt; if set, use traced forward + ckpt post_processing')
    parser.add_argument('--compile', action='store_true', help='wrap model with torch.compile()')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device ID (default: 0)')
    parser.add_argument('--rate', type=float, default=None,
                        help='simulate lidar rate in Hz (e.g. 10); sleep after each frame to match period')
    parser.add_argument('--warmup', type=int, default=5, help='number of warmup frames before measuring latency (default: 5)')
    parser.add_argument('--samples', type=int, default=None,
                        help='number of frames to run (default: all in data_path)')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model (same as test.py)')
    parser.add_argument('--nsight', action='store_true', default=False,
                        help='enable NVTX ranges for Nsight Systems; run with: nsys profile -o out python inference.py ... --nsight')
    parser.add_argument('--preprocess_gpu', action='store_true', default=False,
                        help='offload preprocessing to GPU (GPU voxelization); default is CPU preprocessing')
    parser.add_argument('--compile_voxelizer', action='store_true', default=False,
                        help='when using --preprocess_gpu, wrap voxelization with torch.compile(); no effect without --preprocess_gpu')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable AMP (FP16) inference via torch.autocast')
    parser.add_argument('--channels_last', action='store_true', default=False,
                        help='convert model to channels-last (NHWC) memory format to eliminate NCHW<->NHWC conversion overhead with Tensor Core')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    if getattr(args, 'nsight', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    logger = common_utils.create_logger()
    logger.info('Online inference: single-frame, latency-focused (no DataLoader, no eval metrics)')

    root_path = Path(args.data_path)
    if not root_path.exists():
        raise FileNotFoundError('data_path does not exist: %s' % args.data_path)

    dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=root_path,
        logger=logger,
        ext=args.ext,
    )
    n_total = len(dataset)
    if n_total == 0:
        raise ValueError('No samples found in data_path: %s' % args.data_path)

    n_run = args.samples if args.samples is not None else n_total
    n_run = min(n_run, n_total)
    logger.info('Samples: %d to run (of %d); warmup: %d' % (n_run, n_total, args.warmup))

    # Option A: data source prepared here (file list ready). Each frame: get_raw -> voxelize -> to_gpu -> forward.
    preprocess_gpu = getattr(args, 'preprocess_gpu', False)
    if preprocess_gpu:
        logger.info('Data source ready: %d frames (raw %s); per-frame pipeline: get_raw -> voxelize(GPU) -> forward'
                    % (n_total, args.ext))
        # Resolve voxel config from DATA_CONFIG (same as dataset prepare_data)
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
        max_voxels = int(voxel_cfg.MAX_NUMBER_OF_VOXELS.get('test', voxel_cfg.MAX_NUMBER_OF_VOXELS.get('train', 40000)))
        use_lead_xyz = getattr(getattr(cfg.MODEL, 'VFE', None), 'USE_ABSLOTE_XYZ', True)
        compile_voxelizer = getattr(args, 'compile_voxelizer', False)
        compiled_voxelize = None
        if compile_voxelizer:
            if hasattr(torch, 'compile'):
                logger.info('Voxelizer will be wrapped with torch.compile(dynamic=True)')
                range_min_t = torch.tensor(point_cloud_range[:3], dtype=torch.float32, device='cuda')
                range_max_t = torch.tensor(point_cloud_range[3:], dtype=torch.float32, device='cuda')
                voxel_size_t = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
                grid_size = np.round(
                    (np.array(point_cloud_range[3:]) - np.array(point_cloud_range[:3])) / np.array(voxel_size)
                ).astype(np.int64)
                nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

                def _voxelize_fn(pts):
                    return voxelize_tensor(
                        pts, range_min_t, range_max_t, voxel_size_t,
                        nx, ny, nz, max_pts, max_voxels, use_lead_xyz,
                    )

                compiled_voxelize = torch.compile(_voxelize_fn, mode='default', dynamic=True)
            else:
                compile_voxelizer = False
                logger.warning('--compile_voxelizer set but torch.compile not available; using plain voxelizer')
    else:
        compile_voxelizer = False
        compiled_voxelize = None
        logger.info('Data source ready: %d frames (raw %s); per-frame pipeline: get_raw -> voxelize(CPU) -> to_gpu -> forward'
                    % (n_total, args.ext))

    model = load_model_for_inference(cfg, args, logger, dataset, to_cpu=False)
    if getattr(args, 'channels_last', False):
        model = model.to(memory_format=torch.channels_last)
        logger.info('Model converted to channels-last (NHWC) memory format')

    period_sec = (1.0 / args.rate) if args.rate is not None else None
    if period_sec is not None:
        logger.info('Pacing: %.1f Hz (period %.3f s) - will sleep after each frame until next boundary', args.rate, period_sec)
    else:
        logger.info('Pacing: none (no --rate); frames run back-to-back')
    latencies_ms = []
    use_nvtx = getattr(args, 'nsight', False)
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(n_run):
            # Reset 10 Hz clock after warmup so first measured frame starts the period (avoids being "behind" from slow warmup)
            if period_sec is not None and i == args.warmup:
                t_start = time.perf_counter()

            # Strict 10 Hz: wait until next period boundary before starting this frame
            if period_sec is not None and i >= args.warmup:
                deadline = t_start + (i - args.warmup) * period_sec
                now = time.perf_counter()
                if now < deadline:
                    time.sleep(deadline - now)

            # NVTX only for measured frames (skip warmup)
            nvtx_this = use_nvtx and (i >= args.warmup)
            if preprocess_gpu:
                if nvtx_this:
                    with _nvtx_range('read_points'):
                        raw = dataset.get_raw(i)
                    with _nvtx_range('data_to_gpu'):
                        points_gpu = torch.from_numpy(raw['points']).float().cuda()
                    with _nvtx_range('pre_processing'):
                        if compiled_voxelize is not None:
                            with _nvtx_range('voxelize_compiled'):
                                voxels, coords, num_pts = compiled_voxelize(points_gpu)
                        else:
                            voxels, coords, num_pts = points_to_voxels_gpu(
                                points_gpu,
                                point_cloud_range=point_cloud_range,
                                voxel_size=voxel_size,
                                max_points_per_voxel=max_pts,
                                max_num_voxels=max_voxels,
                                use_lead_xyz=use_lead_xyz,
                                device=torch.device('cuda'),
                            )
                        data_dict = build_batch_dict_from_gpu_voxels(voxels, coords, num_pts, frame_id=i, batch_size=1)
                else:
                    raw = dataset.get_raw(i)
                    points_gpu = torch.from_numpy(raw['points']).float().cuda()
                    if compiled_voxelize is not None:
                        voxels, coords, num_pts = compiled_voxelize(points_gpu)
                    else:
                        voxels, coords, num_pts = points_to_voxels_gpu(
                            points_gpu,
                            point_cloud_range=point_cloud_range,
                            voxel_size=voxel_size,
                            max_points_per_voxel=max_pts,
                            max_num_voxels=max_voxels,
                            use_lead_xyz=use_lead_xyz,
                            device=torch.device('cuda'),
                        )
                    data_dict = build_batch_dict_from_gpu_voxels(voxels, coords, num_pts, frame_id=i, batch_size=1)
            else:
                if nvtx_this:
                    with _nvtx_range('read_points'):
                        raw = dataset.get_raw(i)
                    with _nvtx_range('pre_processing'):
                        data_dict = dataset.collate_batch([dataset.prepare_data(raw)])
                else:
                    raw = dataset.get_raw(i)
                    data_dict = dataset.collate_batch([dataset.prepare_data(raw)])
                if nvtx_this:
                    with _nvtx_range('data_to_gpu'):
                        load_data_to_gpu(data_dict)
                else:
                    load_data_to_gpu(data_dict)
            t0 = time.perf_counter()
            # Original (FP32):
            # if nvtx_this:
            #     with _nvtx_range('forward'):
            #         pred_dicts, _ = model(data_dict)
            # else:
            #     pred_dicts, _ = model(data_dict)
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if args.amp else torch.cuda.amp.autocast(enabled=False)
            with amp_ctx:
                if nvtx_this:
                    with _nvtx_range('forward'):
                        pred_dicts, _ = model(data_dict)
                else:
                    pred_dicts, _ = model(data_dict)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            if i >= args.warmup:
                latencies_ms.append(latency_ms)

            # Strict 10 Hz: sleep until next period boundary after this frame (visible as wait_10hz in Nsight)
            if period_sec is not None and i >= args.warmup:
                next_start = t_start + (i - args.warmup + 1) * period_sec
                now = time.perf_counter()
                if now < next_start:
                    sleep_duration = next_start - now
                    if nvtx_this:
                        with _nvtx_range('wait_10hz'):
                            time.sleep(sleep_duration)
                    else:
                        time.sleep(sleep_duration)

    if not latencies_ms:
        logger.info('No latency samples (warmup >= frames run). Run with --samples > --warmup.')
        return

    cold_start_ms = latencies_ms[0]
    steady = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms
    mean_ms = np.mean(steady)
    p50_ms = np.percentile(steady, 50)
    p99_ms = np.percentile(steady, 99)

    logger.info('Latency (ms) - cold start (first after warmup): %.2f' % cold_start_ms)
    logger.info('Latency (ms) - steady-state mean: %.2f  p50: %.2f  p99: %.2f  (n=%d)' %
                (mean_ms, p50_ms, p99_ms, len(steady)))
    if args.rate is not None:
        logger.info('Simulated rate: %.1f Hz (period %.3f s)' % (args.rate, period_sec))
    logger.info('Online inference done.')


if __name__ == '__main__':
    main()
