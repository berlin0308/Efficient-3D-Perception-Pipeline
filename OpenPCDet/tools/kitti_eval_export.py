"""
Run KITTI validation evaluation once and write numeric metrics to JSON.

Used by collect_research_metrics.py so runs.csv can include official AP (R40).
Must be launched with cwd=OpenPCDet/tools/ (same as test.py).
"""
import _init_path  # noqa: F401
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def _parse_cuda_id_early():
    p = argparse.ArgumentParser()
    p.add_argument('--cuda_id', type=int, default=0)
    args, _ = p.parse_known_args()
    return args.cuda_id


os.environ['CUDA_VISIBLE_DEVICES'] = str(_parse_cuda_id_early())

import torch  # noqa: E402

from eval_utils import eval_utils  # noqa: E402
from model_loader import load_model_for_inference  # noqa: E402
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from preprocess_gpu_loop import (  # noqa: E402
    build_compiled_voxelize_fn,
    build_preprocess_gpu_dataloader,
    dataset_supports_gpu_voxel,
    resolve_voxel_params,
)
from pcdet.utils import common_utils  # noqa: E402


def _json_safe_scalar(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def ret_dict_to_jsonable(ret_dict: dict) -> dict:
    out = {}
    for k, v in ret_dict.items():
        try:
            out[str(k)] = _json_safe_scalar(v)
        except (TypeError, ValueError):
            continue
    return out


def main():
    parser = argparse.ArgumentParser(description='KITTI eval once -> JSON metrics')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--eval_result_dir', type=str, required=True, help='eval logs and pickles')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        '--memory_opt_scatter',
        action='store_true',
        default=False,
        help='HWC coalesced scatter (same as profile_suite)',
    )
    parser.add_argument(
        '--memory_opt_conv2d',
        action='store_true',
        default=False,
        help='channels_last BEV path (same as profile_suite)',
    )
    parser.add_argument('--int8', action='store_true', default=False,
                        help='CPU dynamic Linear INT8 PTQ (same as profile_suite --int8)')
    parser.add_argument(
        '--preprocess_gpu',
        action='store_true',
        default=False,
        help='GPU voxelization DataLoader (inference.py-aligned; batch_size=1)',
    )
    parser.add_argument(
        '--compile_voxelizer',
        action='store_true',
        default=False,
        help='with --preprocess_gpu, torch.compile voxelizer',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=20,
        help='warmup batches before infer_time metering (set --infer_time to use)',
    )
    parser.add_argument('--infer_time', action='store_true', default=False)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    if ns.max_samples is not None and ns.max_samples <= 0:
        ns.max_samples = None

    class Args:
        pass

    args = Args()
    args.cfg_file = ns.cfg_file
    args.ckpt = ns.ckpt
    args.cuda_id = ns.cuda_id
    args.batch_size = ns.batch_size
    args.workers = ns.workers
    args.compile = ns.compile
    args.amp = ns.amp
    args.memory_opt_scatter = ns.memory_opt_scatter
    args.memory_opt_conv2d = ns.memory_opt_conv2d
    args.int8 = ns.int8
    args.warmup = ns.warmup
    args.infer_time = ns.infer_time
    args.max_samples = ns.max_samples
    args.save_to_file = False
    args.launcher = 'none'
    args.profile = False
    args.nsight = False
    args.traced_model = None
    args.pretrained_model = None
    args.compile_debug = False
    args.preprocess_gpu = ns.preprocess_gpu
    args.compile_voxelizer = ns.compile_voxelizer

    cfg_from_yaml_file(ns.cfg_file, cfg)
    cfg.TAG = Path(ns.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(ns.cfg_file.split('/')[1:-1])
    np.random.seed(1024)
    if ns.set_cfgs is not None:
        cfg_from_list(ns.set_cfgs, cfg)

    eval_result_dir = Path(ns.eval_result_dir)
    eval_result_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_result_dir / 'log_kitti_eval.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info('********************** kitti_eval_export **********************')
    for key, val in vars(args).items():
        logger.info('{:24} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    if args.preprocess_gpu:
        if args.int8:
            raise SystemExit('--preprocess_gpu is incompatible with --int8.')
        if int(args.batch_size) != 1:
            raise SystemExit('--preprocess_gpu requires --batch_size 1.')
        if not dataset_supports_gpu_voxel(test_set):
            raise SystemExit('--preprocess_gpu requires KITTI.')
        vp = resolve_voxel_params(cfg)
        compiled_vox = None
        if args.compile_voxelizer:
            compiled_vox = build_compiled_voxelize_fn(
                vp['point_cloud_range'], vp['voxel_size'], vp['max_points_per_voxel'],
                vp['max_num_voxels'], vp['use_lead_xyz'], logger,
            )
        test_loader = build_preprocess_gpu_dataloader(test_set, cfg, args, logger, compiled_vox)
        logger.info('kitti_eval_export: GPU voxelization loader (inference.py-aligned)')

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt else []
    epoch_id = num_list[-1] if len(num_list) > 0 else 'export'

    with torch.inference_mode():
        model = load_model_for_inference(cfg, args, logger, test_set, to_cpu=False)
        ret_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, epoch_id, logger,
            dist_test=False, result_dir=eval_result_dir,
        )

    payload = {
        'metrics': ret_dict_to_jsonable(ret_dict),
        'cfg_file': str(Path(ns.cfg_file).resolve()),
        'ckpt': str(Path(ns.ckpt).resolve()),
        'eval_result_dir': str(eval_result_dir.resolve()),
    }
    out_path = Path(ns.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    logger.info('Wrote %s', out_path)


if __name__ == '__main__':
    main()
