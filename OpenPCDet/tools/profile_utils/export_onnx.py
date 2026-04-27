"""
export_onnx.py — Export PointPillars backbone+head to ONNX for TensorRT.

Uses the same PointPillarTraceWrapper boundary as export.py:
  forward(voxels, voxel_num_points, voxel_coords) -> (batch_cls_preds, batch_box_preds)
NMS / post_processing stays in Python (same as TracedModelWrapper).

Dynamic axes: voxels dim-0 (num_voxels) is dynamic so TRT can handle varying
point cloud densities without rebuilding the engine.

Run from OpenPCDet/tools/:
    python profile_utils/export_onnx.py \
        --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt /path/to/pointpillar_7728.pth \
        --output pointpillar.onnx \
        [--data_path /path/to/velodyne]

Then build TRT engine (run in shell, not Python):
    trtexec --onnx=pointpillar.onnx \
            --fp16 \
            --minShapes=voxels:1x32x4,voxel_num_points:1,voxel_coords:1x4 \
            --optShapes=voxels:10000x32x4,voxel_num_points:10000,voxel_coords:10000x4 \
            --maxShapes=voxels:20000x32x4,voxel_num_points:20000,voxel_coords:20000x4 \
            --saveEngine=pointpillar_fp16.engine \
            --verbose
"""
import _init_path  # noqa: F401
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils

# Re-use the same wrapper and dataset from export.py
from profile_utils.export import DemoDatasetForExport, PointPillarTraceWrapper


def get_parser():
    parser = argparse.ArgumentParser(description='Export PointPillars to ONNX for TensorRT')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--ckpt', type=str, default='ckpt/pointpillar_7728.pth')
    parser.add_argument('--output', type=str, default='pointpillar.onnx')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (17 recommended for TRT 8.6+)')
    return parser


def main():
    args = get_parser().parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    logger = common_utils.create_logger()
    logger.info('Config   : %s', args.cfg_file)
    logger.info('Ckpt     : %s', args.ckpt)
    logger.info('Output   : %s', args.output)
    logger.info('Opset    : %d', args.opset)

    data_path = Path(args.data_path) if args.data_path else (
        Path(cfg.ROOT_DIR) / 'data' / 'kitti' / 'training' / 'velodyne'
    )
    if not data_path.exists():
        raise FileNotFoundError('Data path not found: %s' % data_path)

    dataset = DemoDatasetForExport(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=data_path,
        logger=logger,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()

    data_dict = dataset[0]
    batch_dict = dataset.collate_batch([data_dict])
    voxels           = torch.from_numpy(batch_dict['voxels']).float().cuda()
    voxel_num_points = torch.from_numpy(batch_dict['voxel_num_points']).int().cuda()
    voxel_coords     = torch.from_numpy(batch_dict['voxel_coords']).int().cuda()

    # Patch PFNLayer exactly as in export.py to avoid tensor→bool trace warnings
    from pcdet.models.backbones_3d.vfe import pillar_vfe
    _orig = pillar_vfe.PFNLayer.forward

    def _pfn_trace_friendly(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        return torch.cat([x, x_repeat], dim=2)

    pillar_vfe.PFNLayer.forward = _pfn_trace_friendly

    wrapper = PointPillarTraceWrapper(model)
    wrapper.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Exporting ONNX (voxels %s, coords %s)…', tuple(voxels.shape), tuple(voxel_coords.shape))
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
            torch.onnx.export(
                wrapper,
                (voxels, voxel_num_points, voxel_coords),
                str(out_path),
                opset_version=args.opset,
                input_names=['voxels', 'voxel_num_points', 'voxel_coords'],
                output_names=['batch_cls_preds', 'batch_box_preds'],
                dynamic_axes={
                    'voxels':           {0: 'num_voxels'},
                    'voxel_num_points': {0: 'num_voxels'},
                    'voxel_coords':     {0: 'num_voxels'},
                },
            )

    pillar_vfe.PFNLayer.forward = _orig
    logger.info('Saved ONNX → %s', out_path)
    logger.info('')
    logger.info('Next: build TRT engine with trtexec (see docstring at top of this file).')


if __name__ == '__main__':
    main()
