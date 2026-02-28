"""
Export PointPillar (or the model from the given cfg/ckpt) to TorchScript.
Uses the same config/checkpoint as 06_test.sh by default.
Requires CUDA (model and trace run on GPU).
Output: forward(voxels, voxel_num_points, voxel_coords) -> (batch_cls_preds, batch_box_preds).
"""
import _init_path
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils


def get_parser():
    parser = argparse.ArgumentParser(description='Export PointPillar to TorchScript')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml',
                        help='config file (same as 06_test.sh)')
    parser.add_argument('--ckpt', type=str, default='ckpt/pointpillar_7728.pth',
                        help='checkpoint to load (same as 06_test.sh)')
    parser.add_argument('--output', type=str, default='pointpillar_traced.pt',
                        help='output path for the TorchScript model')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path to one .bin file or a dir of .bin (for trace example). Default: ROOT/data/kitti/training/velodyne')
    parser.add_argument('--compile', action='store_true',
                        help='wrap model with torch.compile() before tracing (PyTorch 2.0+)')
    return parser


class DemoDatasetForExport(DatasetTemplate):
    """Minimal dataset to get one batch for tracing (same preprocessing as demo/test)."""

    def __init__(self, dataset_cfg, class_names, root_path, logger, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = Path(root_path)
        self.ext = ext
        if self.root_path.is_file():
            self.sample_file_list = [str(self.root_path)] if self.root_path.suffix == ext else []
        else:
            self.sample_file_list = [str(p) for p in sorted(self.root_path.glob(f'*{ext}'))]
        if not self.sample_file_list:
            raise FileNotFoundError(f'No *{ext} files under {root_path}')

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        path = self.sample_file_list[index]
        if self.ext == '.bin':
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        else:
            points = np.load(path)
        input_dict = {'points': points, 'frame_id': index}
        return self.prepare_data(data_dict=input_dict)


class PointPillarTraceWrapper(nn.Module):
    """
    Wrapper so we can trace with (voxels, voxel_num_points, voxel_coords).
    Scatter is inlined in forward (no submodule) so the traced graph shows one
    connected flow: voxel_coords and pillar_features both feed into the same
    scatter ops -> spatial_features -> backbone -> head.
    Forward returns (batch_cls_preds, batch_box_preds) for downstream NMS/postprocess.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        scatter = model.map_to_bev_module
        self.num_bev_features = int(scatter.num_bev_features)
        self.nx = int(scatter.nx)
        self.ny = int(scatter.ny)
        self.nz = int(scatter.nz)

    def forward(self, voxels, voxel_num_points, voxel_coords):
        batch_size = int(voxel_coords[:, 0].max().item()) + 1
        batch_dict = {
            'voxels': voxels,
            'voxel_num_points': voxel_num_points,
            'voxel_coords': voxel_coords,
            'batch_size': batch_size,
        }
        batch_dict = self.model.vfe(batch_dict)
        pillar_features = batch_dict['pillar_features']

        # Inlined scatter: spatial is created FROM pillar_features (new_zeros) so the
        # trace has one data flow; voxel_coords feeds indices into index_put_.
        batch_idx = voxel_coords[:, 0].long()
        spatial_idx = (
            voxel_coords[:, 1]
            + voxel_coords[:, 2] * self.nx
            + voxel_coords[:, 3]
        ).long()
        spatial = pillar_features.new_zeros(
            batch_size,
            self.num_bev_features,
            self.nz * self.nx * self.ny,
        )
        spatial[batch_idx, :, spatial_idx] = pillar_features
        spatial_features = spatial.view(
            batch_size, self.num_bev_features * self.nz, self.ny, self.nx
        )
        # Force graph edge: spatial_features explicitly depends on pillar_features
        # (0 * sum is zero; keeps Netron from drawing scatter as a separate branch)
        spatial_features = spatial_features + 0.0 * pillar_features.sum()

        batch_dict['spatial_features'] = spatial_features
        batch_dict = self.model.backbone_2d(batch_dict)
        batch_dict = self.model.dense_head(batch_dict)
        return batch_dict['batch_cls_preds'], batch_dict['batch_box_preds']


def main():
    args = get_parser().parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    logger = common_utils.create_logger()
    logger.info('Export config: %s', args.cfg_file)
    logger.info('Checkpoint: %s', args.ckpt)
    logger.info('Output: %s', args.output)

    root_path = Path(cfg.ROOT_DIR)
    data_path = args.data_path
    if data_path is None:
        data_path = root_path / 'data' / 'kitti' / 'training' / 'velodyne'
    else:
        data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'Data path for trace example not found: {data_path}')

    dataset = DemoDatasetForExport(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=data_path,
        logger=logger,
        ext='.bin',
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    if args.compile:
        if hasattr(torch, 'compile'):
            logger.info('Wrapping model with torch.compile()')
            model = torch.compile(model, mode='default')
        else:
            logger.warning('--compile set but torch.compile not available (PyTorch 2.0+); skipping')

    # One batch for tracing (same preprocessing as demo/test)
    data_dict = dataset[0]
    batch_dict = dataset.collate_batch([data_dict])
    voxels = torch.from_numpy(batch_dict['voxels']).float().cuda()
    voxel_num_points = torch.from_numpy(batch_dict['voxel_num_points']).int().cuda()
    voxel_coords = torch.from_numpy(batch_dict['voxel_coords']).int().cuda()

    wrapper = PointPillarTraceWrapper(model)
    wrapper.eval()

    # TorchScript-friendly: patch PFNLayer to avoid "if inputs.shape[0] > self.part" (tensor->bool) during trace
    from pcdet.models.backbones_3d.vfe import pillar_vfe
    _pfn_forward_orig = pillar_vfe.PFNLayer.forward

    def _pfn_forward_trace_friendly(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        x_concatenated = torch.cat([x, x_repeat], dim=2)
        return x_concatenated

    pillar_vfe.PFNLayer.forward = _pfn_forward_trace_friendly

    with torch.no_grad():
        logger.info('Tracing with example batch (voxels %s, coords %s)...',
                    tuple(voxels.shape), tuple(voxel_coords.shape))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='.*Converting a tensor to a Python (number|integer|boolean).*',
                category=torch.jit.TracerWarning,
            )
            traced = torch.jit.trace(
                wrapper,
                (voxels, voxel_num_points, voxel_coords),
                check_trace=True,
                strict=False,
            )

    pillar_vfe.PFNLayer.forward = _pfn_forward_orig

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    logger.info('Saved TorchScript model to %s', out_path)


if __name__ == '__main__':
    main()
