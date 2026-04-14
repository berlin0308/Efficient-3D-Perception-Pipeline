import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.profiler
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def _load_eval_batch(batch_dict, args):
    if getattr(args, 'int8', False):
        import int8_utils
        int8_utils.load_batch_to_device(batch_dict, torch.device('cpu'))
    else:
        load_data_to_gpu(batch_dict)


def _nvtx_range(name):
    """Context manager for NVTX range (no-op if torch.cuda.nvtx not available)."""
    nvtx = getattr(torch.cuda, 'nvtx', None)
    if nvtx is None:
        from contextlib import nullcontext
        return nullcontext()
    return _NvtxRange(name)


class _NvtxRange:
    def __init__(self, name):
        self.name = name
        self.nvtx = getattr(torch.cuda, 'nvtx', None)

    def __enter__(self):
        if self.nvtx is not None:
            self.nvtx.range_push(self.name)

    def __exit__(self, *args):
        if self.nvtx is not None:
            self.nvtx.range_pop()


def _forward_model(model, batch_dict, args):
    """
    Run model(batch_dict). When --compile is set, pass only tensor/int/float keys
    so torch.compile (Dynamo) does not see numpy.str_ or other unsupported types.
    """
    amp_enabled = bool(getattr(args, 'amp', False)) and not getattr(args, 'int8', False)
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled) \
        if torch.cuda.is_available() and not getattr(args, 'int8', False) else nullcontext()

    if not getattr(args, 'compile', False):
        with amp_ctx:
            return model(batch_dict)
    safe = {k: v for k, v in batch_dict.items() if isinstance(v, (torch.Tensor, int, float))}
    with amp_ctx:
        pred_dicts, ret_dict = model(safe)
    batch_dict.update(safe)
    return pred_dicts, ret_dict


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    # Offline eval: dataloader (disk + CPU preprocess) -> load_data_to_gpu -> forward -> generate_prediction_dicts (GPU->CPU for metrics).
    # Real-time: single-frame, latency from scan ready to detections; no dataset/workers. See README "Test pipeline vs real-time".
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    max_samples = getattr(args, 'max_samples', None)
    num_samples_evaluated = 0

    def batch_size_from_dict(batch_dict):
        if 'frame_id' in batch_dict and batch_dict['frame_id'] is not None:
            return len(batch_dict['frame_id'])
        return getattr(args, 'batch_size', 1)

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    warmup = getattr(args, 'warmup', 20)
    use_profile = getattr(args, 'profile', False)
    if use_profile:
        dataloader_iter = iter(dataloader)
        if warmup > 0 and cfg.LOCAL_RANK == 0:
            logger.info('Running %d warmup steps (no timing/profiling)' % warmup)
        for _ in range(min(warmup, len(dataloader))):
            batch_dict = next(dataloader_iter)
            _load_eval_batch(batch_dict, args)
            with torch.inference_mode():
                _forward_model(model, batch_dict, args)
        profile_steps = min(getattr(args, 'profile_steps', 20), len(dataloader))
        profile_output_path = getattr(args, 'profile_output', None) or (result_dir / 'torch_profile_trace.json')
        if isinstance(profile_output_path, str):
            profile_output_path = Path(profile_output_path)
        if 'result_dir' in str(profile_output_path) or not profile_output_path.parent.exists():
            profile_output_path = result_dir / profile_output_path.name
        profile_output_path.parent.mkdir(parents=True, exist_ok=True)
        do_profile_export = use_profile and (cfg.LOCAL_RANK == 0)

        def run_one_step(batch_dict):
            _load_eval_batch(batch_dict, args)
            step_start = time.time() if getattr(args, 'infer_time', False) else None
            with torch.inference_mode():
                pred_dicts, ret_dict = _forward_model(model, batch_dict, args)
            disp_dict = {}
            if getattr(args, 'infer_time', False) and step_start is not None:
                infer_time_meter.update((time.time() - step_start) * 1000)
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if args.save_to_file else None
            )
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()
            return annos

        if do_profile_export:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            ) as prof:
                for _ in range(profile_steps):
                    if max_samples is not None and num_samples_evaluated >= max_samples:
                        break
                    batch_dict = next(dataloader_iter)
                    annos = run_one_step(batch_dict)
                    det_annos += annos
                    num_samples_evaluated += batch_size_from_dict(batch_dict)
            prof.export_chrome_trace(str(profile_output_path))
            logger.info('Profile trace saved to %s' % profile_output_path)
        else:
            for _ in range(profile_steps):
                if max_samples is not None and num_samples_evaluated >= max_samples:
                    break
                batch_dict = next(dataloader_iter)
                annos = run_one_step(batch_dict)
                det_annos += annos
                num_samples_evaluated += batch_size_from_dict(batch_dict)

        for batch_dict in dataloader_iter:
            if max_samples is not None and num_samples_evaluated >= max_samples:
                break
            annos = run_one_step(batch_dict)
            det_annos += annos
            num_samples_evaluated += batch_size_from_dict(batch_dict)
    elif getattr(args, 'nsight', False):
        dataloader_iter = iter(dataloader)
        if warmup > 0 and cfg.LOCAL_RANK == 0:
            logger.info('Running %d warmup steps (no NVTX/timing)' % warmup)
        for _ in range(min(warmup, len(dataloader))):
            batch_dict = next(dataloader_iter)
            _load_eval_batch(batch_dict, args)
            with torch.inference_mode():
                _forward_model(model, batch_dict, args)
        nsight_steps = min(getattr(args, 'nsight_steps', 20), len(dataloader))
        if cfg.LOCAL_RANK == 0:
            logger.info('Nsight mode: running %d steps with NVTX ranges (get_batch, data_to_gpu, forward, postprocess, generate_prediction_dicts)' % nsight_steps)

        def run_one_step_nsight(batch_dict):
            with _nvtx_range('data_to_gpu'):
                _load_eval_batch(batch_dict, args)
            step_start = time.time() if getattr(args, 'infer_time', False) else None
            with _nvtx_range('forward'):
                with torch.inference_mode():
                    pred_dicts, ret_dict = _forward_model(model, batch_dict, args)
            disp_dict = {}
            if getattr(args, 'infer_time', False) and step_start is not None:
                infer_time_meter.update((time.time() - step_start) * 1000)
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
            with _nvtx_range('postprocess'):
                statistics_info(cfg, ret_dict, metric, disp_dict)
                with _nvtx_range('generate_prediction_dicts'):
                    annos = dataset.generate_prediction_dicts(
                        batch_dict, pred_dicts, class_names,
                        output_path=final_output_dir if args.save_to_file else None
                    )
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()
            return annos

        for _ in range(nsight_steps):
            if max_samples is not None and num_samples_evaluated >= max_samples:
                break
            with _nvtx_range('get_batch'):
                batch_dict = next(dataloader_iter)
            annos = run_one_step_nsight(batch_dict)
            det_annos += annos
            num_samples_evaluated += batch_size_from_dict(batch_dict)
    else:
        for i, batch_dict in enumerate(dataloader):
            if max_samples is not None and num_samples_evaluated >= max_samples:
                break
            _load_eval_batch(batch_dict, args)

            do_timing = getattr(args, 'infer_time', False) and i >= warmup
            if do_timing:
                start_time = time.time()

            with torch.inference_mode():
                pred_dicts, ret_dict = _forward_model(model, batch_dict, args)

            disp_dict = {}

            if do_timing:
                inference_time = time.time() - start_time
                infer_time_meter.update(inference_time * 1000)
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

            statistics_info(cfg, ret_dict, metric, disp_dict)
            # Full GPU->CPU for evaluation (KITTI-style annos); real-time may keep minimal detections on GPU or copy less.
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if args.save_to_file else None
            )
            det_annos += annos
            num_samples_evaluated += batch_size_from_dict(batch_dict)
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    n_eval = num_samples_evaluated if num_samples_evaluated > 0 else len(dataloader.dataset)
    sec_per_example = (time.time() - start_time) / max(n_eval, 1)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    if max_samples is not None and num_samples_evaluated > 0:
        logger.info('Evaluated first %d samples (--max_samples=%d).' % (num_samples_evaluated, max_samples))

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    eval_kwargs = dict(
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )
    if max_samples is not None and len(det_annos) < len(dataset):
        eval_kwargs['max_eval_samples'] = len(det_annos)
    result_str, result_dict = dataset.evaluation(det_annos, class_names, **eval_kwargs)

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
