import torch

from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        nvtx = getattr(torch.cuda, 'nvtx', None)
        if not self.training and nvtx is not None:
            nvtx.range_push('forward')
        for cur_module in self.module_list:
            if not self.training and nvtx is not None:
                nvtx.range_push(cur_module.__class__.__name__)
            batch_dict = cur_module(batch_dict)
            if not self.training and nvtx is not None:
                nvtx.range_pop()
        if not self.training and nvtx is not None:
            nvtx.range_pop()

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if nvtx is not None:
                nvtx.range_push('post_processing')
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if nvtx is not None:
                nvtx.range_pop()
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
