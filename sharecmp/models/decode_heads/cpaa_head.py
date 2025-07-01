from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from torch import Tensor


@MODELS.register_module()
class CPAAHead(BaseDecodeHead):
    def __init__(self,
                 polar_key: str,
                 pred_channels=3,
                 interpolate_mode='bilinear',
                 loss_decode=dict(
                     type='MSELoss',
                     reduction='mean'),
                 **kwargs):
        super().__init__(num_classes=pred_channels, loss_decode=loss_decode,
                         input_transform='multiple_select', **kwargs)
        self.conv_seg = None

        self.polar_key = polar_key
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        self.conv_polar = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.conv_polar.append(
                nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
            )

    def forward(self, inputs) -> List[Tensor]:
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            conv_polar = self.conv_polar[idx]
            x = resize(
                input=conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
            x = conv_polar(x)
            outs.append(x)

        return outs

    def loss_by_feat(self, outs: List[Tensor],
                     batch_data_samples: SampleList) -> dict:
        polar_label = batch_data_samples[0].get(self.polar_key)
        loss = dict()
        outs = [
            resize(
                input=x,
                size=polar_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            for x in outs
        ]

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for i, polar in enumerate(outs):
            idx_polar = self.in_index[i]

            for loss_decode in losses_decode:
                loss_name = f'{loss_decode.loss_name}_{idx_polar}'
                if loss_name not in loss:
                    loss[loss_name] = loss_decode(
                        polar,
                        polar_label)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        polar,
                        polar_label)
        return loss

    def predict_by_feat(self, outs: Tensor,
                        batch_img_metas: List[dict]) -> List[Tensor]:
        outs = [
            resize(
                input=x,
                size=batch_img_metas[0]['img_shape'],
                mode='bilinear',
                align_corners=self.align_corners)
            for x in outs
        ]
        return outs
