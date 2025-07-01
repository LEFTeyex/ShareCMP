from typing import Union, List, Dict

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


class ChannelAttention(nn.Module):
    def __init__(self,
                 channels: int,
                 reduction: int = 1):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        in_channels = channels * 2
        hid_channels = in_channels // reduction
        out_channels = channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.PReLU(),
            nn.Linear(hid_channels, out_channels),
            nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        b, fc, _, _ = x.shape

        avg_attn = self.avg_pool(x).flatten(1)
        max_attn = self.max_pool(x).flatten(1)
        attn = torch.cat((avg_attn, max_attn), dim=1)
        attn = self.channel_attn(attn).reshape(b, fc, 1, 1)
        return attn


class DWConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channel_ratio: Union[int, float] = 1,
                 dilation=1):
        super().__init__()
        hid_channels = int(in_channels * channel_ratio)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1, 1),
            nn.Conv2d(hid_channels, hid_channels, 3, 1,
                      padding=dilation, groups=hid_channels, dilation=dilation),
            nn.PReLU(),
            nn.Conv2d(hid_channels, out_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.dw_conv(x)


@MODELS.register_module()
class PGA(nn.Module):
    def __init__(self,
                 x_img_keys: List[str],
                 x_img_in_channels: List[int],
                 channel_ratio: Union[int, float] = 1,
                 reduction: int = 1):
        super().__init__()
        assert len(x_img_keys) == len(x_img_in_channels)
        self.x_img_keys = x_img_keys
        self.x_img_in_channels = x_img_in_channels
        # multi modal stem use conv(x, 32, 3, 1, 1), p_relu

        # fusion conv use conv(x * 4, 32, 3, 1, 1), p_relu
        #                 conv(32,     3, 3, 1, 1), p_relu
        self.x_img_stems = nn.ModuleDict()
        for x_key, x_in_c in zip(x_img_keys, x_img_in_channels):
            self.x_img_stems[x_key] = nn.Sequential(
                nn.Conv2d(x_in_c, 32, 3, 1, 1),
                nn.PReLU())

        channels = 32 * len(x_img_keys)
        self.attn_forward = DWConv(channels, channels, channel_ratio, dilation=2)
        self.channel_attn = ChannelAttention(channels, reduction)

        self.feed_forward = DWConv(channels, 3, channel_ratio)
        self.act = nn.PReLU()

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        x = []
        for k in self.x_img_keys:
            x.append(self.x_img_stems[k](inputs[k]))
        x = torch.cat(x, dim=1)

        attn = self.channel_attn(self.attn_forward(x))
        x = x + attn * x
        x = self.feed_forward(x)
        return self.act(x)
