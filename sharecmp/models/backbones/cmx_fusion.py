from typing import Tuple, Optional

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.registry import MODELS
from torch import Tensor


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 1):
        super().__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels * 4, self.channels * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels * 4 // reduction, self.channels * 2),
            nn.Sigmoid())

    def forward(self, rgb: Tensor, x: Tensor) -> Tensor:
        b = rgb.shape[0]
        feats = torch.cat((rgb, x), dim=1)
        avg_feats = self.avg_pool(feats).reshape(b, self.channels * 2)
        max_feats = self.max_pool(feats).reshape(b, self.channels * 2)
        feats = torch.cat((avg_feats, max_feats), dim=1)
        attn = self.mlp(feats).reshape(b, 2, self.channels, 1, 1)
        attn = attn.permute(1, 0, 2, 3, 4)  # 2 b c 1 1
        return attn


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 1):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, rgb: Tensor, x: Tensor) -> Tensor:
        b, _, h, w = rgb.shape
        feats = torch.cat((rgb, x), dim=1)
        attn = self.mlp(feats).reshape(b, 2, 1, h, w)
        attn = attn.permute(1, 0, 2, 3, 4)  # 2 b 1 h w
        return attn


class FeatureRectifyModule(nn.Module):
    def __init__(self,
                 channels: int,
                 reduction: int = 1,
                 lambda_ca: float = 0.5,
                 lambda_sa: float = 0.5):
        super().__init__()
        self.lambda_ca = lambda_ca
        self.lambda_sa = lambda_sa
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(channels, reduction)

    def forward(self, rgb: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        ca = self.channel_attn(rgb, x)
        sa = self.spatial_attn(rgb, x)
        rgb_attn_feats = self.lambda_ca * ca[0] * rgb + self.lambda_sa * sa[0] * rgb
        x_attn_feats = self.lambda_ca * ca[1] * x + self.lambda_sa * sa[1] * x
        rgb = rgb + x_attn_feats
        x = x + rgb_attn_feats
        return rgb, x


class CrossAttention(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            f'embedding_dim {embedding_dim} should be divided by num_heads {num_heads}')

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        head_dim = embedding_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv_rgb = nn.Linear(embedding_dim, embedding_dim * 2, bias=qkv_bias)
        self.kv_x = nn.Linear(embedding_dim, embedding_dim * 2, bias=qkv_bias)

    @staticmethod
    def _separate_heads(x: Tensor, num_heads: int) -> Tensor:
        b, n_tokens, c = x.shape
        x = x.reshape(b, n_tokens, num_heads, c // num_heads)
        return x.transpose(1, 2).contiguous()  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head).contiguous()  # B x N_tokens x C

    def forward(self, rgb: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        b, _, c = rgb.shape
        q_rgb = self._separate_heads(rgb, num_heads=self.num_heads)
        q_x = self._separate_heads(x, num_heads=self.num_heads)

        k_rgb, v_rgb = self.kv_rgb(rgb).reshape(
            b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k_x, v_x = self.kv_x(x).reshape(
            b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx_rgb = (k_rgb.transpose(-2, -1) @ v_rgb) * self.scale
        ctx_rgb = ctx_rgb.softmax(dim=-2)
        ctx_x = (k_x.transpose(-2, -1) @ v_x) * self.scale
        ctx_x = ctx_x.softmax(dim=-2)

        rgb = q_rgb @ ctx_x
        x = q_x @ ctx_rgb

        rgb = self._recombine_heads(rgb)
        x = self._recombine_heads(x)

        return rgb, x


class CrossPath(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 reduction: int = 1,
                 num_heads: int = 8,
                 norm_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='LN')

        self.rgb_proj = nn.Linear(embedding_dim, embedding_dim // reduction * 2)
        self.x_proj = nn.Linear(embedding_dim, embedding_dim // reduction * 2)
        self.act_rgb = nn.ReLU(inplace=True)
        self.act_x = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(embedding_dim // reduction, num_heads=num_heads)
        self.out_proj_rgb = nn.Linear(embedding_dim // reduction * 2, embedding_dim)
        self.out_proj_x = nn.Linear(embedding_dim // reduction * 2, embedding_dim)
        self.norm_rgb = build_norm_layer(norm_cfg, embedding_dim)[1]
        self.norm_x = build_norm_layer(norm_cfg, embedding_dim)[1]

    def forward(self, rgb: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        y_rgb, u_rgb = self.act_rgb(self.rgb_proj(rgb)).chunk(2, dim=-1)
        y_x, u_x = self.act_x(self.x_proj(x)).chunk(2, dim=-1)
        v_rgb, v_x = self.cross_attn(u_rgb, u_x)
        y_rgb = torch.cat((y_rgb, v_rgb), dim=-1)
        y_x = torch.cat((y_x, v_x), dim=-1)
        rgb = self.norm_rgb(rgb + self.out_proj_rgb(y_rgb))
        x = self.norm_x(x + self.out_proj_x(y_x))
        return rgb, x


class ChannelEmbed(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 reduction: int = 1,
                 norm_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            build_norm_layer(norm_cfg, out_channels)[1]
        )
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        b, n, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        return self.norm(residual + x)


class FeatureFusionModule(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 reduction: int = 1,
                 num_heads: int = 8,
                 norm_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        self.cross = CrossPath(embedding_dim, reduction, num_heads)
        self.channel_embed = ChannelEmbed(embedding_dim * 2, embedding_dim,
                                          reduction, norm_cfg)

    def forward(self, rgb: Tensor, x: Tensor) -> Tensor:
        b, c, h, w = rgb.shape
        rgb = rgb.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        rgb, x = self.cross(rgb, x)
        fusion = torch.cat((rgb, x), dim=-1)
        fusion = self.channel_embed(fusion, h, w)
        return fusion


@MODELS.register_module()
class CMXFusion(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int = 8,
                 reduction: int = 1,
                 lambda_ca: float = 0.5,
                 lambda_sa: float = 0.5,
                 norm_cfg=None,
                 **kwargs):
        super().__init__()
        self.frm = FeatureRectifyModule(embedding_dim, reduction, lambda_ca, lambda_sa)
        self.ffm = FeatureFusionModule(embedding_dim, reduction, num_heads, norm_cfg)

    def forward(self, rgb: Tensor, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        rgb, x = self.frm(rgb, x)
        fusion = self.ffm(rgb, x)
        return rgb, x, fusion
