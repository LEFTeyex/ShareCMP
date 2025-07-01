from typing import Union, List, Optional

from mmengine.config import Config, ConfigDict
from mmengine.registry import MODELS
from mmseg.models import MixVisionTransformer
from mmseg.models.utils import nlc_to_nchw, PatchEmbed
from torch.nn import ModuleList

cfg = Union[dict, ConfigDict, Config]


@MODELS.register_module()
class ShareCMPEncoder(MixVisionTransformer):
    def __init__(self,
                 x_img_key: str,
                 x_img_in_channel: int,
                 rgbx_attention: dict,
                 *args,
                 x_img_fusion: Optional[cfg] = None,
                 diff_pe: List[int] = [0, 1, 2, 3],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.x_img_key = x_img_key
        self.x_img_in_channel = x_img_in_channel
        self.rgbx_attention = rgbx_attention
        self.diff_pe = diff_pe

        norm_cfg = kwargs.get('norm_cfg')
        if norm_cfg is None:
            norm_cfg = dict(type='LN', eps=1e-6)
        self.norm_cfg = norm_cfg

        self.x_patch_embed_layers = ModuleList()
        self.rgbx_attn_layers = ModuleList()
        for i, num_layer in enumerate(self.num_layers):
            embed_dims_i = self.embed_dims * self.num_heads[i]

            rgbx_attention['embedding_dim'] = embed_dims_i
            rgbx_attn_layer = MODELS.build(rgbx_attention)
            self.rgbx_attn_layers.append(rgbx_attn_layer)

            if i not in diff_pe:
                x_img_in_channel = embed_dims_i
                self.x_patch_embed_layers.append(None)
                continue

            patch_embed = PatchEmbed(
                in_channels=x_img_in_channel,
                embed_dims=embed_dims_i,
                kernel_size=self.patch_sizes[i],
                stride=self.strides[i],
                padding=self.patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            x_img_in_channel = embed_dims_i
            self.x_patch_embed_layers.append(patch_embed)

        self.x_img_fusion = None
        if x_img_fusion is not None:
            self.x_img_fusion = MODELS.build(x_img_fusion)

    def forward(self, inputs: dict):
        rgb = inputs['img']
        if self.x_img_fusion is not None:
            x = self.x_img_fusion(inputs)
        else:
            x = inputs[self.x_img_key]

        outs = []
        for i, layer in enumerate(self.layers):
            rgb, hw_shape = layer[0](rgb)
            layer_0 = self.x_patch_embed_layers[i]
            if layer_0 is None:
                layer_0 = layer[0]

            x, x_hw_shape = layer_0(x)

            for block in layer[1]:
                rgb = block(rgb, hw_shape)
                x = block(x, x_hw_shape)

            rgb = layer[2](rgb)
            x = layer[2](x)

            rgb = nlc_to_nchw(rgb, hw_shape)
            x = nlc_to_nchw(x, x_hw_shape)

            rgbx_attn_layer = self.rgbx_attn_layers[i]
            rgb, x, fusion = rgbx_attn_layer(rgb, x)
            if fusion is None:
                fusion = rgb

            if i in self.out_indices:
                outs.append(fusion)

        return outs
