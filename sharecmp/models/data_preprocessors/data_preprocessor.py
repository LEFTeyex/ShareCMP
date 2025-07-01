import copy
from numbers import Number
from typing import Any, Dict, Tuple, Sequence, Optional, List

import torch
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS
from mmseg.utils import stack_batch, SampleList


@MODELS.register_module()
class RGBXSegDataPreProcessor(SegDataPreProcessor):
    def __init__(
            self,
            packed_keys: List[str] = None,
            x_mean: Sequence[Number] = None,
            x_std: Sequence[Number] = None,
            mean: Sequence[Number] = None,
            std: Sequence[Number] = None,
            size: Optional[tuple] = None,
            size_divisor: Optional[int] = None,
            pad_val: Number = 0,
            seg_pad_val: Number = 255,
            bgr_to_rgb: bool = False,
            rgb_to_bgr: bool = False,
            batch_augments: Optional[List[dict]] = None,
            test_cfg: dict = None,
    ):
        super().__init__(
            mean,
            std,
            size,
            size_divisor,
            pad_val,
            seg_pad_val,
            bgr_to_rgb,
            rgb_to_bgr,
            batch_augments,
            test_cfg,
        )
        self.packed_keys = packed_keys

        if x_mean is not None:
            assert x_std is not None, 'To enable the normalization in ' \
                                      'preprocessing, please specify both ' \
                                      '`x_mean` and `x_std`.'
            # Enable the normalization in preprocessing.
            self._enable_x_normalize = True
            self.register_buffer('x_mean',
                                 torch.tensor(x_mean).view(-1, 1, 1), False)
            self.register_buffer('x_std',
                                 torch.tensor(x_std).view(-1, 1, 1), False)
        else:
            self._enable_x_normalize = False

    def inner_forward(self,
                      inputs,
                      data_samples,
                      training: bool,
                      x_img: bool = False) -> Tuple[torch.Tensor, SampleList]:
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize and not x_img:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if self._enable_x_normalize and x_img:
            inputs = [(_input - self.x_mean) / self.x_std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs), \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)
        return inputs, data_samples

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        data_samples_copy = copy.deepcopy(data_samples)
        img = inputs.pop('img')
        img, data_samples = self.inner_forward(img, data_samples, training)

        for key, value in inputs.items():
            value, _ = self.inner_forward(value, data_samples_copy, training, x_img=True)
            inputs[key] = value
        inputs['img'] = img

        # pack dolp and aolp etc. in data_sample index 0
        if self.packed_keys is not None:
            for key in self.packed_keys:
                value = inputs[key]  # value shape is  b c h w
                data_samples[0].set_data({key: value})

        return dict(inputs=inputs, data_samples=data_samples)
