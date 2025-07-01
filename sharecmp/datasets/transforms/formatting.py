import numpy as np
from mmcv.transforms import to_tensor
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.registry import TRANSFORMS


def pack_x_img_fields(results: dict, packed_results: dict) -> dict:
    x_img_fields = results.get('x_img_fields', [])  # x_img_fields List[str]

    img = packed_results.get('inputs')
    inputs = dict(img=img)
    for x_img_key in x_img_fields:
        if x_img_key in results:
            x_img = results[x_img_key]

            if len(x_img.shape) < 3:
                x_img = np.expand_dims(x_img, -1)

            if not x_img.flags.c_contiguous:
                x_img = np.ascontiguousarray(x_img.transpose(2, 0, 1))
                x_img = to_tensor(x_img)
            else:
                x_img = to_tensor(x_img).permute(2, 0, 1).contiguous()
            inputs[x_img_key] = x_img

    packed_results['inputs'] = inputs
    return packed_results


@TRANSFORMS.register_module()
class RGBXPackSegInputs(PackSegInputs):
    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        packed_results = pack_x_img_fields(results, packed_results)
        return packed_results
