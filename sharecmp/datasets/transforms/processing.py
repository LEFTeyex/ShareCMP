from typing import List, Optional

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CatImages(BaseTransform):
    """Cat images from cat_keys.

    Examples:
        cat_keys is ['a_img', 'b_img'],
        cat_img_key == 'a_b_img'.
    """

    def __init__(self,
                 cat_keys: List[str],
                 release_keys: Optional[List[str]] = None):
        self.cat_keys = cat_keys
        self.release_keys = release_keys

    def transform(self, results: dict) -> dict:
        results.setdefault('x_img_fields', [])

        cat_imgs = [results[key] for key in self.cat_keys]
        for i, img in enumerate(cat_imgs):
            if len(img.shape) == 2:
                cat_imgs[i] = img[:, :, np.newaxis]

        cat_img_key = '_'.join([key.rsplit('_', 1)[0]
                                for key in self.cat_keys]) + '_img'

        cat_img = np.concatenate(cat_imgs, axis=2)
        results[cat_img_key] = cat_img
        results['x_img_fields'].append(cat_img_key)
        results[f'{cat_img_key}_shape'] = cat_img.shape[:2]
        results[f'{cat_img_key}_ori_shape'] = cat_img.shape[:2]

        if self.release_keys is not None:
            for key in self.release_keys:
                if key in results['x_img_fields']:
                    results['x_img_fields'].remove(key)
                    del results[key]
                    del results[f'{key}_shape']
                    del results[f'{key}_ori_shape']

        return results
