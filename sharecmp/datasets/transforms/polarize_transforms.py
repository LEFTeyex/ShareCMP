from typing import List, Dict, Tuple

import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


def stokes(img_0: np.ndarray,
           img_45: np.ndarray,
           img_90: np.ndarray,
           img_135: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stokes Vector form 0, 45, 90, 135 angles.

    Returns:
        s_0 interval is [0, 255], normalized is [0, 1].
        s_1 interval is [-255, 255], normalized is [-1, 1].
        s_2 interval is [-255, 255], normalized is [-1, 1].
        s_3 is not supported.
    """
    img_0 = img_0.astype(np.float32)
    img_45 = img_45.astype(np.float32)
    img_90 = img_90.astype(np.float32)
    img_135 = img_135.astype(np.float32)

    s_0 = (img_0 + img_45 + img_90 + img_135) / 4
    s_1 = img_0 - img_90
    s_2 = img_45 - img_135
    s_3 = np.zeros_like(s_0)
    return s_0, s_1, s_2, s_3


def aolp(s_0: np.ndarray,
         s_1: np.ndarray,
         s_2: np.ndarray,
         s_3: np.ndarray,
         eps: float = 1e-5) -> np.ndarray:
    # s_2 / s_1 interval is (-oxo, +oxo)
    # arctan
    # x interval is (-oxo, +oxo)
    # y interval is (-pi/2, +pi/2)
    out = s_2 / (s_1 + eps)
    out = np.arctan(out) / np.pi + 0.5
    return np.clip(out, 0, 1)


def aolp_sin(s_0: np.ndarray,
             s_1: np.ndarray,
             s_2: np.ndarray,
             s_3: np.ndarray,
             eps: float = 1e-5) -> np.ndarray:
    # s_2 / s_0 interval is (-oxo, +oxo)
    # arcsin
    # x interval is (-1, +1)
    # y interval is (-pi/2, +pi/2)
    out = np.clip(s_2 / (s_0 + eps), -1, 1)
    out = np.arcsin(out) / np.pi + 0.5
    return np.clip(out, 0, 1)


def aolp_cos(s_0: np.ndarray,
             s_1: np.ndarray,
             s_2: np.ndarray,
             s_3: np.ndarray,
             eps: float = 1e-5) -> np.ndarray:
    # s_1 / s_0 interval is (-oxo, +oxo)
    # arccos
    # x interval is (-1, +1)
    # y interval is (0, +pi)
    out = np.clip(-(s_1 / (s_0 + eps)), -1, 1)
    out = np.arccos(out) / np.pi
    return np.clip(out, 0, 1)


def dolp(s_0: np.ndarray,
         s_1: np.ndarray,
         s_2: np.ndarray,
         s_3: np.ndarray,
         eps: float = 1e-5) -> np.ndarray:
    out = np.sqrt(s_1 ** 2 + s_2 ** 2) / (s_0 + eps)
    return np.clip(out, 0, 1)


def dolp1(s_0: np.ndarray,
          s_1: np.ndarray,
          s_2: np.ndarray,
          s_3: np.ndarray,
          eps: float = 1e-5) -> np.ndarray:
    out = s_1 / (s_0 + eps)
    return np.clip(out, 0, 1)


def dolp2(s_0: np.ndarray,
          s_1: np.ndarray,
          s_2: np.ndarray,
          s_3: np.ndarray,
          eps: float = 1e-5) -> np.ndarray:
    out = s_2 / (s_0 + eps)
    return np.clip(out, 0, 1)


def docp(s_0: np.ndarray,
         s_1: np.ndarray,
         s_2: np.ndarray,
         s_3: np.ndarray,
         eps: float = 1e-5) -> np.ndarray:
    out = s_3 / (s_0 + eps)
    return np.clip(out, 0, 1)


def dop(s_0: np.ndarray,
        s_1: np.ndarray,
        s_2: np.ndarray,
        s_3: np.ndarray,
        eps: float = 1e-5) -> np.ndarray:
    out = np.sqrt(s_1 ** 2 + s_2 ** 2 + s_3 ** 2) / (s_0 + eps)
    return np.clip(out, 0, 1)


_ADOP_ = {
    'aolp': aolp,
    'aolp_sin': aolp_sin,
    'aolp_cos': aolp_cos,
    'dolp': dolp,
    'dolp1': dolp1,
    'dolp2': dolp2,
    'docp': docp,
    'dop': dop,
}


def transform_polarize_image(img_0: np.ndarray,
                             img_45: np.ndarray,
                             img_90: np.ndarray,
                             img_135: np.ndarray,
                             transform_keys: List[str],
                             eps: float = 1e-5) -> Dict[str, np.ndarray]:
    """The interval of transformed image is 0 to 1."""
    s_0, s_1, s_2, s_3 = stokes(img_0, img_45, img_90, img_135)
    transformed = {}
    for trans_key in transform_keys:
        trans_func = _ADOP_[trans_key]
        trans_img = trans_func(s_0, s_1, s_2, s_3, eps)
        transformed[trans_key] = (trans_img * 255).astype(np.uint8)
    return transformed


@TRANSFORMS.register_module()
class TransformPolarizeFourAngle(BaseTransform):
    def __init__(self,
                 transform_keys: List[str],
                 img_0_key: str = 'p_img_0',
                 img_45_key: str = 'p_img_45',
                 img_90_key: str = 'p_img_90',
                 img_135_key: str = 'p_img_135',
                 release_polar_img: bool = True,
                 eps: float = 1e-5):
        self.transform_keys = transform_keys
        self.img_0_key = img_0_key
        self.img_45_key = img_45_key
        self.img_90_key = img_90_key
        self.img_135_key = img_135_key
        self.release_polar_img = release_polar_img
        self.eps = eps

    def transform(self, results: dict) -> dict:
        img_0 = results[self.img_0_key]
        img_45 = results[self.img_45_key]
        img_90 = results[self.img_90_key]
        img_135 = results[self.img_135_key]

        transformed_imgs = transform_polarize_image(img_0, img_45, img_90, img_135,
                                                    self.transform_keys, self.eps)
        results.setdefault('x_img_fields', [])
        for trans_key, x_img in transformed_imgs.items():
            x_img_key = f'{trans_key}_img'
            results[x_img_key] = x_img
            results[f'{x_img_key}_shape'] = x_img.shape[:2]
            results[f'{x_img_key}_ori_shape'] = x_img.shape[:2]
            results['x_img_fields'].append(x_img_key)

        if self.release_polar_img:
            for key in (self.img_0_key, self.img_45_key, self.img_90_key, self.img_135_key):
                results['x_img_fields'].remove(key)
                del results[key]
                del results[f'{key}_shape']
                del results[f'{key}_ori_shape']

        return results
