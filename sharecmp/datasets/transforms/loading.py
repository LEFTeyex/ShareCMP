import warnings
from pathlib import Path
from typing import Optional, Union, List, Dict

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadXImageFromFile(LoadImageFromFile):
    def __init__(self,
                 x_img_key: str,
                 x_img_dirs: Union[str, List[Union[str, Path]]],
                 x_img_suffix: str = '.png',
                 x_sub_names: Optional[List[str]] = None,
                 align_rgbx: bool = True,
                 to_float32: bool = False,
                 color_type: str = 'grayscale',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None):
        super().__init__(to_float32,
                         color_type,
                         imdecode_backend,
                         file_client_args,
                         ignore_empty,
                         backend_args=backend_args)
        self.x_img_key = x_img_key
        self.x_img_dirs = x_img_dirs
        self.x_img_suffix = x_img_suffix
        self.x_sub_names = x_sub_names
        self.align_rgbx = align_rgbx

    def _match_x_img_path(self,
                          img_path: Union[str, Path],
                          x_img_dirs: Union[str, List[Union[str, Path]]]) -> Dict[str, str]:
        img_stem = Path(img_path).stem
        if isinstance(x_img_dirs, list) and len(x_img_dirs) > 1:
            assert len(x_img_dirs) == len(self.x_sub_names), (
                'x_img_dirs must be corresponding to x_sub_names')
        else:
            x_img_dirs = [x_img_dirs]

        x_img_paths = {}
        if self.x_sub_names is None:
            x_img_dir = Path(x_img_dirs[0])
            x_img_name = f'{img_stem}{self.x_img_suffix}'
            x_img_paths[self.x_img_key] = str(x_img_dir / x_img_name)
        else:
            if len(x_img_dirs) == 1:
                x_img_dirs = x_img_dirs * len(self.x_sub_names)
            for i, xs_name in enumerate(self.x_sub_names):
                x_img_dir = Path(x_img_dirs[i])
                x_img_key = f'{self.x_img_key}_{xs_name}'
                x_img_name = f'{img_stem}_{xs_name}{self.x_img_suffix}'
                x_img_paths[x_img_key] = str(x_img_dir / x_img_name)

        return x_img_paths

    def _transform_once(self, results: dict, x_img_key: str, filename: str) -> Optional[dict]:
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                x_img_bytes = file_client.get(filename)
            else:
                x_img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            x_img = mmcv.imfrombytes(
                x_img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert x_img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            x_img = x_img.astype(np.float32)

        results[x_img_key] = x_img
        results[f'{x_img_key}_shape'] = x_img.shape[:2]
        results[f'{x_img_key}_ori_shape'] = x_img.shape[:2]

    @staticmethod
    def _align_x_img_to_img(results: dict, x_img_key: str) -> Optional[dict]:
        img = results.get('img', None)
        if img is None:
            warnings.warn(
                'Do not find img, can not align x_img size to img size')
            return results
        x_img = results[x_img_key]

        if x_img.shape[:2] != img.shape[:2]:
            x_img = mmcv.imrescale(x_img, img.shape[:2])

        if x_img.shape[:2] != img.shape[:2]:
            raise ValueError('x_img size can not be aligned when keep it ratio')

        results[x_img_key] = x_img
        results[f'{x_img_key}_shape'] = x_img.shape[:2]
        return results

    def transform(self, results: dict) -> Optional[dict]:
        img_path = results['img_path']
        x_img_paths = self._match_x_img_path(img_path, self.x_img_dirs)

        results.setdefault('x_img_fields', [])
        for x_img_key, filename in x_img_paths.items():
            self._transform_once(results, x_img_key, filename)

            if self.align_rgbx:
                results = self._align_x_img_to_img(results, x_img_key)

            results['x_img_fields'].append(x_img_key)

        return results
