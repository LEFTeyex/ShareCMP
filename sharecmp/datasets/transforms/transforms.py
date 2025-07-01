import mmcv
from mmseg.datasets.transforms import RandomCrop, Resize, RandomFlip
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RGBXResize(Resize):
    def _resize_x_img_fields(self, results: dict) -> None:
        x_img_fields = results.get('x_img_fields', [])
        for x_img_key in x_img_fields:
            x_img = results[x_img_key]
            if self.keep_ratio:
                x_img, scale_factor = mmcv.imrescale(
                    x_img,
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            else:
                x_img, w_scale, h_scale = mmcv.imresize(
                    x_img,
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results[x_img_key] = x_img
            results[f'{x_img_key}_shape'] = x_img.shape[:2]

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        self._resize_x_img_fields(results)
        return results


@TRANSFORMS.register_module()
class RandomRGBXCrop(RandomCrop):
    def transform(self, results: dict) -> dict:
        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop the x image
        x_img_fields = results.get('x_img_fields', [])
        for x_img_key in x_img_fields:
            x_img = results[x_img_key]
            x_img = self.crop(x_img, crop_bbox)
            results[x_img_key] = x_img
            results[f'{x_img_key}_shape'] = img.shape[:2]

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class RandomRGBXFlip(RandomFlip):
    def _flip(self, results: dict) -> None:
        super()._flip(results)
        x_img_fields = results.get('x_img_fields', [])
        for x_img_key in x_img_fields:
            results[x_img_key] = mmcv.imflip(
                results[x_img_key], direction=results['flip_direction'])
