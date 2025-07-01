import mmengine.fileio as fileio
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class UPLightDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('unlabeled', 'mooringmine', 'uuv', 'float', 'reflector',
                 'person', 'bottommine', 'cup', 'ironball',
                 'conch', 'fish', 'shell', 'starfish'),
        # palette color order is BGR
        palette=[[0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192],
                 [0, 0, 192], [128, 128, 0], [64, 64, 128],
                 [192, 128, 128], [192, 64, 0], [192, 192, 192],
                 [0, 255, 64], [192, 0, 64], [192, 64, 255]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], self.backend_args)
