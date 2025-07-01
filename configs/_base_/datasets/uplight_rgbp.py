# dataset settings
import_dir = 'sharecmp.datasets'

dataset_type = 'UPLightDataset'
data_root = 'data/UPLight/'
x_img0_dir = 'images_rgb_0'
x_img45_dir = 'images_rgb_45'
x_img90_dir = 'images_rgb_90'
x_img135_dir = 'images_rgb_135'

img_scale = (512, 612)
crop_size = (512, 612)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadXImageFromFile',
        x_img_key='p_img',
        x_img_dirs=[
            data_root + x_img0_dir,
            data_root + x_img45_dir,
            data_root + x_img90_dir,
            data_root + x_img135_dir,
        ],
        x_sub_names=['0', '45', '90', '135'],
        x_img_suffix='.jpg',
        align_rgbx=True,
        color_type='color',
    ),
    dict(
        type='TransformPolarizeFourAngle',
        # get aolp_img and dolp_img
        transform_keys=['aolp', 'dolp'],
        img_0_key='p_img_0',
        img_45_key='p_img_45',
        img_90_key='p_img_90',
        img_135_key='p_img_135',
        release_polar_img=False,
    ),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        resize_type='RGBXResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomRGBXCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomRGBXFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='RGBXPackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadXImageFromFile',
        x_img_key='p_img',
        x_img_dirs=[
            data_root + x_img0_dir,
            data_root + x_img45_dir,
            data_root + x_img90_dir,
            data_root + x_img135_dir,
        ],
        x_sub_names=['0', '45', '90', '135'],
        x_img_suffix='.jpg',
        align_rgbx=True,
        color_type='color',
    ),
    dict(
        type='TransformPolarizeFourAngle',
        # get aolp_img and dolp_img
        transform_keys=['aolp', 'dolp'],
        img_0_key='p_img_0',
        img_45_key='p_img_45',
        img_90_key='p_img_90',
        img_135_key='p_img_135',
        release_polar_img=False,
    ),
    dict(type='RGBXResize', scale=img_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='RGBXPackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadXImageFromFile',
        x_img_key='p_img',
        x_img_dirs=[
            data_root + x_img0_dir,
            data_root + x_img45_dir,
            data_root + x_img90_dir,
            data_root + x_img135_dir,
        ],
        x_sub_names=['0', '45', '90', '135'],
        x_img_suffix='.jpg',
        align_rgbx=True,
        color_type='color',
    ),
    dict(
        type='TransformPolarizeFourAngle',
        # get aolp_img and dolp_img
        transform_keys=['aolp', 'dolp'],
        img_0_key='p_img_0',
        img_45_key='p_img_45',
        img_90_key='p_img_90',
        img_135_key='p_img_135',
        release_polar_img=False,
    ),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='RGBXResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomRGBXFlip', prob=0., direction='horizontal'),
                dict(type='RandomRGBXFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='RGBXPackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images_rgb', seg_map_path='labels'),
        ann_file='train.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images_rgb', seg_map_path='labels'),
        ann_file='val.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
