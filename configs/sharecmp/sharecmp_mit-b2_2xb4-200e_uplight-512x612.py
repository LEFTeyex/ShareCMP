_base_ = [
    '../_base_/models/segformer_mit-b2.py', '../_base_/datasets/uplight_rgbp.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_200e.py'
]

import_dir = 'sharecmp'
max_epoch = 200
crop_size = (512, 612)
num_classes = 13
data_preprocessor = dict(
    type='RGBXSegDataPreProcessor',
    x_mean=[123.675, 116.28, 103.53],
    x_std=[58.395, 57.12, 57.375],
    size=crop_size,
    packed_keys=['aolp_img', 'dolp_img'],
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ShareCMPEncoder',
        x_img_key='no_img',
        x_img_in_channel=3,
        rgbx_attention=dict(
            type='CMXFusion',
            reduction=1,
            lambda_ca=0.5,
            lambda_sa=0.5,
            norm_cfg=dict(type='BN'),
        ),
        x_img_fusion=dict(
            type='PGA',
            x_img_keys=[
                'p_img_0',
                'p_img_45',
                'p_img_90',
                'p_img_135',
            ],
            x_img_in_channels=[3, 3, 3, 3],
            channel_ratio=1,
            reduction=1,
        ),
    ),
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=[
        dict(
            type='CPAAHead',
            in_index=[2, 3],
            in_channels=[320, 512],
            channels=256,
            norm_cfg=norm_cfg,
            align_corners=False,
            polar_key='aolp_img',
            pred_channels=3,
            loss_decode=dict(
                type='CPALoss',
                loss_name='loss_aolp',
                reduction='mean',
                loss_weight=0.01),
        ),
        dict(
            type='CPAAHead',
            in_index=[2, 3],
            in_channels=[320, 512],
            channels=256,
            norm_cfg=norm_cfg,
            align_corners=False,
            polar_key='dolp_img',
            pred_channels=3,
            loss_decode=dict(
                type='CPALoss',
                loss_name='loss_dolp',
                reduction='mean',
                loss_weight=0.01),
        ),
    ]
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, begin=0, end=5,
        by_epoch=True, convert_to_iter_based=True),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=True),
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
log_processor = dict(by_epoch=True)
