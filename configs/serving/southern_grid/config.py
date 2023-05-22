_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py'
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', 
    '../../_base_/default_runtime.py'
]

file_client_args = dict(
    backend='petrel')
file_client_args = dict(
    backend='petrel')

model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=30,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


# classes = None
# data = dict(
#     samples_per_gpu=10,
#     workers_per_gpu=5,
#     train=dict(
#         type='CocoDataset',
#         classes=classes,
#         ann_file='/mnt/lustre/zongzhuofan/data/nw/nw_train.json',
#         img_prefix='/mnt/lustre/share_data/zongzhuofan/data/',
#         pipeline=train_pipeline),
#     val=dict(
#         type='CocoDataset',
#         classes=classes,
#         ann_file='/mnt/lustre/zongzhuofan/data/nw/nw_val.json',
#         img_prefix='/mnt/lustre/share_data/zongzhuofan/data/',
#         pipeline=test_pipeline),
#     test=dict(
#         type='CocoDataset',
#         classes=classes,
#         ann_file='/mnt/lustre/zongzhuofan/data/nw/nw_val.json',
#         img_prefix='/mnt/lustre/share_data/zongzhuofan/data/',
#         pipeline=test_pipeline))

# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 2000), (512, 2000), 
                            (544, 2000), (576, 2000), (608, 2000), (640, 2000), 
                            (672, 2000), (704, 2000), (736, 2000), (768, 2000), 
                            (800, 2000), (832, 2000), (864, 2000), (896, 2000), 
                            (928, 2000), (960, 2000), (992, 2000), (1024, 2000),
                            (1056, 2000), (1088, 2000), (1120, 2000), (1152, 2000),
                            (1184, 2000), (1216, 2000), (1248, 2000), (1280, 2000)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 2000), (512, 2000), 
                            (544, 2000), (576, 2000), (608, 2000), (640, 2000), 
                            (672, 2000), (704, 2000), (736, 2000), (768, 2000), 
                            (800, 2000), (832, 2000), (864, 2000), (896, 2000), 
                            (928, 2000), (960, 2000), (992, 2000), (1024, 2000),
                            (1056, 2000), (1088, 2000), (1120, 2000), (1152, 2000),
                            (1184, 2000), (1216, 2000), (1248, 2000), (1280, 2000)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2000, 1216),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]



classes = None
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=5,
    train=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='/mnt/cache/share/DSK/datasets/mscoco2017/annotations/instances_train2017.json',
        img_prefix='/mnt/cache/share/DSK/datasets/mscoco2017/train2017',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='/mnt/cache/share/DSK/datasets/mscoco2017/annotations/instances_val2017.json',
        img_prefix='/mnt/cache/share/DSK/datasets/mscoco2017/val2017',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=classes,
        ann_file='/mnt/cache/share/DSK/datasets/mscoco2017/annotations/instances_val2017.json',
        img_prefix='/mnt/cache/share/DSK/datasets/mscoco2017/val2017',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)





