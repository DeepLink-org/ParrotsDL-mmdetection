_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/coco/': 's3://open_dataset_original/Objects365/train',
        'data/coco/images/v1': 's3://open_dataset_original/Objects365/train',
        'data/coco/images/v2': 's3://open_dataset_original/Objects365/train',
        'objects365/val/images/v1': 's3://open_dataset_original/Objects365/val',
        'objects365/val/images/v2': 's3://open_dataset_original/Objects365/val'
    })
)
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/coco/': 's3://',
        'objects365/val/images/v1': 's3://open_dataset_original/Objects365/val',
        'objects365/val/images/v2': 's3://open_dataset_original/Objects365/val'
    })
)
norm_cfg = dict(type='MMSyncBN', requires_grad=True)
head_norm_cfg = dict(type='MMSyncBN', requires_grad=True)
dataset_type = 'Obj365DatasetV2'
data_root = 'data/coco/'
checkpoint_config = dict(interval=1)
resume_from = None
load_from = None
pretrained = '/mnt/lustre/zongzhuofan/models/mixmim/giant_800/ckpt/checkpoint.pth'
pretrained = 's3://home/exp_mxnet/giant_800/ft_21k/checkpoint.pth'
# Use MMSyncBN that handles empty tensor in head. It can be changed to
# SyncBN after https://github.com/pytorch/pytorch/issues/36530 is fixed
# Requires MMCV-full after  https://github.com/open-mmlab/mmcv/pull/1205.
model = dict(
    type='CompositeDetector',
    use_anchor_query=True,
    backbone=dict(
        type='XMNet',
        embed_dims=384,
        depths=[2, 6, 24, 2],
        num_heads=[6, 12, 24, 48],
        window_size=[28, 28, 28, 28],
        use_abs_pos_embed=True,
        pretrain_img_size=224,
        use_global=True,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.5,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[384*2, 384*4, 384*8],
        kernel_size=1,
        out_channels=384,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    rpn_head=dict(
        type='RPNHead',
        in_channels=384,
        feat_channels=384,
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
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0),
        loss_bbox=dict(type='L1Loss', loss_weight=12.0)),
    query_head=dict(
        type='CompositeDeformDETRHead',
        num_query=900,
        num_classes=365,
        in_channels=2048,
        max_pos_priors=900,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        mixed_selection=True,
        transformer=dict(
            type='DeformableDetrTransformer',
            use_anchor_feats=True,
            use_anchor_query=True,
            num_co_heads=2,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=384, num_heads=12, dropout=0.0),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=384,
                        feedforward_channels=3072,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=3072,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                look_forward_twice=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=384,
                            num_heads=12,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=384,
                            num_heads=12,
                            dropout=0.0)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=384,
                        feedforward_channels=3072,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=3072,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=192,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=384,
            featmap_strides=[8, 16, 32, 64],
            finest_scale=112),
        bbox_head=dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=384,
            conv_out_channels=384,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=365,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=12.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=120.0)),)],
    bbox_head=[dict(
        type='ATSSHead',
        num_classes=365,
        in_channels=384,
        stacked_convs=1,
        feat_channels=384,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=12.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0)),],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),],
    test_cfg=[
        dict(
            max_per_img=100),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=100)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='soft_nms', iou_threshold=0.6),
            max_per_img=100),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])

# optimizer
optimizer = dict(
    type='AdamW',
    lr=9e-4,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

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
                    img_scale=[(480, 1600), (512, 1600), 
                            (544, 1600), (576, 1600), (608, 1600), (640, 1600), 
                            (672, 1600), (704, 1600), (736, 1600), (768, 1600), 
                            (800, 1600), (832, 1600), (864, 1600), (896, 1600), 
                            (928, 1600), (960, 1600), (992, 1600), (1024, 1600)],
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
                    img_scale=[(480, 1600), (512, 1600), 
                            (544, 1600), (576, 1600), (608, 1600), (640, 1600), 
                            (672, 1600), (704, 1600), (736, 1600), (768, 1600), 
                            (800, 1600), (832, 1600), (864, 1600), (896, 1600), 
                            (928, 1600), (960, 1600), (992, 1600), (1024, 1600)],
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
        img_scale=(1600, 1024),
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
                    img_scale=[(480, 1333), (512, 1333), 
                            (544, 1333), (576, 1333), (608, 1333), (640, 1333), 
                            (672, 1333), (704, 1333), (736, 1333), (768, 1333), 
                            (800, 1333)],
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
                    img_scale=[(480, 1333), (512, 1333), 
                            (544, 1333), (576, 1333), (608, 1333), (640, 1333), 
                            (672, 1333), (704, 1333), (736, 1333), (768, 1333), 
                            (800, 1333)],
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
        img_scale=(1333, 800),
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

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        ann_file='/mnt/lustre/zongzhuofan/labels/objects365_v2_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/mnt/lustre/zongzhuofan/labels/zhiyuan_objv2_val_5k.json',
        img_prefix='objects365/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/mnt/lustre/zongzhuofan/labels/zhiyuan_objv2_val_5k.json',
        img_prefix='objects365/val',
        pipeline=test_pipeline))
evaluation = dict(metric='bbox')
dist_params = dict(backend='nccl', port=29515)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    step=[24])
runner = dict(type='EpochBasedRunner', max_epochs=26)

#find_unused_parameters=True
#resume_from = 'work_dirs/mb15_pretrain/mixmim_g_encoder/iter_120000.pth'
#resume_from = 'work_dirs/mb15_pretrain/mixmim_g_codet_cascade/iter_20000.pth'
load_from = None
resume_from = 'work_dirs/mb15_pretrain/mixmim_g_codet_cascade_bs4/iter_70000.pth'
resume_from = 'work_dirs/mb15_pretrain/mixmim_g_codet_cascade_bs4_obj/latest.pth'
resume_from = '/mnt/lustre/zongzhuofan/competition/mmdetection/work_dirs/mb15_pretrain/mixmim_g_21k_codet_new_obj_dpr5/latest.pth'
load_from = 'work_dirs/mb15_pretrain/mixmim_g_codet_cascade_bs4_obj/latest.pth'
load_from = 'work_dirs/mb15_pretrain/mixmim_g_codet_cascade_bs4/iter_30000.pth'
load_from = 'work_dirs/mb15_pretrain/mixmim_g_codet_cascade_obj_dpr5/latest.pth'
load_from = None
