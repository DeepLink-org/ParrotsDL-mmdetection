_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='/mnt/lustre/share_data/PAT/datasets/mmdet/pretrain/resnet101-5d3b4d8f.pth')))
