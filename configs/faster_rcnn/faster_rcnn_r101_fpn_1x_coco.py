_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
from utils import checkpoint
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint=checkpoint['resnet101'])))
