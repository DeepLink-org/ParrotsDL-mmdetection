import os
mmlab_path = os.getenv("MMLAB_PATH")
mmlab_path_1984 = '/mnt/lustre/share_data/openmmlab/'
mmlab_path_1424 = '/mnt/lustre/share/openmmlab/'
coco_path = "datasets/detection/coco/"
voc_path = "datasets/detection/VOCdevkit/"
action_path = "datasets/action/"
pretrain_path_1984 = '/mnt/lustre/share_data/jiangyongjiu/model_zoo/'
pretrain_path = ""
if mmlab_path is None:
    if os.path.isdir(mmlab_path_1984):
        coco_path = mmlab_path_1984 + coco_path
        voc_path = mmlab_path_1984 + voc_path
        action_path = mmlab_path_1984 + action_path
        pretrain_path = pretrain_path_1984
    elif os.path.isdir(mmlab_path_1424):
        coco_path = mmlab_path_1424 + coco_path
        voc_path = mmlab_path_1424 + voc_path
        action_path = mmlab_path_1424 + action_path
        pretrain_path = mmlab_path_1424 + "pretrain_model/"
    else:
        raise RuntimeError("Please set MMLAB_DATA_PATH")
else:
    coco_path = mmlab_path + coco_path
    voc_path = mmlab_path + voc_path
    action_path = mmlab_path + action_path
    pretrain_path = mmlab_path + "pretrain_model/"

checkpoint = dict(
    resnet50=pretrain_path + "resnet50-19c8e357.pth",
    resnet101=pretrain_path + "resnet101_caffe-3ad79236.pth",
    darknet53=pretrain_path + "darknet53-a628ea1b.pth",
    vgg16=pretrain_path + "vgg16_caffe-292e1171.pth",
)
