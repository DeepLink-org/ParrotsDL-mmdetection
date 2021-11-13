_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
# fp16 settings
fp16 = dict(loss_scale=512.)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='PaviLoggerHook', init_kwargs={"project": "yyyds112"})
    ])
