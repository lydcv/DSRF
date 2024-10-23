_base_ = [
    './meta-rcnn_r50_c4.py',
]

custom_imports = dict(
    imports=[
        'dsrf.dsrf_detector',
        'dsrf.dsrf_roi_head',
        'dsrf.dsrf_bbox_head'], 
    allow_failed_imports=False)

pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    type='DSRF',
    pretrained=pretrained,
    backbone=dict(depth=101),
    information_fusion=dict(
        type='DynamicInformationFusionModule',
        in_channels=1024,
        inter_channels=1024,
        dimension=2,
        sub_sample=False,
        bn_layer=True),
    roi_head=dict(
        type='DSRFRoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(
            type='DSRFBBoxHead', num_classes=6, num_meta_classes=6)))
