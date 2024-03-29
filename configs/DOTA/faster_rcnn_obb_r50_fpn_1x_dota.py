CLASSES = ('small ship', 'large ship', 'civilian aircraft', 'military aircraft', 'small car',
               'bus', 'truck', 'train', 'crane', 'bridge', 'oil tank', 'dam', 'athletic field',
               'helipad', 'roundabout', 'etc')

# model settings
model = dict(
    type='FasterRCNNOBB',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4],
        anchor_ratios=[0.5, 1.0, 2.0],
        # anchor_ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=len(CLASSES),
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        reg_class_agnostic=False,
        with_module=False,
        hbb_trans='hbbpolyobb',
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr = 0.05,
        nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1),
        max_per_img = 2000))
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05))

# dataset settings
dataset_type = 'ArirangDataset'
data_root = '/home/ace19/dl_data/Arirang_Dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'patch/train.json',
        img_prefix=data_root + 'patch/images',
        img_scale=[(1024, 1024)],
        multiscale_mode='range',
        # img_scale=[(1024, 1024)],
        # multiscale_mode='value',
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        # https://albumentations.readthedocs.io/en/latest/examples.html
        albu_aug=dict(
            transforms=[
                # dict(
                #     type='ShiftScaleRotate',
                #     shift_limit=0.0625,
                #     scale_limit=0.1,
                #     rotate_limit=45,
                #     p=0.5),
                dict(
                    type='RandomRotate90',
                    p=0.3
                ),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(type='GaussNoise', p=1.0),
                        dict(
                            type='MultiplicativeNoise',
                            multiplier=[0.5, 1.5],
                            elementwise=True,
                            p=1),
                        ],
                    p=0.2),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(type='Blur', blur_limit=(15, 15), p=0.2),
                        dict(type='MedianBlur', blur_limit=3, p=0.2),
                    ],
                    p=0.2),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(type='CLAHE', clip_limit=2, p=1.0),
                        dict(type='IAASharpen', blur_limit=3, p=1.0),
                        dict(type='IAAEmboss', p=1.0),
                        dict(type='RandomBrightnessContrast', p=1.0),
                    ],
                    p=0.3),
                dict(
                    type='HueSaturationValue',
                    p=0.3),
                # dict(
                #     type='Cutout',
                #     num_holes=10,
                #     max_h_size=20,
                #     max_w_size=20,
                #     fill_value=0,
                #     p=1),
            ],
            bbox_params=dict(
                type='BboxParams',
                # https://github.com/open-mmlab/mmdetection/pull/1354
                # format='pascal_voc',
                format='coco',
                label_fields=['gt_labels'],
                min_visibility=0.0,
                filter_lost_elements=True),
            keymap={ 'img': 'image', 'gt_masks': 'masks', 'gt_bboxes': 'bboxes' },
            update_pad_shape=False,
            skip_img_without_anno=True
        ),
        rotate_aug=dict(
          scale=1.0,
          rotate_range=(-180, 180),
        ),
        # extra_aug=dict(
        #     photo_metric_distortion=dict(
        #         brightness_delta=32,
        #         contrast_range=(0.1, 1.1),
        #         saturation_range=(0.1, 1.1),
        #         hue_delta=18),
        #     ),
        # ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_fold2/val.json',
        img_prefix=data_root + 'val_fold2/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'baseline_test/test.json',
        img_prefix=data_root + 'patch_test/images',
        # ann_file=data_root + 'test1024_ms/DOTA_test1024_ms.json',
        # img_prefix=data_root + 'test1024_ms/images',
        # img_scale=(1024, 1024),
        img_scale=[(1024, 1024)],
        multiscale_mode='value',
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# The config to build the evaluation hook,
# refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    # gamma=0.2,
    warmup_iters=3000,
    warmup_ratio=0.01,
    step=[6, 11])
checkpoint_config = dict(interval=2)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_obb_r50_fpn_1x_dota'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
