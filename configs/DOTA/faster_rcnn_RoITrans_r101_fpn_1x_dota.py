CLASSES = ('small ship', 'large ship', 'civilian aircraft', 'military aircraft', 'small car', 'bus', 'truck',
           'train', 'crane', 'bridge', 'oil tank', 'dam', 'athletic field', 'helipad', 'roundabout', 'etc')
# model settings
model = dict(
    type='RoITransformer',
    pretrained='modelzoo://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
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
        anchor_scales=[4],  # original 8
        anchor_ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
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
        reg_class_agnostic=True,
        with_module=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    rbbox_roi_extractor=dict(
        type='RboxSingleRoIExtractor',
        roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    rbbox_head=dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=len(CLASSES),
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
)
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
    rcnn=[
        dict(
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
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssignerRbbox',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomRbboxSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ])
test_cfg = dict(
    rpn=dict(
        # TODO: test nms 2000
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.1),
        max_per_img=2000)
    # score_thr = 0.001, nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img = 2000)

    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)

# dataset settings
# dataset_type = 'DOTADataset'
# data_root = 'data/dota1_1024/'
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
        # img_scale=[(1280, 1024)],
        # multiscale_mode='range',
        img_scale=[(1024, 1024)],
        multiscale_mode='value',
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        rotate_aug=dict(
            scale=1.0,
            rotate_range=(-180, 180),
        ),
        extra_aug=dict(
            # # https://albumentations.readthedocs.io/en/latest/examples.html
            # albu=dict(
            #     transforms=[
            #         dict(
            #             type='ShiftScaleRotate',
            #             shift_limit=0.0625,
            #             scale_limit=0.1,
            #             rotate_limit=45,
            #             interpolation=1,
            #             p=0.3),
            #         dict(
            #             type='RandomBrightnessContrast',
            #             brightness_limit=[0.1, 0.1],
            #             contrast_limit=[0.1, 0.1],
            #             p=0.3),
            #         # dict(type='ChannelShuffle', p=0.1),
            #         dict(type='ToGray', p=0.3),
            #         dict(
            #             type='OneOf',
            #             transforms=[
            #                 dict(type='Blur', blur_limit=11, p=1.0),
            #                 dict(type='MedianBlur', blur_limit=3, p=1.0)
            #             ],
            #             p=0.3),
            #         # dict(
            #         #     type='Cutout',
            #         #     num_holes=10,
            #         #     max_h_size=20,
            #         #     max_w_size=20,
            #         #     fill_value=0,
            #         #     p=0.3
            #         # )
            #     ],
            # ),
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.1, 1.1),
                saturation_range=(0.1, 1.1),
                hue_delta=18),
        ),
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
        img_scale=(1024, 1024),
        multiscale_mode='value',
        # img_scale=[(1280, 768)],
        # multiscale_mode='range',
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True)
)
# The config to build the evaluation hook,
# refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # # gamma=0.2,
    # warmup_iters=1000,
    # warmup_ratio=0.01,
    step=[5, 11])  # when using 'step' policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     # warmup='linear',
#     # warmup_iters=1500,
#     # warmup_ratio=0.01,
#     min_lr_ratio=1e-5)

checkpoint_config = dict(interval=1)
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
work_dir = './work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
# workflow = [('train', 5), ('val', 1)]