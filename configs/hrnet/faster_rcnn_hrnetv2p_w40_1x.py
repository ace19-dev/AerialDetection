CLASSES = ('small ship', 'large ship', 'civilian aircraft', 'military aircraft', 'small car', 'bus', 'truck',
           'train', 'crane', 'bridge', 'oil tank', 'dam', 'athletic field', 'helipad', 'roundabout', 'etc')

# model settings
model = dict(
    type='FasterRCNNOBB',  # TODO: FasterRCNNOBB
    pretrained='open-mmlab://msra/hrnetv2_w40',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(40, 80)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(40, 80, 160)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(40, 80, 160, 320)))),
    neck=dict(
        type='HRFPN',
        in_channels=[40, 80, 160, 320],
        out_channels=256),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4],
        anchor_ratios=[0.5, 1.0, 2.0],
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
    # bbox_head=dict(
    #     type='SharedFCBBoxHead',
    #     num_fcs=2,
    #     in_channels=256,
    #     fc_out_channels=1024,
    #     roi_feat_size=7,
    #     num_classes=len(CLASSES),
    #     target_means=[0., 0., 0., 0.],
    #     target_stds=[0.1, 0.1, 0.2, 0.2],
    #     reg_class_agnostic=False,
    #     loss_cls=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #     loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
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
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        # score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
        score_thr = 0.05,
        nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.5),
        max_per_img = 1000))
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)

# dataset settings
dataset_type = 'ArirangDataset'
data_root = '/home/ace19/dl_data/Arirang_Dataset/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_fold2/train.json',
        img_prefix=data_root + 'train_fold2/images',
        img_scale=[(1280, 640)],
        multiscale_mode='range',  # TODO: value
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        rotate_aug=dict(
            scale=1.2,
            rotate_range=(-180, 180),
        ),
        extra_aug=dict(
            # random_crop=dict(
            #
            # ),
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.2, 1.2),
                saturation_range=(0.2, 1.2),
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
        img_prefix=data_root + 'baseline_test/images',
        # ann_file=data_root + 'test1024_ms/DOTA_test1024_ms.json',
        # img_prefix=data_root + 'test1024_ms/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# The config to build the evaluation hook,
# refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
evaluation = dict(interval=5, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    # gamma=0.2,
    warmup_iters=1500,
    warmup_ratio=0.01,
    step=[16, 19])
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_hrnetv2p_w40_1x'
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033-0f236ef4.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
