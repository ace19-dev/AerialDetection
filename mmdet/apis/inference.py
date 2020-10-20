import warnings
import pdb
import cv2
import random
import os

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint
from mmcv.visualization.color import color_val

from mmdet.core import get_classes, tensor2imgs
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


# TODO: merge this method with the one in BaseDetector
def show_result(img, result, class_names, score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file)


def show_obb_result(data, result, img_norm_cfg, classes, show=False,
                    score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        # class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)

    # if dataset is None:
    #     class_names = dataset.CLASSES
    # elif isinstance(dataset, str):
    #     class_names = get_classes(dataset)
    # elif isinstance(dataset, (list, tuple)):
    #     class_names = dataset
    # else:
    #     raise TypeError(
    #         'dataset must be a valid dataset name or a sequence'
    #         ' of class names, not {}'.format(type(dataset)))

    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        if out_file is not None:
            out_file = os.path.join(out_file, 'OUT_' + img_meta['filename'])
        img_show = img[:h, :w, :]

        bboxes = np.vstack(bbox_result)
        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(
                    0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # set score threshold
        if score_thr > 0:
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        results = draw_poly_detections(img_meta['filename'], img_show.copy(),
                                       bboxes, labels, classes, show, out_file)

    return results


def draw_poly_detections(img_name, img, bboxes, labels, class_names, show, out_file):
    """
    :param img_name:
    :param img:
    :param bboxes:
    :param labels:
    :param class_names:
    :param show:
    :param out_file:
    :return:
    """

    assert isinstance(class_names, (tuple, list))
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]

    img = mmcv.imread(img)
    color_white = (255, 255, 255)
    color_green = (0, 128, 0)

    # if score_thr > 0:
    #     scores = bboxes[:, -1]
    #     inds = scores > score_thr
    #     bboxes = bboxes[inds, :]
    #     labels = labels[inds]

    results = []
    for bbox, label in zip(bboxes, labels):
        object = {}
        # color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        score = bbox[-1]
        bbox_int = bbox.astype(np.int32)
        # left_top = (bbox_int[0], bbox_int[1])
        label_text = class_names[label] if class_names is not None else f'cls {label}'

        cv2.circle(img, (bbox_int[0], bbox_int[1]), 3, (0, 0, 255), -1)
        for i in range(3):
            cv2.line(img, (bbox_int[i * 2], bbox_int[i * 2 + 1]), (bbox_int[(i + 1) * 2], bbox_int[(i + 1) * 2 + 1]),
                     color=color_green, thickness=1)
        cv2.line(img, (bbox_int[6], bbox_int[7]), (bbox_int[0], bbox_int[1]), color=color_green, thickness=1)
        cv2.putText(img, '%s %.3f' % (label_text, score), (bbox_int[0], bbox_int[1] - 10),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.8)

        # submit object
        object['file_name'] = img_name
        object['class_id'] = str(label)
        object['confidence'] = str(score)
        object['point1_x'], object['point1_y'] = str(bbox_int[0]), str(bbox_int[1])
        object['point2_x'], object['point2_y'] = str(bbox_int[2]), str(bbox_int[3])
        object['point3_x'], object['point3_y'] = str(bbox_int[4]), str(bbox_int[5])
        object['point4_x'], object['point4_y'] = str(bbox_int[6]), str(bbox_int[7])
        results.append(object)

    # TODO: bugfix
    # if show:
    #     mmcv.imshow(img, img_name, 0)

    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return results
