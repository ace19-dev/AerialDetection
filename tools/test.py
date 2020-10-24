import os
import csv
import json
import argparse
import os.path as osp
import shutil
import tempfile
import time
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist, show_obb_result
from mmdet.core import results2json, coco_eval, tensor2imgs
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.24):
    model.eval()

    results = []
    submit_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # print("###### ", data)
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        # print("####### ", results)

        # if show:
        # model.module.show_result(data, result, dataset.img_norm_cfg)
        ori_file_name = data['img_meta'][0].data[0][0]['filename'].split('__')[0]
        # print(ori_file_name)

        # show DOTA coordinate
        new_result = show_obb_result(data, result, dataset.img_norm_cfg, dataset.CLASSES,
                                     show=show, score_thr=show_score_thr,
                                     out_file=os.path.join(out_dir, 'images', ori_file_name))
        if len(new_result) == 0:
            continue

        # # for submit
        # submit_results.extend(new_result)

        filename = new_result[0]['file_name'][:-4] + '.json'
        out_json_path = os.path.join(out_dir, 'json', ori_file_name)
        if not os.path.exists(out_json_path):
            os.makedirs(out_json_path)

        with open(os.path.join(out_json_path, filename), "w") as json_file:
            anno = {'content': new_result}
            json.dump(anno, json_file)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    # TODO: fix for merge submit
    # if not os.path.exists('../submit'):
    #     os.makedirs('../submit')
    #
    # fout = open('../submit/%s_submission.csv' % 'season-2',
    #             'w', encoding='UTF-8', newline='')
    # writer = csv.writer(fout)
    # writer.writerow(['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y',
    #                  'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y', ])
    # for r in submit_results:
    #     writer.writerow([r['file_name'], r['class_id'], r['confidence'],
    #                      r['point1_x'], r['point1_y'],
    #                      r['point2_x'], r['point2_y'],
    #                      r['point3_x'], r['point3_y'],
    #                      r['point4_x'], r['point4_y']])
    # fout.close()

    return results


def multi_gpu_test(model, data_loader, tmp_dir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmp_dir)

    return results


def collect_results(result_part, size, tmp_dir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmp_dir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmp_dir = tempfile.mkdtemp()
            tmp_dir = torch.tensor(
                bytearray(tmp_dir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmp_dir)] = tmp_dir
        dist.broadcast(dir_tensor, 0)
        tmp_dir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmp_dir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmp_dir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmp_dir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmp_dir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', default='../configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py',
                        help='test config file path')
    parser.add_argument('--checkpoint', default='../pretrained/epoch_18.pth',
                        help='checkpoint file')
    # Filename of the output results in pickle format.
    parser.add_argument('--out', help='output result file in pickle format')
    # parser.add_argument('--out', default='../pretrained/outputs/out.pkl',
    #                     help='output result file in pickle format')

    # EVAL_METRICS: Items to be evaluated on the results. Allowed values depend on the dataset,
    # e.g., proposal_fast, proposal, bbox, segm are available for COCO,
    # mAP, recall for PASCAL VOC.
    # Cityscapes could be evaluated by cityscapes as well as all COCO metrics.
    parser.add_argument('--eval',
                        type=str,
                        nargs='+',
                        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
                        default=['bbox'],
                        help='eval types')
    # If specified, detection results will be plotted on the images and shown in a new window.
    # It is only applicable to single GPU testing and used for debugging and visualization.
    parser.add_argument('--show', default=False, help='show results')
    # set show dir
    parser.add_argument('--show-dir', default='../submit/results',
                        help='directory where painted images will be saved')
    # parser.add_argument('--show-dir',
    #                     help='directory where painted images will be saved')
    parser.add_argument('--tmp_dir', help='tmp dir for writing some results')
    parser.add_argument('--log_dir', help='log the inference speed')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmp_dir)

    # TODO: 나중에 적용.
    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_file = args.out + '.json'
                    results2json(dataset, outputs, result_file)
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}.json'.format(name)
                        results2json(dataset, outputs_, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)


if __name__ == '__main__':
    main()
