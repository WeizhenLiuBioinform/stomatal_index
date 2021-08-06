from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.optim.lr_scheduler import LambdaLR

import _init_paths
import os
import sys
import numpy as np
import argparse
from easydict import EasyDict as edict
import pprint
import pdb
import time
import yaml
import matplotlib.pyplot as plt
import torch
import pickle
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb

from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, \
    clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


def validation(val_dataloader, epoch, model_name, val_imdb, args):
    val_imdb.competition_mode(on=True)
    print('Start Validation')
    val_fasterRCNN = resnet(val_imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    val_fasterRCNN.create_architecture()

    print("load checkpoint %s" % model_name)
    checkpoint = torch.load(model_name)
    val_fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    if args.cuda:
        val_im_data = torch.FloatTensor(1).cuda()
        val_im_info = torch.FloatTensor(1).cuda()
        val_num_boxes = torch.LongTensor(1).cuda()
        val_gt_boxes = torch.FloatTensor(1).cuda()
        val_fasterRCNN.cuda()
        cfg.CUDA = True
    else:
        val_im_data = torch.FloatTensor(1)
        val_im_info = torch.FloatTensor(1)
        val_num_boxes = torch.LongTensor(1)
        val_gt_boxes = torch.FloatTensor(1)

    val_im_data = Variable(val_im_data)
    val_im_info = Variable(val_im_info)
    val_num_boxes = Variable(val_num_boxes)
    val_gt_boxes = Variable(val_gt_boxes)

    start = time.time()
    # 每张图像最大目标检测数量
    max_per_image = 100

    thresh = 0.0

    save_name = 'val_' + args.exp_group
    num_images = len(val_imdb.image_index)
    # 创建[[[],[]...[]],[[],[]...[]]] 1,2,200
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(val_imdb.num_classes)]

    output_dir = get_output_dir(val_imdb, save_name)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    save_dir = os.path.join(output_dir, f"PRCurves_{args.exp_group}")
    os.makedirs(save_dir, exist_ok=True)
    det_file = os.path.join(save_dir, f'epoch_{epoch}_detections.pkl')

    val_fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i, data in enumerate(val_dataloader):
        with torch.no_grad():
            val_im_data.resize_(data[0].size()).copy_(data[0])
            val_im_info.resize_(data[1].size()).copy_(data[1])
            val_gt_boxes.resize_(data[2].size()).copy_(data[2])
            val_num_boxes.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        val_rois, val_cls_prob, val_bbox_pred, \
        val_rpn_loss_cls, val_rpn_loss_box, val_RCNN_loss_cls, \
        val_RCNN_loss_bbox, val_rois_label = val_fasterRCNN(val_im_data, val_im_info, val_gt_boxes, val_num_boxes)

        scores = val_cls_prob.data
        boxes = val_rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = val_bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(val_imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, val_im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        for j in range(1, val_imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, val_imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, val_imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s \r'
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    map = val_imdb.evaluate_detections(all_boxes, epoch, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))

    return map


if __name__ == '__main__':
    hyp_file = "./cfgs/validation.yml"
    with open(hyp_file, 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if hyp_file is not None:
        cfg_from_file(hyp_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    val_imdb, val_roidb, val_ratio_list, val_ratio_index = combined_roidb(args.val_imdb_name, training=False,
                                                                          USE_FLIPPED=False)
    val_imdb.competition_mode(on=True)
    val_size = len(val_roidb)

    print('{:d} roidb entries for validation'.format(val_size))

    val_dataset = roibatchLoader(val_roidb, val_ratio_list, val_ratio_index, 1, val_imdb.num_classes, training=False,
                                 normalize=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                                 pin_memory=True)

    model_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    maps = []
    for epoch in range(1, 31):
        load_name = os.path.join(model_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, epoch, args.checkpoint))
        map = validation(val_dataloader, epoch, load_name, val_imdb, args)
        maps.append(map)
    print(maps)
    plt.figure()
    plt.plot(maps, label=args.exp_group)
    plt.title(args.exp_group + " mAP")
    plt.xlabel("epoch")
    plt.ylabel("mAP")
    plt.savefig(f'./mAP_{args.exp_group}.jpg')
    plt.close()
