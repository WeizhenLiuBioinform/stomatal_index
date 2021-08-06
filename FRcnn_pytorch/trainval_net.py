from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

from torch.optim.lr_scheduler import LambdaLR

import _init_paths
import os
import numpy as np
import argparse
from easydict import EasyDict as edict
import pprint
import time
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb

from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, \
    clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from validation import validation


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def fit_one_epoch(args, fasterRCNN, epoch, output_dir, dataloader, val_dataloader, imdb, val_imdb):
    fasterRCNN.train()
    imdb.competition_mode(on=False)
    loss_temp = 0
    start = time.time()

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.disp_interval == 0:
            end = time.time()
            if step > 0:
                loss_temp /= (args.disp_interval + 1)

            if args.mGPUs:
                loss_rpn_cls = rpn_loss_cls.mean().item()
                loss_rpn_box = rpn_loss_box.mean().item()
                loss_rcnn_cls = RCNN_loss_cls.mean().item()
                loss_rcnn_box = RCNN_loss_bbox.mean().item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt
            else:
                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

            print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                  % (args.session, epoch, step, iters_per_epoch, loss_temp, optimizer.param_groups[0]['lr']))
            print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
            print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                  % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
            if args.use_tfboard:
                info = {
                    'loss': loss_temp,
                    'loss_rpn_cls': loss_rpn_cls,
                    'loss_rpn_box': loss_rpn_box,
                    'loss_rcnn_cls': loss_rcnn_cls,
                    'loss_rcnn_box': loss_rcnn_box
                }
                logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                   (epoch - 1) * iters_per_epoch + step)
                logger.add_scalar("logs_s_{}/lr".format(args.session), optimizer.param_groups[0]['lr'],
                                  (epoch - 1) * iters_per_epoch + step)

            loss_temp = 0
            start = time.time()

    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    if (epoch % args.checkpoint_epoch_interval == 0) or (epoch == args.max_epochs):
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    map = validation(val_dataloader, epoch, save_name, val_imdb, args)
    if args.use_tfboard:
        logger.add_scalar("logs_s_{}/map".format(args.session), map, epoch)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="GPUid for training")
    parser.add_argument(
        "--gpu",
        dest='gpu',
        help="which devices to use",
        default="0,1",
        type=str,
    )
    parm = parser.parse_args()
    return parm


if __name__ == '__main__':

    with open("cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(args.SEED)
    parm = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = parm.gpu
    # 获取配置文件
    cfg_file = "cfgs/{}.yml".format(args.net)
    hyp_file = "cfgs/train.yml"
    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if hyp_file is not None:
        cfg_from_file(hyp_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    cfg.CUDA = args.cuda
    lr = args.lr

    # 打印超参数配置
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)  # For reproducibility(default=3 in config.py)
    # torch.backends.cudnn.benchmark = True
    for k_fold in range(args.K_FOLD):
        shutil.rmtree(args.DATA_DIR + '/cache', ignore_errors=True)
        shutil.rmtree(args.DATA_DIR + '/VOCdevkit2007/results/VOC2007/Main', ignore_errors=True)
        shutil.rmtree(args.DATA_DIR + '/VOCdevkit2007/annotations_cache', ignore_errors=True)
        if os.path.exists(args.DATA_DIR + '/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt_annots.pkl'):
            os.remove(args.DATA_DIR + '/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt_annots.pkl')
            print("Cache cleared successfully")
        with open("cfgs/train.yml", 'r', encoding="utf-8") as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)

        doc['k_fold'] = k_fold
        with open("cfgs/train.yml", 'w', encoding="utf-8") as f:
            yaml.dump(doc, f)
        print('fold_' + str(k_fold) + ' is running...')

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, training=True, USE_FLIPPED=True)
        val_imdb, val_roidb, val_ratio_list, val_ratio_index = combined_roidb(args.val_imdb_name, training=False,
                                                                              USE_FLIPPED=False)
        train_size = len(roidb)
        val_size = len(val_roidb)
        print('{:d} roidb entries for training'.format(train_size))
        print('{:d} roidb entries for validation'.format(val_size))

        sampler_batch = sampler(train_size, args.batch_size)

        train_dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler_batch,
                                                       num_workers=args.num_workers)

        val_dataset = roibatchLoader(val_roidb, val_ratio_list, val_ratio_index, 1, val_imdb.num_classes,
                                     training=False, normalize=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                     num_workers=0, pin_memory=True)


        output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/" + f"_fold_{k_fold}"
        print(f'output_dir is {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if args.cuda:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        # initilize the network here.
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN.create_architecture()

        params = []
        for key, value in dict(fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        if args.cuda:
            fasterRCNN.cuda()

        optimizer = {
            "adam": lambda: optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-8, eps=1e-08),
            "rmsprop": lambda: optim.RMSprop(params, lr=lr, momentum=cfg.TRAIN.MOMENTUM, eps=0.001, weight_decay=1e-8),
            "sgd": lambda: optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        }[args.optimizer]()

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs * len(dataloader), eta_min=0, last_epoch=-1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=10)
        scheduler = LambdaLR(optimizer, lambda x: (((1 + np.cos(x * np.pi / args.max_epochs)) / 2) ** 1.0) * 0.9 + 0.1)

        if args.resume:
            load_name = os.path.join(output_dir,
                                     'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                       args.checkpoint))
            print("loading checkpoint %s" % load_name)
            checkpoint = torch.load(load_name)
            args.session = checkpoint['session']
            args.start_epoch = checkpoint['epoch']
            fasterRCNN.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
            print("loaded checkpoint %s" % load_name)

        if args.mGPUs:
            fasterRCNN = nn.DataParallel(fasterRCNN)

        iters_per_epoch = int(train_size / args.batch_size)
        if args.use_tfboard:
            from tensorboardX import SummaryWriter

            logger = SummaryWriter(logdir=f'runs/{args.dataset}/FOLD_{k_fold}')

        for epoch in range(args.start_epoch, args.max_epochs + 1):
            fit_one_epoch(args, fasterRCNN, epoch, output_dir, train_dataloader, val_dataloader, imdb, val_imdb)
            scheduler.step()

        if args.use_tfboard:
            logger.close()
