from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import _init_paths
import os
import pprint
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import LambdaLR
from easydict import EasyDict as edict
from model.faster_rcnn.resnet import resnet
from torch.utils.data.sampler import Sampler
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list

from trainval_net import set_seed, sampler, fit_one_epoch

if __name__ == '__main__':

    with open("FRcnn_pytorch/cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(args.SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # 获取配置文件
    cfg_file = "FRcnn_pytorch/cfgs/{}.yml".format(args.net)
    hyp_file = "FRcnn_pytorch/cfgs/train.yml"
    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if hyp_file is not None:
        cfg_from_file(hyp_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    cfg.CUDA = args.cuda
    lr = args.lr

    print('Using config:')
    pprint.pprint(cfg)
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