#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks
import glob
import os
import logging

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """返回目录中的id列表"""
    return [os.path.splitext(file)[0] for file in os.listdir(dir) if not file.startswith('.')]  # 图片名字作为图片id


def split_ids(ids, n=2):
    """将每个id拆分为n个，为每个id创建n个元组(id, k)"""
    # 等价于for id in ids:
    #       for i in range(n):
    #           (id, i)
    # 得到元祖列表[(id1,0),(id1,1),(id2,0),(id2,1),...,(idn,0),(idn,1)]
    # 这样的作用是后面会通过后面的0,1作为utils.py中get_square函数的pos参数，pos=0的取左边的部分，pos=1的取右边的部分
    return ((id, i) for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """从元组列表中返回经过剪裁的正确img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)  # 重新设置图片大小为原来的scale倍
        yield get_square(im, pos)  # 然后根据pos选择图片的左边或右边


def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """返回所有组(img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)  # 对图像进行转置，将(H, W, C)变为(C, H, W)
    imgs_normalized = map(normalize, imgs_switched)  # 对像素值进行归一化，由[0,255]变为[0,1]

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)  # 对图像的结果也进行相同的处理

    return zip(imgs_normalized, masks)  # 并将两个结果打包在一起


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def get_input_filenames(args):
    images = []
    in_files = args.input
    if os.path.isfile(in_files) and os.path.splitext(in_files)[1] in [".jpg", ".png", ".bmp", ".tif"]:
        images = [in_files]
    elif os.path.isdir(in_files):
        images = []
        for root, dirs, files in os.walk(in_files):
            for file in files:
                if not os.path.splitext(file)[1] in [".jpg", ".png", ".bmp", ".tif"]:
                    logging.warning(f"{file} is not image file skipped")
                    continue
                images.append(os.path.join(root, file))
        # images = [os.path.join(in_files, file) for file in os.listdir(in_files)]
    return images
