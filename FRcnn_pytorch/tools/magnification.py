#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import cv2
import sys


# 定义图像缩放函数
def process_image(img, min_side):
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    else:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2

    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])  # 从图像边界向上,下,左,右扩的像素数目
    return pad_img


# 你的图片文件夹路径
filename = r"‪D:\DataSet\cell_label\9999.jpg"
filename = r"‪D:\DataSet\cell_label\9999_scale.jpg"
im = cv2.imread(filename)
# 输入图片和尺寸
img_new = process_image(im, 512)
cv2.imwrite(filename, img_new)


# num = 0
# yname = []
# # 批量重命名 保证没有中文名字 后面会改回来
# for filename in os.listdir(path):
#     im = cv2.imread(filename)
#     name = filename.split('.')
#     yname.append(name[0])
#     os.rename(os.path.join(path, filename), os.path.join(path, str(num) + '.JPG'))
#     num += 1
# # 批量转换
# for filename in os.listdir(path):
#     im = cv2.imread(path + filename)
#     # 输入图片和尺寸
#     img_new = process_image(im, 128)
#     cv2.imwrite(path + filename, img_new)
#     num = filename.split('.')
#     num = num[0]
#     name = yname[eval(num)]
#     os.rename(os.path.join(path, filename), os.path.join(path, name + '.JPG'))