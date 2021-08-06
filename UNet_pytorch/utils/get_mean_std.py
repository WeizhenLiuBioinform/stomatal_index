# coding: utf-8

import numpy as np
import cv2
from PIL import Image
import random
import os
import tqdm
import glob
import matplotlib.pyplot as plt

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""

PATH_IMAGES = r"/home/zhucc/stomata_index/CS-X-15-2019.11.02"
# train_txt_path = os.path.join("..", "..", "Data/train.txt")


CNum = 100  # 挑选多少图片进行计算

img_h, img_w = 512, 512
final_means, final_stdevs = [], []

for _ in range(20):
    Path_list = glob.glob(f"{PATH_IMAGES}/*/*.jpg")
    random.shuffle(Path_list)  # shuffle , 随机挑选图片
    imgs = np.zeros([img_w, img_h, 1])
    for i in tqdm.tqdm(range(CNum)):
        img_path = Path_list[i].rstrip()
        # img = cv2.imdecode(np.fromfile(img_path, dtype=np.float32), -1)
        img = Image.open(img_path)
        img = img.convert('L')
        img = img.resize((img_h, img_w), Image.BILINEAR)
        img = np.array(img)
        img = img[:, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=2)

    imgs = imgs.astype(np.float32) / 255.
    means, stdevs = [], []

    pixels = imgs[:, :, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
    # for i in range(3):
    #     pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    #     means.append(np.mean(pixels))
    #     stdevs.append(np.std(pixels))
    print(f"第{_ + 1}批的means为{means}")
    print(f"第{_ + 1}批的stdevs为{stdevs}")
    final_means.append(means)
    final_stdevs.append(stdevs)

    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()

final_mean = np.mean(np.array(final_means), axis=0)
final_std = np.mean(np.array(final_stdevs), axis=0)
print("normMean = {}".format(final_mean))
print("normStd = {}".format(final_std))
print('transforms.Normalize(mean = {}, std = {})'.format(final_mean, final_std))

# gray 1000
# transforms.Normalize(mean = [0.46274137], std = [0.10995317])
# gray 2000
# transforms.Normalize(mean = [0.46152964], std = [0.10963361])

# RGB：
# transforms.Normalize(mean = [0.4976264  0.45133978 0.3993562], std = [0.11552592 0.10886826 0.10727626])



