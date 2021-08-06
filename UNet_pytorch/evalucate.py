import argparse
import logging
import os
import yaml
from functools import wraps

import numpy as np
import torch
import time
import cv2
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import segmentation, measure, color
import skimage.morphology as sm
import matplotlib.patches as mpatches
from tqdm import tqdm
from PIL import Image
from skimage.morphology import skeletonize
from torchvision import transforms
from easydict import EasyDict as edict

from dice_loss import dice_coeff
from unet import UNet
from utils.data_vis import plot_img_and_mask, myskeletonize
from utils.dataset import BasicDataset
from utils.load import get_input_filenames, get_output_filenames


from utils.segmentation_metric import SegmentationMetric


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5,
                init_shape=(1024, 1360),
                input_size=(512, 512)):
    net.eval()

    img = cv2.resize(src=full_img, dsize=input_size, interpolation=cv2.INTER_NEAREST)
    # img = cv2.normalize(img, 0, 1.0, norm_type=cv2.NORM_MINMAX)
    # img = np.expand_dims(img, 0)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = img / 255.0

    img = (img - img.min()) / (img.max() - img.min())
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # starter.record()
        output = net(img)
        # ender.record()

        # WAIT FOR GPU SYNC
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # timings[index] = curr_time
        if args.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=init_shape, interpolation=Image.NEAREST),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def mask_to_image(mask):
    return (mask * 255).astype(np.uint8)


def count(image, flag, filter_factor):
    # bifilter = cv2.bilateralFilter(image, 9, 75, 75)  # 双边滤波模糊去噪，保留边缘信息
    # ret, binary = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    # skeletonized = myskeletonize(1 - binary / 255)
    # skeletonized = skeletonize(1 - binary / 255)

    rects = []
    count = 0
    if flag == "mask_true" or flag == "mask_pred":
        cleared = image.copy()  # 复制
        segmentation.clear_border(cleared)  # 清除与边界相连的目标物

        label_image = measure.label(image, connectivity=2)  # 2代表8连通，1代表4联通区域标记
        borders = np.logical_xor(image, cleared)  # 异或扩
        label_image[borders] = -1
        dst = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示
        props = measure.regionprops(label_image)
        count = len(props)
        regions = measure.regionprops(label_image)
        areas = [region.area for region in regions]
        for region in regions:  # 循环得到每一个连通区域属性集
            # 忽略小区域
            if region.area < (np.mean(areas) / 10):
                count -= 1

    if flag == "mask_pred_Post_treatment":
        cleared = image.copy()  # 复制
        segmentation.clear_border(cleared)  # 清除与边界相连的目标物

        label_image = measure.label(image, connectivity=2)  # 2代表8连通，1代表4联通区域标记
        borders = np.logical_xor(image, cleared)  # 异或扩
        label_image[borders] = -1
        dst = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示
        props = measure.regionprops(label_image)
        count = len(props)
        regions = measure.regionprops(label_image)
        areas = [region.area for region in regions]

        for region in regions:  # 循环得到每一个连通区域属性集
            # 忽略小区域
            if region.area < (np.mean(areas) / filter_factor):
                count -= 1
                # continue

            # 绘制外包矩形
            # minr, minc, maxr, maxc = region.bbox
            #
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1.0)
            # rects.append(rect)
            # ax1.add_patch(rect)
            # label = img_as_ubyte(label)
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 12))
    # ax0.imshow(cleared, cmap="gray")
    # ax1.imshow(dst)
    return count, rects


if __name__ == "__main__":
    from warnings import simplefilter

    simplefilter(action='ignore', category=FutureWarning)

    with open("./cfgs/predict.yml", 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))
    if_save_out_image = False
    if_count = True
    if_eval = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    dir_checkpoint = "checkpoint"
    groups = ["10x", "20x", "com"]
    folds = ["0", "1", "2", "3", "4"]
    epoch_list = np.append(np.array([1]), (np.arange(0, 201, 5)[1:]))
    metric = SegmentationMetric(2)
    for group in groups:
        for fold in folds:
            for e in epoch_list:
                modelpath = os.path.join(dir_checkpoint, group, f"checkpoints{group}_fold{fold}")
                model = os.path.join(modelpath, f"CP_epoch{str(e)}.pth")
                input_size = args.inputsize
                with open(f"data{group}/CV/fold_{fold}/val.txt") as f:
                    images_list = list(map(lambda line: line.strip() + '.jpg', f))
                out_dir = os.path.join("evaluation", f"detect_output_{group}", f"fold_{fold}")
                os.makedirs(out_dir, exist_ok=True)
                # out_files = get_output_filenames(args)

                net = UNet(n_channels=1, n_classes=1)
                logging.info("Loading model {}".format(model))

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                logging.info(f'Using device {device}')
                net.to(device=device)
                net.load_state_dict(torch.load(model, map_location=device))
                logging.info("Model loaded !")


                with open(out_dir + f'/cells_epoch{e}.csv', 'w+', newline='', encoding='utf-8-sig') as f:
                    csv_write = csv.writer(f)
                    csv_head = ['path', 'cells_UNet']
                    column = 1
                    if if_eval:
                        csv_head.append('cells_GT')
                        csv_head.append('error')
                        csv_head.append('accuracy(%)')
                        csv_head.append('pixelACC')
                        csv_head.append('mIoU')
                        column = column + 1
                    csv_write.writerow(csv_head)
                    accuracies = []
                    errors = []
                    pixelACCs = []
                    mIoUs = []
                    for fn in tqdm(images_list):
                        try:
                            # logging.info("\nPredicting image {} ...".format(fn))
                            img = cv2.imread(os.path.join(f"data{group}/imgs", fn), 0)
                            init_shape = img.shape

                            mask_prediction = predict_img(net=net,
                                                          full_img=img,
                                                          device=device,
                                                          init_shape=init_shape,
                                                          out_threshold=args.mask_threshold)

                            mask_pred = mask_to_image(mask_prediction)
                            if if_save_out_image:
                                out_file = "{}_OUT{}".format(out_dir, os.path.splitext(fn)[1])
                                # cv2.imwrite(os.path.join(out_dir, os.path.basename(out_file)), mask_pred)
                                # logging.info("Mask saved to {}".format(out_file))

                            if if_count:
                                fig, axes = plt.subplots(1, column, figsize=(20, 12))
                                ret, mask_pred = cv2.threshold(mask_pred, 230, 1, cv2.THRESH_BINARY)
                                cells_Unet, rects_Unet = count(mask_pred, flag="mask_pred")
                                # cells_Unet2, label_Unet2, rects_Unet2 = count(mask_pred, flag="mask_pred_Post_treatment")
                                # axes[0].imshow(label_Unet)
                                # for rect_Unet in rects_Unet:
                                #     axes[0].add_patch(rect_Unet)
                                # axes[0].set_title("Unet_predicted:" + str(cells_Unet), fontsize=18, color='r')
                                # plt.xticks([]), plt.yticks([])
                                line = [fn, cells_Unet]
                                if if_eval:
                                    mask_true = cv2.imread(f"data{group}/masks" + '/' + os.path.basename(fn),
                                                           cv2.IMREAD_GRAYSCALE)
                                    ret, mask_true = cv2.threshold(mask_true, 230, 1, cv2.THRESH_BINARY)
                                    cells_GT, rects_GT = count(mask_true, flag="mask_true")
                                    line.append(cells_GT)
                                    # axes[1].imshow(label_GT)
                                    # axes[1].set_title("Ground Truth:" + str(cells_GT), fontsize=18, color='r')
                                    # plt.xticks([]), plt.yticks([])
                                    for rect_GT in rects_GT:
                                        axes[1].add_patch(rect_GT)
                                    # if mask_true.max() > 1:
                                    #     mask_true = mask_true / 255.0
                                    #
                                    # mask_true[mask_true >= 0.6] = 1
                                    # mask_true[mask_true < 0.6] = 0

                                    # if args.n_classes > 1:
                                    #     dice = F.cross_entropy(mask_pred, mask_true).item()
                                    # else:
                                    #     # pred = torch.sigmoid(torch.from_numpy(mask_prediction))
                                    #     pred = torch.from_numpy(mask_prediction)
                                    #     pred = (pred > args.mask_threshold).float
                                    #     dice = dice_coeff(pred, mask_true).item()

                                    accuracy = (1 - (abs(cells_GT - cells_Unet) / cells_GT)) * 100

                                    error = abs(cells_GT - cells_Unet)

                                    metric.addBatch(mask_pred, mask_true)
                                    pixelACC = metric.pixelAccuracy()
                                    mIoU = metric.meanIntersectionOverUnion()

                                    accuracies.append(accuracy)
                                    errors.append(error)
                                    pixelACCs.append(pixelACC)
                                    mIoUs.append(mIoU)

                                    line.append(error)
                                    line.append(accuracy)
                                    line.append(pixelACC)
                                    line.append(mIoU)

                                    csv_write.writerow(line)
                        except IOError:
                            print(os.path.join(f"data{group}/imgs", fn) + "error!!")

                        plt.close()
                    average_accuracy = np.mean(accuracies)
                    average_error = np.mean(errors)
                    average_pixelACCs = np.mean(pixelACCs)
                    average_mIoUs = np.mean(mIoUs)
                    average_line = ['average', '-', '-', average_error, average_accuracy, average_pixelACCs,
                                    average_mIoUs]
                    csv_write.writerow(average_line)
