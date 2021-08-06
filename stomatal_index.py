from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import cv2
import os
import torch
import pprint
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.utils.config import cfg
from stoma_detect_vis import stomata_count, stomata_vis, load_faster_rcnn
from cell_predict_vis import load_unet, cell_count, cell_vis

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda = torch.cuda.is_available()
    image_dir = "/home/zhucc/stomata_index/SI/images"
    image_list = os.listdir(image_dir)

    is_vis = True
    output_dir = r"/home/zhucc/stomata_index/SI/vis"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(cfg.RNG_SEED)

    # Stomatal counting environment
    frcnn_load_name = r"/home/zhucc/stomata_index/SI/FRcnn_pytorch/checkpoints/faster_rcnn_1_20_298.pth"
    pascal_classes = np.asarray(['__background__', 'stomata'])
    Faster_RCNN = load_faster_rcnn(frcnn_load_name, cuda, pascal_classes)
    # print('Faster R-CNN config:')
    # pprint.pprint(cfg)

    # Cell counting environment
    device = torch.device('cuda:0' if cuda else 'cpu')
    unet_load_name = r"/home/zhucc/stomata_index/SI/UNet_pytorch/checkpoints/CP_epoch200.pth"
    UNet = load_unet(n_channels=1, n_classes=1, model_path=unet_load_name, device=device)


    with open(os.path.join(output_dir, "stomatal_index.csv"), "w", encoding='utf-8-sig') as F:
        csv_write = csv.writer(F)
        csv_head = ['image', 'stomata', 'cells', 'SI(%)']
        csv_write.writerow(csv_head)

        for image_name in tqdm(image_list):
            if '_stoma_out' in image_name or '_cell_out' in image_name or '_SI_out' in image_name or (
                    os.path.splitext(image_name)[1] not in ['.jpg', '.jpeg', '.png', '.tif', '.bmp']):
                print(image_name + "has been skipped")
                continue

            image = cv2.imread(os.path.join(image_dir, image_name), 1)
            num_stomata, label_stomata = stomata_count(Faster_RCNN, image, cuda, pascal_classes)
            # stomata_vis(output_dir, image_name, num_stomata, im2show)

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            num_connected_domains, num_cells, label_cells, cell_rects = cell_count(UNet, image_gray, device)
            # cell_vis(output_dir, image_name, label_Unet, cell_rects, num_cells)

            si = round((num_stomata / (num_stomata + num_cells) * 100), 3)

            line = [image_name, num_stomata, num_cells, si]
            csv_write.writerow(line)
            if is_vis:
                fig = plt.figure(figsize=(8, 4), dpi=200)
                fig.suptitle(f"Stomatal index is {si} %", fontsize=12)
                ax1 = plt.subplot(121)
                plt.imshow(cv2.cvtColor(label_stomata, cv2.COLOR_BGR2RGB))
                ax1.set_title(f"Number of stomata: {num_stomata}", fontsize=12)
                plt.xticks([])
                plt.yticks([])

                ax2 = plt.subplot(122)
                plt.imshow(label_cells)
                for rect in cell_rects:
                    ax2.add_patch(rect)

                ax2.set_title(f"Number of cells: {num_cells}", fontsize=12)
                plt.xticks([])
                plt.yticks([])

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_SI_out.jpg"))
                plt.close()
