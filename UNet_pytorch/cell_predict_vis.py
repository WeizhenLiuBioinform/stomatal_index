import os
import cv2
import yaml
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from unet import UNet
from PIL import Image
from torchvision import transforms
from easydict import EasyDict as edict
from skimage import segmentation, measure, color


def predict_img(net, full_img, device, out_threshold=0.5, input_size=(512, 512)):
    net.eval()
    img = cv2.resize(src=full_img, dsize=input_size, interpolation=cv2.INTER_NEAREST)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = img / 255.0

    img = (img - img.min()) / (img.max() - img.min())
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(size=full_img.shape, interpolation=Image.NEAREST),
        #         transforms.ToTensor()
        #     ]
        # )
        #
        # probs = probs.cpu()
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def mask_to_image(mask):
    return (mask * 255).astype(np.uint8)


def count(predicted, img_shape):
    cleared = cv2.resize(predicted, (int(img_shape[1]/2), int(img_shape[0]/2)), interpolation=cv2.INTER_NEAREST)
    # cleared = img.copy()
    segmentation.clear_border(cleared)  # 清除与边界相连的目标物

    label_image = measure.label(cleared, connectivity=2)  # 2代表8连通，1代表4联通区域标记
    props = measure.regionprops(label_image)
    count1 = len(props)
    count2 = len(props)

    # borders = np.logical_xor(img, cleared)  # 异或扩
    # label_image[borders] = -1
    dst = color.label2rgb(label_image, image=cleared, bg_label=0)  # 不同标记用不同颜色显示

    regions = measure.regionprops(label_image)
    areas = [region.area for region in regions]
    rects = []
    for region in regions:
        if region.area < (np.mean(areas) / 10):
            count2 -= 1
            continue
        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.5)
        rects.append(rect)
    return count1, count2, dst, rects


def load_unet(n_channels, n_classes, model_path, device):
    net = UNet(n_channels=n_channels, n_classes=n_classes)
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("U-Net load checkpoint %s" % model_path)
    return net


def cell_count(net, img, device):
    mask_prediction = predict_img(net=net,
                                  full_img=img,
                                  device=device,
                                  out_threshold=0.5)
    mask_pred = mask_to_image(mask_prediction)
    bifilter = cv2.bilateralFilter(mask_pred, 9, 75, 75)  # 双边滤波模糊去噪，保留边缘信息
    ret, binary = cv2.threshold(bifilter, 230, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    cells_count1, cells_count2, label_Unet, rects_Unet = count(opening.astype(np.uint8), img.shape)
    return cells_count1, cells_count2, label_Unet, rects_Unet


def cell_vis(output_dir, image_name, label_Unet, rects_Unet, cells_count):
    figure, ax = plt.figure()
    plt.imshow(label_Unet)
    for rect_Unet in rects_Unet:
        print(rect_Unet)
        ax.add_patch(rect_Unet)
    plt.title(f"Number of cells after area filtration : {cells_count}", fontsize=12)
    plt.xticks([])
    plt.yticks([])

    # filtration = cv2.cvtColor((label_Unet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # for rect_Unet in rects_Unet:
    #     cv2.rectangle(filtration,
    #                   rect_Unet.get_xy(),
    #                   (rect_Unet.get_xy()[0] + rect_Unet.get_width(),
    #                    rect_Unet.get_xy()[1] + rect_Unet.get_height()),
    #                   (0, 0, 255),
    #                   2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{image_name}_cell_out.png"))


if __name__ == "__main__":
    with open("./cfgs/predict.yml", 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    unet_load_name = r"/home/zhucc/stomata_index/frcnn_pytorch/models/res101/pascal_voc_datacom/_fold_0/faster_rcnn_1_20_298.pth"
    UNet = load_unet(n_channels=1, n_classes=1, model_path=unet_load_name, device=device)

    output_dir = r"/home/zhucc/stomata_index/vis"
    os.makedirs(output_dir, exist_ok=True)
    image_dir = "/home/zhucc/stomata_index/Semantic_segmentation/Pytorch-UNet/data/imgs"
    image_list = os.listdir(image_dir)[1:2]
    for image_name in image_list:
        if '_stoma_det' in image_name or (
                os.path.splitext(image_name)[1] not in ['.jpg', '.jpeg', '.png', '.tif', '.bmp']):
            print(image_name + "has been skipped")
            continue
        im_file = os.path.join(image_dir, image_name)
        image = cv2.imread(im_file, 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        num_connected_domains, num_cells, label_cells, cell_rects = cell_count(UNet, image, device)
        cell_vis(output_dir, image_name, label_cells, cell_rects, num_cells)

        # plt.figure(figsize=(16, 10), dpi=300)
        #
        # plt.subplot(231)
        # plt.imshow(image)
        # plt.title("label_GT", fontsize=12)
        # plt.xticks([])
        # plt.yticks([])
        # # cv2.imwrite("origin.jpg", img)
        #
        # plt.subplot(232)
        # plt.imshow(mask_pred)
        # plt.title("mask_pred", fontsize=12)
        # plt.xticks([])
        # plt.yticks([])
        # # cv2.imwrite("mask_pred.jpg", mask_pred)
        #
        # plt.subplot(233)
        # plt.imshow(binary)
        # plt.title("bilateralFilter and binary", fontsize=12)
        # plt.xticks([])
        # plt.yticks([])
        # # cv2.imwrite("bb.jpg", binary)
        #
        # plt.subplot(234)
        # plt.imshow(opening)
        # plt.title("opening", fontsize=12)
        # plt.xticks([])
        # plt.yticks([])
        # # cv2.imwrite("opening.jpg", opening)
        #
        # plt.subplot(235)
        # plt.imshow(label_Unet)
        # plt.title(f"Number of connected domains  : {cells_count1}", fontsize=12)
        # plt.xticks([])
        # plt.yticks([])
        # # cv2.imwrite("domains.jpg", cv2.cvtColor((label_Unet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        #
        # ax6 = plt.subplot(236)
        # plt.imshow(label_Unet)
        # for rect_Unet in rects_Unet:
        #     print(rect_Unet)
        #     ax6.add_patch(rect_Unet)
        # plt.title(f"Number of cells after area filtration : {cells_count2}", fontsize=12)
        # plt.xticks([])
        # plt.yticks([])
        #
        # filtration = cv2.cvtColor((label_Unet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # for rect_Unet in rects_Unet:
        #     cv2.rectangle(filtration,
        #                   rect_Unet.get_xy(),
        #                   (rect_Unet.get_xy()[0] + rect_Unet.get_width(),
        #                    rect_Unet.get_xy()[1] + rect_Unet.get_height()),
        #                   (0, 0, 255),
        #                   2)
        # # cv2.imwrite("filtration.jpg", filtration)
        #
        # plt.tight_layout()
        # plt.savefig(os.path.join(args.output, f"{os.path.splitext(image)}_out.png"))
