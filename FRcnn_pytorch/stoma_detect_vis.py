from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import cv2

import time
import torch
import pprint
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

from imageio import imread
from model.rpn.bbox_transform import clip_boxes

from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def _get_image_blob(im):
    """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def load_faster_rcnn(load_name, cuda, pascal_classes):
    # initilize the network here.
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()

    if cuda:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    print("Faster R-CNN load checkpoint %s" % load_name)

    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    if cuda:
        cfg.CUDA = True

    if cuda:
        fasterRCNN.cuda()
    fasterRCNN.eval()
    return fasterRCNN


def stomata_count(fasterRCNN, image, cuda, pascal_classes):
    if cuda:
        cfg.USE_GPU_NMS = True
    im_in = image
    if len(im_in.shape) == 2:
        im_in = im_in[:, :, np.newaxis]
        im_in = np.concatenate((im_in, im_in, im_in), axis=2)

    blobs, im_scales = _get_image_blob(im_in)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor()
    im_info = torch.FloatTensor()
    num_boxes = torch.LongTensor()
    gt_boxes = torch.FloatTensor()

    # ship to cuda
    if cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    class_agnostic = False
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            if class_agnostic:
                if cuda:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if cuda:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    num_stomata = 0
    label_stomata = np.copy(image)
    for j in xrange(1, len(pascal_classes)):
        inds = torch.nonzero(scores[:, j] > int(0.5)).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            dets = cls_dets.cpu().numpy()
            label_stomata, num_stomata = vis_detections(label_stomata, pascal_classes[j], dets, 0.9)
    return num_stomata, label_stomata


def stomata_vis(output_dir, image_name, num_stomata, im2show):
    im2show = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(im2show)
    plt.title(f"The remaining bboxes filtered by a threshold of 0.9 : {num_stomata}", fontsize=8)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(os.path.join(output_dir, f"/{image_name}_stoma_out.jpg"))
    plt.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    image_dir = "/home/zhucc/stomata_index/Semantic_segmentation/Pytorch-UNet/data/imgs"
    image_list = os.listdir(image_dir)[1:2]
    load_name = r"/home/zhucc/stomata_index/frcnn_pytorch/models/res101/pascal_voc_datacom/_fold_0/faster_rcnn_1_20_298.pth"
    output_dir = r"/home/zhucc/stomata_index/frcnn_pytorch/vis"
    os.makedirs(output_dir, exist_ok=True)
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    cuda = torch.cuda.is_available()
    pascal_classes = np.asarray(['__background__', 'stomata'])
    Faster_RCNN = load_faster_rcnn(load_name, cuda, pascal_classes)
    for image_name in image_list:
        if '_stoma_det' in image_name or (
                os.path.splitext(image_name)[1] not in ['.jpg', '.jpeg', '.png', '.tif', '.bmp']):
            print(image_name + "has been skipped")
            continue
        im_file = os.path.join(image_dir, image_name)
        image = cv2.imread(im_file)
        num_stomata, im2show = stomata_count(Faster_RCNN, image, cuda, pascal_classes)
        stomata_vis(output_dir, image_name, num_stomata, im2show)
