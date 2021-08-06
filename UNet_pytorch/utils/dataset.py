import random
import os
from os.path import splitext
from os import listdir

import cv2
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from utils.augment import Augmentation, strong_aug
import matplotlib.pyplot as plt
from torchvision import transforms
import pdb


def collate_fn(data):
    images = [s['image'] for s in data]
    # masks = [cv2.cvtColor(s['mask'], cv2.COLOR_RGB2GRAY) for s in data]
    masks = [s['mask'] for s in data]

    images = np.expand_dims(images, 0)
    masks = np.expand_dims(masks, 0)

    images = torch.from_numpy(np.stack(images, axis=0)).type(torch.float32)
    masks = torch.from_numpy(np.stack(masks, axis=0)).type(torch.float32)

    images = images.permute(1, 0, 2, 3)
    masks = masks.permute(1, 0, 2, 3)

    assert images.shape == masks.shape, f'Images and masks should be the same shape, but are {images.shape} and {masks.shape}'
    # print(images.shape, masks.shape)
    return {
        'image': images,
        'mask': masks
    }


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, ids, transform=None, mask_suffix='', dir_augs="data/aug"):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.ids = ids
        self.transform = transform
        self.dir_augs = dir_augs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i].strip()

        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        # print(idx)
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img_ = cv2.imread(img_file[0], 1)
        mask_ = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)

        img_RGB = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        mask_gray = mask_

        # data = {"image": img_RGB, "mask": mask_RGB}

        if self.transform is not None:
            # augmented = self.transform(**data)
            augmented = self.transform(image=img_RGB, mask=mask_gray)
            img, mask = augmented["image"], augmented["mask"]

        os.makedirs(self.dir_augs, exist_ok=True)
        if len(os.listdir(self.dir_augs)) < 100:
            plt.figure()
            plt.subplot(121), plt.imshow(img, cmap='gray'), plt.xlabel("image_aug"), plt.axis('off')
            plt.subplot(122), plt.imshow(mask, cmap='gray'), plt.xlabel("mask_aug"), plt.axis('off')
            plt.savefig(os.path.join(self.dir_augs, str(idx) + '_' + str(random.randint(1, 10000)) + '_aug.png'))
            plt.close()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if img.max() > 1:
            img = img / 255.0

        if mask.max() > 1:
            mask = mask / 255.0

        img = (img - img.min()) / (img.max() - img.min())
        # mask = (mask - mask.min()) / (mask.max() - mask.min())

        mask[mask >= 0.6] = 1
        mask[mask < 0.6] = 0

        return {
            'image': img,
            'mask': mask
        }
        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        #     # 'image': img.type(torch.FloatTensor),
        #     # 'mask': mask.type(torch.FloatTensor)
        # }


# @classmethod
# def augmentdata(cls, img, mask):
#     aug = Augmentation()
#     aug_list = ['rotate', 'flip', 'randomResizeCrop']
#
#     n = random.randint(0, 3)
#     aug_now = random.sample(aug_list, n)
#     for a in aug_now:
#         if a == 'rotate':
#             img, mask = aug.rotate(img, mask)
#         if a == 'flip':
#             img, mask = aug.flip(img, mask)
#         if a == 'randomResizeCrop':
#             img, mask = aug.randomResizeCrop(img, mask)
#     return img, mask

# @classmethod
# def preprocess(cls, pil_img, inputsize=(512, 512)):
#     gray_img = pil_img.convert("L")
#     assert inputsize[0] > 0 and inputsize[1] > 0, 'Scale is too small'
#     img_reszie = gray_img.resize(inputsize, Image.BILINEAR)
#
#     img_nd = np.array(img_reszie)
#     if len(img_nd.shape) == 2:
#         img_nd = np.expand_dims(img_nd, axis=2)
#
#     # HWC to CHW
#     img_trans = img_nd.transpose((2, 0, 1))
#
#     # 归一化
#     if img_nd.max() > 1:
#         img_trans = img_trans / 255
#     return img_trans

# @classmethod
# def augment(cls, img, mask, index, inputsize=(512, 512), save_dir="data/aug"):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)
#         print('create aug dir.....')
#     gray_img = img.convert("L")
#     gray_mask = mask.convert("L")
#     assert inputsize[0] > 0 and inputsize[1] > 0, 'Scale is too small'
#     img_reszie = gray_img.resize(inputsize, Image.NEAREST)
#     mask_reszie = gray_mask.resize(inputsize, Image.NEAREST)
#
#     img_nd = np.array(img_reszie, dtype=np.float32)
#     mask_nd = np.array(mask_reszie, dtype=np.float32)
#
#     # img_aug, mask_aug = cls.augmentdata(img_reszie, mask_reszie)
#     augmentation = strong_aug(p=0.9)
#     data = {"image": img_nd, "mask": mask_nd}
#     augmented = augmentation(**data)
#     img_aug, mask_aug = augmented["image"], augmented["mask"]
#
#     # plt.figure()
#     # plt.subplot(121), plt.imshow(img_aug, cmap='gray'), plt.xlabel("image_aug"), plt.axis('off')
#     # plt.subplot(122), plt.imshow(mask_aug, cmap='gray'), plt.xlabel("mask_aug"), plt.axis('off')
#     # plt.savefig(os.path.join(save_dir, str(index) +'_'+ str(random.randint(1, 10000)) + '_aug.png'))
#     # plt.close()
#
#     # img_aug.save(os.path.join(save_dir, 'img', str(index) + '_aug.png'))
#     # mask_aug.save(os.path.join(save_dir, 'mask', str(index) + '_aug.png'))
#
#     # img_nd = np.array(img_aug)
#     # mask_nd = np.array(mask_aug)
#
#     if len(img_aug.shape) == 2:
#         img_aug = np.expand_dims(img_aug, axis=2)
#
#     if len(mask_aug.shape) == 2:
#         mask_aug = np.expand_dims(mask_aug, axis=2)
#
#     # HWC to CHW
#     img_trans = img_aug.transpose((2, 0, 1))
#     mask_trans = mask_aug.transpose((2, 0, 1))
#
#     # 归一化
#     if img_trans.max() > 1:
#         img_trans = img_trans / 255.0
#     if mask_trans.max() > 1:
#         mask_trans = mask_trans / 255.0
#     return img_trans, mask_trans

class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
