import random
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
# 图像变换函数
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, CLAHE, RandomRotate90, ElasticTransform, RandomGamma,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomSizedCrop, PadIfNeeded,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


# from albumentations
def strong_aug(p=0.5):
    return Compose([
        # 非破坏性转换
        RandomRotate90(0.5),  # Randomly rotate the input by 90 degrees zero or more times.
        Flip(p=0.5),  # Flip the input either horizontally, vertically or both horizontally and vertically.
        # Transpose(),  # Transpose the input by swapping rows and columns.
        # OneOf([
        #     IAAAdditiveGaussianNoise(),
        #     GaussNoise(),  # Apply gaussian noise to the input image.
        # ], p=0.2),
        # OneOf([
        #     MotionBlur(p=0.2),  # Apply motion blur to the input image using a random-sized kernel.
        #     MedianBlur(blur_limit=3, p=0.1),  # Blur the input image using using a median filter with a random aperture linear size.
        #     Blur(blur_limit=3, p=0.1),  # Blur the input image using a random-sized kernel.
        # ], p=0.2),
        ShiftScaleRotate(p=0.7),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, interpolation=cv2.INTER_CUBIC,
        #                  border_mode=cv2.BORDER_REFLECT, p=0.5),  # andomly apply affine transforms: translate, scale and rotate the input.
        # OneOf([
        #     OpticalDistortion(p=0.3),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),

        # OneOf([
        #     CLAHE(clip_limit=2),
        #     IAASharpen(),
        #     IAAEmboss(),
        #     RandomBrightnessContrast(),
        # ], p=0.3),
        # HueSaturationValue(p=0.3),
        # OneOf([RandomSizedCrop(min_max_height=(512, 512),
        #                        height=512, width=512, p=0.5),
        #        PadIfNeeded(min_height=512,
        #                    min_width=512, p=0.5)], p=1),
        # 非刚体转换
        # ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        # OneOf([
        #     ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     # GridDistortion(p=0.5),
        #     OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        # ], p=0.8),
        # 非空间性转换
        # CLAHE(p=0.8),
        # RandomBrightnessContrast(p=0.8),
        # RandomGamma(p=0.8)
    ], p=p)


class Augmentation:
    def __init__(self):
        pass

    def rotate(self, image, mask, angle=None):
        if angle is None:
            angle = transforms.RandomRotation.get_params([-180, 180])  # -180~180随机选一个角度旋转
        if isinstance(angle, list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def flip(self, image, mask):  # 水平翻转和垂直翻转
        if random.random() > 0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        if random.random() < 0.5:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def randomResizeCrop(self, image, mask, scale=(0.4, 1.0),
                         ratio=(1, 1)):  # scale表示随机crop出来的图片会在的0.3倍至1倍之间，ratio表示长宽比
        img = np.array(image)
        h_image, w_image = img.shape
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image, mask

    def adjustContrast(self, image, mask):
        factor = transforms.RandomRotation.get_params([0, 10])  # 这里调增广后的数据的对比度
        image = tf.adjust_contrast(image, factor)
        # mask = tf.adjust_contrast(mask,factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def adjustBrightness(self, image, mask):
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_brightness(image, factor)
        # mask = tf.adjust_contrast(mask, factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def centerCrop(self, image, mask, size=None):  # 中心裁剪
        if size == None: size = image.size  # 若不设定size，则是原图。
        image = tf.center_crop(image, size)
        mask = tf.center_crop(mask, size)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def adjustSaturation(self, image, mask):  # 调整饱和度
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_saturation(image, factor)
        # mask = tf.adjust_saturation(mask, factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask


def augmentationData(image_path, mask_path, option=[1, 2, 3, 4, 5, 6, 7], save_dir=None):
    '''
    :param image_path: 图片的路径
    :param mask_path: mask的路径
    :param option: 需要哪种增广方式：1为旋转，2为翻转，3为随机裁剪并恢复原本大小，4为调整对比度，5为中心裁剪(不恢复原本大小)，6为调整亮度,7为饱和度
    :param save_dir: 增广后的数据存放的路径
    '''
    aug_image_savedDir = os.path.join(save_dir, 'img')
    aug_mask_savedDir = os.path.join(save_dir, 'mask')
    if not os.path.exists(aug_image_savedDir):
        os.makedirs(aug_image_savedDir)
        print('create aug image dir.....')
    if not os.path.exists(aug_mask_savedDir):
        os.makedirs(aug_mask_savedDir)
        print('create aug mask dir.....')
    aug = Augmentation()
    res = os.walk(image_path)
    images = []
    masks = []
    for root, dirs, files in res:
        for f in files:
            images.append(os.path.join(root, f))
    res = os.walk(mask_path)
    for root, dirs, files in res:
        for f in files:
            masks.append(os.path.join(root, f))
    datas = list(zip(images, masks))
    num = len(datas)

    for (image_path, mask_path) in datas:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        if 1 in option:
            num += 1
            image_tensor, mask_tensor = aug.rotate(image, mask)
            image_rotate = transforms.ToPILImage()(image_tensor).save(
                os.path.join(save_dir, 'img', str(num) + '__rotate.jpg'))
            mask_rotate = transforms.ToPILImage()(mask_tensor).save(
                os.path.join(save_dir, 'mask', str(num) + '_rotate_mask.jpg'))
        if 2 in option:
            num += 1
            image_tensor, mask_tensor = aug.flip(image, mask)
            image_filp = transforms.ToPILImage()(image_tensor).save(
                os.path.join(save_dir, 'img', str(num) + '_filp.jpg'))
            mask_filp = transforms.ToPILImage()(mask_tensor).save(
                os.path.join(save_dir, 'mask', str(num) + '_filp_mask.jpg'))
        if 3 in option:
            num += 1
            image_tensor, mask_tensor = aug.randomResizeCrop(image, mask)
            image_ResizeCrop = transforms.ToPILImage()(image_tensor).save(
                os.path.join(save_dir, 'img', str(num) + '_ResizeCrop.jpg'))
            mask_ResizeCrop = transforms.ToPILImage()(mask_tensor).save(
                os.path.join(save_dir, 'mask', str(num) + '_ResizeCrop_mask.jpg'))
        if 4 in option:
            num += 1
            image_tensor, mask_tensor = aug.adjustContrast(image, mask)
            image_Contrast = transforms.ToPILImage()(image_tensor).save(
                os.path.join(save_dir, 'img', str(num) + '_Contrast.jpg'))
            mask_Contrast = transforms.ToPILImage()(mask_tensor).save(
                os.path.join(save_dir, 'mask', str(num) + '_Contrast_mask.jpg'))
        if 5 in option:
            num += 1
            image_tensor, mask_tensor = aug.centerCrop(image, mask)
            image_centerCrop = transforms.ToPILImage()(image_tensor).save(
                os.path.join(save_dir, 'img', str(num) + '_centerCrop.jpg'))
            mask_centerCrop = transforms.ToPILImage()(mask_tensor).save(
                os.path.join(save_dir, 'mask', str(num) + '_centerCrop_mask.jpg'))
        if 6 in option:
            num += 1
            image_tensor, mask_tensor = aug.adjustBrightness(image, mask)
            image_Brightness = transforms.ToPILImage()(image_tensor).save(
                os.path.join(save_dir, 'img', str(num) + '_Brightness.jpg'))
            mask_Brightness = transforms.ToPILImage()(mask_tensor).save(
                os.path.join(save_dir, 'mask', str(num) + '_Brightness.jpg'))
        if 7 in option:
            num += 1
            image_tensor, mask_tensor = aug.adjustSaturation(image, mask)
            image_Saturation = transforms.ToPILImage()(image_tensor).save(
                os.path.join(save_dir, 'img', str(num) + '_Saturation.jpg'))
            mask_Saturation = transforms.ToPILImage()(mask_tensor).save(
                os.path.join(save_dir, 'mask', str(num) + '_Saturation.jpg'))

# augmentationData(r'E:\datasets\isbi\train\images', r'E:\datasets\isbi\train\label', save_dir=r'E:\代码\mytext\suanfa\aug')
