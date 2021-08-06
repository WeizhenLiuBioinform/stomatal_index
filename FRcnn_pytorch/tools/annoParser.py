import os
import random
import argparse
import glob
import shutil

import yaml
from easydict import EasyDict as edict

from tqdm import tqdm

from tools.rotateAlbumentations import BboxAlbumentation

if __name__ == '__main__':

    with open("../cfgs/train.yml", 'r', encoding="utf-8") as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    random.seed(cfg.SEED)
    traindata_path = f'{cfg.DATA_DIR}/VOCdevkit2007/VOC2007/JPEGImages/'
    DIR_CV = f'{cfg.DATA_DIR}/VOCdevkit2007/VOC2007/ImageSets/train_val/'
    img_list = os.listdir(traindata_path)
    random.shuffle(img_list)
    val_list = []
    train_list = []
    for k in range(0, cfg.K_FOLD):
        txtpath = DIR_CV + 'fold_' + str(k)
        os.makedirs(txtpath, exist_ok=True)
        val_list = img_list[(len(img_list) // cfg.K_FOLD) * k:(len(img_list) // cfg.K_FOLD) * (k + 1)]
        train_list = [image for image in img_list if image not in val_list]
        with open(txtpath + '/train.txt', 'a')as fr:
            for img in train_list:
                name = img[:-4] + '\n'
                fr.write(name)

        with open(txtpath + '/val.txt', 'a')as fv:
            for img in val_list:
                name = img[:-4] + '\n'
                fv.write(name)

        # train set augmentation
        imgs_path = f'{cfg.DATA_DIR}/VOCdevkit2007/VOC2007/JPEGImages'
        xmls_path = f'{cfg.DATA_DIR}/VOCdevkit2007/VOC2007/Annotations/'
        train_img_save_path = f'{cfg.DATA_DIR}/VOCdevkit2007/VOC2007/JPEGImages/JPEGImages_fold_{k}/train'
        train_xml_save_path = f'{cfg.DATA_DIR}/VOCdevkit2007/VOC2007/Annotations/Annotations_fold_{k}/train'

        val_img_save_path = f'{cfg.DATA_DIR}/VOCdevkit2007/VOC2007/JPEGImages/JPEGImages_fold_{k}/val'
        val_xml_save_path = f'{cfg.DATA_DIR}/VOCdevkit2007/VOC2007/Annotations/Annotations_fold_{k}/val'

        os.makedirs(train_img_save_path, exist_ok=True)
        os.makedirs(train_xml_save_path, exist_ok=True)
        os.makedirs(val_img_save_path, exist_ok=True)
        os.makedirs(val_xml_save_path, exist_ok=True)

        for val_image in tqdm(val_list):
            shutil.copyfile(f"{imgs_path}/{val_image}", f"{val_img_save_path}/{val_image}")
            shutil.copyfile(f"{xmls_path}/{os.path.splitext(val_image)[0] + '.xml'}",
                            f"{val_xml_save_path}/{os.path.splitext(val_image)[0] + '.xml'}")

        for angle in [45, 90]:
            bboxAug = BboxAlbumentation(angle=angle, imgs_path=imgs_path, xmls_path=xmls_path,
                                        img_save_path=train_img_save_path,
                                        xml_save_path=train_xml_save_path)

            for image in tqdm(train_list):
                shutil.copyfile(f"{imgs_path}/{image}", f"{train_img_save_path}/{image}")
                shutil.copyfile(f"{xmls_path}/{os.path.splitext(image)[0] + '.xml'}",
                                f"{train_xml_save_path}/{os.path.splitext(image)[0] + '.xml'}")
                filename = os.path.join(imgs_path, image)
                bboxAug.aug(filename)

        for path in [train_img_save_path, train_xml_save_path, val_img_save_path, val_xml_save_path]:
            name = ''
            if 'train' in path:
                startNumber = '1'
            elif 'val' in path:
                startNumber = '1201'
            else:
                print('startNumber error')
                break

            if 'JPEGImages' in path:
                fileType = '.jpg'
            elif 'Annotations' in path:
                fileType = '.xml'
            else:
                print('fileType error')
                break

            print("正在生成以" + name + startNumber + fileType + "迭代的文件名")
            count = 0
            filelist = os.listdir(path)
            filelist.sort()
            for files in filelist:
                Olddir = os.path.join(path, files)
                if os.path.isdir(Olddir):
                    continue
                Newdir = os.path.join(os.path.abspath(os.path.join(path, "../")), name + str(count + int(startNumber)).rjust(6, '0') + fileType)
                os.rename(Olddir, Newdir)
                count += 1
            print("一共修改了" + str(count) + "个文件")
            os.rmdir(path)



