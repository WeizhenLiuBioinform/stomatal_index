# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：kaggle -> 12
@IDE    ：PyCharm
@Author ：zhucc
@Date   ：2020/5/13 21:17
@Desc   ：
==================================================
"""
# !/usr/bin/env python

import cv2
import math
import numpy as np
import os
import pdb
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from urllib.request import urlopen

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    Rotate,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose,
    BboxParams
)


class BboxAlbumentation:
    def __init__(self, angle, imgs_path, xmls_path, img_save_path, xml_save_path):
        self.angle = angle
        self.imgs_path = imgs_path
        self.xmls_path = xmls_path
        self.img_save_path = img_save_path
        self.xml_save_path = xml_save_path

    def get_aug(self, aug, min_area=625, min_visibility=0.5):
        return Compose(aug,
                       bbox_params=BboxParams(format='pascal_voc', min_area=min_area, min_visibility=min_visibility))

    def getAnnotations(self, filename):
        bboxes = self.getBbox(filename)
        image = cv2.imread(filename, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotations = {'image': image,
                       'bboxes': bboxes}
        return annotations

    def getBbox(self, filename):
        bboxes = []
        xml_name = self.getXmlName(filename)
        old_tree = ET.parse(xml_name)
        old_root = old_tree.getroot()
        for box in old_root.iter('bndbox'):
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            b = [xmin, ymin, xmax, ymax, "stomata"]
            bboxes.append(b)
        return bboxes

    def getXmlName(self, filename):
        return os.path.join(self.xmls_path, os.path.split(filename)[1].split('.')[0] + '.xml')

    def aug(self, _filename):
        d, n = os.path.split(_filename)
        name = os.path.splitext(n)[0]
        transform = self.get_aug(
            [Rotate(limit=(self.angle, self.angle), interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT,
                    value=None, mask_value=None, always_apply=True, p=1)])
        annotations = self.getAnnotations(_filename)
        transformed = transform(image=annotations['image'], bboxes=annotations['bboxes'])
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        transformed_image_name = name + '_' + str(self.angle) + "d.jpg"
        transformed_xml_name = name + '_' + str(self.angle) + "d.xml"
        image_save_name = os.path.join(self.img_save_path, transformed_image_name)
        xml_save_name = os.path.join(self.xml_save_path, transformed_xml_name)
        # 写入图像
        # cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_save_name, transformed_image)

        # xml_path = os.path.join(self.xmls_path, name + '.xml')
        # old_tree = ET.parse(xml_path)
        # file_name = old_tree.find('filename').text  # it is origin name
        # path = old_tree.find('path').text  # it is origin path

        root = ET.Element('annotation')

        folder = ET.SubElement(root, 'folder')
        folder.text = 'stomata_label1000'

        filename = ET.SubElement(root, 'filename')
        filename.text = transformed_image_name

        path = ET.SubElement(root, 'path')
        path.text = os.path.join(d, transformed_image_name)

        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'

        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = '1360'

        height = ET.SubElement(size, 'height')
        height.text = '1024'

        depth = ET.SubElement(size, 'depth')
        depth.text = '3'

        segmented = ET.SubElement(root, 'segmented')
        root.find('segmented').text = '0'

        for box in transformed_bboxes:
            object = ET.SubElement(root, 'object')

            name = ET.SubElement(object, 'name')
            name.text = 'stomata'
            pose = ET.SubElement(object, 'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(object, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(object, 'difficult')
            difficult.text = '0'
            bndbox = ET.SubElement(object, 'bndbox')

            if self.angle == 45:
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(box[0]) + 7)
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(box[1]) + 7)
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(box[2]) - 7)
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(box[3]) - 7)
            else:
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(box[0]))
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(box[1]))
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(box[2]))
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(box[3]))

        # write into new xml
        tree = ET.ElementTree(root)
        tree.write(xml_save_name)


if __name__ == '__main__':
    imgs_path = r'/home/zhucc/stomata_index/frcnn_pytorch/datacom/VOCdevkit2007/VOC2007/JPEGImages-backup/'
    xmls_path = r'/home/zhucc/stomata_index/frcnn_pytorch/datacom/VOCdevkit2007/VOC2007/Annotations-backup/'
    img_save_path = r'/home/zhucc/stomata_index/frcnn_pytorch/datacom/VOCdevkit2007/VOC2007/ImageAugmentations/'
    xml_save_path = r'/home/zhucc/stomata_index/frcnn_pytorch/datacom/VOCdevkit2007/VOC2007/XmlAugmentations/'

    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(xml_save_path, exist_ok=True)

    for angle in [45, 90]:
        bboxAug = BboxAlbumentation(angle=angle, imgs_path=imgs_path, xmls_path=xmls_path, img_save_path=img_save_path,
                                    xml_save_path=xml_save_path)
        for image in tqdm(os.listdir(imgs_path)):
            if not image.endswith('.jpg'):
                continue
            filename = os.path.join(imgs_path, image)
            bboxAug.aug(filename)


    # for name in tqdm(os.listdir(xmls_path)):
    #     new_name = name[3:]
    #     os.rename(os.path.join(xmls_path, name), os.path.join(xmls_path, new_name))
