import numpy as np
import cv2
import os
import tqdm

__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N

P      TP    FP

N      FN    TN

"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    # def Frequency_Weighted_Intersection_over_Union(self):
    #     # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
    #     freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
    #     iu = np.diag(self.confusion_matrix) / (
    #             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
    #             np.diag(self.confusion_matrix))
    #     FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    #     return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def genComparedImages(self, predictedpath, labelpath, file_path):
        predictedfilepath = os.path.join(predictedpath, file_path)
        imgPredict = cv2.imread(predictedfilepath, cv2.IMREAD_GRAYSCALE)
        bifilter = cv2.bilateralFilter(imgPredict, 9, 75, 75)  # 双边滤波模糊去噪，保留边缘信息
        normalized = cv2.normalize(bifilter, None, 0, 255, cv2.NORM_MINMAX)
        ret, binaryPredict = cv2.threshold(normalized, 230, 1, cv2.THRESH_BINARY)
        labeledfilepath = os.path.join(labelpath, file_path)
        imgLabel = cv2.imread(labeledfilepath, cv2.IMREAD_GRAYSCALE)
        ret, binaryLabel = cv2.threshold(imgLabel, 230, 1, cv2.THRESH_BINARY)
        Predict = np.array(binaryPredict)
        Label = np.array(binaryLabel)
        imagePair = [Predict, Label]
        yield imagePair


if __name__ == '__main__':
    labelpath = r"/home/zhucc/stomata_index/Semantic_segmentation/0510_Unet/0826forpaper/data/membrane/test/label_U8"
    predictedpath= r"/home/zhucc/stomata_index/Semantic_segmentation/0510_Unet/0826forpaper/data/membrane/test/predicted"
    metric = SegmentationMetric(2)
    quantity = len(os.listdir(predictedpath))
    print(quantity)
    acc = 0
    mIoU = 0
    for iteration in tqdm.tqdm(range(quantity)):
        # print(iteration)
        imagepair = metric.genComparedImages(predictedpath, labelpath, iteration)
        Predict, Label = next(imagepair)
        metric.addBatch(Predict, Label)
        acc = acc + metric.pixelAccuracy()
        mIoU = mIoU + metric.meanIntersectionOverUnion()
    print("评The average acc is {:.4f}\nThe average mIoU is {:.4f}".format(acc/quantity, mIoU/quantity))