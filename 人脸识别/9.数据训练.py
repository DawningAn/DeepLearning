# -*- coding = utf-8 -*-
# @Time : 2022/10/7 20:15
# @Author : Juyi
# @File : 9.数据训练.py
# @Software : PyCharm
import os
import cv2 as cv
from PIL import Image
import numpy as np


# 数据训练可以寻找图像来训练，或是采用自己制作
def getImageLabel(path):
    # 存储人脸数据
    facesSamples = []
    # 存储姓名数据
    ids = []
    # 存储图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv.CascadeClassifier(
        'E:/Anaconda/envs/juyi/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    # 遍历保存的图像
    for imagePath in imagePaths:
        # 打开图片，灰度化PIL有九种不同模式：1(黑白)，L（灰度图），P，RGB,RGBA,CMYK,YCbCr,I,F
        PIL_img = Image.open(imagePath).convert('L')
        # 将图像转换为数组
        img_numpy = np.array(PIL_img, 'uint8')
        # 获取人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0])
        # 预防空人脸图像
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y + h, x:x + w])
        # 打印脸部特征和id
    print('id:', id)
    print('faces:', facesSamples)
    return facesSamples, ids


if __name__ == "__main__":
    # 要训练的图片路径
    path = './IMG/'  # 找到图片路径
    # 获取人脸特征和ID
    faces, ids = getImageLabel(path)
    # 加载识别器
    recognizer = cv.face.LBPHFaceRecognizer_create()
    # 训练过程
    recognizer.train(faces, np.array(ids))
    # 保存文件
    recognizer.write("train/trainer.yml")
