# -*- coding = utf-8 -*-
# @Time : 2022/10/13 20:49
# @Author : Juyi
# @File : 10.人脸识别.py
# @Software : PyCharm

import cv2 as cv
import os
import urllib
import urllib.request

#加载训练好的数据集文件
recogizer = cv.face.LBPHFaceRecognizer_create()
#加载数据
recogizer.read('train/trainer.yml')
