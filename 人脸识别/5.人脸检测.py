# -*- coding = utf-8 -*-
# @Time : 2022/10/2 10:23
# @Author : Juyi
# @File : 5.人脸检测.py
# @Software : PyCharm
import cv2 as cv


# 检测人脸的函数
def face_detectDemo(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 转换为灰度图
    # 加载分类器（做出人脸识别的关键，调用cv里的分类器）
    face_detect = cv.CascadeClassifier(
        'E:/Anaconda/envs/juyi/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    face = face_detect.detectMultiScale(gray)  # 多尺度检测  (传入灰度图)
    # gray, 1.05,5, 0, (10, 10), (50,50)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
    cv.imshow('result', img)


img = cv.imread("111.jpg")
# img = cv.resize(img, (1063, 554))
face_detectDemo(img)
# 等待(检测到q后退出)
while True:
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
