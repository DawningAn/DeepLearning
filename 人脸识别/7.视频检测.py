# -*- coding = utf-8 -*-
# @Time : 2022/10/2 21:01
# @Author : Juyi
# @File : 7.视频检测.py
# @Software : PyCharm
import cv2 as cv

# 检测人脸的函数
# 要读取摄像头的图像
# cap = cv.VideoCapture(0)  # 0为默认摄像头
cap = cv.VideoCapture("01.mp4")


def face_detectDemo(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 转换为灰度图

    # 加载分类器（做出人脸识别的关键，调用cv里的分类器）
    face_detect = cv.CascadeClassifier(
        'E:/Anaconda/envs/juyi/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    face = face_detect.detectMultiScale(gray, 1.06)  # 多尺度检测  (传入灰度图)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y, x + w, y + h), color=(0, 0, 255), thickness=1)
    cv.imshow('result', img)


# 等待(检测到q后退出)
while True:
    # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵
    flag, frame = cap.read()
    if not flag:
        break
    face_detectDemo(frame)  # 识别每一帧
    if ord('q') == cv.waitKey(1):
        break
# 释放内存
cv.destroyAllWindows()
