# -*- coding = utf-8 -*-
# @Time : 2022/10/6 21:38
# @Author : Juyi
# @File : 8.人脸录入.py
# @Software : PyCharm

import cv2 as cv

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv.VideoCapture(0)
flag = 1
num = 1

while (cap.isOpened()):  # 检测摄像头开启状态
    ret_flag, Vshow = cap.read()
    cv.imshow("Capture_Test", Vshow)
    k = cv.waitkey(1) & 0xFF  # 按键中断
    if k == ord('s'):
        cv.imwrite("C:/Users/Judy/Desktop/save_Test/" + str(num) + ".name" + ".jpg", Vshow)  # 保存当前帧图像格式（name可以自设定）
        print("Success to save!" + str(num) + ".jpg")
        print("--------------------------")
        num += 1
    elif k == ord(' '):
        break
# 释放摄像头
cap.release()
# 释放内存
cv.destroyAllWindows()
