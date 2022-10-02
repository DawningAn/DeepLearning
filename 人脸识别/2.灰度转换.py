# -*- coding = utf-8 -*-
# @Time : 2022/10/2 10:02
# @Author : Juyi
# @File : 2.灰度转换.py
# @Software : PyCharm
import cv2 as cv

img = cv.imread("face1.png")
# 灰度图
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 显示灰度
cv.imshow("gray_img", gray_img)
# 保存灰度图
cv.imwrite("gray_face1.png", gray_img)
cv.imshow("read_img", img)
# 等待
cv.waitKey(0)
# 释放内存
cv.destroyAllWindows()
