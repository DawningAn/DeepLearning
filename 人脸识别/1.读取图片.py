# -*- coding = utf-8 -*-
# @Time : 2022/10/2 9:57
# @Author : Juyi
# @File : 1.读取图片.py
# @Software : PyCharm

import cv2 as cv

img = cv.imread("face1.png")
cv.imshow("read_img", img)
# 等待
cv.waitKey(0)
# 释放内存
cv.destroyAllWindows()
