# -*- coding = utf-8 -*-
# @Time : 2022/10/2 10:09
# @Author : Juyi
# @File : 3.修改尺寸.py
# @Software : PyCharm
import cv2 as cv

img = cv.imread("face1.png")
# 修改尺寸
resize_img = cv.resize(img, dsize=(200, 200))
# 显示修改后的
cv.imshow("resize_img", resize_img)
# 打印原图大小
print("未修改", img.shape)
# 打印修改后的大小
print("修改后", resize_img.shape)

# 等待(检测到q后退出)
while True:
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
