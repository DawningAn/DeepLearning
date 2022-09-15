# -*- coding = utf-8 -*-
# @Time : 2022/9/14 21:37
# @Author : Juyi
# @File : mnistDemo.py
# @Software : PyCharm
'''
导入mnist数字图片数据集

'''
# 用mnist.py中的load_mnist()函数，就可以按下述方式轻松读入MNIST数据。
import sys, os
sys.path.append(os.pardir) # 为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist
# 第一次调用会花费几分钟 ……
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False)
'''
第 1 个参数normalize设置是否将输入图像正规化为0.0～1.0的值。如果将该参数设置为False，则输入图像的像素会保持原来的0～255。
第2个参数flatten设置是否展开输入图像（变成一维数组）。如果将该参数设置为False，则输入图像为1 × 28 × 28的三维数组；
                                                            若设置为True，则输入图像会保存为由784个元素构成的一维数组。
'''


# 输出各个数据的形状

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)