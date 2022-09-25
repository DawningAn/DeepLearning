# -*- coding = utf-8 -*-
# @Time : 2022/9/25 18:39
# @Author : Juyi
# @File : Demo01.py
# @Software : PyCharm
import torch


#考虑深度学习Pytorch的基本流程实现

# batch：把数据集分为多少个batch（块），以块为单位送入神经网络
# epoch：把所有batch送完，我们需要训练epoch轮，送入神经网络epoch次

'''
考虑通俗一点的流程

# 1. 数据处理如何输入
# 2. 训练数据的路径
# 3. epoch轮数
# 4. batch_size
# 5. 输出（得到了什么，图片image和目标值label）
'''

dir = "./data/"     # 假设的训练数据的存放目录,假设有100张图片

epoch = 5

def 挤牙膏操作(dir, batch_size):
    # 读取文件夹
    读取文件夹(dir)
    image, label = 按照batch_size进行数据拆分(batch_size)
    image = 数据预处理(image)
    return image,label

def 神经网络(image):
    return pre_label

for i in range(epoch):  # 5轮
    for j in range(10):  # batch_size 为10
        image, label = 挤牙膏操作(dir,batch_size = 10)  # 需要返回image和label
        pre_label = 神经网络(image)
        lossFunction(pre_label,label)

