# -*- coding = utf-8 -*-
# @Time : 2022/7/26 20:31
# @Author : Juyi
# @File : 01.py
# @Software : PyCharm
import numpy as np

# 读入波士顿房价训练数据
datafile = 'D:\Python\pythonProject\ML\PaddlePaddle\housing.data'
data = np.fromfile(datafile, sep=' ')
print(data)
# [6.320e-03 1.800e+01 2.310e+00 ... 3.969e+02 7.880e+00 1.190e+01]
# 可以看到，读入之后的数据被转化成1维array，观察原数据集，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推....
# 这里对原始数据做reshape，变成 N x 14的形式
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])  # 把data数据变为 N x 14

# 查看数据
x = data[0]
print(x.shape)
print(x)  # x为第一个样本的各项属性值

# 设置分割比率为8：2 （训练集：测试集）
ratio = 0.8
# 80%的数据用作训练集，20%用作测试集
offset = int(data.shape[0] * ratio)
# 506个样本数进行分割

training_data = data[:offset]
print(training_data.shape)  # 训练集共有404个样本
# 归一化处理
# 对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间
# 计算train数据集的最大值，最小值，平均值
maximums, minimums, avgs = \
                            training_data.max(axis=0), \
                            training_data.min(axis=0), \
                training_data.sum(axis=0) / training_data.shape[0]

# 对数据进行归一化处理
for i in range(feature_num):
    # print(maximums[i], minimums[i], avgs[i])
    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

# 封装为函数
def load_data():
    # 从文件导入数据
    datafile = './housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

if __name__ == "__main__":
# 获取数据
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]

    # 查看数据
    print(x[0])
    print(y[0])