# -*- coding = utf-8 -*-
# @Time : 2022/9/15 21:03
# @Author : Juyi
# @File : neuralnet_mnist.py
# @Software : PyCharm
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import *
'''
推理一下这个神经网络
该网络的输入层有784个神经元，输出层有10个神经元。输入层的784这个数字来源于图像大小的28 × 28 = 784
输出层的10这个数字来源于10类别分类（数字0到9，共10类别）。
此外，这个神经网络有2个隐藏层，第1个隐藏层有50个神经元，第2个隐藏层有100个神经元。这个50和100可以设置为任何值
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)  # 防止溢出
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y

# 加载数据集
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return x_test, t_test

'''
init_network()会读入保存在pickle文件sample_weight.pkl中的学习到的权重参数
'''
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y



# Accuracy 精度
'''
for语句逐一取出保存在x中的图像数据，用predict()函数进行分类。
predict()函数以NumPy数组的形式输出各个标签对应的概率。
比如输出[0.1, 0.3, 0.2, ..., 0.04]的数组，该数组表示“0”的概率为0.1，“1”的概率为0.3，等等。
然后，取出这个概率列表中的最大值的索引（第几个元素的概率最高），作为预测结果。
可以用np.argmax(x)函数取出数组中的最大值的索引，np.argmax(x)将获取被赋给参数x的数组中的最大值元素的索引。
最后，比较神经网络所预测的答案和正确解标签，将回答正确的概率作为识别精度
'''
'''
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1  # +1表示当前这个图像（i）已经正确识别
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352
'''

'''
考虑打包输入多张图像的情形。比如，我们想用predict()函数一次性打包处理100张图像。为此，可以把x的形状改为100 × 784，将100张图像打包作为输入数据
输入数据的形状为 100 × 784，输出数据的形状为100 × 10。这表示输入的100张图像的结果被一次性输出了。
比如，x[0]和y[0]中保存了第0张图像及其推理结果，x[1]和y[1]中保存了第1张图像及其推理结果，等等。

这种打包式的输入数据称为批（batch）

批处理一次性计算大型数组要比分开逐步计算各个小型数组速度更快
'''
# 模拟批处理实现
x ,t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])  # 比较运算符（==）生成由True/False构成的布尔型数组，并计算True的个数

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# TypeError: only integer scalar arrays can be converted to a scalar index
'''
这是因为最新版本的python、numpy的问题。版本升级，有些方法已经发生改变，使将单个元素数组作为标量进行索引成为一个错误
'''

