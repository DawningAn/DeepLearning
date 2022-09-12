# -*- coding = utf-8 -*-
# @Time : 2022/4/17 21:56
# @Author : Juyi
# @File : test01.py
# @Software : PyCharm
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# print(sklearn.__file__)  查看软件包的存储位置

def datasets_demo():
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)  # 形如字典
    print("查看数据集描述：\n", iris["DESCR"])
    print("查看特征值的名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape)  # 最后结果(150, 4)表示有150个样本，4个特征

    # 数据集划分
    # 传入特征值 和 目标值
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    # test_size=0.2表示测试集范围为0.2，即20%;random_state=22表示设置随机数种子
    print("训练集的特征值：\n", x_train, x_train.shape)  # 二维数组 120行4列
    # x_train.shape 查看多少行多少列
    return None


if __name__ == "__main__":
    datasets_demo()
