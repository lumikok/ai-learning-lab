"""Iris 二分类数据处理，沿用 05-iris-classification 的划分与标准化。"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# train_test_split() 用于划分训练集和验证集

def load_iris_raw_split(random_state=42):
    """返回未经标准化的 X_train、X_val、y_train、y_val。

    沿用原实验的两类、四特征及分层 3:1 划分，便于各训练子集
    分别拟合预处理统计量。标签 0 为 versicolor，1 为 virginica。
    """
    iris = load_iris()
    mask = iris.target != 0
    X = iris.data[mask]
    y = iris.target[mask] - 1
    return train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )


def load_iris_split(random_state=42):
    """返回 X_train、X_val、y_train、y_val；标准化统计量只来自训练集。

    标签 0 为 versicolor，标签 1 为 virginica，保留全部四项特征。
    训练/验证比例为 3:1，使用分层随机划分。
    """
    iris = load_iris()
    mask = iris.target != 0
    X = iris.data[mask]
    y = iris.target[mask]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    y_train = np.where(y_train == 1, 0, 1)
    y_val = np.where(y_val == 1, 0, 1)

    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    return X_train, X_val, y_train, y_val
