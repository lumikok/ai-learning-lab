"""已有Iris逻辑回归基线。

以下计算与数据处理复制自02-l2-regularization.py，保留原实验。
该模块仅定义函数，供分类指标实验通过普通import复用。

分别包括：
- sigmoid函数
- 二元交叉熵损失函数
- 逻辑回归梯度计算函数
- 逻辑回归训练函数
- Iris数据集加载与划分函数
补充：不包括L2正则化
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def gradients_logistic(X, y, w, b):
    predictions = sigmoid(X @ w + b)
    dw = X.T @ (predictions - y) / len(y)
    db = np.sum(predictions - y) / len(y)
    return dw, db


def train_logistic(X, y, w, b, learning_rate, steps):
    history = binary_cross_entropy(y, sigmoid(X @ w + b))
    w = w.copy()
    for step in range(steps):
        dw, db = gradients_logistic(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        current_loss = binary_cross_entropy(y, sigmoid(X @ w + b))
        history = np.append(history, current_loss)
    return w, b, history


def load_iris_split(random_state=42):
    """四特征、75/25分层划分；标准化统计量只来自训练集。

    返回X_train、X_val、y_train、y_val；0为versicolor，1为virginica。
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
