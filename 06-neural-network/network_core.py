"""Reusable functions from the two-layer sigmoid network experiment.

Samples are rows; weight columns correspond to neurons.
Gradient functions return results; callers control display and training.
"""

import numpy as np


def sigmoid(z):
    """把线性得分转换为 sigmoid 激活值，支持标量和数组。"""
    return 1 / (1 + np.exp(-z))


def dense_forward(inputs, W, b):
    """计算当前层的线性得分 z 和激活值 a。"""
    z = np.dot(inputs, W) + b
    a = sigmoid(z)
    return z, a


def backward_single(x,y,W1,b1,W2,b2):
    """计算单样本的前向传播和反向传播，返回解析梯度"""
    """语法：
        dz2[:,newaxis] -> newaxis 在指定位置增加一个维度，便于广播计算。
        W2.flatten() -> 将 W2 展平为一维数组，便于与 dz2 相乘。
    """
    # dense_forward，得到两层的得分和激活值
    z1, a1 = dense_forward(x, W1, b1)
    z2, a2 = dense_forward(a1, W2, b2)

    dz2 = a2 - y  # 损失对输出得分的梯度
    dW2 = dz2[:, np.newaxis] * a1[:, np.newaxis]  # 输出层权重的梯度
    db2 = dz2  # 输出层偏置的梯度
    # 对每个隐藏层神经元计算梯度
    dz1 = (W2.flatten() * dz2) * a1 * (1 - a1)  # 隐藏层得分的梯度
    dW1 =  x[:, np.newaxis] * dz1[np.newaxis, :] # 隐藏层权重的梯度
    db1 = dz1  # 隐藏层偏置的梯度
    return dW1, db1, dW2, db2


def loss_single(x, y, W1, b1, W2, b2):
    """计算单样本的前向传播和损失"""
    z1, a1 = dense_forward(x, W1, b1)
    z2, a2 = dense_forward(a1, W2, b2)
    p = float(a2[0])
    loss = -np.log(p) if y == 1 else -np.log(1 - p)
    return loss, p
