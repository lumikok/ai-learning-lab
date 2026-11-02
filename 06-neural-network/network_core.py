"""
神经网络核心模块，包含前向传播、反向传播和训练函数。
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


def backward_batch(X,y,W1,b1,W2,b2):
    """
    逐样本计算梯度，累加后返回平均梯度。
    所有样本使用同一组原始参数。
    """
    # 初始化梯度累加器
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)

    # 遍历每个样本
    for i in range(X.shape[0]):
        # 计算单个样本的梯度
        dW1_i, db1_i, dW2_i, db2_i = backward_single(X[i], y[i], W1, b1, W2, b2)
        # 累加梯度
        dW1 += dW1_i
        db1 += db1_i
        dW2 += dW2_i
        db2 += db2_i

    # 计算平均梯度
    m = X.shape[0]
    dW1 /= m
    db1 /= m
    dW2 /= m
    db2 /= m

    return dW1, db1, dW2, db2


def loss_batch(X, y, W1, b1, W2, b2):
    """
    逐样本计算损失，返回平均损失和预测概率。
    """
    # 初始化总损失
    total_loss = 0.0
    p = []

    # 遍历每个样本
    for i in range(X.shape[0]):
        # 计算单个样本的损失
        loss_i,p_i = loss_single(X[i], y[i], W1, b1, W2, b2)
        # 累加损失
        total_loss += loss_i
        # 累加预测概率
        p.append(p_i)

    # 计算平均损失
    mean_loss = total_loss / X.shape[0]
    p = np.array(p)  # 将预测概率转换为 NumPy 数组
    return mean_loss,p


def train_network(X, y, W1, b1, W2, b2, learning_rate=0.1, steps=100):
    """
    使用批量梯度下降训练神经网络。
    """
    W1_train = W1.copy()
    b1_train = b1.copy()
    W2_train = W2.copy()
    b2_train = b2.copy()

    loss_before, p_before = loss_batch(X, y, W1_train, b1_train, W2_train, b2_train)
    loss_history, p_history = [loss_before], [p_before]
    for step in range(steps):
        # 计算平均梯度
        dW1, db1, dW2, db2 = backward_batch(X, y, W1_train, b1_train, W2_train, b2_train)
        # 更新参数
        W1_train -= learning_rate * dW1
        b1_train -= learning_rate * db1
        W2_train -= learning_rate * dW2
        b2_train -= learning_rate * db2
        loss_after, p_after = loss_batch(X, y, W1_train, b1_train, W2_train, b2_train)
        loss_history.append(loss_after)
        p_history.append(p_after)

    return W1_train, b1_train, W2_train, b2_train, loss_history, p_history
