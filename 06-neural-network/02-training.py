"""
多样本训练：批量梯度下降
"""
import numpy as np
from network_core import backward_single, loss_single

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

def main():
    print("批量梯度下降\n")
    X = np.array([[1.0, 2.0], [0.0, 0.0], [2.0, -1.0]])
    y = np.array([1.0, 0.0, 1.0])

    W1 = np.array([[0.5, -1.0], [-0.25, 0.0]])
    b1 = np.array([0.2, 0.0])
    W2 = np.array([[1.0], [-2.0]])
    b2 = np.array([0.0])
    learning_rate = 0.1
    steps = 100
    # 更新前的平均损失和预测概率
    loss_before, p_before = loss_batch(X, y, W1, b1, W2, b2)
    print("\n更新前")
    print(f"平均损失: {loss_before:.8f}")
    print("预测概率:", p_before)
    print("预测概率形状:", p_before.shape)

    # 训练网络
    W1_updated, b1_updated, W2_updated, b2_updated, loss_history, p_history = train_network(X, y, W1, b1, W2, b2, learning_rate=learning_rate, steps=steps)
    print("\n更新后")
    print(f"平均损失: {loss_history[-1]:.8f}")
    print("预测概率:", p_history[-1])
    print("预测概率形状:", p_history[-1].shape)
    print("训练步数:", steps)
    print("损失记录条数:", len(loss_history))
    print("记录条数是否等于步数加一:", len(loss_history) == steps + 1)
    print("所有损失是否为有限数:", np.isfinite(loss_history).all())

    # 用函数实际返回的参数重新评价，核对返回模型和训练记录一致。
    returned_loss, returned_p = loss_batch(
        X, y, W1_updated, b1_updated, W2_updated, b2_updated
    )
    print("返回参数重新计算的平均损失:", returned_loss)
    print("返回参数重新计算的预测概率:", returned_p)
    print("返回模型损失是否与最后记录一致:", np.isclose(returned_loss, loss_history[-1]))
    print("返回模型概率是否与最后记录一致:", np.allclose(returned_p, p_history[-1]))

    # 比较
    print("\n比较")
    print(f"更新前平均损失: {loss_before:.8f}")
    print(f"更新后平均损失: {loss_history[-1]:.8f}")
    print(f"损失变化量: {loss_history[-1] - loss_before:.8f}")

    if loss_history[-1] < loss_before:
        print("结果：平均损失下降，参数更新有效。")
    else:
        print("结果：平均损失没有下降，需要检查。")

if __name__ == "__main__":
    main()
