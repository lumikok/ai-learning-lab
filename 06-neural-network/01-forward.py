"""两层神经网络：前向传播、非线性与单个权重的梯度检查。

命名约定（样本按行存放）：
- W1、b1：第1层（隐藏层）的权重、偏置；W2、b2：第2层（输出层）。
- z1、a1：单样本第1层的线性得分、激活值；z2、a2：单样本第2层。
- Z1、A1、Z2、A2：批量样本对应的得分、激活值。
- 数学 a^[1]_2 对应单样本 a1[1]，即隐藏层第2个神经元的激活值。
- q 是本次选中检查的单个权重 W1[1, 1]；grad_q 是损失对 q 的梯度。
- W 的每一列是一名神经元的权重，形状为（输入数，本层神经元数）。
"""

import numpy as np

from network_core import sigmoid, dense_forward, backward_single, loss_single


def demo_single_and_batch(x, X, W1, b1, W2, b2):
    """同一个网络分别处理单样本和批量输入。
    结果应该一致，但形状不同。
    """
    z1, a1 = dense_forward(x, W1, b1)
    z2, a2 = dense_forward(a1, W2, b2)
    print("\n单样本前向传播")
    print("隐藏层 z1、a1：", z1, a1)
    print("隐藏层 z1、a1 的形状：", z1.shape, a1.shape)
    print("输出层 z2、a2：", z2, a2)
    print("输出层 z2、a2 的形状：", z2.shape, a2.shape)

    Z1, A1 = dense_forward(X, W1, b1)
    Z2, A2 = dense_forward(A1, W2, b2)
    print("\n批量前向传播")
    print("隐藏层 Z1、A1：", Z1, A1)
    print("隐藏层 Z1、A1 的形状：", Z1.shape, A1.shape)
    print("输出层 Z2、A2：", Z2, A2)
    print("输出层 Z2、A2 的形状：", Z2.shape, A2.shape)
    print("批量第一条与单样本结果一致：", np.allclose(A2[0], a2))
    return a2, A2


def demo_hidden_width(x):
    """隐藏层扩大为三个神经元，参数只在本实验内使用。
        网络宽度变化后，参数和输出应该怎么变化？
        -> 只要输入和输出层不变，隐藏层的宽度变化不会影响最终输出的形状。
        -> 但是隐藏层的激活值和线性得分会随宽度变化而变化。
    """
    W1 = np.array([
        [0.5, -1.0, 0.3],
        [-0.25, 0.0, 0.8],
    ])
    b1 = np.array([0.2, 0.0, -0.1])
    W2 = np.array([[1.0], [-2.0], [0.5]])
    b2 = np.array([0.0])

    z1, a1 = dense_forward(x, W1, b1)
    z2, a2 = dense_forward(a1, W2, b2)
    print("\n三隐藏神经元变式")
    print("隐藏层 z1、a1 的形状：", z1.shape, a1.shape)
    print("输出层 z2、a2 的形状：", z2.shape, a2.shape)
    print("最终概率：", a2)
    print("预测类别：", int(a2[0] >= 0.5))
    return a2


def demo_linear_equivalence(X, W1, b1, W2, b2):
    """比较去掉隐藏层激活后的两层计算和合并参数计算进行比较。"""
    Z1, _ = dense_forward(X, W1, b1)
    Z2_linear, A2_linear = dense_forward(Z1, W2, b2)

    W_merged = W1 @ W2
    b_merged = b1 @ W2 + b2
    Z_merged = X @ W_merged + b_merged
    A_merged = sigmoid(Z_merged)

    print("\n去掉隐藏层激活后的合并检查")
    print("两层计算的得分和概率：", Z2_linear, A2_linear)
    print("合并参数 W_merged、b_merged：", W_merged, b_merged)
    print("合并计算的得分和概率：", Z_merged, A_merged)
    print("两种计算结果一致：", np.allclose(A_merged, A2_linear))
    return A2_linear, A_merged


def loss_with_q(q, x, W1, b1, W2, b2):
    """只替换 W1[1, 1]，返回单样本（真实标签 y=1）的损失和概率。"""
    W1_trial = W1.copy()
    W1_trial[1, 1] = q
    z1, a1 = dense_forward(x, W1_trial, b1)
    z2, a2 = dense_forward(a1, W2, b2)
    p = float(a2[0])
    loss = -np.log(p)
    return loss, p


def demo_q_gradient(x, W1, b1, W2, b2):
    """先观察 q 扰动，再核对解析梯度、数值梯度和一次更新。"""
    y = 1.0
    q_initial = float(W1[1, 1])
    print("\n改变单个权重 q = W1[1, 1]")
    trial_results = []
    for q_trial in (-0.01, 0.0, 0.01):
        loss_trial, p_trial = loss_with_q(q_trial, x, W1, b1, W2, b2)
        trial_results.append([float(loss_trial), p_trial])
        print(f"q={q_trial:+.2f}，概率={p_trial:.9f}，损失={loss_trial:.9f}")

    loss_before, p_before = loss_with_q(q_initial, x, W1, b1, W2, b2)
    z1, a1 = dense_forward(x, W1, b1)

    # 链式法则计算解析梯度 grad_q：
    # q -> z1[1] -> a1[1] -> z2[0] -> 损失
    # 各局部因子：x[1]、a1[1]*(1-a1[1])、W2[1,0]、p_before-y。
    # a1[1] 是第1层第2个神经元的激活值；a2 是第2层输出。
    grad_q = (p_before - y) * W2[1, 0] * a1[1] * (1 - a1[1]) * x[1]

    epsilon = 1e-5
    loss_plus, _ = loss_with_q(q_initial + epsilon, x, W1, b1, W2, b2)
    loss_minus, _ = loss_with_q(q_initial - epsilon, x, W1, b1, W2, b2)
    numerical_grad_q = (loss_plus - loss_minus) / (2 * epsilon)
    gradient_error = abs(numerical_grad_q - grad_q)
    print("解析梯度 grad_q：", grad_q)
    print("数值梯度 numerical_grad_q：", numerical_grad_q)
    print("梯度绝对差 gradient_error：", gradient_error)

    # 一次梯度下降更新 q，观察损失是否减小。
    learning_rate = 0.1
    q_updated = q_initial - learning_rate * grad_q
    loss_after, p_after = loss_with_q(q_updated, x, W1, b1, W2, b2)
    print("原权重 q_initial：", q_initial)
    print("更新后权重 q_updated：", q_updated)
    print("更新前后概率：", p_before, p_after)
    print("更新前后损失：", loss_before, loss_after)
    print("更新后的损失减小：", loss_after < loss_before)
    return {
        "q_trials": trial_results,
        "analytic_grad": float(grad_q),
        "numerical_grad": float(numerical_grad_q),
        "q_updated": float(q_updated),
        "loss_after": float(loss_after),
        "p_after": p_after,
    }


def demo_loss_single(x, y, W1, b1, W2, b2):
    """检查四个参数的数值梯度和解析梯度
        用于检查反向传播的正确性
    """

    epsilon = 1e-5

    # 计算解析梯度
    dW1, db1, dW2, db2 = backward_single(
        x, y, W1, b1, W2, b2
    )

    for name, gradient in zip(("dW1", "db1", "dW2", "db2"), (dW1, db1, dW2, db2)):
        print(f"{name}, shape={gradient.shape}:", gradient)

    # 创建参数副本
    W1_plus, W1_minus = W1.copy(), W1.copy()
    b1_plus, b1_minus = b1.copy(), b1.copy()
    W2_plus, W2_minus = W2.copy(), W2.copy()
    b2_plus, b2_minus = b2.copy(), b2.copy()

    # 分别扰动四个参数
    W1_plus[0, 0] += epsilon
    W1_minus[0, 0] -= epsilon

    b1_plus[1] += epsilon
    b1_minus[1] -= epsilon

    W2_plus[0, 0] += epsilon
    W2_minus[0, 0] -= epsilon

    b2_plus[0] += epsilon
    b2_minus[0] -= epsilon

    # 计算 W1[0,0] 数值梯度
    lp, _ = loss_single(x, y, W1_plus, b1, W2, b2)
    lm, _ = loss_single(x, y, W1_minus, b1, W2, b2)
    numeric_W1 = (lp - lm) / (2 * epsilon)

    # 计算 b1[1] 数值梯度
    lp, _ = loss_single(x, y, W1, b1_plus, W2, b2)
    lm, _ = loss_single(x, y, W1, b1_minus, W2, b2)
    numeric_b1 = (lp - lm) / (2 * epsilon)

    # 计算 W2[0,0] 数值梯度
    lp, _ = loss_single(x, y, W1, b1, W2_plus, b2)
    lm, _ = loss_single(x, y, W1, b1, W2_minus, b2)
    numeric_W2 = (lp - lm) / (2 * epsilon)

    # 计算 b2[0] 数值梯度
    lp, _ = loss_single(x, y, W1, b1, W2, b2_plus)
    lm, _ = loss_single(x, y, W1, b1, W2, b2_minus)
    numeric_b2 = (lp - lm) / (2 * epsilon)

    # 比较结果
    results = [
        ("W1[0,0]", dW1[0, 0], numeric_W1),
        ("b1[1]", db1[1], numeric_b1),
        ("W2[0,0]", dW2[0, 0], numeric_W2),
        ("b2[0]", db2[0], numeric_b2),
    ]

    print("\n数值梯度检查")

    for name, analytic, numeric in results:
        diff = abs(analytic - numeric)
        status = "通过" if diff < 1e-6 else "未通过"

        print(
            f"{name:<10} "
            f"解析梯度: {analytic: .8f}  "
            f"数值梯度: {numeric: .8f}  "
            f"绝对差: {diff:.10f}  "
            f"{status}"
        )


def demo_update_single(x, y, W1, b1, W2, b2, learning_rate=0.1):
    """演示单样本的梯度下降更新"""
    dW1, db1, dW2, db2 = backward_single(x, y, W1, b1, W2, b2)

    # 更新参数
    W1_updated = W1 - learning_rate * dW1
    b1_updated = b1 - learning_rate * db1
    W2_updated = W2 - learning_rate * dW2
    b2_updated = b2 - learning_rate * db2

    # 计算更新前后的损失
    loss_before, p_before = loss_single(x, y, W1, b1, W2, b2)
    loss_after, p_after = loss_single(x, y, W1_updated, b1_updated, W2_updated, b2_updated)

    print("\n单样本梯度下降更新")
    print("更新前损失: {:.8f}, 概率: {:.8f}".format(loss_before, p_before))
    print("更新后损失: {:.8f}, 概率: {:.8f}".format(loss_after, p_after))
    print("损失是否减小: {}".format(loss_after < loss_before))
    return W1_updated, b1_updated, W2_updated, b2_updated


def main():
    # 输入层为第0层：单样本 x，批量 X。各实验中的临时量使用局部作用域。
    x = np.array([1.0, 2.0])
    X = np.array([
        [1.0, 2.0],
        [0.0, 0.0],
        [2.0, -1.0],
    ])
    # 第1层：隐藏层两个神经元，每列为一个神经元的权重。
    W1 = np.array([[0.5, -1.0], [-0.25, 0.0]])
    b1 = np.array([0.2, 0.0])
    # 第2层：输出层一个神经元。
    W2 = np.array([[1.0], [-2.0]])
    b2 = np.array([0.0])

    demo_single_and_batch(x, X, W1, b1, W2, b2)
    demo_hidden_width(x)
    demo_linear_equivalence(X, W1, b1, W2, b2)
    demo_q_gradient(x, W1, b1, W2, b2)
    demo_loss_single(x, 1.0, W1, b1, W2, b2)
    demo_update_single(x, 1.0, W1, b1, W2, b2)


if __name__ == "__main__":
    main()
