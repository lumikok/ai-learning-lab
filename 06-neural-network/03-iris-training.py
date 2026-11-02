"""神经网络训练实验：
使用iris数据集训练一个简单的神经网络，并输出训练过程和评价结果。
"""

import numpy as np
from iris_data import load_iris_split
from network_core import train_network, loss_batch

def initialize_parameters(n_features, hidden_units, seed=42):
    """随机初始化网络参数
        输入特征的数量: n_features
        隐藏层单元的数量: hidden_units
        随机数种子: seed
    """

    rng = np.random.default_rng(seed)
    W1 = rng.normal(
        loc=0.0, scale=0.1,
        size=(n_features, hidden_units)
    )
    b1 = np.zeros((hidden_units,))
    W2 = rng.normal(
        loc=0.0, scale=0.1,
        size=(hidden_units, 1)
    )
    b2 = np.zeros((1,))

    return W1, b1, W2, b2

def demo_iris_training():
    """iris数据集训练"""

    # 1. 加载iris数据集
    X_train, X_val, y_train, y_val = load_iris_split()

    # 2. 初始化网络参数
    n_features = X_train.shape[1]
    hidden_units = 4
    W1, b1, W2, b2 = initialize_parameters(n_features, hidden_units)

    # 3. 训练网络
    learning_rate = 0.1
    steps = 1000
    W1_updated, b1_updated, W2_updated, b2_updated, loss_history, p_history = train_network(
        X_train, y_train, W1, b1, W2, b2,
        learning_rate=learning_rate,
        steps=steps
    )

    # 4. 检查训练过程
    print("\n========== 训练过程检查 ==========")
    print("损失记录数量:", len(loss_history))
    print("损失是否全部有限:", np.all(np.isfinite(loss_history)))

    # 5. 分别评价训练集和验证集
    train_loss, train_p = loss_batch(X_train, y_train, W1_updated, b1_updated, W2_updated, b2_updated)
    val_loss, val_p = loss_batch(X_val, y_val, W1_updated, b1_updated, W2_updated, b2_updated)

    # 6. 检查重新计算的训练损失
    print("训练损失是否和历史记录一致：",
          np.isclose(train_loss, loss_history[-1])
    )

    # 7. 计算错误率、FP、FN
    threshold = 0.5
    train_pred = (train_p >= threshold).astype(int)
    val_pred = (val_p >= threshold).astype(int)

    train_error_rate = np.mean(train_pred != y_train)
    val_error_rate = np.mean(val_pred != y_val)

    train_FP = np.sum((train_pred == 1) & (y_train == 0))
    train_FN = np.sum((train_pred == 0) & (y_train == 1))

    val_FP = np.sum((val_pred == 1) & (y_val == 0))
    val_FN = np.sum((val_pred == 0) & (y_val == 1))

    # 8. 并列输出评价结果
    print("\n========== 模型评价 ==========")

    print(f"{'指标':<16}{'训练集':>12}{'验证集':>12}")
    print("-" * 40)

    results = [
        ("交叉熵", train_loss, val_loss),
        ("错误率", train_error_rate, val_error_rate),
        ("FP", train_FP, val_FP),
        ("FN", train_FN, val_FN),
    ]

    for name, train_value, val_value in results:
        if name in ("FP", "FN"):
            print(f"{name:<16}{train_value:>12}{val_value:>12}")
        else:
            print(
                f"{name:<16}"
                f"{train_value:>12.4f}"
                f"{val_value:>12.4f}"
            )

    print("\n训练集预测概率形状:", train_p.shape)
    print("验证集预测概率形状:", val_p.shape)

    return W1_updated, b1_updated, W2_updated, b2_updated, loss_history

if __name__ == "__main__":
    demo_iris_training()