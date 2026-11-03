"""
泛化诊断和学习曲线绘制
泛化诊断：容量诊断 -> 神经元数量对照
          样本量诊断 -> 训练样本数量对照
"""
import numpy as np
from network_core import train_network, loss_batch
from iris_data import load_iris_raw_split
import matplotlib.pyplot as plt

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

def run_experiment(X_train_raw, X_val_raw, y_train, y_val, hidden_units):
    """当前训练数据拟合标准化 -> 初始化 -> 训练 -> 评价"""

    learning_rate = 0.1
    steps = 1000
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    W1, b1, W2, b2 = initialize_parameters(X_train.shape[1], hidden_units)
    W1_train, b1_train, W2_train, b2_train,loss_history,p_history = train_network(X_train, y_train, W1, b1, W2, b2,learning_rate=learning_rate, steps=steps)
    train_loss, train_p = loss_batch(X_train, y_train, W1_train, b1_train, W2_train, b2_train)
    val_loss, val_p = loss_batch(X_val, y_val, W1_train, b1_train, W2_train, b2_train)
    return train_loss, val_loss, loss_history

if __name__ == "__main__":
    X_train_raw, X_val_raw, y_train, y_val = load_iris_raw_split()

    result_4 = run_experiment(X_train_raw, X_val_raw, y_train, y_val, hidden_units=4)
    result_8 = run_experiment(X_train_raw, X_val_raw, y_train, y_val, hidden_units=8)

    print("========== 容量对照 ==========")

    print(f"{'指标':<16}{'4个神经元':>15}{'8个神经元':>15}")

    results = [
        ("训练集交叉熵", result_4[0], result_8[0]),
        ("验证集交叉熵", result_4[1], result_8[1]),
    ]

    for name, value_4, value_8 in results:
        print(f"{name:<16}{value_4:>15.4f}{value_8:>15.4f}")

    """样本量对照"""

    print("\n========== 样本量对照 ==========")

    # 固定随机种子，生成可复现的随机顺序
    rng = np.random.default_rng(42)
    indices = rng.permutation(X_train_raw.shape[0])

    # 不同训练样本量
    sample_sizes = [25, 50, 75]

    print(f"{'训练样本数':<16}{'训练集交叉熵':>15}{'验证集交叉熵':>15}")

    # 记录训练集和验证集的损失
    train_losses = []
    val_losses = []

    for sample_size in sample_sizes:

        # 随机抽取指定数量的训练样本
        selected_indices = indices[:sample_size]

        X_subset = X_train_raw[selected_indices]
        y_subset = y_train[selected_indices]

        # 固定隐藏层宽度为4
        train_loss, val_loss, loss_history = run_experiment(
            X_subset,
            X_val_raw,
            y_subset,
            y_val,
            hidden_units=4
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"{sample_size:<16}"
            f"{train_loss:>15.4f}"
            f"{val_loss:>15.4f}"
        )

    # 绘制学习曲线
    plt.plot(sample_sizes, train_losses, marker='o', label='Train')
    plt.plot(sample_sizes, val_losses, marker='o', label='Validation')
    plt.title('Learning Curves')
    plt.xlabel('Sample Size')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    """语法：
    plot(x, y, marker='o', label='Train')  # 绘制训练集损失曲线
    # 设置标题、坐标轴标签和图例，marker='o' 表示数据点标记为圆圈，label 用于图例显示
    plt.xlabel('Sample Size')  # 设置 x 轴标签
    plt.ylabel('Loss')  # 设置 y 轴标签
    legend()  # 显示图例
    show()  # 显示图形
    """
