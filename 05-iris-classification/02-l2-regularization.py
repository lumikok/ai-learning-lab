"""
L2正则化-逻辑回归-iris数据集实验-看L2正则化对训练过程和结果的影响
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# sigmoid函数：把线性得分映射到0-1之间
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 二分类交叉熵损失函数
def binary_cross_entropy(y_true, y_pred):
    # 避免对数零的情况
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 计算逻辑回归的梯度
def gradients_logistic(X,y,w,b):
    predictions = sigmoid(X @ w + b)
    dw = X.T @ (predictions - y) / len(y)
    db = np.sum(predictions - y) / len(y)
    return dw, db

# 计算纯逻辑回归的损失和附带L2正则化后的损失
def regularized_loss(X, y, w, b, l2_lambda):
    predictions  = sigmoid(X @ w + b)
    data_loss = binary_cross_entropy(y, predictions)
    l2_loss = (l2_lambda / (2 * len(y))) * np.sum(w ** 2)
    total_loss = data_loss + l2_loss
    return data_loss, total_loss

# 计算带L2正则化的梯度
def gradients_logistic_l2(X, y, w, b, l2_lambda):
    # 计算原始梯度
    dw_data, db = gradients_logistic(X, y, w, b)
    dw_l2 = l2_lambda/len(y) * w
    dw = dw_data + dw_l2
    db = db  # 偏置项不受L2正则化影响
    return dw, db

# L2正则化练习
def train_logistic_l2(X, y, w, b, learning_rate, steps, l2_lambda):
    original_data_loss, original_total_loss = regularized_loss(X, y, w, b, l2_lambda)
    data_history, total_history = np.array([]), np.array([])
    data_history = np.append(data_history, original_data_loss)
    total_history = np.append(total_history, original_total_loss)
    w = w.copy()
    for step in range(steps):
        w_original, b_original = w.copy(), b
        dw, db = gradients_logistic_l2(X, y, w, b, l2_lambda)
        w -= learning_rate * dw
        b -= learning_rate * db
        data_loss, total_loss = regularized_loss(X, y, w, b, l2_lambda)
        data_history = np.append(data_history, data_loss)
        total_history = np.append(total_history, total_loss)
    return w, b, data_history, total_history

# 训练逻辑回归模型（复制自main.py，用于对照）
def train_logistic(X,y,w,b,learning_rate,steps):
    # 计算初始交叉熵
    history = binary_cross_entropy(y, sigmoid(X @ w + b))
    w = w.copy()
    for step in range(steps):
        dw, db = gradients_logistic(X,y,w,b)
        w -= learning_rate * dw
        b -= learning_rate * db
        # 计算当前交叉熵
        current_loss = binary_cross_entropy(y, sigmoid(X @ w + b))
        history = np.append(history, current_loss)
    return w, b, history


def load_iris_split(random_state=42):
    """沿用已有Iris实验：四特征、75/25分层划分、训练统计量标准化。

    返回X_train、X_val、y_train、y_val；0为versicolor，1为virginica。
    数据处理复用06-neural-network/iris_data.py的同名函数。
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

def run_iris_l2_experiment(l2_lambda):
    """在Iris数据集上运行带L2正则化的逻辑回归实验"""

    # 1. 加载数据集
    X_train, X_val, y_train, y_val = load_iris_split()

    # 2. 初始化参数，每组实验都从0开始
    n_features = X_train.shape[1]
    w = np.zeros(n_features)
    b = 0.0

    # 3. 固定训练参数
    learning_rate = 0.1
    steps = 1000

    # 4. 训练模型w
    w_trained, b_trained, data_history, total_history = train_logistic_l2(
        X_train, y_train, w, b, learning_rate, steps, l2_lambda
    )

    # 5. 计算训练集数据损失和总目标损失
    train_data_loss, train_total_loss = regularized_loss(X_train, y_train, w_trained, b_trained, l2_lambda)

    # 6. 计算验证集纯交叉熵
    val_predictions = sigmoid(X_val @ w_trained + b_trained)
    val_loss = binary_cross_entropy(y_val, val_predictions)

    # 7. 计算验证集错误率
    val_pred = (val_predictions >= 0.5).astype(int)
    val_error_rate = np.mean(val_pred != y_val)

    # 8. 计算最终权重平方和
    weight_square_sum = np.sum(w_trained ** 2)

    return {
        "lambda": l2_lambda,
        "train_data_loss": train_data_loss,
        "train_total_loss": train_total_loss,
        "val_loss": val_loss,
        "val_error_rate": val_error_rate,
        "weight_square_sum": weight_square_sum,
    }

if __name__ == "__main__":
    X = np.array([[1.0, 2.0], [-1.0, 1.0]])
    y = np.array([1.0, 0.0])
    w = np.array([1.0, -2.0])
    b = 0.5

    # 原始逻辑回归的损失和梯度
    predictions = sigmoid(X @ w + np.float64(b))
    original_loss = binary_cross_entropy(y, predictions)
    dw_original, db_original = gradients_logistic(X, y, w, b)

    # 情况一：L2正则化参数为0
    l2_lambda_0 = 0.0

    data_loss_0, total_loss_0 = regularized_loss(X, y, w, b, l2_lambda_0)

    dw_0, db_0 = gradients_logistic_l2(X, y, w, b, l2_lambda_0)

    print("========== L2 强度为 0 ==========")
    print(f"原始交叉熵: {original_loss:.6f}")
    print(f"正则化总损失: {total_loss_0:.6f}")
    print(f"原始梯度 dw: {dw_original}")
    print(f"正则化梯度 dw: {dw_0}")
    print(f"原始梯度 db: {db_original:.6f}")
    print(f"正则化梯度 db: {db_0:.6f}")

    print("损失是否一致:", np.isclose(original_loss, total_loss_0))
    print("权重梯度是否一致:", np.allclose(dw_original, dw_0))
    print("偏置梯度是否一致:", np.isclose(db_original, db_0))

    # 情况二：L2正则化参数为1
    l2_lambda_1 = 1.0

    data_loss_1, total_loss_1 = regularized_loss(X, y, w, b, l2_lambda_1)

    dw_1, db_1 = gradients_logistic_l2(X, y, w, b, l2_lambda_1)

    # 计算正则化惩罚值
    l2_penalty = total_loss_1 - data_loss_1

    # 新旧梯度差值
    dw_diff = dw_1 - dw_original
    db_diff = db_1 - db_original
    print("\n========== L2 强度为 1 ==========")
    print(f"原始交叉熵: {data_loss_1:.6f}")
    print(f"L2 惩罚值: {l2_penalty:.6f}")
    print(f"正则化总损失: {total_loss_1:.6f}")

    print(f"原始梯度 dw: {dw_original}")
    print(f"正则化梯度 dw: {dw_1}")
    print(f"新旧权重梯度之差: {dw_diff}")

    print(f"原始梯度 db: {db_original:.6f}")
    print(f"正则化梯度 db: {db_1:.6f}")
    print(f"新旧偏置梯度之差: {db_diff:.6f}")

    learning_rate = 0.1

    # 使用正则化梯度更新一次参数
    w_new = w - learning_rate * dw_1
    b_new = b - learning_rate * db_1

    print("\n========== 参数更新 ==========")
    print(f"原始权重: {w}")
    print(f"更新后权重: {w_new}")
    print(f"原始偏置: {b}")
    print(f"更新后偏置: {b_new}")

    # 对照检验L2正则化的训练过程
    steps = 100
    w_l2, b_l2, data_history_l2, total_history_l2 = train_logistic_l2(X, y, w, b, learning_rate, steps, l2_lambda_1)
    w_no_l2, b_no_l2, history_no_l2 = train_logistic(X, y, w, b, learning_rate, steps)
    w_l2_0, b_l2_0, data_history_l2_0, total_history_l2_0 = train_logistic_l2(X, y, w, b, learning_rate, steps, l2_lambda_0)
    print("\n========== 训练对照 ==========")
    print(f"带L2正则化的数据损失历史（不算L2惩罚）有多少条: {len(data_history_l2)}")
    print(f"带L2正则化的总损失历史（包含L2惩罚）有多少条: {len(total_history_l2)}")
    # 用返回的参数重算损失，并与历史记录对照
    final_data_loss_l2, final_total_loss_l2 = regularized_loss(X, y, w_l2, b_l2, l2_lambda_1)
    print(f"最终数据损失（不算L2惩罚），与历史记录是否一致: {np.isclose(final_data_loss_l2, data_history_l2[-1])}")
    print(f"最终总损失（包含L2惩罚），与历史记录是否一致: {np.isclose(final_total_loss_l2, total_history_l2[-1])}")

    # 核对强度0能够复现原训练结果
    print(f"带L2正则化（强度0）的数据损失历史，是否与原训练结果一致: {np.allclose(data_history_l2_0, history_no_l2)}")
    print(f"带L2正则化（强度0）的总损失历史，是否与原训练结果一致: {np.allclose(total_history_l2_0, history_no_l2)}")
    print(f"带L2正则化（强度0）训练后的权重，是否与原训练结果一致: {np.allclose(w_l2_0, w_no_l2)}")
    print(f"带L2正则化（强度0）训练后的偏置，是否与原训练结果一致: {np.allclose(b_l2_0, b_no_l2)}")

    # 在Iris数据集上运行实验
    lambdas = [0.0, 0.01, 0.1, 1.0]
    results = []

    for l2_lambda in lambdas:
        result = run_iris_l2_experiment(l2_lambda)
        results.append(result)

    print("\n========== Iris L2 强度对照 ==========")

    print(
        f"{'Lambda':<10}"
        f"{'训练数据损失':>16}"
        f"{'训练总目标':>16}"
        f"{'验证交叉熵':>16}"
        f"{'验证错误率':>14}"
        f"{'权重平方和':>16}"
    )

    print("-" * 88)

    for result in results:
        print(
            f"{result['lambda']:<10.2f}"
            f"{result['train_data_loss']:>16.6f}"
            f"{result['train_total_loss']:>16.6f}"
            f"{result['val_loss']:>16.6f}"
            f"{result['val_error_rate']:>13.2%}"
            f"{result['weight_square_sum']:>16.6f}"
        )