"""分类指标：用于评估分类模型性能的指标。

P：精确率
R：召回率
F1：F1分数
混淆矩阵：用于可视化分类模型性能的工具，显示了预测类别与实际类别之间的关系。实际就是FN、FP、TN、TP的矩阵表示。
并据此做不同阈值下的分类指标比较，用于评估模型性能和选择符合需求的候选阈值。
"""
import numpy as np
from iris_baseline import load_iris_split, train_logistic, sigmoid

# 定义真实标签和预测概率
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])

def load_iris_validation_predictions():
    """使用已有零正则化基线，返回固定验证标签和预测概率。

    只运行一次已有训练；随后比较阈值时复用返回的同一组概率。
    iris_baseline保存从02入口复制的已有函数。
    """
    X_train, X_val, y_train, y_val = load_iris_split()
    w, b, loss_history = train_logistic(
        X_train, y_train, np.zeros(X_train.shape[1]), 0.0,
        learning_rate=0.1, steps=1000,
    )
    val_probabilities = sigmoid(X_val @ w + b)
    return y_val, val_probabilities


def classification_metrics(y_true, y_prob, threshold=0.5):
    """
    计算分类指标：精确率、召回率、F1分数和混淆矩阵。
    约定：分母为0时，精确率、召回率和F1分数均返回0.0
    """
    # 1. 将预测概率转换为预测标签
    y_pred = (y_prob >= threshold).astype(int)

    # 2. 计算混淆矩阵
    TP = np.sum((y_true == 1) & (y_pred == 1))  # 真阳性
    TN = np.sum((y_true == 0) & (y_pred == 0))  # 真阴性
    FP = np.sum((y_true == 0) & (y_pred == 1))  # 假阳性
    FN = np.sum((y_true == 1) & (y_pred == 0))  # 假阴性

    # 3. 计算精确率、召回率和F1分数
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # 精确率
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 召回率
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0  # F1分数

    # 4. 返回结果
    return {
        "Precision": P,
        "Recall": R,
        "F1 Score": F1,
        "Confusion Matrix": {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        }
    }

# 调用函数并打印结果
iris_y_val, iris_p_val = load_iris_validation_predictions()
metrics = classification_metrics(iris_y_val, iris_p_val, threshold=0.3)
metrics_2 = classification_metrics(iris_y_val, iris_p_val, threshold=0.5)
metrics_3 = classification_metrics(iris_y_val, iris_p_val, threshold=0.7)
print("阈值为0.3时的分类指标：")
for key, value in metrics.items():
    if key == "Confusion Matrix":
        print(f"{key}:")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    else:
        print(f"{key}: {value}")

print("\n阈值为0.5时的指标：")

for key, value in metrics_2.items():
    if key == "Confusion Matrix":
        print(f"{key}:")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    else:
        print(f"{key}: {value}")

print("\n阈值为0.7时的指标：")

for key, value in metrics_3.items():
    if key == "Confusion Matrix":
        print(f"{key}:")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    else:
        print(f"{key}: {value}")
