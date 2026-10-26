import numpy as np

X = np.array([[-2], [-0.5], [0.2], [0.5], [2]], dtype=float)
y = np.array([0, 1, 0, 1, 1])

w = np.array([1.0])
b = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = X @ w + b
predictions = sigmoid(z)
print("预测值概率:", predictions)

# 阈值
threshold1 = 0.5
threshold2 = 0.7

# 根据阈值进行分类
predictions_class1 = (predictions >= threshold1).astype(int)
predictions_class2 = (predictions >= threshold2).astype(int)
# 语法糖：使用astype(int)将布尔值转换为整数类型（0或1），以便表示分类结果。

print(f"阈值 {threshold1} 的分类结果:", predictions_class1)
print(f"阈值 {threshold2} 的分类结果:", predictions_class2)
print("真实标签:", y)

def binary_cross_entropy(y_true, y_pred):
    # 避免对数零的情况
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

loss1 = binary_cross_entropy(y, predictions)
print(f"损失：", loss1)

# 对两组预测标签分别统计
# 错误总数
error_count1 = np.sum(predictions_class1 != y)
error_count2 = np.sum(predictions_class2 != y)
print(f"阈值 {threshold1} 的错误总数:", error_count1)
print(f"阈值 {threshold2} 的错误总数:", error_count2)
# FP,FN
predictions_class1_fp = np.sum((predictions_class1 == 1) & (y == 0))
predictions_class1_fn = np.sum((predictions_class1 == 0) & (y == 1))
predictions_class2_fp = np.sum((predictions_class2 == 1) & (y == 0))
predictions_class2_fn = np.sum((predictions_class2 == 0) & (y == 1))
print(f"阈值 {threshold1} 的假阳性数 (FP):", predictions_class1_fp)
print(f"阈值 {threshold1} 的假阴性数 (FN):", predictions_class1_fn)
print(f"阈值 {threshold2} 的假阳性数 (FP):", predictions_class2_fp)
print(f"阈值 {threshold2} 的假阴性数 (FN):", predictions_class2_fn)
