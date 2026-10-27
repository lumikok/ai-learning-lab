# 分类训练，检查是否能够降低交叉熵
import numpy as np

X = np.array([[-2], [-0.5], [0.2], [0.5], [2]], dtype=float)
y = np.array([0, 1, 0, 1, 1])

w = np.zeros(1)
b = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 计算二元交叉熵损失
def binary_cross_entropy(y_true, y_pred):
    # 避免对数零的情况
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 逻辑回归的梯度计算
def gradients_logistic(X,y,w,b):
    predictions = sigmoid(X @ w + b)
    dw = X.T @ (predictions - y) / len(y)
    db = np.sum(predictions - y) / len(y)
    return dw, db

# 逻辑回归的训练
def train_logistic(X,y,w,b,learning_rate,steps):
    # 计算初始交叉熵
    history = binary_cross_entropy(y, sigmoid(X @ w + b))
    for step in range(steps):
        dw, db = gradients_logistic(X,y,w,b)
        w -= learning_rate * dw
        b -= learning_rate * db
        # 计算当前交叉熵
        current_loss = binary_cross_entropy(y, sigmoid(X @ w + b))
        history = np.append(history, current_loss)
    return w, b, history

# 初始错误数量
start_error_count = (np.sum((sigmoid(X @ w + b) >= 0.5) != y))

# 进行训练
learning_rate = 0.1
steps = 1000
w1,b1,history1 = train_logistic(X,y,w,b,learning_rate,1)

print("一步更新参数:", w1, b1)

w = np.zeros(1)
b = 0.0

# 输出训练结果
print("初始梯度:", gradients_logistic(X,y,w,b))

w, b, history = train_logistic(X,y,w,b,learning_rate,steps)
print("最终参数:", w, b)
print("初始损失:", history[0])
print("最终损失:", history[-1])

# 阈值
threshold = 0.5

error_count = (np.sum((sigmoid(X @ w + b) >= threshold) != y))
print("错误数量:", error_count)
print("初始错误数量:", start_error_count)

fp_count = np.sum((sigmoid(X @ w + b) >= threshold) & (y == 0))
fn_count = np.sum((sigmoid(X @ w + b) < threshold) & (y == 1))
print("假阳性数量:", fp_count)
print("假阴性数量:", fn_count)

# 选例验证
# 阈值固定为0.5，边界满足w*x+b=0
decision_boundary = -b / w[0]
print("当前决策边界:", decision_boundary)

new_samples = np.array([[-1],[3],[-0.4069]], dtype=float)
print("新样本预测结果:", sigmoid(new_samples @ w + b))
print("新样本预测类别:", (sigmoid(new_samples @ w + b) >= threshold).astype(int))

# 一般化阈值
def boundary_x(weight, bias, threshold):
    return (np.log(threshold/(1-threshold)) - bias) / weight

decision_boundary_05 = boundary_x(w[0], b, 0.5)
decision_boundary_07 = boundary_x(w[0], b, 0.7)

print("阈值为0.5时的边界:", decision_boundary_05)
print("阈值为0.7时的边界:", decision_boundary_07)
new_samples2 = np.array([[decision_boundary_05-0.01],[decision_boundary_05+0.01]], dtype=float)
new_samples3 = np.array([[decision_boundary_07-0.01],[decision_boundary_07+0.01]], dtype=float)

print("阈值为0.5时的边界附近样本预测结果:", sigmoid(new_samples2 @ w + b))
print("阈值为0.5时的边界附近样本预测类别:", (sigmoid(new_samples2 @ w + b) >= 0.5).astype(int))
print("阈值为0.7时的边界附近样本预测结果:", sigmoid(new_samples3 @ w + b))
print("阈值为0.7时的边界附近样本预测类别:", (sigmoid(new_samples3 @ w + b) >= 0.7).astype(int))
