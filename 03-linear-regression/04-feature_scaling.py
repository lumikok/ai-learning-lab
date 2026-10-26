# 特征缩放实验
import numpy as np

X_train = np.array([
    [-100,  0],
    [ 100,  0],
    [   0, -1],
    [   0,  1],
    [ 100,  1],
], dtype=float)

y_train = np.array([4, 0, -1, 5, 3], dtype=float)

def cost_multi(X,y,w,b):
    m = len(y)
    predictions = X @ w + b
    e = predictions - y
    return np.sum(e ** 2) / (2 * m)

def gradients_multi(X,y,w,b):
    m = len(y)
    predictions = X @ w + b
    e = predictions - y
    dw = (X.T @ e) / m
    db = np.sum(e) / m
    return dw,db

def train_multi(X,y,w,b,learning_rate,steps):
    # 训练过程中记录损失值
    history = [cost_multi(X,y,w,b)]
    for i in range(steps):
        dw,db = gradients_multi(X,y,w,b)
        w -= learning_rate * dw
        b -= learning_rate * db
        history.append(cost_multi(X,y,w,b))
    return w,b,history

# 标准化
def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_scaled = (X - mu) / sigma
    return X_scaled, mu, sigma

# 一组：标准化后的特征
X_scaled, mu, sigma = standardize(X_train)

print("训练集均值 mu:", mu)
print("训练集标准差 sigma:", sigma)

w1 = np.zeros(X_train.shape[1])
b1 = 0.0
w2 = np.zeros(X_scaled.shape[1])
b2 = 0.0
learning_rate = 0.1

steps = 40

w_raw,b_raw,history_raw = train_multi(X_train, y_train, w1, b1, learning_rate, steps)
w_scaled,b_scaled,history_scaled = train_multi(X_scaled, y_train, w2, b2, learning_rate, steps)

# 检查损失记录数
print("原始特征训练损失记录数:", len(history_raw))
print("标准化特征训练损失记录数:", len(history_scaled))

# 打印指定步数的损失
check_steps = [0, 1, 2, 5, 10, 20, 40]

for step in check_steps:
    if step < len(history_raw):
        print(f"原始特征训练步数 {step} 的损失: {history_raw[step]}")
    if step < len(history_scaled):
        print(f"标准化特征训练步数 {step} 的损失: {history_scaled[step]}")

# 查找首次达到 J < 0.01 的步数
def first_below_threshold(history, threshold=0.01):
    for step, loss in enumerate(history):
        if loss < threshold:
            return step
    return None

for name,history in [
    ("原始输入", history_raw),
    ("标准化输入", history_scaled)
]:
    step = first_below_threshold(history)
    if step is not None:
        print(f"{name}首次达到 J < 0.01 的步数: {step}")
    else:
        print(f"{name}未达到 ")

# 对新样本使用训练集的均值和标准差进行变换
X_new = np.array([[200, -1]], dtype=float)
X_new_scaled = (X_new - mu) / sigma

# 使用标准化输入训练出的模型预测
prediction = X_new_scaled @ w_scaled + b_scaled

print("\n新样本:", X_new)
print("新样本标准化后:", X_new_scaled)
print("标准化模型预测值:", prediction[0])
