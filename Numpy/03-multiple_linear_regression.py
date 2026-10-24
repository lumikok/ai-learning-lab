import numpy as np

X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
], dtype=float)

y = np.array([1, 3, 0, 2], dtype=float)

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
    for i in range(steps):
        dw,db = gradients_multi(X,y,w,b)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w,b

w = np.zeros(2)
b = 0.0
print("初始损失：", cost_multi(X,y,w,b))
print("初始梯度：", gradients_multi(X,y,w,b))
learning_rate = 0.1
w,b = train_multi(X,y,w,b,learning_rate,1000)
print("训练1000步后参数：", w,b)
print("训练1000步后损失：", cost_multi(X,y,w,b))

# 只保留原数据的第一列特征
X_single = X[:,0:1]
w_single = np.array([1.0])
b_single = 1
print("单特征初始损失：", cost_multi(X_single,y,w_single,b_single))
print("单特征初始梯度：", gradients_multi(X_single,y,w_single,b_single))
w_single,b_single = train_multi(X_single,y,w_single,b_single,learning_rate,1000)
print("单特征训练1000步后参数：", w_single,b_single)
print("单特征训练1000步后损失：", cost_multi(X_single,y,w_single,b_single))

# 换样本数和数据
X2 = np.array([
    [-1,  0],
    [ 1,  0],
    [ 0, -1],
    [ 0,  1],
    [ 1,  1],
], dtype=float)

y2 = np.array([4, 0, -1, 5, 3], dtype=float)

w2 = np.zeros(2)
b2 = 0.0
print("新样本初始损失：", cost_multi(X2,y2,w2,b2))
print("新样本初始梯度：", gradients_multi(X2,y2,w2,b2))
w2,b2 = train_multi(X2,y2,w2,b2,learning_rate,1000)
print("新样本训练1000步后参数：", w2,b2)
print("新样本训练1000步后损失：", cost_multi(X2,y2,w2,b2))
print("新输入[2, -1]的预测：", np.array([2.0, -1.0]) @ w2 + b2)

# 与昨日单变量实验使用相同数据和初始参数，对照一步更新
X_previous = np.array([[0], [1], [2], [3]], dtype=float)
y_previous = np.array([1, 3, 5, 7], dtype=float)
w_previous = np.array([1.0])
b_previous = 1.0
print("昨日数据初始损失：", cost_multi(X_previous,y_previous,w_previous,b_previous))
print("昨日数据初始梯度：", gradients_multi(X_previous,y_previous,w_previous,b_previous))
w_previous,b_previous = train_multi(X_previous,y_previous,w_previous,b_previous,0.1,1)
print("昨日数据一步更新后参数：", w_previous,b_previous)
