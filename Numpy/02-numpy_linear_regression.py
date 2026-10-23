# 一元线性回归模型模拟实验（使用numpy实现）
import numpy as np

data = np.array([[0,1],
                 [1,3],
                 [2,5],
                 [3,7]],dtype=float)

x = data[:,0]
y = data[:,1]

def cost_np(x,y,w,b):
    return np.sum((w*x+b-y) ** 2) / (2 * len(x))

def gradients_np(x,y,w,b):
    e = w*x+b-y
    dw = np.dot(e,x) / len(x)
    db = np.sum(e) / len(x)
    return dw,db

def train_np(x,y,w,b,learning_rate,steps):
    for i in range(steps):
        dw,db = gradients_np(x,y,w,b)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w,b

w,b = 1,1
learning_rate = 0.1
print("初始损失：", cost_np(x,y,w,b))
print("初始梯度：", gradients_np(x,y,w,b))
print("训练一步后参数：", train_np(x,y,w,b,learning_rate,1))
w,b = 1,1
print("训练1000步后参数：", train_np(x,y,w,b,learning_rate,1000))

data2 = np.array([[-2,  5],
                  [ 0,  1],
                  [ 1, -1]], dtype=float)
x2 = data2[:,0]
y2 = data2[:,1]

w,b = 0,0

print("x.y的shape:",x2.shape,y2.shape)
print("残差e的shape:",(w*x2+b-y2).shape)
w,b = train_np(x2,y2,w,b,learning_rate,1000)
print("训练1000步后参数：",w,b)
print("训练1000步后损失：", cost_np(x2,y2,w,b))

# 多特征表示
# 每行表示一个样本，每列表示一个特征
X = np.array([[2, 3],
              [1, 4],
              [0, 2]], dtype=float)
# 权重
w = np.array([4, -1], dtype=float)
b = 5.0
# @ 运算符表示矩阵乘法
predictions = X @ w + b
print("预测值：",predictions)
