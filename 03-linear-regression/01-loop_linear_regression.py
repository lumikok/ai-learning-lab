# 一元线性回归模型模拟实验

# 预测函数
def predict(x,w,b):
    return w * x + b

# 损失函数
def cost(data,w,b):
    a = 0
    for x,y in data:
        a += (predict(x,w,b)-y) ** 2
    return a / (2 * len(data))

# 梯度计算函数
def gradients(data,w,b):
    dw = 0
    db = 0
    for x,y in data:
        dw += (predict(x,w,b)-y) * x
        db += (predict(x,w,b)-y)
    return dw / len(data), db / len(data)

# 训练函数
def train(data,w,b,learning_rate,steps):
    for i in range(steps):
        dw, db = gradients(data,w,b)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w,b

# 初始数据
data = [(0,1),(1,3),(2,5),(3,7)]
# 初始参数
w = 1
b = 1
learning_rate = 0.1

print("初始损失：", cost(data,w,b))
print("初始梯度：", gradients(data,w,b))

w,b = train(data,w,b,learning_rate,1)
cost_after_one_step = cost(data,w,b)

print("一步训练后参数：",w,b)
print("一步训练后损失：", cost_after_one_step)

# 训练1000步（从初始参数开始）
w,b = 1,1
w,b = train(data,w,b,learning_rate,1000)
print("训练1000步后参数：",w,b)
print("训练1000步后损失：", cost(data,w,b))

# 验证集
validation_data = [(0.5,2),(2.5,6)]
MAE = 0
for x,y in validation_data:
    MAE += abs(predict(x,w,b)-y)

MAE /= len(validation_data)
print("验证集平均绝对误差：",MAE)

# 变式
train_data2 = [(-1,4),(0,1),(1,-2),(2,-5)]
validation_data2 = [(-0.5,2.5),(1.5,-3.5)]
w ,b = 0,0
# 训练1000步
w,b = train(train_data2,w,b,learning_rate,1000)
cost_after_training = cost(train_data2,w,b)
print("变式训练1000步后参数：",w,b)
print("变式训练1000步后损失：", cost_after_training)

MAE2 = 0
for x,y in validation_data2:
    MAE2 += abs(predict(x,w,b)-y)
MAE2 /= len(validation_data2)
print("变式验证集平均绝对误差：",MAE2)

# 模拟不同学习率对训练结果的影响
learning_rates = [0.01, 0.1,1.0,0.5]
for lr in learning_rates:
    w,b = 1,1
    cost_training = [cost(data,w,b)]
    for step in range(20):
        w,b = train(data,w,b,lr,1)
        cost_training.append(cost(data,w,b))
    print(f"学习率：{lr}，训练0,1,2,3,5,10,20步后的损失：", cost_training[0], cost_training[1], cost_training[2], cost_training[3], cost_training[5], cost_training[10], cost_training[20])
    print(f"学习率：{lr}，训练20步后参数：",w,b)
