import numpy as np

data = np.array([[0,1],
                [1,3],
                [2,5],
                [3,7]], dtype=float)

x = data[:,0]
y = data[:,1]
w,b = 1,1

predictions = w * x + b
residuals = predictions - y
J = np.mean(residuals ** 2) / 2
print("预测值：", predictions)
print("残差：", residuals)
print("损失：", J)
print("x,y为：", x.shape, y.shape)
print("预测值大小：", predictions.shape)

# 小变式
data2 = np.array([[101, 0, 2],
                  [102, 1, 5],
                  [103, 2, 8]], dtype=float)

id = data2[:,0]
x2 = data2[:,1]
y2 = data2[:,2]
w,b = 2,1
predictions2 = w * x2 + b
residuals2 = predictions2 - y2
J2 = np.mean(residuals2 ** 2) / 2
print("预测值：", predictions2)
print("残差：", residuals2)
print("损失：", J2)
print("x,y为：", x2.shape, y2.shape)
print("预测值大小：", predictions2.shape)