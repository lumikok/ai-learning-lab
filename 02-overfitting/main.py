# 过拟合-模拟实验

# 案例1

train_data = [(1, 2), (2,5), (3,6)]

def predict_a(x):
    return 2 * x

def predict_b(x):
    return 2 * x + 1 - (x - 2) ** 2

# 平均绝对误差-MAE
def mae(predict, data):
    return sum(abs(predict(x) - y) for x, y in data) / len(data)

print("MAE for predict_a:", mae(predict_a, train_data))
print("MAE for predict_b:", mae(predict_b, train_data))

new_data = [(1.5,3),(2.5,5),(4,8)]

print("MAE for predict_a on new data:", mae(predict_a, new_data))
print("MAE for predict_b on new data:", mae(predict_b, new_data))

