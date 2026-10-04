
def total_error(b):
    """
    计算三次预测的绝对误差之和，
    """
    error1 = abs(3*1 + b - 8)
    error2 = abs(3*2 + b - 12)
    error3 = abs(3*3 + b - 13)
    return error1 + error2 + error3

# 遍历0到10的整数，输出最小的b和对应的误差
errors = []
for b in range(11):
    errors.append(total_error(b))

min_error = min(errors)
min_b = errors.index(min_error)
print(f"b={min_b}, minimum error={min_error}")

# 变式1：把，枚举范围改成 range(3,11)
errors = []
for b in range(3, 11):
    errors.append(total_error(b))
min_error = min(errors)
min_b = errors.index(min_error) + 3  # 因为枚举从3开始，所以需要加上偏移量
print(f"b={min_b}, minimum error={min_error}")

# 变式2：改成固定参数列表
example_bs = [0,2,5,9]
errors = []
for b in example_bs:
    errors.append(total_error(b))
min_error = min(errors)
min_b = example_bs[errors.index(min_error)]
print(f"b={min_b}, minimum error={min_error}")

# 小实验：计算训练集和测试集的误差之和
train = [(1, 8), (2, 12), (3, 13)]
test = [(4, 19), (5, 22)]

def total_error_with_data(b,data):
    """
    计算数据集的绝对误差之和
    """
    total = 0
    for x, y in data:
        error = abs(3*x + b - y)
        total += error
    return total

errors = [total_error_with_data(b, train) for b in range(11)]
b_min = errors.index(min(errors))
# 输出平均绝对误差
test_errors = total_error_with_data(b_min, test) / len(test)
print(f"b={b_min}, test average error={test_errors}")

train_data = [1,2,3]
test_data = [4,10]

max_data_train = max(train_data)
max_data_train_test = max(max_data_train, max(test_data))

ans1 = [i / max_data_train for i in train_data + test_data]
ans2 = [i / max_data_train_test for i in train_data + test_data]

print(ans1)
print(ans2)
