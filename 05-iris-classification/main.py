"""NumPy逻辑回归核心，复用自04单元的训练实现。
    该文件用于在iris数据集上进行真实的二分类任务的逻辑回归训练。
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

"""逻辑回归核心函数
    1. sigmoid函数：把线性得分映射到0-1之间
    2. 二分类交叉熵损失函数：衡量预测的优劣
    3. 计算逻辑回归的梯度：用于更新权重和偏置
    4. 训练逻辑回归模型：迭代更新权重和偏置，记录每次迭代的损失"""

# sigmoid函数：把线性得分映射到0-1之间
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 二分类交叉熵损失函数
def binary_cross_entropy(y_true, y_pred):
    # 避免对数零的情况
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 计算逻辑回归的梯度
def gradients_logistic(X,y,w,b):
    predictions = sigmoid(X @ w + b)
    dw = X.T @ (predictions - y) / len(y)
    db = np.sum(predictions - y) / len(y)
    return dw, db

# 训练逻辑回归模型
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

"""数据预处理
    1. 取出非0类别的样本中的特征和标签
    2. 将数据集划分为训练集和验证集(3:1)
    3. 映射标签为0和1，便于二分类任务

"""

# 取出非0类别的样本中的特征和标签
mask = iris.target != 0
X = iris.data[mask]                   # 特征:(n_samples, n_features)
y = iris.target[mask]                 # 标签:(n_samples,)

X_train, X_val,y_train,y_val = train_test_split(
    X,y,
    test_size=0.25,     # 25%验证集，75%训练集
    random_state=42,    # 保证每次划分的随机性一致
    stratify=y          # 保证训练集和验证集中各类别样本的比例一致
)

y_train = np.where(y_train == 1, 0, 1)  # 将标签映射为0和1
y_val = np.where(y_val == 1, 0, 1)      # 将标签映射为0和1

"""打印数据集信息，用于确认数据集划分是否正确"""

# 原始数据集信息
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")

# 筛选后的数据集信息
print(f"筛选后的特征形状: {X.shape}")
print(f"筛选后的标签形状: {y.shape}")

# 划分后的训练集和验证集信息
print(f"训练集特征形状: {X_train.shape}")
print(f"训练集标签形状: {y_train.shape}")
print(f"验证集特征形状: {X_val.shape}")
print(f"验证集标签形状: {y_val.shape}")

"""处理数据并进行逻辑回归训练，并打印训练集和验证集的最终结果
    1.使用训练集的均值和标准差对训练集和验证集进行标准化
    2.打印标准化后的训练集每列均值和标准差
    3.统计训练集和验证集中各类别样本的数量
    补充：初始更新核对：打印零参数下的db和一步后偏置
    4.训练逻辑回归模型
    5.整理为评价函数，计算错误率、错误样本数、假阳性样本数、假阴性样本数、交叉熵损失
    6.打印训练集和验证集的最终结果
    7.基线模型：均预测为训练集中样本数较多的类别，并计算其错误率、假阳性样本数、假阴性样本数
    补充：基线模型用于检查逻辑回归模型的预测效果是否优于简单的多数类预测。
"""

# 保留验证集原始信息
X_val_raw = X_val.copy()

X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std

# 打印标准化后的训练集标准化后的每列均值和标准差
print(f"训练集标准化后的每列均值: {X_train.mean(axis=0)}")
print(f"训练集标准化后的每列标准差: {X_train.std(axis=0)}")

# 统计训练集和验证集中各类别样本的数量
y_train_0 = np.sum(y_train == 0)
y_train_1 = np.sum(y_train == 1)
y_val_0 = np.sum(y_val == 0)
y_val_1 = np.sum(y_val == 1)

print(f"训练集中标签为0的样本数: {y_train_0}")
print(f"训练集中标签为1的样本数: {y_train_1}")

print(f"验证集中标签为0的样本数: {y_val_0}")
print(f"验证集中标签为1的样本数: {y_val_1}")

# 训练逻辑回归模型
w = np.zeros(X_train.shape[1])  # 初始化权重（四个特征对应四个权重）
b = 0                           # 初始化偏置
learning_rate = 0.1            # 学习率
steps = 1000                   # 迭代次数

# 单独核对模块
w_init = np.zeros(X_train.shape[1])  # 初始化权重（四个特征对应四个权重）
b_init = 0                           # 初始化偏置
print("初始参数下，偏置b的梯度db和一步后的偏置b:")
dw_init, db_init = gradients_logistic(X_train, y_train, w_init, b_init)
print(f"初始偏置b的梯度db: {db_init}")
w_init, b_init, _ = train_logistic(X_train, y_train, w_init, b_init, learning_rate, 1)
print(f"一步后更新的偏置b: {b_init}")

w,b,history = train_logistic(X_train,y_train,w,b,learning_rate,steps)

# 整理为评价函数
def evaluate_model(y_true,y_pred,y_pred_prob):
    error_rate = np.mean(y_pred != y_true)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    loss = binary_cross_entropy(y_true, y_pred_prob)
    error_count = fp + fn
    return error_rate, error_count, fp, fn, loss

# 验证集预测（使用更新后的权重和偏置进行预测）
threshold = 0.5                                           # 预测阈值
y_val_p = sigmoid(X_val @ w + b)                          # 预测概率
y_val_pred = (y_val_p >= threshold).astype(int)           # 将预测概率转换成0或1的预测标签

# 验证集的错误率、错误样本数、假阳性样本数、假阴性样本数、交叉熵损失
val_error_rate, val_error_count, val_fp, val_fn, val_loss = evaluate_model(y_val,y_val_pred,y_val_p)

train_loss = history[-1]                                  # 最终训练集的交叉熵损失
train_error_rate = np.mean((sigmoid(X_train @ w + b) >= threshold).astype(int) != y_train)  # 训练集的错误率

# 打印训练集和验证集的最终结果
print(f"训练集最终交叉熵损失: {train_loss:.4f}")
print(f"训练集最终错误率: {train_error_rate:.4f}")
print(f"验证集最终交叉熵损失: {val_loss:.4f}")
print(f"验证集最终错误率: {val_error_rate:.4f}")
print(f"验证集错误样本数: {val_error_count}")
print(f"验证集假阳性样本数: {val_fp}")
print(f"验证集假阴性样本数: {val_fn}")

# 基线模型：均预测为训练集中样本数较多的类别
baseline_class = np.full(y_val.shape, 0) if y_train_0 > y_train_1 else np.full(y_val.shape, 1)
baseline_p = np.full(y_val.shape, 0) if y_train_0 > y_train_1 else np.full(y_val.shape, 1) # 预测概率为1或0，取决于训练集中样本数较多的类别
baseline_error_rate, baseline_error_count, baseline_fp, baseline_fn, _ = evaluate_model(y_val, baseline_class, baseline_p)
print(f"基线模型错误率: {baseline_error_rate:.4f}")
print(f"基线模型假阳性样本数: {baseline_fp}")
print(f"基线模型假阴性样本数: {baseline_fn}")

"""错误分析
    1. 找出验证集中预测错误的记录
    2. 打印验证集中预测错误的记录的索引、四个原始测量值、真实标签、预测概率、预测标签、是FP还是FN
"""

# 错误的记录的四个原始测量值
# np.where()函数返回的是一个元组，其中第一个元素是满足条件的索引数组
error_indices = np.where(y_val != y_val_pred)[0]        # 找出验证集中预测错误的记录的索引
error_real_records = X_val_raw[error_indices]                 # 错误的记录的四个原始测量值
error_real_labels = y_val[error_indices]                              # 错误的记录的真实标签
error_pred_p = y_val_p[error_indices]                              # 错误的记录的预测概率
error_pred_labels = y_val_pred[error_indices]                              # 错误的记录的预测标签
error_fp_or_fn = ["假阳性" if pred == 1 and real == 0 else "假阴性"              # zip()函数将两个列表打包成一个元组列表，方便同时遍历预测标签和真实标签
                  for pred, real in zip(error_pred_labels, error_real_labels)]  # 错误的记录是FP还是FN

print(f"验证集中预测错误的记录的索引: {error_indices}")
print(f"验证集中预测错误的记录的四个原始测量值: {error_real_records}")
print(f"验证集中预测错误的记录的真实标签: {error_real_labels}")
print(f"验证集中预测错误的记录的预测概率: {error_pred_p}")
print(f"验证集中预测错误的记录的预测标签: {error_pred_labels}")
print(f"验证集中预测错误的记录是FP还是FN: {error_fp_or_fn}")

"""特征消融实验 -> 验证被消融的特征对模型性能的影响
    1. 取双特征组合（排除萼片长度和萼片宽度的组合），进行逻辑回归训练
    2. 比较四特征与花瓣双特征模型的训练、验证指标
"""

# 选取花瓣两特征
X_train_petals = X_train[:, 2:4]  # 花瓣长度和花瓣宽度
X_val_petals = X_val[:, 2:4]      # 花瓣长度和花瓣宽度

w_petals = np.zeros(X_train_petals.shape[1])  # 初始化权重（两个特征对应两个权重）
b_petals = 0                                  # 初始化偏置

w_petals, b_petals, history_petals = train_logistic(
    X_train_petals, y_train, w_petals, b_petals, learning_rate, steps
)

# 训练集和验证集的最终结果
train_loss_petals = history_petals[-1]                                  # 最终训练集的交叉熵损失
train_p_petals = sigmoid(X_train_petals @ w_petals + b_petals)                  # 训练集预测概率
train_error_rate_petals = np.mean((train_p_petals >= threshold).astype(int) != y_train)  # 训练集的错误率

val_p_petals = sigmoid(X_val_petals @ w_petals + b_petals)                      # 验证集预测概率
val_error_rate_petals, val_error_count_petals, val_fp_petals, val_fn_petals, val_loss_petals = evaluate_model(y_val, (val_p_petals >= threshold).astype(int), val_p_petals)

print(f"{'指标':<20}{'四特征模型':>15}{'花瓣特征模型':>18}")

results = [
    ("训练集交叉熵", train_loss, train_loss_petals),
    ("训练集错误率", train_error_rate, train_error_rate_petals),
    ("验证集交叉熵", val_loss, val_loss_petals),
    ("验证集错误率", val_error_rate, val_error_rate_petals),
    ("验证集 FP", val_fp, val_fp_petals),
    ("验证集 FN", val_fn, val_fn_petals),
]

for name, full, petals in results:
    print(f"{name:<20}{full:>15.4f}{petals:>18.4f}")

# 检查原错误样本是否仍然错误
y_val_pred_petals = (val_p_petals >= threshold).astype(int)

for i in error_indices:
    print(f"验证集索引: {i}")
    print(f"真实标签: {y_val[i]}")
    print(f"原模型预测: {y_val_pred[i]}")
    print(f"花瓣模型预测: {y_val_pred_petals[i]}")

    if y_val_pred_petals[i] == y_val[i]:
        print("结果：原错误已纠正")
    else:
        print("结果：仍然错误")

    print("-" * 50)

"""修改划分进行训练和验证
    1. 将"划分 -> 标准化 -> 两组训练 -> 评价"整理为函数，便于随机种子的修改和多次训练验证
    2. 使用不同的随机种子进行多次训练和验证，打印每次的训练集和验证集的最终结果
"""

def run_experiment(random_state):
    # 1.将原始数据重新划分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.25,
        random_state=random_state,
        stratify=y
    )

    # 2.将标签映射为0和1
    y_train = np.where(y_train == 1, 0, 1)
    y_val = np.where(y_val == 1, 0, 1)

    # 3.使用当前训练集计算均值和标准差，并对训练集和验证集进行标准化
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma

    # 4.训练四特征逻辑回归模型，每次从零初始化权重和偏置开始训练
    w = np.zeros(X_train.shape[1])
    b = 0

    w, b, history = train_logistic(X_train, y_train, w, b, learning_rate, steps)

    # 四特征模型的训练集和验证集的最终结果
    train_loss = history[-1]
    train_p = sigmoid(X_train @ w + b)

    val_p = sigmoid(X_val @ w + b)
    val_pred = (val_p >= threshold).astype(int)

    val_error_rate, val_error_count, val_fp, val_fn, val_loss = evaluate_model(y_val, val_pred, val_p)

    # 5.训练花瓣特征逻辑回归模型，每次从零初始化权重和偏置开始训练
    X_train_petals = X_train[:, 2:4]
    X_val_petals = X_val[:, 2:4]

    w_petals = np.zeros(X_train_petals.shape[1])
    b_petals = 0

    w_petals, b_petals, history_petals = train_logistic(
        X_train_petals, y_train, w_petals, b_petals, learning_rate, steps
    )

    # 花瓣特征模型的训练集和验证集的最终结果
    train_loss_petals = history_petals[-1]
    train_p_petals = sigmoid(X_train_petals @ w_petals + b_petals)

    val_p_petals = sigmoid(X_val_petals @ w_petals + b_petals)
    val_pred_petals = (val_p_petals >= threshold).astype(int)

    val_error_rate_petals, val_error_count_petals, val_fp_petals, val_fn_petals, val_loss_petals = evaluate_model(y_val, val_pred_petals, val_p_petals)

    # 6.返回结果
    return {
        "seed": random_state,
        "full_train_loss": train_loss,
        "full_error_rate": val_error_rate,
        "full_fp": val_fp,
        "full_fn": val_fn,
        "full_val_loss": val_loss,
        "full-error_count": val_error_count,
        "petals_train_loss": train_loss_petals,
        "petals_error_rate": val_error_rate_petals,
        "petals_fp": val_fp_petals,
        "petals_fn": val_fn_petals,
        "petals_val_loss": val_loss_petals,
        "petals-error_count": val_error_count_petals,
    }

# 运行实验，使用不同的随机种子进行多次训练和验证
seeds = [0,1,2,42]
results = [run_experiment(seed) for seed in seeds]

for r in results:
    print(f"\n========== 随机种子 {r['seed']} ==========")
    print(f"{'指标':<20}{'四特征模型':>14}{'花瓣特征模型':>16}")
    print("-" * 50)

    rows = [
        ("训练集交叉熵", r["full_train_loss"], r["petals_train_loss"]),
        ("验证集交叉熵", r["full_val_loss"], r["petals_val_loss"]),
        ("验证集错误率", r["full_error_rate"], r["petals_error_rate"]),
        ("验证集 FP", r["full_fp"], r["petals_fp"]),
        ("验证集 FN", r["full_fn"], r["petals_fn"]),
        ("验证集错误数", r["full-error_count"], r["petals-error_count"]),
    ]

    for name, full, petals in rows:
        if name in ("验证集 FP", "验证集 FN", "验证集错误数"):
            print(f"{name:<20}{full:>14}{petals:>16}")
        elif name == "验证集错误率":
            print(f"{name:<20}{full:>13.2%}{petals:>15.2%}")
        else:
            print(f"{name:<20}{full:>14.4f}{petals:>16.4f}")
