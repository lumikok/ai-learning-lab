"""
新数据迁移练习
scikit-learn工具使用
逻辑回归基线
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

data = load_breast_cancer()
X = data.data
y_raw = data.target
y = (y_raw == 0).astype(int)  # 调换标签，0表示良性，1表示恶性

print("特征的shape:", X.shape)
print("标签的shape:", y.shape)
print("恶性样本数：",(y == 1).sum())

"""数据集划分"""

# 第一次划分：80%开发集，20%测试集
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 第二次划分：从开发集中划分出25%作为验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

# 打印数据集形状
print("\n========== 数据集划分结果 ==========")
print("训练集:", X_train.shape, y_train.shape)
print("验证集:", X_val.shape, y_val.shape)
print("最终测试集:", X_test.shape, y_test.shape)

print("\n========== 标签分布 ==========")
print("训练集恶性样本:", np.sum(y_train == 1))
print("验证集恶性样本:", np.sum(y_val == 1))
print("测试集恶性样本:", np.sum(y_test == 1))

"""模型训练与评估"""
# 创建流水线(pipeline)，包含标准化和逻辑回归
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 使用训练集训练模型
model.fit(X_train, y_train)

# 查看类别顺序
print("\n模型类别顺序:", model.classes_)

# 获取验证集预测概率
val_prob = model.predict_proba(X_val)

# 取出恶性类别的预测概率
val_prob_malignant = val_prob[:, 1]
# 取0.5为分类阈值
val_pred = (val_prob_malignant >= 0.5).astype(int)

# 计算TP和FN
TP = np.sum((val_pred == 1) & (y_val == 1))
FN = np.sum((val_pred == 0) & (y_val == 1))

# 计算召回率
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

print("\n========== 验证集评价 ==========")
print("恶性样本总数:", TP + FN)
print("正确检出的恶性样本 TP:", TP)
print("漏检的恶性样本 FN:", FN)
print(f"恶性召回率: {recall:.4f}")