# AI Learning Lab

个人 AI 实验仓库，包含线性回归、逻辑回归、Iris 二分类和两层神经网络的 NumPy 实现。

通过对照实验观察数据、特征、参数与实现方式如何影响结果。各单元说明包含运行方法、关键机制、实验结果和易错点。

## 实验索引

| 单元 | 主题 | 实验 |
|---|---|---|
| [01-train-and-predict](01-train-and-predict/README.md) | ML 基础 | 参数枚举、训练/测试误差、预处理信息泄漏 |
| [02-overfitting](02-overfitting/README.md) | ML 基础 | 给定模型的训练/新数据 MAE 对比与符号错误定位 |
| [03-linear-regression](03-linear-regression/README.md) | ML 基础 | 循环与数组梯度下降、多特征训练、学习率/删列/标准化对照 |
| [04-logistic-regression](04-logistic-regression/README.md) | ML 基础 | 概率预测、交叉熵、梯度训练、阈值与边界检查 |
| [05-iris-classification](05-iris-classification/README.md) | ML 基础 | Iris 二分类、训练/验证、简单基线、错误分析与多划分特征对照 |
| [06-neural-network](06-neural-network/README.md) | DL 基础 | 两层网络前向、非线性、反向传播、数值梯度检查与三样本短训练 |
| [Numpy](Numpy/README.md) | 数值工具基础 | 数组、shape、索引与批量预测/损失计算 |

## 目录组织

按实验单元组织目录，每个目录包含代码和说明；相关对照与变式放在同一目录。根目录统一管理依赖与运行环境。

## 运行方式

使用 Python 3.11 或更高版本，在仓库根目录通过 uv 安装锁定依赖并运行：

```powershell
uv sync --locked
uv run --frozen python .\01-train-and-predict\main.py
uv run --frozen python .\02-overfitting\main.py
uv run --frozen python .\03-linear-regression\01-loop_linear_regression.py
uv run --frozen python .\Numpy\01-numpy_basics.py
uv run --frozen python .\03-linear-regression\02-numpy_linear_regression.py
uv run --frozen python .\03-linear-regression\03-multiple_linear_regression.py
uv run --frozen python .\03-linear-regression\04-feature_scaling.py
uv run --frozen python .\04-logistic-regression\01-prediction_and_loss.py
uv run --frozen python .\04-logistic-regression\02-logistic_training.py
uv run --frozen python .\05-iris-classification\main.py
uv run --frozen python .\06-neural-network\01-forward.py
uv run --frozen python .\06-neural-network\02-training.py
```

01、02单元和线性回归循环版使用 Python 标准库；数组运算使用 NumPy 2.4.3。Iris 实验还使用 scikit-learn 加载数据和划分样本，训练核心使用 NumPy 实现。依赖版本由 `pyproject.toml` 和 `uv.lock` 管理，各单元共用 `.venv`。

## 数据与实验产物

手工小数据直接保存在脚本中，外部数据的来源与加载方式见对应实验说明。虚拟环境、缓存、密钥、大型数据和模型权重由 `.gitignore` 排除。
