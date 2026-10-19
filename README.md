# AI Learning Lab

个人 AI 学习与实验仓库，用代码、对照实验和结果分析连接理论与实践。

从机器学习的训练与评价基础起步，逐步扩展到深度学习、Transformer、大语言模型（LLM）及图文多模态。当前内容以小规模基础实验为主，后续随实际学习逐步补充。

## 仓库用途

- 实现学习中的关键机制，观察参数、数据和实现方式如何影响结果。
- 保存可复现的小实验、必要说明及调试中的发现。
- 在同一学习单元中加入对比和变式，检验理解与迁移能力。

实验说明保留问题、运行方法、关键结果与局限；长期知识总结另行整理。涉及课程时注明对应内容与来源，自行设计的练习和官方作业分别标明。

## 当前实验

| 单元 | 主题 | 实验 |
|---|---|---|
| [01-train-and-predict](01-train-and-predict/README.md) | ML 基础 | 参数枚举、训练/测试误差、预处理信息泄漏 |
| [02-overfitting](02-overfitting/README.md) | ML 基础 | 给定模型的训练/新数据 MAE 对比与符号错误定位 |
| [03-linear-regression](03-linear-regression/README.md) | ML 基础 | 手写梯度下降、训练/验证评价与负斜率变式 |
| [Numpy](Numpy/README.md) | 数值工具基础 | 数组、shape、取列与批量预测/损失计算 |

## 目录组织

按学习单元建立目录，例如 `03-linear-regression/`。每个目录保存该单元的代码与 `README.md`，同一概念的对照和变式优先放在同一目录。随着实验需要，可以增加小数据、图表或 Notebook，无需预先创建空目录。

## 运行方式

使用 Python 3.11 或更高版本，在仓库根目录通过 uv 安装锁定依赖并运行：

```powershell
uv sync --locked
uv run --frozen python .\01-train-and-predict\main.py
uv run --frozen python .\02-overfitting\main.py
uv run --frozen python .\03-linear-regression\main.py
uv run --frozen python .\Numpy\01-numpy_basics.py
```

具体实验的背景、结果和限制见各单元的说明。

前三个实验只用Python标准库，NumPy实验依赖NumPy 2.4.3。依赖与版本锁在根目录统一管理，各单元共用`.venv`，不重复建立环境。

## 数据与实验产物

仓库保存代码、必要说明、手工小数据和依赖配置。虚拟环境、密钥、大型原始数据、模型权重、缓存及批量生成的输出由 `.gitignore` 排除。使用外部数据或模型时，在实验说明中记录来源、获取方式和复现条件。
