# 模型评估

## 01 训练与验证基线

用 Breast Cancer Wisconsin 数据集练习分层划分、训练集标准化和验证集评估。数据由 scikit-learn 的 `load_breast_cancer()` 提供，共 569 条样本、30 个特征。实验将恶性映射为正类 1，良性映射为 0。

在仓库根目录运行。

```powershell
uv sync --locked
uv run --frozen python .\07-model-evaluation\01-validation-baseline.py
```

先分层留出 20% 作为最终测试集，再从开发集中分层留出 25% 作为验证集，两次划分都使用 `random_state=42`。

| 用途 | 样本数 | 恶性样本数 |
| --- | ---: | ---: |
| 训练 | 341 | 127 |
| 验证 | 114 | 43 |
| 最终测试 | 114 | 42 |

`StandardScaler` 和 `LogisticRegression(max_iter=1000)` 组成流水线，只对训练集调用 `fit`。验证时将原始特征交给 `predict_proba`，流水线使用训练集的统计量进行变换。根据 `classes_ = [0, 1]` 取第二列当作恶性概率，用 0.5 作为分类阈值。

验证集中检出 42 条恶性样本，漏判 1 条，恶性召回率为 42 / 43 ≈ 97.67%。这个结果来自本次固定划分。最终测试集留待选型完成后评估。
