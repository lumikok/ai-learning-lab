# AI 基础学习实验

按学习单元保存可运行的小实验，同一概念的对比和变式放在同一目录。

| 单元 | 主题 | 实验 |
|---|---|---|
| [01-train-and-predict](01-train-and-predict/README.md) | ML 基础 | 参数枚举、训练/测试误差、预处理信息泄漏 |
| [02-overfitting](02-overfitting/README.md) | ML 基础 | 给定模型的训练/新数据 MAE 对比与符号错误定位 |

## 运行

目前实验只使用 Python 标准库，在本目录运行：

```powershell
python .\01-train-and-predict\main.py
```

需要外部依赖时再在 AI 根目录统一建立 uv 环境，提交 pyproject.toml 和 uv.lock，各单元不重复建环境。

## 日常 Git 工作流

完成一个有意义的学习单元后：运行实验 → 检查实际改动 → 更新必要的实验说明 → 精确暂存相关文件 → 检查暂存内容 → 用户执行提交 → 核对提交结果。

教师在用户说“完成了”后自动核查代码；核查通过后给出本次实际文件清单及可直接执行的提交命令。提交信息冒号后写中文。用户执行提交，尚未执行的命令不记为已提交。当前未配置远程仓库。

```powershell
git status --short
git diff
# git add -- 后填写本次已核对的具体文件，不使用全目录暂存。
git diff --cached
git diff --cached --check
# 暂存内容正确且检查通过后，执行本次提供的 git commit 命令。
git log -1 --oneline
git status --short
```

虚拟环境、密钥、大数据、模型权重与生成输出不提交；可复现代码、必要说明、手工小数据和依赖锁文件提交。长期知识笔记在笔记库，能力证据与下一断点在学习项目 state，不复制完整学习流水账。
