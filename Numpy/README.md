# NumPy数组与基础计算

用已有线性模型连接数组创建、形状、取列和逐元素计算。本目录沿用当前学习文件位置。

在仓库根目录运行：

```powershell
uv sync --locked
uv run --frozen python Numpy/01-numpy_basics.py
```

`01-numpy_basics.py` 使用 `dtype=float` 创建二维数组，取出一维输入与标签，用数组表达式计算预测、残差和半均方误差；不执行参数训练。

- 四行两列数据：x、y及预测的shape均为(4,)，预测[1,2,3,4]、残差[0,-1,-2,-3]，J=1.75，与逐样本循环一致。
- 三列变式的列依次为编号、x、y，编号不参与预测。取后两列，用w=2、b=1得到预测[1,3,5]、残差[-1,-2,-3]、J=7/3，数组shape均为(3,)。模型参数未拟合该数据，非零损失不是取列错误。

二维数组的shape表示两条轴的长度；一维(4,)与二维(4,1)不同。`data[:, 1]`取整列并得到一维数组，`data[1, :]`取一条样本。数组与标量的基本算术逐元素计算，同形数组按对应位置运算，`np.mean`求平均。

参考：[NumPy入门](https://numpy.org/doc/stable/user/absolute_beginners.html)的Array fundamentals与Array attributes；本地练习不作为官方作业完成记录。
