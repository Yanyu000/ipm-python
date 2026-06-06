# 从零手写一个基于 Cholesky 分解的 Interior Point LP Solver

这份笔记的目标不是让你复制一个完整 solver，而是帮助你**一步一步独立写出一个基础的 primal-dual interior point LP solver**。我们先只用 `numpy`，在 Mac 上运行，先解小规模标准型 LP：

$$
\min_x c^T x
$$

subject to

$$
Ax=b, \quad x \ge 0.
$$

对应的 dual 是：

$$
\max_y b^T y
$$

subject to

$$
A^T y+s=c, \quad s \ge 0.
$$

其中：

- $x \in \mathbb{R}^n$ 是 primal variable；
- $y \in \mathbb{R}^m$ 是 equality constraint 的 dual variable；
- $s \in \mathbb{R}^n$ 是 dual slack variable。

---

## 1. 在 Mac 上配置 Python 环境

### 1.1 创建项目文件夹

打开 Terminal，运行：

```bash
mkdir ipm-lp-solver
cd ipm-lp-solver
```

这会创建一个新的项目文件夹，并进入该文件夹。

---

### 1.2 创建虚拟环境

运行：

```bash
python3 -m venv .venv
```

这句话的意思是：

- `python3`：使用 Mac 上的 Python 3；
- `-m venv`：调用 Python 自带的虚拟环境模块；
- `.venv`：创建一个名叫 `.venv` 的虚拟环境文件夹。

创建完成后，你的项目结构大概是：

```text
ipm-lp-solver/
└── .venv/
```

`.venv` 里存放这个项目独立的 Python 和 package。这样做的好处是：

- 不污染系统 Python；
- 每个项目可以有自己的 package 版本；
- 以后别人复现实验更容易。

---

### 1.3 激活虚拟环境

每次打开一个新的 Terminal，进入这个项目后，都需要激活环境：

```bash
source .venv/bin/activate
```

激活后，你通常会看到命令行前面出现：

```text
(.venv)
```

例如：

```text
(.venv) alice@Mac ipm-lp-solver %
```

这表示你现在使用的是 `.venv` 里的 Python。

#### 每次跑代码前都要 activate 吗？

如果你开了一个新的 Terminal，是的，通常需要重新运行：

```bash
source .venv/bin/activate
```

如果你还在同一个 Terminal 里，而且前面已经激活过，就不用重复激活。

可以用下面命令检查当前 Python 路径：

```bash
which python
```

如果看到类似：

```text
/Users/yourname/ipm-lp-solver/.venv/bin/python
```

说明你正在用虚拟环境里的 Python。

---

### 1.4 安装需要的包

激活环境后，安装：

```bash
pip install numpy scipy matplotlib
```

目前最重要的是：

- `numpy`：表示向量、矩阵，做线性代数；
- `scipy`：之后可以用 `scipy.optimize.linprog` 对比结果；
- `matplotlib`：之后画 convergence plot。

---

### 1.5 创建文件

运行：

```bash
touch solver.py test_solver.py README.md
```

建议项目结构：

```text
ipm-lp-solver/
├── .venv/
├── solver.py
├── test_solver.py
└── README.md
```

其中：

- `solver.py` 放 solver 的函数；
- `test_solver.py` 放测试例子；
- `README.md` 放笔记和运行方法。

---

## 2. Python 函数基础：`def` 怎么用？

Python 里用 `def` 定义函数。

最简单的例子：

```python
def add(a, b):
    return a + b

result = add(2, 3)
print(result)
```

输出：

```text
5
```

函数的基本结构是：

```python
def 函数名(输入1, 输入2, ...):
    # 函数内部代码
    return 输出
```

注意：Python 用缩进表示代码块。函数内部的代码必须缩进，通常是 4 个空格。

---

### 2.1 多个返回值

Python 函数可以返回多个值：

```python
def split_sum_and_product(a, b):
    s = a + b
    p = a * b
    return s, p

x, y = split_sum_and_product(2, 3)
print(x)
print(y)
```

输出：

```text
5
6
```

在我们的 LP solver 里，经常会写：

```python
return rp, rd, rc, mu
```

然后用：

```python
rp, rd, rc, mu = residuals(A, b, c, x, y, s)
```

---

### 2.2 默认参数

函数可以设置默认参数：

```python
def step_length(x, dx, tau=0.99):
    ...
```

这里 `tau=0.99` 表示如果调用时没有给 `tau`，就默认用 `0.99`。

例如：

```python
alpha = step_length(x, dx)
```

等价于：

```python
alpha = step_length(x, dx, tau=0.99)
```

---

## 3. 用 NumPy 表示向量和矩阵

在 Python 中，LP 里的矩阵和向量通常用 `numpy.array`。

先导入 NumPy：

```python
import numpy as np
```

`np` 是 `numpy` 的常用缩写。

---

### 3.1 表示向量

数学里的向量：

$$
x = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
$$

在 NumPy 里可以写成：

```python
x = np.array([1., 2., 3.])
```

这里写 `1.` 而不是 `1`，是为了让它成为 floating point number。

检查 shape：

```python
print(x.shape)
```

输出：

```text
(3,)
```

这表示它是长度为 3 的一维数组。

---

### 3.2 表示矩阵

数学里的矩阵：

$$
A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

在 NumPy 里写成：

```python
A = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])
```

检查 shape：

```python
print(A.shape)
```

输出：

```text
(2, 3)
```

表示 $A$ 有 2 行、3 列。

---

### 3.3 矩阵乘向量

数学里：

$$
Ax
$$

NumPy 里用：

```python
A @ x
```

例子：

```python
import numpy as np

A = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])

x = np.array([1., 1., 1.])

print(A @ x)
```

输出：

```text
[ 6. 15.]
```

因为：

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
\begin{bmatrix}
1 \\ 1 \\ 1
\end{bmatrix}
=
\begin{bmatrix}
6 \\ 15
\end{bmatrix}.
$$

---

### 3.4 转置矩阵

数学里的：

$$
A^T
$$

NumPy 里写：

```python
A.T
```

所以：

```python
A.T @ y
```

对应数学里的：

$$
A^T y.
$$

---

### 3.5 向量逐元素乘法

在 NumPy 里：

```python
x * s
```

不是 dot product，而是逐元素乘法。

例如：

```python
x = np.array([1., 2., 3.])
s = np.array([10., 20., 30.])

print(x * s)
```

输出：

```text
[10. 40. 90.]
```

对应数学里的：

$$
x \circ s.
$$

---

### 3.6 向量内积

数学里的：

$$
x^T s
$$

NumPy 里可以写：

```python
x @ s
```

例子：

```python
x = np.array([1., 2., 3.])
s = np.array([10., 20., 30.])

print(x @ s)
```

输出：

```text
140.0
```

因为：

$$
1\cdot 10 + 2\cdot 20 + 3\cdot 30 = 140.
$$

---

## 4. 我们的第一个 LP 例子

先用最简单的 LP：

$$
\min x_1 + 2x_2
$$

subject to

$$
x_1+x_2=1,
$$

$$
x_1,x_2 \ge 0.
$$

显然，因为 $x_1$ 的成本是 1，$x_2$ 的成本是 2，所以应该尽量选 $x_1$。

最优解是：

$$
x^* = \begin{bmatrix}1 \\ 0\end{bmatrix},
$$

目标值是：

$$
c^T x^* = 1.
$$

在 NumPy 里：

```python
import numpy as np

A = np.array([[1., 1.]])
b = np.array([1.])
c = np.array([1., 2.])
```

这里：

- `A.shape == (1, 2)`，因为有 1 个 equality constraint，2 个变量；
- `b.shape == (1,)`；
- `c.shape == (2,)`。

---

## 5. KKT residuals

Primal-dual LP 的 KKT 条件是：

$$
Ax=b,
$$

$$
A^T y+s=c,
$$

$$
x_i s_i = 0, \quad i=1,\dots,n,
$$

$$
x \ge 0, \quad s \ge 0.
$$

Interior point method 里，我们保持：

$$
x > 0, \quad s > 0,
$$

然后逐渐让：

$$
x_i s_i \to 0.
$$

定义三个 residual：

$$
r_p = Ax-b,
$$

$$
r_d = A^T y+s-c,
$$

$$
r_c = x \circ s.
$$

还定义 duality measure：

$$
\mu = \frac{x^T s}{n}.
$$

---

## 练习 A：写 residuals 函数

在 `solver.py` 里写：

```python
import numpy as np


def residuals(A, b, c, x, y, s):
    """
    Compute primal residual, dual residual, complementarity residual,
    and duality measure mu.

    Parameters
    ----------
    A : numpy array, shape (m, n)
    b : numpy array, shape (m,)
    c : numpy array, shape (n,)
    x : numpy array, shape (n,)
    y : numpy array, shape (m,)
    s : numpy array, shape (n,)

    Returns
    -------
    rp : A @ x - b
    rd : A.T @ y + s - c
    rc : x * s
    mu : x @ s / n
    """
    # TODO: write this yourself
    pass
```

你要把 `pass` 替换成自己的代码。

提示：

```python
n = len(x)
rp = A @ x - b
rd = A.T @ y + s - c
rc = x * s
mu = x @ s / n
return rp, rd, rc, mu
```

但请先自己写一遍，不要直接复制。

---

### 测试 residuals

在 `test_solver.py` 里写：

```python
import numpy as np
from solver import residuals

A = np.array([[1., 1.]])
b = np.array([1.])
c = np.array([1., 2.])

x = np.array([0.5, 0.5])
y = np.array([0.])
s = np.array([1., 2.])

rp, rd, rc, mu = residuals(A, b, c, x, y, s)

print("rp =", rp)
print("rd =", rd)
print("rc =", rc)
print("mu =", mu)
```

运行：

```bash
python test_solver.py
```

你应该看到：

```text
rp = [0.]
rd = [0. 0.]
rc = [0.5 1. ]
mu = 0.75
```

解释：

- `rp = [0.]` 表示当前点满足 $Ax=b$；
- `rd = [0., 0.]` 表示当前点满足 $A^Ty+s=c$；
- `rc = [0.5, 1.]` 表示 complementarity 还没满足；
- `mu = 0.75` 是平均 complementarity。

---

## 6. Step length：如何保持 x 和 s 为正？

Interior point method 的点必须保持：

$$
x > 0, \quad s > 0.
$$

如果当前有方向 $\Delta x$，下一步是：

$$
x^+ = x + \alpha \Delta x.
$$

我们要选 $\alpha$，使得：

$$
x_i + \alpha \Delta x_i > 0.
$$

如果 $\Delta x_i \ge 0$，这个变量不会限制步长。

如果 $\Delta x_i < 0$，那么需要：

$$
\alpha < -\frac{x_i}{\Delta x_i}.
$$

所以最大步长是：

$$
\alpha = \min\left(1, \tau \min_{\Delta x_i < 0} -\frac{x_i}{\Delta x_i}\right),
$$

其中 $\tau=0.99$ 是 safety factor。

---

## 练习 B：写 step_length 函数

在 `solver.py` 里添加：

```python
def step_length(x, dx, tau=0.99):
    """
    Return the largest alpha in (0, 1] such that x + alpha * dx stays positive.
    """
    # TODO: write this yourself
    pass
```

提示：

```python
idx = dx < 0
if not np.any(idx):
    return 1.0
return min(1.0, tau * np.min(-x[idx] / dx[idx]))
```

---

### 测试 step_length

在 `test_solver.py` 里：

```python
import numpy as np
from solver import step_length

x = np.array([1., 2., 3.])
dx = np.array([-0.5, 1., -6.])

alpha = step_length(x, dx)
print(alpha)
```

运行：

```bash
python test_solver.py
```

你应该得到：

```text
0.495
```

因为第三个分量限制最严格：

$$
3 + \alpha(-6) > 0
$$

所以：

$$
\alpha < 0.5.
$$

乘上 safety factor $0.99$ 后：

$$
0.99 \times 0.5 = 0.495.
$$

---

## 7. Newton direction 和 Cholesky

Interior point method 每一步要求解 perturbed KKT system。

线性化后：

$$
A\Delta x = -r_p,
$$

$$
A^T\Delta y + \Delta s = -r_d,
$$

$$
S\Delta x + X\Delta s = -r_c + \sigma \mu e.
$$

其中：

$$
X = \operatorname{diag}(x), \quad S = \operatorname{diag}(s).
$$

从第二行：

$$
\Delta s = -r_d - A^T\Delta y.
$$

代入第三行：

$$
S\Delta x + X(-r_d - A^T\Delta y) = -r_c + \sigma \mu e.
$$

得到：

$$
\Delta x = S^{-1}(-r_c + \sigma\mu e + Xr_d) + S^{-1}X A^T \Delta y.
$$

定义：

$$
D = S^{-1}X = \operatorname{diag}(x/s).
$$

代入第一行：

$$
ADA^T \Delta y = -r_p - A S^{-1}(-r_c + \sigma\mu e + Xr_d).
$$

这里：

$$
M = ADA^T
$$

在 $A$ 满行秩且 $x,s>0$ 时是 symmetric positive definite，可以用 Cholesky 分解：

$$
M = LL^T.
$$

---

## 练习 C：写 compute_direction 函数

在 `solver.py` 里添加：

```python
def compute_direction(A, b, c, x, y, s, sigma=0.1):
    """
    Compute Newton direction dx, dy, ds using the normal equation
    and Cholesky factorization.
    """
    # TODO: write this yourself
    pass
```

你需要完成这些步骤：

1. 调用 `residuals` 得到 `rp, rd, rc, mu`；
2. 设置 `e = np.ones(n)`；
3. 计算 `D = x / s`；
4. 计算 normal equation matrix：

```python
AD = A * D
M = AD @ A.T
```

注意：`A * D` 会把 `A` 的第 `j` 列乘以 `D[j]`。

5. 计算右端项：

```python
w = (-rc + sigma * mu * e + x * rd) / s
rhs = -rp - A @ w
```

6. Cholesky 求解：

```python
L = np.linalg.cholesky(M)
dy = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
```

7. 回代：

```python
dx = w + D * (A.T @ dy)
ds = -rd - A.T @ dy
```

8. 返回：

```python
return dx, dy, ds
```

---

### 测试 compute_direction

在 `test_solver.py` 里：

```python
import numpy as np
from solver import residuals, compute_direction

A = np.array([[1., 1.]])
b = np.array([1.])
c = np.array([1., 2.])

x = np.array([0.5, 0.5])
y = np.array([0.])
s = np.array([1., 2.])

sigma = 0.1

dx, dy, ds = compute_direction(A, b, c, x, y, s, sigma=sigma)
rp, rd, rc, mu = residuals(A, b, c, x, y, s)

n = len(x)
e = np.ones(n)

print("dx =", dx)
print("dy =", dy)
print("ds =", ds)

print("check primal equation:", np.linalg.norm(A @ dx + rp))
print("check dual equation:", np.linalg.norm(A.T @ dy + ds + rd))
print("check complementarity equation:", np.linalg.norm(s * dx + x * ds + rc - sigma * mu * e))
```

三个 check 的结果都应该非常接近 0，比如：

```text
1e-16
```

如果不是，最常见的问题是：

- `rhs` 符号写反；
- `dx` 的回代符号写反；
- `ds = -rd - A.T @ dy` 写错；
- `A * D` 写成了 `D * A.T` 之类的维度错误。

---

## 8. 写第一个完整 solve_lp 主循环

先写一个很基础的版本，固定使用：

```python
sigma = 0.1
```

在 `solver.py` 里添加：

```python
def solve_lp(A, b, c, max_iter=50, tol=1e-8, verbose=True):
    """
    Solve min c^T x subject to A x = b, x >= 0
    using a simple primal-dual interior point method.
    """
    m, n = A.shape

    x = np.ones(n)
    y = np.zeros(m)
    s = np.ones(n)

    for k in range(max_iter):
        rp, rd, rc, mu = residuals(A, b, c, x, y, s)

        primal_res = np.linalg.norm(rp)
        dual_res = np.linalg.norm(rd)
        obj = c @ x

        if verbose:
            print(k, "obj=", obj, "primal=", primal_res, "dual=", dual_res, "mu=", mu)

        if primal_res < tol and dual_res < tol and mu < tol:
            break

        dx, dy, ds = compute_direction(A, b, c, x, y, s, sigma=0.1)

        alpha_p = step_length(x, dx)
        alpha_d = step_length(s, ds)

        x = x + alpha_p * dx
        y = y + alpha_d * dy
        s = s + alpha_d * ds

    return x, y, s
```

---

### 测试完整 solver

在 `test_solver.py` 里：

```python
import numpy as np
from solver import solve_lp

A = np.array([[1., 1.]])
b = np.array([1.])
c = np.array([1., 2.])

x, y, s = solve_lp(A, b, c)

print("final x =", x)
print("final objective =", c @ x)
print("final y =", y)
print("final s =", s)
```

运行：

```bash
python test_solver.py
```

你希望看到：

```text
final x ≈ [1. 0.]
final objective ≈ 1.0
```

注意：interior point method 通常不会真的得到精确的 0，而是得到类似：

```text
[9.99999999e-01 1.23456789e-09]
```

这是正常的。

---

## 9. 常见 Python / NumPy 错误

### 9.1 忘记导入 numpy

错误：

```text
NameError: name 'np' is not defined
```

原因：你用了 `np.array`，但没有写：

```python
import numpy as np
```

---

### 9.2 矩阵维度不匹配

错误：

```text
ValueError: matmul: Input operand 1 has a mismatch in its core dimension
```

检查：

```python
print(A.shape)
print(x.shape)
print(y.shape)
print(s.shape)
```

对于标准型 LP：

```text
A.shape = (m, n)
x.shape = (n,)
y.shape = (m,)
s.shape = (n,)
b.shape = (m,)
c.shape = (n,)
```

---

### 9.3 Cholesky 失败

错误：

```text
numpy.linalg.LinAlgError: Matrix is not positive definite
```

可能原因：

1. `A` 不是 full row rank；
2. `x` 或 `s` 出现了非正数；
3. normal equation 写错了；
4. 问题太病态；
5. 初始化太差。

初学时先用很小、很干净的问题测试。

---

## 10. 每次写代码的推荐流程

不要一次写完全部。建议这样做：

1. 写 `residuals`；
2. 只测试 `residuals`；
3. 写 `step_length`；
4. 只测试 `step_length`；
5. 写 `compute_direction`；
6. 只检查 Newton equations；
7. 最后再写 `solve_lp` 主循环。

这比一口气写完整 solver 容易 debug 很多。

---

## 11. 第一阶段 checklist

完成后你应该能回答：

- `python3 -m venv .venv` 是干什么的？
- 为什么每次开新 Terminal 要 `source .venv/bin/activate`？
- `def` 如何定义函数？
- NumPy 里怎么写向量、矩阵？
- `A @ x` 和 `x * s` 有什么区别？
- `A.T @ y + s - c` 对应哪个 residual？
- 为什么 step length 要检查 `dx < 0`？
- 为什么 normal equation 是 $ADA^T \Delta y = rhs$？
- 为什么可以用 Cholesky？

---

## 12. 下一阶段可以升级什么？

等这个基础版本跑通之后，可以继续做：

1. 加入 Mehrotra predictor-corrector；
2. 改进初始化；
3. 支持 inequality constraints；
4. 支持 general bound；
5. 用 scipy sparse matrix；
6. 对比 `scipy.optimize.linprog`；
7. 画 convergence history；
8. 研究为什么实际 LP solver 不直接形成 dense normal equation。

