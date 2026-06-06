# 从 CPU Cholesky 到多 GPU Dense Cholesky 的分步练习

这份笔记的目标不是让你复制一个完整的多 GPU solver，而是帮助你**不靠 AI，一步一步把已经写好的 IPM solver 升级成可以把 dense normal equation 分块并分配到不同 GPU 上做 Cholesky decomposition 的版本**。

你现在已经有一个基础 primal-dual interior point LP solver。它每次迭代会形成 normal equation：

$$
M \Delta y = rhs,
$$

其中：

$$
M = A D A^T,
$$

$$
D = \operatorname{diag}(x/s).
$$

当 $A$ 满行秩且 $x,s>0$ 时，$M$ 是 symmetric positive definite，所以可以做：

$$
M = LL^T.
$$

这份练习只关注一件事：

> 把当前 `np.linalg.cholesky(M)` 逐步替换成你自己能理解、能测试、最后能迁移到多 GPU 的 dense blocked Cholesky 求解器。

---

## 0. 先明确路线

不要一开始就写多 GPU。推荐路线是：

1. 先把 IPM 里的 linear solve 抽象出来；
2. 写 CPU reference test，保证任何新 solver 都能和 NumPy 对齐；
3. 写 CPU blocked Cholesky；
4. 写 CPU tiled matrix 数据结构；
5. 写单 GPU tile Cholesky；
6. 写单 GPU blocked Cholesky；
7. 写多 GPU tile 分配；
8. 写多 GPU blocked Cholesky；
9. 接回 IPM 主循环；
10. 做 correctness、performance、memory 三类测试。

这条路线的关键思想是：

- 每一步都只改变一个东西；
- 每一步都有可以打印和检查的数值；
- GPU 代码永远先和 CPU reference 对齐；
- 多 GPU 之前，先把单 GPU 版本跑稳。

---

## 1. 先把 Cholesky solve 从 IPM 里抽出来

你现在的 `compute_direction` 里大概有这样的代码：

```python
L = np.linalg.cholesky(M)
dy = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
```

第一步不是写 GPU，而是把它抽成函数。

---

## 练习 A：写 dense_cholesky_solve 函数

在 `solver.py` 或新的 `linear_solver.py` 里写：

```python
def dense_cholesky_solve(M, rhs):
    """
    Solve M x = rhs for symmetric positive definite dense M.

    Parameters
    ----------
    M : array, shape (m, m)
        Symmetric positive definite matrix.
    rhs : array, shape (m,)
        Right hand side.

    Returns
    -------
    x : array, shape (m,)
        Solution of M x = rhs.
    """
    # TODO: use np.linalg.cholesky and two triangular solves
    pass
```

提示：

```python
L = np.linalg.cholesky(M)
z = np.linalg.solve(L, rhs)
x = np.linalg.solve(L.T, z)
return x
```

然后在 `compute_direction` 里改成：

```python
dy = dense_cholesky_solve(M, rhs)
```

---

### 测试 dense_cholesky_solve

单独创建一个 SPD 矩阵：

```python
import numpy as np
from linear_solver import dense_cholesky_solve

np.random.seed(0)

B = np.random.randn(5, 5)
M = B @ B.T + 1e-3 * np.eye(5)
rhs = np.random.randn(5)

x = dense_cholesky_solve(M, rhs)

print("residual =", np.linalg.norm(M @ x - rhs))
print("numpy =", np.linalg.norm(x - np.linalg.solve(M, rhs)))
```

你希望看到：

```text
residual ≈ 1e-12
numpy ≈ 1e-12
```

这一关过了，说明你已经把 IPM 和 linear algebra 后端分离了。

---

## 2. 理解 dense blocked Cholesky

普通 Cholesky 可以看成一次处理一列。Blocked Cholesky 一次处理一个矩阵块。

假设矩阵按 block size `bs` 分成很多 tile：

```text
M =
[
  M00 M01 M02
  M10 M11 M12
  M20 M21 M22
]
```

我们只存 lower triangular part：

```text
M =
[
  M00
  M10 M11
  M20 M21 M22
]
```

Blocked Cholesky 第 `k` 步做三件事：

1. factor diagonal block：

$$
M_{kk} = L_{kk}L_{kk}^T
$$

2. triangular solve panel：

$$
L_{ik}L_{kk}^T = M_{ik}, \quad i > k
$$

3. update trailing matrix：

$$
M_{ij} \leftarrow M_{ij} - L_{ik}L_{jk}^T, \quad i,j > k.
$$

对应 LAPACK / BLAS 里的名字通常是：

- `potrf`：Cholesky factorization；
- `trsm`：triangular solve；
- `syrk`：symmetric rank-k update；
- `gemm`：matrix multiplication update。

多 GPU 版本本质上就是把这些 tile operation 分配到不同 GPU 上执行。

---

## 练习 B：先写 CPU blocked_cholesky

新建一个函数：

```python
def blocked_cholesky_cpu(M, block_size):
    """
    Compute lower Cholesky factor L using blocked algorithm.

    This function may overwrite a copy of M, but should not modify
    the user's input matrix.
    """
    # TODO
    pass
```

推荐先假设：

- `M` 是 square matrix；
- `M` 是 dense NumPy array；
- `M.shape[0]` 可以不是 `block_size` 的整数倍；
- 函数返回完整 dense lower triangular matrix `L`。

你需要写出这样的循环：

```python
n = M.shape[0]
A = M.copy()

for k0 in range(0, n, block_size):
    k1 = min(k0 + block_size, n)

    # 1. factor A[k0:k1, k0:k1]

    # 2. solve blocks below the diagonal block

    # 3. update trailing blocks

return np.tril(A)
```

第 1 步：

```python
A[k0:k1, k0:k1] = np.linalg.cholesky(A[k0:k1, k0:k1])
```

第 2 步要解：

$$
L_{ik}L_{kk}^T = A_{ik}.
$$

在 NumPy 里可以先写成：

```python
Lkk = A[k0:k1, k0:k1]
for i0 in range(k1, n, block_size):
    i1 = min(i0 + block_size, n)
    A[i0:i1, k0:k1] = np.linalg.solve(Lkk, A[i0:i1, k0:k1].T).T
```

第 3 步：

```python
for i0 in range(k1, n, block_size):
    i1 = min(i0 + block_size, n)
    Lik = A[i0:i1, k0:k1]

    for j0 in range(k1, i1, block_size):
        # 这里先不要写，思考为什么这个 loop 有问题
        pass
```

上面的 loop 故意不完整。你真正需要的是更新 lower triangular trailing blocks：

```python
for i0 in range(k1, n, block_size):
    i1 = min(i0 + block_size, n)
    Lik = A[i0:i1, k0:k1]

    for j0 in range(k1, i0 + 1, block_size):
        j1 = min(j0 + block_size, n)
        Ljk = A[j0:j1, k0:k1]
        A[i0:i1, j0:j1] -= Lik @ Ljk.T
```

注意：

- 只更新 lower triangular blocks；
- 当 `i0 == j0` 时，这是 diagonal block update；
- 当 `i0 > j0` 时，这是 off-diagonal block update。

---

### 测试 blocked_cholesky_cpu

```python
import numpy as np
from linear_solver import blocked_cholesky_cpu

np.random.seed(1)

for n in [1, 2, 5, 16, 31, 64]:
    for bs in [1, 2, 4, 8, 16]:
        B = np.random.randn(n, n)
        M = B @ B.T + 1e-2 * np.eye(n)

        L = blocked_cholesky_cpu(M, bs)

        err_factor = np.linalg.norm(L @ L.T - M) / np.linalg.norm(M)
        err_numpy = np.linalg.norm(L - np.linalg.cholesky(M)) / np.linalg.norm(np.linalg.cholesky(M))

        print("n=", n, "bs=", bs, "factor_err=", err_factor, "numpy_err=", err_numpy)
```

你希望：

```text
factor_err < 1e-10
numpy_err < 1e-10
```

如果失败，先打印：

```python
print("M shape:", M.shape)
print("block:", k0, k1)
print("diag eig min:", np.linalg.eigvalsh(A[k0:k1, k0:k1]).min())
```

---

## 3. 写 triangular solve

Cholesky factorization 之后，还要解：

$$
LL^T x = rhs.
$$

如果你的 factorization 返回 `L`，可以先用 NumPy：

```python
z = np.linalg.solve(L, rhs)
x = np.linalg.solve(L.T, z)
```

但是为了之后上 GPU，你应该先写 block triangular solve。

---

## 练习 C：写 blocked_forward_solve_cpu

目标是解：

$$
Lz = rhs.
$$

函数接口：

```python
def blocked_forward_solve_cpu(L, rhs, block_size):
    """
    Solve L z = rhs, where L is lower triangular.
    """
    # TODO
    pass
```

思路：

```python
n = L.shape[0]
z = rhs.copy()

for k0 in range(0, n, block_size):
    k1 = min(k0 + block_size, n)

    z[k0:k1] = np.linalg.solve(L[k0:k1, k0:k1], z[k0:k1])

    for i0 in range(k1, n, block_size):
        i1 = min(i0 + block_size, n)
        z[i0:i1] -= L[i0:i1, k0:k1] @ z[k0:k1]

return z
```

再写 backward solve：

```python
def blocked_backward_solve_cpu(L, z, block_size):
    """
    Solve L.T x = z.
    """
    # TODO
    pass
```

提示：方向反过来，从最后一个 block 往前走。

---

### 测试 blocked solve

```python
np.random.seed(2)

n = 50
bs = 8
B = np.random.randn(n, n)
M = B @ B.T + 1e-2 * np.eye(n)
rhs = np.random.randn(n)

L = blocked_cholesky_cpu(M, bs)
z = blocked_forward_solve_cpu(L, rhs, bs)
x = blocked_backward_solve_cpu(L, z, bs)

print(np.linalg.norm(M @ x - rhs) / np.linalg.norm(rhs))
```

你希望看到：

```text
< 1e-10
```

---

## 4. 给 IPM 加一个 linear_solver 参数

现在可以把 IPM solver 改成支持不同 linear solver。

---

## 练习 D：让 compute_direction 接收 solver 函数

把：

```python
dy = dense_cholesky_solve(M, rhs)
```

改成：

```python
dy = linear_solver(M, rhs)
```

函数接口可以变成：

```python
def compute_direction(A, b, c, x, y, s, sigma=0.1, linear_solver=None):
    if linear_solver is None:
        linear_solver = dense_cholesky_solve

    ...
    dy = linear_solver(M, rhs)
    ...
```

然后 `solve_lp` 也接收：

```python
def solve_lp(A, b, c, max_iter=50, tol=1e-8, verbose=True, linear_solver=None):
    ...
    dx, dy, ds = compute_direction(
        A, b, c, x, y, s,
        sigma=0.1,
        linear_solver=linear_solver,
    )
```

---

### 测试 IPM 不变

先用默认 solver：

```python
x, y, s = solve_lp(A, b, c)
```

再用 blocked solver：

```python
def my_solver(M, rhs):
    L = blocked_cholesky_cpu(M, block_size=16)
    z = blocked_forward_solve_cpu(L, rhs, block_size=16)
    return blocked_backward_solve_cpu(L, z, block_size=16)

x2, y2, s2 = solve_lp(A, b, c, linear_solver=my_solver)
```

检查：

```python
print("x diff =", np.linalg.norm(x - x2))
print("obj diff =", abs(c @ x - c @ x2))
```

你希望两者非常接近。

---

## 5. 选择 GPU 工具

这一阶段开始进入 GPU。不要急着做多 GPU。先选一个你能独立安装、能跑小例子的工具。

常见选择：

- `CuPy`：Python 里最像 NumPy，适合从现有代码迁移；
- `PyTorch`：也能做 GPU matrix operation，但线性代数接口不是专门为 solver 后端设计；
- `CUDA C++ + cuBLAS/cuSOLVER`：更接近生产级，但学习曲线更高；
- `MAGMA` / `SLATE` / `cuSOLVERMg`：更接近多 GPU dense linear algebra，但要先理解 blocked algorithm。

如果你的目标是从 Python IPM 慢慢走到多 GPU，建议顺序是：

1. `NumPy` CPU reference；
2. `CuPy` 单 GPU prototype；
3. `CuPy` 多 GPU data movement prototype；
4. 再研究 CUDA C++ / cuSOLVERMg / MAGMA / SLATE。

---

## 练习 E：确认单 GPU 能工作

先不要写 Cholesky。只确认 GPU array 能创建、能矩阵乘法、能传回 CPU。

```python
import cupy as cp

x = cp.ones(10)
y = cp.ones(10)

print(cp.dot(x, y))
print(cp.cuda.runtime.getDeviceCount())
```

如果这一步失败，不要继续写 solver。先解决：

- CUDA driver；
- CUDA toolkit；
- CuPy wheel 版本；
- GPU 是否被系统识别；
- Python 环境是否正确。

---

## 6. 写单 GPU dense Cholesky solver

---

## 练习 F：写 cupy_cholesky_solve

函数接口：

```python
def cupy_cholesky_solve(M, rhs):
    """
    Solve M x = rhs using one GPU.
    M and rhs are NumPy arrays.
    Return a NumPy array.
    """
    # TODO
    pass
```

思路：

```python
import cupy as cp

Mg = cp.asarray(M)
rhsg = cp.asarray(rhs)

L = cp.linalg.cholesky(Mg)
z = cp.linalg.solve(L, rhsg)
xg = cp.linalg.solve(L.T, z)

return cp.asnumpy(xg)
```

---

### 测试 cupy_cholesky_solve

用第 1 节同样的 SPD test：

```python
x_gpu = cupy_cholesky_solve(M, rhs)
x_cpu = dense_cholesky_solve(M, rhs)

print(np.linalg.norm(x_gpu - x_cpu) / np.linalg.norm(x_cpu))
print(np.linalg.norm(M @ x_gpu - rhs) / np.linalg.norm(rhs))
```

你希望：

```text
< 1e-8
```

GPU 和 CPU 的 floating point 结果不需要 bitwise identical。

---

## 7. 写 tile 数据结构

多 GPU 的关键不是一开始就分解，而是先能表达：

> 第 `(i, j)` 个 tile 在哪张 GPU 上？

---

## 练习 G：写 tile_ranges

```python
def tile_ranges(n, block_size):
    """
    Return [(start, end), ...] for block partition.
    """
    # TODO
    pass
```

期望：

```python
print(tile_ranges(10, 4))
```

输出：

```text
[(0, 4), (4, 8), (8, 10)]
```

---

## 练习 H：写 owner_gpu

先用最简单的 2D block cyclic 分配。

如果有 `num_gpus` 张 GPU，可以写：

```python
def owner_gpu(i_tile, j_tile, num_gpus):
    return (i_tile + j_tile) % num_gpus
```

测试：

```python
for i in range(4):
    for j in range(4):
        if i >= j:
            print((i, j), "-> gpu", owner_gpu(i, j, 2))
```

你应该能画出 lower triangular tile 的归属。

---

## 8. 写多 GPU tile container

这一步仍然不做 Cholesky，只做数据放置。

---

## 练习 I：把 CPU matrix 分发到 GPU tiles

目标接口：

```python
def scatter_lower_tiles_to_gpus(M, block_size, num_gpus):
    """
    Split lower triangular part of M into tiles.
    Put each tile on its owner GPU.
    Return a dictionary:

        tiles[(i_tile, j_tile)] = gpu_array
    """
    # TODO
    pass
```

思路：

```python
tiles = {}
ranges = tile_ranges(M.shape[0], block_size)

for i, (i0, i1) in enumerate(ranges):
    for j, (j0, j1) in enumerate(ranges):
        if i < j:
            continue

        gpu = owner_gpu(i, j, num_gpus)
        with cp.cuda.Device(gpu):
            tiles[(i, j)] = cp.asarray(M[i0:i1, j0:j1])

return tiles
```

---

### 测试 scatter

```python
tiles = scatter_lower_tiles_to_gpus(M, block_size=16, num_gpus=2)

for key, value in tiles.items():
    print(key, value.device, value.shape)
```

检查：

- tile 数量是否正确；
- diagonal tile 是否存在；
- lower triangular tile 是否存在；
- upper triangular tile 是否没有存；
- 每个 tile 的 shape 是否正确；
- tile 的 device 是否符合 `owner_gpu`。

---

## 9. 多 GPU Cholesky 的核心依赖关系

Blocked Cholesky 第 `k` 步：

```text
potrf(k, k)
trsm(i, k), for i > k
update(i, j, k), for i >= j > k
```

依赖关系是：

```text
trsm(i, k) depends on potrf(k, k)
update(i, j, k) depends on trsm(i, k) and trsm(j, k)
potrf(k+1, k+1) depends on all updates to (k+1, k+1)
```

你可以先写同步版本：

```python
for k in range(num_tiles):
    potrf(k, k)
    synchronize_all_gpus()

    for i in range(k + 1, num_tiles):
        trsm(i, k)
    synchronize_all_gpus()

    for i in range(k + 1, num_tiles):
        for j in range(k + 1, i + 1):
            update(i, j, k)
    synchronize_all_gpus()
```

这个版本性能不是最优，但最容易 debug。

等正确性稳定后，再考虑 streams、overlap、look-ahead。

---

## 练习 J：先写伪多 GPU版本

在真正移动 GPU 数据之前，先用 NumPy 模拟多 GPU。

目标：

- tile dictionary；
- `owner_gpu`；
- blocked Cholesky；
- 每一步打印哪个 tile 由哪个 GPU 负责；
- 数值结果仍然由 NumPy 计算。

伪代码：

```python
def simulated_multigpu_cholesky(M, block_size, num_gpus):
    tiles = scatter_lower_tiles_cpu(M, block_size)

    for k in range(num_tiles):
        print("potrf", (k, k), "gpu", owner_gpu(k, k, num_gpus))

        for i in range(k + 1, num_tiles):
            print("trsm", (i, k), "gpu", owner_gpu(i, k, num_gpus))

        for i in range(k + 1, num_tiles):
            for j in range(k + 1, i + 1):
                print("update", (i, j), "gpu", owner_gpu(i, j, num_gpus))
```

这一步的目的不是快，而是确保你知道：

- 哪些任务可以并行；
- 哪些任务必须等前一步；
- 哪些 tile 需要跨 GPU 传输。

---

## 10. 真正写多 GPU tile operation

多 GPU 的困难点通常不是 `potrf`，而是：

- `trsm(i, k)` 需要拿到 `Lkk`；
- `update(i, j, k)` 需要拿到 `Lik` 和 `Ljk`；
- 这些 tile 可能在不同 GPU；
- 你需要决定复制、移动、还是重新分配。

初学时使用最简单策略：

> 谁负责输出 tile，就把输入 tile 复制到谁的 GPU。

例如 `update(i, j, k)` 输出 tile 是 `(i, j)`，它的 owner 是：

```python
gpu_out = owner_gpu(i, j, num_gpus)
```

那么就把 `Lik` 和 `Ljk` 都复制到 `gpu_out` 上，然后做：

```python
tiles[(i, j)] -= Lik_on_gpu_out @ Ljk_on_gpu_out.T
```

这个策略会有很多数据传输，但非常容易验证。

---

## 练习 K：写 copy_tile_to_gpu

```python
def copy_tile_to_gpu(tile, gpu):
    """
    Return a copy of tile on target gpu.
    """
    # TODO
    pass
```

思路：

```python
with cp.cuda.Device(gpu):
    return cp.array(tile, copy=True)
```

然后每个 operation 都先确保输入 tile 在当前 GPU。

---

## 练习 L：写 multigpu_cholesky_sync

目标接口：

```python
def multigpu_cholesky_sync(M, block_size, num_gpus):
    """
    Synchronous multi-GPU blocked Cholesky prototype.

    Input M is NumPy.
    Return L as NumPy.
    """
    # TODO
    pass
```

建议顺序：

1. `scatter_lower_tiles_to_gpus`；
2. loop over `k`；
3. `potrf(k, k)`；
4. copy `Lkk` to GPUs that need panel solve；
5. `trsm(i, k)`；
6. synchronize；
7. `update(i, j, k)`；
8. synchronize；
9. gather tiles back to CPU；
10. reconstruct dense lower triangular `L`。

每完成一个 stage，就用小矩阵测试：

```python
for n in [8, 17, 32]:
    for bs in [4, 8]:
        for num_gpus in [1, 2]:
            ...
```

检查：

```python
err = np.linalg.norm(L @ L.T - M) / np.linalg.norm(M)
print(err)
```

先接受：

```text
err < 1e-8
```

---

## 11. 接回 IPM solver

当 `multigpu_cholesky_sync` 能通过 SPD matrix test 后，再写：

```python
def multigpu_cholesky_solve(M, rhs, block_size=512, num_gpus=2):
    L = multigpu_cholesky_sync(M, block_size, num_gpus)
    z = np.linalg.solve(L, rhs)
    x = np.linalg.solve(L.T, z)
    return x
```

注意：这里 triangular solve 暂时还在 CPU 上。这样做是故意的：

- 先验证 factorization 正确；
- 再验证 IPM 能跑；
- 最后再把 triangular solve 也搬到 GPU。

---

## 练习 M：让 IPM 使用多 GPU solver

```python
def gpu_solver(M, rhs):
    return multigpu_cholesky_solve(
        M,
        rhs,
        block_size=512,
        num_gpus=2,
    )

x, y, s = solve_lp(A, b, c, linear_solver=gpu_solver)
```

检查：

```python
rp, rd, rc, mu = residuals(A, b, c, x, y, s)
print("primal =", np.linalg.norm(rp))
print("dual =", np.linalg.norm(rd))
print("mu =", mu)
print("obj =", c @ x)
```

如果 IPM 失败，先不要怀疑整个算法。按顺序排查：

1. `M` 是否 SPD；
2. `multigpu_cholesky_sync(M)` 是否满足 `L @ L.T ≈ M`；
3. `multigpu_cholesky_solve(M, rhs)` 是否满足 `M @ dy ≈ rhs`；
4. `compute_direction` 的 Newton equation check 是否接近 0；
5. `step_length` 后 `x` 和 `s` 是否仍然为正。

---

## 12. 性能测试不要太早做

正确性没稳定之前，不要看速度。GPU 版本一开始很可能比 CPU 慢，因为：

- Python 调度 overhead 大；
- tile 太小；
- CPU 和 GPU 来回复制；
- 多 GPU 同步太多；
- 没有 stream overlap；
- 没有复用已经复制过的 panel tile。

先做 correctness，再做 performance。

---

## 练习 N：写 benchmark

写一个只测试 linear solver 的 benchmark：

```python
import time
import numpy as np

def make_spd(n, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    return B @ B.T + 1e-2 * np.eye(n)

def benchmark_solver(solver, n, repeats=3):
    M = make_spd(n)
    rhs = np.random.randn(n)

    # warmup
    solver(M, rhs)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        x = solver(M, rhs)
        t1 = time.perf_counter()
        err = np.linalg.norm(M @ x - rhs) / np.linalg.norm(rhs)
        times.append(t1 - t0)

    return min(times), err
```

测试：

```python
for n in [512, 1024, 2048, 4096]:
    t, err = benchmark_solver(gpu_solver, n)
    print(n, t, err)
```

记录：

- matrix size；
- block size；
- GPU 数量；
- residual error；
- factorization time；
- solve time；
- CPU-GPU transfer time；
- total IPM iteration time。

---

## 13. 优化顺序

正确性通过后，再按这个顺序优化：

1. 调整 `block_size`；
2. 避免每一步重复复制同一个 tile；
3. 把 triangular solve 也放到 GPU；
4. 使用 CUDA streams；
5. 对 panel factorization 做 look-ahead；
6. 减少全局 synchronize；
7. 改成 2D process grid 分配；
8. 用专业库替换 Python prototype。

不要一开始就做 streams。先用同步版本把依赖关系跑通。

---

## 14. 为什么实际 solver 很少直接做 dense normal equation

这个项目很适合学习，但你也要知道限制。

如果 $A$ 很大，形成：

$$
M = A D A^T
$$

会带来几个问题：

- $M$ 可能非常 dense；
- memory 是 $O(m^2)$；
- Cholesky 时间是 $O(m^3)$；
- normal equation 会放大 condition number；
- 稀疏 LP 通常更适合 sparse factorization 或 iterative methods。

所以这个目标更像是：

> 学会 dense linear algebra、blocked algorithm、多 GPU 数据分布和 IPM backend abstraction。

这本身非常有价值，但不要误以为 dense multi-GPU Cholesky 一定是所有 LP 的最佳路线。

---

## 15. 每一步的推荐检查表

写代码时，每完成一个函数都问自己：

- 输入 shape 是什么？
- 输出 shape 是什么？
- 是否修改了用户传入的矩阵？
- 是否处理了最后一个不完整 block？
- 是否只存 lower triangular part？
- 是否和 NumPy reference 对齐？
- residual 是否小于阈值？
- GPU array 是否在预期 device 上？
- CPU-GPU copy 是否发生在我预期的位置？
- 失败时能不能打印出当前 tile index？

---

## 16. 第二阶段 checklist

完成后你应该能回答：

- 为什么 IPM 里要求解 $ADA^T \Delta y = rhs$？
- 为什么 $ADA^T$ 可以做 Cholesky？
- dense Cholesky 的 `potrf`、`trsm`、`syrk`、`gemm` 分别是什么？
- blocked Cholesky 的依赖关系是什么？
- 为什么多 GPU 版本要先设计 tile ownership？
- 为什么同步版本更容易 debug？
- 为什么一开始可以让 triangular solve 留在 CPU？
- 如何检查 `L @ L.T` 是否还原了 `M`？
- 如何检查 linear solver 是否满足 `M @ x = rhs`？
- 如何检查 IPM 的 Newton equation？
- 为什么 GPU 版本可能一开始比 CPU 慢？

---

## 17. 不靠 AI 的学习方法

如果你想独立完成，不要让 AI 直接写代码。建议这样做：

1. 每次只实现一个函数；
2. 先写函数 docstring；
3. 先写 shape print；
4. 先写最小测试；
5. 再写真实实现；
6. 和 NumPy reference 比较；
7. 把失败 case 记录到 README；
8. 每天只优化一个 bottleneck。

遇到问题时，优先查：

- NumPy / CuPy 官方文档；
- BLAS / LAPACK routine 名字；
- cuBLAS / cuSOLVER 文档；
- MAGMA 或 SLATE 的 Cholesky examples；
- 线性代数教材里的 blocked Cholesky；
- 自己的 residual 和 shape print。

推荐你维护一个调试日志：

```text
date:
matrix size:
block size:
num gpus:
function under test:
expected:
actual:
residual:
next hypothesis:
```

这会比“问一次拿一大段答案”更可靠，也更能真正长到你自己手上。

---

## 18. 最小里程碑

按照下面顺序打勾：

- [ ] IPM 可以通过 `linear_solver` 参数切换后端；
- [ ] `dense_cholesky_solve` 和 NumPy reference 对齐；
- [ ] `blocked_cholesky_cpu` 通过 SPD test；
- [ ] `blocked_forward_solve_cpu` 通过 triangular solve test；
- [ ] `blocked_backward_solve_cpu` 通过 triangular solve test；
- [ ] blocked CPU solver 可以接回 IPM；
- [ ] `cupy_cholesky_solve` 单 GPU 通过 SPD test；
- [ ] `tile_ranges` 正确处理最后一个小 block；
- [ ] `owner_gpu` 能打印清楚 tile 分配；
- [ ] `scatter_lower_tiles_to_gpus` 能把 tile 放到预期 GPU；
- [ ] simulated multi-GPU 能打印正确 dependency order；
- [ ] synchronous multi-GPU factorization 通过 `L @ L.T` test；
- [ ] multi-GPU factorization 接回 IPM；
- [ ] benchmark 记录 CPU、单 GPU、多 GPU 的时间和 residual；
- [ ] 写出下一轮优化计划。
