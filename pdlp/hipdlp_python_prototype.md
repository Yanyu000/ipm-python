# HiPDLP Python Prototype Guide

这份文档的目标是帮助你先用 Python 写一个足够清楚、可测试、可替换后端的 HiPDLP 原型。重点不是一开始追求最快，而是把接口边界设计好：算法主循环只依赖线性算子、LP 数据和 heuristic 策略；矩阵存储、CPU/GPU 后端、SpMV/SpMM 实现都可以独立替换。

## 0. 先统一术语

你提到想用 sparse matrix dense vector multiplication 代替 sparse matrix vector product。工程上更建议把目标抽象成：

- `SpMV`: sparse matrix dense vector product，`A @ x`，其中 `x.shape == (n,)`。
- `SpMM`: sparse matrix dense matrix product，`A @ X`，其中 `X.shape == (n, k)`。当 `k == 1` 时它退化为 SpMV。

原型接口应该同时支持 `matvec` 和 `matmat`。这样你可以先用单向量验证算法，再把多个候选方向、多个 restart trial、batched power method 或其它 heuristic 放进 dense RHS matrix。

## 1. 当前代码里的可借鉴点

仓库里现有的 hipdlp 线性代数入口主要是围绕 `HighsLp` 的两个函数：

- [`highs/pdlp/hipdlp/linalg.hpp`](../highs/pdlp/hipdlp/linalg.hpp): `ax(lp, x, result)` 和 `aTy(lp, y, result)`。
- [`highs/pdlp/hipdlp/pdhg.hpp`](../highs/pdlp/hipdlp/pdhg.hpp): `PDLPSolver` 同时持有 LP、iterate、cache、restart、scaling 和 GPU 资源。

ipx 里有一个小而干净的线性算子抽象：

- [`highs/ipm/ipx/linear_operator.h`](../highs/ipm/ipx/linear_operator.h): `LinearOperator::Apply(rhs, lhs, rhs_dot_lhs)`。

Python 原型建议借鉴 ipx 的精神，但扩展成四个矩阵接口：

- `matvec(x)`: `A @ x`
- `rmatvec(y)`: `A.T @ y`
- `matmat(X)`: `A @ X`
- `rmatmat(Y)`: `A.T @ Y`

不要把 LP 上下界、目标、scaling、restart、step-size 放进 `LinearOperator`。它只管矩阵乘法和少量矩阵元信息。

## 2. 推荐项目结构

先不要接入现有 CMake。把原型放到一个独立目录，例如：

```text
prototypes/hipdlp_py/
  pyproject.toml
  src/hipdlp_proto/
    __init__.py
    arrays.py
    data.py
    generators.py
    heuristics.py
    io_highs.py
    operators.py
    pdhg.py
    residuals.py
    scaling.py
    stopping.py
    timing.py
    cli.py
  tests/
    test_highs_io.py
    test_operators.py
    test_generators.py
    test_pdhg_smoke.py
```

如果你想更快起步，可以先写单文件 `prototype.py`，但文件内部仍然按这几个区块组织。等接口稳定后再拆成包。

## 3. Python 环境

最小 CPU 原型依赖：

```toml
[project]
name = "hipdlp-proto"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = [
  "highspy",
  "numpy",
  "scipy",
  "pytest",
  "rich",
]

[project.optional-dependencies]
gpu = [
  "cupy-cuda12x",
]
```

备注：

- `highspy` 用来读取 MPS 并获得 `HighsLp`。
- `scipy.sparse` 是 CPU sparse matrix 的基线。
- `cupy` 是 Python 里最直接的 GPU sparse 路线，因为它提供 `cupyx.scipy.sparse`，接口接近 SciPy。

## 4. 核心数据结构

### 4.1 `LpProblem`

`LpProblem` 保存 LP 语义，不保存算法状态。

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class LpProblem:
    A: "LinearOperator"
    c: np.ndarray
    col_lower: np.ndarray
    col_upper: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    sense: Any = 1
    offset: float = 0.0
    name: str = ""

    @property
    def num_col(self) -> int:
        return self.A.shape[1]

    @property
    def num_row(self) -> int:
        return self.A.shape[0]
```

第一版可以把所有向量都保留为 NumPy 数组。GPU 后端通过 `problem_to_backend(problem, "cupy")` 做一次性迁移，避免算法中频繁 host/device copy。

### 4.2 `IterationState`

算法状态单独放，不要塞回 problem。

```python
@dataclass
class IterationState:
    x: Any
    y: Any
    x_next: Any
    y_next: Any
    ax: Any
    aty: Any
    ax_next: Any
    aty_next: Any
    iteration: int = 0
```

如果要实验 SpMM，把 `x` 和 `y` 的 RHS 维度统一成二维：

- 单 RHS: `x.shape == (n, 1)`，`y.shape == (m, 1)`
- 多 RHS: `x.shape == (n, k)`，`y.shape == (m, k)`

算法外层可以在需要时用 `x[:, 0]` 提取主 iterate。不要在 `LinearOperator` 内部猜测哪个列是主 iterate。

## 5. `LinearOperator` 接口

第一版用 `Protocol` 定义接口，而不是抽象基类。这样 SciPy、CuPy、mock operator 都容易接上。

```python
from typing import Protocol, Any

class LinearOperator(Protocol):
    shape: tuple[int, int]
    dtype: Any
    backend: str

    def matvec(self, x: Any) -> Any:
        ...

    def rmatvec(self, y: Any) -> Any:
        ...

    def matmat(self, X: Any) -> Any:
        ...

    def rmatmat(self, Y: Any) -> Any:
        ...

    def to_backend(self, backend: str) -> "LinearOperator":
        ...
```

建议第一版实现一个具体类：

```python
@dataclass
class SparseMatrixOperator:
    A_csr: Any
    A_csc: Any
    backend: str = "numpy"

    @property
    def shape(self) -> tuple[int, int]:
        return self.A_csr.shape

    @property
    def dtype(self) -> Any:
        return self.A_csr.dtype

    def matvec(self, x):
        return self.A_csr @ x

    def rmatvec(self, y):
        return self.A_csc.T @ y

    def matmat(self, X):
        return self.A_csr @ X

    def rmatmat(self, Y):
        return self.A_csc.T @ Y

    def to_backend(self, backend: str) -> "SparseMatrixOperator":
        if backend == self.backend:
            return self
        if backend == "cupy":
            return to_cupy_operator(self)
        raise ValueError(f"Unsupported backend: {backend}")
```

为什么同时存 CSR 和 CSC：

- `A @ x` 和 `A @ X` 用 CSR 通常更自然。
- `A.T @ y` 和 `A.T @ Y` 如果只用 CSR 的转置视图，底层可能临时转换。保留 CSC 或 `A.T.tocsr()` 可以让转置乘法路径明确。
- GPU 后端上，尽量一次性构造好需要的 sparse descriptor，避免每次迭代转换格式。

## 6. 从 MPS 读取 LP

`highspy` 可以直接读取 MPS：

```python
import highspy

h = highspy.Highs()
status = h.readModel("check/instances/avgas.mps")
lp = h.getLp()
```

把 `HighsLp.a_matrix_` 转成 SciPy sparse 时要看矩阵方向：

```python
import numpy as np
import scipy.sparse as sp
import highspy

def scipy_matrices_from_highs_lp(lp):
    m = int(lp.num_row_)
    n = int(lp.num_col_)
    mat = lp.a_matrix_
    start = np.asarray(mat.start_, dtype=np.int64)
    index = np.asarray(mat.index_, dtype=np.int64)
    value = np.asarray(mat.value_, dtype=np.float64)

    if mat.format_ == highspy.MatrixFormat.kColwise:
        A_csc = sp.csc_array((value, index, start), shape=(m, n))
        return A_csc.tocsr(), A_csc

    if mat.format_ == highspy.MatrixFormat.kRowwise:
        A_csr = sp.csr_array((value, index, start), shape=(m, n))
        return A_csr, A_csr.tocsc()

    raise ValueError(f"Unsupported Highs matrix format: {mat.format_}")
```

然后构造 `LpProblem`：

```python
def read_mps(path: str) -> LpProblem:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(path)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"Could not read model: {path}")

    lp = h.getLp()
    A_csr, A_csc = scipy_matrices_from_highs_lp(lp)
    A = SparseMatrixOperator(A_csr=A_csr, A_csc=A_csc)

    return LpProblem(
        A=A,
        c=np.asarray(lp.col_cost_, dtype=np.float64),
        col_lower=np.asarray(lp.col_lower_, dtype=np.float64),
        col_upper=np.asarray(lp.col_upper_, dtype=np.float64),
        row_lower=np.asarray(lp.row_lower_, dtype=np.float64),
        row_upper=np.asarray(lp.row_upper_, dtype=np.float64),
        sense=lp.sense_,
        offset=float(lp.offset_),
        name=str(lp.model_name_),
    )
```

第一批测试：

- 读取 `check/instances/avgas.mps`。
- 检查 `problem.A.shape == (lp.num_row_, lp.num_col_)`。
- 用随机 `x` 验证 `A.matvec(x)` 和 `A.A_csr.toarray() @ x` 一致。
- 用随机 `Y` 验证 `A.rmatmat(Y)` 和 `A.A_csr.toarray().T @ Y` 一致。

## 7. 自己生成 sparse matrix A

原型需要两类 generator：

### 7.1 只生成矩阵

用于测试 `LinearOperator` 和 benchmark。

```python
def random_sparse_operator(m, n, density, seed=0) -> SparseMatrixOperator:
    rng = np.random.default_rng(seed)
    A = sp.random(
        m,
        n,
        density=density,
        format="csr",
        dtype=np.float64,
        random_state=rng,
    )
    A.data = rng.normal(size=A.nnz)
    A.sum_duplicates()
    A.eliminate_zeros()
    return SparseMatrixOperator(A_csr=A, A_csc=A.tocsc())
```

### 7.2 生成一个有可行点的 LP

用于 PDHG smoke test。思路是先采样 `x0`，再令 row bounds 包住 `A @ x0`。

```python
def random_feasible_lp(m, n, density, seed=0) -> LpProblem:
    rng = np.random.default_rng(seed)
    A = random_sparse_operator(m, n, density, seed)
    x0 = rng.uniform(0.0, 1.0, size=n)
    ax0 = A.matvec(x0)
    width = rng.uniform(0.1, 10.0, size=m)

    return LpProblem(
        A=A,
        c=rng.normal(size=n),
        col_lower=np.zeros(n),
        col_upper=np.full(n, np.inf),
        row_lower=ax0 - width,
        row_upper=ax0 + width,
        name=f"random_feasible_{m}x{n}",
    )
```

不要一开始生成复杂 MIP、free variables、半连续变量。先让 continuous LP 跑通，再逐步扩大覆盖。

## 8. Heuristic 接口

把 heuristic 做成纯策略对象，不直接拥有矩阵或 LP。

```python
class StepSizeRule(Protocol):
    def initialize(self, problem: LpProblem) -> tuple[float, float]:
        ...

    def update(self, stats: "IterationStats") -> tuple[float, float]:
        ...

class RestartRule(Protocol):
    def should_restart(self, stats: "IterationStats") -> bool:
        ...

class ScalingRule(Protocol):
    def apply(self, problem: LpProblem) -> tuple[LpProblem, "ScalingState"]:
        ...

class StoppingRule(Protocol):
    def check(self, stats: "IterationStats") -> "Termination | None":
        ...
```

建议先实现这些最小策略：

- `FixedStepSize`: 手动给定 `tau` 和 `sigma`。
- `PowerMethodStepSize`: 通过 `A.T @ (A @ x)` 估计 `||A||_2`。
- `NoRestart`: 保证主循环最简单。
- `FixedFrequencyRestart`: 每 `k` 次 restart，测试状态迁移。
- `NoScaling`: 第一版避免把 scaling 和 solver bug 混在一起。
- `PockChambolleDiagonalScaling`: 第二阶段再加。

## 9. PDHG 主循环的第一版

第一版只支持 bound projection 和 box row projection，先不追求完全等价现有 C++。

主循环依赖的接口应该只有：

- `problem.A.matvec` / `rmatvec`
- `project_box`
- `step_size_rule`
- `restart_rule`
- `stopping_rule`

伪代码：

```python
def solve(problem, options, hooks=()):
    A = problem.A
    tau, sigma = options.step_size_rule.initialize(problem)
    state = initialize_state(problem)

    for k in range(options.max_iter):
        state.aty = A.rmatvec(state.y)
        x_bar = project_box(
            state.x - tau * (problem.c - state.aty),
            problem.col_lower,
            problem.col_upper,
        )

        reflected_x = 2.0 * x_bar - state.x
        ax_reflected = A.matvec(reflected_x)
        y_bar = project_row_dual(
            state.y + sigma * ax_reflected,
            problem.row_lower,
            problem.row_upper,
            sigma,
        )

        stats = compute_stats(problem, state, x_bar, y_bar, k)
        if options.restart_rule.should_restart(stats):
            x_bar, y_bar = restart_update(state, stats)

        term = options.stopping_rule.check(stats)
        state = advance_state(state, x_bar, y_bar, stats)

        for hook in hooks:
            hook(stats)

        if term is not None:
            return SolverResult(state=state, termination=term, stats=stats)
```

把每个 vector update 写成独立函数，并用小矩阵单元测试。不要在第一版就把 logging、benchmark、GPU、adaptive step 全部塞进 `solve`。

## 10. SpMM 实验怎么接入

当接口支持 `matmat` 后，先不要改整个 PDHG。先找这几类可批处理的计算：

1. Power method: 同时跑多个随机初始向量，`A @ X` 和 `A.T @ Y` 用 SpMM。
2. 多个候选 step-size trial: 同一 iterate 下生成多个 reflected primal，组成 `X_trial`。
3. 多个 restart / averaging candidate: 把 candidate columns 批量评估 fixed-point error。
4. Benchmark-only path: 固定 `A`，比较 `k = 1, 2, 4, 8, 16, 32` 的 `A @ X`。

建议先写一个独立 benchmark：

```python
def benchmark_operator(A, batch_sizes=(1, 2, 4, 8, 16, 32), repeats=20):
    n = A.shape[1]
    for k in batch_sizes:
        X = np.random.default_rng(0).normal(size=(n, k))
        A.matmat(X)  # warmup
        # time repeats here
```

GPU benchmark 必须注意：

- warmup 后再计时；
- 每次计时后 `cupy.cuda.Stream.null.synchronize()`；
- 不要把 host/device transfer 计入 kernel timing，除非你明确想测端到端时间；
- 单 RHS 的 GPU SpMV 未必比 CPU 快，矩阵足够大或 RHS batch 足够大才更可能受益。

## 11. GPU 后端路线

Python 可以做 GPU 原型，但要说准确一点：Python 语言本身不适合写高性能 GPU 内核，Python 生态适合调用高性能 GPU 库。

推荐顺序：

1. CPU SciPy 后端先跑通接口和算法。
2. CuPy 后端复用同一 `LinearOperator` 接口。
3. 发现瓶颈后，用 CuPy RawKernel、Numba CUDA，或直接回到 C++/CUDA 写专用 kernel。
4. 接近生产时，把稳定接口映射回 hipdlp C++ 类，底层继续用 cuSPARSE/cuBLAS 或自写 kernel。

CuPy 后端示意：

```python
def to_cupy_operator(op: SparseMatrixOperator) -> SparseMatrixOperator:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    A_csr = cpsp.csr_matrix(op.A_csr)
    A_csc = cpsp.csc_matrix(op.A_csc)
    return SparseMatrixOperator(A_csr=A_csr, A_csc=A_csc, backend="cupy")
```

关键约束：

- 不要在每次迭代里 `cp.asarray` 或 `.get()`。
- 所有 iterate、bounds、cost、row bounds 都要提前搬到 GPU。
- 如果 heuristic 需要很多标量 diagnostics，可以先只在每隔 `k` 次迭代同步一次。
- 如果多 RHS 宽度很小，SpMM 未必比多次 SpMV 快，必须实测。

## 12. Python 还是 Julia

如果目标是这个仓库的 hipdlp 重构，我建议先用 Python。

原因：

- `highspy` 直接读取 MPS 并暴露 `HighsLp`，和当前 HiGHS 数据结构最贴近。
- SciPy/CuPy 的 sparse API 很适合快速验证 `CSR/CSC + matvec/matmat` 的接口边界。
- Python 原型更容易和现有 benchmark 脚本、测试数据、日志处理共存。
- 未来迁移回 C++/CUDA 时，Python 的类边界可以一一对应到 C++ interface。

Julia 更适合的情况：

- 你打算长期在 Julia 里做算法研究，而不是只做 refactor prototype。
- 你需要写较多自定义 GPU kernel，并希望代码比 C++/CUDA 更短。
- 你愿意把 MPS/HiGHS 数据桥接问题也纳入原型成本。

实用结论：

- 短期原型和接口设计：Python。
- GPU sparse baseline：Python + CuPy，重点验证 SpMM 是否值得。
- 大量自定义 GPU kernel 研究：Julia/CUDA.jl 可以并行试，但不要让它阻塞 Python 原型。
- 生产落地：C++/CUDA，接口从 Python 原型反推。

## 13. 里程碑清单

### Milestone 1: 矩阵接口

- 建 `LinearOperator` protocol。
- 建 `SparseMatrixOperator` CPU 实现。
- 写 `matvec/rmatvec/matmat/rmatmat` 的 dense reference 测试。
- 用随机 sparse matrix 覆盖 `m < n`、`m > n`、空行、空列。

### Milestone 2: MPS 输入

- 写 `read_mps(path) -> LpProblem`。
- 对 `check/instances/avgas.mps` 和 `check/instances/25fv47.mps` 做 smoke test。
- 验证 highspy 的 `num_col_`、`num_row_`、`num_nz` 与 SciPy matrix 一致。

### Milestone 3: 随机 LP generator

- 写只生成矩阵的 generator。
- 写有可行点的 continuous LP generator。
- 保存 seed、shape、density、nnz 到 metadata。

### Milestone 4: 最小 PDHG

- 实现 `NoScaling + FixedStepSize + NoRestart`。
- 只支持 continuous LP。
- 每 `k` 次输出 primal residual、dual residual、objective、gap。
- 对小随机 LP 跑到合理 residual，不要求和 HiGHS 完全一致。

### Milestone 5: Heuristic 插件化

- 加 `PowerMethodStepSize`。
- 加 `FixedFrequencyRestart`。
- 加一个你想实验的新 heuristic。
- 确保新增 heuristic 不需要改 `LinearOperator` 和 `LpProblem`。

### Milestone 6: SpMM

- 把 power method 改成 batched version。
- 写 `A @ X` benchmark。
- 比较 CPU SciPy 和 GPU CuPy 下不同 RHS batch size。
- 记录哪些矩阵族真的受益。

### Milestone 7: GPU 后端

- 写 `to_backend(problem, "cupy")`。
- 保证整个 solve loop 没有隐式 host/device copy。
- 加 GPU smoke test，缺少 CuPy 或 CUDA 时自动 skip。
- 单独计时 matrix multiply、projection、diagnostics。

### Milestone 8: 回迁 C++

- 把 Python 的 `LinearOperator` 对应成 C++ 抽象。
- 把 `SparseMatrixOperator` 对应成 CSR/CSC 或 descriptor-owning 类。
- 把 heuristic protocol 对应成小类或函数对象。
- 把 PDHG 主循环从资源管理里剥离，让 GPU memory 和 cuSPARSE descriptors 由 operator/backend 管理。

## 14. C++ 重构时的目标形状

Python 原型稳定后，C++ 里可以考虑这组边界：

```text
PdlpProblemData
  owns objective, bounds, row types, scaling metadata

LinearOperator
  shape()
  apply(x, y)
  applyTranspose(y, x)
  applyBatch(X, Y)
  applyTransposeBatch(Y, X)

PdhgState
  owns iterates, averages, residual caches

PdhgHeuristics
  step-size rule
  restart rule
  scaling rule
  termination rule

PdhgSolver
  owns loop, logging, stats
  does not know CSR/CSC/cusparse details
```

这个拆分能直接解决当前 `PDLPSolver` 过度持有所有资源的问题，也能给未来的 sparse matrix dense matrix product 留出位置。

## 15. 参考链接

- HiGHS Python interface: <https://ergo-code.github.io/HiGHS/stable/interfaces/python/>
- SciPy sparse arrays: <https://scipy.github.io/devdocs/tutorial/sparse.html>
- CuPy sparse matrices: <https://docs.cupy.dev/en/stable/reference/scipy_sparse.html>
- NVIDIA cuSPARSE: <https://docs.nvidia.com/cuda/cusparse/index.html>
- Julia CUDA.jl cuSPARSE: <https://cuda.juliagpu.org/stable/lib/cusparse/>
