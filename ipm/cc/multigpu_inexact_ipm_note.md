# 多 GPU 分块 Cholesky 与 Inexact IPM/QIPM 的初步研究笔记

> 研究背景：在线性规划（LP）或更一般的凸优化问题中，Interior Point Method（IPM）每一步都需要求解 Newton system。对于大规模、稠密或近似稠密的 normal equation，直接 Cholesky 分解可能成为主要瓶颈。一个自然想法是：把 normal equation 的矩阵分成多个 block，放到多张 GPU 上分别做 dense Cholesky 或 block Cholesky，并用 inexact IPM / QIPM 中允许 Newton direction 近似求解的理论来吸收 GPU 低精度或分块近似带来的误差。

---

## 0. 当前原型审计状态（2026-06-10）

当前仓库中的 Python 原型仍然是 **CPU-only reference implementation**。
它已经可以支持小规模 dense random LP 上的 ExactCholesky 与
AdaptiveRefinement 收敛性检查，也可以用
`experiments/benchmark_newton_systems.py` 独立冻结 Newton normal equation：

\[
M_k = A D_k^2 A^T,\qquad M_k \Delta y = r
\]

并比较 dense Cholesky、完整 blocked dense Cholesky、CG、block-Jacobi PCG、
FP32 simulated Cholesky/refinement、SuperLU sparse direct，以及可选的
CHOLMOD/CuPy 后端。

因此目前可以诚实声称的是：

- CPU 原型验证了若干 linear solver failure modes；
- diagonal-block Cholesky 只是 block-Jacobi preconditioner，必须配合
  CG/refinement；
- 完整 blocked Cholesky 必须包含 off-diagonal coupling 和 Schur-complement
  / trailing updates，才是 direct solver；
- AdaptiveRefinement 在小规模 CPU dense random LP 上能恢复接近 FP64
  Cholesky baseline 的收敛。

目前还不能声称：

- GPU speedup；
- multi-GPU scaling；
- Netlib 或 HIPO production instance 上的加速；
- GPU/block-wise dense Cholesky 已经可以替换 HIPO 的现有线性求解器。

GPU speedup 只能由真实 GPU benchmark，或从 HIPO 导出的 frozen Newton
systems 上的严格同容差 benchmark 支撑。

---

## 1. 问题设定：从 LP 的 Newton system 出发

考虑标准形式线性规划：

\[
\min_x\ c^T x, \quad \text{s.t. } Ax=b,\ x\ge 0,
\]

其对偶为：

\[
\max_y\ b^T y, \quad \text{s.t. } A^T y+s=c,\ s\ge 0.
\]

在 primal-dual IPM 中，每一步需要求解一个 Newton system。常见做法是消元得到 normal equation，例如形式上类似：

\[
A D^2 A^T \Delta y = r,
\]

其中

\[
D = X^{1/2}S^{-1/2},
\]

或者等价地，矩阵为

\[
M = A D^2 A^T.
\]

如果 \(A\) 是大规模稠密矩阵，或者 \(D\) 使得 \(A D^2 A^T\) 变成较难处理的稠密矩阵，那么 Cholesky 分解

\[
M = LL^T
\]

会成为每一步最昂贵的部分。

---

## 2. 多 GPU 分块 Cholesky 的想法

### 2.1 基本动机

假设 normal equation 矩阵可以写成 block matrix：

\[
M =
\begin{bmatrix}
M_{11} & M_{12} & \cdots & M_{1p} \\
M_{21} & M_{22} & \cdots & M_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
M_{p1} & M_{p2} & \cdots & M_{pp}
\end{bmatrix},
\]

其中每个 diagonal block \(M_{ii}\) 可以分配到一张 GPU 上做 dense Cholesky：

\[
M_{ii} \approx L_iL_i^T.
\]

这个方向的核心目标是：

1. **降低单张 GPU 显存压力**：每张 GPU 只存储一部分 block。
2. **增加并行度**：多个 diagonal blocks 可以并行分解。
3. **利用 dense BLAS/GPU tensor core 高吞吐**：如果每个 block 足够大，dense Cholesky、TRSM、SYRK/GEMM 等 kernel 可以有较高效率。
4. **用近似求解代替精确求解**：不一定要求每一步 Newton system 被求得非常精确，而是满足 IPM 理论允许的 residual tolerance。

### 2.2 一个简单的 block Cholesky 结构

对于二分块矩阵：

\[
M =
\begin{bmatrix}
A & B \\
B^T & C
\end{bmatrix},
\]

如果 \(A\) 正定，则 block Cholesky 可以写成：

\[
M =
\begin{bmatrix}
L_A & 0 \\
B^T L_A^{-T} & L_S
\end{bmatrix}
\begin{bmatrix}
L_A^T & L_A^{-1}B \\
0 & L_S^T
\end{bmatrix},
\]

其中

\[
A=L_A L_A^T,
\]

Schur complement 为：

\[
S = C - B^T A^{-1}B,
\]

然后

\[
S = L_S L_S^T.
\]

这说明：即使 diagonal block 可以并行分解，真正困难的是 **off-diagonal coupling** 和 **Schur complement update**。如果只对每个 diagonal block 独立分解，而忽略 off-diagonal block，那么得到的不是完整 Cholesky，而是一个 block Jacobi preconditioner。

---

## 3. 这个想法最自然的定位：不是“精确 direct solver”，而是“低精度 preconditioner + iterative refinement / Krylov correction”

### 3.1 为什么不能简单地各 block 独立 Cholesky？

如果直接把 \(M\) 切成多个 block，并让每张 GPU 分解一个 diagonal block：

\[
P = \operatorname{blkdiag}(M_{11},M_{22},\dots,M_{pp}),
\]

那么 \(P\) 只是 \(M\) 的近似。除非 off-diagonal blocks 很小，否则

\[
P^{-1}r
\]

不是 Newton equation 的准确解。

因此这个方向更合理的解释是：

> 多 GPU dense block Cholesky 生成一个强 preconditioner，然后用 CG / MINRES / iterative refinement 修正误差。

对于 SPD normal equation，可以考虑：

\[
M\Delta y = r.
\]

先用低精度 block Cholesky 近似解：

\[
\Delta y_0 \approx P^{-1}r.
\]

然后计算 residual：

\[
q_0 = r - M\Delta y_0.
\]

再解 correction equation：

\[
M e_0 = q_0,
\]

更新：

\[
\Delta y_1 = \Delta y_0 + e_0.
\]

如果 correction equation 继续用同一个低精度 factorization 或 block preconditioner 来近似求解，这就是 mixed-precision iterative refinement 或 preconditioned iterative refinement 的思路。

---

## 4. GPU 精度问题：不是“GPU 只能 single”，而是“低精度更快”

需要澄清一点：GPU 支持多种浮点精度，包括 FP64、FP32、TF32、FP16、BF16 等。真正的问题通常是：

- FP64 精度高，但在很多 GPU 上吞吐较低；
- FP32 / TF32 / FP16 吞吐更高，尤其适合 GEMM/SYRK/TRSM 等 dense kernel；
- Cholesky 在 IPM 后期会遇到病态矩阵，低精度分解可能不稳定；
- normal equation 本身会放大条件数，因为 \(A D^2 A^T\) 的 conditioning 可能比原 KKT system 更差。

因此比较合理的路线是：

1. 在 GPU 上用 FP32 / TF32 做 block Cholesky；
2. 用 FP64 计算 residual；
3. 用 iterative refinement 或 Krylov method 修正；
4. IPM 外层不要求每一步 Newton direction 精确，只要求 residual 满足理论上允许的 inexactness 条件。

---

## 5. 与 inexact IPM 的连接

### 5.1 Inexact IPM 的核心思想

传统 IPM 假设 Newton system 被精确求解：

\[
M d = r.
\]

Inexact IPM 允许只求近似解 \(\hat d\)：

\[
M\hat d = r + e,
\]

其中 \(e\) 是线性系统误差。只要 \(e\) 足够小，或者满足与 central path neighbourhood 相关的条件，IPM 仍然可以保持收敛。

这正好对应你的 GPU 设定：

- 分块 Cholesky 误差；
- 低精度 factorization 误差；
- iterative solver 未完全收敛的误差；
- 多 GPU 通信和 Schur complement 近似带来的误差。

这些误差都可以统一视为 Newton equation residual：

\[
\eta_k = \frac{\|M_k \hat d_k - r_k\|}{\|r_k\|}.
\]

然后根据 IPM 当前阶段调节 \(\eta_k\)。

### 5.2 一个可能的动态精度策略

IPM 早期：

- centrality 要求较宽松；
- Newton direction 不必特别精确；
- 可以使用低精度 block Cholesky；
- iterative refinement 次数少；
- 目标 residual tolerance 可以较松，例如 \(10^{-2}\) 到 \(10^{-4}\)。

IPM 中后期：

- \(\mu\) 变小；
- normal equation 更病态；
- 需要更精确的 direction；
- 增加 iterative refinement 次数；
- 必要时切换到 FP64 factorization 或 KKT augmented system；
- 目标 residual tolerance 逐步收紧，例如 \(10^{-6}\) 到 \(10^{-10}\)。

一个简单规则可以是：

\[
\eta_k \leq c \mu_k^\alpha,
\]

其中 \(c>0\)，\(\alpha\in(0,1]\)。也可以采用 practical rule：只要 predictor/corrector step 后 primal residual、dual residual、complementarity 都下降，就接受当前 inexact direction。

---

## 6. 与 QIPM / IF-QIPM 的联系

### 6.1 QIPM 为什么需要 inexact IPM？

Quantum Interior Point Method 试图用 quantum linear system algorithm 加速 Newton system 的求解。但 quantum solver 输出的是近似量子态，之后还需要 tomography 或 classical extraction。因此 QIPM 天然得到的是 inexact Newton direction。

这和 GPU 低精度求解非常类似：

| QIPM 中的误差来源 | GPU 分块 Cholesky 中的对应物 |
|---|---|
| QLSA 近似误差 | FP32/TF32 Cholesky roundoff error |
| Quantum tomography error | 低精度 residual / communication error |
| Quantum hardware noise | GPU mixed precision / kernel numerical error |
| QLSA 对 condition number 敏感 | normal equation conditioning 恶化 |
| Iterative refinement 修正 QLSA 误差 | Iterative refinement 修正低精度 Cholesky 误差 |

所以可以借鉴 QIPM 的理论语言，把你的方法解释成：

> 一个 classical GPU analogue of inexact feasible QIPM: 用低精度、多 GPU direct factorization 作为 approximate Newton solver，再用 iterative refinement 和 feasibility-preserving Newton formulation 保证 IPM 外层收敛。

### 6.2 Feasible vs infeasible 的重要区别

普通 Newton system 如果被 inexact solve，可能破坏 primal-dual feasibility：

\[
Ax=b,\quad A^Ty+s=c.
\]

QIPM 文献中特别关心 **inexact feasible IPM**，因为 quantum solver 的误差可能导致 direction 不再严格保持可行性。

对于你的 classical GPU 方法也一样：

- 如果直接解 normal equation，得到的 \(\Delta y\) 近似误差会传播到 \(\Delta x,\Delta s\)；
- 如果 direction 不满足线性化可行性条件，outer IPM 会变成 infeasible IPM；
- infeasible IPM 不是不可以，但理论和 stopping rule 要重新设计。

因此值得研究两条路线：

1. **Inexact infeasible IPM**：允许 primal/dual residual 存在，用 residual reduction 控制收敛；
2. **Inexact feasible IPM**：设计 Newton reformulation，使得即使线性系统近似求解，也能保持某种可行性结构。

---

## 7. 其他可考虑的 inexact feasible IPM 方向

### 7.1 Orthogonal Subspace System / Null-space formulation

如果 \(V\) 是 \(A\) 的 null space basis，那么可以把 direction 限制在保持 primal feasibility 的子空间里。这样，即使求解存在误差，也更容易保持

\[
A(x+\Delta x)=b.
\]

缺点是：

- 构造和存储 \(V\) 可能很贵；
- 对大规模稠密 \(A\)，null-space basis 可能比原问题更稠密；
- 多 GPU 上维护 \(V\) 的通信成本可能高。

### 7.2 Augmented KKT system 而不是 normal equation

Normal equation：

\[
A D^2 A^T \Delta y = r
\]

优点是 SPD，可以用 Cholesky / CG。

缺点是：

- 条件数可能被平方放大；
- IPM 后期 \(D\) 极端变化，矩阵更病态；
- 低精度 Cholesky 风险增加。

另一种选择是 augmented KKT system：

\[
\begin{bmatrix}
-H & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta y
\end{bmatrix}
= r.
\]

它通常是 indefinite system，可以用 LDL\(^T\)、MINRES、SQMR 或 preconditioned Krylov 方法。虽然不能直接用 SPD Cholesky，但可能数值上比 normal equation 更稳。

### 7.3 Regularized inexact IPM

在 normal equation 或 KKT system 中加入 regularization：

\[
A D^2 A^T + \rho I,
\]

或者对 KKT system 加 primal/dual regularization。这样可以：

- 改善条件数；
- 避免 rank-deficient \(A\) 的问题；
- 提高低精度 Cholesky 的稳定性；
- 让 iterative refinement 更容易收敛。

代价是：regularization 会改变 Newton direction，需要在 outer IPM 里控制 regularization bias。

### 7.4 Predictor-corrector 中复用 factorization

Mehrotra predictor-corrector 方法中，同一个系数矩阵常被用于 predictor 和 corrector RHS。因此可以：

1. 对同一个 block normal equation 做一次低精度 Cholesky；
2. 用它解 predictor RHS；
3. 复用 factorization 解 corrector RHS；
4. 对两个 RHS 做 batched triangular solve；
5. 用 iterative refinement 分别修正。

这对 GPU 特别友好，因为多个 RHS 可以形成 dense matrix solve：

\[
M Z = R,
\]

其中 \(R\) 包含 predictor RHS、corrector RHS，甚至多个 regularization / centering parameter 的 candidate RHS。

### 7.5 Block Jacobi / block IC / hierarchical preconditioner

如果完整 block Cholesky 通信太贵，可以退一步做 preconditioner：

- block Jacobi：只分解 diagonal blocks；
- block Gauss-Seidel：顺序或颜色并行；
- incomplete block Cholesky：保留部分 off-diagonal coupling；
- hierarchical low-rank block approximation：用 H-matrix / low-rank Schur complement 降低通信与存储。

这类方法不直接给 exact Newton direction，但非常适合与 inexact IPM 结合。

---

## 8. 能否推广到一般问题？

这个想法不只适用于 LP，但最自然、最清楚的起点是 LP。

### 8.1 LP

LP 的 Newton system 结构最标准，normal equation 通常是：

\[
A D^2 A^T \Delta y = r.
\]

这是最适合先做 proof-of-concept 的问题。

### 8.2 Convex QP

对于 convex QP，Newton/KKT system 包含 Hessian：

\[
\begin{bmatrix}
Q + \Theta^{-1} & A^T \\
A & 0
\end{bmatrix}.
\]

如果消元后得到 SPD system，也可以用 Cholesky 或 preconditioned iterative refinement。Inexact feasible IPM 对 convex QP 已有相关理论，适合作为第二阶段推广。

### 8.3 Conic optimization / SDP / SOCP

对于 SOCP、SDP，IPM 也需要解 Newton system，但矩阵结构更复杂。尤其 SDP 的 Schur complement 可能非常稠密，理论上很适合 dense GPU linear algebra，但显存和通信压力更大。

### 8.4 非凸问题

对于一般非凸 NLP，Newton system 可能 indefinite，line search/trust region 和 global convergence 更复杂。此时不能简单套用 LP 的 inexact IPM 理论。可以借鉴 inexact Newton / trust-region IPM，但理论和实现都要更谨慎。

---

## 9. 可能的算法框架

### Algorithm: Multi-GPU Mixed-Precision Inexact IPM

输入：LP 数据 \((A,b,c)\)，初始 interior point \((x^0,y^0,s^0)\)，GPU 数量 \(p\)。

For \(k=0,1,2,\dots\):

1. 计算 scaling：

   \[
   D_k = X_k^{1/2}S_k^{-1/2}.
   \]

2. 构造 normal equation matrix：

   \[
   M_k = A D_k^2 A^T.
   \]

3. 对 \(M_k\) 做 block partition，并分配到多张 GPU。

4. 在每张 GPU 上用低精度 dense Cholesky 分解 diagonal block 或局部 block：

   \[
   M_{ii} \approx L_iL_i^T.
   \]

5. 构造 preconditioner：

   \[
   P_k \approx M_k.
   \]

6. 用 \(P_k\) 作为 preconditioner，求解：

   \[
   M_k\Delta y_k = r_k.
   \]

   可选求解器：

   - preconditioned CG；
   - flexible CG；
   - iterative refinement；
   - mixed-precision refinement；
   - batched RHS triangular solve。

7. 检查 linear solve residual：

   \[
   \eta_k = \frac{\|M_k\Delta y_k-r_k\|}{\|r_k\|}.
   \]

8. 如果 \(\eta_k\) 不满足当前 IPM tolerance：

   - 增加 refinement 次数；
   - 切换部分 block 到 FP64；
   - 加强 regularization；
   - 或改用 augmented KKT formulation。

9. 从 \(\Delta y_k\) recover \(\Delta x_k,\Delta s_k\)。

10. 选择 step length，更新：

\[
(x^{k+1},y^{k+1},s^{k+1}) = (x^k,y^k,s^k) + \alpha_k(\Delta x_k,\Delta y_k,\Delta s_k).
\]

11. 检查 primal residual、dual residual、duality gap、centrality。

---

## 10. 需要重点验证的研究问题

### Q1. Block partition 是否真的降低总体时间？

需要比较：

- 单 GPU full Cholesky；
- 多 GPU full distributed Cholesky；
- 多 GPU block diagonal Cholesky preconditioner；
- CPU sparse/direct solver；
- CPU/GPU iterative solver。

指标：

- factorization time；
- solve time；
- communication time；
- IPM total time；
- iteration count；
- final accuracy。

### Q2. Low precision 是否足够？

需要比较：

- FP64 Cholesky；
- FP32 Cholesky + FP64 residual refinement；
- TF32 Cholesky + FP64 residual refinement；
- mixed precision block strategy：diagonal block FP64，off-diagonal update FP32/TF32。

### Q3. Normal equation 是否过于病态？

需要记录：

\[
\kappa(M_k),\quad \eta_k,\quad \mu_k,
\]

并观察 IPM 后期是否出现：

- Cholesky breakdown；
- refinement stagnation；
- CG iteration count 爆炸；
- step length 变小；
- centrality 破坏。

### Q4. Inexact tolerance 如何设计？

需要比较：

- 固定 tolerance：\(10^{-4},10^{-6},10^{-8}\)；
- 随 \(\mu_k\) 收紧；
- 根据 residual reduction 自适应；
- 根据 predictor-corrector 成功与否自适应。

### Q5. Feasible formulation 是否优于 infeasible formulation？

需要比较：

- normal equation based inexact infeasible IPM；
- null-space / orthogonal-subspace feasible formulation；
- regularized inexact IPM；
- augmented KKT based IPM。

---

## 11. 初步判断

这个想法是有研究价值的，但需要把表述从

> “多 GPU 分块 Cholesky 精确求解大矩阵”

调整为

> “多 GPU mixed-precision block Cholesky 作为 approximate Newton solver / preconditioner，并通过 inexact feasible IPM、iterative refinement 和 regularization 控制误差”。

原因是：

1. 独立分解 diagonal blocks 通常不能等价于完整 Cholesky；
2. normal equation 在 IPM 后期会严重病态；
3. GPU 低精度需要 residual correction；
4. inexact IPM/QIPM 的理论正好提供了允许近似 Newton direction 的分析框架；
5. 对 LP 是最容易开始的 setting，但可以推广到 convex QP、SOCP、SDP 等更一般凸优化问题。

---

## 12. 可以写成论文的 tentative contribution

一个可能的论文贡献表述：

> We propose a multi-GPU mixed-precision inexact interior point framework for large-scale dense linear optimization. The method partitions the normal equation matrix into dense blocks, applies GPU-accelerated low-precision block Cholesky factorizations as preconditioners, and uses iterative refinement to recover Newton directions satisfying adaptive inexact IPM residual conditions. Inspired by inexact feasible QIPM, we investigate feasibility-preserving Newton reformulations and characterize the relationship between linear solve accuracy, centrality, and outer IPM convergence.

中文版本：

> 本文提出一种面向大规模稠密线性优化的多 GPU mixed-precision inexact IPM 框架。该方法将 normal equation 矩阵划分为多个稠密 block，在多张 GPU 上使用低精度 block Cholesky 构造近似 Newton solver 或 preconditioner，并通过 iterative refinement 使 Newton direction 满足自适应 inexact IPM 误差条件。受 inexact feasible QIPM 启发，本文进一步研究保持可行性的 Newton reformulation，并分析线性系统求解精度、centrality 与外层 IPM 收敛之间的关系。

---

## 13. 下一步建议

### Step 1: 理论最小模型

先从 LP normal equation 开始：

\[
M_k = A D_k^2 A^T.
\]

假设求解误差满足：

\[
\|M_k\hat d_k-r_k\| \leq \eta_k\|r_k\|.
\]

研究 \(\eta_k\) 与 IPM 收敛、centrality neighbourhood 的关系。

### Step 2: 数值原型

先不写完整 IPM，可以只 benchmark linear solver：

- 随机 dense LP；
- Netlib 中较小 dense-after-normal-equation 的实例；
- synthetic block-structured LP；
- 比较 full Cholesky vs block preconditioned CG vs iterative refinement。

### Step 3: 放入 IPM 外层

实现 simple primal-dual IPM 或基于现有 solver 框架替换 linear solver。

记录：

- outer iteration count；
- average inner iteration count；
- refinement count；
- residual tolerance；
- total runtime；
- failure cases。

### Step 4: 写成 QIPM-inspired classical paper

重点不是声称 quantum speedup，而是说：

- QIPM 发展出了一套处理 inexact Newton direction 的理论；
- 这套思想也可以服务于 classical heterogeneous computing；
- GPU mixed precision 和 QLSA error 在数学上都可以进入 inexact Newton residual model。

---

## 14. 关键词

- Interior Point Method
- Linear Programming
- Normal Equation System
- Dense Cholesky
- Block Cholesky
- Multi-GPU Linear Algebra
- Mixed Precision
- Iterative Refinement
- Inexact IPM
- Inexact Feasible IPM
- Quantum Interior Point Method
- Preconditioned Conjugate Gradient
- Regularized IPM
- Feasibility-Preserving Newton Direction
