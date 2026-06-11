# inexact-ipm

CPU reference prototype for studying inexact Newton solves inside primal-dual
interior point methods for linear programming.

The current code does **not** demonstrate GPU speedup and does **not** call
HiGHS/HIPO.  It provides a controlled Python baseline for checking Newton
normal-equation accuracy, failure modes, and IPM convergence before deciding
whether a GPU/block-wise dense Cholesky backend is worth integrating into HIPO.

## Current Status

Implemented:

- Dense random LP generation with reproducible seeds.
- A small Mehrotra-style primal-dual IPM outer loop.
- Linear solver backends:
  - `ExactCholeskySolver`: FP64 dense Cholesky baseline.
  - `BlockedCholeskySolver`: complete blocked dense Cholesky with trailing
    Schur-complement updates; this is a direct solver.
  - `LowPrecisionCholeskySolver`: CPU FP32 simulation of low-precision
    Cholesky.
  - `ConjugateGradientSolver`: plain CG baseline.
  - `BlockJacobiPCGSolver`: diagonal-block Cholesky preconditioner plus CG;
    this is not complete block Cholesky.
  - `AdaptiveRefinementSolver`: FP32 simulated factorization plus FP64
    residual refinement and FP64 fallback.
- End-to-end tests showing that ExactCholesky and AdaptiveRefinement converge
  on small dense random LPs.
- Frozen Newton-system benchmark for comparing linear solver behavior outside
  the IPM outer loop.

Not implemented yet:

- Real CUDA/CuPy multi-GPU block Cholesky kernels.
- Distributed block Cholesky with off-diagonal updates and communication.
- Netlib MPS loading into the Python prototype.
- HIPO/HiGHS linear-solver backend integration.

## Setup

```bash
pip install -r requirements.txt
pytest tests/
```

The optional GPU and sparse-Cholesky benchmark rows require extra packages:

- CuPy with a working CUDA device for `gpu_cupy_dense_cholesky`.
- scikit-sparse/CHOLMOD for `sparse_cholesky_cholmod`.

If these are unavailable, `experiments/benchmark_newton_systems.py` reports
those methods as `skip` instead of failing the benchmark.

## Experiments

Frozen Newton normal-equation benchmark:

```bash
python experiments/benchmark_newton_systems.py --sizes 32 64 --repeats 1 --skip-gpu
```

With optional GPU detection:

```bash
python experiments/benchmark_newton_systems.py --sizes 64 128 --repeats 3
```

Full IPM solver benchmark and plots:

```bash
python experiments/benchmark_solvers.py
```

Numerical validation:

```bash
python experiments/validate_numerics.py
```

## Interpreting Claims

Current CPU results support only these claims:

- FP32-only simulated Cholesky and fixed-tolerance block-Jacobi PCG can fail
  as the normal equations become ill-conditioned.
- Adaptive residual refinement can recover small dense CPU test convergence
  comparable to the FP64 Cholesky baseline.
- Complete blocked Cholesky is a direct solver only when off-diagonal
  coupling and Schur-complement/trailing updates are included.
- Diagonal block Cholesky alone is a block-Jacobi preconditioner and must be
  paired with CG/refinement to solve the original Newton equation.

Claims that still require evidence:

- Any GPU speedup.
- Any multi-GPU scaling.
- Performance on Netlib or HIPO production instances.
- Whether normal equations are preferable to augmented KKT systems in HIPO.

## HIPO Integration Gate

Before integrating with
`C:\Users\Yanyu\Documents\GitHub\HiGHS\highs\ipm\hipo`, the next evidence
needed is:

1. Export or intercept frozen HIPO Newton systems `(M_k, r_k, tolerance)`.
2. Reproduce CPU dense/sparse baseline timings on those frozen systems.
3. Run an actual GPU dense or block-wise benchmark with the same residual
   tolerance and explicit transfer/formation timing policy.
4. Decide whether the GPU backend should be a direct solver, a preconditioner
   for Krylov/refinement, or only a fallback/experimental option.

## License

MIT — until the paper is published, this is a private research prototype.
