# inexact-ipm

Research prototype for **Multi-GPU Mixed-Precision Inexact Interior Point
Methods for Linear Programming**.

This is a CPU-only Python reference implementation.  GPU and mixed-precision
extensions will be added in a later phase.

## Structure

```
inexact_ipm/          # core library
├── lp_problem.py     # random & curated LP instance generation
├── ipm_solver.py     # Mehrotra predictor-corrector IPM outer loop
├── linear_solver.py  # pluggable Newton system solvers (Cholesky, CG, MINRES, …)
└── utils.py          # residuals, condition numbers, convergence checks, logging

tests/                # pytest test suite
├── test_lp_problem.py
└── test_linear_solver.py

experiments/          # benchmark & plotting scripts
└── benchmark_solvers.py
```

## Quick start

```bash
pip install -r requirements.txt
pytest tests/
```

## Progress

- [ ] LP problem generation (``lp_problem.py``)
- [ ] IPM outer loop (``ipm_solver.py``)
- [ ] Exact linear solvers (Cholesky, LU)
- [ ] Inexact linear solvers (CG, MINRES)
- [ ] Mixed-precision iterative refinement solver
- [ ] Benchmark driver and plots
- [ ] GPU offload (CUDA / CuPy)
- [ ] Multi-GPU data-parallel IPM

## License

MIT — until the paper is published, this is a private research prototype.
