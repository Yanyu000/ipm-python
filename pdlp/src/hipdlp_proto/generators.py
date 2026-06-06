"""Random problem generators."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import LpProblem
from .operators import SparseMatrixOperator


def random_sparse_operator(m: int, n: int, density: float, seed: int = 0) -> SparseMatrixOperator:
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=density, format="csr", dtype=np.float64, random_state=rng)
    A.data = rng.normal(size=A.nnz)
    A.sum_duplicates()
    A.eliminate_zeros()
    return SparseMatrixOperator(A_csr=A, A_csc=A.tocsc())


def random_feasible_lp(m: int, n: int, density: float, seed: int = 0) -> LpProblem:
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