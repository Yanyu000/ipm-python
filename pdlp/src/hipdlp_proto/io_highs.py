"""Highs I/O helpers."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import LpProblem
from .operators import SparseMatrixOperator


def scipy_matrices_from_highs_lp(lp):
    import highspy

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


def read_mps(path: str) -> LpProblem:
    import highspy

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