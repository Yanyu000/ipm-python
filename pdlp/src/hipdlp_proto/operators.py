"""Linear operator abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


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


def to_cupy_operator(op: SparseMatrixOperator) -> SparseMatrixOperator:
    import cupyx.scipy.sparse as cpsp

    A_csr = cpsp.csr_matrix(op.A_csr)
    A_csc = cpsp.csc_matrix(op.A_csc)
    return SparseMatrixOperator(A_csr=A_csr, A_csc=A_csc, backend="cupy")