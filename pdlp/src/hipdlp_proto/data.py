"""Core LP data structures."""

from __future__ import annotations

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