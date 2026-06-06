"""Array helpers for the HiPDLP prototype."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_float64_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def zeros(shape: tuple[int, ...], backend: str = "numpy") -> Any:
    if backend == "numpy":
        return np.zeros(shape, dtype=np.float64)
    if backend == "cupy":
        import cupy as cp

        return cp.zeros(shape, dtype=cp.float64)
    raise ValueError(f"Unsupported backend: {backend}")