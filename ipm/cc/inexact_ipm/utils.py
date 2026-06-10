"""
Utility routines: residuals, condition numbers, convergence checks, logging.

Provides functions that are shared across the IPM outer loop and the linear
solvers:

* Primal / dual residual and duality gap computation.
* Condition number estimation of the Newton system matrix.
* Convergence checks against the IPM tolerance.
* Structured logging for per-iteration diagnostics.

Usage
-----
    from inexact_ipm.utils import ipm_residuals, condition_number

    info = ipm_residuals(A, b, c, x, y, s)
    print(info["primal_inf"])   # e.g. 3.2e-06
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Residuals and duality measures
# ---------------------------------------------------------------------------


def ipm_residuals(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
) -> dict[str, float]:
    """Compute primal infeasibility, dual infeasibility, and duality gap.

    Parameters
    ----------
    A : (m, n) ndarray
    b : (m,) ndarray
    c : (n,) ndarray
    x : (n,) ndarray   (primal iterate)
    y : (m,) ndarray   (dual iterate)
    s : (n,) ndarray   (dual slacks)

    Returns
    -------
    dict[str, float]
        Keys: ``primal_inf``, ``dual_inf``, ``duality_gap``, ``mu``.
    """
    m = A.shape[0]

    rp = b - A @ x                      # primal residual
    rd = c - A.T @ y - s                # dual residual
    gap = np.dot(x, s)                  # complementarity gap
    mu = gap / len(x) if len(x) > 0 else 0.0

    return {
        "primal_inf": np.linalg.norm(rp) / (1.0 + np.linalg.norm(b)),
        "dual_inf": np.linalg.norm(rd) / (1.0 + np.linalg.norm(c)),
        "duality_gap": gap,
        "mu": mu,
    }


# ---------------------------------------------------------------------------
# Condition number estimation
# ---------------------------------------------------------------------------


def condition_number(
    A: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
    use_normal_eq: bool = True,
) -> float:
    """Estimate the condition number of the Newton system matrix.

    Parameters
    ----------
    A : (m, n) ndarray
        Constraint matrix.
    x, s : (n,) ndarray
        Current primal iterate and dual slacks.
    use_normal_eq : bool
        If True, estimate cond(A D^{-1} A^T).  Otherwise, estimate
        cond( [0 A; A^T -D] ) for the augmented system.

    Returns
    -------
    float
        Estimated condition number (ratio of largest to smallest singular
        value).  Returns ``np.inf`` if the system appears singular.
    """
    d = x / s                           # diagonal of D^{-1}
    D_inv_sqrt = np.sqrt(np.maximum(d, 1e-15))

    if use_normal_eq:
        # Normal equations: A D^{-1} A^T
        A_scaled = A * D_inv_sqrt[None, :]
        M = A_scaled @ A_scaled.T
    else:
        # Full augmented system (we only estimate via normal eq. for now)
        A_scaled = A * D_inv_sqrt[None, :]
        M = A_scaled @ A_scaled.T

    try:
        sv = np.linalg.svd(M, compute_uv=False)
        cond = sv[0] / sv[-1] if sv[-1] > 1e-15 else np.inf
    except np.linalg.LinAlgError:
        cond = np.inf

    return cond


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------


def converged(
    residuals: dict[str, float],
    tol: float,
) -> bool:
    """Return True if the IPM has reached the convergence tolerance.

    A primal-dual iterate is considered converged when all of the
    following are below ``tol``:

    - Relative primal infeasibility.
    - Relative dual infeasibility.
    - Duality gap.

    Parameters
    ----------
    residuals : dict
        Output of ``ipm_residuals``.
    tol : float
        Tolerance (e.g. 1e-8).

    Returns
    -------
    bool
    """
    return (
        residuals["primal_inf"] < tol
        and residuals["dual_inf"] < tol
        and residuals["duality_gap"] < tol
    )


# ---------------------------------------------------------------------------
# Iteration logger
# ---------------------------------------------------------------------------


class IterationLogger:
    """Collects and optionally prints per-iteration IPM diagnostics.

    Parameters
    ----------
    verbose : bool
        If True, print a header and one line per iteration.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.history: list[dict[str, Any]] = []
        self._start_time: float | None = None

    def start(self) -> None:
        """Record start time and optionally print a table header."""
        self._start_time = time.perf_counter()
        self.history.clear()
        if self.verbose:
            print(
                f"{'iter':>5s}  {'primal_inf':>10s}  {'dual_inf':>10s}  "
                f"{'gap':>12s}  {'mu':>12s}  {'alpha':>8s}  {'cond':>10s}"
            )

    def log(
        self,
        iteration: int,
        residuals: dict[str, float],
        alpha: float,
        cond: float,
    ) -> None:
        """Record and optionally print diagnostics for one iteration.

        Parameters
        ----------
        iteration : int
            IPM outer-loop iteration number.
        residuals : dict
            Output of ``ipm_residuals``.
        alpha : float
            Step length taken in this iteration (primal).
        cond : float
            Condition number estimate for the Newton system.
        """
        record = {
            "iter": iteration,
            **residuals,
            "alpha": alpha,
            "cond": cond,
            "time": time.perf_counter() - (self._start_time or 0.0),
        }
        self.history.append(record)

        if self.verbose:
            print(
                f"{iteration:5d}  "
                f"{residuals['primal_inf']:10.2e}  "
                f"{residuals['dual_inf']:10.2e}  "
                f"{residuals['duality_gap']:12.4e}  "
                f"{residuals['mu']:12.4e}  "
                f"{alpha:8.4f}  "
                f"{cond:10.2e}"
            )
