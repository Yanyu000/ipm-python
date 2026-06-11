"""
Pluggable linear system solvers for the IPM Newton step.

The Newton system at each IPM iteration reduces to the normal equations

    M @ delta_y = r,   M = A @ diag(d^2) @ A.T  (SPD),

where d = sqrt(x / s) in the current prototype and r is the assembled
right-hand side.  This module
provides three solver strategies with a uniform ``solve(M, r)`` interface:

* ``ExactCholeskySolver``       — FP64 dense Cholesky (exact baseline).
* ``BlockedCholeskySolver``     — FP64 blocked dense Cholesky with
                                   Schur-complement/trailing updates.
* ``LowPrecisionCholeskySolver`` — FP32 downcast Cholesky (simulates GPU low-precision).
* ``ConjugateGradientSolver``    — unpreconditioned CG baseline.
* ``BlockJacobiPCGSolver``       — Conjugate gradient with a block-diagonal
                                   Jacobi preconditioner (inexact iterative).
* ``AdaptiveRefinementSolver``  — Dynamic-precision iterative refinement
                                   driven by the IPM duality measure.

Each solver tracks ``solve_time``, ``final_eta``, and ``num_iterations``
after every call to ``solve()``.

Usage
-----
    from inexact_ipm.linear_solver import ExactCholeskySolver

    solver = ExactCholeskySolver()
    delta_y, eta = solver.solve(M, r)
    print(f"relative residual: {eta:.2e}, time: {solver.solve_time:.4f}s")
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy import linalg
from scipy.sparse import linalg as splinalg
from scipy.sparse.linalg import LinearOperator


# ===================================================================
# Base class
# ===================================================================


class NewtonSolver:
    """Base class for Newton-system solvers operating on the normal equations.

    Subclasses must implement ``solve(M, r) -> (delta_y, eta)``.

    Attributes set after each ``solve`` call
    ----------------------------------------
    solve_time : float
        Wall-clock seconds for the most recent solve (factorization + solve).
    final_eta : float
        Relative residual  ‖M·Δy − r‖₂ / ‖r‖₂  **in FP64** regardless of
        the internal precision used by the solver.
    num_iterations : int
        1 for direct solvers; inner iteration count for iterative solvers.
    """

    def __init__(self) -> None:
        self.solve_time: float = 0.0
        self.final_eta: float = 0.0
        self.num_iterations: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        M: np.ndarray,
        r: np.ndarray,
        mu: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Solve  M @ delta_y = r.

        Parameters
        ----------
        M : (m, m) ndarray
            Symmetric positive-definite normal-equations matrix.
        r : (m,) ndarray
            Right-hand side vector.
        mu : float or None
            Current duality measure (optional).  Solvers that implement
            dynamic-precision strategies can use ``mu`` to adjust their
            target residual tolerance.

        Returns
        -------
        delta_y : (m,) ndarray
            Solution vector.
        eta : float
            Relative residual  ‖M·Δy − r‖₂ / ‖r‖₂  (in float64).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_eta(self, M: np.ndarray, delta_y: np.ndarray, r: np.ndarray) -> float:
        """Compute relative residual in float64."""
        residual = M @ delta_y - r
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-15:
            return 0.0
        return float(np.linalg.norm(residual) / r_norm)


# ===================================================================
# Exact FP64 Cholesky solver
# ===================================================================


class ExactCholeskySolver(NewtonSolver):
    """Dense Cholesky solve in native FP64 — the "exact" baseline.

    Uses ``scipy.linalg.cho_factor`` (lower-triangular) followed by
    ``scipy.linalg.cho_solve``.  Because the normal-equations matrix is
    SPD, Cholesky is unconditionally stable (no pivoting required).
    """

    def solve(
        self,
        M: np.ndarray,
        r: np.ndarray,
        mu: float | None = None,
    ) -> tuple[np.ndarray, float]:
        _ = mu  # unused — FP64 Cholesky doesn't need dynamic tolerance
        t0 = time.perf_counter()

        # --- factor & solve ---
        L_and_lower = linalg.cho_factor(M, lower=True)
        delta_y = linalg.cho_solve(L_and_lower, r)

        self.solve_time = time.perf_counter() - t0
        self.num_iterations = 1
        self.final_eta = self._compute_eta(M, delta_y, r)

        return delta_y, self.final_eta


# ===================================================================
# Blocked FP64 Cholesky solver with trailing Schur updates
# ===================================================================


class BlockedCholeskySolver(NewtonSolver):
    """Dense blocked Cholesky solve in FP64.

    This is a direct solver, not a block-Jacobi approximation.  Each
    diagonal panel is factored, the off-diagonal panel is solved, and the
    trailing matrix is updated by the Schur-complement term
    ``A22 -= L21 @ L21.T``.  It is included as an algorithmic reference for
    "complete block Cholesky with off-diagonal updates"; SciPy/LAPACK's
    native Cholesky remains the performance baseline.
    """

    def __init__(self, block_size: int = 64, regularize: float = 0.0) -> None:
        super().__init__()
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if regularize < 0:
            raise ValueError("regularize must be non-negative")
        self.block_size = block_size
        self.regularize = regularize

    def solve(
        self,
        M: np.ndarray,
        r: np.ndarray,
        mu: float | None = None,
    ) -> tuple[np.ndarray, float]:
        _ = mu
        t0 = time.perf_counter()

        L = _blocked_cholesky_factor(
            M,
            block_size=self.block_size,
            regularize=self.regularize,
        )
        y = linalg.solve_triangular(L, r, lower=True, check_finite=False)
        delta_y = linalg.solve_triangular(
            L.T,
            y,
            lower=False,
            check_finite=False,
        )

        self.solve_time = time.perf_counter() - t0
        self.num_iterations = max(1, int(np.ceil(M.shape[0] / self.block_size)))
        self.final_eta = self._compute_eta(M, delta_y, r)

        return delta_y, self.final_eta


# ===================================================================
# Low-precision (FP32) Cholesky solver
# ===================================================================


class LowPrecisionCholeskySolver(NewtonSolver):
    """Cholesky solver that downcasts to float32 before factorization.

    This simulates a GPU tensor-core Cholesky (e.g., CUDA ``cusolverDn``
    in FP32 mode) without requiring actual GPU hardware.  The factor and
    solve step run entirely in float32; the residual ``eta`` is always
    evaluated in float64 so the accuracy penalty from reduced precision
    is faithfully reported.

    Parameters
    ----------
    regularize : float
        Small positive value added to the diagonal before factorization
        (guards against singularity due to round-off in float32).
    """

    def __init__(self, regularize: float = 1e-10) -> None:
        super().__init__()
        self.regularize = regularize

    def solve(
        self,
        M: np.ndarray,
        r: np.ndarray,
        mu: float | None = None,
    ) -> tuple[np.ndarray, float]:
        _ = mu  # unused — FP32 Cholesky doesn't need dynamic tolerance
        t0 = time.perf_counter()

        # --- downcast, guarding against fp32 overflow ---
        M32 = _safe_cast_f32(M)
        r32 = _safe_cast_f32(r)

        # --- regularize (safety) ---
        if self.regularize > 0:
            M32.flat[:: M32.shape[0] + 1] += np.float32(self.regularize)

        # --- factor & solve in fp32 ---
        try:
            L_and_lower = linalg.cho_factor(M32, lower=True)
            delta_y32 = linalg.cho_solve(L_and_lower, r32)
        except linalg.LinAlgError:
            # Fall back to eigh if Cholesky fails (should be rare)
            w, V = np.linalg.eigh(M32)
            w = np.maximum(w, 1e-12)
            delta_y32 = V @ ((V.T @ r32) / w)

        # --- upcast result ---
        delta_y = delta_y32.astype(np.float64)

        # If fp32 produced NaN, fall back to exact FP64 solve.
        if not np.all(np.isfinite(delta_y)):
            delta_y = np.linalg.solve(M, r)

        self.solve_time = time.perf_counter() - t0
        self.num_iterations = 1
        self.final_eta = self._compute_eta(M, delta_y, r)

        return delta_y, self.final_eta


# ===================================================================
# Unpreconditioned Conjugate Gradient
# ===================================================================


class ConjugateGradientSolver(NewtonSolver):
    """Unpreconditioned Conjugate Gradient baseline for SPD systems."""

    def __init__(self, cg_tol: float = 1e-8, max_iter: int = 1000) -> None:
        super().__init__()
        if cg_tol <= 0:
            raise ValueError("cg_tol must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        self.cg_tol = cg_tol
        self.max_iter = max_iter
        self.info: int = 0

    def solve(
        self,
        M: np.ndarray,
        r: np.ndarray,
        mu: float | None = None,
    ) -> tuple[np.ndarray, float]:
        _ = mu
        t0 = time.perf_counter()

        cg_iters: list[int] = [0]

        def _cb(_xk: np.ndarray) -> None:
            cg_iters[0] += 1

        delta_y, info = splinalg.cg(
            M,
            r,
            rtol=self.cg_tol,
            maxiter=self.max_iter,
            callback=_cb,
        )

        self.solve_time = time.perf_counter() - t0
        self.info = int(info)
        self.num_iterations = cg_iters[0] if info == 0 else int(info)
        self.final_eta = self._compute_eta(M, delta_y, r)

        return delta_y, self.final_eta


# ===================================================================
# Block-Jacobi preconditioned Conjugate Gradient
# ===================================================================


class BlockJacobiPCGSolver(NewtonSolver):
    """Conjugate Gradient with a block-diagonal Jacobi preconditioner.

    The SPD matrix ``M`` is partitioned into ``num_blocks`` diagonal blocks

        P = blkdiag(M₁₁, M₂₂, …, Mₚₚ),

    where each ``Mᵢᵢ`` is factored via Cholesky and its inverse is applied
    as the CG preconditioner.  This is a natural way to inject inexactness:
    with ``num_blocks = 1`` the preconditioner is exactly ``M`` itself and
    CG converges in 1 iteration; with ``num_blocks = m`` the preconditioner
    is diagonal (Jacobi) and convergence is slower but each iteration is
    extremely cheap.

    Parameters
    ----------
    num_blocks : int
        Number of diagonal blocks (1 ≤ num_blocks ≤ m).
    cg_tol : float
        Relative residual tolerance for CG termination.
    max_iter : int
        Maximum number of CG iterations.
    """

    def __init__(
        self,
        num_blocks: int = 4,
        cg_tol: float = 1e-6,
        max_iter: int = 500,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be ≥ 1")
        self.num_blocks = num_blocks
        self.cg_tol = cg_tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        M: np.ndarray,
        r: np.ndarray,
        mu: float | None = None,
    ) -> tuple[np.ndarray, float]:
        _ = mu  # unused — CG tolerance is fixed at construction time
        m = M.shape[0]
        t0 = time.perf_counter()

        # --- build block-diagonal preconditioner ---
        precond = _build_block_diagonal_preconditioner(M, self.num_blocks)

        # --- track iteration count ---
        cg_iters: list[int] = [0]

        def _cb(_xk: np.ndarray) -> None:
            cg_iters[0] += 1

        # --- run CG ---
        delta_y, info = splinalg.cg(
            M,
            r,
            M=precond,
            rtol=self.cg_tol,
            maxiter=self.max_iter,
            callback=_cb,
        )

        self.solve_time = time.perf_counter() - t0
        self.num_iterations = cg_iters[0] if info == 0 else info
        self.final_eta = self._compute_eta(M, delta_y, r)

        return delta_y, self.final_eta


# ===================================================================
# Adaptive-precision iterative refinement solver
# ===================================================================


class AdaptiveRefinementSolver(NewtonSolver):
    """Dynamic-precision iterative refinement solver (cf. Section 5.2).

    Uses a low-precision (FP32) Cholesky factorization for the initial
    solve, then iteratively refines the solution in FP64 until the
    relative residual meets a target tolerance **that depends on the
    current IPM duality measure** ``mu``:

    ============  ==============
    mu range      eta_target
    ============  ==============
    mu > 1e-2     1e-4  (early IPM — loose)
    mu > 1e-6     1e-7  (mid IPM)
    mu ≤ 1e-6     1e-10 (late IPM — tight, near exact)
    ============  ==============

    If the target is not met after ``max_refinement_steps``, the solver
    falls back to a full FP64 Cholesky solve.

    Parameters
    ----------
    base_precision : str
        ``"float32"`` to use FP32 Cholesky in the inner loop (default).
        ``"float64"`` can be set for testing (degenerates to exact solve).
    max_refinement_steps : int
        Maximum number of iterative refinement corrections.

    Attributes set after each ``solve`` call
    ----------------------------------------
    num_refinement_steps : int
        Number of refinement corrections actually performed.
    used_fallback : bool
        True if the solver fell back to a full FP64 solve.
    """

    def __init__(
        self,
        base_precision: str = "float32",
        max_refinement_steps: int = 5,
    ) -> None:
        super().__init__()
        if base_precision not in ("float32", "float64"):
            raise ValueError(
                f"base_precision must be 'float32' or 'float64', "
                f"got {base_precision!r}"
            )
        self.base_precision = base_precision
        self.max_refinement_steps = max_refinement_steps

        # Additional diagnostics
        self.num_refinement_steps: int = 0
        self.used_fallback: bool = False

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        M: np.ndarray,
        r: np.ndarray,
        mu: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Solve  M @ delta_y = r  with adaptive-precision refinement.

        Parameters
        ----------
        M : (m, m) ndarray (SPD)
        r : (m,) ndarray
        mu : float or None
            Current IPM duality measure.  Used to select the residual
            tolerance target.

        Returns
        -------
        delta_y : (m,) ndarray
        eta : float
            Relative residual in FP64.
        """
        t0 = time.perf_counter()
        dtype: np.dtype = np.dtype(self.base_precision)
        self.num_refinement_steps = 0
        self.used_fallback = False

        # ---- step 1: initial low-precision solve ----
        M_lo = _safe_cast_dtype(M, dtype)
        r_lo = _safe_cast_dtype(r, dtype)

        # Regularize in working precision.
        M_lo.flat[:: M_lo.shape[0] + 1] += dtype.type(1e-10)

        # ``factor`` holds either a Cholesky factor tuple or an eigen-
        # decomposition (w, V) if Cholesky failed in low precision.
        factor: tuple | None = None
        use_eigh: bool = False

        try:
            L_and_lower = linalg.cho_factor(M_lo, lower=True)
            x_lo = linalg.cho_solve(L_and_lower, r_lo)
            factor = L_and_lower
        except linalg.LinAlgError:
            # Fall back to eigh in working precision.
            w, V = np.linalg.eigh(M_lo)
            w = np.maximum(w, 1e-12)
            x_lo = V @ ((V.T @ r_lo) / w)
            factor = (w, V)
            use_eigh = True

        delta_y = x_lo.astype(np.float64)

        if not np.all(np.isfinite(delta_y)):
            # Precision breakdown — fall back immediately.
            delta_y = np.linalg.solve(M, r)
            self.solve_time = time.perf_counter() - t0
            self.num_iterations = 1
            self.num_refinement_steps = 0
            self.used_fallback = True
            self.final_eta = self._compute_eta(M, delta_y, r)
            return delta_y, self.final_eta

        # ---- step 2: compute residual in FP64 ----
        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-15:
            self.solve_time = time.perf_counter() - t0
            self.num_iterations = 1
            self.final_eta = 0.0
            return delta_y, 0.0

        residual = r - M @ delta_y
        eta = float(np.linalg.norm(residual) / r_norm)

        # ---- step 3: target tolerance from mu ----
        eta_target = _mu_to_eta_target(mu)

        # ---- step 4: iterative refinement loop ----
        for step in range(self.max_refinement_steps):
            # If precision has broken down, stop refining.
            if not np.isfinite(eta) or not np.all(np.isfinite(delta_y)):
                break

            if eta <= eta_target:
                break

            self.num_refinement_steps += 1

            # Solve correction  M @ e = residual  in working precision.
            q_lo = _safe_cast_dtype(residual, dtype)
            try:
                if use_eigh:
                    w_eig, V_eig = factor
                    e_lo = V_eig @ ((V_eig.T @ q_lo) / w_eig)
                else:
                    e_lo = linalg.cho_solve(factor, q_lo)
            except (linalg.LinAlgError, ValueError):
                break  # factorization degraded; fall through to fallback

            # Update in FP64.
            delta_y = delta_y + e_lo.astype(np.float64)

            # Recompute residual in FP64.
            residual = r - M @ delta_y
            eta = float(np.linalg.norm(residual) / r_norm)

        # ---- step 5-6: fallback if target not met ----
        if not np.isfinite(eta) or eta > eta_target:
            self.used_fallback = True
            delta_y = np.linalg.solve(M, r)
            eta = self._compute_eta(M, delta_y, r)

        # ---- bookkeeping ----
        self.solve_time = time.perf_counter() - t0
        self.num_iterations = 1 + self.num_refinement_steps
        self.final_eta = eta

        return delta_y, self.final_eta


# ===================================================================
# Helpers
# ===================================================================


def _mu_to_eta_target(mu: float | None) -> float:
    """Map IPM duality measure to linear-solve tolerance.

    Section 5.2 dynamic precision strategy:
      - Early IPM (large mu):   loose tolerance, cheap solves.
      - Late IPM (small mu):   tight tolerance, near-exact solves.
    """
    if mu is None or mu > 1e-2:
        return 1e-4
    if mu > 1e-6:
        return 1e-7
    return 1e-10


def _safe_cast_dtype(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast to the requested dtype, guarding against overflow."""
    if dtype == np.float32:
        return _safe_cast_f32(arr)
    return arr.astype(dtype)


def _safe_cast_f32(arr: np.ndarray) -> np.ndarray:
    """Cast to float32, clipping values that would overflow FP32 range."""
    # FP32 max is ~3.4e38; leave headroom.
    clip = np.float32(1e37)
    return np.clip(arr, -clip, clip).astype(np.float32)


# ===================================================================
# Helper: blocked dense Cholesky
# ===================================================================


def _blocked_cholesky_factor(
    M: np.ndarray,
    block_size: int,
    regularize: float = 0.0,
) -> np.ndarray:
    """Return a lower Cholesky factor using explicit block updates.

    The returned ``L`` satisfies ``L @ L.T ~= M + regularize * I``.  This
    routine is intentionally simple and diagnostic; it exposes the Schur
    complement updates that a distributed/GPU block Cholesky must perform.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")

    n = M.shape[0]
    A_work = np.array(M, dtype=np.float64, copy=True)
    if regularize > 0:
        A_work.flat[:: n + 1] += regularize

    L = np.zeros_like(A_work)

    for k in range(0, n, block_size):
        end = min(k + block_size, n)

        panel = A_work[k:end, k:end]
        panel = 0.5 * (panel + panel.T)
        Lkk = linalg.cholesky(panel, lower=True, check_finite=False)
        L[k:end, k:end] = Lkk

        if end == n:
            continue

        # M21 = L21 Lkk.T, so solve Lkk L21.T = M21.T.
        L21_T = linalg.solve_triangular(
            Lkk,
            A_work[end:n, k:end].T,
            lower=True,
            check_finite=False,
        )
        L21 = L21_T.T
        L[end:n, k:end] = L21

        # Schur-complement/trailing update.
        A_work[end:n, end:n] -= L21 @ L21.T
        A_work[end:n, end:n] = 0.5 * (
            A_work[end:n, end:n] + A_work[end:n, end:n].T
        )

    return L


# ===================================================================
# Helper: block-diagonal preconditioner builder
# ===================================================================


def _build_block_diagonal_preconditioner(
    M: np.ndarray,
    num_blocks: int,
) -> LinearOperator:
    """Build a LinearOperator that applies  P⁻¹  where
    P = blkdiag(M₁₁, …, Mₚₚ)  with each diagonal block factored via Cholesky.

    Parameters
    ----------
    M : (m, m) ndarray (SPD)
    num_blocks : int
        Number of diagonal blocks.

    Returns
    -------
    LinearOperator
        Applies P⁻¹ @ v for any vector v of length m.
    """
    m = M.shape[0]
    p = min(num_blocks, m)

    block_slices: list[tuple[int, int]] = []
    factors: list[tuple[np.ndarray, bool]] = []

    for i in range(p):
        start = i * m // p
        end = (i + 1) * m // p if i < p - 1 else m
        block = M[start:end, start:end].copy()

        # Regularize tiny diagonal entries to keep Cholesky stable.
        block.flat[:: block.shape[0] + 1] += 1e-12

        fac = linalg.cho_factor(block, lower=True)
        block_slices.append((start, end))
        factors.append(fac)

    def _matvec(v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v)
        for (s, e), fac in zip(block_slices, factors):
            out[s:e] = linalg.cho_solve(fac, v[s:e])
        return out

    return LinearOperator((m, m), matvec=_matvec, dtype=M.dtype)
