"""
Primal-dual interior point method (IPM) outer loop.

Implements a simplified Mehrotra predictor-corrector IPM for linear
programming problems in standard form::

    min  c^T x      s.t.   A x = b,   x >= 0.

At each iteration the algorithm forms the normal-equations system::

    M @ Δy = r,     M = A @ diag(d^2) @ A^T,    d = √(x / s),

solves it with a pluggable ``NewtonSolver``, recovers primal-dual step
directions, and updates the iterate with a fraction-to-boundary rule.

Usage
-----
    from inexact_ipm.lp_problem import generate_random_dense_lp
    from inexact_ipm.ipm_solver import IPMSolver, IPMResult
    from inexact_ipm.linear_solver import ExactCholeskySolver

    A, b, c = generate_random_dense_lp(m=20, n=50, seed=42)
    solver = IPMSolver(A, b, c, solver=ExactCholeskySolver())
    result = solver.solve()
    print(result.history[-1])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from inexact_ipm.lp_problem import compute_normal_equation
from inexact_ipm.linear_solver import NewtonSolver


# ===================================================================
# Result dataclass
# ===================================================================


@dataclass
class IPMResult:
    """Output of an IPM solve.

    Attributes
    ----------
    x : (n,) ndarray
        Primal solution (or final iterate if not converged).
    y : (m,) ndarray
        Dual solution.
    s : (n,) ndarray
        Dual slacks.
    converged : bool
        Whether the solver met the convergence tolerance.
    n_iter : int
        Number of main iterations performed.
    history : list[dict]
        Per-iteration diagnostics (mu, primal_res, dual_res, eta,
        cond_M, step lengths, solve_time, …).
    """

    x: np.ndarray | None = None
    y: np.ndarray | None = None
    s: np.ndarray | None = None
    converged: bool = False
    n_iter: int = 0
    history: list[dict] = field(default_factory=list)


# ===================================================================
# IPM Solver
# ===================================================================


class IPMSolver:
    """Primal-dual path-following IPM with a Mehrotra-style predictor-corrector.

    The solver reduces the 2×2 augmented system to the normal equations
    and delegates the linear solve to a pluggable ``NewtonSolver``.

    Parameters
    ----------
    A, b, c : ndarray
        Problem data for  min c^T x  s.t. A x = b, x >= 0.
    solver : NewtonSolver
        Any concrete ``NewtonSolver`` instance (exact Cholesky,
        low-precision Cholesky, block-Jacobi CG, …).
    tol : float
        Convergence tolerance for the duality gap and residuals.
    max_iter : int
        Maximum number of IPM outer-loop iterations.
    tau : float
        Fraction-to-boundary parameter (0 < tau < 1).  Smaller values
        give more conservative step lengths.
    verbose : bool
        If True, print per-iteration diagnostics.
    """

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        solver: NewtonSolver,
        tol: float = 1e-8,
        max_iter: int = 100,
        tau: float = 0.99,
        verbose: bool = True,
    ) -> None:
        self.A = A
        self.b = b
        self.c = c
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.tau = tau
        self.verbose = verbose

        self.m, self.n = A.shape

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self) -> IPMResult:
        """Run the predictor-corrector IPM.

        Returns
        -------
        IPMResult
        """
        A, b, c = self.A, self.b, self.c
        m, n = self.m, self.n

        # --- initialisation ---
        x = np.ones(n, dtype=np.float64)
        s = np.ones(n, dtype=np.float64)
        y = np.zeros(m, dtype=np.float64)

        history: list[dict] = []

        if self.verbose:
            print(f"{'iter':>4s}  {'mu':>10s}  {'p_inf':>10s}  "
                  f"{'d_inf':>10s}  {'eta':>10s}  {'cond(M)':>10s}  "
                  f"{'a_p':>7s}  {'a_d':>7s}  {'t_slv':>8s}")

        # --- main loop ---
        converged = False
        for it in range(self.max_iter):
            # 2a. scaling vector  d = √(x/s)
            d = np.sqrt(x / s)

            # residuals
            rp = b - A @ x
            rd = c - A.T @ y - s
            mu = float(np.dot(x, s) / n)

            primal_res = float(np.linalg.norm(rp) / (1.0 + np.linalg.norm(b)))
            dual_res = float(np.linalg.norm(rd) / (1.0 + np.linalg.norm(c)))

            # --- convergence check ---
            if mu < self.tol and primal_res < self.tol and dual_res < self.tol:
                converged = True
                if self.verbose:
                    print(f"  Converged at iteration {it}")
                break

            # --- divergence safeguard ---
            if mu > 1e10 or primal_res > 1e6 or dual_res > 1e6:
                if self.verbose:
                    print(f"  Diverged at iteration {it} — stopping.")
                break

            # --- detect loss of feasibility in late iterations ---
            if mu < 1e-6 and primal_res > 1e-3:
                if self.verbose:
                    print(f"  mu={mu:.1e} but pr_inf={primal_res:.1e} — "
                          f"inexact solve lost primal feasibility. Stopping.")
                break

            # --- numerical health check ---
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(s))):
                if self.verbose:
                    print(f"  Non-finite iterate at iteration {it} — stopping.")
                break

            # 2b. form normal-equations matrix  M = A diag(d^2) A^T
            M = compute_normal_equation(A, d)

            try:
                cond_M = float(np.linalg.cond(M))
            except np.linalg.LinAlgError:
                cond_M = np.inf

            # ----------------------------------------------------------
            # 2c.   Affine (predictor) step  (σ = 0)
            # ----------------------------------------------------------
            # r_aff = b + A @ ((x/s) * rd)
            r_aff = b + A @ ((x / s) * rd)

            dy_aff, eta_aff = self.solver.solve(M, r_aff, mu=mu)

            # recover Δx, Δs from Δy  (affine)
            dx_aff = -x - (x / s) * rd + (x / s) * (A.T @ dy_aff)
            ds_aff = rd - A.T @ dy_aff

            # ----------------------------------------------------------
            # Step lengths for affine direction
            # ----------------------------------------------------------
            alpha_p_aff = self._step_length(x, dx_aff)
            alpha_d_aff = self._step_length(s, ds_aff)

            # ----------------------------------------------------------
            # Centering parameter  σ = (μ_aff / μ)³
            # ----------------------------------------------------------
            mu_aff = float(
                np.dot(x + alpha_p_aff * dx_aff, s + alpha_d_aff * ds_aff) / n
            )
            sigma = (mu_aff / mu) ** 3 if mu > 1e-16 else 0.0

            # ----------------------------------------------------------
            # 2c (continued).  Combined corrector RHS
            #
            #   r = b  +  A @ (x/s * rd)
            #         -  σ μ A @ (1/s)
            #         +  A @ (Δx_aff * Δs_aff / s)
            # ----------------------------------------------------------
            rhs_corr_1 = b + A @ ((x / s) * rd)
            rhs_corr_2 = sigma * mu * (A @ (1.0 / s))
            rhs_corr_3 = A @ (dx_aff * ds_aff / s)
            r_corr = rhs_corr_1 - rhs_corr_2 + rhs_corr_3

            # 2d. solve normal equations
            dy, eta = self.solver.solve(M, r_corr, mu=mu)

            # 2e. recover Δx, Δs
            #     Δx = -x + σμ/s - (x/s)*rd + (x/s)*A^T Δy - (Δx_aff·Δs_aff)/s
            #     Δs = rd - A^T Δy
            dx = (
                -x
                + sigma * mu / s
                - (x / s) * rd
                + (x / s) * (A.T @ dy)
                - dx_aff * ds_aff / s
            )
            ds = rd - A.T @ dy

            # 2f. step lengths
            alpha_p = self._step_length(x, dx)
            alpha_d = self._step_length(s, ds)

            # 2g. update
            x = x + alpha_p * dx
            s = s + alpha_d * ds
            y = y + alpha_d * dy  # dual step uses α_d (same as slack)

            # 2h. log
            record = {
                "iter": it,
                "mu": mu,
                "primal_res": primal_res,
                "dual_res": dual_res,
                "eta": eta,
                "cond_M": cond_M,
                "solve_time": self.solver.solve_time,
                "alpha_p": alpha_p,
                "alpha_d": alpha_d,
                "sigma": sigma,
            }
            history.append(record)

            if self.verbose:
                print(
                    f"{it:4d}  {mu:10.2e}  {primal_res:10.2e}  "
                    f"{dual_res:10.2e}  {eta:10.2e}  {cond_M:10.2e}  "
                    f"{alpha_p:7.4f}  {alpha_d:7.4f}  "
                    f"{self.solver.solve_time:8.4f}"
                )

        return IPMResult(
            x=x,
            y=y,
            s=s,
            converged=converged,
            n_iter=len(history),
            history=history,
        )

    # ------------------------------------------------------------------
    # Fraction-to-boundary step length
    # ------------------------------------------------------------------

    def _step_length(
        self,
        v: np.ndarray,
        dv: np.ndarray,
    ) -> float:
        """Largest α ∈ (0, τ] such that v + α·dv ≥ 0.

        Parameters
        ----------
        v : (k,) ndarray
            Current iterate (all entries must be > 0).
        dv : (k,) ndarray
            Step direction.

        Returns
        -------
        float
            Step length α ∈ (0, τ].
        """
        # Only consider components where dv < 0 (moving towards boundary)
        neg = dv < 0
        if not np.any(neg):
            return self.tau  # no boundary hit — full step
        ratios = -v[neg] / dv[neg]
        alpha_max = float(np.min(ratios))
        return min(self.tau * alpha_max, self.tau)


# ===================================================================
# Convenience: run with dense random LP
# ===================================================================

if __name__ == "__main__":
    from inexact_ipm.lp_problem import generate_random_dense_lp
    from inexact_ipm.linear_solver import (
        BlockJacobiPCGSolver,
        ExactCholeskySolver,
        LowPrecisionCholeskySolver,
    )

    A, b, c = generate_random_dense_lp(m=20, n=50, seed=42)

    for name, slv in [
        ("ExactCholesky", ExactCholeskySolver()),
        ("LowPrecisionCholesky", LowPrecisionCholeskySolver()),
        ("BlockJacobiPCG(4)", BlockJacobiPCGSolver(num_blocks=4)),
    ]:
        print(f"\n{'='*70}\n  {name}\n{'='*70}")
        ipm = IPMSolver(A, b, c, solver=slv, tol=1e-8, max_iter=40)
        result = ipm.solve()
        final = result.history[-1]
        print(
            f"  iters={result.n_iter:3d}  converged={result.converged}  "
            f"final_mu={final['mu']:.2e}  final_pr_inf={final['primal_res']:.2e}"
        )
