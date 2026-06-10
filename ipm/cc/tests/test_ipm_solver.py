"""
End-to-end integration tests for ``inexact_ipm.ipm_solver``.

Runs the full IPM on a small LP instance and verifies convergence
consistency across different linear solver backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from inexact_ipm.ipm_solver import IPMSolver, IPMResult
from inexact_ipm.linear_solver import (
    AdaptiveRefinementSolver,
    ExactCholeskySolver,
)
from inexact_ipm.lp_problem import generate_random_dense_lp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_lp() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reproducible small LP (m=20, n=50)."""
    A, b, c = generate_random_dense_lp(m=20, n=50, seed=42)
    return A, b, c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_converged(result: IPMResult, label: str) -> None:
    """Check standard convergence criteria on an IPMResult.

    Note: ``result.history[-1]`` contains the *last logged* iteration,
    which is one step *before* the convergence break.  The actual final
    duality measure is therefore re-computed from the solution vectors.
    """
    assert result.converged, (
        f"{label}: IPM did not converge "
        f"(n_iter={result.n_iter})"
    )

    # Compute actual final mu from the solution.
    n = len(result.x)
    final_mu = float(np.dot(result.x, result.s) / n)
    assert final_mu < 1e-8, (
        f"{label}: final mu = {final_mu:.2e} > 1e-8"
    )

    # The logged residuals are from the penultimate iteration, which
    # should already be small.  Allow slightly looser tolerance here.
    last = result.history[-1]
    assert last["primal_res"] < 1e-6, (
        f"{label}: primal_res = {last['primal_res']:.2e}"
    )
    assert last["dual_res"] < 1e-6, (
        f"{label}: dual_res = {last['dual_res']:.2e}"
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIPMSolverIntegration:
    """End-to-end IPM runs with different solver backends."""

    def test_exact_cholesky_converges(self, small_lp):
        """ExactCholeskySolver should converge on a small LP."""
        A, b, c = small_lp
        solver = ExactCholeskySolver()
        ipm = IPMSolver(
            A, b, c, solver=solver, tol=1e-8, max_iter=50, verbose=False
        )
        result = ipm.solve()
        _assert_converged(result, "ExactCholesky")

    def test_adaptive_refinement_converges(self, small_lp):
        """AdaptiveRefinementSolver should also converge on a small LP."""
        A, b, c = small_lp
        solver = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=5
        )
        ipm = IPMSolver(
            A, b, c, solver=solver, tol=1e-8, max_iter=50, verbose=False
        )
        result = ipm.solve()
        _assert_converged(result, "AdaptiveRefinement")

    def test_objective_values_agree(self, small_lp):
        """Both solvers should reach the same objective value."""
        A, b, c = small_lp

        slv_exact = ExactCholeskySolver()
        ipm_exact = IPMSolver(
            A, b, c, solver=slv_exact,
            tol=1e-8, max_iter=50, verbose=False,
        )
        res_exact = ipm_exact.solve()

        slv_adapt = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=5
        )
        ipm_adapt = IPMSolver(
            A, b, c, solver=slv_adapt,
            tol=1e-8, max_iter=50, verbose=False,
        )
        res_adapt = ipm_adapt.solve()

        assert res_exact.converged and res_adapt.converged

        obj_exact = float(c @ res_exact.x)
        obj_adapt = float(c @ res_adapt.x)

        rel_diff = abs(obj_exact - obj_adapt) / max(abs(obj_exact), 1.0)
        assert rel_diff < 1e-6, (
            f"Objective mismatch: exact={obj_exact:.10e}, "
            f"adapt={obj_adapt:.10e}, rel_diff={rel_diff:.2e}"
        )

    def test_ipm_result_attributes(self, small_lp):
        """IPMResult should contain all expected fields."""
        A, b, c = small_lp
        solver = ExactCholeskySolver()
        ipm = IPMSolver(
            A, b, c, solver=solver, tol=1e-8, max_iter=50, verbose=False
        )
        result = ipm.solve()

        assert result.converged
        assert result.n_iter > 0
        assert result.n_iter <= 50
        assert result.x is not None and len(result.x) == 50
        assert result.y is not None and len(result.y) == 20
        assert result.s is not None and len(result.s) == 50
        assert len(result.history) == result.n_iter

        # Spot-check that history records have expected keys.
        h0 = result.history[0]
        for key in (
            "iter", "mu", "primal_res", "dual_res",
            "eta", "cond_M", "solve_time", "alpha_p", "alpha_d", "sigma",
        ):
            assert key in h0, f"history record missing key: {key}"

        # Primal solution should be non-negative (up to tiny feasibility tol).
        assert np.all(result.x > -1e-8), (
            f"primal variable negative: min(x)={result.x.min():.2e}"
        )

    def test_reproducibility(self, small_lp):
        """Same seed → same result (deterministic)."""
        A, b, c = small_lp

        solver1 = ExactCholeskySolver()
        ipm1 = IPMSolver(
            A, b, c, solver=solver1, tol=1e-8, max_iter=50, verbose=False
        )
        r1 = ipm1.solve()

        solver2 = ExactCholeskySolver()
        ipm2 = IPMSolver(
            A, b, c, solver=solver2, tol=1e-8, max_iter=50, verbose=False
        )
        r2 = ipm2.solve()

        assert r1.n_iter == r2.n_iter
        assert np.allclose(r1.x, r2.x, rtol=1e-12, atol=1e-12)
        assert abs(float(c @ r1.x) - float(c @ r2.x)) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
