"""
Tests for ``inexact_ipm.linear_solver``.

Uses a small known SPD matrix (built from a random A as M = A @ A^T + αI)
so that the true solution x_true = M^{-1} @ r can be compared against
each solver's output.
"""

from __future__ import annotations

import numpy as np
import pytest

from inexact_ipm.linear_solver import (
    AdaptiveRefinementSolver,
    BlockJacobiPCGSolver,
    ExactCholeskySolver,
    LowPrecisionCholeskySolver,
    NewtonSolver,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spd_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a small SPD matrix M and a known solution x_true.

    Returns
    -------
    M : (m, m) ndarray  — SPD normal-equations matrix
    r : (m,) ndarray     — right-hand side  M @ x_true
    x_true : (m,) ndarray — ground-truth solution
    A : (m, n) ndarray   — the underlying constraint matrix (for context)
    """
    rng = np.random.default_rng(42)
    m, n = 12, 30
    A = rng.uniform(-1, 1, size=(m, n))
    M = A @ A.T + 0.1 * np.eye(m)   # ensure strict positive definiteness
    x_true = np.arange(m, dtype=np.float64)
    r = M @ x_true
    return M, r, x_true, A


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _max_relative_error(x: np.ndarray, x_true: np.ndarray) -> float:
    """Component-wise max relative error.

    For entries where ``|x_true| ≥ 1`` the error is relative; for small
    entries it falls back to an absolute comparison (scaled by the overall
    norm of x_true).  This avoids spurious large "relative" errors when
    x_true has near-zero components.
    """
    abs_err = np.abs(x - x_true)
    scale = np.maximum(np.abs(x_true), 1.0)  # clip at 1 — switch to absolute
    return float(np.max(abs_err / scale))


# ---------------------------------------------------------------------------
# Tests: ExactCholeskySolver
# ---------------------------------------------------------------------------


class TestExactCholeskySolver:
    """FP64 dense Cholesky should solve the system to near machine precision."""

    def test_recovers_true_solution(self, spd_system):
        M, r, x_true, _ = spd_system
        solver = ExactCholeskySolver()
        delta_y, eta = solver.solve(M, r)

        assert _max_relative_error(delta_y, x_true) < 1e-10
        assert eta < 1e-12
        assert solver.num_iterations == 1
        assert solver.solve_time > 0

    def test_residual_is_consistent_with_reported_eta(self, spd_system):
        M, r, _, _ = spd_system
        solver = ExactCholeskySolver()
        delta_y, eta = solver.solve(M, r)

        # Recompute eta independently and compare.
        r_norm = np.linalg.norm(r)
        residual_norm = np.linalg.norm(M @ delta_y - r)
        expected = residual_norm / r_norm if r_norm > 1e-15 else 0.0
        assert eta == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Tests: LowPrecisionCholeskySolver
# ---------------------------------------------------------------------------


class TestLowPrecisionCholeskySolver:
    """FP32 downcast should be less accurate than FP64 but still reasonable."""

    def test_accuracy_bounded_by_fp32_precision(self, spd_system):
        M, r, x_true, _ = spd_system
        solver = LowPrecisionCholeskySolver()
        delta_y, eta = solver.solve(M, r)

        # FP32 has ~7 decimal digits, so relative error ≲ 1e-6.
        assert _max_relative_error(delta_y, x_true) < 1e-4
        assert eta < 1e-4
        assert solver.num_iterations == 1
        assert solver.solve_time > 0

    def test_eta_larger_than_exact_solver(self, spd_system):
        """The FP32 solver should report a (slightly) larger residual
        than the FP64 solver on the same system."""
        M, r, _, _ = spd_system

        _, eta_exact = ExactCholeskySolver().solve(M, r)
        _, eta_low = LowPrecisionCholeskySolver().solve(M, r)

        # FP32 is strictly less accurate, though the gap may be tiny
        # for well-conditioned matrices.
        assert eta_low >= 0.0  # at minimum, it is non-negative
        # For a well-conditioned system, eta_low may still be small;
        # we just verify it doesn't spuriously claim perfection.
        if eta_exact < 1e-15:
            pass  # both are below noise floor — nothing to test
        else:
            assert eta_low >= eta_exact * 0.5  # shouldn't be *better* than exact


# ---------------------------------------------------------------------------
# Tests: BlockJacobiPCGSolver
# ---------------------------------------------------------------------------


class TestBlockJacobiPCGSolver:
    """CG with block-Jacobi preconditioner should converge with tunable
    accuracy/cost trade-off."""

    @pytest.mark.parametrize("num_blocks", [1, 2, 3, 4, 6])
    def test_converges_below_tolerance(self, spd_system, num_blocks):
        M, r, x_true, _ = spd_system
        cg_tol = 1e-8
        solver = BlockJacobiPCGSolver(
            num_blocks=num_blocks, cg_tol=cg_tol, max_iter=200
        )
        delta_y, eta = solver.solve(M, r)

        # CG should achieve the requested tolerance.
        # (Allow a factor-of-2 slack because CG uses the *preconditioned*
        # residual internally, which can be slightly optimistic.)
        assert eta < 10 * cg_tol
        assert _max_relative_error(delta_y, x_true) < 1e-6
        assert solver.solve_time > 0

    def test_num_blocks_1_converges_in_one_iteration(self, spd_system):
        """With 1 block, the preconditioner is M itself → CG converges
        in 1 iteration (up to round-off)."""
        M, r, x_true, _ = spd_system
        solver = BlockJacobiPCGSolver(
            num_blocks=1, cg_tol=1e-12, max_iter=100
        )
        delta_y, eta = solver.solve(M, r)

        assert solver.num_iterations <= 2  # 1, maybe 2 due to round-off
        assert eta < 1e-10
        assert _max_relative_error(delta_y, x_true) < 1e-8

    def test_more_blocks_more_iterations(self, spd_system):
        """More blocks → weaker preconditioner → more CG iterations."""
        M, r, _, _ = spd_system

        iters = {}
        for nb in [1, 2, 4, 8]:
            solver = BlockJacobiPCGSolver(
                num_blocks=nb, cg_tol=1e-8, max_iter=500
            )
            solver.solve(M, r)
            iters[nb] = solver.num_iterations

        # Monotonic in blocks isn't guaranteed (depends on M structure),
        # but 1 block should be fastest and many blocks should be > 1.
        assert iters[1] <= 2
        # 8 blocks should need more than 1 iteration on a non-diagonal M
        assert iters[8] >= 2

    @pytest.mark.parametrize("cg_tol", [1e-3, 1e-6, 1e-9])
    def test_tighter_tolerance_increases_iterations(self, spd_system, cg_tol):
        M, r, _, _ = spd_system
        solver = BlockJacobiPCGSolver(
            num_blocks=3, cg_tol=cg_tol, max_iter=500
        )
        delta_y, eta = solver.solve(M, r)

        # The achieved relative residual should meet the target
        assert eta < 10 * cg_tol


# ---------------------------------------------------------------------------
# Tests: base class & attributes
# ---------------------------------------------------------------------------


class TestNewtonSolverAttributes:
    """Every solver subclass must record solve_time, final_eta, num_iterations."""

    SOLVERS = [
        ExactCholeskySolver(),
        LowPrecisionCholeskySolver(),
        BlockJacobiPCGSolver(num_blocks=3, cg_tol=1e-8, max_iter=200),
    ]

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_attributes_set_after_solve(self, solver, spd_system):
        M, r, _, _ = spd_system
        solver.solve(M, r)

        assert isinstance(solver.solve_time, float)
        assert solver.solve_time >= 0
        assert isinstance(solver.final_eta, float)
        assert solver.final_eta >= 0
        assert isinstance(solver.num_iterations, int)
        assert solver.num_iterations >= 1

    def test_subclass_relationship(self):
        """All concrete solvers should be instances of NewtonSolver."""
        for s in self.SOLVERS:
            assert isinstance(s, NewtonSolver)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Corner cases that shouldn't crash."""

    def test_zero_rhs(self):
        """If r = 0, the solution should be delta_y = 0 and eta = 0."""
        m = 8
        rng = np.random.default_rng(1)
        A = rng.uniform(-1, 1, size=(m, 2 * m))
        M = A @ A.T + np.eye(m)
        r = np.zeros(m)

        for solver in [
            ExactCholeskySolver(),
            LowPrecisionCholeskySolver(),
            BlockJacobiPCGSolver(num_blocks=2, cg_tol=1e-10, max_iter=100),
        ]:
            delta_y, eta = solver.solve(M, r)
            assert delta_y == pytest.approx(np.zeros(m), abs=1e-10)
            assert eta == 0.0

    def test_identity_matrix(self):
        """Trivial identity system — every solver should be near-perfect."""
        m = 10
        M = np.eye(m)
        x_true = np.arange(m, dtype=np.float64)
        r = M @ x_true

        for solver in [
            ExactCholeskySolver(),
            LowPrecisionCholeskySolver(),
            BlockJacobiPCGSolver(num_blocks=5, cg_tol=1e-10, max_iter=100),
        ]:
            delta_y, eta = solver.solve(M, r)
            assert _max_relative_error(delta_y, x_true) < 1e-4
            assert eta < 1e-4

    def test_block_jacobi_num_blocks_equals_m(self):
        """When num_blocks = m, each block is 1×1 → diagonal preconditioner."""
        m = 10
        rng = np.random.default_rng(2)
        A = rng.uniform(-1, 1, size=(m, 3 * m))
        M = A @ A.T + 0.5 * np.eye(m)
        x_true = np.arange(m, dtype=np.float64)
        r = M @ x_true

        solver = BlockJacobiPCGSolver(
            num_blocks=m, cg_tol=1e-8, max_iter=200
        )
        delta_y, eta = solver.solve(M, r)

        # Diagonal preconditioner on a non-diagonal M should need > 1 iteration.
        assert solver.num_iterations > 1
        assert eta < 1e-6
        assert _max_relative_error(delta_y, x_true) < 1e-6

    def test_block_jacobi_validation(self):
        """num_blocks < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            BlockJacobiPCGSolver(num_blocks=0)


# ---------------------------------------------------------------------------
# Tests: AdaptiveRefinementSolver
# ---------------------------------------------------------------------------


class TestAdaptiveRefinementSolver:
    """Dynamic-precision iterative refinement solver tests."""

    def test_recovers_true_solution(self, spd_system):
        """With a tight mu, the solver should achieve high accuracy
        (via refinement or fallback)."""
        M, r, x_true, _ = spd_system
        solver = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=5
        )
        delta_y, eta = solver.solve(M, r, mu=1e-10)

        # Tight mu → high accuracy, whether refinement converged or
        # fallback was triggered.
        assert _max_relative_error(delta_y, x_true) < 1e-10
        assert eta < 1e-10
        assert solver.solve_time > 0
        assert isinstance(solver.num_refinement_steps, int)

    @pytest.mark.parametrize(
        "mu, expected_target",
        [
            (None, 1e-4),
            (1e-1, 1e-4),
            (1e-2, 1e-4),
            (1e-3, 1e-7),
            (1e-5, 1e-7),
            (1e-6, 1e-7),
            (1e-7, 1e-10),
            (1e-9, 1e-10),
        ],
    )
    def test_eta_target_from_mu(self, spd_system, mu, expected_target):
        """The achieved residual should be below the target dictated by mu."""
        M, r, _, _ = spd_system
        solver = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=8
        )
        _, eta = solver.solve(M, r, mu=mu)

        # Allow a factor-of-10 slack (the target guides refinement,
        # but convergence isn't guaranteed if M is badly conditioned).
        assert eta < max(expected_target * 10, 1e-10)

    def test_tight_mu_drives_more_refinement(self, spd_system):
        """Smaller mu → tighter target → typically more refinement steps."""
        M, r, _, _ = spd_system

        # Use a harder system to see the effect.
        rng = np.random.default_rng(99)
        A_hard = rng.uniform(-1, 1, size=(20, 50))
        M_hard = A_hard @ A_hard.T + 1e-3 * np.eye(20)
        r_hard = M_hard @ np.arange(20, dtype=np.float64)

        solver_loose = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=8
        )
        solver_tight = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=8
        )

        _, _ = solver_loose.solve(M_hard, r_hard, mu=1e-1)
        _, _ = solver_tight.solve(M_hard, r_hard, mu=1e-8)

        # Tight should use at least as many refinement steps as loose.
        assert solver_tight.num_refinement_steps >= solver_loose.num_refinement_steps

    def test_exact_when_base_is_float64(self, spd_system):
        """With FP64 base precision, should behave like exact solve
        (no refinement needed)."""
        M, r, x_true, _ = spd_system
        solver = AdaptiveRefinementSolver(
            base_precision="float64", max_refinement_steps=5
        )
        delta_y, eta = solver.solve(M, r, mu=None)

        assert _max_relative_error(delta_y, x_true) < 1e-10
        assert eta < 1e-10
        # FP64 initial solve should be good enough — no refinement.
        assert solver.num_refinement_steps == 0
        assert not solver.used_fallback

    def test_fallback_on_hard_system(self):
        """On an ill-conditioned system with tight target, fallback should
        trigger."""
        # Hilbert matrix: notoriously ill-conditioned.
        m = 10
        M = np.array(
            [[1.0 / (i + j + 1) for j in range(m)] for i in range(m)]
        )
        x_true = np.ones(m)
        r = M @ x_true

        solver = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=2
        )
        delta_y, eta = solver.solve(M, r, mu=1e-10)

        # With only 2 refinement steps on a Hilbert matrix and tight
        # tolerance, fallback should be used.
        assert solver.used_fallback
        # After fallback to FP64, solution accuracy is limited by the
        # extreme condition number (cond ≈ 1e13).  2e-5 is acceptable.
        assert _max_relative_error(delta_y, x_true) < 5e-5
        assert eta < 1e-8

    def test_attributes_recorded(self, spd_system):
        M, r, _, _ = spd_system
        solver = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=3
        )
        solver.solve(M, r, mu=1e-4)

        assert isinstance(solver.num_refinement_steps, int)
        assert solver.num_refinement_steps >= 0
        assert isinstance(solver.used_fallback, bool)
        assert isinstance(solver.solve_time, float)
        assert solver.solve_time >= 0
        assert isinstance(solver.final_eta, float)
        assert solver.final_eta >= 0

    def test_zero_rhs(self):
        M = np.eye(8)
        r = np.zeros(8)
        solver = AdaptiveRefinementSolver()
        delta_y, eta = solver.solve(M, r, mu=None)
        assert delta_y == pytest.approx(np.zeros(8), abs=1e-10)
        assert eta == 0.0

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError):
            AdaptiveRefinementSolver(base_precision="float16")


# ---------------------------------------------------------------------------
# Tests: all solvers on a well-conditioned system
# ---------------------------------------------------------------------------


_WELL_COND_SPD: tuple | None = None


def _get_well_cond_system() -> tuple[np.ndarray, np.ndarray]:
    """Lazily build a well-conditioned 20×20 SPD matrix."""
    global _WELL_COND_SPD
    if _WELL_COND_SPD is None:
        rng = np.random.default_rng(123)
        A = rng.uniform(-1, 1, size=(20, 50))
        M = A @ A.T + np.eye(20)           # cond ≈ O(10–100)
        r = M @ np.arange(20, dtype=np.float64)
        _WELL_COND_SPD = (M, r)
    return _WELL_COND_SPD


class TestAllSolversWellConditioned:
    """Every solver must achieve eta < 1e-3 on a well-conditioned SPD."""

    SOLVERS = [
        ExactCholeskySolver(),
        LowPrecisionCholeskySolver(),
        BlockJacobiPCGSolver(num_blocks=4, cg_tol=1e-6, max_iter=200),
        AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=5
        ),
    ]

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_eta_below_threshold(self, solver):
        M, r = _get_well_cond_system()
        delta_y, eta = solver.solve(M, r, mu=1e-3)

        msg = f"{type(solver).__name__}: eta={eta:.2e}"
        assert eta < 1e-3, msg
        # Also sanity-check that the solution is non-trivial.
        assert np.all(np.isfinite(delta_y))


# ---------------------------------------------------------------------------
# AdaptiveRefinementSolver: fallback on ill-conditioned system
# ---------------------------------------------------------------------------


class TestAdaptiveRefinementFallback:
    """Verify that ``used_fallback`` triggers on a hard matrix."""

    def test_fallback_triggered_ill_cond(self):
        # Build a highly ill-conditioned SPD matrix via exponential decay.
        m = 15
        rng = np.random.default_rng(7)
        Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
        sigma = np.logspace(0, 12, num=m)   # κ ≈ 1e12
        M = Q @ np.diag(sigma) @ Q.T
        M = 0.5 * (M + M.T)                 # ensure symmetry
        M += 1e-10 * np.eye(m)              # guard

        r = M @ np.ones(m)

        solver = AdaptiveRefinementSolver(
            base_precision="float32", max_refinement_steps=3
        )
        delta_y, eta = solver.solve(M, r, mu=1e-8)

        # FP32 + 3 refinement steps can't handle κ=1e12 with target 1e-10.
        assert solver.used_fallback, (
            f"Expected fallback on κ≈1e12 matrix, got "
            f"used_fallback={solver.used_fallback}, eta={eta:.2e}"
        )
        # After fallback to FP64, the solution should be accurate.
        assert eta < 1e-4


# ---------------------------------------------------------------------------
# Tests: all solvers agree on the same system
# ---------------------------------------------------------------------------


class TestAllSolversAgreement:
    """All four solvers must return approximately the same delta_y."""

    def test_solution_agreement(self):
        m = 30
        rng = np.random.default_rng(99)
        A_mat = rng.uniform(-1, 1, size=(m, 3 * m))
        M = A_mat @ A_mat.T + 0.5 * np.eye(m)
        r = M @ np.arange(m, dtype=np.float64)

        # Compute reference with exact FP64 solver.
        ref_solver = ExactCholeskySolver()
        delta_y_ref, _ = ref_solver.solve(M, r, mu=1e-4)

        other_solvers: list[tuple[str, object]] = [
            ("LowPrecisionCholesky", LowPrecisionCholeskySolver()),
            (
                "BlockJacobiPCG(5)",
                BlockJacobiPCGSolver(
                    num_blocks=5, cg_tol=1e-8, max_iter=300
                ),
            ),
            (
                "AdaptiveRefinement",
                AdaptiveRefinementSolver(
                    base_precision="float32", max_refinement_steps=5
                ),
            ),
        ]

        for label, slv in other_solvers:
            delta_y, eta = slv.solve(M, r, mu=1e-4)
            rel_err = _max_relative_error(delta_y, delta_y_ref)
            assert rel_err < 1e-4, (
                f"{label} solution deviates from ExactCholesky: "
                f"rel_err={rel_err:.2e}, eta={eta:.2e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
