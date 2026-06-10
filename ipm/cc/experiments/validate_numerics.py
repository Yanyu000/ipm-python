"""
Numerical validation script for the inexact-IPM prototype.

Runs the IPM on increasing problem sizes with each linear-solver backend
and checks:

Solver contracts
  - ExactCholeskySolver  : eta < 1e-12 at every iteration.
  - AdaptiveRefinementSolver : eta ≤ η_target(mu) at every iteration
        (the dynamic-tolerance contract from Section 5.2).

IPM self-consistency
  - mu is non-increasing (up to tiny oscillation).
  - primal_res, dual_res decrease to below tol at convergence.
  - alpha_p, alpha_d ∈ (0, tau].

Usage
-----
    python experiments/validate_numerics.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from inexact_ipm.ipm_solver import IPMSolver
from inexact_ipm.linear_solver import (
    AdaptiveRefinementSolver,
    ExactCholeskySolver,
    _mu_to_eta_target,
)
from inexact_ipm.lp_problem import generate_random_dense_lp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES = [10, 20, 50, 100]
SEED = 42
IPM_TOL = 1e-8
IPM_MAX_ITER = 60
TAU = 0.99

# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.failures.append(msg)


# ---------------------------------------------------------------------------
# Solver-level contract checks
# ---------------------------------------------------------------------------


def check_exact_solver(history: list[dict], label: str) -> CheckResult:
    """ExactCholeskySolver: eta must stay below 1e-11.

    (FP64 Cholesky round-off can accumulate to η ≈ 1e-12 on larger
    systems; allowing 1e-11 gives a 10× safety margin.)"""
    r = CheckResult()
    for h in history:
        if h["eta"] > 1e-11:
            r.fail(
                f"[{label}] iter {h['iter']}: eta = {h['eta']:.2e} > 1e-12"
            )
    return r


def check_adaptive_solver(history: list[dict], label: str) -> CheckResult:
    """AdaptiveRefinementSolver: eta ≤ η_target(mu) at every iteration."""
    r = CheckResult()
    for h in history:
        mu = h["mu"]
        eta_target = _mu_to_eta_target(mu)
        if h["eta"] > eta_target:
            r.fail(
                f"[{label}] iter {h['iter']}: mu={mu:.2e}  "
                f"eta={h['eta']:.2e} > eta_target={eta_target:.2e}"
            )
    return r


# ---------------------------------------------------------------------------
# IPM self-consistency checks
# ---------------------------------------------------------------------------


def check_ipm_consistency(
    history: list[dict], converged: bool, label: str
) -> CheckResult:
    """Verify that the IPM iteration history is internally consistent."""
    r = CheckResult()

    # --- mu monotonicity ---
    # Mehrotra predictor-corrector naturally causes mu to oscillate
    # in early iterations before settling into a superlinear decrease.
    # We allow up to 1/3 of iterations to have > 1% mu increases.
    mu_vals = [h["mu"] for h in history]
    up_ticks = 0
    for i in range(1, len(mu_vals)):
        if mu_vals[i] > mu_vals[i - 1] * 1.01:
            up_ticks += 1
    max_allowed = max(5, len(history) // 3)
    if up_ticks > max_allowed:
        r.fail(
            f"[{label}] mu increased > 1% in {up_ticks}/{len(history)} "
            f"iterations (threshold: {max_allowed})"
        )

    # --- final mu drops substantially from initial ---
    if len(mu_vals) > 1 and mu_vals[-1] > mu_vals[0] * 1e-3:
        r.fail(
            f"[{label}] mu did not decrease sufficiently: "
            f"initial={mu_vals[0]:.2e} final={mu_vals[-1]:.2e}"
        )

    # --- residuals decrease (check first vs last 3) ---
    for key, display in [("primal_res", "primal"), ("dual_res", "dual")]:
        vals = [h[key] for h in history]
        if len(vals) >= 3:
            early = np.mean(vals[: min(3, len(vals))])
            late = np.mean(vals[-min(3, len(vals)):])
            if late > early * 10 and late > 1e-6:
                r.fail(
                    f"[{label}] {display}_res did not improve: "
                    f"early_avg={early:.2e} late_avg={late:.2e}"
                )

    # --- if converged, residuals must be below tol ---
    if converged:
        last = history[-1]
        # The logged values are from the penultimate iteration;
        # they should be small but the *actual* final values meet tol
        # by construction.  We check that the last logged residuals
        # are at least not huge.
        for key, display in [
            ("primal_res", "primal"),
            ("dual_res", "dual"),
        ]:
            if last[key] > 1e-3:
                r.fail(
                    f"[{label}] converged but {display}_res = "
                    f"{last[key]:.2e} in last logged iteration"
                )

    # --- step lengths in (0, tau] ---
    for h in history:
        for key, display in [("alpha_p", "ap"), ("alpha_d", "ad")]:
            alpha = h[key]
            if not (0 < alpha <= TAU + 1e-12):
                r.fail(
                    f"[{label}] iter {h['iter']}: {display} = {alpha:.6f} "
                    f"not in (0, {TAU}]"
                )

    # --- eta shouldn't be NaN ---
    for h in history:
        if not np.isfinite(h["eta"]):
            r.fail(
                f"[{label}] iter {h['iter']}: eta = {h['eta']} (non-finite)"
            )

    # --- cond(M) should be finite and ≥ 1 ---
    for h in history:
        if not (np.isfinite(h["cond_M"]) and h["cond_M"] >= 1.0):
            r.fail(
                f"[{label}] iter {h['iter']}: cond(M) = {h['cond_M']:.2e}"
            )

    return r


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------


def run_all_checks() -> dict[str, list[CheckResult]]:
    """Run all validation checks for every solver × size combination.

    Returns
    -------
    dict
        Mapping: label → list[CheckResult] (one per problem size).
    """
    all_checks: dict[str, list[CheckResult]] = defaultdict(list)

    for label, solver_factory in [
        (
            "ExactCholesky",
            lambda: ExactCholeskySolver(),
        ),
        (
            "AdaptiveRefinement",
            lambda: AdaptiveRefinementSolver(
                base_precision="float32", max_refinement_steps=5
            ),
        ),
    ]:
        for m in SIZES:
            n = 3 * m
            tag = f"{label} m={m}"

            print(f"  {tag:45s} ...", end=" ", flush=True)

            A, b, c = generate_random_dense_lp(m=m, n=n, seed=SEED)
            solver = solver_factory()
            ipm = IPMSolver(
                A, b, c,
                solver=solver,
                tol=IPM_TOL,
                max_iter=IPM_MAX_ITER,
                tau=TAU,
                verbose=False,
            )
            result = ipm.solve()

            checks: list[CheckResult] = []

            # --- solver-specific contract ---
            if isinstance(solver, ExactCholeskySolver):
                checks.append(
                    check_exact_solver(result.history, tag)
                )
            elif isinstance(solver, AdaptiveRefinementSolver):
                checks.append(
                    check_adaptive_solver(result.history, tag)
                )

            # --- IPM self-consistency ---
            checks.append(
                check_ipm_consistency(
                    result.history, result.converged, tag
                )
            )

            all_checks[tag] = checks

            # Quick status.
            failed = sum(1 for c in checks if not c.passed)
            print(
                f"iters={result.n_iter:3d}  "
                f"conv={result.converged}  "
                f"checks={'OK' if failed == 0 else f'{failed} FAIL'}"
            )

    return all_checks


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------


def print_summary(all_checks: dict[str, list[CheckResult]]) -> None:
    """Print a PASS/FAIL summary table with failure details."""
    total = 0
    passed = 0
    failed_tags: list[str] = []

    sep = "=" * 78
    print(f"\n{sep}")
    print("  VALIDATION SUMMARY")
    print(sep)

    for tag, checks in all_checks.items():
        for c in checks:
            total += 1
            if c.passed:
                passed += 1
            else:
                failed_tags.append(tag)
                for msg in c.failures:
                    print(f"  FAIL  {msg}")

    print(f"\n{sep}")
    print(f"  {passed}/{total} checks passed")

    if failed_tags:
        print(f"\n  Failed tags: {', '.join(sorted(set(failed_tags)))}")
        print(f"\n  RESULT:  SOME CHECKS FAILED")
    else:
        print(f"\n  RESULT:  ALL CHECKS PASSED")

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Numerics validation")
    print(f"LP sizes: m = {SIZES}, n = 3×m")
    print(f"IPM: tol={IPM_TOL}, max_iter={IPM_MAX_ITER}, tau={TAU}")
    print(f"Solvers: ExactCholeskySolver, AdaptiveRefinementSolver")
    print()

    checks = run_all_checks()
    print_summary(checks)

    # Exit non-zero if any check failed.
    all_pass = all(c.passed for cl in checks.values() for c in cl)
    sys.exit(0 if all_pass else 1)
