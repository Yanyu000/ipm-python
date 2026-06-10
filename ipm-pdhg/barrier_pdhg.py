"""Prototype: solve LP log-barrier subproblems with PDHG.

The internal problem form is

    minimize    c^T x - mu * sum(log(x))
    subject to  A x = b
                x > 0

This is not a production LP solver. It is a compact experiment for replacing the
Newton solve inside a barrier method with matrix-free primal-dual iterations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp


BOUND_INF = 1.0e20


@dataclass(frozen=True)
class StandardLP:
    A: sp.csr_matrix
    b: np.ndarray
    c: np.ndarray
    name: str = ""
    objective_offset: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def num_rows(self) -> int:
        return self.A.shape[0]

    @property
    def num_cols(self) -> int:
        return self.A.shape[1]


@dataclass(frozen=True)
class PdhgOptions:
    max_iter: int = 3000
    tol: float = 1.0e-7
    theta: float = 1.0
    step_safety: float = 0.95
    norm_power_iters: int = 30
    log_every: int = 500
    verbose: bool = True


@dataclass(frozen=True)
class Diagnostics:
    iteration: int
    mu: float
    lp_objective: float
    barrier_objective: float
    primal_residual: float
    stationarity_residual: float
    min_x: float


@dataclass(frozen=True)
class PdhgResult:
    x: np.ndarray
    y: np.ndarray
    mu: float
    tau: float
    sigma: float
    operator_norm: float
    iterations: int
    converged: bool
    history: list[Diagnostics]

    @property
    def final(self) -> Diagnostics:
        return self.history[-1]


def generate_feasible_standard_lp(
    m: int = 20,
    n: int = 60,
    density: float = 0.15,
    seed: int = 0,
) -> StandardLP:
    """Generate a bounded equality-form LP with a strictly positive feasible point."""
    if m < 1:
        raise ValueError("m must be at least 1")
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")

    rng = np.random.default_rng(seed)
    first_row = sp.csr_matrix(np.ones((1, n), dtype=np.float64))

    if m == 1:
        A = first_row
    else:
        tail = sp.random(
            m - 1,
            n,
            density=density,
            format="csr",
            random_state=rng,
            dtype=np.float64,
        )
        tail.data = rng.normal(size=tail.nnz)
        tail.sum_duplicates()
        tail.eliminate_zeros()
        A = sp.vstack([first_row, tail], format="csr")

    x_feasible = rng.uniform(0.05, 2.0, size=n)
    b = np.asarray(A @ x_feasible, dtype=np.float64)
    c = rng.normal(size=n)

    return StandardLP(
        A=A,
        b=b,
        c=c,
        name=f"generated_{m}x{n}_density_{density:g}",
        metadata={
            "seed": seed,
            "feasible_min_x": float(np.min(x_feasible)),
            "feasible_max_x": float(np.max(x_feasible)),
            "feasible_sum_x": float(np.sum(x_feasible)),
        },
    )


def prox_linear_log_barrier(
    v: np.ndarray,
    c: np.ndarray,
    tau: float,
    mu: float,
) -> np.ndarray:
    """Compute prox_{tau * (c^T x - mu sum log x)}(v) over x > 0."""
    z = v - tau * c
    return 0.5 * (z + np.sqrt(z * z + 4.0 * tau * mu))


def estimate_operator_norm(
    A: sp.csr_matrix,
    iterations: int = 30,
    seed: int = 0,
) -> float:
    """Estimate ||A||_2 by power iteration on A^T A."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=A.shape[1])
    norm_x = np.linalg.norm(x)
    if norm_x == 0.0:
        return 0.0
    x /= norm_x

    for _ in range(iterations):
        y = A @ x
        x = A.T @ y
        norm_x = np.linalg.norm(x)
        if norm_x == 0.0:
            return 0.0
        x /= norm_x

    return float(np.linalg.norm(A @ x))


def default_initial_x(problem: StandardLP) -> np.ndarray:
    """Pick a positive starting point, matching an all-positive first row when easy."""
    n = problem.num_cols
    x0 = np.ones(n, dtype=np.float64)
    if problem.num_rows == 0:
        return x0

    row = problem.A.getrow(0)
    if row.nnz == n and np.all(row.data > 0.0) and problem.b[0] > 0.0:
        scale = float(problem.b[0] / row.data.sum())
        if scale > 0.0 and np.isfinite(scale):
            x0.fill(scale)
    return x0


def compute_diagnostics(
    problem: StandardLP,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    iteration: int,
) -> Diagnostics:
    Ax_minus_b = problem.A @ x - problem.b
    stationarity = problem.c - mu / x + problem.A.T @ y

    primal_den = 1.0 + np.linalg.norm(problem.b)
    stationarity_den = 1.0 + np.linalg.norm(problem.c)
    lp_objective = float(problem.objective_offset + problem.c @ x)
    barrier_objective = float(lp_objective - mu * np.sum(np.log(x)))

    return Diagnostics(
        iteration=iteration,
        mu=mu,
        lp_objective=lp_objective,
        barrier_objective=barrier_objective,
        primal_residual=float(np.linalg.norm(Ax_minus_b) / primal_den),
        stationarity_residual=float(np.linalg.norm(stationarity) / stationarity_den),
        min_x=float(np.min(x)),
    )


def print_diagnostics_header(mu: float) -> None:
    print(f"\nmu = {mu:g}")
    print(
        " iter |     lp obj | barrier obj |  primal res | station res |      min x"
    )
    print("-" * 76)


def print_diagnostics(row: Diagnostics) -> None:
    print(
        f"{row.iteration:5d} | "
        f"{row.lp_objective:10.4e} | "
        f"{row.barrier_objective:11.4e} | "
        f"{row.primal_residual:11.4e} | "
        f"{row.stationarity_residual:11.4e} | "
        f"{row.min_x:10.4e}"
    )


def solve_barrier_pdhg(
    problem: StandardLP,
    mu: float,
    options: PdhgOptions,
    x0: np.ndarray | None = None,
    y0: np.ndarray | None = None,
) -> PdhgResult:
    """Solve one fixed-mu barrier subproblem using PDHG."""
    if mu <= 0.0:
        raise ValueError("mu must be positive")

    A = problem.A
    operator_norm = estimate_operator_norm(A, options.norm_power_iters)
    if operator_norm == 0.0:
        tau = sigma = 1.0
    else:
        tau = sigma = options.step_safety / operator_norm

    x = default_initial_x(problem) if x0 is None else np.maximum(x0.copy(), 1.0e-14)
    y = np.zeros(problem.num_rows, dtype=np.float64) if y0 is None else y0.copy()
    x_bar = x.copy()
    history: list[Diagnostics] = []
    converged = False

    if options.verbose:
        print_diagnostics_header(mu)

    for iteration in range(options.max_iter + 1):
        should_log = (
            iteration == 0
            or iteration == options.max_iter
            or iteration % options.log_every == 0
        )
        if should_log:
            row = compute_diagnostics(problem, x, y, mu, iteration)
            history.append(row)
            if options.verbose:
                print_diagnostics(row)
            if (
                row.primal_residual <= options.tol
                and row.stationarity_residual <= options.tol
            ):
                converged = True
                break

        y_next = y + sigma * (A @ x_bar - problem.b)
        x_next = prox_linear_log_barrier(x - tau * (A.T @ y_next), problem.c, tau, mu)
        x_bar = x_next + options.theta * (x_next - x)
        x = x_next
        y = y_next

    if not history or history[-1].iteration != iteration:
        history.append(compute_diagnostics(problem, x, y, mu, iteration))

    return PdhgResult(
        x=x,
        y=y,
        mu=mu,
        tau=tau,
        sigma=sigma,
        operator_norm=operator_norm,
        iterations=iteration,
        converged=converged,
        history=history,
    )


def solve_barrier_path(
    problem: StandardLP,
    mus: Iterable[float],
    options: PdhgOptions,
) -> list[PdhgResult]:
    """Warm-start a sequence of fixed-mu PDHG solves."""
    results: list[PdhgResult] = []
    x: np.ndarray | None = None
    y: np.ndarray | None = None

    for mu in mus:
        result = solve_barrier_pdhg(problem, mu, options, x0=x, y0=y)
        results.append(result)
        x = result.x
        y = result.y

    return results


def _is_finite_bound(value: float) -> bool:
    return bool(np.isfinite(value) and abs(value) < BOUND_INF)


def _highs_matrix_to_scipy(lp: object) -> sp.csr_matrix:
    import highspy

    m = int(lp.num_row_)
    n = int(lp.num_col_)
    matrix = lp.a_matrix_
    start = np.asarray(matrix.start_, dtype=np.int64)
    index = np.asarray(matrix.index_, dtype=np.int64)
    value = np.asarray(matrix.value_, dtype=np.float64)

    if matrix.format_ == highspy.MatrixFormat.kColwise:
        return sp.csc_matrix((value, index, start), shape=(m, n)).tocsr()
    if matrix.format_ == highspy.MatrixFormat.kRowwise:
        return sp.csr_matrix((value, index, start), shape=(m, n))
    raise ValueError(f"Unsupported HiGHS matrix format: {matrix.format_}")


def read_mps_as_standard_form(path: str | Path) -> StandardLP:
    """Read a simple continuous MPS model and convert it to equality form.

    Supported variables must have finite lower bounds. Finite row and column
    upper bounds are converted with nonnegative slack variables.
    """
    import highspy

    path = Path(path)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(str(path))
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"Could not read MPS model: {path}")

    lp = h.getLp()
    A = _highs_matrix_to_scipy(lp)
    c = np.asarray(lp.col_cost_, dtype=np.float64)
    col_lower = np.asarray(lp.col_lower_, dtype=np.float64)
    col_upper = np.asarray(lp.col_upper_, dtype=np.float64)
    row_lower = np.asarray(lp.row_lower_, dtype=np.float64)
    row_upper = np.asarray(lp.row_upper_, dtype=np.float64)
    objective_offset = float(getattr(lp, "offset_", 0.0))

    if hasattr(highspy, "ObjSense") and lp.sense_ == highspy.ObjSense.kMaximize:
        c = -c
        objective_offset = -objective_offset

    finite_col_lower = np.array([_is_finite_bound(v) for v in col_lower])
    if not np.all(finite_col_lower):
        unsupported = np.flatnonzero(~finite_col_lower)[:5]
        raise ValueError(
            "This prototype requires finite variable lower bounds. "
            f"Unsupported column examples: {unsupported.tolist()}"
        )

    shift = col_lower
    shifted_row_activity = np.asarray(A @ shift, dtype=np.float64)
    shifted_row_lower = row_lower - shifted_row_activity
    shifted_row_upper = row_upper - shifted_row_activity
    objective_offset += float(c @ shift)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    slack_rows: list[int] = []

    def add_constraint(
        indices: np.ndarray,
        values: np.ndarray,
        rhs_value: float,
        with_slack: bool,
    ) -> None:
        row_index = len(rhs)
        rows.extend([row_index] * len(indices))
        cols.extend(indices.tolist())
        data.extend(values.tolist())
        rhs.append(float(rhs_value))
        if with_slack:
            slack_rows.append(row_index)

    for i in range(A.shape[0]):
        row = A.getrow(i)
        lower_finite = _is_finite_bound(shifted_row_lower[i])
        upper_finite = _is_finite_bound(shifted_row_upper[i])

        if lower_finite and upper_finite and np.isclose(
            shifted_row_lower[i],
            shifted_row_upper[i],
            rtol=1.0e-9,
            atol=1.0e-10,
        ):
            add_constraint(row.indices, row.data, shifted_row_upper[i], False)
            continue

        if upper_finite:
            add_constraint(row.indices, row.data, shifted_row_upper[i], True)
        if lower_finite:
            add_constraint(row.indices, -row.data, -shifted_row_lower[i], True)

    for j, upper in enumerate(col_upper):
        if not _is_finite_bound(upper):
            continue
        width = float(upper - col_lower[j])
        if width <= 0.0:
            raise ValueError(
                "Fixed or contradictory variable bounds are not yet supported "
                f"by the strict-positive barrier prototype. Column {j} has "
                f"width {width:g}."
            )
        add_constraint(
            np.asarray([j], dtype=np.int64),
            np.asarray([1.0], dtype=np.float64),
            width,
            True,
        )

    n_original = A.shape[1]
    n_slack = len(slack_rows)
    for slack_index, row_index in enumerate(slack_rows):
        rows.append(row_index)
        cols.append(n_original + slack_index)
        data.append(1.0)

    A_standard = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(rhs), n_original + n_slack),
        dtype=np.float64,
    )
    c_standard = np.concatenate([c, np.zeros(n_slack, dtype=np.float64)])

    return StandardLP(
        A=A_standard,
        b=np.asarray(rhs, dtype=np.float64),
        c=c_standard,
        name=str(lp.model_name_) or path.stem,
        objective_offset=objective_offset,
        metadata={
            "source_mps": str(path),
            "original_rows": A.shape[0],
            "original_cols": A.shape[1],
            "slack_cols": n_slack,
        },
    )


def parse_mu_list(text: str) -> list[float]:
    mus = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not mus:
        raise ValueError("At least one mu value is required")
    if any(mu <= 0.0 for mu in mus):
        raise ValueError("All mu values must be positive")
    return mus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps", type=Path, help="Optional MPS file to read with highspy")
    parser.add_argument("--m", type=int, default=20, help="Generated LP row count")
    parser.add_argument("--n", type=int, default=60, help="Generated LP column count")
    parser.add_argument(
        "--density",
        type=float,
        default=0.15,
        help="Generated random tail-row density",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--mu",
        default="1e-1,1e-2",
        help="Comma-separated barrier parameters",
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--step-safety", type=float, default=0.95)
    parser.add_argument("--norm-power-iters", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--quiet", action="store_true", help="Only print final summaries")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mps is None:
        problem = generate_feasible_standard_lp(args.m, args.n, args.density, args.seed)
    else:
        problem = read_mps_as_standard_form(args.mps)

    mus = parse_mu_list(args.mu)
    options = PdhgOptions(
        max_iter=args.max_iter,
        tol=args.tol,
        theta=args.theta,
        step_safety=args.step_safety,
        norm_power_iters=args.norm_power_iters,
        log_every=args.log_every,
        verbose=not args.quiet,
    )

    print(
        f"Problem {problem.name}: rows={problem.num_rows}, cols={problem.num_cols}, "
        f"nnz={problem.A.nnz}"
    )
    if problem.metadata:
        print(f"Metadata: {problem.metadata}")

    results = solve_barrier_path(problem, mus, options)

    print("\nSummary")
    print("    mu | iters | conv |     lp obj |  primal res | station res |      min x")
    print("-" * 78)
    for result in results:
        row = result.final
        print(
            f"{result.mu:6.1e} | "
            f"{result.iterations:5d} | "
            f"{str(result.converged):>4s} | "
            f"{row.lp_objective:10.4e} | "
            f"{row.primal_residual:11.4e} | "
            f"{row.stationarity_residual:11.4e} | "
            f"{row.min_x:10.4e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
