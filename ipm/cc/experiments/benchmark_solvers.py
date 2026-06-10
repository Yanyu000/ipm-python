"""
Benchmark driver for comparing linear solvers within the IPM framework.

Runs a suite of dense LP instances of increasing size through the IPM
solver, each time with a different ``NewtonSolver`` backend, and produces:

1. Convergence trajectory (mu vs iteration) — one subplot per problem size.
2. Condition number of the normal-equations matrix vs iteration.
3. Linear-solve residual eta vs iteration.
4. Total wall-clock time comparison (bar chart).

All figures are saved to ``experiments/figures/`` in a clean academic
style (serif fonts, no gridlines).  A summary table is printed to stdout.

Usage
-----
    python experiments/benchmark_solvers.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inexact_ipm.ipm_solver import IPMSolver
from inexact_ipm.linear_solver import (
    AdaptiveRefinementSolver,
    BlockJacobiPCGSolver,
    ExactCholeskySolver,
    LowPrecisionCholeskySolver,
)
from inexact_ipm.lp_problem import generate_random_dense_lp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROBLEM_SIZES = [20, 50, 100, 200]          # m values
N_RATIO = 3                                  # n = N_RATIO * m
SEED = 42
IPM_TOL = 1e-8
IPM_MAX_ITER = 60
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# Suppress overflow / divide-by-zero warnings from FP32 solvers in the
# final ill-conditioned IPM iterations.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Font / style setup (academic, clean)
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Distinct, colourblind-friendly palette.
COLORS = {
    "ExactCholesky": "#0072B2",
    "LowPrecisionCholesky": "#E69F00",
    "BlockJacobiPCG(4)": "#009E73",
    "AdaptiveRefinement": "#CC79A7",
}
MARKERS = {
    "ExactCholesky": "o",
    "LowPrecisionCholesky": "s",
    "BlockJacobiPCG(4)": "^",
    "AdaptiveRefinement": "D",
}
LINESTYLES = {
    "ExactCholesky": "-",
    "LowPrecisionCholesky": "--",
    "BlockJacobiPCG(4)": "-.",
    "AdaptiveRefinement": ":",
}

# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------


def _build_solvers() -> list[tuple[str, object]]:
    """Return (label, solver_instance) for all 4 strategies."""
    return [
        ("ExactCholesky", ExactCholeskySolver()),
        ("LowPrecisionCholesky", LowPrecisionCholeskySolver()),
        (
            "BlockJacobiPCG(4)",
            BlockJacobiPCGSolver(num_blocks=4, cg_tol=1e-6, max_iter=500),
        ),
        (
            "AdaptiveRefinement",
            AdaptiveRefinementSolver(
                base_precision="float32", max_refinement_steps=5
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict:
    """Run all solver × size combinations and collect results.

    Returns
    -------
    dict
        Nested mapping:  results[size][label] = {
            "history": list[dict],    # per-iteration IPM diagnostics
            "result": IPMResult,
            "total_time": float,
            "obj_val": float,
        }
    """
    all_results: dict[int, dict[str, dict]] = {}

    for m in PROBLEM_SIZES:
        n = N_RATIO * m
        print(f"\n{'='*60}\n  Generating LP: m={m}, n={n}\n{'='*60}")

        A, b, c = generate_random_dense_lp(m=m, n=n, seed=SEED)
        size_results: dict[str, dict] = {}

        for label, solver in _build_solvers():
            print(f"  {label:25s} ...", end=" ", flush=True)
            t_start = time.perf_counter()

            ipm = IPMSolver(
                A, b, c,
                solver=solver,
                tol=IPM_TOL,
                max_iter=IPM_MAX_ITER,
                tau=0.99,
                verbose=False,
            )
            result = ipm.solve()
            total_time = time.perf_counter() - t_start

            obj_val = float(c @ result.x) if result.x is not None else np.nan

            status = "CONV" if result.converged else "DIV"
            print(
                f"{status}  iters={result.n_iter:3d}  "
                f"mu={result.history[-1]['mu']:.2e}  "
                f"time={total_time:.2f}s"
            )

            size_results[label] = {
                "history": result.history,
                "result": result,
                "total_time": total_time,
                "obj_val": obj_val,
            }

        all_results[m] = size_results

    return all_results


# ---------------------------------------------------------------------------
# Solution verification
# ---------------------------------------------------------------------------


def verify_solutions(all_results: dict) -> None:
    """Assert that all converged solvers give the same objective value."""
    print(f"\n{'='*60}\n  Solution verification\n{'='*60}")

    for m, size_results in all_results.items():
        ref = size_results["ExactCholesky"]["obj_val"]
        print(f"  m={m:4d}  ExactCholesky  obj = {ref:.8e}")
        for label in [
            "LowPrecisionCholesky",
            "BlockJacobiPCG(4)",
            "AdaptiveRefinement",
        ]:
            r = size_results[label]
            if not r["result"].converged:
                print(f"         {label:25s}  did NOT converge — skipping check")
                continue
            diff = abs(r["obj_val"] - ref) / max(abs(ref), 1.0)
            assert diff < 1e-3, (
                f"Objective mismatch for m={m}, {label}: "
                f"{r['obj_val']:.8e} vs ref {ref:.8e} (diff={diff:.2e})"
            )
            print(
                f"         {label:25s}  obj = {r['obj_val']:.8e}  "
                f"(Delta = {diff:.2e})  OK"
            )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_all(all_results: dict) -> None:
    """Generate and save all four figure types."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _plot_convergence(all_results)
    _plot_condition_number(all_results)
    _plot_eta(all_results)
    _plot_total_time(all_results)


def _plot_convergence(all_results: dict) -> None:
    """Figure 1: duality measure mu vs iteration."""
    n_sizes = len(PROBLEM_SIZES)
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 8), sharex=False, sharey=False
    )
    axes = axes.flatten()

    for idx, m in enumerate(PROBLEM_SIZES):
        ax = axes[idx]
        for label in all_results[m]:
            hist = all_results[m][label]["history"]
            mu_vals = [h["mu"] for h in hist]
            ax.semilogy(
                range(len(mu_vals)),
                mu_vals,
                color=COLORS[label],
                marker=MARKERS[label],
                linestyle=LINESTYLES[label],
                markersize=3,
                markevery=max(1, len(mu_vals) // 10),
                linewidth=1.0,
                label=label,
            )
        ax.set_title(f"m = {m}, n = {N_RATIO * m}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\mu$ (duality measure)")
        ax.axhline(IPM_TOL, color="gray", linestyle=":", linewidth=0.8)

    # Shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIGURES_DIR / "convergence_mu.pdf")
    fig.savefig(FIGURES_DIR / "convergence_mu.png")
    plt.close(fig)


def _plot_condition_number(all_results: dict) -> None:
    """Figure 2: condition number of the normal-equations matrix."""
    n_sizes = len(PROBLEM_SIZES)
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 8), sharex=False, sharey=False
    )
    axes = axes.flatten()

    for idx, m in enumerate(PROBLEM_SIZES):
        ax = axes[idx]
        for label in all_results[m]:
            hist = all_results[m][label]["history"]
            cond_vals = [h["cond_M"] for h in hist]
            ax.semilogy(
                range(len(cond_vals)),
                cond_vals,
                color=COLORS[label],
                marker=MARKERS[label],
                linestyle=LINESTYLES[label],
                markersize=3,
                markevery=max(1, len(cond_vals) // 10),
                linewidth=1.0,
                label=label,
            )
        ax.set_title(f"m = {m}, n = {N_RATIO * m}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\kappa(M)$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIGURES_DIR / "condition_number.pdf")
    fig.savefig(FIGURES_DIR / "condition_number.png")
    plt.close(fig)


def _plot_eta(all_results: dict) -> None:
    """Figure 3: linear-solve relative residual eta vs iteration."""
    n_sizes = len(PROBLEM_SIZES)
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 8), sharex=False, sharey=False
    )
    axes = axes.flatten()

    for idx, m in enumerate(PROBLEM_SIZES):
        ax = axes[idx]
        for label in all_results[m]:
            hist = all_results[m][label]["history"]
            eta_vals = [h["eta"] for h in hist]
            ax.semilogy(
                range(len(eta_vals)),
                eta_vals,
                color=COLORS[label],
                marker=MARKERS[label],
                linestyle=LINESTYLES[label],
                markersize=3,
                markevery=max(1, len(eta_vals) // 10),
                linewidth=1.0,
                label=label,
            )
        ax.set_title(f"m = {m}, n = {N_RATIO * m}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\eta$ (linear solve residual)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIGURES_DIR / "eta_residual.pdf")
    fig.savefig(FIGURES_DIR / "eta_residual.png")
    plt.close(fig)


def _plot_total_time(all_results: dict) -> None:
    """Figure 4: total solve time bar chart."""
    n_sizes = len(PROBLEM_SIZES)
    n_solvers = 4
    solver_labels = [
        "ExactCholesky",
        "LowPrecisionCholesky",
        "BlockJacobiPCG(4)",
        "AdaptiveRefinement",
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_sizes)
    width = 0.18

    for i, label in enumerate(solver_labels):
        times = [
            all_results[m][label]["total_time"] for m in PROBLEM_SIZES
        ]
        bars = ax.bar(
            x + i * width - width * 1.5,
            times,
            width,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        # Annotate with iteration count.
        for bar, m in zip(bars, PROBLEM_SIZES):
            n_iter = all_results[m][label]["result"].n_iter
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(times) if max(times) > 0 else 0.1,
                f"{n_iter}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}" for m in PROBLEM_SIZES])
    ax.set_ylabel("Total solve time (s)")
    ax.set_xlabel("Problem size")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "total_time.pdf")
    fig.savefig(FIGURES_DIR / "total_time.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(all_results: dict) -> None:
    """Print a formatted summary table to stdout."""
    header = (
        f"{'Solver':<25s}  {'m':>4s}  {'Iters':>5s}  "
        f"{'Final mu':>10s}  {'Total time':>10s}  {'Avg eta':>10s}  "
        f"{'Conv':>4s}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}\n{header}\n{sep}")

    for label in [
        "ExactCholesky",
        "LowPrecisionCholesky",
        "BlockJacobiPCG(4)",
        "AdaptiveRefinement",
    ]:
        for m in PROBLEM_SIZES:
            r = all_results[m][label]
            hist = r["history"]
            final_mu = hist[-1]["mu"] if hist else np.nan
            avg_eta = (
                np.mean([h["eta"] for h in hist]) if hist else np.nan
            )
            n_iter = r["result"].n_iter
            conv = "CONV" if r["result"].converged else "DIV"

            print(
                f"{label:<25s}  {m:4d}  {n_iter:5d}  "
                f"{final_mu:10.2e}  {r['total_time']:10.3f}  "
                f"{avg_eta:10.2e}  {conv:>4s}"
            )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Inexact IPM — solver benchmark")
    print(f"Problem sizes: m = {PROBLEM_SIZES}, n = {N_RATIO}×m")
    print(f"IPM tolerance: {IPM_TOL}, max iterations: {IPM_MAX_ITER}")
    print(f"Figures saved to: {FIGURES_DIR}")

    results = run_benchmark()
    verify_solutions(results)
    plot_all(results)
    print_summary_table(results)

    print(f"\nDone.  Figures written to {FIGURES_DIR.resolve()}")
