"""
LP problem generation and data structures.

Provides utilities to construct Linear Programming problems of the form

    min  c^T x
    s.t. A x = b,  x >= 0,

with dual

    max  b^T y
    s.t. A^T y + s = c,  s >= 0.

Both random (controlled condition number) and curated test instances are
supported.  The module also contains a lightweight ``LPProblem`` dataclass
that bundles the problem data together for consumption by the IPM solver.

Usage
-----
    from inexact_ipm.lp_problem import generate_random_lp, LPProblem

    lp = generate_random_lp(m=100, n=200, seed=42)
    print(lp.A.shape)   # (100, 200)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path to HiGHS build (for MPS file reading) and Netlib instances
# ---------------------------------------------------------------------------
HIGHS_ROOT = Path(r"C:\Users\Yanyu\Documents\GitHub\HiGHS")
NETLIB_DIR = Path(r"C:\Users\Yanyu\Documents\GitHub\hipdlp2\instances\netlib")


@dataclass
class LPProblem:
    """Canonical-form linear program.

    Primal (standard form):
        min  c^T x
        s.t. A x = b
             x >= 0

    Dual:
        max  b^T y
        s.t. A^T y + s = c
             s >= 0

    Attributes
    ----------
    A : (m, n) ndarray
        Constraint matrix.
    b : (m,) ndarray
        Right-hand side vector.
    c : (n,) ndarray
        Cost vector.
    name : str
        Human-readable label for this instance.
    """

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    name: str = "lp"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def generate_random_lp(
    m: int = 100,
    n: int = 200,
    density: float = 0.3,
    condition_number: float | None = None,
    seed: int | None = None,
) -> LPProblem:
    """Generate a random LP instance in standard form.

    Parameters
    ----------
    m : int
        Number of constraints (rows of A).
    n : int
        Number of variables (columns of A).
    density : float
        Fraction of non-zero entries in the constraint matrix A.
    condition_number : float or None
        If given, A is constructed so that its singular values are
        log-linearly spaced between 1 and ``condition_number``.  When
        ``None``, A is a plain random sparse matrix with no explicit
        conditioning control.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    LPProblem
        A feasible primal-dual pair with a known interior starting point.

    Notes
    -----
    The instance is constructed so that a strictly feasible primal-dual
    point is available analytically, which is essential for a primal-dual
    IPM to start from the interior of the feasible region.
    """
    rng = np.random.default_rng(seed)

    # --- constraint matrix ---
    A = _make_constraint_matrix(m, n, density, condition_number, rng)

    # --- feasible interior point ---
    x = rng.uniform(1.0, 10.0, size=n)   # primal > 0
    y = rng.standard_normal(size=m)       # unconstrained dual variable
    s = rng.uniform(1.0, 10.0, size=n)   # dual slack > 0

    # --- right-hand side and cost ---
    b = A @ x
    c = A.T @ y + s

    return LPProblem(A=A, b=b, c=c, name=f"random_{m}x{n}")


def _make_constraint_matrix(
    m: int,
    n: int,
    density: float,
    condition_number: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the constraint matrix A, optionally controlling conditioning."""
    if condition_number is not None:
        # Build A = U @ diag(sigma) @ V^T with controlled singular values.
        U = rng.standard_normal((m, m))
        U, _ = np.linalg.qr(U)              # orthogonal

        V = rng.standard_normal((n, n))
        V, _ = np.linalg.qr(V)              # orthogonal

        sigma = np.logspace(
            0, np.log10(condition_number), num=min(m, n)
        )
        S = np.zeros((m, n))
        np.fill_diagonal(S, sigma)

        A = U @ S @ V.T
    else:
        A = (rng.uniform(-1, 1, size=(m, n))
             * (rng.random(size=(m, n)) < density))

    return A


def generate_random_dense_lp(
    m: int = 100,
    n: int = 200,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a feasible dense LP: min c^T x  s.t. A x = b, x >= 0.

    The instance is feasible by construction:
        x0 = ones(n),   b = A @ x0,   c is a random positive vector.

    This provides a *primal*-feasible starting point (x0 > 0) but does not
    guarantee a strictly feasible dual pair — useful for testing the IPM
    initial-point heuristic.

    Parameters
    ----------
    m : int
        Number of constraints (rows of A).
    n : int
        Number of variables (columns of A).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    A : (m, n) ndarray
        Dense constraint matrix with entries ~ Uniform(-1, 1).
    b : (m,) ndarray
        Right-hand side vector (b = A @ ones(n)).
    c : (n,) ndarray
        Cost vector with positive entries ~ Uniform(0, 10).
    """
    rng = np.random.default_rng(seed)

    A = rng.uniform(-1.0, 1.0, size=(m, n))
    x0 = np.ones(n)
    b = A @ x0
    c = rng.uniform(0.0, 10.0, size=n)  # positive cost

    return A, b, c


def compute_normal_equation(
    A: np.ndarray,
    D_diag: np.ndarray,
) -> np.ndarray:
    """Form the normal-equations matrix M = A @ diag(d^2) @ A^T.

    In the IPM context, D = X^{-1} S = diag(x)^{-1} diag(s), so
    D_diag = sqrt(x_i / s_i) (or equivalently sqrt(s_i / x_i), depending on
    convention).  This function multiplies by the *squared* diagonal entries
    as expected when forming the SPD normal-equations system.

    Parameters
    ----------
    A : (m, n) ndarray
        Constraint matrix.
    D_diag : (n,) ndarray
        Diagonal scaling vector whose elements are *squared* before use
        (i.e. this function computes A @ diag(D_diag**2) @ A^T).

    Returns
    -------
    M : (m, m) ndarray
        Symmetric positive (semi-)definite matrix.
    """
    # M[i,j] = Σ_k A[i,k] * D_diag[k]^2 * A[j,k]
    #         = (A * D_diag) @ (A * D_diag)^T
    A_scaled = A * D_diag[None, :]
    return A_scaled @ A_scaled.T


def load_netlib_lp(
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a Netlib LP instance from an MPS file using HiGHS.

    Requires a working HiGHS build at ``HIGHS_ROOT`` and MPS files under
    ``NETLIB_DIR``.  The function shells out to HiGHS to read the MPS file
    and extract the constraint matrix, right-hand side, and cost vector in
    standard form.

    Parameters
    ----------
    name : str
        Netlib problem name (e.g. ``"afiro"``, ``"adlittle"``,
        ``"25fv47"``).  The corresponding ``<name>.mps`` file is expected
        in ``NETLIB_DIR``.

    Returns
    -------
    A : (m, n) ndarray
        Constraint matrix (sparse or dense, depending on the instance).
    b : (m,) ndarray
        Right-hand side vector.
    c : (n,) ndarray
        Cost vector.

    Raises
    ------
    NotImplementedError
        Always — this is a stub to be wired up in a later iteration.
    FileNotFoundError
        If the MPS file does not exist under ``NETLIB_DIR``.
    """
    mps_path = NETLIB_DIR / f"{name}.mps"
    if not mps_path.exists():
        raise FileNotFoundError(
            f"MPS file not found: {mps_path}\n"
            f"Available files: {list(NETLIB_DIR.glob('*.mps'))[:10]} ..."
        )

    # TODO: use HiGHS (via subprocess or ctypes) to read the MPS file
    # and return (A, b, c).  Prototype sketch:
    #
    #   import subprocess
    #   result = subprocess.run(
    #       [str(HIGHS_ROOT / "build" / "bin" / "highs"), "--read", str(mps_path)],
    #       capture_output=True, text=True, check=True,
    #   )
    #   # Parse output ...
    #
    raise NotImplementedError(
        f"Netlib loader not yet implemented.  MPS file exists at {mps_path}."
    )


def generate_klee_minty(
    n: int = 5,
) -> LPProblem:
    """Construct the Klee-Minty cube — a worst-case LP for simplex methods.

    The Klee-Minty cube is an LP over n variables with 2 n constraints
    that forces the simplex method (with Dantzig's pivot rule) to visit
    2^n vertices.  It is a useful stress test for IPM implementations
    because the central path has high curvature near the optimum.

    Parameters
    ----------
    n : int
        Problem dimension.

    Returns
    -------
    LPProblem
        The Klee-Minty instance in standard form (requires conversion
        from inequality to equality form with slack variables).
    """
    # We construct the LP in inequality form and then add slacks.
    raise NotImplementedError("Klee-Minty generator not yet implemented")


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate a small feasible LP.
    m, n = 20, 50
    A, b, c = generate_random_dense_lp(m=m, n=n, seed=42)

    print(f"A  shape : {A.shape}")
    print(f"b  shape : {b.shape}")
    print(f"c  shape : {c.shape}")
    print(f"b  range : [{b.min():.4f}, {b.max():.4f}]")
    print(f"c  range : [{c.min():.4f}, {c.max():.4f}]")

    # Build the normal-equation matrix for a typical scaling vector.
    D_diag = np.ones(n)  # placeholder: identity scaling
    M = compute_normal_equation(A, D_diag)
    print(f"\nM = A @ diag(D_diag^2) @ A^T  shape : {M.shape}")

    # Condition number of the normal equations.
    sv = np.linalg.svd(M, compute_uv=False)
    cond_M = sv[0] / sv[-1] if sv[-1] > 1e-15 else np.inf
    print(f"cond(M) : {cond_M:.2e}")

    # Also show condition with a random (positive) D_diag.
    rng = np.random.default_rng(99)
    D_rand = rng.uniform(0.1, 10.0, size=n)
    M_rand = compute_normal_equation(A, D_rand)
    sv_r = np.linalg.svd(M_rand, compute_uv=False)
    cond_M_rand = sv_r[0] / sv_r[-1] if sv_r[-1] > 1e-15 else np.inf
    print(f"cond(M) with random D_diag : {cond_M_rand:.2e}")
