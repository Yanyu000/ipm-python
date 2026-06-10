"""
inexact_ipm — A research prototype for Multi-GPU Mixed-Precision Inexact
Interior Point Methods for Linear Programming.

This package provides a modular implementation of a primal-dual interior point
method where the Newton system at each iteration can be solved inexactly
(e.g., via iterative methods, low-precision arithmetic, or early termination),
enabling the study of trade-offs between per-iteration cost and convergence rate.

Modules
-------
lp_problem     : Generate random LP instances with controlled conditioning.
ipm_solver     : Primal-dual IPM outer loop with predictor-corrector steps.
linear_solver  : Pluggable linear system solvers for the Newton step.
utils          : Residual computation, condition number estimation, logging.
"""

__version__ = "0.1.0"
