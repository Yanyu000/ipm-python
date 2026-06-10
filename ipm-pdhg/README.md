# Barrier PDHG Prototype

This directory is a small experiment for solving the log-barrier subproblem
that appears in an interior point method, but using primal-dual hybrid gradient
(PDHG) instead of Newton linear solves.

The first target is the equality-form LP

```text
minimize    c^T x
subject to  A x = b
            x >= 0
```

For a fixed barrier parameter `mu > 0`, replace the LP by the strictly convex
barrier problem

```text
minimize    c^T x - mu * sum_i log(x_i)
subject to  A x = b
            x > 0
```

An interior point method normally follows the central path by solving a sequence
of these problems for decreasing `mu`. This prototype asks a different question:
can each barrier subproblem be solved by a first-order primal-dual method that
uses only `A @ x` and `A.T @ y`?

## Saddle Point Form

Write

```text
f_mu(x) = c^T x - mu * sum_i log(x_i) + indicator{x > 0}
g(z)    = indicator{z = b}
```

The barrier subproblem is

```text
min_x f_mu(x) + g(A x).
```

PDHG uses the convex conjugate `g*(y) = b^T y` and iterates on the saddle
problem

```text
min_x max_y f_mu(x) + y^T (A x - b).
```

With step sizes `tau`, `sigma` satisfying

```text
tau * sigma * ||A||_2^2 < 1,
```

the Chambolle-Pock style update is

```text
y^{k+1} = y^k + sigma * (A xbar^k - b)
x^{k+1} = prox_{tau f_mu}(x^k - tau * A^T y^{k+1})
xbar^{k+1} = x^{k+1} + theta * (x^{k+1} - x^k)
```

The useful part is that the primal prox is separable and has a closed form. For
each coordinate, solve

```text
min_{x_i > 0} 0.5 / tau * (x_i - v_i)^2 + c_i x_i - mu log(x_i).
```

The positive root is

```text
prox_i(v) = 0.5 * (z_i + sqrt(z_i^2 + 4 * tau * mu))
z_i       = v_i - tau * c_i
```

So every PDHG iteration is just sparse matrix-vector products plus this cheap
coordinate update.

## What To Measure

For a fixed `mu`, the barrier KKT conditions are

```text
A x = b
c - mu ./ x + A^T y = 0
x > 0
```

The prototype reports:

- LP objective `c^T x`.
- Barrier objective `c^T x - mu * sum(log(x))`.
- Relative primal residual `||A x - b|| / (1 + ||b||)`.
- Relative stationarity residual `||c - mu ./ x + A^T y|| / (1 + ||c||)`.
- Minimum primal component, to see whether iterates approach the boundary as
  `mu` decreases.

## Experiment Shape

The script supports two sources of LP data:

1. A generated strictly feasible standard-form LP. The generator includes a
   dense all-ones row, so the feasible region is bounded by a simplex-like
   constraint and the random objective is well behaved.
2. An optional MPS reader through `highspy`. The reader converts a simple
   continuous LP with finite variable lower bounds into standard form by adding
   slack variables for finite row and column upper bounds.

The MPS path is intentionally conservative. It is meant to get real models into
the experiment, not to be a full production presolver. Free variables, integer
variables, and fixed or contradictory variable bounds should be handled later by
a real standard-form conversion or by HiGHS presolve.

## Running

Generated problem:

```powershell
py -3.10 .\ipm-pdhg\barrier_pdhg.py --m 20 --n 60 --density 0.15 --mu 1e-1,1e-2
```

MPS problem, if `highspy` is installed:

```powershell
py -3.10 .\ipm-pdhg\barrier_pdhg.py --mps path\to\model.mps --mu 1e-1,1e-2,1e-3
```

Important options:

```text
--max-iter       PDHG iterations per barrier value
--tol            stopping tolerance for primal and stationarity residuals
--step-safety    safety factor for tau = sigma = step_safety / ||A||
--log-every      print frequency inside each barrier solve
```

## Next Ideas

- Add diagonal primal-dual scaling before estimating `||A||`.
- Add adaptive PDHG step sizes.
- Compare the barrier path with the Newton IPM prototype in `ipm/solver.py`.
- Use batched SpMM in the power method or for candidate step-size trials.
- Preserve the map from standard-form variables back to original MPS variables.
