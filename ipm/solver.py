import numpy as np

def residuals(A,b,c,x,y,s):
    n = len(x)
    rp = A @ x - b
    rd = A.T @ y + s -c
    rc = x * s 
    mu = x @ s / n
    return rp, rd, rc, mu

def step_length(x, dx, tau = 0.99):
    idx = dx < 0
    if not np.any(idx): 
        return 1.0
    return min(1.0, tau * np.min(-x[idx] / dx[idx]))

def compute_direction(A, b, c, x, y, s, sigma = 0.1):
    rp, rd, rc, mu = residuals(A,b,c,x,y,s)
    n = len(x)
    e = np.ones(n)
    D = x / s
    AD = A * D
    M = AD @ A.T
    w = (-rc + sigma * mu * e + x * rd) / s
    rhs = -rp - A @ w
    L = np.linalg.cholesky(M)
    dy = np.linalg.solve(L.T, np.linalg.solve(L,rhs))
    dx = w + D * (A.T @ dy)
    ds = -rd - A.T @ dy
    return dx,dy,ds

def solve_lp(A,b,c, max_iter = 50, tol = 1e-8, verbose = True):
    m,n = A.shape

    x = np.ones(n)
    y = np.zeros(m)
    s = np.ones(n)
    
    for k in range(max_iter):
        rp, rd, rc, mu = residuals(A,b,c,x,y,s)

        primal_res = np.linalg.norm(rp)
        dual_res = np.linalg.norm(rd)
        obj = c @ x

        if verbose:
            print(k, "obj=", obj, " primal=", primal_res, " dual=", dual_res, " mu=", mu)
        if primal_res < tol and dual_res < tol and mu < tol:
            break

        dx, dy, ds = compute_direction(A,b,c,x,y,s,sigma=0.1)

        alpha_p = step_length(x,dx)
        alpha_d = step_length(s,ds)

        x = x + alpha_p * dx
        y = y + alpha_d * dy
        s = s + alpha_d * ds
    return x,y,s
    