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
        alpha = 1.0
    alpha = min (1.0, tau * np.min(-x[idx] / dx [idx]))
    return alpha