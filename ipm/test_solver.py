import numpy as np
from solver import residuals, step_length, compute_direction, solve_lp

A = np.array([[1., 1.]])
b = np.array([1.])
c = np.array([1., 2.])

x = np.array([0.5, 0.5])
y = np.array([0.])
s = np.array([1. , 2.])

sigma = 0.1

dx, dy, ds = compute_direction(A,b,c,x,y,s,sigma = sigma)
rp, rd, rc, mu = residuals(A,b,c,x,y,s)

print("rp =", rp)
print("rd =", rd)
print("rc =", rc)
print("mu =", mu)

n = len(x)
e = np.ones(n)

print("check primal equation:", np.linalg.norm(A @ dx + rp))
print("check dual equation:", np.linalg.norm(A.T @ dy + ds + rd))
print("check complementarity equation:", np.linalg.norm(s * dx + x * ds + rc - sigma * mu * e))

x,y,s = solve_lp(A,b,c)

print("final x =", x)
print("final objective =", c @ x)
print("final y =", y)
print("final s =", s)

x = np.array([1., 2., 3.])
dx = np.array([-0.5, 1., -6.])

alpha = step_length(x, dx)
print(alpha)