import numpy as np
from solver import residuals, step_length

A = np.array([[1., 1.]])
b = np.array([1.])
c = np.array([1., 2.])

x = np.array([0.5, 0.5])
y = np.array([0.])
s = np.array([1. , 2.])

rp, rd, rc, mu = residuals(A,b,c,x,y,s)

print("rp =", rp)
print("rd =", rd)
print("rc =", rc)
print("mu =", mu)

x = np.array([1., 2., 3.])
dx = np.array([-0.5, 1., -6.])

alpha = step_length(x, dx)
print(alpha)