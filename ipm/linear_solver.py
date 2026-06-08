def dense_cholesky_solve(M, rhs):
    m,n = M.shape
    x = np.linalg.solve(M,rhs)

    return x