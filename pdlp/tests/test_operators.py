from __future__ import annotations

import numpy as np

from hipdlp_proto.generators import random_sparse_operator


def test_rmatmat_matches_dense():
    op = random_sparse_operator(6, 4, 0.5, seed=3)
    Y = np.random.default_rng(1).normal(size=(6, 2))
    dense = op.A_csr.toarray()
    np.testing.assert_allclose(op.rmatmat(Y), dense.T @ Y)