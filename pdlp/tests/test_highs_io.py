from __future__ import annotations

import numpy as np

from hipdlp_proto.generators import random_sparse_operator


def test_random_sparse_operator_shape():
    op = random_sparse_operator(4, 3, 0.5, seed=1)
    assert op.shape == (4, 3)


def test_random_sparse_operator_matvec_matches_dense():
    op = random_sparse_operator(5, 4, 0.6, seed=2)
    x = np.random.default_rng(0).normal(size=4)
    dense = op.A_csr.toarray()
    np.testing.assert_allclose(op.matvec(x), dense @ x)