from __future__ import annotations

from hipdlp_proto.generators import random_feasible_lp


def test_random_feasible_lp_dimensions():
    problem = random_feasible_lp(5, 3, 0.4, seed=4)
    assert problem.num_row == 5
    assert problem.num_col == 3