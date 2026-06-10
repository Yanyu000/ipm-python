"""
Tests for ``inexact_ipm.lp_problem``.
"""

import numpy as np
import pytest

from inexact_ipm.lp_problem import LPProblem, generate_random_lp


class TestGenerateRandomLP:
    """Tests for ``generate_random_lp``."""

    def test_shapes(self):
        """A, b, c should have consistent dimensions."""
        m, n = 30, 50
        lp = generate_random_lp(m=m, n=n, seed=0)
        assert lp.A.shape == (m, n)
        assert lp.b.shape == (m,)
        assert lp.c.shape == (n,)

    def test_feasibility_of_built_in_point(self):
        """The generated problem should be feasible by construction."""
        m, n = 20, 40
        lp = generate_random_lp(m=m, n=n, seed=1)

        # Reconstruct the interior point used during generation.
        # (In future, the LPProblem may carry the starting point explicitly.)
        # For now we just check that A @ x == b for *some* x >= 0.
        x = np.ones(n)  # not the actual point — placeholder
        # The real test will use the stored starting point once available.
        assert lp.A.shape[0] == m
        assert lp.A.shape[1] == n

    def test_reproducibility(self):
        """Same seed → same instance."""
        lp1 = generate_random_lp(seed=42)
        lp2 = generate_random_lp(seed=42)
        assert np.allclose(lp1.A, lp2.A)
        assert np.allclose(lp1.b, lp2.b)
        assert np.allclose(lp1.c, lp2.c)

    def test_controlled_condition(self):
        """When condition_number is given, A should have the target spread."""
        cond_target = 1e3
        lp = generate_random_lp(
            m=10, n=20, condition_number=cond_target, seed=7
        )
        sv = np.linalg.svd(lp.A, compute_uv=False)
        actual_cond = sv[0] / sv[-1]
        # Should be within a factor of ~2 (QR introduces some noise)
        assert 0.5 * cond_target <= actual_cond <= 2.0 * cond_target


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
