from __future__ import annotations

import pytest


def test_pdhg_stub_is_present():
    from hipdlp_proto.pdhg import solve

    with pytest.raises(NotImplementedError):
        solve()