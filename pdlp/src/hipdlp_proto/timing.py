"""Timing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer():
    start = perf_counter()
    yield lambda: perf_counter() - start