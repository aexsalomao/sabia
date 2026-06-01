"""Shared pytest fixtures and comparison helpers for the sabia suite.

OHLCV fixtures (single-symbol series and a complete-cross-section panel) plus a null-tolerant,
float-tolerant series comparison used by the cross-cutting invariant harness.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from synthetic import make_panel, make_series

# Long enough to cover the largest default lookback (LONG = 504) plus headroom for warmup.
N_BARS = 600


@pytest.fixture(scope="session")
def series() -> pl.DataFrame:
    return make_series(N_BARS, seed=0)


@pytest.fixture(scope="session")
def panel() -> pl.DataFrame:
    return make_panel(N_BARS, seed=0)


def assert_series_close(
    actual: pl.Series,
    expected: pl.Series,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Assert two series match in length, null mask, and value (within tolerance) on non-nulls."""
    assert actual.len() == expected.len(), f"length {actual.len()} != {expected.len()}"
    actual_nulls = actual.is_null().to_list()
    expected_nulls = expected.is_null().to_list()
    assert actual_nulls == expected_nulls, "null masks differ"
    a = actual.drop_nulls().to_numpy()
    e = expected.drop_nulls().to_numpy()
    # Reject any inf that slipped through (degenerate inputs must yield null, never inf).
    assert np.isfinite(a).all(), "actual contains non-finite values"
    np.testing.assert_allclose(a, e, rtol=rtol, atol=atol)
