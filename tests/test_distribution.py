"""Reference-value and degenerate-input tests for the distribution family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sabia.distribution import downside_dev, kurtosis, skew
from sabia.spec import Column

TOL = 1e-9


def _frame(closes: list[float]) -> pl.DataFrame:
    n = len(closes)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            Column.TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            Column.SYMBOL: ["AAA"] * n,
            Column.CLOSE: closes,
        }
    )


def test_downside_dev_is_zero_when_all_returns_positive() -> None:
    closes = [float(i) for i in range(1, 30)]  # strictly increasing -> no downside
    out = _frame(closes).select(downside_dev(window=5)).to_series()
    assert out[-1] == pytest.approx(0.0, abs=TOL)


def test_downside_dev_is_positive_with_losses() -> None:
    closes = [100.0, 90.0, 99.0, 89.0, 98.0, 88.0, 97.0, 87.0]
    out = _frame(closes).select(downside_dev(window=4)).to_series().drop_nulls()
    assert out.min() > 0.0


def test_skew_flat_series_is_null_not_nan() -> None:
    out = _frame([100.0] * 80).select(skew(window=63)).to_series()
    assert out[-1] is None


def test_kurtosis_is_finite_on_varied_returns() -> None:
    closes = [100.0 + (i % 7) - 3 for i in range(80)]
    out = _frame([float(c) for c in closes]).select(kurtosis(window=63)).to_series().drop_nulls()
    assert out.is_finite().all()
