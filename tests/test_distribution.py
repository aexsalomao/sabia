"""Reference-value and degenerate-input tests for the distribution family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import CLOSE, SCHEMA, SYMBOL, TIMESTAMP

from sabia.distribution import downside_dev, kurt, skew, up_down_vol_ratio

TOL = 1e-9


def _frame(closes: list[float]) -> pl.DataFrame:
    n = len(closes)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            SYMBOL: ["AAA"] * n,
            CLOSE: closes,
        }
    )


def test_downside_dev_is_zero_when_all_returns_positive() -> None:
    closes = [float(i) for i in range(1, 30)]  # strictly increasing -> no downside
    out = _frame(closes).select(downside_dev(window=5).expr(SCHEMA)).to_series()
    assert out[-1] == pytest.approx(0.0, abs=TOL)


def test_downside_dev_is_positive_with_losses() -> None:
    closes = [100.0, 90.0, 99.0, 89.0, 98.0, 88.0, 97.0, 87.0]
    out = _frame(closes).select(downside_dev(window=4).expr(SCHEMA)).to_series().drop_nulls()
    assert out.min() > 0.0


def test_skew_flat_series_is_null_not_nan() -> None:
    out = _frame([100.0] * 80).select(skew(window=63).expr(SCHEMA)).to_series()
    assert out[-1] is None


def test_kurt_is_finite_on_varied_returns() -> None:
    closes = [100.0 + (i % 7) - 3 for i in range(80)]
    out = _frame([float(c) for c in closes]).select(kurt(window=63).expr(SCHEMA)).to_series()
    assert out.drop_nulls().is_finite().all()


def test_up_down_vol_ratio_is_null_when_no_losses() -> None:
    closes = [float(i) for i in range(1, 30)]  # strictly increasing -> zero downside
    out = _frame(closes).select(up_down_vol_ratio(window=5).expr(SCHEMA)).to_series()
    assert out[-1] is None  # division by zero downside -> null, not inf


def test_up_down_vol_ratio_is_positive_with_mixed_returns() -> None:
    closes = [100.0, 110.0, 99.0, 108.0, 97.0, 106.0, 95.0, 104.0]
    out = _frame(closes).select(up_down_vol_ratio(window=4).expr(SCHEMA)).to_series().drop_nulls()
    assert out.min() > 0.0
