"""Reference-value and degenerate-input tests for the momentum family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sabia.momentum import rsi
from sabia.spec import Column

PERIOD = 14


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


def _rsi_last(closes: list[float]) -> float | None:
    return _frame(closes).select(rsi(period=PERIOD)).to_series().tail(1).item()


def test_rsi_saturates_at_100_for_pure_gains() -> None:
    closes = [float(i) for i in range(1, 40)]  # strictly increasing
    assert _rsi_last(closes) == pytest.approx(100.0)


def test_rsi_saturates_at_0_for_pure_losses() -> None:
    closes = [float(i) for i in range(40, 1, -1)]  # strictly decreasing
    assert _rsi_last(closes) == pytest.approx(0.0)


def test_rsi_flat_series_is_null_not_inf() -> None:
    closes = [100.0] * 40
    assert _rsi_last(closes) is None


def test_rsi_stays_within_bounds() -> None:
    closes = [100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 107.0, 110.0] * 5
    out = _frame(closes).select(rsi(period=PERIOD)).to_series().drop_nulls()
    assert out.min() >= 0.0 and out.max() <= 100.0
