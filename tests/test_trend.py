"""Reference-value and degenerate-input tests for the trend family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sabia.spec import Column
from sabia.trend import adx, ema, sma

TOL = 1e-9


def _frame(
    closes: list[float], *, highs: list[float] | None = None, lows: list[float] | None = None
) -> pl.DataFrame:
    n = len(closes)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            Column.TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            Column.SYMBOL: ["AAA"] * n,
            Column.HIGH: highs if highs is not None else closes,
            Column.LOW: lows if lows is not None else closes,
            Column.CLOSE: closes,
        }
    )


def test_sma_matches_window_mean() -> None:
    out = _frame([2.0, 4.0, 6.0, 8.0]).select(sma(window=3)).to_series()
    assert out[2] == pytest.approx(4.0, abs=TOL)  # mean(2,4,6)
    assert out[3] == pytest.approx(6.0, abs=TOL)  # mean(4,6,8)


def test_sma_leading_values_are_null() -> None:
    out = _frame([1.0, 2.0, 3.0]).select(sma(window=3)).to_series()
    assert out[0] is None and out[1] is None


def test_ema_seeds_on_first_value_and_tracks_constant() -> None:
    out = _frame([5.0] * 30).select(ema(span=12)).to_series()
    assert out[-1] == pytest.approx(5.0, abs=TOL)  # constant series -> EMA equals the level


def test_adx_is_high_for_a_strong_uptrend() -> None:
    n = 80
    closes = [100.0 + i for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    out = _frame(closes, highs=highs, lows=lows).select(adx(window=14)).to_series()
    # A clean monotone trend drives ADX toward its ceiling.
    assert out[-1] > 90.0


def test_adx_stays_within_bounds() -> None:
    n = 120
    closes = [100.0 + (5 if i % 2 else -5) for i in range(n)]
    highs = [c + 2.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    out = _frame(closes, highs=highs, lows=lows).select(adx(window=14)).to_series().drop_nulls()
    assert out.min() >= 0.0 and out.max() <= 100.0
