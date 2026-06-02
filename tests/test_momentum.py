"""Reference-value and degenerate-input tests for the momentum family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import CLOSE, HIGH, LOW, SCHEMA, SYMBOL, TIMESTAMP

from sabia.momentum import roc, rsi, stoch_k, williams_r

PERIOD = 14


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


def _ohlc(highs: list[float], lows: list[float], closes: list[float]) -> pl.DataFrame:
    n = len(closes)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            SYMBOL: ["AAA"] * n,
            HIGH: highs,
            LOW: lows,
            CLOSE: closes,
        }
    )


def _rsi_last(closes: list[float]) -> float | None:
    return _frame(closes).select(rsi(period=PERIOD).expr(SCHEMA)).to_series().tail(1).item()


# RSI emits null until its effective_warmup (~249 bars), so reference series must clear that.
_RSI_BARS = 320


def test_rsi_saturates_at_100_for_pure_gains() -> None:
    closes = [float(i) for i in range(1, _RSI_BARS)]  # strictly increasing
    assert _rsi_last(closes) == pytest.approx(100.0)


def test_rsi_saturates_at_0_for_pure_losses() -> None:
    closes = [float(i) for i in range(_RSI_BARS, 1, -1)]  # strictly decreasing
    assert _rsi_last(closes) == pytest.approx(0.0)


def test_rsi_flat_series_is_null_not_inf() -> None:
    closes = [100.0] * _RSI_BARS
    assert _rsi_last(closes) is None


def test_rsi_stays_within_bounds() -> None:
    closes = [100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 107.0, 110.0] * 40
    out = _frame(closes).select(rsi(period=PERIOD).expr(SCHEMA)).to_series().drop_nulls()
    assert out.min() >= 0.0 and out.max() <= 100.0


def test_roc_matches_percent_change_over_window() -> None:
    closes = [100.0] * 10 + [110.0]
    out = _frame(closes).select(roc(window=10).expr(SCHEMA)).to_series()
    assert out[10] == pytest.approx(0.10)


def test_roc_zero_base_is_null() -> None:
    closes = [0.0] + [100.0] * 10
    out = _frame(closes).select(roc(window=10).expr(SCHEMA)).to_series()
    assert out[10] is None


def test_williams_r_matches_hand_value() -> None:
    n = 14
    df = _ohlc([110.0] * n, [90.0] * n, [100.0] * n)
    out = df.select(williams_r(window=14).expr(SCHEMA)).to_series()
    # -100 * (110 - 100) / (110 - 90) = -50
    assert out[13] == pytest.approx(-50.0)


def test_stoch_k_matches_hand_value() -> None:
    n = 14
    df = _ohlc([110.0] * n, [90.0] * n, [100.0] * n)
    out = df.select(stoch_k(window=14).expr(SCHEMA)).to_series()
    # 100 * (100 - 90) / (110 - 90) = 50
    assert out[13] == pytest.approx(50.0)


def test_stoch_k_flat_range_is_null() -> None:
    n = 14
    df = _ohlc([100.0] * n, [100.0] * n, [100.0] * n)
    out = df.select(stoch_k(window=14).expr(SCHEMA)).to_series()
    assert out[13] is None
