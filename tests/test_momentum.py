"""Reference-value and degenerate-input tests for the momentum family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import CLOSE, HIGH, LOW, SCHEMA, SYMBOL, TIMESTAMP

from sabia.momentum import cci, roc, rsi, stoch_k, williams_r
from sabia.spec import DEFAULT_FLOAT_TOLERANCE

PERIOD = 14
TOL = DEFAULT_FLOAT_TOLERANCE


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


# Canonical Lambert CCI = (TP - SMA(TP)) / (0.015 * MAD), MAD = (1/n)*sum|TP_i - SMA(TP)| with
# SMA(TP) the single current-window mean held constant across all n terms. Below, h = l = c so
# TP = c, and the reference values are hand-computed (see the test docstrings).
@pytest.mark.parametrize(
    ("typical_prices", "expected"),
    [
        # Ramp TP = 1..20: SMA = 10.5; TP_last - SMA = 9.5; sum|i - 10.5| = 100 -> MAD = 5;
        # CCI = 9.5 / (0.015 * 5) = 9.5 / 0.075 = 126.6666...
        ([float(i) for i in range(1, 21)], 126.66666666666667),
        # Flat-then-jump TP = [10]*19 + [20]: SMA = 210/20 = 10.5; TP_last - SMA = 9.5;
        # MAD = (19*0.5 + 9.5)/20 = 19/20 = 0.95; CCI = 9.5 / (0.015 * 0.95) = 666.6666...
        ([10.0] * 19 + [20.0], 666.6666666666667),
    ],
    ids=["ramp", "jump"],
)
def test_cci_matches_canonical_single_window_mad(
    typical_prices: list[float], expected: float
) -> None:
    df = _ohlc(typical_prices, typical_prices, typical_prices)
    out = df.select(cci(window=20).expr(SCHEMA)).to_series()
    assert out[19] == pytest.approx(expected, abs=TOL)


def test_cci_flat_series_is_null() -> None:
    n = 20
    df = _ohlc([100.0] * n, [100.0] * n, [100.0] * n)
    out = df.select(cci(window=20).expr(SCHEMA)).to_series()
    assert out[19] is None


def test_cci_min_history_is_window() -> None:
    assert cci(window=20).spec.min_history == 20
