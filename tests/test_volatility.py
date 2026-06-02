"""Reference-value and degenerate-input tests for the volatility family."""

from datetime import UTC, datetime, timedelta
from math import log, sqrt

import polars as pl
import pytest
from synthetic import CLOSE, HIGH, LOW, OPEN, SCHEMA, SYMBOL, TIMESTAMP

from sabia.volatility import atr, vol_cc, vol_gk, vol_parkinson, vol_rs, vol_yz

TOL = 1e-9


def _ohlc(
    opens: list[float], highs: list[float], lows: list[float], closes: list[float]
) -> pl.DataFrame:
    n = len(closes)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            SYMBOL: ["AAA"] * n,
            OPEN: opens,
            HIGH: highs,
            LOW: lows,
            CLOSE: closes,
        }
    )


def _flat(n: int, price: float = 100.0) -> pl.DataFrame:
    return _ohlc([price] * n, [price] * n, [price] * n, [price] * n)


def test_vol_cc_matches_rolling_std_of_log_returns() -> None:
    closes = [100.0, 110.0, 121.0, 133.1]  # constant +10% log returns
    out = _ohlc(closes, closes, closes, closes).select(vol_cc(window=3).expr(SCHEMA)).to_series()
    # All log returns equal -> rolling std is 0.
    assert out[3] == pytest.approx(0.0, abs=TOL)


def test_parkinson_matches_closed_form() -> None:
    df = _ohlc([100.0, 100.0], [110.0, 110.0], [90.0, 90.0], [100.0, 100.0])
    out = df.select(vol_parkinson(window=2).expr(SCHEMA)).to_series()
    expected = sqrt((log(110.0 / 90.0) ** 2) / (4.0 * log(2.0)))
    assert out[1] == pytest.approx(expected, abs=TOL)


def test_rogers_satchell_is_zero_for_flat_bars() -> None:
    out = _flat(5).select(vol_rs(window=3).expr(SCHEMA)).to_series()
    assert out[4] == pytest.approx(0.0, abs=TOL)


def test_garman_klass_is_finite_and_nonnegative() -> None:
    df = _ohlc(
        [100.0, 101.0, 102.0, 103.0, 104.0],
        [105.0, 106.0, 107.0, 108.0, 109.0],
        [98.0, 99.0, 100.0, 101.0, 102.0],
        [101.0, 102.0, 103.0, 104.0, 105.0],
    )
    out = df.select(vol_gk(window=3).expr(SCHEMA)).to_series().drop_nulls()
    assert out.min() >= 0.0


def test_yang_zhang_is_zero_for_flat_bars() -> None:
    out = _flat(6).select(vol_yz(window=3).expr(SCHEMA)).to_series()
    assert out[5] == pytest.approx(0.0, abs=TOL)


def test_atr_equals_true_range_level_for_constant_range() -> None:
    # Constant high-low range of 10, no gaps -> ATR converges to 10. ATR emits null until its
    # effective_warmup (~249 bars), so the series must clear that.
    n = 320
    out = (
        _ohlc([100.0] * n, [105.0] * n, [95.0] * n, [100.0] * n)
        .select(atr(window=14).expr(SCHEMA))
        .to_series()
    )
    assert out[-1] == pytest.approx(10.0, abs=1e-6)


def test_volatility_leading_values_are_null() -> None:
    out = _flat(10).select(vol_parkinson(window=5).expr(SCHEMA)).to_series()
    assert out.head(4).null_count() == 4
