"""Reference-value and degenerate-input tests for the mean-reversion family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import CLOSE, SCHEMA, SYMBOL, TIMESTAMP

from sabia.mean_reversion import autocorr, half_life, var_ratio, zscore_close

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


def test_zscore_close_is_zero_at_the_rolling_mean() -> None:
    # Window [98, 102, 100] has mean 100; the last close (100) equals it -> z = 0.
    out = _frame([98.0, 102.0, 100.0]).select(zscore_close(window=3).expr(SCHEMA)).to_series()
    assert out[2] == pytest.approx(0.0, abs=TOL)


def test_zscore_close_flat_window_is_null_not_inf() -> None:
    out = _frame([100.0] * 5).select(zscore_close(window=3).expr(SCHEMA)).to_series()
    assert out[-1] is None


def test_autocorr_is_negative_for_alternating_returns() -> None:
    # Up/down/up/down closes give returns that flip sign each bar -> negative lag-1 autocorrelation.
    closes = [100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0]
    out = _frame(closes).select(autocorr(lag=1, window=5).expr(SCHEMA)).to_series().drop_nulls()
    assert out.max() < 0.0


def test_autocorr_stays_within_unit_interval() -> None:
    closes = [100.0 + (i % 5) for i in range(40)]
    out = (
        _frame([float(c) for c in closes])
        .select(autocorr(lag=1, window=10).expr(SCHEMA))
        .to_series()
        .drop_nulls()
    )
    assert out.min() >= -1.0 - TOL and out.max() <= 1.0 + TOL


def test_var_ratio_flat_returns_is_null() -> None:
    # Flat prices -> zero one-bar return variance -> null, never inf.
    closes = [100.0] * 30
    out = _frame(closes).select(var_ratio(q=2, window=10).expr(SCHEMA)).to_series()
    assert out[-1] is None


def test_var_ratio_is_positive_and_finite_on_varied_series() -> None:
    closes = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0, 95.0, 106.0]
    out = _frame(closes).select(var_ratio(q=2, window=5).expr(SCHEMA)).to_series().drop_nulls()
    assert out.len() > 0
    assert out.min() > 0.0 and out.is_finite().all()


def test_half_life_finite_and_positive_for_mean_reverting_series() -> None:
    import numpy as np

    # AR(1) reverting to 100 with slope beta = -0.2 (within the mean-reverting band).
    rng = np.random.default_rng(0)
    x = [100.0]
    for _ in range(150):
        x.append(x[-1] - 0.2 * (x[-1] - 100.0) + rng.normal(0.0, 1.0))
    out = _frame(x).select(half_life(window=60).expr(SCHEMA)).to_series().drop_nulls()
    assert out.len() > 0
    assert out.min() > 0.0


def test_half_life_is_null_for_a_pure_trend() -> None:
    closes = [100.0 + i for i in range(120)]  # monotone trend -> not mean-reverting
    out = _frame(closes).select(half_life(window=60).expr(SCHEMA)).to_series()
    assert out.drop_nulls().len() == 0
