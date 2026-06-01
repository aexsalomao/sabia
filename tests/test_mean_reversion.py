"""Reference-value and degenerate-input tests for the mean-reversion family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sabia.mean_reversion import bollinger_pctb, dist_ma, half_life, zdist
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


def test_zdist_is_zero_at_the_rolling_mean() -> None:
    # Window [98, 102, 100] has mean 100; the last close (100) equals it -> z = 0.
    out = _frame([98.0, 102.0, 100.0]).select(zdist(window=3)).to_series()
    assert out[2] == pytest.approx(0.0, abs=TOL)


def test_bollinger_pctb_is_half_at_the_mean() -> None:
    out = _frame([98.0, 102.0, 100.0]).select(bollinger_pctb(window=3)).to_series()
    assert out[2] == pytest.approx(0.5, abs=TOL)


def test_dist_ma_is_zero_when_price_equals_average() -> None:
    out = _frame([100.0] * 60).select(dist_ma(window=50)).to_series()
    assert out[-1] == pytest.approx(0.0, abs=TOL)


def test_dist_ma_is_positive_above_the_average() -> None:
    closes = [100.0] * 49 + [110.0]
    out = _frame(closes).select(dist_ma(window=50)).to_series()
    assert out[49] > 0.0


def test_half_life_finite_and_positive_for_mean_reverting_series() -> None:
    import numpy as np

    # AR(1) reverting to 100 with slope beta = -0.2 (within the mean-reverting band).
    rng = np.random.default_rng(0)
    x = [100.0]
    for _ in range(150):
        x.append(x[-1] - 0.2 * (x[-1] - 100.0) + rng.normal(0.0, 1.0))
    out = _frame(x).select(half_life(window=60)).to_series().drop_nulls()
    assert out.len() > 0
    assert out.min() > 0.0


def test_half_life_is_null_for_a_pure_trend() -> None:
    closes = [100.0 + i for i in range(120)]  # monotone trend -> not mean-reverting
    out = _frame(closes).select(half_life(window=60)).to_series()
    assert out.drop_nulls().len() == 0
