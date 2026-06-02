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


def test_var_ratio_lo_mackinlay_reference_value() -> None:
    # Lo-MacKinlay (1988) unbiased VR(q=2) over a window of T=5 one-bar returns, hand-computed.
    # closes = [100, 101, 99, 102, 98, 103, 97, 104]; at the last bar (index 7) the window covers
    # the 5 one-bar log returns r1 = ln(c_i/c_{i-1}) for i = 3..7:
    #   r1 = [0.02985296314968113, -0.04000533461369913, 0.0497615095590638,
    #         -0.06001800972625292, 0.0696799206379898]
    #   mean1 = 0.009854209801356536
    #   var_a = sum((r1 - mean1)^2) / (T - 1) = 0.00323493990438461     [unbiased, divisor T-1=4]
    # The T-q+1 = 4 overlapping 2-bar log returns rq = ln(c_i/c_{i-2}) for i = 4..7:
    #   rq = [-0.010152371464017962, 0.009756174945364656,
    #         -0.01025650016718911, 0.00966191091173689]
    #   m = q*(T-q+1)*(1 - q/T) = 2 * 4 * (1 - 2/5) = 4.8     [overlap-bias correction]
    #   var_c = sum((rq - q*mean1)^2) / m = 0.0004144880771969206
    # VR = var_c / var_a = 0.0004144880771969206 / 0.00323493990438461 = 0.1281285246242525
    closes = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0]
    out = _frame(closes).select(var_ratio(q=2, window=5).expr(SCHEMA)).to_series()
    assert out[7] == pytest.approx(0.1281285246242525, abs=TOL)


def test_var_ratio_centers_near_one_for_a_random_walk() -> None:
    import numpy as np

    # The LM bias correction makes VR an unbiased estimator: a random walk should hover near 1.
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, 5000)
    closes = [100.0, *(100.0 * np.exp(np.cumsum(rets))).tolist()]
    out = _frame(closes).select(var_ratio(q=2, window=250).expr(SCHEMA)).to_series().drop_nulls()
    assert out.mean() == pytest.approx(1.0, abs=0.05)


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
