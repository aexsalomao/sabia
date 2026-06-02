"""Degenerate-input contract tests (FEATURES.md 4.5): every feature yields null on a degenerate
window, never inf or NaN. Regression coverage for code-review findings #2 (log-return NaN/inf) and
#3 (mfi saturating on a flat window).
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import CLOSE, HIGH, LOW, SCHEMA, SYMBOL, TIMESTAMP, VOLUME

from sabia._math import log_return
from sabia.distribution import skew, up_down_vol_ratio
from sabia.mean_reversion import autocorr, var_ratio
from sabia.volatility import semivar_down, vol_cc
from sabia.volume import mfi


def _closes(values: list[float]) -> pl.DataFrame:
    n = len(values)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {TIMESTAMP: [start + timedelta(days=i) for i in range(n)], SYMBOL: ["A"] * n, CLOSE: values}
    )


def _ohlcv(close: float, n: int, *, high: float, low: float, vol: float = 1000.0) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            SYMBOL: ["A"] * n,
            HIGH: [high] * n,
            LOW: [low] * n,
            CLOSE: [close] * n,
            VOLUME: [vol] * n,
        }
    )


# --- log_return helper (#2) --------------------------------------------------------------------


def test_log_return_nulls_every_degenerate_ratio() -> None:
    # negative ratio (-5/10), zero numerator over negative (0/-5), and zero base (4/0) -> all null,
    # never NaN or inf. Before the fix, distribution used safe_log(c/c.shift) (inf on zero base) and
    # mean_reversion used safe_div(...).log() (NaN on negative ratio).
    df = _closes([10.0, -5.0, 0.0, 4.0])
    out = df.select(log_return(pl.col(CLOSE), pl.col(CLOSE).shift(1))).to_series()
    assert out.to_list() == [None, None, None, None]


@pytest.mark.parametrize(
    "feature",
    [
        autocorr(lag=1, window=10),
        var_ratio(q=2, window=10),
        skew(window=10),
        semivar_down(window=10),
        vol_cc(window=10),
    ],
    ids=["autocorr", "var_ratio", "skew", "semivar_down", "vol_cc"],
)
def test_log_return_features_never_emit_nonfinite_on_sign_flips(feature) -> None:  # type: ignore[no-untyped-def]
    # Sign-flipping closes drive log-return ratios negative on every bar; the output must be null
    # there, never NaN/inf. (autocorr/var_ratio produced NaN before the safe-log fix.)
    closes = [(10.0 if i % 2 else -10.0) for i in range(60)]
    out = _closes(closes).select(feature.expr(SCHEMA)).to_series().drop_nulls()
    assert out.is_finite().all()


def test_up_down_vol_ratio_null_when_no_downside() -> None:
    closes = [float(i) for i in range(1, 40)]  # strictly increasing -> zero downside -> null
    out = _closes(closes).select(up_down_vol_ratio(window=5).expr(SCHEMA)).to_series()
    assert out[-1] is None


# --- mfi flat-window guard (#3) ----------------------------------------------------------------


def test_mfi_flat_window_is_null() -> None:
    # Flat typical price -> no up- or down-flow -> degenerate -> null (was saturating at 100.0).
    out = _ohlcv(10.0, 30, high=10.0, low=10.0).select(mfi(window=14).expr(SCHEMA)).to_series()
    assert out.drop_nulls().to_list() == [] or out.tail(1).item() is None
    assert out[-1] is None


def test_mfi_saturates_at_100_with_only_upflow() -> None:
    # Strictly rising typical price -> only positive flow -> genuine 100 (not the degenerate case).
    n = 30
    start = datetime(2024, 1, 1, tzinfo=UTC)
    closes = [100.0 + i for i in range(n)]
    df = pl.DataFrame(
        {
            TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            SYMBOL: ["A"] * n,
            HIGH: [c + 1 for c in closes],
            LOW: [c - 1 for c in closes],
            CLOSE: closes,
            VOLUME: [1000.0] * n,
        }
    )
    out = df.select(mfi(window=14).expr(SCHEMA)).to_series().drop_nulls()
    assert out.tail(1).item() == pytest.approx(100.0)
