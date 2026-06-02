"""Tests for the normalization layer (zscore, xs_zscore, xs_rank, frac_diff).

Each transform factory returns a ``BoundTransform``; ``.apply(expr)`` resolves it onto an expr.
"""

from datetime import UTC, datetime

import polars as pl
import pytest

from sabia.normalize import frac_diff, xs_rank, xs_zscore, zscore
from sabia.spec import DEFAULT_FLOAT_TOLERANCE

TOL = DEFAULT_FLOAT_TOLERANCE


def _ts(n: int) -> list[datetime]:
    return [datetime(2024, 1, 1 + i, tzinfo=UTC) for i in range(n)]


# --- zscore ------------------------------------------------------------------------------------


def test_zscore_matches_hand_computed_value() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = df.select(zscore(3).apply(pl.col("x"))).to_series()
    # At index 2: window (1,2,3) -> mean 2, sample std 1 -> (3 - 2) / 1 = 1.0
    assert out[2] == pytest.approx(1.0, abs=TOL)


def test_zscore_leading_values_are_null() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out = df.select(zscore(3).apply(pl.col("x"))).to_series()
    assert out[0] is None and out[1] is None


def test_zscore_flat_window_yields_null_not_inf() -> None:
    df = pl.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
    out = df.select(zscore(3).apply(pl.col("x"))).to_series()
    assert out[3] is None


def test_zscore_over_does_not_bleed_across_symbols() -> None:
    df = pl.DataFrame(
        {
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "x": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        }
    )
    out = df.select(zscore(3, over="symbol").apply(pl.col("x"))).to_series()
    # B's first window starts fresh: its index-3 value must be null, not pulled from A.
    assert out[3] is None
    assert out[5] == pytest.approx(1.0, abs=TOL)


# --- xs_zscore / xs_rank -----------------------------------------------------------------------


def test_xs_zscore_is_centered_within_each_timestamp() -> None:
    df = pl.DataFrame(
        {
            "timestamp": _ts(1) * 3,
            "symbol": ["A", "B", "C"],
            "x": [1.0, 2.0, 3.0],
        }
    )
    out = df.select(xs_zscore().apply(pl.col("x"))).to_series()
    assert out.mean() == pytest.approx(0.0, abs=TOL)


def test_xs_zscore_winsorize_clips_outlier_before_standardizing() -> None:
    # FEATURES.md 4.6: xs_zscore(winsorize=k) clips each slice to mean +/- k*std, then standardizes.
    # Slice x = [1, 2, 3, 100]: mean = 26.5, sample std (ddof=1) = 49.00680224893955.
    #   k=1 bounds = [26.5 - 49.0068, 26.5 + 49.0068] = [-22.50680, 75.50680].
    #   1, 2, 3 are inside the band; 100 clips down to 75.50680224893955.
    # Clipped = [1, 2, 3, 75.50680224893955]: clipped mean = 20.376700562234888,
    #   clipped std (ddof=1) = 36.76246946116165.
    # z = (clipped - clipped_mean) / clipped_std:
    #   1  -> -0.5270783178128373
    #   2  -> -0.499876663118327
    #   3  -> -0.4726750084238167
    #   75.50680... -> 1.4996299893549812
    df = pl.DataFrame(
        {
            "timestamp": _ts(1) * 4,
            "symbol": ["A", "B", "C", "D"],
            "x": [1.0, 2.0, 3.0, 100.0],
        }
    )
    out = df.select(xs_zscore(winsorize=1.0).apply(pl.col("x"))).to_series()
    assert out.to_list() == pytest.approx(
        [-0.5270783178128373, -0.499876663118327, -0.4726750084238167, 1.4996299893549812],
        abs=TOL,
    )
    # The winsorized slice still standardizes to mean 0.
    assert out.mean() == pytest.approx(0.0, abs=TOL)


def test_xs_zscore_winsorize_none_matches_plain_zscore() -> None:
    # Default winsorize=None must preserve the prior (unclipped) behavior exactly.
    df = pl.DataFrame(
        {
            "timestamp": _ts(1) * 4,
            "symbol": ["A", "B", "C", "D"],
            "x": [1.0, 2.0, 3.0, 100.0],
        }
    )
    plain = df.select(xs_zscore().apply(pl.col("x"))).to_series().to_list()
    # mean 26.5, std 49.00680224893955 -> z = (x - 26.5) / 49.00680224893955.
    assert plain == pytest.approx(
        [-0.520335929499497, -0.49993059893088926, -0.47952526836228154, 1.4997917967926677],
        abs=TOL,
    )


def test_xs_rank_is_monotone_in_value() -> None:
    df = pl.DataFrame(
        {
            "timestamp": _ts(1) * 3,
            "symbol": ["A", "B", "C"],
            "x": [10.0, 30.0, 20.0],
        }
    )
    out = df.select(xs_rank().apply(pl.col("x"))).to_series()
    # Ranks: 10 -> 1/3, 20 -> 2/3, 30 -> 3/3.
    assert out.to_list() == pytest.approx([1 / 3, 1.0, 2 / 3], abs=TOL)


def test_xs_rank_does_not_pool_across_time() -> None:
    df = pl.DataFrame(
        {
            "timestamp": _ts(2) + _ts(2),
            "symbol": ["A", "A", "B", "B"],
            "x": [1.0, 2.0, 100.0, 200.0],
        }
    ).sort("timestamp", "symbol")
    out = df.with_columns(rank=xs_rank().apply(pl.col("x")))
    # At each timestamp there are two symbols, so ranks are exactly {0.5, 1.0}.
    assert set(out.get_column("rank").round(6).to_list()) == {0.5, 1.0}


# --- frac_diff ---------------------------------------------------------------------------------


def test_frac_diff_d_zero_is_identity() -> None:
    df = pl.DataFrame({"x": [1.0, 4.0, 9.0, 16.0]})
    out = df.select(frac_diff(0.0).apply(pl.col("x"))).to_series()
    assert out.to_list() == pytest.approx([1.0, 4.0, 9.0, 16.0], abs=TOL)


def test_frac_diff_d_one_is_first_difference() -> None:
    df = pl.DataFrame({"x": [1.0, 4.0, 9.0, 16.0]})
    out = df.select(frac_diff(1.0).apply(pl.col("x"))).to_series()
    expected = df.select(pl.col("x").diff()).to_series()
    assert out.to_list()[1:] == pytest.approx(expected.to_list()[1:], abs=TOL)
    assert out[0] is None


def test_frac_diff_is_causal() -> None:
    base = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    extended = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    out_base = base.select(frac_diff(0.5).apply(pl.col("x"))).to_series().to_list()
    out_ext = extended.select(frac_diff(0.5).apply(pl.col("x"))).to_series().to_list()[:5]
    assert out_ext == pytest.approx(out_base, abs=TOL, nan_ok=True)
