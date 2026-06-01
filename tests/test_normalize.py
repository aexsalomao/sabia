"""Tests for the normalization layer (zscore, xs_zscore, xs_rank, frac_diff)."""

from datetime import UTC, datetime

import polars as pl
import pytest

from sabia.normalize import frac_diff, xs_rank, xs_zscore, zscore
from sabia.spec import DEFAULT_FLOAT_TOLERANCE, Column

TOL = DEFAULT_FLOAT_TOLERANCE


def _ts(n: int) -> list[datetime]:
    return [datetime(2024, 1, 1 + i, tzinfo=UTC) for i in range(n)]


# --- zscore ------------------------------------------------------------------------------------


def test_zscore_matches_hand_computed_value() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = df.select(zscore(pl.col("x"), window=3)).to_series()
    # At index 2: window (1,2,3) -> mean 2, sample std 1 -> (3 - 2) / 1 = 1.0
    assert out[2] == pytest.approx(1.0, abs=TOL)


def test_zscore_leading_values_are_null() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out = df.select(zscore(pl.col("x"), window=3)).to_series()
    assert out[0] is None and out[1] is None


def test_zscore_flat_window_yields_null_not_inf() -> None:
    df = pl.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
    out = df.select(zscore(pl.col("x"), window=3)).to_series()
    assert out[3] is None


def test_zscore_over_does_not_bleed_across_symbols() -> None:
    df = pl.DataFrame(
        {
            Column.SYMBOL: ["A", "A", "A", "B", "B", "B"],
            "x": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        }
    )
    out = df.select(zscore(pl.col("x"), window=3, over=Column.SYMBOL)).to_series()
    # B's first window starts fresh: its index-3 value must be null, not pulled from A.
    assert out[3] is None
    assert out[5] == pytest.approx(1.0, abs=TOL)


# --- xs_zscore / xs_rank -----------------------------------------------------------------------


def test_xs_zscore_is_centered_within_each_timestamp() -> None:
    df = pl.DataFrame(
        {
            Column.TIMESTAMP: _ts(1) * 3,
            Column.SYMBOL: ["A", "B", "C"],
            "x": [1.0, 2.0, 3.0],
        }
    )
    out = df.select(xs_zscore(pl.col("x"))).to_series()
    assert out.mean() == pytest.approx(0.0, abs=TOL)


def test_xs_rank_is_monotone_in_value() -> None:
    df = pl.DataFrame(
        {
            Column.TIMESTAMP: _ts(1) * 3,
            Column.SYMBOL: ["A", "B", "C"],
            "x": [10.0, 30.0, 20.0],
        }
    )
    out = df.select(xs_rank(pl.col("x"))).to_series()
    # Ranks: 10 -> 1/3, 20 -> 2/3, 30 -> 3/3.
    assert out.to_list() == pytest.approx([1 / 3, 1.0, 2 / 3], abs=TOL)


def test_xs_rank_does_not_pool_across_time() -> None:
    df = pl.DataFrame(
        {
            Column.TIMESTAMP: _ts(2) + _ts(2),
            Column.SYMBOL: ["A", "A", "B", "B"],
            "x": [1.0, 2.0, 100.0, 200.0],
        }
    ).sort(Column.TIMESTAMP, Column.SYMBOL)
    out = df.with_columns(rank=xs_rank(pl.col("x")))
    # At each timestamp there are two symbols, so ranks are exactly {0.5, 1.0}.
    assert set(out.get_column("rank").round(6).to_list()) == {0.5, 1.0}


# --- frac_diff ---------------------------------------------------------------------------------


def test_frac_diff_d_zero_is_identity() -> None:
    df = pl.DataFrame({"x": [1.0, 4.0, 9.0, 16.0]})
    out = df.select(frac_diff(pl.col("x"), 0.0)).to_series()
    assert out.to_list() == pytest.approx([1.0, 4.0, 9.0, 16.0], abs=TOL)


def test_frac_diff_d_one_is_first_difference() -> None:
    df = pl.DataFrame({"x": [1.0, 4.0, 9.0, 16.0]})
    out = df.select(frac_diff(pl.col("x"), 1.0)).to_series()
    expected = df.select(pl.col("x").diff()).to_series()
    assert out.to_list()[1:] == pytest.approx(expected.to_list()[1:], abs=TOL)
    assert out[0] is None


def test_frac_diff_is_causal() -> None:
    base = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    extended = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    out_base = base.select(frac_diff(pl.col("x"), 0.5)).to_series().to_list()
    out_ext = extended.select(frac_diff(pl.col("x"), 0.5)).to_series().to_list()[:5]
    assert out_ext == pytest.approx(out_base, abs=TOL, nan_ok=True)
