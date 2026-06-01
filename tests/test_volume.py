"""Reference-value and degenerate-input tests for the volume family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sabia.spec import Column
from sabia.volume import amihud, cmf, dollar_vol, signed_vol

TOL = 1e-9


def _frame(
    closes: list[float],
    volumes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
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
            Column.VOLUME: volumes,
        }
    )


def test_dollar_vol_is_price_times_volume() -> None:
    out = _frame([10.0, 20.0], [100.0, 50.0]).select(dollar_vol()).to_series()
    assert out.to_list() == pytest.approx([1000.0, 1000.0], abs=TOL)


def test_cmf_is_plus_one_when_close_equals_high() -> None:
    n = 5
    df = _frame([10.0] * n, [1000.0] * n, highs=[10.0] * n, lows=[8.0] * n)
    out = df.select(cmf(window=3)).to_series()
    assert out[4] == pytest.approx(1.0, abs=TOL)


def test_cmf_is_minus_one_when_close_equals_low() -> None:
    n = 5
    df = _frame([8.0] * n, [1000.0] * n, highs=[10.0] * n, lows=[8.0] * n)
    out = df.select(cmf(window=3)).to_series()
    assert out[4] == pytest.approx(-1.0, abs=TOL)


def test_cmf_flat_range_is_null() -> None:
    n = 5
    df = _frame([10.0] * n, [1000.0] * n, highs=[10.0] * n, lows=[10.0] * n)
    out = df.select(cmf(window=3)).to_series()
    assert out[4] is None


def test_signed_vol_sums_up_volume_in_an_uptrend() -> None:
    closes = [float(i) for i in range(1, 6)]  # strictly increasing -> all +1
    volumes = [100.0] * 5
    out = _frame(closes, volumes).select(signed_vol(window=3)).to_series()
    assert out[4] == pytest.approx(300.0, abs=TOL)  # 3 bars * 100, all positive


def test_amihud_zero_volume_is_null() -> None:
    closes = [100.0, 101.0, 102.0]
    volumes = [0.0, 0.0, 0.0]
    out = _frame(closes, volumes).select(amihud(window=2)).to_series()
    assert out[2] is None
