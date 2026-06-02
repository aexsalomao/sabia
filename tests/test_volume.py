"""Reference-value and degenerate-input tests for the volume family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import CLOSE, DOLLAR_VOLUME, HIGH, LOW, SCHEMA, SYMBOL, TIMESTAMP, VOLUME

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
            TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            SYMBOL: ["AAA"] * n,
            HIGH: highs if highs is not None else closes,
            LOW: lows if lows is not None else closes,
            CLOSE: closes,
            VOLUME: volumes,
            DOLLAR_VOLUME: [c * v for c, v in zip(closes, volumes, strict=True)],
        }
    )


def test_dollar_vol_is_price_times_volume() -> None:
    out = _frame([10.0, 20.0], [100.0, 50.0]).select(dollar_vol().expr(SCHEMA)).to_series()
    assert out.to_list() == pytest.approx([1000.0, 1000.0], abs=TOL)


def test_cmf_is_plus_one_when_close_equals_high() -> None:
    n = 5
    df = _frame([10.0] * n, [1000.0] * n, highs=[10.0] * n, lows=[8.0] * n)
    out = df.select(cmf(window=3).expr(SCHEMA)).to_series()
    assert out[4] == pytest.approx(1.0, abs=TOL)


def test_cmf_is_minus_one_when_close_equals_low() -> None:
    n = 5
    df = _frame([8.0] * n, [1000.0] * n, highs=[10.0] * n, lows=[8.0] * n)
    out = df.select(cmf(window=3).expr(SCHEMA)).to_series()
    assert out[4] == pytest.approx(-1.0, abs=TOL)


def test_cmf_all_flat_range_is_null() -> None:
    # Every bar in the window is a doji -> zero money-flow numerator, but the denominator (summed
    # volume) is non-zero, so CMF is 0.0 (not null): there is genuinely no directional flow.
    n = 5
    df = _frame([10.0] * n, [1000.0] * n, highs=[10.0] * n, lows=[10.0] * n)
    out = df.select(cmf(window=3).expr(SCHEMA)).to_series()
    assert out[4] == pytest.approx(0.0, abs=TOL)


def test_cmf_single_doji_bar_does_not_null_the_window() -> None:
    # A single flat (doji) bar contributes ZERO money flow but still counts its volume in the
    # denominator (canonical Chaikin CMF), so the window emits a value instead of nulling.
    # Window=3 at bar 4 covers bars 2,3,4. Bar 2 is a doji (high==low==10) -> multiplier 0.
    #   bar 2: doji              -> mult 0,    mfv 0
    #   bar 3: (2*10-10-8)/2 = +1 -> mult +1,  mfv +1000
    #   bar 4: (2*8.5-10-8)/2 = -0.5 -> mult -0.5, mfv -500
    # CMF = (0 + 1000 - 500) / (1000 + 1000 + 1000) = 500 / 3000 = 0.16666...
    highs = [10.0, 10.0, 10.0, 10.0, 10.0]
    lows = [8.0, 8.0, 10.0, 8.0, 8.0]
    closes = [10.0, 9.0, 10.0, 10.0, 8.5]
    df = _frame(closes, [1000.0] * 5, highs=highs, lows=lows)
    out = df.select(cmf(window=3).expr(SCHEMA)).to_series()
    assert out[4] == pytest.approx(1.0 / 6.0, abs=TOL)


def test_amihud_uses_absolute_log_return() -> None:
    # Amihud_21 averages |log return| / dollar_volume (returns log unless named, FEATURES.md 4.6).
    # closes = [100, 110, 121], volumes = [10, 10, 10] -> dvol = [1000, 1100, 1210].
    #   bar 1: |ln(110/100)| / 1100 = 0.09531017980432493 / 1100 = 8.664561800393176e-05
    #   bar 2: |ln(121/110)| / 1210 = 0.09531017980432493 / 1210 = 7.876874363993796e-05
    # rolling mean (window=2) at bar 2 = (8.664561800393176e-05 + 7.876874363993796e-05) / 2
    #                                  = 8.270718082193485e-05
    closes = [100.0, 110.0, 121.0]
    volumes = [10.0, 10.0, 10.0]
    out = _frame(closes, volumes).select(amihud(window=2).expr(SCHEMA)).to_series()
    assert out[2] == pytest.approx(8.270718082193485e-05, abs=TOL)


def test_signed_vol_sums_up_volume_in_an_uptrend() -> None:
    closes = [float(i) for i in range(1, 6)]  # strictly increasing -> all +1
    volumes = [100.0] * 5
    out = _frame(closes, volumes).select(signed_vol(window=3).expr(SCHEMA)).to_series()
    assert out[4] == pytest.approx(300.0, abs=TOL)  # 3 bars * 100, all positive


def test_amihud_zero_volume_is_null() -> None:
    closes = [100.0, 101.0, 102.0]
    volumes = [0.0, 0.0, 0.0]
    out = _frame(closes, volumes).select(amihud(window=2).expr(SCHEMA)).to_series()
    assert out[2] is None
