"""Reference-value tests for the seasonality family (resolved via the UTC SessionCalendar)."""

from datetime import UTC, datetime, timedelta

import polars as pl
from synthetic import SCHEMA, SYMBOL, TIMESTAMP

from sabia.seasonality import season_dow, season_tom


def _frame(dates: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({TIMESTAMP: dates, SYMBOL: ["AAA"] * len(dates)})


def test_season_dow_is_monday_zero_sunday_six() -> None:
    # 2024-01-01 is a Monday; the UTC calendar maps Monday -> 0.
    dates = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(7)]
    out = _frame(dates).select(season_dow().expr(SCHEMA)).to_series()
    assert out.to_list() == [0, 1, 2, 3, 4, 5, 6]


def test_season_dow_is_int8() -> None:
    out = _frame([datetime(2024, 1, 1, tzinfo=UTC)]).select(season_dow().expr(SCHEMA)).to_series()
    assert out.dtype == pl.Int8


def test_season_tom_flags_month_boundaries_only() -> None:
    # January has 31 days; with k=3 the flag is True for days <= 3 or > 28.
    dates = [datetime(2024, 1, d, tzinfo=UTC) for d in (1, 3, 15, 26, 31)]
    out = _frame(dates).select(season_tom(k=3).expr(SCHEMA)).to_series()
    assert out.to_list() == [True, True, False, False, True]


def test_season_tom_is_boolean() -> None:
    out = (
        _frame([datetime(2024, 1, 15, tzinfo=UTC)]).select(season_tom(k=3).expr(SCHEMA)).to_series()
    )
    assert out.dtype == pl.Boolean
