"""Reference-value tests for the seasonality family."""

from datetime import UTC, datetime, timedelta

import polars as pl

from sabia.seasonality import day_of_week, month_of_year, turn_of_month
from sabia.spec import Column


def _frame(dates: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({Column.TIMESTAMP: dates, Column.SYMBOL: ["AAA"] * len(dates)})


def test_day_of_week_is_monday_one_sunday_seven() -> None:
    # 2024-01-01 is a Monday.
    dates = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(7)]
    out = _frame(dates).select(day_of_week()).to_series()
    assert out.to_list() == [1, 2, 3, 4, 5, 6, 7]


def test_day_of_week_is_int8() -> None:
    out = _frame([datetime(2024, 1, 1, tzinfo=UTC)]).select(day_of_week()).to_series()
    assert out.dtype == pl.Int8


def test_month_of_year_matches_calendar_month() -> None:
    dates = [datetime(2024, m, 15, tzinfo=UTC) for m in (1, 6, 12)]
    out = _frame(dates).select(month_of_year()).to_series()
    assert out.to_list() == [1, 6, 12]


def test_turn_of_month_flags_boundaries_only() -> None:
    dates = [datetime(2024, 1, d, tzinfo=UTC) for d in (1, 3, 15, 26, 31)]
    out = _frame(dates).select(turn_of_month()).to_series()
    assert out.to_list() == [True, True, False, True, True]


def test_turn_of_month_is_boolean() -> None:
    out = _frame([datetime(2024, 1, 15, tzinfo=UTC)]).select(turn_of_month()).to_series()
    assert out.dtype == pl.Boolean
