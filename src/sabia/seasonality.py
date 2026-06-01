# Seasonality family: deterministic calendar features of the bar timestamp. Pure Polars datetime
# accessors -- vectorized, no per-row Python and no quando calls in the hot path (FEATURES.md 9).
# All are causal: a bar's calendar position is known at t (unlike "last trading day of month",
# which is only knowable after the month ends, so we use calendar day-of-month proximity instead).

from __future__ import annotations

import polars as pl

from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence

_TURN_OF_MONTH_HEAD = 3
_TURN_OF_MONTH_TAIL = 26


def day_of_week(timestamp: str = Column.TIMESTAMP) -> pl.Expr:
    """Day of week, Monday=1 .. Sunday=7. FINITE. Citation: French (1980) weekend effect."""
    return pl.col(timestamp).dt.weekday().cast(pl.Int8).alias("day_of_week")


def month_of_year(timestamp: str = Column.TIMESTAMP) -> pl.Expr:
    """Calendar month, 1 .. 12. FINITE. Citation: Rozeff & Kinney (1976) January effect."""
    return pl.col(timestamp).dt.month().cast(pl.Int8).alias("month_of_year")


def turn_of_month(timestamp: str = Column.TIMESTAMP) -> pl.Expr:
    """Turn-of-month flag: True near the month boundary by calendar day. FINITE. Ariel (1987)."""
    day = pl.col(timestamp).dt.day()
    return ((day <= _TURN_OF_MONTH_HEAD) | (day >= _TURN_OF_MONTH_TAIL)).alias("turn_of_month")


_INT8 = pl.Int8()
_BOOL = pl.Boolean()

FEATURES: tuple[RegisteredFeature, ...] = (
    make_feature(
        day_of_week,
        build=day_of_week,
        name="day_of_week",
        family=Family.SEASONALITY,
        native_band=(Horizon.SHORT,),
        lookback=1,
        min_history=1,
        recurrence=Recurrence.FINITE,
        effective_warmup=1,
        cost_class=Cost.O1,
        inputs=(Column.TIMESTAMP,),
        output_dtype=_INT8,
        citation="French (1980)",
        params={},
    ),
    make_feature(
        month_of_year,
        build=month_of_year,
        name="month_of_year",
        family=Family.SEASONALITY,
        native_band=(Horizon.LONG,),
        lookback=1,
        min_history=1,
        recurrence=Recurrence.FINITE,
        effective_warmup=1,
        cost_class=Cost.O1,
        inputs=(Column.TIMESTAMP,),
        output_dtype=_INT8,
        citation="Rozeff & Kinney (1976)",
        params={},
    ),
    make_feature(
        turn_of_month,
        build=turn_of_month,
        name="turn_of_month",
        family=Family.SEASONALITY,
        native_band=(Horizon.SHORT, Horizon.MEDIUM),
        lookback=1,
        min_history=1,
        recurrence=Recurrence.FINITE,
        effective_warmup=1,
        cost_class=Cost.O1,
        inputs=(Column.TIMESTAMP,),
        output_dtype=_BOOL,
        citation="Ariel (1987)",
        params={},
    ),
)


__all__ = ["FEATURES", "day_of_week", "month_of_year", "turn_of_month"]
