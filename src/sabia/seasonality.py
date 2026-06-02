# Seasonality family: deterministic calendar position of the bar timestamp. Resolved through the
# SessionCalendar seam (FEATURES.md 4.6, calendar.py) so no 252 or weekday convention is hardcoded
# in feature code; v1's UtcCalendar is a calendar-day approximation, an exchange calendar arrives
# later as a quando adapter. All causal: a bar's calendar position is known at t. Timestamp is a
# fixed canonical column (FEATURES.md 2.1), so these features declare no input roles.

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from sabia.calendar import get_calendar
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, Evidence, Family, Horizon, Recurrence, Unit

_INT8: pl.DataType = pl.Int8()
_BOOL: pl.DataType = pl.Boolean()


def season_dow() -> BoundFeature:
    """Session weekday, Monday=0 .. Sunday=6, via the frame's calendar. FINITE, UNITLESS.

    Citation: French (1980), the weekend effect.
    """
    name = "season_dow"

    def build(s: BarSchema) -> pl.Expr:
        cal = get_calendar(s.calendar)
        return cal.session_weekday(pl.col(s.timestamp_col)).cast(_INT8).alias(name)

    return _calendar_feature(
        build,
        name=name,
        bands=(Horizon.SHORT,),
        output_dtype=_INT8,
        formula=Reference("French", 1980),
        params=FrozenParams(),
    )


def season_tom(*, k: int = 3) -> BoundFeature:
    """Turn-of-month flag: True within ``k`` sessions of a month boundary. FINITE, UNITLESS.

    Causal proxy for the turn-of-month effect: the first ``k`` and last ``k`` calendar days of the
    month (the last-session test uses ``days_in_month``, knowable at t, not a look-ahead). Citation:
    Ariel (1987).
    """
    name = naming("season_tom", k)

    def build(s: BarSchema) -> pl.Expr:
        cal = get_calendar(s.calendar)
        ts = pl.col(s.timestamp_col)
        day = cal.day_of_month(ts)
        value = (day <= k) | (day > cal.days_in_month(ts) - k)
        return value.cast(_BOOL).alias(name)

    return _calendar_feature(
        build,
        name=name,
        bands=(Horizon.SHORT, Horizon.MEDIUM),
        output_dtype=_BOOL,
        formula=Reference("Ariel", 1987),
        params=FrozenParams(k=k),
    )


def _calendar_feature(
    build: Callable[[BarSchema], pl.Expr],
    *,
    name: str,
    bands: tuple[Horizon, ...],
    output_dtype: pl.DataType,
    formula: Reference,
    params: FrozenParams,
) -> BoundFeature:
    return bind_feature(
        build,
        name=name,
        family=Family.SEASONALITY,
        native_band=bands,
        lookback=1,
        min_history=1,
        recurrence=Recurrence.FINITE,
        effective_warmup=1,
        cost_class=Cost.O1,
        input_roles=(),
        output_unit=Unit.UNITLESS,
        output_dtype=output_dtype,
        evidence=Evidence.ACADEMIC_SINGLE,
        citation=Citation(formula=formula),
        params=params,
    )


FEATURES: tuple[BoundFeature, ...] = (
    season_dow(),
    season_tom(k=3),
)


__all__ = [
    "FEATURES",
    "season_dow",
    "season_tom",
]
