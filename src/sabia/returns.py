# Returns family: log and simple returns over closed bars, plus overnight / intraday decompositions
# and a trailing drawdown. Close-based returns use close@tr (total return), so close-to-close is the
# dividend+split-adjusted return (FEATURES.md 2.2). All strictly trailing and panel-safe.

from __future__ import annotations

import polars as pl

from sabia._expr import grouped
from sabia._math import safe_div, safe_log
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, Evidence, Family, Horizon, Recurrence, Unit
from sabia.typing import CLOSE_TR, OPEN_TR, Adjustment, PriceRole

_BANDS = (Horizon.SHORT, Horizon.MEDIUM)
_FORMULA = Reference("Campbell, Lo & MacKinlay", 1997, "The Econometrics of Financial Markets")


def ret_log(*, period: int = 1, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Log return over ``period`` bars: ``ln(close / close.shift(period))``. FINITE, LOG_RETURN.

    A non-positive ratio (split artifact / bad data) yields ``null`` (FEATURES.md 4.5).
    """
    name = naming("ret_log", period, role=close, default_adjustment=Adjustment.TR)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        value = safe_log(safe_div(c, c.shift(period)))
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.RETURNS,
        native_band=_BANDS,
        lookback=period,
        min_history=period + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=period + 1,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=Unit.LOG_RETURN,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=_FORMULA),
        params=FrozenParams(period=period),
    )


def ret_simple(*, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Simple one-bar return: ``close / close.shift(1) - 1``. FINITE, RATIO. (Extra, beyond §12.)"""
    name = naming("ret_simple", role=close, default_adjustment=Adjustment.TR)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        value = safe_div(c, c.shift(1)) - 1
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.RETURNS,
        native_band=(Horizon.SHORT,),
        lookback=1,
        min_history=2,
        recurrence=Recurrence.FINITE,
        effective_warmup=2,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=Unit.RATIO,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=_FORMULA),
        params=FrozenParams(),
    )


def ret_overnight(*, open_: PriceRole = OPEN_TR, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Overnight return: ``ln(open / close.shift(1))``. FINITE, LOG_RETURN."""
    name = "ret_overnight"

    def build(s: BarSchema) -> pl.Expr:
        o = pl.col(s.column(open_))
        c_prev = pl.col(s.column(close)).shift(1)
        value = safe_log(safe_div(o, c_prev))
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.RETURNS,
        native_band=(Horizon.SHORT,),
        lookback=1,
        min_history=2,
        recurrence=Recurrence.FINITE,
        effective_warmup=2,
        cost_class=Cost.O1,
        input_roles=(open_, close),
        output_unit=Unit.LOG_RETURN,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=_FORMULA),
        params=FrozenParams(),
    )


def ret_intraday(*, open_: PriceRole = OPEN_TR, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Intraday (open-to-close) return: ``ln(close / open)``. FINITE, LOG_RETURN."""
    name = "ret_intraday"

    def build(s: BarSchema) -> pl.Expr:
        value = safe_log(safe_div(pl.col(s.column(close)), pl.col(s.column(open_))))
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.RETURNS,
        native_band=(Horizon.SHORT,),
        lookback=1,
        min_history=1,
        recurrence=Recurrence.FINITE,
        effective_warmup=1,
        cost_class=Cost.O1,
        input_roles=(open_, close),
        output_unit=Unit.LOG_RETURN,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=_FORMULA),
        params=FrozenParams(),
    )


def drawdown(*, window: int = 252, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Trailing drawdown: ``close / max(close, window) - 1``, in [-1, 0]. FINITE, RATIO."""
    name = naming("drawdown", window, role=close, default_adjustment=Adjustment.TR)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        peak = c.rolling_max(window, min_samples=window)
        value = safe_div(c, peak) - 1
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.RETURNS,
        native_band=(Horizon.LONG,),
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=Unit.RATIO,
        output_range=(-1.0, 0.0),
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=_FORMULA),
        params=FrozenParams(window=window),
    )


FEATURES: tuple[BoundFeature, ...] = (
    ret_log(period=1),
    ret_log(period=5),
    ret_log(period=21),
    ret_log(period=252),
    ret_simple(),
    ret_overnight(),
    ret_intraday(),
    drawdown(window=252),
)


__all__ = [
    "FEATURES",
    "drawdown",
    "ret_intraday",
    "ret_log",
    "ret_overnight",
    "ret_simple",
]
