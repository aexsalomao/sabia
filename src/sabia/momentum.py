# Momentum family: oscillators and rate-of-change measures of directional persistence. Close-based
# measures (RSI, ROC, momentum) use close@tr; range oscillators (Williams %R, stochastic, CCI) use
# split-only OHLC (FEATURES.md 2.2). RSI is RECURSIVE_DECAY (emit null until effective_warmup); the
# rest are FINITE. All strictly trailing and panel-safe via .over(symbol).

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from sabia._expr import emit_after, grouped
from sabia._math import safe_div, safe_log
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import (
    Cost,
    Evidence,
    Family,
    Horizon,
    Recurrence,
    Unit,
    ewm_effective_warmup,
)
from sabia.typing import (
    CLOSE_SPLIT,
    CLOSE_TR,
    HIGH_SPLIT,
    LOW_SPLIT,
    Adjustment,
    PriceRole,
)

_BANDS = (Horizon.SHORT, Horizon.MEDIUM)
_CCI_SCALE = 0.015  # Lambert's constant, so ~70-80% of CCI values fall in [-100, 100].


def rsi(*, period: int = 14, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Wilder's RSI, in [0, 100]. RECURSIVE_DECAY (Wilder RMA = EWM, alpha=1/period).

    A flat series (no gains, no losses) yields ``null``; pure gains saturate at 100, pure losses
    at 0 (FEATURES.md 4.5). Citation: Wilder (1978).
    """
    name = naming("rsi", period, role=close, default_adjustment=Adjustment.TR)
    warmup = ewm_effective_warmup(1 / period)
    alpha = 1 / period

    def build(s: BarSchema) -> pl.Expr:
        delta = pl.col(s.column(close)).diff()
        gain = delta.clip(lower_bound=0).ewm_mean(alpha=alpha, adjust=False, min_samples=period)
        loss = (-delta).clip(lower_bound=0).ewm_mean(alpha=alpha, adjust=False, min_samples=period)
        rs = gain / loss
        value = (
            pl.when((gain == 0) & (loss == 0))
            .then(None)
            .when(loss == 0)
            .then(pl.lit(100.0))
            .otherwise(100 - 100 / (1 + rs))
        )
        return emit_after(grouped(value, s.symbol_col), warmup, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MOMENTUM,
        native_band=_BANDS,
        lookback=period,
        min_history=warmup,
        recurrence=Recurrence.RECURSIVE_DECAY,
        effective_warmup=warmup,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=Unit.INDEX_0_100,
        output_range=(0.0, 100.0),
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Wilder", 1978)),
        params=FrozenParams(period=period),
    )


def roc(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Rate of change: ``close / close.shift(window) - 1``. FINITE, RATIO. Zero base yields null."""
    name = naming("roc", window, role=close, default_adjustment=Adjustment.TR)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        value = safe_div(c, c.shift(window)) - 1
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MOMENTUM,
        native_band=(Horizon.SHORT, Horizon.MEDIUM),
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=Unit.RATIO,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Pring", 2002)),
        params=FrozenParams(window=window),
    )


def mom(*, formation: int = 252, skip: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Time-series momentum: ``ln(close.shift(skip) / close.shift(formation))``. FINITE, LOG_RETURN.

    ``mom_252_21`` is the canonical 12-1 momentum: a 252-bar formation, the most recent 21 bars
    skipped to avoid short-term reversal. Citation: Jegadeesh & Titman (1993).
    """
    name = naming("mom", formation, skip, role=close, default_adjustment=Adjustment.TR)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        value = safe_log(safe_div(c.shift(skip), c.shift(formation)))
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MOMENTUM,
        native_band=(Horizon.LONG,),
        lookback=formation,
        min_history=formation + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=formation + 1,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=Unit.LOG_RETURN,
        evidence=Evidence.ACADEMIC_REPLICATED,
        citation=Citation(
            formula=Reference("Jegadeesh & Titman", 1993),
            empirical=(Reference("Asness, Moskowitz & Pedersen", 2013),),
        ),
        params=FrozenParams(formation=formation, skip=skip),
    )


def williams_r(
    *,
    window: int = 14,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Williams %R, in [-100, 0]. A flat range yields null. FINITE. Citation: Williams (1979)."""
    name = naming("williams_r", window)

    def build(s: BarSchema) -> pl.Expr:
        highest, lowest = _range_extremes(s, high, low, window)
        value = safe_div(highest - pl.col(s.column(close)), highest - lowest) * -100
        return grouped(value, s.symbol_col).alias(name)

    return _finite_osc(
        build, name, window, (high, low, close), Reference("Williams", 1979), (-100.0, 0.0)
    )


def stoch_k(
    *,
    window: int = 14,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Stochastic %K, in [0, 100]. A flat range yields null. FINITE. Citation: Lane (1984)."""
    name = naming("stoch_k", window)

    def build(s: BarSchema) -> pl.Expr:
        return grouped(_stoch_k_core(s, high, low, close, window), s.symbol_col).alias(name)

    return _finite_osc(
        build, name, window, (high, low, close), Reference("Lane", 1984), (0.0, 100.0)
    )


def stoch_d(
    *,
    window: int = 14,
    smooth: int = 3,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Stochastic %D: a ``smooth``-bar SMA of %K. FINITE. Citation: Lane (1984)."""
    name = naming("stoch_d", window, smooth)

    def build(s: BarSchema) -> pl.Expr:
        k = _stoch_k_core(s, high, low, close, window)
        value = k.rolling_mean(smooth, min_samples=smooth)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MOMENTUM,
        native_band=_BANDS,
        lookback=window,
        min_history=window + smooth - 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + smooth - 1,
        cost_class=Cost.LINEAR,
        input_roles=(high, low, close),
        output_unit=Unit.INDEX_0_100,
        output_range=(0.0, 100.0),
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Lane", 1984)),
        params=FrozenParams(window=window, smooth=smooth),
    )


def cci(
    *,
    window: int = 20,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Commodity Channel Index: typical price vs its rolling mean, scaled by mean deviation. FINITE.

    Uses the deviation-from-rolling-mean form of the mean absolute deviation (a pure-expression
    variant). A flat window (zero deviation) yields null. Citation: Lambert (1980).
    """
    name = naming("cci", window)
    # The mean-deviation is a rolling mean of (tp - sma_tp), and sma_tp is itself a window-bar
    # rolling mean -- so the first non-null CCI lands only after 2*window-1 bars, not window.
    min_history = 2 * window - 1

    def build(s: BarSchema) -> pl.Expr:
        tp = (pl.col(s.column(high)) + pl.col(s.column(low)) + pl.col(s.column(close))) / 3.0
        sma_tp = tp.rolling_mean(window, min_samples=window)
        mad = (tp - sma_tp).abs().rolling_mean(window, min_samples=window)
        value = safe_div(tp - sma_tp, _CCI_SCALE * mad)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MOMENTUM,
        native_band=_BANDS,
        lookback=window,
        min_history=min_history,
        recurrence=Recurrence.FINITE,
        effective_warmup=min_history,
        cost_class=Cost.LINEAR,
        input_roles=(high, low, close),
        output_unit=Unit.UNITLESS,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Lambert", 1980)),
        params=FrozenParams(window=window),
    )


def _range_extremes(
    s: BarSchema, high: PriceRole, low: PriceRole, window: int
) -> tuple[pl.Expr, pl.Expr]:
    return (
        pl.col(s.column(high)).rolling_max(window, min_samples=window),
        pl.col(s.column(low)).rolling_min(window, min_samples=window),
    )


def _stoch_k_core(
    s: BarSchema, high: PriceRole, low: PriceRole, close: PriceRole, window: int
) -> pl.Expr:
    highest, lowest = _range_extremes(s, high, low, window)
    return safe_div(pl.col(s.column(close)) - lowest, highest - lowest) * 100


def _finite_osc(
    build: Callable[[BarSchema], pl.Expr],
    name: str,
    window: int,
    roles: tuple[PriceRole, ...],
    formula: Reference,
    output_range: tuple[float, float] | None,
) -> BoundFeature:
    unit = Unit.INDEX_0_100 if output_range is not None else Unit.UNITLESS
    return bind_feature(
        build,
        name=name,
        family=Family.MOMENTUM,
        native_band=_BANDS,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=roles,
        output_unit=unit,
        output_range=output_range,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=formula),
        params=FrozenParams(window=window),
    )


FEATURES: tuple[BoundFeature, ...] = (
    rsi(period=14),
    roc(window=21),
    roc(window=10),
    mom(formation=252, skip=21),
    williams_r(window=14),
    stoch_k(window=14),
    stoch_d(window=14, smooth=3),
    cci(window=20),
)


__all__ = [
    "FEATURES",
    "cci",
    "mom",
    "roc",
    "rsi",
    "stoch_d",
    "stoch_k",
    "williams_r",
]
