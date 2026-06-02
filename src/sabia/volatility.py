# Volatility family: per-bar volatility / range estimators. Close-to-close, RiskMetrics EWMA, and
# downside semivariance use close@tr; the OHLC range estimators (Parkinson, Garman-Klass,
# Rogers-Satchell, Yang-Zhang) and ATR use split-only OHLC -- dividend adjustment distorts ranges,
# so range vol on it measures adjustment artifacts (FEATURES.md 2.2). EWMA and ATR are
# RECURSIVE_DECAY (emit null until effective_warmup); the rest are FINITE. Per-bar, never inf/NaN.

from __future__ import annotations

from collections.abc import Callable
from math import log

import polars as pl

from sabia._expr import emit_after, grouped
from sabia._math import log_return, safe_log, safe_sqrt
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
    OPEN_SPLIT,
    PriceRole,
)

_LN2 = log(2.0)
_BANDS = (Horizon.SHORT, Horizon.MEDIUM)
_STD_PER_BAR = Unit.RETURN_STD_PER_BAR


def vol_cc(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Close-to-close volatility: rolling std of one-bar log returns. FINITE, per-bar."""
    name = naming("vol_cc", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        r = log_return(c, c.shift(1))
        return grouped(r.rolling_std(window, min_samples=window), s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLATILITY,
        native_band=_BANDS,
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=_STD_PER_BAR,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=Reference("Campbell, Lo & MacKinlay", 1997)),
        params=FrozenParams(window=window),
    )


def vol_ewma(*, lam: float = 0.94, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """RiskMetrics EWMA volatility: ``sqrt(EWMA(r^2, alpha=1-lam))``. RECURSIVE_DECAY, per-bar.

    Named by the lambda's percentage (``vol_ewma_94`` for lambda=0.94). RiskMetrics (1996).
    """
    name = naming("vol_ewma", round(lam * 100))
    alpha = 1.0 - lam
    warmup = ewm_effective_warmup(alpha)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        r = log_return(c, c.shift(1))
        var = (r**2).ewm_mean(alpha=alpha, adjust=False, min_samples=2)
        return emit_after(grouped(safe_sqrt(var), s.symbol_col), warmup, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLATILITY,
        native_band=_BANDS,
        lookback=None,
        min_history=warmup,
        recurrence=Recurrence.RECURSIVE_DECAY,
        effective_warmup=warmup,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=_STD_PER_BAR,
        evidence=Evidence.ACADEMIC_SINGLE,
        citation=Citation(formula=Reference("J.P. Morgan / Reuters RiskMetrics", 1996)),
        params=FrozenParams(lam=lam),
    )


def semivar_down(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Realized downside semivariance: ``sqrt(mean(min(r, 0)^2))`` over the window. FINITE."""
    name = naming("semivar_down", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        r = log_return(c, c.shift(1))
        # clip (not min_horizontal) so a null leading return stays null, never imputed to 0.
        downside = r.clip(upper_bound=0.0) ** 2
        value = safe_sqrt(downside.rolling_mean(window, min_samples=window))
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLATILITY,
        native_band=_BANDS,
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=_STD_PER_BAR,
        evidence=Evidence.ACADEMIC_SINGLE,
        citation=Citation(formula=Reference("Markowitz", 1959)),
        params=FrozenParams(window=window),
    )


def vol_parkinson(
    *, window: int = 21, high: PriceRole = HIGH_SPLIT, low: PriceRole = LOW_SPLIT
) -> BoundFeature:
    """Parkinson (1980) high-low range volatility. FINITE, per-bar."""
    name = naming("vol_parkinson", window)

    def build(s: BarSchema) -> pl.Expr:
        term = safe_log(pl.col(s.column(high)) / pl.col(s.column(low))) ** 2
        variance = term.rolling_mean(window, min_samples=window) / (4.0 * _LN2)
        return grouped(safe_sqrt(variance), s.symbol_col).alias(name)

    return _finite_range(build, name, window, (high, low), Reference("Parkinson", 1980))


def vol_gk(
    *,
    window: int = 21,
    open_: PriceRole = OPEN_SPLIT,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Garman-Klass (1980) OHLC volatility. FINITE, per-bar."""
    name = naming("vol_gk", window)

    def build(s: BarSchema) -> pl.Expr:
        hl = safe_log(pl.col(s.column(high)) / pl.col(s.column(low)))
        co = safe_log(pl.col(s.column(close)) / pl.col(s.column(open_)))
        term = 0.5 * hl**2 - (2.0 * _LN2 - 1.0) * co**2
        variance = term.rolling_mean(window, min_samples=window)
        return grouped(safe_sqrt(variance), s.symbol_col).alias(name)

    return _finite_range(
        build, name, window, (open_, high, low, close), Reference("Garman & Klass", 1980)
    )


def vol_rs(
    *,
    window: int = 21,
    open_: PriceRole = OPEN_SPLIT,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Rogers-Satchell (1991) drift-independent OHLC volatility. FINITE, per-bar."""
    name = naming("vol_rs", window)

    def build(s: BarSchema) -> pl.Expr:
        variance = _rs_term(s, open_, high, low, close).rolling_mean(window, min_samples=window)
        return grouped(safe_sqrt(variance), s.symbol_col).alias(name)

    return _finite_range(
        build, name, window, (open_, high, low, close), Reference("Rogers & Satchell", 1991)
    )


def vol_yz(
    *,
    window: int = 21,
    open_: PriceRole = OPEN_SPLIT,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Yang-Zhang (2000) volatility: overnight + open-close + Rogers-Satchell. FINITE, per-bar."""
    name = naming("vol_yz", window)

    def build(s: BarSchema) -> pl.Expr:
        o, c = pl.col(s.column(open_)), pl.col(s.column(close))
        overnight = safe_log(o / c.shift(1))
        open_close = safe_log(c / o)
        sigma_o2 = overnight.rolling_var(window, min_samples=window)
        sigma_c2 = open_close.rolling_var(window, min_samples=window)
        sigma_rs2 = _rs_term(s, open_, high, low, close).rolling_mean(window, min_samples=window)
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        variance = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
        return grouped(safe_sqrt(variance), s.symbol_col).alias(name)

    return _finite_range(
        build,
        name,
        window,
        (open_, high, low, close),
        Reference("Yang & Zhang", 2000),
        extra_history=1,
    )


def atr(
    *,
    window: int = 14,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Average True Range (Wilder 1978), RMA-smoothed. RECURSIVE_DECAY, price units."""
    name = naming("atr", window)
    warmup = ewm_effective_warmup(1 / window)

    def build(s: BarSchema) -> pl.Expr:
        h, low_, c = pl.col(s.column(high)), pl.col(s.column(low)), pl.col(s.column(close))
        prev_close = c.shift(1)
        true_range = pl.max_horizontal(h - low_, (h - prev_close).abs(), (low_ - prev_close).abs())
        value = true_range.ewm_mean(alpha=1 / window, adjust=False, min_samples=window)
        return emit_after(grouped(value, s.symbol_col), warmup, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLATILITY,
        native_band=_BANDS,
        lookback=window,
        min_history=warmup,
        recurrence=Recurrence.RECURSIVE_DECAY,
        effective_warmup=warmup,
        cost_class=Cost.O1,
        input_roles=(high, low, close),
        output_unit=Unit.PRICE_UNITS,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Wilder", 1978)),
        params=FrozenParams(window=window),
    )


def _rs_term(
    s: BarSchema, open_: PriceRole, high: PriceRole, low: PriceRole, close: PriceRole
) -> pl.Expr:
    h, low_, c, o = (
        pl.col(s.column(high)),
        pl.col(s.column(low)),
        pl.col(s.column(close)),
        pl.col(s.column(open_)),
    )
    return safe_log(h / c) * safe_log(h / o) + safe_log(low_ / c) * safe_log(low_ / o)


def _finite_range(
    build: Callable[[BarSchema], pl.Expr],
    name: str,
    window: int,
    roles: tuple[PriceRole, ...],
    formula: Reference,
    *,
    extra_history: int = 0,
) -> BoundFeature:
    return bind_feature(
        build,
        name=name,
        family=Family.VOLATILITY,
        native_band=_BANDS,
        lookback=window,
        min_history=window + extra_history,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + extra_history,
        cost_class=Cost.LINEAR,
        input_roles=roles,
        output_unit=_STD_PER_BAR,
        evidence=Evidence.ACADEMIC_SINGLE,
        citation=Citation(formula=formula),
        params=FrozenParams(window=window),
    )


FEATURES: tuple[BoundFeature, ...] = (
    vol_cc(window=21),
    vol_cc(window=63),
    vol_ewma(lam=0.94),
    semivar_down(window=21),
    vol_parkinson(window=21),
    vol_gk(window=21),
    vol_rs(window=21),
    vol_yz(window=21),
    atr(window=14),
)


__all__ = [
    "FEATURES",
    "atr",
    "semivar_down",
    "vol_cc",
    "vol_ewma",
    "vol_gk",
    "vol_parkinson",
    "vol_rs",
    "vol_yz",
]
