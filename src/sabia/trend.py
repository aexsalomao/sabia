# Trend family: moving-average distances, 52-week-high distance, rolling price percentile, an OLS
# trend slope, MACD, and (beyond §12) raw moving averages + ADX. Close-based, close@tr. SMAs / OLS /
# percentile / 52w-high are FINITE; EMA-based (ema, ema_dist, macd) and ADX are RECURSIVE_DECAY and
# emit null until effective_warmup. All strictly trailing and panel-safe via .over(symbol).

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from sabia._expr import emit_after, grouped
from sabia._math import safe_div, safe_log
from sabia._validate_params import int_at_least, less_than
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
    PriceRole,
)

_MED = (Horizon.MEDIUM,)
_MED_LONG = (Horizon.MEDIUM, Horizon.LONG)


def sma_dist(*, window: int = 50, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Distance of close from its SMA: ``close / SMA(window) - 1``. FINITE, RATIO."""
    int_at_least("window", window, 2)
    name = naming("sma_dist", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        sma_ = c.rolling_mean(window, min_samples=window)
        return grouped(safe_div(c, sma_) - 1, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=Unit.RATIO,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Brock, Lakonishok & LeBaron", 1992)),
        params=FrozenParams(window=window),
    )


def ema_dist(*, span: int = 50, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Distance of close from its EMA: ``close / EMA(span) - 1``. RECURSIVE_DECAY, RATIO."""
    int_at_least("span", span, 2)  # EWM alpha = 2/(span+1) needs span >= 2 for alpha < 1
    name = naming("ema_dist", span)
    warmup = ewm_effective_warmup(2 / (span + 1))

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        ema_ = c.ewm_mean(span=span, adjust=False, min_samples=span)
        return emit_after(grouped(safe_div(c, ema_) - 1, s.symbol_col), warmup, s.symbol_col).alias(
            name
        )

    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED,
        lookback=span,
        min_history=warmup,
        recurrence=Recurrence.RECURSIVE_DECAY,
        effective_warmup=warmup,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=Unit.RATIO,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Brock, Lakonishok & LeBaron", 1992)),
        params=FrozenParams(span=span),
    )


def dist_52w_high(*, window: int = 252, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Distance from the trailing high: ``close / max(close, window) - 1``, in [-1, 0]. FINITE.

    The 52-week-high anomaly (George & Hwang 2004). Citation: George & Hwang (2004).
    """
    int_at_least("window", window, 2)
    name = naming("dist_52w_high", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        peak = c.rolling_max(window, min_samples=window)
        return grouped(safe_div(c, peak) - 1, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED_LONG,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=Unit.RATIO,
        output_range=(-1.0, 0.0),
        evidence=Evidence.ACADEMIC_REPLICATED,
        citation=Citation(formula=Reference("George & Hwang", 2004)),
        params=FrozenParams(window=window),
    )


def price_pctile(*, window: int = 252, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Percentile of current close within the trailing window, in [0, 1]. FINITE, RANK_0_1.

    Uses a rolling kernel (the fraction of the window <= current close) -- a HEAVY escape hatch
    (FEATURES.md 10), carried by the eager-vs-lazy and benchmark gates.
    """
    int_at_least("window", window, 2)
    name = naming("price_pctile", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        pctile = c.rolling_map(
            lambda w: (w <= w.last()).mean(), window_size=window, min_samples=window
        )
        return grouped(pctile, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED_LONG,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.HEAVY,
        input_roles=(close,),
        output_unit=Unit.RANK_0_1,
        output_range=(0.0, 1.0),
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=Reference("Campbell, Lo & MacKinlay", 1997)),
        params=FrozenParams(window=window),
    )


def ols_slope(*, window: int = 63, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """OLS slope of ``ln(close) ~ t`` over the trailing window (per-bar log drift). FINITE.

    Computed in closed form from rolling sums (the window's x-values are a fixed ramp 0..w-1, so the
    denominator is constant) -- exact and vectorized, no per-row Python.
    """
    int_at_least("window", window, 2)  # denom = window*(window^2 - 1)/12 is 0 at window=1
    name = naming("ols_slope", window)
    denom = window * (window * window - 1) / 12.0
    x_mean = (window - 1) / 2.0

    def build(s: BarSchema) -> pl.Expr:
        y = safe_log(pl.col(s.column(close)))
        g = pl.int_range(pl.len()).cast(pl.Float64)
        s_gy = (g * y).rolling_sum(window, min_samples=window)
        s_y = y.rolling_sum(window, min_samples=window)
        # Local position k = g - (g_current - w + 1); centered numerator = Σ k·y - x_mean·Σ y.
        numerator = s_gy - (g - window + 1) * s_y - x_mean * s_y
        return grouped(numerator / denom, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=Unit.LOG_RETURN,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=Reference("Campbell, Lo & MacKinlay", 1997)),
        params=FrozenParams(window=window),
    )


def _check_macd_params(fast: int, slow: int, signal: int) -> None:
    # Each leg is an EWM span (alpha = 2/(n+1) needs n >= 2), and the line is EMA(fast) - EMA(slow),
    # so a meaningful MACD needs fast < slow.
    int_at_least("fast", fast, 2)
    int_at_least("slow", slow, 2)
    int_at_least("signal", signal, 2)
    less_than("fast", fast, "slow", slow)


def _macd_raw(s: BarSchema, close: PriceRole, fast: int, slow: int) -> pl.Expr:
    c = pl.col(s.column(close))
    ema_fast = c.ewm_mean(span=fast, adjust=False, min_samples=fast)
    ema_slow = c.ewm_mean(span=slow, adjust=False, min_samples=slow)
    return ema_fast - ema_slow


def _macd_warmup(slow: int, signal: int) -> tuple[int, int]:
    line = ewm_effective_warmup(2 / (slow + 1))
    sig = line + ewm_effective_warmup(2 / (signal + 1))
    return line, sig


def macd(
    *, fast: int = 12, slow: int = 26, signal: int = 9, close: PriceRole = CLOSE_TR
) -> BoundFeature:
    """MACD line: ``EMA(fast) - EMA(slow)`` of close. RECURSIVE_DECAY, LOG_RETURN. Appel (1979)."""
    _check_macd_params(fast, slow, signal)
    name = naming("macd", fast, slow, signal)
    warmup, _ = _macd_warmup(slow, signal)

    def build(s: BarSchema) -> pl.Expr:
        raw = grouped(_macd_raw(s, close, fast, slow), s.symbol_col)
        return emit_after(raw, warmup, s.symbol_col).alias(name)

    params = FrozenParams(fast=fast, slow=slow, signal=signal)
    return _macd_feature(build, name, warmup, close, params)


def macd_signal(
    *, fast: int = 12, slow: int = 26, signal: int = 9, close: PriceRole = CLOSE_TR
) -> BoundFeature:
    """MACD signal: ``EMA(signal)`` of the MACD line. RECURSIVE_DECAY, LOG_RETURN."""
    _check_macd_params(fast, slow, signal)
    name = naming("macd", fast, slow, signal, suffix="signal")
    _, warmup = _macd_warmup(slow, signal)

    def build(s: BarSchema) -> pl.Expr:
        raw = _macd_raw(s, close, fast, slow)
        sig = raw.ewm_mean(span=signal, adjust=False, min_samples=signal)
        return emit_after(grouped(sig, s.symbol_col), warmup, s.symbol_col).alias(name)

    params = FrozenParams(fast=fast, slow=slow, signal=signal)
    return _macd_feature(build, name, warmup, close, params)


def macd_hist(
    *, fast: int = 12, slow: int = 26, signal: int = 9, close: PriceRole = CLOSE_TR
) -> BoundFeature:
    """MACD histogram: ``MACD line - signal``. RECURSIVE_DECAY, LOG_RETURN."""
    _check_macd_params(fast, slow, signal)
    name = naming("macd", fast, slow, signal, suffix="hist")
    _, warmup = _macd_warmup(slow, signal)

    def build(s: BarSchema) -> pl.Expr:
        raw = _macd_raw(s, close, fast, slow)
        sig = raw.ewm_mean(span=signal, adjust=False, min_samples=signal)
        return emit_after(grouped(raw - sig, s.symbol_col), warmup, s.symbol_col).alias(name)

    params = FrozenParams(fast=fast, slow=slow, signal=signal)
    return _macd_feature(build, name, warmup, close, params)


def _macd_feature(
    build: Callable[[BarSchema], pl.Expr],
    name: str,
    warmup: int,
    close: PriceRole,
    params: FrozenParams,
) -> BoundFeature:
    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED,
        lookback=None,
        min_history=warmup,
        recurrence=Recurrence.RECURSIVE_DECAY,
        effective_warmup=warmup,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=Unit.LOG_RETURN,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Appel", 1979)),
        params=params,
    )


# --- extras beyond §12 (no §12 equivalent): raw moving averages and ADX --------------------------


def sma(*, window: int = 50, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Simple moving average level of close. FINITE, PRICE_UNITS. (Extra, beyond §12.)"""
    int_at_least("window", window, 2)
    name = naming("sma", window)

    def build(s: BarSchema) -> pl.Expr:
        value = pl.col(s.column(close)).rolling_mean(window, min_samples=window)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED_LONG,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=Unit.PRICE_UNITS,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Brock, Lakonishok & LeBaron", 1992)),
        params=FrozenParams(window=window),
    )


def ema(*, span: int = 12, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Exponential moving average level of close. RECURSIVE_DECAY. (Extra, beyond §12.)"""
    int_at_least("span", span, 2)
    name = naming("ema", span)
    warmup = ewm_effective_warmup(2 / (span + 1))

    def build(s: BarSchema) -> pl.Expr:
        value = pl.col(s.column(close)).ewm_mean(span=span, adjust=False, min_samples=span)
        return emit_after(grouped(value, s.symbol_col), warmup, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED,
        lookback=span,
        min_history=warmup,
        recurrence=Recurrence.RECURSIVE_DECAY,
        effective_warmup=warmup,
        cost_class=Cost.O1,
        input_roles=(close,),
        output_unit=Unit.PRICE_UNITS,
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Appel", 1979)),
        params=FrozenParams(span=span),
    )


def adx(
    *,
    window: int = 14,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
) -> BoundFeature:
    """Average Directional Index, in [0, 100] (Wilder 1978). RECURSIVE_DECAY. (Extra.)"""
    int_at_least("window", window, 2)  # RMA alpha = 1/window needs window >= 2 for alpha < 1
    name = naming("adx", window)
    warmup = 2 * ewm_effective_warmup(1 / window) + 2 * window

    def build(s: BarSchema) -> pl.Expr:
        value = _adx_core(s, high, low, close, window)
        return emit_after(grouped(value, s.symbol_col), warmup, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.TREND,
        native_band=_MED_LONG,
        lookback=window,
        min_history=warmup,
        recurrence=Recurrence.RECURSIVE_DECAY,
        effective_warmup=warmup,
        cost_class=Cost.O1,
        input_roles=(high, low, close),
        output_unit=Unit.INDEX_0_100,
        output_range=(0.0, 100.0),
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Wilder", 1978)),
        params=FrozenParams(window=window),
    )


def _rma(expr: pl.Expr, window: int) -> pl.Expr:
    return expr.ewm_mean(alpha=1 / window, adjust=False, min_samples=window)


def _adx_core(
    s: BarSchema, high: PriceRole, low: PriceRole, close: PriceRole, window: int
) -> pl.Expr:
    h, low_, c = pl.col(s.column(high)), pl.col(s.column(low)), pl.col(s.column(close))
    up_move = h - h.shift(1)
    down_move = low_.shift(1) - low_
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
    prev_close = c.shift(1)
    true_range = pl.max_horizontal(h - low_, (h - prev_close).abs(), (low_ - prev_close).abs())
    atr_ = _rma(true_range, window)
    plus_di = 100 * safe_div(_rma(plus_dm, window), atr_)
    minus_di = 100 * safe_div(_rma(minus_dm, window), atr_)
    dx = 100 * safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return _rma(dx, window)


FEATURES: tuple[BoundFeature, ...] = (
    sma_dist(window=50),
    ema_dist(span=50),
    dist_52w_high(window=252),
    price_pctile(window=252),
    ols_slope(window=63),
    macd(fast=12, slow=26, signal=9),
    macd_signal(fast=12, slow=26, signal=9),
    macd_hist(fast=12, slow=26, signal=9),
    sma(window=50),
    sma(window=200),
    ema(span=12),
    ema(span=26),
    adx(window=14),
)


__all__ = [
    "FEATURES",
    "adx",
    "dist_52w_high",
    "ema",
    "ema_dist",
    "macd",
    "macd_hist",
    "macd_signal",
    "ols_slope",
    "price_pctile",
    "sma",
    "sma_dist",
]
