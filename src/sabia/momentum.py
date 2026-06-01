# Momentum family: oscillators and rate-of-change measures of directional persistence.
# All features are strictly trailing and panel-safe -- the whole expression is evaluated within
# each symbol via .over(symbol), so windows never bleed across symbols.

from __future__ import annotations

from functools import partial

import polars as pl

from sabia._expr import grouped
from sabia._math import safe_div
from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence, ewm_effective_warmup

_BANDS = (Horizon.SHORT, Horizon.MEDIUM)


def rsi(
    close: str = Column.CLOSE, *, period: int = 14, symbol: str | None = Column.SYMBOL
) -> pl.Expr:
    """Wilder's Relative Strength Index, in [0, 100]. Strictly trailing, RECURSIVE.

    A flat series (no gains and no losses) yields ``null`` -- there is no information to oscillate
    on (FEATURES.md 3.5); a series of pure gains saturates at 100, pure losses at 0. Wilder's RMA
    smoothing is an EWM with ``alpha = 1 / period`` and ``adjust=False``. Citation: Wilder (1978).
    """
    delta = pl.col(close).diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    avg_gain = gain.ewm_mean(alpha=1 / period, adjust=False, min_samples=period)
    avg_loss = loss.ewm_mean(alpha=1 / period, adjust=False, min_samples=period)
    rs = avg_gain / avg_loss
    value = (
        pl.when((avg_gain == 0) & (avg_loss == 0))
        .then(None)
        .when(avg_loss == 0)
        .then(pl.lit(100.0))
        .otherwise(100 - 100 / (1 + rs))
    )
    return grouped(value, symbol).alias(f"rsi_{period}")


def roc(
    close: str = Column.CLOSE, *, window: int = 10, symbol: str | None = Column.SYMBOL
) -> pl.Expr:
    """Rate of change: percent return over ``window`` bars. A zero base yields null. FINITE."""
    base = pl.col(close).shift(window)
    value = safe_div(pl.col(close) - base, base) * 100
    return grouped(value, symbol).alias(f"roc_{window}")


def williams_r(
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    *,
    window: int = 14,
    symbol: str | None = Column.SYMBOL,
) -> pl.Expr:
    """Williams %R, in [-100, 0]. A flat range yields null. FINITE. Citation: Williams (1979)."""
    highest, lowest = _range_extremes(high, low, window)
    value = safe_div(highest - pl.col(close), highest - lowest) * -100
    return grouped(value, symbol).alias(f"williams_r_{window}")


def stoch_k(
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    *,
    window: int = 14,
    symbol: str | None = Column.SYMBOL,
) -> pl.Expr:
    """Stochastic %K, in [0, 100]. A flat range yields null. FINITE. Citation: Lane (1984)."""
    return grouped(_stoch_k_core(high, low, close, window), symbol).alias(f"stoch_k_{window}")


def stoch_d(
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    *,
    window: int = 14,
    smooth: int = 3,
    symbol: str | None = Column.SYMBOL,
) -> pl.Expr:
    """Stochastic %D: an ``smooth``-bar SMA of %K. FINITE. Citation: Lane (1984)."""
    k = _stoch_k_core(high, low, close, window)
    value = k.rolling_mean(smooth, min_samples=smooth)
    return grouped(value, symbol).alias(f"stoch_d_{window}_{smooth}")


def macd(
    close: str = Column.CLOSE,
    *,
    fast: int = 12,
    slow: int = 26,
    symbol: str | None = Column.SYMBOL,
) -> pl.Expr:
    """MACD line: fast EMA minus slow EMA of close. RECURSIVE. Citation: Appel (1979)."""
    ema_fast = pl.col(close).ewm_mean(span=fast, adjust=False, min_samples=fast)
    ema_slow = pl.col(close).ewm_mean(span=slow, adjust=False, min_samples=slow)
    return grouped((ema_fast - ema_slow), symbol).alias(f"macd_{fast}_{slow}")


def _range_extremes(high: str, low: str, window: int) -> tuple[pl.Expr, pl.Expr]:
    return (
        pl.col(high).rolling_max(window, min_samples=window),
        pl.col(low).rolling_min(window, min_samples=window),
    )


def _stoch_k_core(high: str, low: str, close: str, window: int) -> pl.Expr:
    highest, lowest = _range_extremes(high, low, window)
    return safe_div(pl.col(close) - lowest, highest - lowest) * 100


def _rsi_warmup(period: int) -> int:
    # RSI is an EWM applied to a diff(), so the analytic EWM warmup is one decay-bar short in
    # practice; +period absorbs the diff offset and recursive accumulation (validated by parity).
    return ewm_effective_warmup(1 / period) + period


def _ewm_warmup(span: int) -> int:
    return ewm_effective_warmup(2 / (span + 1)) + span


FEATURES: tuple[RegisteredFeature, ...] = (
    make_feature(
        rsi,
        build=partial(rsi, period=14),
        name="rsi_14",
        family=Family.MOMENTUM,
        native_band=_BANDS,
        lookback=14,
        min_history=15,
        recurrence=Recurrence.RECURSIVE,
        effective_warmup=_rsi_warmup(14),
        cost_class=Cost.O1,
        inputs=(Column.CLOSE,),
        citation="Wilder (1978)",
        params={"period": 14},
    ),
    make_feature(
        roc,
        build=partial(roc, window=10),
        name="roc_10",
        family=Family.MOMENTUM,
        native_band=(Horizon.SHORT,),
        lookback=10,
        min_history=11,
        recurrence=Recurrence.FINITE,
        effective_warmup=11,
        cost_class=Cost.O1,
        inputs=(Column.CLOSE,),
        citation="rate of change",
        params={"window": 10},
    ),
    make_feature(
        williams_r,
        build=partial(williams_r, window=14),
        name="williams_r_14",
        family=Family.MOMENTUM,
        native_band=_BANDS,
        lookback=14,
        min_history=14,
        recurrence=Recurrence.FINITE,
        effective_warmup=14,
        cost_class=Cost.LINEAR,
        inputs=(Column.HIGH, Column.LOW, Column.CLOSE),
        citation="Williams (1979)",
        params={"window": 14},
    ),
    make_feature(
        stoch_k,
        build=partial(stoch_k, window=14),
        name="stoch_k_14",
        family=Family.MOMENTUM,
        native_band=_BANDS,
        lookback=14,
        min_history=14,
        recurrence=Recurrence.FINITE,
        effective_warmup=14,
        cost_class=Cost.LINEAR,
        inputs=(Column.HIGH, Column.LOW, Column.CLOSE),
        citation="Lane (1984)",
        params={"window": 14},
    ),
    make_feature(
        stoch_d,
        build=partial(stoch_d, window=14, smooth=3),
        name="stoch_d_14_3",
        family=Family.MOMENTUM,
        native_band=_BANDS,
        lookback=14,
        min_history=16,
        recurrence=Recurrence.FINITE,
        effective_warmup=16,
        cost_class=Cost.LINEAR,
        inputs=(Column.HIGH, Column.LOW, Column.CLOSE),
        citation="Lane (1984)",
        params={"window": 14, "smooth": 3},
    ),
    make_feature(
        macd,
        build=partial(macd, fast=12, slow=26),
        name="macd_12_26",
        family=Family.MOMENTUM,
        native_band=(Horizon.MEDIUM,),
        lookback=26,
        min_history=26,
        recurrence=Recurrence.RECURSIVE,
        effective_warmup=_ewm_warmup(26),
        cost_class=Cost.O1,
        inputs=(Column.CLOSE,),
        citation="Appel (1979)",
        params={"fast": 12, "slow": 26},
    ),
)


__all__ = [
    "FEATURES",
    "macd",
    "roc",
    "rsi",
    "stoch_d",
    "stoch_k",
    "williams_r",
]
