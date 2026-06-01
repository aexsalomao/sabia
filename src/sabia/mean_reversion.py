# Mean-reversion family: how far price sits from its recent center, and how fast it pulls back.
# All FINITE, strictly trailing, panel-safe via .over(symbol). The OU half-life uses a rolling OLS
# slope expressed entirely through rolling moments (no per-row Python). FEATURES.md 3.5 / 9.

from __future__ import annotations

from functools import partial
from math import log

import polars as pl

from sabia._math import safe_div
from sabia.normalize import zscore
from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence

_LN2 = log(2.0)
_BOLLINGER_K = 2.0


def zdist(close: str = Column.CLOSE, *, window: int = 20, symbol: str = Column.SYMBOL) -> pl.Expr:
    """Rolling z-distance of close from its own mean: the mean-reversion signal. FINITE."""
    return zscore(pl.col(close), window, over=symbol).alias(f"zdist_{window}")


def bollinger_pctb(
    close: str = Column.CLOSE, *, window: int = 20, symbol: str = Column.SYMBOL
) -> pl.Expr:
    """Bollinger %b: position within the +/-2 sigma bands, 0 at the lower, 1 at the upper. FINITE.

    Equivalent to ``0.5 + zscore / (2k)``; a flat window (zero std) -> null. Citation: Bollinger.
    """
    standardized = zscore(pl.col(close), window, over=symbol)
    return (0.5 + standardized / (2 * _BOLLINGER_K)).alias(f"bollinger_pctb_{window}")


def dist_ma(close: str = Column.CLOSE, *, window: int = 50, symbol: str = Column.SYMBOL) -> pl.Expr:
    """Fractional distance of close from its ``window``-bar SMA (close / SMA - 1). FINITE."""
    moving_average = pl.col(close).rolling_mean(window, min_samples=window)
    value = safe_div(pl.col(close), moving_average) - 1
    return value.over(symbol).alias(f"dist_ma_{window}")


def half_life(
    close: str = Column.CLOSE, *, window: int = 60, symbol: str = Column.SYMBOL
) -> pl.Expr:
    """Ornstein-Uhlenbeck mean-reversion half-life (in bars), from a rolling OLS slope. FINITE.

    Regresses the one-bar change on the prior level over ``window`` bars; the slope ``beta`` gives a
    half-life ``-ln(2) / ln(1 + beta)`` only when ``-1 < beta < 0`` (genuinely mean-reverting),
    otherwise null. The slope is built from rolling moments, so it stays a vectorized expression.
    """
    level = pl.col(close).shift(1)
    change = pl.col(close).diff()
    beta = _rolling_slope(level, change, window)
    reverting = (beta > -1) & (beta < 0)
    value = pl.when(reverting).then(-_LN2 / (1 + beta).log()).otherwise(None)
    return value.over(symbol).alias(f"half_life_{window}")


def _rolling_slope(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    # OLS slope of y on x over the window, from population moments: cov(x, y) / var(x).
    mean_x = x.rolling_mean(window, min_samples=window)
    mean_y = y.rolling_mean(window, min_samples=window)
    mean_xy = (x * y).rolling_mean(window, min_samples=window)
    mean_xx = (x * x).rolling_mean(window, min_samples=window)
    return safe_div(mean_xy - mean_x * mean_y, mean_xx - mean_x * mean_x)


FEATURES: tuple[RegisteredFeature, ...] = (
    make_feature(
        zdist,
        build=partial(zdist, window=20),
        name="zdist_20",
        family=Family.MEAN_REVERSION,
        native_band=(Horizon.SHORT, Horizon.MEDIUM),
        lookback=20,
        min_history=20,
        recurrence=Recurrence.FINITE,
        effective_warmup=20,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="rolling z-score",
        params={"window": 20},
    ),
    make_feature(
        bollinger_pctb,
        build=partial(bollinger_pctb, window=20),
        name="bollinger_pctb_20",
        family=Family.MEAN_REVERSION,
        native_band=(Horizon.SHORT, Horizon.MEDIUM),
        lookback=20,
        min_history=20,
        recurrence=Recurrence.FINITE,
        effective_warmup=20,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="Bollinger",
        params={"window": 20},
    ),
    make_feature(
        dist_ma,
        build=partial(dist_ma, window=50),
        name="dist_ma_50",
        family=Family.MEAN_REVERSION,
        native_band=(Horizon.MEDIUM,),
        lookback=50,
        min_history=50,
        recurrence=Recurrence.FINITE,
        effective_warmup=50,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="distance from moving average",
        params={"window": 50},
    ),
    make_feature(
        half_life,
        build=partial(half_life, window=60),
        name="half_life_60",
        family=Family.MEAN_REVERSION,
        native_band=(Horizon.MEDIUM,),
        lookback=60,
        min_history=61,
        recurrence=Recurrence.FINITE,
        effective_warmup=61,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="Ornstein-Uhlenbeck half-life",
        params={"window": 60},
    ),
)


__all__ = ["FEATURES", "bollinger_pctb", "dist_ma", "half_life", "zdist"]
