# Trend family: moving averages and directional-strength measures. SMAs are FINITE; EMAs and ADX
# (Wilder, double-RMA-smoothed) are RECURSIVE. Native to the MEDIUM/LONG bands. All strictly
# trailing and panel-safe via .over(symbol).

from __future__ import annotations

from functools import partial

import polars as pl

from sabia._math import safe_div
from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence, ewm_effective_warmup


def sma(close: str = Column.CLOSE, *, window: int = 50, symbol: str = Column.SYMBOL) -> pl.Expr:
    """Simple moving average of close over ``window`` bars. FINITE."""
    value = pl.col(close).rolling_mean(window, min_samples=window)
    return value.over(symbol).alias(f"sma_{window}")


def ema(close: str = Column.CLOSE, *, span: int = 12, symbol: str = Column.SYMBOL) -> pl.Expr:
    """Exponential moving average of close (``adjust=False``). RECURSIVE."""
    value = pl.col(close).ewm_mean(span=span, adjust=False, min_samples=span)
    return value.over(symbol).alias(f"ema_{span}")


def adx(
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    *,
    window: int = 14,
    symbol: str = Column.SYMBOL,
) -> pl.Expr:
    """Average Directional Index, in [0, 100]: Wilder's trend-strength measure. RECURSIVE.

    Built from smoothed directional movement (+DM/-DM) and true range, then a second RMA over the
    directional index DX. A bar with no directional spread yields null. Citation: Wilder (1978).
    """
    return _adx_core(high, low, close, window).over(symbol).alias(f"adx_{window}")


def _rma(expr: pl.Expr, window: int) -> pl.Expr:
    # Wilder's smoothing == EWM with alpha = 1/window.
    return expr.ewm_mean(alpha=1 / window, adjust=False, min_samples=window)


def _adx_core(high: str, low: str, close: str, window: int) -> pl.Expr:
    up_move = pl.col(high) - pl.col(high).shift(1)
    down_move = pl.col(low).shift(1) - pl.col(low)
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)

    prev_close = pl.col(close).shift(1)
    true_range = pl.max_horizontal(
        pl.col(high) - pl.col(low),
        (pl.col(high) - prev_close).abs(),
        (pl.col(low) - prev_close).abs(),
    )
    atr = _rma(true_range, window)
    plus_di = 100 * safe_div(_rma(plus_dm, window), atr)
    minus_di = 100 * safe_div(_rma(minus_dm, window), atr)
    dx = 100 * safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return _rma(dx, window)


def _ewm_warmup(span: int) -> int:
    return ewm_effective_warmup(2 / (span + 1)) + span


FEATURES: tuple[RegisteredFeature, ...] = (
    make_feature(
        sma,
        build=partial(sma, window=50),
        name="sma_50",
        family=Family.TREND,
        native_band=(Horizon.MEDIUM,),
        lookback=50,
        min_history=50,
        recurrence=Recurrence.FINITE,
        effective_warmup=50,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="moving average",
        params={"window": 50},
    ),
    make_feature(
        sma,
        build=partial(sma, window=200),
        name="sma_200",
        family=Family.TREND,
        native_band=(Horizon.LONG,),
        lookback=200,
        min_history=200,
        recurrence=Recurrence.FINITE,
        effective_warmup=200,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="moving average",
        params={"window": 200},
    ),
    make_feature(
        ema,
        build=partial(ema, span=12),
        name="ema_12",
        family=Family.TREND,
        native_band=(Horizon.MEDIUM,),
        lookback=12,
        min_history=12,
        recurrence=Recurrence.RECURSIVE,
        effective_warmup=_ewm_warmup(12),
        cost_class=Cost.O1,
        inputs=(Column.CLOSE,),
        citation="exponential moving average",
        params={"span": 12},
    ),
    make_feature(
        ema,
        build=partial(ema, span=26),
        name="ema_26",
        family=Family.TREND,
        native_band=(Horizon.MEDIUM,),
        lookback=26,
        min_history=26,
        recurrence=Recurrence.RECURSIVE,
        effective_warmup=_ewm_warmup(26),
        cost_class=Cost.O1,
        inputs=(Column.CLOSE,),
        citation="exponential moving average",
        params={"span": 26},
    ),
    make_feature(
        adx,
        build=partial(adx, window=14),
        name="adx_14",
        family=Family.TREND,
        native_band=(Horizon.MEDIUM, Horizon.LONG),
        lookback=14,
        min_history=27,
        recurrence=Recurrence.RECURSIVE,
        effective_warmup=2 * ewm_effective_warmup(1 / 14) + 2 * 14,
        cost_class=Cost.O1,
        inputs=(Column.HIGH, Column.LOW, Column.CLOSE),
        citation="Wilder (1978)",
        params={"window": 14},
    ),
)


__all__ = ["FEATURES", "adx", "ema", "sma"]
