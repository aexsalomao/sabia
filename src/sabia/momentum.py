# Momentum family: oscillators and rate-of-change measures of directional persistence.
# All features are strictly trailing and panel-safe -- the whole expression is evaluated within
# each symbol via .over(symbol), so windows never bleed across symbols.

from __future__ import annotations

from functools import partial

import polars as pl

from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence, ewm_effective_warmup


def rsi(close: str = Column.CLOSE, *, period: int = 14, symbol: str = Column.SYMBOL) -> pl.Expr:
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
    return value.over(symbol).alias(f"rsi_{period}")


def _rsi_warmup(period: int) -> int:
    # RSI is an EWM applied to a diff(), so the analytic EWM warmup is one decay-bar short in
    # practice; +period absorbs the diff offset and recursive accumulation (validated by parity).
    return ewm_effective_warmup(1 / period) + period


_RSI_PERIODS = (14,)

FEATURES: tuple[RegisteredFeature, ...] = tuple(
    make_feature(
        rsi,
        build=partial(rsi, period=period),
        name=f"rsi_{period}",
        family=Family.MOMENTUM,
        native_band=(Horizon.SHORT, Horizon.MEDIUM),
        lookback=period,
        min_history=period + 1,
        recurrence=Recurrence.RECURSIVE,
        effective_warmup=_rsi_warmup(period),
        cost_class=Cost.O1,
        inputs=(Column.CLOSE,),
        citation="Wilder (1978)",
        params={"period": period},
    )
    for period in _RSI_PERIODS
)


__all__ = ["FEATURES", "rsi"]
