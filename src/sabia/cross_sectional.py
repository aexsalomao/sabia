# Cross-sectional family: rank / standardize a per-symbol signal across the universe at each
# timestamp. Evaluated in two passes (registry.evaluate): the per-symbol SIGNAL (trailing,
# .over(symbol)) is materialized first, then the cross-sectional REDUCTION runs within each
# timestamp slice (.over(timestamp)) -- Polars cannot nest those two groupings in one expression.
# The frame must carry the complete cross-section at each timestamp (validate cross_sectional=True).

from __future__ import annotations

from functools import partial

import polars as pl

from sabia._math import safe_log
from sabia.normalize import xs_rank, xs_zscore
from sabia.registry import XS_SIGNAL_COLUMN, RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence


def momentum_signal(
    close: str = Column.CLOSE, *, window: int, symbol: str = Column.SYMBOL
) -> pl.Expr:
    """Per-symbol ``window``-bar log-return signal (the input to a cross-sectional reduction)."""
    return safe_log(pl.col(close) / pl.col(close).shift(window)).over(symbol)


def volatility_signal(
    close: str = Column.CLOSE, *, window: int, symbol: str = Column.SYMBOL
) -> pl.Expr:
    """Per-symbol realized-volatility signal over ``window`` bars."""
    log_return = safe_log(pl.col(close) / pl.col(close).shift(1))
    return log_return.rolling_std(window, min_samples=window).over(symbol)


def _xs_rank(name: str, timestamp: str = Column.TIMESTAMP) -> pl.Expr:
    return xs_rank(pl.col(XS_SIGNAL_COLUMN), over=timestamp).alias(name)


def _xs_zscore(name: str, timestamp: str = Column.TIMESTAMP) -> pl.Expr:
    return xs_zscore(pl.col(XS_SIGNAL_COLUMN), over=timestamp).alias(name)


FEATURES: tuple[RegisteredFeature, ...] = (
    make_feature(
        momentum_signal,
        build=partial(_xs_rank, "xs_rank_mom_252"),
        signal=partial(momentum_signal, window=252),
        name="xs_rank_mom_252",
        family=Family.CROSS_SECTIONAL,
        native_band=(Horizon.LONG,),
        lookback=252,
        min_history=253,
        recurrence=Recurrence.FINITE,
        effective_warmup=253,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="Jegadeesh & Titman (1993)",
        params={"window": 252, "reduction": "rank"},
    ),
    make_feature(
        momentum_signal,
        build=partial(_xs_zscore, "xs_zscore_ret_21"),
        signal=partial(momentum_signal, window=21),
        name="xs_zscore_ret_21",
        family=Family.CROSS_SECTIONAL,
        native_band=(Horizon.MEDIUM,),
        lookback=21,
        min_history=22,
        recurrence=Recurrence.FINITE,
        effective_warmup=22,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="cross-sectional relative strength",
        params={"window": 21, "reduction": "zscore"},
    ),
    make_feature(
        volatility_signal,
        build=partial(_xs_rank, "xs_rank_vol_63"),
        signal=partial(volatility_signal, window=63),
        name="xs_rank_vol_63",
        family=Family.CROSS_SECTIONAL,
        native_band=(Horizon.MEDIUM,),
        lookback=63,
        min_history=64,
        recurrence=Recurrence.FINITE,
        effective_warmup=64,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="low-volatility anomaly",
        params={"window": 63, "reduction": "rank"},
    ),
)


__all__ = ["FEATURES", "momentum_signal", "volatility_signal"]
