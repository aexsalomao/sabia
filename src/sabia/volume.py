# Volume / liquidity family. All FINITE (a non-decaying cumulative like raw OBV cannot satisfy the
# windowed-recompute parity guarantee, so OBV ships only in its windowed signed-volume form).
# Strictly trailing, panel-safe via .over(symbol); degenerate divides yield null (FEATURES.md 3.5).

from __future__ import annotations

from functools import partial

import polars as pl

from sabia._expr import grouped
from sabia._math import safe_div
from sabia.normalize import zscore
from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence

_BANDS = (Horizon.SHORT, Horizon.MEDIUM)


def amihud(
    close: str = Column.CLOSE,
    volume: str = Column.VOLUME,
    *,
    window: int = 21,
    symbol: str | None = Column.SYMBOL,
) -> pl.Expr:
    """Amihud (2002) illiquidity: rolling mean of |return| per unit dollar volume. FINITE.

    Zero dollar volume (a halted bar) yields null, not inf -- there is no price impact to measure.
    """
    prev_close = pl.col(close).shift(1)
    abs_return = safe_div(pl.col(close) - prev_close, prev_close).abs()
    dollar_volume = pl.col(close) * pl.col(volume)
    ratio = safe_div(abs_return, dollar_volume)
    value = ratio.rolling_mean(window, min_samples=window)
    return grouped(value, symbol).alias(f"amihud_{window}")


def cmf(
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    volume: str = Column.VOLUME,
    *,
    window: int = 21,
    symbol: str | None = Column.SYMBOL,
) -> pl.Expr:
    """Chaikin Money Flow over ``window`` bars, in [-1, 1]. FINITE. Citation: Chaikin."""
    multiplier = safe_div(
        2 * pl.col(close) - pl.col(high) - pl.col(low), pl.col(high) - pl.col(low)
    )
    money_flow_volume = multiplier * pl.col(volume)
    value = safe_div(
        money_flow_volume.rolling_sum(window, min_samples=window),
        pl.col(volume).rolling_sum(window, min_samples=window),
    )
    return grouped(value, symbol).alias(f"cmf_{window}")


def vol_zscore(
    volume: str = Column.VOLUME, *, window: int = 21, symbol: str | None = Column.SYMBOL
) -> pl.Expr:
    """Rolling z-score of volume: how unusual today's volume is vs its recent norm. FINITE."""
    return zscore(pl.col(volume), window, over=symbol).alias(f"vol_zscore_{window}")


def dollar_vol(
    close: str = Column.CLOSE, volume: str = Column.VOLUME, *, symbol: str | None = Column.SYMBOL
) -> pl.Expr:
    """Per-bar dollar volume (price times volume). FINITE."""
    return grouped((pl.col(close) * pl.col(volume)), symbol).alias("dollar_vol")


def adv(
    close: str = Column.CLOSE,
    volume: str = Column.VOLUME,
    *,
    window: int = 21,
    symbol: str | None = Column.SYMBOL,
) -> pl.Expr:
    """Average daily dollar volume over ``window`` bars: a liquidity scale. FINITE."""
    value = (pl.col(close) * pl.col(volume)).rolling_mean(window, min_samples=window)
    return grouped(value, symbol).alias(f"adv_{window}")


def signed_vol(
    close: str = Column.CLOSE,
    volume: str = Column.VOLUME,
    *,
    window: int = 21,
    symbol: str | None = Column.SYMBOL,
) -> pl.Expr:
    """Windowed signed volume (Granville's OBV, FINITE form): net up/down volume over ``window``."""
    signed = pl.col(close).diff().sign() * pl.col(volume)
    value = signed.rolling_sum(window, min_samples=window)
    return grouped(value, symbol).alias(f"signed_vol_{window}")


FEATURES: tuple[RegisteredFeature, ...] = (
    make_feature(
        amihud,
        build=partial(amihud, window=21),
        name="amihud_21",
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=21,
        min_history=22,
        recurrence=Recurrence.FINITE,
        effective_warmup=22,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE, Column.VOLUME),
        citation="Amihud (2002)",
        params={"window": 21},
    ),
    make_feature(
        cmf,
        build=partial(cmf, window=21),
        name="cmf_21",
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=21,
        min_history=21,
        recurrence=Recurrence.FINITE,
        effective_warmup=21,
        cost_class=Cost.LINEAR,
        inputs=(Column.HIGH, Column.LOW, Column.CLOSE, Column.VOLUME),
        citation="Chaikin",
        params={"window": 21},
    ),
    make_feature(
        vol_zscore,
        build=partial(vol_zscore, window=21),
        name="vol_zscore_21",
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=21,
        min_history=21,
        recurrence=Recurrence.FINITE,
        effective_warmup=21,
        cost_class=Cost.LINEAR,
        inputs=(Column.VOLUME,),
        citation="volume z-score",
        params={"window": 21},
    ),
    make_feature(
        dollar_vol,
        build=dollar_vol,
        name="dollar_vol",
        family=Family.VOLUME,
        native_band=(Horizon.SHORT,),
        lookback=1,
        min_history=1,
        recurrence=Recurrence.FINITE,
        effective_warmup=1,
        cost_class=Cost.O1,
        inputs=(Column.CLOSE, Column.VOLUME),
        citation="dollar volume",
        params={},
    ),
    make_feature(
        adv,
        build=partial(adv, window=21),
        name="adv_21",
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=21,
        min_history=21,
        recurrence=Recurrence.FINITE,
        effective_warmup=21,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE, Column.VOLUME),
        citation="average dollar volume",
        params={"window": 21},
    ),
    make_feature(
        signed_vol,
        build=partial(signed_vol, window=21),
        name="signed_vol_21",
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=21,
        min_history=22,
        recurrence=Recurrence.FINITE,
        effective_warmup=22,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE, Column.VOLUME),
        citation="Granville (1963), windowed",
        params={"window": 21},
    ),
)


__all__ = [
    "FEATURES",
    "adv",
    "amihud",
    "cmf",
    "dollar_vol",
    "signed_vol",
    "vol_zscore",
]
