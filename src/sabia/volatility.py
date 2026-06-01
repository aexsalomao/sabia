# Volatility family: per-bar volatility / range estimators. Close-to-close and the OHLC range
# estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) are FINITE; ATR is Wilder-RMA
# RECURSIVE. All strictly trailing, panel-safe via .over(symbol), and null (never inf/NaN) on a
# log-domain breach. Estimators are per-bar (annualization is a downstream choice).

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import log

import polars as pl

from sabia._math import safe_log, safe_sqrt
from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence, ewm_effective_warmup

_LN2 = log(2.0)
_VOL_BANDS = (Horizon.SHORT, Horizon.MEDIUM)
_OHLC = (Column.OPEN, Column.HIGH, Column.LOW, Column.CLOSE)


def vol_close(
    close: str = Column.CLOSE, *, window: int = 21, symbol: str = Column.SYMBOL
) -> pl.Expr:
    """Close-to-close volatility: rolling std of one-bar log returns. FINITE. Citation: classic."""
    log_return = safe_log(pl.col(close) / pl.col(close).shift(1))
    value = log_return.rolling_std(window, min_samples=window)
    return value.over(symbol).alias(f"vol_close_{window}")


def vol_parkinson(
    high: str = Column.HIGH, low: str = Column.LOW, *, window: int = 21, symbol: str = Column.SYMBOL
) -> pl.Expr:
    """Parkinson (1980) high-low range volatility. FINITE."""
    term = safe_log(pl.col(high) / pl.col(low)) ** 2
    variance = term.rolling_mean(window, min_samples=window) / (4.0 * _LN2)
    return safe_sqrt(variance).over(symbol).alias(f"vol_parkinson_{window}")


def vol_gk(
    open_: str = Column.OPEN,
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    *,
    window: int = 21,
    symbol: str = Column.SYMBOL,
) -> pl.Expr:
    """Garman-Klass (1980) OHLC volatility. FINITE."""
    hl = safe_log(pl.col(high) / pl.col(low))
    co = safe_log(pl.col(close) / pl.col(open_))
    term = 0.5 * hl**2 - (2.0 * _LN2 - 1.0) * co**2
    variance = term.rolling_mean(window, min_samples=window)
    return safe_sqrt(variance).over(symbol).alias(f"vol_gk_{window}")


def vol_rs(
    open_: str = Column.OPEN,
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    *,
    window: int = 21,
    symbol: str = Column.SYMBOL,
) -> pl.Expr:
    """Rogers-Satchell (1991) drift-independent OHLC volatility. FINITE."""
    variance = _rs_term(open_, high, low, close).rolling_mean(window, min_samples=window)
    return safe_sqrt(variance).over(symbol).alias(f"vol_rs_{window}")


def vol_yz(
    open_: str = Column.OPEN,
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    *,
    window: int = 21,
    symbol: str = Column.SYMBOL,
) -> pl.Expr:
    """Yang-Zhang (2000) volatility: overnight + open-close + Rogers-Satchell. FINITE."""
    overnight = safe_log(pl.col(open_) / pl.col(close).shift(1))
    open_close = safe_log(pl.col(close) / pl.col(open_))
    sigma_o2 = overnight.rolling_var(window, min_samples=window)
    sigma_c2 = open_close.rolling_var(window, min_samples=window)
    sigma_rs2 = _rs_term(open_, high, low, close).rolling_mean(window, min_samples=window)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    variance = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
    return safe_sqrt(variance).over(symbol).alias(f"vol_yz_{window}")


def atr(
    high: str = Column.HIGH,
    low: str = Column.LOW,
    close: str = Column.CLOSE,
    *,
    window: int = 14,
    symbol: str = Column.SYMBOL,
) -> pl.Expr:
    """Average True Range (Wilder 1978), RMA-smoothed. RECURSIVE."""
    prev_close = pl.col(close).shift(1)
    true_range = pl.max_horizontal(
        pl.col(high) - pl.col(low),
        (pl.col(high) - prev_close).abs(),
        (pl.col(low) - prev_close).abs(),
    )
    value = true_range.ewm_mean(alpha=1 / window, adjust=False, min_samples=window)
    return value.over(symbol).alias(f"atr_{window}")


def _rs_term(open_: str, high: str, low: str, close: str) -> pl.Expr:
    hc = safe_log(pl.col(high) / pl.col(close))
    ho = safe_log(pl.col(high) / pl.col(open_))
    lc = safe_log(pl.col(low) / pl.col(close))
    lo = safe_log(pl.col(low) / pl.col(open_))
    return hc * ho + lc * lo


def _finite(
    fn: Callable[..., pl.Expr],
    name_fn: Callable[[int], str],
    *,
    windows: tuple[int, ...],
    inputs: tuple[Column, ...],
    citation: str,
    extra_history: int = 0,
) -> list[RegisteredFeature]:
    return [
        make_feature(
            fn,
            build=partial(fn, window=w),
            name=name_fn(w),
            family=Family.VOLATILITY,
            native_band=_VOL_BANDS,
            lookback=w,
            min_history=w + extra_history,
            recurrence=Recurrence.FINITE,
            effective_warmup=w + extra_history,
            cost_class=Cost.LINEAR,
            inputs=inputs,
            citation=citation,
            params={"window": w},
        )
        for w in windows
    ]


FEATURES: tuple[RegisteredFeature, ...] = (
    *_finite(
        vol_close,
        lambda w: f"vol_close_{w}",
        windows=(21, 63),
        inputs=(Column.CLOSE,),
        citation="close-to-close",
        extra_history=1,
    ),
    *_finite(
        vol_parkinson,
        lambda w: f"vol_parkinson_{w}",
        windows=(21,),
        inputs=(Column.HIGH, Column.LOW),
        citation="Parkinson (1980)",
    ),
    *_finite(
        vol_gk,
        lambda w: f"vol_gk_{w}",
        windows=(21,),
        inputs=_OHLC,
        citation="Garman & Klass (1980)",
    ),
    *_finite(
        vol_rs,
        lambda w: f"vol_rs_{w}",
        windows=(21,),
        inputs=_OHLC,
        citation="Rogers & Satchell (1991)",
    ),
    *_finite(
        vol_yz,
        lambda w: f"vol_yz_{w}",
        windows=(21,),
        inputs=_OHLC,
        citation="Yang & Zhang (2000)",
        extra_history=1,
    ),
    make_feature(
        atr,
        build=partial(atr, window=14),
        name="atr_14",
        family=Family.VOLATILITY,
        native_band=_VOL_BANDS,
        lookback=14,
        min_history=14,
        recurrence=Recurrence.RECURSIVE,
        effective_warmup=ewm_effective_warmup(1 / 14) + 14,
        cost_class=Cost.O1,
        inputs=(Column.HIGH, Column.LOW, Column.CLOSE),
        citation="Wilder (1978)",
        params={"window": 14},
    ),
)


__all__ = [
    "FEATURES",
    "atr",
    "vol_close",
    "vol_gk",
    "vol_parkinson",
    "vol_rs",
    "vol_yz",
]
