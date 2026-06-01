# Distribution family: rolling shape and downside moments of the return distribution. All FINITE,
# strictly trailing, panel-safe via .over(symbol). Zero-variance windows yield null, never NaN.

from __future__ import annotations

from functools import partial

import polars as pl

from sabia._math import safe_log, safe_sqrt
from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence

_BANDS = (Horizon.SHORT, Horizon.MEDIUM)


def _log_return(close: str) -> pl.Expr:
    return safe_log(pl.col(close) / pl.col(close).shift(1))


def skew(close: str = Column.CLOSE, *, window: int = 63, symbol: str = Column.SYMBOL) -> pl.Expr:
    """Rolling skewness of log returns over ``window`` bars. FINITE. A flat window yields null."""
    value = _log_return(close).rolling_skew(window, min_samples=window).fill_nan(None)
    return value.over(symbol).alias(f"skew_{window}")


def kurtosis(
    close: str = Column.CLOSE, *, window: int = 63, symbol: str = Column.SYMBOL
) -> pl.Expr:
    """Rolling excess kurtosis of log returns over ``window`` bars. FINITE. Flat window -> null."""
    value = _log_return(close).rolling_kurtosis(window, min_samples=window).fill_nan(None)
    return value.over(symbol).alias(f"kurtosis_{window}")


def downside_dev(
    close: str = Column.CLOSE, *, window: int = 21, symbol: str = Column.SYMBOL
) -> pl.Expr:
    """Downside deviation: RMS of negative log returns over ``window`` bars. FINITE.

    The semivariance behind the Sortino ratio -- only losses contribute. Citation: Sortino (1991).
    """
    # clip (not min_horizontal) so a null return stays null instead of being imputed to 0.
    downside = _log_return(close).clip(upper_bound=0.0)
    variance = (downside**2).rolling_mean(window, min_samples=window)
    return safe_sqrt(variance).over(symbol).alias(f"downside_dev_{window}")


FEATURES: tuple[RegisteredFeature, ...] = (
    make_feature(
        skew,
        build=partial(skew, window=63),
        name="skew_63",
        family=Family.DISTRIBUTION,
        native_band=_BANDS,
        lookback=63,
        min_history=64,
        recurrence=Recurrence.FINITE,
        effective_warmup=64,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="rolling third moment",
        params={"window": 63},
    ),
    make_feature(
        kurtosis,
        build=partial(kurtosis, window=63),
        name="kurtosis_63",
        family=Family.DISTRIBUTION,
        native_band=_BANDS,
        lookback=63,
        min_history=64,
        recurrence=Recurrence.FINITE,
        effective_warmup=64,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="rolling fourth moment",
        params={"window": 63},
    ),
    make_feature(
        downside_dev,
        build=partial(downside_dev, window=21),
        name="downside_dev_21",
        family=Family.DISTRIBUTION,
        native_band=_BANDS,
        lookback=21,
        min_history=22,
        recurrence=Recurrence.FINITE,
        effective_warmup=22,
        cost_class=Cost.LINEAR,
        inputs=(Column.CLOSE,),
        citation="Sortino (1991)",
        params={"window": 21},
    ),
)


__all__ = ["FEATURES", "downside_dev", "kurtosis", "skew"]
