# Returns family: simple and log price changes over one or more bars. These underpin most other
# features. All FINITE, strictly trailing, panel-safe via .over(symbol). Degenerate inputs
# (non-positive prices, zero base) yield null rather than inf/NaN (FEATURES.md 3.5).

from __future__ import annotations

from functools import partial

import polars as pl

from sabia._expr import grouped
from sabia.registry import RegisteredFeature, make_feature
from sabia.spec import Column, Cost, Family, Horizon, Recurrence


def ret_simple(close: str = Column.CLOSE, *, symbol: str | None = Column.SYMBOL) -> pl.Expr:
    """One-bar simple (arithmetic) return. A zero prior price yields null. FINITE."""
    prev = pl.col(close).shift(1)
    value = pl.when(prev == 0).then(None).otherwise(pl.col(close) / prev - 1)
    return grouped(value, symbol).alias("ret_simple")


def ret_log(
    close: str = Column.CLOSE, *, period: int = 1, symbol: str | None = Column.SYMBOL
) -> pl.Expr:
    """``period``-bar log return ``ln(P_t / P_{t-period})``. Non-positive ratio -> null. FINITE."""
    ratio = pl.col(close) / pl.col(close).shift(period)
    value = pl.when(ratio <= 0).then(None).otherwise(ratio.log())
    return grouped(value, symbol).alias(f"ret_log_{period}")


def _band(period: int) -> Horizon:
    return Horizon.SHORT if period <= 10 else Horizon.MEDIUM


_LOG_PERIODS = (1, 5, 21)

FEATURES: tuple[RegisteredFeature, ...] = (
    make_feature(
        ret_simple,
        build=ret_simple,
        name="ret_simple",
        family=Family.RETURNS,
        native_band=(Horizon.SHORT,),
        lookback=1,
        min_history=2,
        recurrence=Recurrence.FINITE,
        effective_warmup=2,
        cost_class=Cost.O1,
        inputs=(Column.CLOSE,),
        citation="Campbell, Lo & MacKinlay (1997)",
        params={},
    ),
    *(
        make_feature(
            ret_log,
            build=partial(ret_log, period=period),
            name=f"ret_log_{period}",
            family=Family.RETURNS,
            native_band=(_band(period),),
            lookback=period,
            min_history=period + 1,
            recurrence=Recurrence.FINITE,
            effective_warmup=period + 1,
            cost_class=Cost.O1,
            inputs=(Column.CLOSE,),
            citation="Campbell, Lo & MacKinlay (1997)",
            params={"period": period},
        )
        for period in _LOG_PERIODS
    ),
)


__all__ = ["FEATURES", "ret_log", "ret_simple"]
