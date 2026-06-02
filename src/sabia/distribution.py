# Distribution family: rolling shape and downside moments of the return distribution. All FINITE,
# strictly trailing, panel-safe via .over(symbol). Close-based, so they use close@tr (FEATURES.md
# 2.2). Zero-variance / flat windows yield null, never NaN (FEATURES.md 4.5).

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from sabia._expr import grouped
from sabia._math import log_return, safe_div, safe_sqrt
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, Evidence, Family, Horizon, Recurrence, Unit
from sabia.typing import CLOSE_TR, PriceRole

_BANDS = (Horizon.SHORT, Horizon.MEDIUM)
_CLM = Reference("Campbell, Lo & MacKinlay", 1997)


def _log_return(s: BarSchema, close: PriceRole) -> pl.Expr:
    c = pl.col(s.column(close))
    return log_return(c, c.shift(1))


def skew(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Rolling skewness of log returns over ``window`` bars. FINITE, UNITLESS.

    A flat window (zero dispersion) yields null. Citation: Campbell, Lo & MacKinlay (1997).
    """
    name = naming("skew", window)

    def build(s: BarSchema) -> pl.Expr:
        value = _log_return(s, close).rolling_skew(window, min_samples=window).fill_nan(None)
        return grouped(value, s.symbol_col).alias(name)

    return _finite_dist(build, name, window, (close,), Unit.UNITLESS, _CLM)


def kurt(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Rolling excess kurtosis of log returns over ``window`` bars. FINITE, UNITLESS.

    A flat window yields null. Citation: Campbell, Lo & MacKinlay (1997).
    """
    name = naming("kurt", window)

    def build(s: BarSchema) -> pl.Expr:
        value = _log_return(s, close).rolling_kurtosis(window, min_samples=window).fill_nan(None)
        return grouped(value, s.symbol_col).alias(name)

    return _finite_dist(build, name, window, (close,), Unit.UNITLESS, _CLM)


def downside_dev(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Downside deviation: RMS of negative log returns over ``window`` bars. FINITE.

    The semivariance behind the Sortino ratio -- only losses contribute. Clipping (not
    ``min_horizontal``) keeps a null return null rather than imputing it to 0. Citation: Sortino &
    Van der Meer (1991).
    """
    name = naming("downside_dev", window)

    def build(s: BarSchema) -> pl.Expr:
        downside = _log_return(s, close).clip(upper_bound=0.0)
        variance = (downside**2).rolling_mean(window, min_samples=window)
        return grouped(safe_sqrt(variance), s.symbol_col).alias(name)

    return _finite_dist(
        build,
        name,
        window,
        (close,),
        Unit.RETURN_STD_PER_BAR,
        Reference("Sortino & Van der Meer", 1991),
        evidence=Evidence.ACADEMIC_SINGLE,
    )


def up_down_vol_ratio(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Ratio of upside to downside semi-deviation over ``window`` bars. FINITE, RATIO.

    ``sqrt(mean(max(r,0)^2)) / sqrt(mean(min(r,0)^2))`` -- asymmetry of the return distribution.
    Zero downside (no losses in the window) yields null. Citation: Campbell, Lo & MacKinlay (1997).
    """
    name = naming("up_down_vol_ratio", window)

    def build(s: BarSchema) -> pl.Expr:
        r = _log_return(s, close)
        up = (r.clip(lower_bound=0.0) ** 2).rolling_mean(window, min_samples=window)
        down = (r.clip(upper_bound=0.0) ** 2).rolling_mean(window, min_samples=window)
        value = safe_div(safe_sqrt(up), safe_sqrt(down))
        return grouped(value, s.symbol_col).alias(name)

    return _finite_dist(build, name, window, (close,), Unit.RATIO, _CLM)


def _finite_dist(
    build: Callable[[BarSchema], pl.Expr],
    name: str,
    window: int,
    roles: tuple[PriceRole, ...],
    unit: Unit,
    formula: Reference,
    *,
    evidence: Evidence = Evidence.FORMULA_ONLY,
) -> BoundFeature:
    return bind_feature(
        build,
        name=name,
        family=Family.DISTRIBUTION,
        native_band=_BANDS,
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.LINEAR,
        input_roles=roles,
        output_unit=unit,
        evidence=evidence,
        citation=Citation(formula=formula),
        params=FrozenParams(window=window),
    )


FEATURES: tuple[BoundFeature, ...] = (
    skew(window=21),
    kurt(window=21),
    downside_dev(window=21),
    up_down_vol_ratio(window=21),
)


__all__ = [
    "FEATURES",
    "downside_dev",
    "kurt",
    "skew",
    "up_down_vol_ratio",
]
