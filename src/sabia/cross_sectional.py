# Cross-sectional family: rank / standardize a per-symbol signal across the universe at each
# timestamp. Evaluated in two passes (registry.evaluate): the per-symbol SIGNAL (trailing,
# .over(symbol)) is materialized first, then the cross-sectional REDUCTION runs within each
# timestamp slice (.over(timestamp)) -- Polars cannot nest those two groupings in one expression.
# The frame must carry the complete cross-section at each timestamp (validate complete_panel=True).
# Ranks are ascending: high momentum -> high rank (FEATURES.md 4.6). Close-based, close@tr.

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from sabia._expr import grouped
from sabia._math import log_return, safe_div, safe_sqrt
from sabia._validate_params import int_at_least, less_than, non_negative_int, positive_int
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import XS_SIGNAL_COLUMN, BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, Evidence, Family, Horizon, Recurrence, Unit
from sabia.typing import CLOSE_TR, MARKET_RET, FactorRole, PriceRole

_JT = Reference("Jegadeesh & Titman", 1993)


def _xs_rank_reduce(s: BarSchema) -> pl.Expr:
    # Ascending percentile rank in (0, 1] within each timestamp slice; ties take the average rank.
    # Null behavior (FEATURES.md 4.5): a null signal is excluded from BOTH the rank and the
    # denominator -- `count()` counts non-null values -- and stays null in the output. So with k
    # valid names and one null at a timestamp, the k valid names rank over k, and the null is null.
    sig = pl.col(XS_SIGNAL_COLUMN)
    return sig.rank(method="average").over(s.timestamp_col) / sig.count().over(s.timestamp_col)


def _xs_zscore_reduce(s: BarSchema) -> pl.Expr:
    # Cross-sectional standardization within each timestamp slice; zero dispersion -> null.
    sig = pl.col(XS_SIGNAL_COLUMN)
    mean = sig.mean().over(s.timestamp_col)
    std = sig.std().over(s.timestamp_col)
    return pl.when(std == 0).then(None).otherwise((sig - mean) / std)


def xs_rank_mom(
    *, formation: int = 252, skip: int = 21, close: PriceRole = CLOSE_TR
) -> BoundFeature:
    """Cross-sectional percentile rank of ``mom_{formation}_{skip}``. FINITE, RANK_0_1.

    The canonical Jegadeesh-Titman momentum factor: rank each name's 12-1 momentum across the
    universe at each date. Citation: Jegadeesh & Titman (1993).
    """
    positive_int("formation", formation)
    non_negative_int("skip", skip)
    less_than("skip", skip, "formation", formation)
    name = naming("xs_rank_mom", formation, skip)

    def signal(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        return grouped(safe_div(c.shift(skip), c.shift(formation)).log(), s.symbol_col)

    return _xs_feature(
        _xs_rank_reduce,
        signal,
        name,
        formation,
        close,
        Unit.RANK_0_1,
        FrozenParams(formation=formation, skip=skip),
    )


def xs_z_mom(*, formation: int = 252, skip: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Cross-sectional z-score of ``mom_{formation}_{skip}`` across the universe. FINITE, ZSCORE.

    Citation: Jegadeesh & Titman (1993).
    """
    positive_int("formation", formation)
    non_negative_int("skip", skip)
    less_than("skip", skip, "formation", formation)
    name = naming("xs_z_mom", formation, skip)

    def signal(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        return grouped(safe_div(c.shift(skip), c.shift(formation)).log(), s.symbol_col)

    return _xs_feature(
        _xs_zscore_reduce,
        signal,
        name,
        formation,
        close,
        Unit.ZSCORE,
        FrozenParams(formation=formation, skip=skip),
    )


def rev_1m(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Short-term reversal: cross-sectional rank of the negated ``window``-bar return. RANK_0_1.

    Recent losers rank high (they tend to rebound). Citation: Jegadeesh (1990).
    """
    positive_int("window", window)
    name = naming("rev_1m", window)

    def signal(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        return grouped(-safe_div(c, c.shift(window)).log(), s.symbol_col)

    return _xs_feature(
        _xs_rank_reduce,
        signal,
        name,
        window,
        close,
        Unit.RANK_0_1,
        FrozenParams(window=window),
        formula=Reference("Jegadeesh", 1990),
    )


def _xs_feature(  # noqa: PLR0913 -- a cross-sectional spec genuinely carries these axes
    reduce: Callable[[BarSchema], pl.Expr],
    signal: Callable[[BarSchema], pl.Expr],
    name: str,
    formation: int,
    close: PriceRole,
    unit: Unit,
    params: FrozenParams,
    *,
    formula: Reference = _JT,
) -> BoundFeature:
    return bind_feature(
        reduce,
        name=name,
        family=Family.CROSS_SECTIONAL,
        native_band=(Horizon.LONG,),
        lookback=formation,
        min_history=formation + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=formation + 1,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=unit,
        output_range=(0.0, 1.0) if unit is Unit.RANK_0_1 else None,
        evidence=Evidence.ACADEMIC_REPLICATED,
        citation=Citation(formula=formula),
        params=params,
        requires_universe=True,
        requires_complete_panel=True,
        signal=signal,
    )


# --- single-factor market model (per symbol, not a cross-section reduction) -----------------------
# beta / idiosyncratic vol regress each name's returns on the market factor over a trailing window.
# These are per-symbol time-series (no XS reduction, so signal=None), but they belong to the
# cross-sectional family: the market factor links a name to the rest of the universe, and the
# residual (idio vol) and slope (beta) are the inputs to the size/value/low-vol factor zoo
# (FEATURES.md 12). Solved in closed form from rolling population moments -- no per-row Python.

_SHARPE = Reference("Sharpe", 1964)
_AHXZ = Reference("Ang, Hodrick, Xing & Zhang", 2006)


def _roll_cov(a: pl.Expr, b: pl.Expr, window: int) -> pl.Expr:
    # Population covariance of two aligned series over the trailing window: E[ab] - E[a]E[b]. With
    # a == b this is the variance. min_samples=window emits null until the window is full.
    mean_ab = (a * b).rolling_mean(window, min_samples=window)
    mean_a = a.rolling_mean(window, min_samples=window)
    mean_b = b.rolling_mean(window, min_samples=window)
    return mean_ab - mean_a * mean_b


def beta(
    *, window: int = 252, close: PriceRole = CLOSE_TR, market: FactorRole = MARKET_RET
) -> BoundFeature:
    """Rolling market beta: OLS slope of asset log returns on the market, ``Cov(r,m)/Var(m)``.

    A flat market window (zero variance) -> null. FINITE, UNITLESS. The CAPM slope (Sharpe 1964),
    estimated rolling per name. Citation: Sharpe (1964); empirically Fama & MacBeth (1973).
    """
    int_at_least("window", window, 2)
    name = naming("beta", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        r = log_return(c, c.shift(1))
        m = pl.col(s.column(market))
        value = safe_div(_roll_cov(r, m, window), _roll_cov(m, m, window))
        return grouped(value, s.symbol_col).alias(name)

    return _market_feature(
        build,
        name,
        window,
        (close, market),
        Unit.UNITLESS,
        Citation(formula=_SHARPE, empirical=(Reference("Fama & MacBeth", 1973),)),
    )


def idio_vol(
    *, window: int = 252, close: PriceRole = CLOSE_TR, market: FactorRole = MARKET_RET
) -> BoundFeature:
    """Idiosyncratic volatility: per-bar std of the residual from the market-model regression.

    ``sqrt(Var(r) - Cov(r,m)^2 / Var(m))`` -- the part of return variance the market does not
    explain. A flat market window -> null. FINITE, per-bar. Citation: Ang, Hodrick, Xing & Zhang
    (2006) (the idiosyncratic-volatility anomaly).
    """
    int_at_least("window", window, 2)
    name = naming("idio_vol", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        r = log_return(c, c.shift(1))
        m = pl.col(s.column(market))
        cov_rm = _roll_cov(r, m, window)
        resid_var = _roll_cov(r, r, window) - safe_div(cov_rm * cov_rm, _roll_cov(m, m, window))
        return grouped(safe_sqrt(resid_var), s.symbol_col).alias(name)

    return _market_feature(
        build, name, window, (close, market), Unit.RETURN_STD_PER_BAR, Citation(formula=_AHXZ)
    )


def _market_feature(
    build: Callable[[BarSchema], pl.Expr],
    name: str,
    window: int,
    roles: tuple[PriceRole | FactorRole, ...],
    unit: Unit,
    citation: Citation,
) -> BoundFeature:
    # Per-symbol factor-model feature: FINITE over the return window (window+1 closes), no XS panel
    # requirement (the market factor is carried per row, session-aligned by the caller, FEATURES.md
    # 2.3). Closed form, so cost is LINEAR like the trend OLS slope, not a HEAVY kernel.
    return bind_feature(
        build,
        name=name,
        family=Family.CROSS_SECTIONAL,
        native_band=(Horizon.LONG,),
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.LINEAR,
        input_roles=roles,
        output_unit=unit,
        evidence=Evidence.ACADEMIC_REPLICATED,
        citation=citation,
        params=FrozenParams(window=window),
    )


FEATURES: tuple[BoundFeature, ...] = (
    xs_rank_mom(formation=252, skip=21),
    xs_z_mom(formation=252, skip=21),
    rev_1m(window=21),
    beta(window=252),
    idio_vol(window=252),
)


__all__ = [
    "FEATURES",
    "beta",
    "idio_vol",
    "rev_1m",
    "xs_rank_mom",
    "xs_z_mom",
]
