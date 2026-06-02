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
from sabia._math import safe_div
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import XS_SIGNAL_COLUMN, BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, Evidence, Family, Horizon, Recurrence, Unit
from sabia.typing import CLOSE_TR, PriceRole

_JT = Reference("Jegadeesh & Titman", 1993)


def _xs_rank_reduce(s: BarSchema) -> pl.Expr:
    # Ascending percentile rank in (0, 1] within each timestamp slice; ties take the average rank.
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
    name = naming("xs_rank_mom", formation)

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
    name = naming("xs_z_mom", formation)

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


FEATURES: tuple[BoundFeature, ...] = (
    xs_rank_mom(formation=252, skip=21),
    xs_z_mom(formation=252, skip=21),
    rev_1m(window=21),
)


__all__ = [
    "FEATURES",
    "rev_1m",
    "xs_rank_mom",
    "xs_z_mom",
]
