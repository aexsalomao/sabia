# Volume / liquidity family. All FINITE (a non-decaying cumulative like raw OBV cannot satisfy the
# windowed-recompute parity guarantee, so OBV ships only in its windowed signed-volume form). Volume
# uses volume@split; Amihud pairs |TR return| with raw dollar volume (dvol@raw); range-based flow
# measures use split OHLC (FEATURES.md 2.2). Strictly trailing, panel-safe; degenerate divides null.

from __future__ import annotations

from collections.abc import Callable
from math import sqrt

import polars as pl

from sabia._expr import grouped
from sabia._math import log_return, safe_div, safe_log, safe_sqrt
from sabia._validate_params import int_at_least
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, Evidence, Family, Horizon, Recurrence, Unit
from sabia.typing import (
    CLOSE_SPLIT,
    CLOSE_TR,
    DVOL_RAW,
    HIGH_SPLIT,
    LOW_SPLIT,
    VOLUME_SPLIT,
    VWAP_SPLIT,
    PriceRole,
    VolumeRole,
)

_BANDS = (Horizon.SHORT, Horizon.MEDIUM)
_CS_DENOM = 3.0 - 2.0 * sqrt(2.0)  # Corwin-Schultz constant
_CLM = Reference("Campbell, Lo & MacKinlay", 1997)


def vol_z(*, window: int = 21, volume: VolumeRole = VOLUME_SPLIT) -> BoundFeature:
    """Rolling z-score of volume: how unusual today's volume is vs its recent norm. FINITE."""
    int_at_least("window", window, 2)
    name = naming("vol_z", window)

    def build(s: BarSchema) -> pl.Expr:
        v = pl.col(s.column(volume))
        mean = v.rolling_mean(window, min_samples=window)
        std = v.rolling_std(window, min_samples=window)
        value = pl.when(std == 0).then(None).otherwise((v - mean) / std)
        return grouped(value, s.symbol_col).alias(name)

    return _finite_vol(build, name, window, (volume,), Unit.ZSCORE, _CLM)


def rel_volume(*, window: int = 21, volume: VolumeRole = VOLUME_SPLIT) -> BoundFeature:
    """Relative volume: ``volume / SMA(volume, window)``. FINITE, RATIO."""
    int_at_least("window", window, 2)
    name = naming("rel_volume", window)

    def build(s: BarSchema) -> pl.Expr:
        v = pl.col(s.column(volume))
        value = safe_div(v, v.rolling_mean(window, min_samples=window))
        return grouped(value, s.symbol_col).alias(name)

    return _finite_vol(build, name, window, (volume,), Unit.RATIO, _CLM)


def amihud(
    *, window: int = 21, close: PriceRole = CLOSE_TR, dvol: VolumeRole = DVOL_RAW
) -> BoundFeature:
    """Amihud (2002) illiquidity: rolling mean of ``|ret@tr| / dollar_volume@raw``. FINITE, RATIO.

    ``ret@tr`` is the log return ``ln(close / close.shift(1))`` (returns are log unless named
    otherwise, FEATURES.md 4.6); illiquidity averages ``|log return| / dollar_volume@raw``.
    Zero dollar volume (a halted bar) yields null, not inf -- there is no price impact to measure.
    """
    int_at_least("window", window, 2)
    name = naming("amihud", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        abs_return = log_return(c, c.shift(1)).abs()
        ratio = safe_div(abs_return, pl.col(s.column(dvol)))
        value = ratio.rolling_mean(window, min_samples=window)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.LINEAR,
        input_roles=(close, dvol),
        output_unit=Unit.RATIO,
        evidence=Evidence.ACADEMIC_REPLICATED,
        citation=Citation(formula=Reference("Amihud", 2002)),
        params=FrozenParams(window=window),
    )


def vwap_dist_close(
    *, close: PriceRole = CLOSE_SPLIT, vwap: PriceRole = VWAP_SPLIT
) -> BoundFeature:
    """Distance of close from VWAP on the same basis: ``close / vwap - 1``. FINITE, RATIO."""
    name = "vwap_dist_close"

    def build(s: BarSchema) -> pl.Expr:
        value = safe_div(pl.col(s.column(close)), pl.col(s.column(vwap))) - 1
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=(Horizon.SHORT,),
        lookback=1,
        min_history=1,
        recurrence=Recurrence.FINITE,
        effective_warmup=1,
        cost_class=Cost.O1,
        input_roles=(close, vwap),
        output_unit=Unit.RATIO,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=Reference("Berkowitz, Logue & Noser", 1988)),
        params=FrozenParams(),
    )


def cmf(
    *,
    window: int = 21,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
    volume: VolumeRole = VOLUME_SPLIT,
) -> BoundFeature:
    """Chaikin Money Flow over ``window`` bars, in [-1, 1]. FINITE. Citation: Chaikin.

    A flat (doji) bar where ``high == low`` contributes ZERO money flow (multiplier = 0), per
    canonical Chaikin CMF -- its volume still counts toward the denominator. This is the benign
    flat-bar case, distinct from a halt (FEATURES.md 2.4); without it a single doji would null the
    whole REQUIRE_FULL_WINDOW window (FEATURES.md 4.5).
    """
    int_at_least("window", window, 2)
    name = naming("cmf", window)

    def build(s: BarSchema) -> pl.Expr:
        h, low_, c, v = (
            pl.col(s.column(high)),
            pl.col(s.column(low)),
            pl.col(s.column(close)),
            pl.col(s.column(volume)),
        )
        multiplier = pl.when(h == low_).then(0.0).otherwise(safe_div(2 * c - h - low_, h - low_))
        money_flow_volume = multiplier * v
        value = safe_div(
            money_flow_volume.rolling_sum(window, min_samples=window),
            v.rolling_sum(window, min_samples=window),
        )
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=(high, low, close, volume),
        output_unit=Unit.UNITLESS,
        output_range=(-1.0, 1.0),
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Chaikin", 1982)),
        params=FrozenParams(window=window),
    )


def mfi(
    *,
    window: int = 14,
    high: PriceRole = HIGH_SPLIT,
    low: PriceRole = LOW_SPLIT,
    close: PriceRole = CLOSE_SPLIT,
    volume: VolumeRole = VOLUME_SPLIT,
) -> BoundFeature:
    """Money Flow Index, in [0, 100]: volume-weighted RSI on typical price. FINITE."""
    int_at_least("window", window, 2)
    name = naming("mfi", window)

    def build(s: BarSchema) -> pl.Expr:
        h, low_, c, v = (
            pl.col(s.column(high)),
            pl.col(s.column(low)),
            pl.col(s.column(close)),
            pl.col(s.column(volume)),
        )
        tp = (h + low_ + c) / 3.0
        raw_flow = tp * v
        prev = tp.shift(1)
        # Keep a missing typical price null (not imputed to a 0 flow); only a genuine non-rising bar
        # contributes 0. Without the validity guard, a null tp falls through `otherwise` to 0.
        valid = tp.is_not_null() & prev.is_not_null()
        pos_flow = pl.when(valid).then(pl.when(tp > prev).then(raw_flow).otherwise(0.0))
        neg_flow = pl.when(valid).then(pl.when(tp < prev).then(raw_flow).otherwise(0.0))
        pos = pos_flow.rolling_sum(window, min_samples=window)
        neg = neg_flow.rolling_sum(window, min_samples=window)
        ratio = safe_div(pos, neg)
        # A window with no flow at all (flat typical price / halted bar) is degenerate -> null, like
        # RSI; only a window with up-flow but no down-flow saturates at 100 (FEATURES.md 4.5).
        value = (
            pl.when((pos == 0) & (neg == 0))
            .then(None)
            .when(neg == 0)
            .then(pl.lit(100.0))
            .otherwise(100 - 100 / (1 + ratio))
        )
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.LINEAR,
        input_roles=(high, low, close, volume),
        output_unit=Unit.INDEX_0_100,
        output_range=(0.0, 100.0),
        evidence=Evidence.TA_CANON,
        citation=Citation(formula=Reference("Quong & Soudack", 1989)),
        params=FrozenParams(window=window),
    )


def roll_spread(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Roll (1984) implied spread: ``2*sqrt(-cov(dp_t, dp_{t-1}))`` over the window. FINITE, RATIO.

    A non-negative serial covariance (no bid-ask bounce) yields null. Computed from rolling
    moments (no per-row Python).
    """
    int_at_least("window", window, 2)
    name = naming("roll_spread", window)

    def build(s: BarSchema) -> pl.Expr:
        dp = pl.col(s.column(close)).diff()
        dp_prev = dp.shift(1)
        cov = dp.rolling_mean(window, min_samples=window) * dp_prev.rolling_mean(
            window, min_samples=window
        )
        cov = (dp * dp_prev).rolling_mean(window, min_samples=window) - cov
        value = pl.when(cov < 0).then(2 * safe_sqrt(-cov)).otherwise(None)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=window,
        min_history=window + 2,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 2,
        cost_class=Cost.HEAVY,
        input_roles=(close,),
        output_unit=Unit.RATIO,
        evidence=Evidence.ACADEMIC_SINGLE,
        citation=Citation(formula=Reference("Roll", 1984)),
        params=FrozenParams(window=window),
    )


def spread_corwin_schultz(
    *, high: PriceRole = HIGH_SPLIT, low: PriceRole = LOW_SPLIT
) -> BoundFeature:
    """Corwin-Schultz (2012) two-day high-low spread estimator, clipped at 0. FINITE, RATIO."""
    name = "spread_corwin_schultz"

    def build(s: BarSchema) -> pl.Expr:
        h, low_ = pl.col(s.column(high)), pl.col(s.column(low))
        hl = safe_log(h / low_) ** 2
        beta = hl + hl.shift(1)
        hi2 = pl.max_horizontal(h, h.shift(1))
        lo2 = pl.min_horizontal(low_, low_.shift(1))
        gamma = safe_log(hi2 / lo2) ** 2
        alpha = (safe_sqrt(2 * beta) - safe_sqrt(beta)) / _CS_DENOM - safe_sqrt(gamma / _CS_DENOM)
        exp_alpha = alpha.exp()
        spread = 2 * (exp_alpha - 1) / (1 + exp_alpha)
        value = pl.when(spread < 0).then(pl.lit(0.0)).otherwise(spread)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=2,
        min_history=2,
        recurrence=Recurrence.FINITE,
        effective_warmup=2,
        cost_class=Cost.HEAVY,
        input_roles=(high, low),
        output_unit=Unit.RATIO,
        evidence=Evidence.ACADEMIC_SINGLE,
        citation=Citation(formula=Reference("Corwin & Schultz", 2012)),
        params=FrozenParams(),
    )


# --- extras beyond §12 (no §12 equivalent) -----------------------------------------------------


def dollar_vol(
    *, close: PriceRole = CLOSE_SPLIT, volume: VolumeRole = VOLUME_SPLIT
) -> BoundFeature:
    """Per-bar dollar volume (price times volume). FINITE. (Extra, beyond §12.)"""
    name = "dollar_vol"

    def build(s: BarSchema) -> pl.Expr:
        value = pl.col(s.column(close)) * pl.col(s.column(volume))
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=(Horizon.SHORT,),
        lookback=1,
        min_history=1,
        recurrence=Recurrence.FINITE,
        effective_warmup=1,
        cost_class=Cost.O1,
        input_roles=(close, volume),
        output_unit=Unit.PRICE_UNITS,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=Reference("Campbell, Lo & MacKinlay", 1997)),
        params=FrozenParams(),
    )


def adv(
    *, window: int = 21, close: PriceRole = CLOSE_SPLIT, volume: VolumeRole = VOLUME_SPLIT
) -> BoundFeature:
    """Average daily dollar volume over ``window`` bars: a liquidity scale. FINITE. (Extra.)"""
    int_at_least("window", window, 2)
    name = naming("adv", window)

    def build(s: BarSchema) -> pl.Expr:
        dv = pl.col(s.column(close)) * pl.col(s.column(volume))
        return grouped(dv.rolling_mean(window, min_samples=window), s.symbol_col).alias(name)

    return _finite_vol(build, name, window, (close, volume), Unit.PRICE_UNITS, _CLM)


def signed_vol(
    *, window: int = 21, close: PriceRole = CLOSE_TR, volume: VolumeRole = VOLUME_SPLIT
) -> BoundFeature:
    """Windowed signed volume (Granville's OBV, FINITE form): net up/down volume. (Extra.)"""
    int_at_least("window", window, 2)
    name = naming("signed_vol", window)

    def build(s: BarSchema) -> pl.Expr:
        signed = pl.col(s.column(close)).diff().sign() * pl.col(s.column(volume))
        return grouped(signed.rolling_sum(window, min_samples=window), s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.LINEAR,
        input_roles=(close, volume),
        output_unit=Unit.UNITLESS,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=Reference("Granville", 1963)),
        params=FrozenParams(window=window),
    )


def _finite_vol(
    build: Callable[[BarSchema], pl.Expr],
    name: str,
    window: int,
    roles: tuple[PriceRole | VolumeRole, ...],
    unit: Unit,
    formula: Reference,
) -> BoundFeature:
    return bind_feature(
        build,
        name=name,
        family=Family.VOLUME,
        native_band=_BANDS,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=roles,
        output_unit=unit,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=formula),
        params=FrozenParams(window=window),
    )


FEATURES: tuple[BoundFeature, ...] = (
    vol_z(window=21),
    rel_volume(window=21),
    amihud(window=21),
    vwap_dist_close(),
    cmf(window=21),
    mfi(window=14),
    roll_spread(window=21),
    spread_corwin_schultz(),
    dollar_vol(),
    adv(window=21),
    signed_vol(window=21),
)


__all__ = [
    "FEATURES",
    "adv",
    "amihud",
    "cmf",
    "dollar_vol",
    "mfi",
    "rel_volume",
    "roll_spread",
    "signed_vol",
    "spread_corwin_schultz",
    "vol_z",
    "vwap_dist_close",
]
