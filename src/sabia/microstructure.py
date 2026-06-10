# Intraday microstructure family (FEATURES.md 13). Computes on the intraday bars the
# ``sabia.adapters`` tick->bar layer produces (MINUTE tier), reading the adapter-derived flow
# aggregates and (where present) L1 quotes on the raw basis. Strictly trailing, panel-safe via
# .over(symbol); degenerate divides yield null, never inf. Every feature declares
# ``data_tier=DataTier.MINUTE`` -- never the bind_feature DAILY default -- so ``Registry.available``
# offers them only on intraday-or-finer input. Three groups: realized volatility (high-frequency
# econometrics), order flow (directional pressure), and liquidity / spread.

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import pi, sqrt

import polars as pl

from sabia._expr import grouped
from sabia._math import bar_return, log_return, rolling_corr, rolling_slope, safe_div, safe_sqrt
from sabia._validate_params import int_at_least, positive_int
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, DataTier, Evidence, Family, Horizon, Recurrence, Unit
from sabia.typing import (
    ASK_RAW,
    ASK_SIZE_RAW,
    BID_RAW,
    BID_SIZE_RAW,
    CLOSE_RAW,
    SIGNED_VOLUME_RAW,
    VOLUME_RAW,
    Adjustment,
    DepthRole,
    FlowRole,
    InputRole,
    PriceRole,
    QuoteField,
    QuoteRole,
    VolumeRole,
)

# Intraday bands: microstructure signals are primary at the MICRO/INTRADAY end (FEATURES.md 6).
_BANDS = (Horizon.MICRO, Horizon.INTRADAY)
# Bipower scaling 1/mu_1^2 = pi/2, mu_1 = E|Z| = sqrt(2/pi) (Barndorff-Nielsen & Shephard 2004).
_BIPOWER_SCALE = pi / 2.0
_ABDL = Reference("Andersen, Bollerslev, Diebold & Labys", 2003)


def _realized_var(r: pl.Expr, window: int) -> pl.Expr:
    # Realized variance: sum of squared intraday returns over the window (the integrated-variance
    # estimator). A null warmup return in the window keeps the sum null until the window is full.
    return (r * r).rolling_sum(window, min_samples=window)


def _bipower_var(r: pl.Expr, window: int) -> pl.Expr:
    # Bipower variation: (pi/2) * sum |r_t| |r_{t-1}| over the SAME `window` returns RV uses --
    # window - 1 adjacent products span returns t-window+1 .. t (Huang & Tauchen 2005), so RV - BV
    # compares like for like. Robust to jumps: a lone large |r| pairs with a finite neighbour.
    return _BIPOWER_SCALE * (r.abs() * r.abs().shift(1)).rolling_sum(
        window - 1, min_samples=window - 1
    )


# --- order flow --------------------------------------------------------------------------------


def trade_imbalance(
    *,
    window: int = 12,
    signed_volume: FlowRole = SIGNED_VOLUME_RAW,
    volume: VolumeRole = VOLUME_RAW,
) -> BoundFeature:
    """Net order-flow imbalance over ``window`` bars: signed volume / total volume. FINITE.

    ``sum(signed_volume) / sum(volume)`` across the trailing window, where ``signed_volume`` is the
    adapter's buy-minus-sell volume (trade-signed at aggregation time, FEATURES.md 13). The result
    lies in ``[-1, 1]``: +1 is all buyer-initiated flow, -1 all seller-initiated. Persistent
    imbalance proxies directional pressure / informed trading and predicts short-horizon returns.
    A zero-volume window yields null, never inf. Citation: Chordia & Subrahmanyam (2004).
    """
    int_at_least("window", window, 2)
    name = naming("trade_imbalance", window)

    def build(s: BarSchema) -> pl.Expr:
        signed = pl.col(s.column(signed_volume)).rolling_sum(window, min_samples=window)
        total = pl.col(s.column(volume)).rolling_sum(window, min_samples=window)
        return grouped(safe_div(signed, total), s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MICROSTRUCTURE,
        native_band=_BANDS,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        data_tier=DataTier.MINUTE,
        input_roles=(signed_volume, volume),
        output_unit=Unit.UNITLESS,
        output_range=(-1.0, 1.0),
        evidence=Evidence.ACADEMIC_REPLICATED,
        citation=Citation(formula=Reference("Chordia & Subrahmanyam", 2004)),
        params=FrozenParams(window=window),
    )


def vpin(
    *,
    n_buckets: int = 50,
    signed_volume: FlowRole = SIGNED_VOLUME_RAW,
    volume: VolumeRole = VOLUME_RAW,
) -> BoundFeature:
    """Volume-synchronized PIN (flow toxicity) over ``n_buckets`` volume buckets. FINITE, in [0, 1].

    Easley, Lopez de Prado & O'Hara (2012): the mean over the trailing ``n_buckets`` equal-volume
    buckets of the per-bucket order imbalance ``|buy - sell| / bucket_volume``
    (``= |signed_volume| / volume``). High VPIN flags toxic, one-sided flow -- a documented
    flash-crash precursor. **Feed this volume bars** (``build_bars(kind=VOLUME)``): each bar is then
    one equal-volume bucket, so the trailing mean is exactly VPIN. On time bars it degrades to a
    bar-clock approximation. A zero-volume bucket yields null. Citation: Easley, Lopez de Prado &
    O'Hara (2012).
    """
    int_at_least("n_buckets", n_buckets, 2)
    name = naming("vpin", n_buckets)

    def build(s: BarSchema) -> pl.Expr:
        oi = safe_div(pl.col(s.column(signed_volume)).abs(), pl.col(s.column(volume)))
        value = oi.rolling_mean(n_buckets, min_samples=n_buckets)
        return grouped(value, s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        n_buckets,
        (signed_volume, volume),
        Unit.UNITLESS,
        Reference("Easley, Lopez de Prado & O'Hara", 2012),
        output_range=(0.0, 1.0),
        evidence=Evidence.ACADEMIC_REPLICATED,
        params=FrozenParams(n_buckets=n_buckets),
    )


def sign_autocorr(
    *,
    lag: int = 1,
    window: int = 78,
    signed_volume: FlowRole = SIGNED_VOLUME_RAW,
    volume: VolumeRole = VOLUME_RAW,
) -> BoundFeature:
    """Autocorrelation of the per-bar order-flow imbalance at ``lag``. FINITE, UNITLESS.

    Order flow is famously long-memory: signed trades persist far longer than returns do (Lillo &
    Farmer 2004). This is the rolling Pearson autocorrelation of the per-bar imbalance
    ``signed_volume / volume`` with its ``lag``-bar-lagged self. A flat window (zero variance)
    yields null. Citation: Lillo & Farmer (2004).
    """
    positive_int("lag", lag)
    int_at_least("window", window, 2)
    name = naming("sign_autocorr", lag, window)

    def build(s: BarSchema) -> pl.Expr:
        imbalance = safe_div(pl.col(s.column(signed_volume)), pl.col(s.column(volume)))
        value = rolling_corr(imbalance, imbalance.shift(lag), window)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MICROSTRUCTURE,
        native_band=_BANDS,
        lookback=window,
        min_history=window + lag,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + lag,
        cost_class=Cost.LINEAR,
        data_tier=DataTier.MINUTE,
        input_roles=(signed_volume, volume),
        output_unit=Unit.UNITLESS,
        evidence=Evidence.ACADEMIC_REPLICATED,
        citation=Citation(formula=Reference("Lillo & Farmer", 2004)),
        params=FrozenParams(lag=lag, window=window),
    )


# --- realized volatility -----------------------------------------------------------------------


def rvar(*, window: int = 78, close: PriceRole = CLOSE_RAW) -> BoundFeature:
    """Realized variance: sum of squared intraday log returns over ``window`` bars. FINITE.

    The model-free integrated-variance estimator of Andersen, Bollerslev, Diebold & Labys (2001-03):
    as sampling gets finer, ``sum r_i^2`` converges to the integrated variance of the price process.
    The workhorse intraday volatility input (e.g. to a HAR forecast). Citation: ABDL (2003).
    """
    int_at_least("window", window, 2)
    name = naming("rvar", window)

    def build(s: BarSchema) -> pl.Expr:
        return grouped(_realized_var(bar_return(s.column(close)), window), s.symbol_col).alias(name)

    return _finite_micro(
        build, name, window + 1, (close,), Unit.UNITLESS, _ABDL, params=FrozenParams(window=window)
    )


def bipower(*, window: int = 78, close: PriceRole = CLOSE_RAW) -> BoundFeature:
    """Bipower variation: jump-robust integrated variance over ``window`` bars. FINITE.

    ``(pi/2) * sum |r_t| |r_{t-1}|`` over the ``window - 1`` adjacent pairs of the same ``window``
    returns ``rvar`` uses (Barndorff-Nielsen & Shephard 2004; windowing as in Huang & Tauchen
    2005). Multiplying adjacent absolute returns makes a lone jump contribute only finitely, so BV
    estimates the *continuous* variance and ``RV - BV`` isolates jumps. Citation:
    Barndorff-Nielsen & Shephard (2004).
    """
    int_at_least("window", window, 2)
    name = naming("bipower", window)

    def build(s: BarSchema) -> pl.Expr:
        return grouped(_bipower_var(bar_return(s.column(close)), window), s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window + 1,
        (close,),
        Unit.UNITLESS,
        Reference("Barndorff-Nielsen & Shephard", 2004),
        params=FrozenParams(window=window),
    )


def jump_rj(*, window: int = 78, close: PriceRole = CLOSE_RAW) -> BoundFeature:
    """Relative jump variation: ``max(RV - BV, 0) / RV`` over ``window`` bars. FINITE, in [0, 1].

    The share of realized variance attributable to discontinuous jumps (Huang & Tauchen 2005),
    flooring the difference at zero (sampling noise can make BV exceed RV). RV and BV are computed
    over the **same** ``window`` returns, so the comparison is like for like. A flat window
    (RV == 0) yields null. Citation: Huang & Tauchen (2005).
    """
    int_at_least("window", window, 2)
    name = naming("jump_rj", window)

    def build(s: BarSchema) -> pl.Expr:
        r = bar_return(s.column(close))
        rv = _realized_var(r, window)
        bv = _bipower_var(r, window)
        value = safe_div((rv - bv).clip(lower_bound=0.0), rv)
        return grouped(value, s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window + 1,
        (close,),
        Unit.UNITLESS,
        Reference("Huang & Tauchen", 2005),
        output_range=(0.0, 1.0),
        params=FrozenParams(window=window),
    )


def rsemivar_up(*, window: int = 78, close: PriceRole = CLOSE_RAW) -> BoundFeature:
    """Upside realized semivariance: sum of squared *positive* returns over ``window``. FINITE.

    Barndorff-Nielsen, Kinnebrock & Shephard (2010) split RV by return sign; the upside half
    captures good volatility. Citation: Barndorff-Nielsen, Kinnebrock & Shephard (2010).
    """
    return _semivar_feature(window, close, positive=True)


def rsemivar_dn(*, window: int = 78, close: PriceRole = CLOSE_RAW) -> BoundFeature:
    """Downside realized semivariance: sum of squared *negative* returns over ``window``. FINITE.

    The bad-volatility half (Barndorff-Nielsen, Kinnebrock & Shephard 2010): downside variation
    carries most of the risk premium. Citation: Barndorff-Nielsen, Kinnebrock & Shephard (2010).
    """
    return _semivar_feature(window, close, positive=False)


def signed_jump(*, window: int = 78, close: PriceRole = CLOSE_RAW) -> BoundFeature:
    """Signed jump variation: upside minus downside semivariance over ``window``. FINITE.

    ``RS+ - RS- = sum r_t |r_t|`` (Patton & Sheppard 2015): the "good minus bad" volatility
    asymmetry. Positive when up-moves dominate the variation; it predicts future volatility with a
    sign the symmetric RV cannot. Citation: Patton & Sheppard (2015).
    """
    int_at_least("window", window, 2)
    name = naming("signed_jump", window)

    def build(s: BarSchema) -> pl.Expr:
        r = bar_return(s.column(close))
        value = (r * r.abs()).rolling_sum(window, min_samples=window)
        return grouped(value, s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window + 1,
        (close,),
        Unit.UNITLESS,
        Reference("Patton & Sheppard", 2015),
        params=FrozenParams(window=window),
    )


def rskew(*, window: int = 78, close: PriceRole = CLOSE_RAW) -> BoundFeature:
    """Realized skewness over ``window`` bars: ``sqrt(N) * sum r^3 / RV^(3/2)``. FINITE, UNITLESS.

    Amaya, Christoffersen, Jacobs & Vasquez (2015): intraday realized skewness predicts the
    cross-section of next-period returns (more negative skew -> higher returns). ``N`` is the number
    of returns in the window. A flat window (RV == 0) yields null. Citation: Amaya et al. (2015).
    """
    int_at_least("window", window, 2)
    name = naming("rskew", window)
    root_n = sqrt(float(window))

    def build(s: BarSchema) -> pl.Expr:
        r = bar_return(s.column(close))
        rv = _realized_var(r, window)
        sum3 = (r * r * r).rolling_sum(window, min_samples=window)
        value = safe_div(root_n * sum3, rv * safe_sqrt(rv))
        return grouped(value, s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window + 1,
        (close,),
        Unit.UNITLESS,
        Reference("Amaya, Christoffersen, Jacobs & Vasquez", 2015),
        params=FrozenParams(window=window),
    )


def rkurt(*, window: int = 78, close: PriceRole = CLOSE_RAW) -> BoundFeature:
    """Realized kurtosis over ``window`` bars: ``N * sum r^4 / RV^2``. FINITE, UNITLESS.

    The intraday tail-heaviness companion to ``rskew`` (Amaya et al. 2015). A flat window (RV == 0)
    yields null. Citation: Amaya et al. (2015).
    """
    int_at_least("window", window, 2)
    name = naming("rkurt", window)
    n = float(window)

    def build(s: BarSchema) -> pl.Expr:
        r = bar_return(s.column(close))
        rv = _realized_var(r, window)
        sum4 = (r * r * r * r).rolling_sum(window, min_samples=window)
        value = safe_div(n * sum4, rv * rv)
        return grouped(value, s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window + 1,
        (close,),
        Unit.UNITLESS,
        Reference("Amaya, Christoffersen, Jacobs & Vasquez", 2015),
        params=FrozenParams(window=window),
    )


# --- liquidity / spread ------------------------------------------------------------------------


def quoted_spread(
    *, window: int = 78, bid: QuoteRole = BID_RAW, ask: QuoteRole = ASK_RAW
) -> BoundFeature:
    """Relative quoted spread over ``window`` bars: rolling mean of ``(ask - bid) / mid``. FINITE.

    The most direct round-trip cost of liquidity (tier L1). ``mid = (bid + ask) / 2``; a crossed
    (``bid > ask``) or zero-mid quote yields null -- ``validate`` rejects crossed quotes outright,
    and the expression guard keeps the promise on the trusted (``ValidationMode.OFF``) path too.
    Citation: Amihud & Mendelson (1986).
    """
    int_at_least("window", window, 2)
    name = naming("quoted_spread", window)

    def build(s: BarSchema) -> pl.Expr:
        b, a = pl.col(s.column(bid)), pl.col(s.column(ask))
        spread = pl.when(b > a).then(None).otherwise(safe_div(a - b, (a + b) / 2.0))
        return grouped(spread.rolling_mean(window, min_samples=window), s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window,
        (bid, ask),
        Unit.RATIO,
        Reference("Amihud & Mendelson", 1986),
        params=FrozenParams(window=window),
    )


def eff_spread(
    *,
    window: int = 78,
    close: PriceRole = CLOSE_RAW,
    bid: QuoteRole = BID_RAW,
    ask: QuoteRole = ASK_RAW,
) -> BoundFeature:
    """Effective spread over ``window`` bars: rolling mean of ``2 |close - mid| / mid``. FINITE.

    What a marketable order actually pays relative to the midpoint (tier L1): the trade price's
    distance from the prevailing mid, doubled for the round trip (Lee & Ready 1991). Citation:
    Lee & Ready (1991).
    """
    int_at_least("window", window, 2)
    name = naming("eff_spread", window)

    def build(s: BarSchema) -> pl.Expr:
        c, b, a = (pl.col(s.column(role)) for role in (close, bid, ask))
        mid = (a + b) / 2.0
        eff = safe_div(2.0 * (c - mid).abs(), mid)
        return grouped(eff.rolling_mean(window, min_samples=window), s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window,
        (close, bid, ask),
        Unit.RATIO,
        Reference("Lee & Ready", 1991),
        params=FrozenParams(window=window),
    )


def amihud_intraday(
    *, window: int = 78, close: PriceRole = CLOSE_RAW, volume: VolumeRole = VOLUME_RAW
) -> BoundFeature:
    """Intraday Amihud illiquidity: rolling mean of ``|return| / dollar_volume``. FINITE.

    Amihud (2002) price impact per unit traded, at the intraday tier (dollar volume taken as
    ``close * volume`` on the bar). A zero-volume (halted) bar yields null -- no impact to measure.
    Citation: Amihud (2002).
    """
    int_at_least("window", window, 2)
    name = naming("amihud_intraday", window)

    def build(s: BarSchema) -> pl.Expr:
        c, v = pl.col(s.column(close)), pl.col(s.column(volume))
        ratio = safe_div(log_return(c, c.shift(1)).abs(), c * v)
        return grouped(ratio.rolling_mean(window, min_samples=window), s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window + 1,
        (close, volume),
        Unit.RATIO,
        Reference("Amihud", 2002),
        params=FrozenParams(window=window),
    )


def kyle_lambda(
    *,
    window: int = 78,
    close: PriceRole = CLOSE_RAW,
    signed_volume: FlowRole = SIGNED_VOLUME_RAW,
) -> BoundFeature:
    """Kyle's lambda over ``window`` bars: price-impact slope of ``dclose`` on signed flow. FINITE.

    The depth parameter of Kyle (1985): the rolling OLS slope of the bar price change on signed
    order flow, ``cov(dp, q) / var(q)``. A larger lambda means a thinner book (more impact per unit
    of flow). Estimated as in Hasbrouck (2009). Zero flow variance yields null. Cite: Kyle (1985).
    """
    int_at_least("window", window, 2)
    name = naming("kyle_lambda", window)

    def build(s: BarSchema) -> pl.Expr:
        dp = pl.col(s.column(close)).diff()
        q = pl.col(s.column(signed_volume))
        return grouped(rolling_slope(q, dp, window), s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window + 1,
        (close, signed_volume),
        Unit.UNITLESS,
        Reference("Kyle", 1985),
        evidence=Evidence.ACADEMIC_SINGLE,
        params=FrozenParams(window=window),
    )


def depth_imbalance(
    *,
    window: int = 78,
    bid_size: QuoteRole = BID_SIZE_RAW,
    ask_size: QuoteRole = ASK_SIZE_RAW,
) -> BoundFeature:
    """Inside-book (L1) depth imbalance over ``window`` bars. FINITE, in [-1, 1].

    Rolling mean of ``(bid_size - ask_size) / (bid_size + ask_size)`` at the best quote: a standing
    size asymmetry that predicts the next move (more bid depth -> upward pressure). Tier L1. An
    empty book (zero total size) yields null. Citation: Cont, Kukanov & Stoikov (2014).
    """
    int_at_least("window", window, 2)
    name = naming("depth_imbalance", window)

    def build(s: BarSchema) -> pl.Expr:
        bs, as_ = pl.col(s.column(bid_size)), pl.col(s.column(ask_size))
        imbalance = safe_div(bs - as_, bs + as_)
        return grouped(imbalance.rolling_mean(window, min_samples=window), s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window,
        (bid_size, ask_size),
        Unit.UNITLESS,
        Reference("Cont, Kukanov & Stoikov", 2014),
        output_range=(-1.0, 1.0),
        params=FrozenParams(window=window),
    )


def book_imbalance(*, levels: int = 3) -> BoundFeature:
    """Depth-weighted order-book imbalance across the top ``levels`` (L2). FINITE, in [-1, 1].

    ``(sum bid_size - sum ask_size) / (sum bid_size + sum ask_size)`` over book levels
    ``0..levels-1`` (level 0 = inside). The deep-book generalization of ``depth_imbalance``; a
    fuller picture of standing liquidity pressure (Cao, Hansch & Wang 2009). Requires L2
    ``DepthRole`` columns, so it
    is **not in the default registry** (the bars-only / L1 tiers do not carry per-level depth) --
    construct it explicitly with a schema that maps ``bid_size_l{k}`` / ``ask_size_l{k}``. An empty
    book yields null. Citation: Cao, Hansch & Wang (2009).
    """
    int_at_least("levels", levels, 1)
    name = naming("book_imbalance", levels)
    bid_roles = tuple(DepthRole(QuoteField.BID_SIZE, lvl, Adjustment.RAW) for lvl in range(levels))
    ask_roles = tuple(DepthRole(QuoteField.ASK_SIZE, lvl, Adjustment.RAW) for lvl in range(levels))

    def build(s: BarSchema) -> pl.Expr:
        bid = sum((pl.col(s.column(r)) for r in bid_roles), pl.lit(0.0))
        ask = sum((pl.col(s.column(r)) for r in ask_roles), pl.lit(0.0))
        return grouped(safe_div(bid - ask, bid + ask), s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        1,
        (*bid_roles, *ask_roles),
        Unit.UNITLESS,
        Reference("Cao, Hansch & Wang", 2009),
        output_range=(-1.0, 1.0),
        params=FrozenParams(levels=levels),
    )


# --- shared binder + helpers -------------------------------------------------------------------


def _semivar_feature(window: int, close: PriceRole, *, positive: bool) -> BoundFeature:
    int_at_least("window", window, 2)
    name = naming("rsemivar_up" if positive else "rsemivar_dn", window)

    def build(s: BarSchema) -> pl.Expr:
        r = bar_return(s.column(close))
        mask = (r > 0) if positive else (r < 0)
        # r*r is null on the warmup bar, so the masked square (and its rolling sum) stays null until
        # the window is full; opposite-sign bars contribute a genuine 0, never a null.
        value = (r * r * mask.cast(pl.Float64)).rolling_sum(window, min_samples=window)
        return grouped(value, s.symbol_col).alias(name)

    return _finite_micro(
        build,
        name,
        window + 1,
        (close,),
        Unit.UNITLESS,
        Reference("Barndorff-Nielsen, Kinnebrock & Shephard", 2010),
        params=FrozenParams(window=window),
    )


def _finite_micro(
    build: Callable[[BarSchema], pl.Expr],
    name: str,
    min_history: int,
    roles: Iterable[InputRole],
    unit: Unit,
    formula: Reference,
    *,
    params: FrozenParams,
    output_range: tuple[float, float] | None = None,
    evidence: Evidence = Evidence.ACADEMIC_SINGLE,
) -> BoundFeature:
    # Shared binder for the FINITE, MINUTE-tier microstructure features: same family / band / tier /
    # recurrence, so each factory states only what differs (math, warmup, roles, unit, citation).
    return bind_feature(
        build,
        name=name,
        family=Family.MICROSTRUCTURE,
        native_band=_BANDS,
        lookback=min_history,
        min_history=min_history,
        recurrence=Recurrence.FINITE,
        effective_warmup=min_history,
        cost_class=Cost.LINEAR,
        data_tier=DataTier.MINUTE,
        input_roles=roles,
        output_unit=unit,
        output_range=output_range,
        evidence=evidence,
        citation=Citation(formula=formula),
        params=params,
    )


FEATURES: tuple[BoundFeature, ...] = (
    trade_imbalance(window=12),
    vpin(n_buckets=50),
    sign_autocorr(lag=1, window=78),
    rvar(window=78),
    bipower(window=78),
    jump_rj(window=78),
    rsemivar_up(window=78),
    rsemivar_dn(window=78),
    signed_jump(window=78),
    rskew(window=78),
    rkurt(window=78),
    quoted_spread(window=78),
    eff_spread(window=78),
    depth_imbalance(window=78),
    amihud_intraday(window=78),
    kyle_lambda(window=78),
)
# ``book_imbalance`` (L2) is intentionally absent: the default bars-only / L1 tiers carry no
# per-level depth columns, so it ships as a factory for callers with an L2 schema (FEATURES.md 13).


__all__ = [
    "FEATURES",
    "amihud_intraday",
    "bipower",
    "book_imbalance",
    "depth_imbalance",
    "eff_spread",
    "jump_rj",
    "kyle_lambda",
    "quoted_spread",
    "rkurt",
    "rsemivar_dn",
    "rsemivar_up",
    "rskew",
    "rvar",
    "sign_autocorr",
    "signed_jump",
    "trade_imbalance",
    "vpin",
]
