# Tick->bar aggregation (FEATURES.md 13): the EDGE layer that turns raw trade/quote ticks into the
# canonical intraday bar frame sabia features consume. Pure frame->frame transform (no I/O, no
# clocks) -- referentially transparent, so the same causality/determinism invariants as core hold.
#
# Information-driven bars (volume / dollar / tick bars; Lopez de Prado, Advances in Financial ML
# ch. 2) sample on activity rather than the clock, giving returns closer to IID than time bars. Each
# bar's observation timestamp is the LAST tick time within it (FEATURES.md 2.3), so downstream
# trailing features inherit point-in-time correctness for free.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import polars as pl

from sabia._math import safe_div
from sabia.adapters.signing import SIGN_COLUMN, sign_ticks
from sabia.schema import DEFAULT_CLOSED_COLUMN
from sabia.spec import ValidationMode
from sabia.typing import FlowField, PriceField, VolumeField
from sabia.validate import SabiaValidationError, validate_ticks

# Internal scratch columns (dunder-prefixed; dropped before the canonical frame is returned).
_BUCKET = "__sabia_bucket__"
_WEIGHT = "__sabia_weight__"
_PRIOR = "__sabia_prior__"
_PV = "__sabia_pv__"
_OBS_TS = "__sabia_obs_ts__"

# Canonical output columns, spelled once: derived from the role-field enums (typing.py), which are
# also what BarSchema.trades() defaults resolve -- the adapter's output vocabulary and the schema's
# role targets cannot drift apart. DEFAULT_CLOSED_COLUMN is the bars-closed marker (FEATURES.md
# 8.3) that BarSchema.trades()/.quotes() map by default.
_OPEN = PriceField.OPEN.value
_HIGH = PriceField.HIGH.value
_LOW = PriceField.LOW.value
_CLOSE = PriceField.CLOSE.value
_VWAP = PriceField.VWAP.value
_VOLUME = VolumeField.VOLUME.value
_TRADE_COUNT = FlowField.TRADE_COUNT.value
_SIGNED_VOLUME = FlowField.SIGNED_VOLUME.value
_BUY_VOLUME = FlowField.BUY_VOLUME.value
_SELL_VOLUME = FlowField.SELL_VOLUME.value
_SIGNED_DOLLAR = FlowField.SIGNED_DOLLAR.value
_OUTPUT_VALUE_COLUMNS = (
    _OPEN,
    _HIGH,
    _LOW,
    _CLOSE,
    _VOLUME,
    _VWAP,
    _TRADE_COUNT,
    _SIGNED_VOLUME,
    _BUY_VOLUME,
    _SELL_VOLUME,
    _SIGNED_DOLLAR,
)


class BarKind(Enum):
    """How a bar boundary is drawn -- by the clock or by accumulated activity (FEATURES.md 13)."""

    TIME = "time"  # fixed wall-clock interval (e.g. "1m", "5m")
    VOLUME = "volume"  # fixed share volume per bar
    DOLLAR = "dollar"  # fixed notional (price*size) per bar
    TICK = "tick"  # fixed number of trades per bar


class SignRule(Enum):
    """Trade-direction rule applied before aggregation (FEATURES.md 13)."""

    NONE = "none"  # no signing; signed aggregates are zero
    TICK_RULE = "tick_rule"  # Lee-Ready tick rule (trades only)
    LEE_READY = "lee_ready"  # trade vs quote midpoint, tick-rule fallback (needs L1 bid/ask)


@dataclass(frozen=True, slots=True)
class BarSpec:
    """How to aggregate raw ticks into bars: the bar kind, its one size knob, and the sign rule.

    Frozen and hashable so it can be pinned in a manifest alongside the features it feeds. Exactly
    one size knob is set per kind: ``interval`` for TIME, ``threshold`` for VOLUME/DOLLAR,
    ``n_ticks`` for TICK. ``sign_rule`` selects trade signing; LEE_READY needs bid/ask columns.

    LEE_READY quote-timing contract: ``bid_col``/``ask_col`` on a trade row must carry the quote
    **prevailing** at trade time (the pre-trade L1, e.g. as-of-joined backward from a quote feed).
    A feed that stamps the *post-trade* (already-updated) quote on trade rows systematically
    misclassifies aggressor direction -- a buy that lifts the ask looks below the new midpoint.
    """

    kind: BarKind
    interval: str | None = None
    threshold: float | None = None
    n_ticks: int | None = None
    sign_rule: SignRule = SignRule.NONE
    price_col: str = "price"
    size_col: str = "size"
    bid_col: str = "bid"
    ask_col: str = "ask"
    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"

    def __post_init__(self) -> None:
        # One size knob per kind (the branches are mutually exclusive by ``kind``).
        if self.kind is BarKind.TIME and self.interval is None:
            raise ValueError("BarKind.TIME requires interval (e.g. '1m')")
        weighted = self.kind in (BarKind.VOLUME, BarKind.DOLLAR)
        if weighted and (self.threshold is None or self.threshold <= 0):
            raise ValueError(f"{self.kind} requires threshold > 0, got {self.threshold}")
        if self.kind is BarKind.TICK and (self.n_ticks is None or self.n_ticks <= 0):
            raise ValueError(f"BarKind.TICK requires n_ticks >= 1, got {self.n_ticks}")


def build_bars(
    ticks: pl.DataFrame | pl.LazyFrame,
    spec: BarSpec,
    *,
    validate: ValidationMode = ValidationMode.OFF,
) -> pl.LazyFrame:
    """Aggregate raw ticks into the canonical intraday bar frame (lazy; FEATURES.md 13).

    ``ticks`` carries (at least) ``spec.symbol_col``, ``spec.timestamp_col`` (tz-aware UTC),
    ``spec.price_col``, ``spec.size_col`` (and bid/ask for LEE_READY). The result has
    ``symbol``/``timestamp`` plus OHLCV, vwap, trade_count, the signed-flow aggregates, and a
    boolean ``closed`` marker -- the columns ``BarSchema.trades()`` resolves. Each bar's timestamp
    is the last tick time within it. The transform is pure and causal: sorting then accumulating
    forward never looks ahead.

    Tied timestamps (legal per the tick contract) are handled deterministically: the sort is
    stable, so simultaneous ticks keep their input (tape) order, and bucket membership is
    tie-atomic -- ticks sharing a (symbol, timestamp) always land in the same bar, so bar
    timestamps are strictly increasing per symbol and bucket composition is independent of the
    relative order of tied ticks (only intra-bar open/close reflect tape order).

    The symbol's final bar is emitted with ``closed = False``: it is still in progress (a later
    tick may extend it), so its values are not point-in-time stable. ``validate``'s finalization
    gate rejects open bars -- filter on ``closed`` before computing features.

    ``validate`` runs the raw-tick contract (``validate_ticks``) first -- STRICT raises on
    non-monotonic timestamps, non-positive price, negative size, or a crossed book. Defaults to OFF
    (the caller is trusted / validates upstream); pass STRICT for untrusted feeds.
    """
    if validate is not ValidationMode.OFF:
        validate_ticks(ticks, spec, mode=validate)
    lf = ticks.lazy()
    if spec.symbol_col not in lf.collect_schema().names():
        # Fail loudly at the boundary (FEATURES.md 8.3): aggregation groups by symbol, so a raw
        # Polars ColumnNotFoundError deep in the lazy plan would otherwise be the failure surface.
        raise SabiaValidationError(
            f"build_bars requires a symbol column ({spec.symbol_col!r}); a single-instrument "
            "feed must still carry a constant symbol"
        )
    lf = lf.sort(spec.symbol_col, spec.timestamp_col, maintain_order=True)
    signed = sign_ticks(lf, spec)
    if spec.kind is BarKind.TIME:
        agg = (
            signed.group_by_dynamic(
                spec.timestamp_col,
                every=_require(spec.interval),
                closed="left",
                group_by=spec.symbol_col,
            )
            .agg(_agg_exprs(spec))
            .drop(spec.timestamp_col)  # drop the window-start; _OBS_TS is the observation time
        )
        return _finalize(agg, spec)
    keyed = _assign_bucket(signed, spec)
    agg = keyed.group_by([spec.symbol_col, _BUCKET], maintain_order=True).agg(_agg_exprs(spec))
    return _finalize(agg, spec)


def _require(value: str | None) -> str:
    # __post_init__ guarantees interval is set for TIME bars; this narrows the type for mypy.
    assert value is not None
    return value


def _assign_bucket(signed: pl.LazyFrame, spec: BarSpec) -> pl.LazyFrame:
    # Map each tick to its bar index. VOLUME/DOLLAR accumulate a weight (size, or size*price) and
    # cut a new bar each time the running total -- exclusive of the current tick -- crosses a
    # threshold multiple, so the boundary-crossing tick completes the bar it lands in (causal). TICK
    # bars cut every n_ticks trades. Done in steps to avoid nesting .over inside .over.
    if spec.kind is BarKind.TICK:
        n = spec.n_ticks
        assert n is not None
        idx = pl.int_range(pl.len(), dtype=pl.Int64).over(spec.symbol_col)
        return _atomic_ties(signed.with_columns((idx // n).alias(_BUCKET)), spec)

    threshold = spec.threshold
    assert threshold is not None
    weight = (
        pl.col(spec.size_col)
        if spec.kind is BarKind.VOLUME
        else pl.col(spec.size_col) * pl.col(spec.price_col)
    )
    keyed = signed.with_columns(weight.alias(_WEIGHT))
    keyed = keyed.with_columns(
        (pl.col(_WEIGHT).cum_sum().over(spec.symbol_col) - pl.col(_WEIGHT)).alias(_PRIOR)
    )
    keyed = keyed.with_columns((pl.col(_PRIOR) / threshold).floor().cast(pl.Int64).alias(_BUCKET))
    return _atomic_ties(keyed, spec)


def _atomic_ties(keyed: pl.LazyFrame, spec: BarSpec) -> pl.LazyFrame:
    # Ticks sharing a (symbol, timestamp) are atomic: all join the bucket of the first tied tick
    # (buckets are non-decreasing in time, so first == min). A bucket boundary therefore only falls
    # between DISTINCT timestamps, which guarantees strictly increasing bar timestamps per symbol
    # (the bar contract) and makes bucket membership independent of tied ticks' relative order.
    return keyed.with_columns(
        pl.col(_BUCKET).first().over(spec.symbol_col, spec.timestamp_col).alias(_BUCKET)
    )


def _agg_exprs(spec: BarSpec) -> list[pl.Expr]:
    # The per-bar reduction. Input is sorted by (symbol, timestamp) and grouped maintain_order, so
    # first()/last() are the bar's open/close. Empty buy/sell filters sum to 0, never null.
    price, size, sign = pl.col(spec.price_col), pl.col(spec.size_col), pl.col(SIGN_COLUMN)
    return [
        price.first().alias(_OPEN),
        price.max().alias(_HIGH),
        price.min().alias(_LOW),
        price.last().alias(_CLOSE),
        size.sum().alias(_VOLUME),
        (price * size).sum().alias(_PV),
        pl.len().cast(pl.Int64).alias(_TRADE_COUNT),
        (sign * size).sum().alias(_SIGNED_VOLUME),
        size.filter(sign > 0).sum().alias(_BUY_VOLUME),
        size.filter(sign < 0).sum().alias(_SELL_VOLUME),
        (sign * size * price).sum().alias(_SIGNED_DOLLAR),
        pl.col(spec.timestamp_col).last().alias(_OBS_TS),
    ]


def _finalize(agg: pl.LazyFrame, spec: BarSpec) -> pl.LazyFrame:
    # Derive vwap (null on a zero-volume bar, never inf), set the observation timestamp as the
    # canonical timestamp, mark completed bars, project to the canonical columns, and sort. A bar
    # is provably complete only once a strictly later tick exists for its symbol (a future tick may
    # legally share the last bar's timestamp and, tie-atomically, join it), so the symbol's last
    # bar -- and only that bar -- is open (closed = False).
    ts = pl.col(spec.timestamp_col)
    return (
        agg.with_columns(safe_div(pl.col(_PV), pl.col(_VOLUME)).alias(_VWAP))
        .rename({_OBS_TS: spec.timestamp_col})
        .with_columns((ts < ts.max().over(spec.symbol_col)).alias(DEFAULT_CLOSED_COLUMN))
        .select(spec.symbol_col, spec.timestamp_col, *_OUTPUT_VALUE_COLUMNS, DEFAULT_CLOSED_COLUMN)
        .sort(spec.symbol_col, spec.timestamp_col)
    )


__all__ = ["BarKind", "BarSpec", "SignRule", "build_bars"]
