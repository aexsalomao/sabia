# Column roles (FEATURES.md 2.2): the adjustment-tagged input identities features declare and a
# BarSchema resolves to physical columns. A role is `field@adjustment` -- `close@tr` and
# `high@split` are distinct roles, so sabia never lets one `*_adj` column do two jobs. Roles are
# frozen and hashable (so they sit in `frozenset[InputRole]` and fold into the fingerprint).

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Adjustment(StrEnum):
    """The adjustment basis a column carries; part of role identity (FEATURES.md 2.2)."""

    TR = "tr"  # total return: close-to-close incl. dividends + splits
    SPLIT = "split"  # split-only: preserves historical ranges for range estimators
    RAW = "raw"  # unadjusted: actual traded prices / notional


class PriceField(StrEnum):
    """The OHLC(+VWAP) field a ``PriceRole`` refers to."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VWAP = "vwap"


class VolumeField(StrEnum):
    """The volume field a ``VolumeRole`` refers to."""

    VOLUME = "volume"
    DOLLAR_VOLUME = "dollar_volume"


class QuoteField(StrEnum):
    """An L1 quote field a ``QuoteRole`` (or ``DepthRole`` side) refers to (FEATURES.md 13).

    BID/ASK/MID are price-like (float); BID_SIZE/ASK_SIZE are size-like (numeric).
    """

    BID = "bid"
    ASK = "ask"
    MID = "mid"
    BID_SIZE = "bid_size"
    ASK_SIZE = "ask_size"


class FlowField(StrEnum):
    """A trade-flow aggregate a ``FlowRole`` refers to -- derived by the adapter at bar-build time.

    Signing (tick rule / Lee-Ready) happens in ``sabia.adapters``; a feature reads the precomputed
    aggregate (``signed_volume@raw`` ...) and never re-derives a trade sign, so it stays pure.
    """

    SIGNED_VOLUME = "signed_volume"
    BUY_VOLUME = "buy_volume"
    SELL_VOLUME = "sell_volume"
    SIGNED_DOLLAR = "signed_dollar"
    TRADE_COUNT = "trade_count"


# Quote fields that carry a price (float) vs a size (numeric); drives role dtype validation.
_PRICE_QUOTE_FIELDS: frozenset[QuoteField] = frozenset(
    {QuoteField.BID, QuoteField.ASK, QuoteField.MID}
)
# Book sides a ``DepthRole`` may carry (MID is not a per-level book side).
_DEPTH_SIDES: frozenset[QuoteField] = frozenset(
    {QuoteField.BID, QuoteField.ASK, QuoteField.BID_SIZE, QuoteField.ASK_SIZE}
)


@dataclass(frozen=True, slots=True)
class PriceRole:
    """A price input identity: ``(field, adjustment)``, rendered ``field@adjustment``.

    Frozen + slots over two hashable enums, so the synthesized hash is stable and order-independent
    inside a ``frozenset`` -- which is what lets ``input_roles`` fold deterministically into the
    fingerprint (FEATURES.md 4.4).
    """

    field: PriceField
    adjustment: Adjustment

    def __str__(self) -> str:
        return f"{self.field.value}@{self.adjustment.value}"


@dataclass(frozen=True, slots=True)
class VolumeRole:
    """A volume input identity: ``(field, adjustment)``, rendered ``field@adjustment``."""

    field: VolumeField
    adjustment: Adjustment

    def __str__(self) -> str:
        return f"{self.field.value}@{self.adjustment.value}"


@dataclass(frozen=True, slots=True)
class QuoteRole:
    """An L1 quote input identity: ``(field, adjustment)`` (FEATURES.md 13, tier L1).

    Same frozen/hashable shape as ``PriceRole`` so it folds deterministically into the fingerprint.
    ``is_price`` distinguishes a price field (BID/ASK/MID -> float) from a size field (numeric).
    """

    field: QuoteField
    adjustment: Adjustment

    @property
    def is_price(self) -> bool:
        return self.field in _PRICE_QUOTE_FIELDS

    def __str__(self) -> str:
        return f"{self.field.value}@{self.adjustment.value}"


@dataclass(frozen=True, slots=True)
class FlowRole:
    """A trade-flow aggregate identity ``(field, adjustment)``; adapter-derived (FEATURES.md 13)."""

    field: FlowField
    adjustment: Adjustment

    def __str__(self) -> str:
        return f"{self.field.value}@{self.adjustment.value}"


@dataclass(frozen=True, slots=True)
class DepthRole:
    """An L2 book input identity: a quote ``side`` at book ``level`` (0 = inside; FEATURES.md 13).

    Rendered ``side_l{level}@adjustment`` (``bid_l1@raw``) -- the level is in the identity, so each
    book level is a distinct role in the fingerprint. ``side`` is restricted to the bid/ask
    price/size sides (not MID); ``level`` is non-negative.
    """

    side: QuoteField
    level: int
    adjustment: Adjustment

    def __post_init__(self) -> None:
        if self.side not in _DEPTH_SIDES:
            raise ValueError(
                f"DepthRole side must be one of {sorted(s.value for s in _DEPTH_SIDES)}"
            )
        if self.level < 0:
            raise ValueError(f"DepthRole level must be non-negative, got {self.level}")

    @property
    def is_price(self) -> bool:
        return self.side in _PRICE_QUOTE_FIELDS

    def __str__(self) -> str:
        return f"{self.side.value}_l{self.level}@{self.adjustment.value}"


class FactorRole(StrEnum):
    """A cross-asset factor series (FEATURES.md 2.2). Carries no adjustment axis; the caller is
    responsible for session-aligning factor observations (FEATURES.md 2.3)."""

    MARKET_RET = "market_ret"

    def __str__(self) -> str:
        return self.value


class CalendarRole(StrEnum):
    """A calendar/session input (FEATURES.md 2.2), resolved against ``BarSchema.calendar``."""

    SESSION = "session"

    def __str__(self) -> str:
        return self.value


# The tagged union every feature declares. `symbol` / `timestamp` are NOT roles -- they are fixed
# canonical column names on BarSchema (FEATURES.md 2.1). Quote/Flow/Depth roles are the intraday
# microstructure tier (FEATURES.md 13); Flow aggregates are adapter-derived.
InputRole = PriceRole | VolumeRole | QuoteRole | FlowRole | DepthRole | FactorRole | CalendarRole


@dataclass(frozen=True, slots=True)
class FeatureRef:
    """A manifest pointer to a bound feature by identity, not the object (FEATURES.md 4.4)."""

    name: str
    version: int
    fingerprint: str


# --- named role constants (factory default args; keep family signatures terse) -----------------

CLOSE_TR = PriceRole(PriceField.CLOSE, Adjustment.TR)
CLOSE_RAW = PriceRole(PriceField.CLOSE, Adjustment.RAW)
OPEN_TR = PriceRole(PriceField.OPEN, Adjustment.TR)
OPEN_SPLIT = PriceRole(PriceField.OPEN, Adjustment.SPLIT)
HIGH_SPLIT = PriceRole(PriceField.HIGH, Adjustment.SPLIT)
LOW_SPLIT = PriceRole(PriceField.LOW, Adjustment.SPLIT)
CLOSE_SPLIT = PriceRole(PriceField.CLOSE, Adjustment.SPLIT)
VWAP_SPLIT = PriceRole(PriceField.VWAP, Adjustment.SPLIT)
VOLUME_SPLIT = VolumeRole(VolumeField.VOLUME, Adjustment.SPLIT)
VOLUME_RAW = VolumeRole(VolumeField.VOLUME, Adjustment.RAW)
DVOL_RAW = VolumeRole(VolumeField.DOLLAR_VOLUME, Adjustment.RAW)
MARKET_RET = FactorRole.MARKET_RET

# Intraday microstructure roles (FEATURES.md 13). Quotes/flow live on the raw (traded) basis -- one
# session has no split/dividend boundary, and the adapter emits raw aggregates.
BID_RAW = QuoteRole(QuoteField.BID, Adjustment.RAW)
ASK_RAW = QuoteRole(QuoteField.ASK, Adjustment.RAW)
MID_RAW = QuoteRole(QuoteField.MID, Adjustment.RAW)
BID_SIZE_RAW = QuoteRole(QuoteField.BID_SIZE, Adjustment.RAW)
ASK_SIZE_RAW = QuoteRole(QuoteField.ASK_SIZE, Adjustment.RAW)
SIGNED_VOLUME_RAW = FlowRole(FlowField.SIGNED_VOLUME, Adjustment.RAW)
BUY_VOLUME_RAW = FlowRole(FlowField.BUY_VOLUME, Adjustment.RAW)
SELL_VOLUME_RAW = FlowRole(FlowField.SELL_VOLUME, Adjustment.RAW)
SIGNED_DOLLAR_RAW = FlowRole(FlowField.SIGNED_DOLLAR, Adjustment.RAW)
TRADE_COUNT_RAW = FlowRole(FlowField.TRADE_COUNT, Adjustment.RAW)


__all__ = [
    "ASK_RAW",
    "ASK_SIZE_RAW",
    "BID_RAW",
    "BID_SIZE_RAW",
    "BUY_VOLUME_RAW",
    "CLOSE_RAW",
    "CLOSE_SPLIT",
    "CLOSE_TR",
    "DVOL_RAW",
    "HIGH_SPLIT",
    "LOW_SPLIT",
    "MARKET_RET",
    "MID_RAW",
    "OPEN_SPLIT",
    "OPEN_TR",
    "SELL_VOLUME_RAW",
    "SIGNED_DOLLAR_RAW",
    "SIGNED_VOLUME_RAW",
    "TRADE_COUNT_RAW",
    "VOLUME_RAW",
    "VOLUME_SPLIT",
    "VWAP_SPLIT",
    "Adjustment",
    "CalendarRole",
    "DepthRole",
    "FactorRole",
    "FeatureRef",
    "FlowField",
    "FlowRole",
    "InputRole",
    "PriceField",
    "PriceRole",
    "QuoteField",
    "QuoteRole",
    "VolumeField",
    "VolumeRole",
]
