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
# canonical column names on BarSchema (FEATURES.md 2.1).
InputRole = PriceRole | VolumeRole | FactorRole | CalendarRole


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


__all__ = [
    "CLOSE_RAW",
    "CLOSE_SPLIT",
    "CLOSE_TR",
    "DVOL_RAW",
    "HIGH_SPLIT",
    "LOW_SPLIT",
    "MARKET_RET",
    "OPEN_SPLIT",
    "OPEN_TR",
    "VOLUME_RAW",
    "VOLUME_SPLIT",
    "VWAP_SPLIT",
    "Adjustment",
    "CalendarRole",
    "FactorRole",
    "FeatureRef",
    "InputRole",
    "PriceField",
    "PriceRole",
    "VolumeField",
    "VolumeRole",
]
