# BarSchema (FEATURES.md 2.2): the caller-supplied map from column roles to physical column names.
# Features declare roles; `.column(role)` resolves them at build time. sabia adjusts and infers
# nothing -- the schema records the adjustment basis each column carries, and the manifest pins it.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sabia.typing import (
    Adjustment,
    InputRole,
    PriceField,
    PriceRole,
    VolumeField,
    VolumeRole,
)


@dataclass(frozen=True, slots=True)
class BarSchema:
    """Resolves column roles to physical columns for one frame/manifest (FEATURES.md 2.2, 2.4).

    ``symbol`` and ``timestamp`` are fixed canonical column names (FEATURES.md 2.1), not
    role-tagged; the OHLCV/factor columns are role-tagged. ``calendar`` is the exchange code (one
    per frame, 2.4); ``closed_col`` is the bars-closed marker (8.3). Frozen, with ``roles`` wrapped
    in an immutable view so the guarantee holds even if the caller mutates the dict they passed.
    """

    roles: Mapping[InputRole, str]
    closed_col: str | None = None
    calendar: str = "UTC"
    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))

    @classmethod
    def ohlcv(
        cls,
        *,
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
        tr_close: str | None = None,
        symbol_col: str = "symbol",
        timestamp_col: str = "timestamp",
        closed_col: str | None = None,
        calendar: str = "UTC",
    ) -> BarSchema:
        """Build a schema from plain OHLCV column names -- the common case, no hand-rolled roles.

        Maps the OHLC columns to the **split-only** basis (the range-safe basis the estimators use,
        FEATURES.md 2.2) and ``volume`` to volume@split. ``open``/``close`` also back the ``@tr``
        return roles so close-to-close features resolve; pass ``tr_close`` when the total-return
        close is a separate column (e.g. an ``adj_close``). For richer inputs (VWAP, dollar volume,
        a market factor) construct ``BarSchema(roles={...})`` directly -- this covers OHLCV only.
        """
        close_tr_col = tr_close if tr_close is not None else close
        roles: dict[InputRole, str] = {
            PriceRole(PriceField.OPEN, Adjustment.SPLIT): open,
            PriceRole(PriceField.HIGH, Adjustment.SPLIT): high,
            PriceRole(PriceField.LOW, Adjustment.SPLIT): low,
            PriceRole(PriceField.CLOSE, Adjustment.SPLIT): close,
            VolumeRole(VolumeField.VOLUME, Adjustment.SPLIT): volume,
            PriceRole(PriceField.OPEN, Adjustment.TR): open,
            PriceRole(PriceField.CLOSE, Adjustment.TR): close_tr_col,
        }
        return cls(
            roles=roles,
            closed_col=closed_col,
            calendar=calendar,
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
        )

    def column(self, role: InputRole) -> str:
        """Physical column backing ``role``; raises a precise ``KeyError`` if undeclared."""
        try:
            return self.roles[role]
        except KeyError:
            declared = sorted(str(r) for r in self.roles)
            raise KeyError(
                f"BarSchema has no column for role {role}; declared roles: {declared}"
            ) from None

    def has(self, role: InputRole) -> bool:
        return role in self.roles


__all__ = ["BarSchema"]
