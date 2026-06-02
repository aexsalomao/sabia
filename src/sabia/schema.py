# BarSchema (FEATURES.md 2.2): the caller-supplied map from column roles to physical column names.
# Features declare roles; `.column(role)` resolves them at build time. sabia adjusts and infers
# nothing -- the schema records the adjustment basis each column carries, and the manifest pins it.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sabia.typing import InputRole


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
