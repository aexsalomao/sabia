# Naming grammar (FEATURES.md 4.3): `{measure}_{params...}`, params in declared order. Rule A -- the
# name encodes the *non-default* role/adjustment: `rsi_14` is close@tr, `rsi_raw_14` is close@raw.
# Multi-output groups carry a `suffix` (`macd_signal_12_26_9`, `bb_pctb_20_2`); `_ann` variants a
# `trailing` token. A single name is the contract id, so the snake_case pattern is enforced here.

from __future__ import annotations

import re
from collections.abc import Iterable

from sabia.spec import NAME_PATTERN
from sabia.typing import Adjustment, InputRole, PriceRole, VolumeRole

_NAME_RE = re.compile(NAME_PATTERN)


def naming(
    measure: str,
    *params: int | str,
    role: InputRole | None = None,
    default_adjustment: Adjustment | None = None,
    suffix: str | None = None,
    trailing: str | None = None,
) -> str:
    """Compose a feature's canonical name (FEATURES.md 4.3).

    Order: ``measure`` [``suffix``] [adjustment token, Rule A] ``params...`` [``trailing``]. The
    adjustment token is emitted only when ``role``'s adjustment deviates from ``default_adjustment``
    (so the common case yields the bare ``rsi_14`` and a ``@raw`` rebinding yields ``rsi_raw_14``).
    Raises if the result is not snake_case -- the generated id must be a valid contract identifier.
    """
    parts: list[str] = [measure]
    if suffix is not None:
        parts.append(suffix)
    if (
        isinstance(role, PriceRole | VolumeRole)
        and default_adjustment is not None
        and role.adjustment is not default_adjustment
    ):
        parts.append(role.adjustment.value)
    parts.extend(str(p) for p in params)
    if trailing is not None:
        parts.append(trailing)
    name = "_".join(parts)
    if not _NAME_RE.match(name):
        raise ValueError(f"generated name {name!r} violates {NAME_PATTERN}")
    return name


def assert_unique(names: Iterable[str]) -> None:
    """Raise on the first repeated name -- the collision guard for a registry / one ``compute``."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate feature name {name!r}")
        seen.add(name)


__all__ = ["assert_unique", "naming"]
