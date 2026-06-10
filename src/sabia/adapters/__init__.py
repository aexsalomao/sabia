"""Edge adapters (FEATURES.md 13): pure transforms that turn raw, un-canonical inputs into the
canonical frames sabia core consumes. Side effects (I/O) belong to the caller, never here.

``build_bars`` aggregates raw trade/quote ticks into intraday bars (time / volume / dollar / tick),
tagging trade direction via ``sign_ticks``. The output resolves against ``BarSchema.trades()`` /
``BarSchema.quotes()`` and feeds the ``microstructure`` family.
"""

from __future__ import annotations

from sabia.adapters.bars import BarKind, BarSpec, SignRule, build_bars
from sabia.adapters.signing import sign_ticks

__all__ = ["BarKind", "BarSpec", "SignRule", "build_bars", "sign_ticks"]
