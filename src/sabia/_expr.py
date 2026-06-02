# Shared expression plumbing: per-symbol grouping and the emit-and-buffer mask. A trailing feature
# evaluates its window within each symbol so panels never bleed across instruments. Centralized here
# so every family applies them identically (and so the fingerprint covers them once, transitively).

from __future__ import annotations

import polars as pl


def grouped(expr: pl.Expr, symbol: str | None) -> pl.Expr:
    """Evaluate ``expr`` within each ``symbol`` group, or as-is when ``symbol`` is None."""
    return expr.over(symbol) if symbol is not None else expr


def emit_after(value: pl.Expr, min_history: int, symbol: str | None) -> pl.Expr:
    """Null the first ``min_history - 1`` rows per symbol; emit ``value`` after (FEATURES.md 4.5).

    For RECURSIVE_DECAY features, ``min_history == effective_warmup``: the recursive value has not
    converged before then, so the spec requires emitting null until the buffer is full. Masking by a
    per-symbol row index enforces that and makes windowed-recompute parity hold (the last row of a
    ``min_history``-bar window is at index ``min_history - 1`` -> never masked -> emits).
    """
    index = pl.int_range(pl.len())
    if symbol is not None:
        index = index.over(symbol)
    return pl.when(index < min_history - 1).then(None).otherwise(value)


__all__ = ["emit_after", "grouped"]
