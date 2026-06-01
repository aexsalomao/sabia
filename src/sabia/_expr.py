# Shared expression plumbing: conditional per-symbol grouping. A trailing feature evaluates its
# window within each symbol so panels never bleed across instruments; a bare single series carries
# no symbol column, so passing symbol=None evaluates the expression ungrouped. Centralized here so
# every family applies grouping identically (and so the fingerprint covers it once, transitively).

from __future__ import annotations

import polars as pl


def grouped(expr: pl.Expr, symbol: str | None) -> pl.Expr:
    """Evaluate ``expr`` within each ``symbol`` group, or as-is when ``symbol`` is None."""
    return expr.over(symbol) if symbol is not None else expr


__all__ = ["grouped"]
