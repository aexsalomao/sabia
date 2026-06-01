# Shared expression guards enforcing the no-inf/no-NaN policy (FEATURES.md 3.5): degenerate inputs
# yield null, never inf or NaN. Used across feature families wherever a divide, log, or sqrt could
# otherwise escape into downstream math.

from __future__ import annotations

import polars as pl


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """``numerator / denominator``, or null where the denominator is zero."""
    return pl.when(denominator == 0).then(None).otherwise(numerator / denominator)


def safe_log(expr: pl.Expr) -> pl.Expr:
    """Natural log, or null where the argument is non-positive (log-domain breach)."""
    return pl.when(expr <= 0).then(None).otherwise(expr.log())


def safe_sqrt(expr: pl.Expr) -> pl.Expr:
    """Square root, or null where the argument is negative."""
    return pl.when(expr < 0).then(None).otherwise(expr.sqrt())


__all__ = ["safe_div", "safe_log", "safe_sqrt"]
