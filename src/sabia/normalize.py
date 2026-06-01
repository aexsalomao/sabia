# Normalization layer (FEATURES.md 4): composable trailing / cross-sectional transforms with the
# same causality invariant as features. These operate on a pl.Expr and return a pl.Expr; they are
# building blocks, not registered features. Cross-sectional transforms compute within a single
# timestamp slice (never pooling across time); time-series transforms are strictly trailing.

from __future__ import annotations

import functools
import operator

import polars as pl

from sabia.spec import Column

# Fractional-differentiation weights below this magnitude are dropped (fixed-width window, FFD).
_FFD_WEIGHT_THRESHOLD = 1e-5
_FFD_MAX_LAG = 100


def zscore(expr: pl.Expr, window: int, *, over: str | None = None) -> pl.Expr:
    """Trailing rolling z-score over ``window`` bars.

    Standardizes ``expr`` against its own rolling mean and (sample, ddof=1) std. A flat window
    (std == 0) yields ``null`` rather than ``inf`` (FEATURES.md 3.5). Pass ``over`` to compute the
    rolling statistics within each group (e.g. per symbol on a panel).
    """
    mean = expr.rolling_mean(window, min_samples=window)
    std = expr.rolling_std(window, min_samples=window)
    if over is not None:
        mean = mean.over(over)
        std = std.over(over)
    standardized = (expr - mean) / std
    return pl.when(std == 0).then(None).otherwise(standardized)


def xs_zscore(expr: pl.Expr, *, over: str = Column.TIMESTAMP) -> pl.Expr:
    """Cross-sectional z-score within each ``over`` slice (a single timestamp by default).

    A degenerate slice (zero dispersion) yields ``null``, never ``inf``.
    """
    mean = expr.mean().over(over)
    std = expr.std().over(over)
    standardized = (expr - mean) / std
    return pl.when(std == 0).then(None).otherwise(standardized)


def xs_rank(expr: pl.Expr, *, over: str = Column.TIMESTAMP) -> pl.Expr:
    """Cross-sectional percentile rank in (0, 1] within each ``over`` slice.

    Ties take the average rank. Computed per timestamp slice, so it never pools across time.
    """
    return expr.rank(method="average").over(over) / expr.count().over(over)


def frac_diff(
    expr: pl.Expr,
    d: float,
    *,
    threshold: float = _FFD_WEIGHT_THRESHOLD,
    max_lag: int = _FFD_MAX_LAG,
    over: str | None = None,
) -> pl.Expr:
    """Fixed-width fractional differentiation (Lopez de Prado): stationarity with memory.

    Applies the binomial FFD weights ``w_k`` (truncated where ``|w_k| < threshold``) to trailing
    lags of ``expr``. ``d == 0`` returns ``expr`` unchanged; ``d == 1`` reduces to the first
    difference. Pass ``over`` to lag within each group (per symbol on a panel).
    """
    weights = _ffd_weights(d, threshold=threshold, max_lag=max_lag)
    terms = []
    for lag, weight in enumerate(weights):
        if lag == 0:
            lagged = expr  # the current bar; shift(0) is a no-op and never crosses a group
        elif over is None:
            lagged = expr.shift(lag)
        else:
            lagged = expr.shift(lag).over(over)  # lag within each group on a panel
        terms.append(lagged * weight)
    return functools.reduce(operator.add, terms)


def _ffd_weights(d: float, *, threshold: float, max_lag: int) -> list[float]:
    """Binomial fractional-difference weights, truncated at ``threshold`` or ``max_lag``."""
    weights = [1.0]
    for k in range(1, max_lag + 1):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
    return weights


__all__ = ["frac_diff", "xs_rank", "xs_zscore", "zscore"]
