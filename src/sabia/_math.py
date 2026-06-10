# Shared expression guards enforcing the no-inf/no-NaN policy (FEATURES.md 3.5): degenerate inputs
# yield null, never inf or NaN. Used across feature families wherever a divide, log, or sqrt could
# otherwise escape into downstream math. Also home to the cross-family rolling-moment helpers, so
# every family computes the same covariance/correlation/slope from one definition.

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


def log_return(current: pl.Expr, base: pl.Expr) -> pl.Expr:
    """``ln(current / base)``, or null on a degenerate ratio -- never inf or NaN.

    Both guards are required: ``safe_div`` nulls a zero base (which raw ``/`` would turn into inf),
    and ``safe_log`` nulls a non-positive ratio (which raw ``.log()`` would turn into NaN). The
    single source of truth for every close-to-close return in the library (FEATURES.md 4.5).
    """
    return safe_log(safe_div(current, base))


def bar_return(column: str) -> pl.Expr:
    """One-bar log return of the physical ``column`` -- the building block of return moments.

    The shared close-to-close form every family uses (``ln(c_t / c_{t-1})``, null on the seed bar
    and on degenerate ratios via ``log_return``); call as ``bar_return(schema.column(close))``.
    """
    c = pl.col(column)
    return log_return(c, c.shift(1))


def rolling_cov(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Population covariance of two aligned series over the trailing window: ``E[xy] - E[x]E[y]``.

    With ``x is y`` this is the variance. ``min_samples=window`` emits null until the window is
    full. The single rolling-moment primitive ``rolling_corr`` / ``rolling_slope`` compose.
    """
    mean_xy = (x * y).rolling_mean(window, min_samples=window)
    mean_x = x.rolling_mean(window, min_samples=window)
    mean_y = y.rolling_mean(window, min_samples=window)
    return mean_xy - mean_x * mean_y


def rolling_corr(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Pearson correlation over the window, from population moments: cov / (std_x * std_y).

    A flat window (zero variance in either series) yields null, never inf or NaN.
    """
    return safe_div(
        rolling_cov(x, y, window), safe_sqrt(rolling_cov(x, x, window) * rolling_cov(y, y, window))
    )


def rolling_slope(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """OLS slope of ``y`` on ``x`` over the window, from population moments: cov(x, y) / var(x).

    A flat ``x`` window (zero variance) yields null, never inf.
    """
    return safe_div(rolling_cov(x, y, window), rolling_cov(x, x, window))


__all__ = [
    "bar_return",
    "log_return",
    "rolling_corr",
    "rolling_cov",
    "rolling_slope",
    "safe_div",
    "safe_log",
    "safe_sqrt",
]
