"""Small, dependency-light numpy stat helpers for the notebook charts.

These keep the notebook cells focused on *plotting* rather than re-deriving textbook statistics.
They are deliberately plain numpy (the notebooks already convert Polars columns to arrays).
"""

from __future__ import annotations

import numpy as np

__all__ = ["acf", "adf_tstat", "fit_line", "normal_pdf"]


def acf(x: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Sample autocorrelation at lags ``0..nlags`` (lag 0 is 1.0 by construction).

    NaNs are dropped first. Returns an array of length ``nlags + 1``.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.zeros(nlags + 1)
    out = np.empty(nlags + 1)
    for lag in range(nlags + 1):
        out[lag] = float(np.dot(x[: len(x) - lag], x[lag:])) / denom
    return out


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian density at ``x`` — for overlaying a reference normal on a return histogram."""
    x = np.asarray(x, dtype=np.float64)
    density: np.ndarray = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    return density


def adf_tstat(x: np.ndarray) -> float:
    """Dickey-Fuller t-statistic (no augmentation, with constant) for a unit root.

    Regresses ``Δy_t`` on a constant and ``y_{t-1}`` and returns the t-stat of the lagged-level
    coefficient. More negative => stronger evidence the series is stationary; the 5% critical value
    is about ``-2.86``. A pure random walk hovers near 0. NaNs are dropped first.
    """
    y = np.asarray(x, dtype=np.float64)
    y = y[~np.isnan(y)]
    dy = np.diff(y)
    design = np.column_stack([np.ones(dy.shape[0]), y[:-1]])
    beta, *_ = np.linalg.lstsq(design, dy, rcond=None)
    resid = dy - design @ beta
    dof = dy.shape[0] - design.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(design.T @ design)
    return float(beta[1] / np.sqrt(cov[1, 1]))


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary-least-squares ``(slope, intercept)`` over paired points, ignoring NaNs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)
