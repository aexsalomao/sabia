# Normalization layer (FEATURES.md 5): composable trailing / cross-sectional transforms with the
# same causality invariant as features. A transform factory returns a ``BoundTransform`` -- a pinned
# ``TransformSpec`` plus ``apply(expr) -> expr`` -- so the manifest pins transforms too.
# Cross-sectional transforms compute within a single timestamp slice (never pooling across time);
# time-series transforms are strictly trailing. The caller resolves a feature to an expr and pipes
# it through ``apply``: ``bt.apply(bf.expr(schema))``.

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from sabia._validate_params import int_at_least, positive, positive_int
from sabia.params import FrozenParams
from sabia.spec import (
    REQUIRE_FULL_WINDOW,
    NullPolicy,
    Unit,
    transform_fingerprint,
)
from sabia.typing import FeatureRef

# Default canonical timestamp column for cross-sectional slicing. `timestamp` is a fixed canonical
# name (FEATURES.md 2.1), not role-tagged; callers with a different name pass `over=`.
_TIMESTAMP_COL = "timestamp"

# Fractional-differentiation weights below this magnitude are dropped (fixed-width window, FFD).
# 1e-6 (tighter than the looser 1e-5) keeps more of the slow-decaying tail for small ``d``, where
# the binomial weights shrink very gradually -- preserving the long memory FFD exists to retain --
# while ``_FFD_MAX_LAG`` still bounds the window. Truncating too early would discard real
# low-frequency content; this is the dropped-weight contribution to the FFD approximation error.
_FFD_WEIGHT_THRESHOLD = 1e-6
_FFD_MAX_LAG = 100

_DEFAULT_DTYPE: pl.DataType = pl.Float64()


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Immutable metadata for a normalization transform (FEATURES.md 5), pinned by the manifest."""

    name: str
    version: int
    fingerprint: str
    lookback: int | None
    min_history: int
    null_policy: NullPolicy
    input_unit: Unit | None  # None = unit-agnostic (zscore accepts anything)
    output_unit: Unit
    in_dtype: pl.DataType
    out_dtype: pl.DataType
    causal: bool
    dependencies: tuple[FeatureRef, ...] = ()


@dataclass(frozen=True, slots=True)
class BoundTransform:
    """A bound transform: an immutable spec + ``apply(expr) -> expr`` (FEATURES.md 5)."""

    spec: TransformSpec
    apply: Callable[[pl.Expr], pl.Expr]


def zscore(window: int, *, over: str | None = None) -> BoundTransform:
    """Trailing rolling z-score over ``window`` bars.

    Standardizes against the rolling mean and (sample, ddof=1) std. A flat window (std == 0) yields
    ``null`` rather than ``inf`` (FEATURES.md 4.5). Pass ``over`` to compute the rolling statistics
    within each group (e.g. per symbol on a panel).
    """
    int_at_least("window", window, 2)

    def apply(expr: pl.Expr) -> pl.Expr:
        mean = expr.rolling_mean(window, min_samples=window)
        std = expr.rolling_std(window, min_samples=window)
        if over is not None:
            mean = mean.over(over)
            std = std.over(over)
        standardized = (expr - mean) / std
        return pl.when(std == 0).then(None).otherwise(standardized)

    return _bound(
        f"zscore_{window}",
        apply,
        params=FrozenParams(window=window),
        lookback=window,
        min_history=window,
        output_unit=Unit.ZSCORE,
    )


def xs_zscore(*, winsorize: float | None = None, over: str = _TIMESTAMP_COL) -> BoundTransform:
    """Cross-sectional z-score within each ``over`` slice (a single timestamp by default).

    A degenerate slice (zero dispersion) yields ``null``, never ``inf``. Pass ``winsorize=k`` to
    clip each slice symmetrically to mean +/- ``k``*std (std measured within the slice) BEFORE
    standardizing, capping the influence of outliers on the standardizing moments (FEATURES.md 4.6:
    "xs_zscore optionally winsorizes before standardizing"). ``winsorize`` folds into the spec
    params and fingerprint; the default ``None`` (no clipping) preserves the prior behavior.
    """
    if winsorize is not None:
        positive("winsorize", winsorize)

    def apply(expr: pl.Expr) -> pl.Expr:
        if winsorize is not None:
            slice_mean = expr.mean().over(over)
            slice_std = expr.std().over(over)
            expr = expr.clip(slice_mean - winsorize * slice_std, slice_mean + winsorize * slice_std)
        mean = expr.mean().over(over)
        std = expr.std().over(over)
        standardized = (expr - mean) / std
        return pl.when(std == 0).then(None).otherwise(standardized)

    return _bound(
        "xs_zscore",
        apply,
        params=FrozenParams(winsorize=winsorize),
        lookback=None,
        min_history=1,
        output_unit=Unit.ZSCORE,
    )


def xs_rank(*, over: str = _TIMESTAMP_COL) -> BoundTransform:
    """Cross-sectional percentile rank in (0, 1] within each ``over`` slice.

    ``rank(average)/count`` spans ``(0, 1]``: the minimum is ``1/n`` (never exactly ``0`` -- the
    smallest valid name still ranks 1) and the maximum is exactly ``1.0``. Ties take the average
    rank. Computed per timestamp slice, so it never pools across time.
    """

    def apply(expr: pl.Expr) -> pl.Expr:
        return expr.rank(method="average").over(over) / expr.count().over(over)

    return _bound(
        "xs_rank",
        apply,
        params=FrozenParams(),
        lookback=None,
        min_history=1,
        output_unit=Unit.RANK_0_1,
    )


def frac_diff(
    d: float,
    *,
    threshold: float = _FFD_WEIGHT_THRESHOLD,
    max_lag: int = _FFD_MAX_LAG,
    over: str | None = None,
) -> BoundTransform:
    """Fixed-width fractional differentiation (Lopez de Prado): stationarity with memory.

    Applies the binomial FFD weights ``w_k`` (truncated where ``|w_k| < threshold``) to trailing
    lags of the input. ``d == 0`` returns the input unchanged; ``d == 1`` reduces to a first
    difference. Pass ``over`` to lag within each group (per symbol on a panel).

    The weight count ``len(weights)`` (capped at ``max_lag``) IS this transform's effective warmup
    / long-memory buffer: it is surfaced as both ``lookback`` and ``min_history`` on the
    ``TransformSpec``, so the manifest carries the long-memory horizon without a separate
    ``effective_warmup`` field (FEATURES.md 8.2). Smaller ``d`` (or a tighter ``threshold``) keeps
    more weights and so a longer warmup.
    """
    positive("threshold", threshold)
    positive_int("max_lag", max_lag)
    weights = _ffd_weights(d, threshold=threshold, max_lag=max_lag)

    def apply(expr: pl.Expr) -> pl.Expr:
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

    name = f"frac_diff_{str(d).replace('.', 'p').replace('-', 'm')}"
    return _bound(
        name,
        apply,
        params=FrozenParams(d=d),
        lookback=len(weights),
        min_history=len(weights),
        output_unit=Unit.UNITLESS,
    )


def _bound(
    name: str,
    apply: Callable[[pl.Expr], pl.Expr],
    *,
    params: FrozenParams,
    lookback: int | None,
    min_history: int,
    output_unit: Unit,
    version: int = 1,
) -> BoundTransform:
    spec = TransformSpec(
        name=name,
        version=version,
        fingerprint=transform_fingerprint(
            canonical_id=name, version=version, params=params, apply=apply
        ),
        lookback=lookback,
        min_history=min_history,
        null_policy=REQUIRE_FULL_WINDOW,
        input_unit=None,
        output_unit=output_unit,
        in_dtype=_DEFAULT_DTYPE,
        out_dtype=_DEFAULT_DTYPE,
        causal=True,
    )
    return BoundTransform(spec=spec, apply=apply)


def _ffd_weights(d: float, *, threshold: float, max_lag: int) -> list[float]:
    """Binomial fractional-difference weights, truncated at ``threshold`` or ``max_lag``."""
    weights = [1.0]
    for k in range(1, max_lag + 1):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
    return weights


__all__ = [
    "BoundTransform",
    "TransformSpec",
    "frac_diff",
    "xs_rank",
    "xs_zscore",
    "zscore",
]
