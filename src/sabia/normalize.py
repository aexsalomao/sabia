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
_FFD_WEIGHT_THRESHOLD = 1e-5
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


def xs_zscore(*, over: str = _TIMESTAMP_COL) -> BoundTransform:
    """Cross-sectional z-score within each ``over`` slice (a single timestamp by default).

    A degenerate slice (zero dispersion) yields ``null``, never ``inf``.
    """

    def apply(expr: pl.Expr) -> pl.Expr:
        mean = expr.mean().over(over)
        std = expr.std().over(over)
        standardized = (expr - mean) / std
        return pl.when(std == 0).then(None).otherwise(standardized)

    return _bound(
        "xs_zscore",
        apply,
        params=FrozenParams(),
        lookback=None,
        min_history=1,
        output_unit=Unit.ZSCORE,
    )


def xs_rank(*, over: str = _TIMESTAMP_COL) -> BoundTransform:
    """Cross-sectional percentile rank in (0, 1] within each ``over`` slice.

    Ties take the average rank. Computed per timestamp slice, so it never pools across time.
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
    """
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
