"""
sabia -- Polars-native technical features for trading pipelines.

Pure factories over OHLCV bars. A factory binds its params and returns a ``BoundFeature`` -- an
immutable ``.spec`` plus ``.expr(schema) -> pl.Expr`` that resolves column *roles* (``close@tr``,
``high@split``) against a caller-supplied ``BarSchema``. Strictly trailing, point-in-time correct,
batch-first / online-ready. See ``FEATURES.md`` for the full specification.

    import polars as pl
    import sabia
    from sabia import BarSchema, PriceRole, PriceField, Adjustment

    schema = BarSchema(roles={PriceRole(PriceField.CLOSE, Adjustment.TR): "close"})
    df = sabia.compute(frame, sabia.momentum.rsi(period=14), schema=schema)

    reg = sabia.Registry.default()
    reg.where(lambda s: sabia.Horizon.MEDIUM in s.native_band)
    reg.available(sabia.DataTier.DAILY)
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from sabia import (
    cross_sectional,
    distribution,
    mean_reversion,
    momentum,
    normalize,
    returns,
    seasonality,
    trend,
    volatility,
    volume,
)
from sabia.calendar import SessionCalendar, UtcCalendar, get_calendar
from sabia.manifest import FeatureSetManifest, TransformRef
from sabia.naming import assert_unique, naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import (
    BoundFeature,
    FrozenRegistryError,
    Registry,
    bind_feature,
    evaluate,
)
from sabia.schema import BarSchema
from sabia.spec import (
    Cost,
    DataTier,
    Evidence,
    Family,
    FeatureSpec,
    Horizon,
    NullPolicy,
    Recurrence,
    Unit,
    ValidationMode,
)
from sabia.typing import (
    Adjustment,
    CalendarRole,
    FactorRole,
    FeatureRef,
    InputRole,
    PriceField,
    PriceRole,
    VolumeField,
    VolumeRole,
)
from sabia.validate import SabiaValidationError, validate

__version__ = "0.2.0"


def compute(
    frame: pl.DataFrame | pl.LazyFrame,
    bound_feature: BoundFeature,
    *more: BoundFeature,
    schema: BarSchema,
    validation: ValidationMode = ValidationMode.STRICT,
    universe: Sequence[str] | None = None,
    membership_asof: object = None,
) -> pl.DataFrame:
    """Materialize bound features, resolving their roles against ``schema`` (FEATURES.md 4.2).

    Time-series features fuse into a single ``select``; cross-sectional features (which carry a
    per-symbol ``signal``) are evaluated two-pass each. A single ``compute`` rejects two same-named
    expressions (FEATURES.md 4.3). ``validation`` runs the input contract first (STRICT by default).
    ``universe`` / ``membership_asof`` declare the cross-sectional universe (FEATURES.md 2.5); pass
    them when any feature ``requires_universe`` so completeness is checked against the declared
    universe rather than inferred from the symbols that happen to be present.
    """
    feats = (bound_feature, *more)
    assert_unique(f.spec.name for f in feats)
    if validation is not ValidationMode.OFF:
        required: frozenset[InputRole] = frozenset().union(*(f.spec.input_roles for f in feats))
        if universe is None and any(f.spec.requires_universe for f in feats):
            raise ValueError(
                "a cross-sectional feature requires_universe=True; pass universe=... to compute() "
                "(sabia asserts completeness against the declared universe, never infers it)"
            )
        validate(
            frame,
            schema=schema,
            required_roles=required,
            complete_panel=any(f.spec.requires_complete_panel for f in feats),
            universe=universe,
            membership_asof=membership_asof,
            mode=validation,
        )
    base = frame.lazy()
    ts_feats = [f for f in feats if f.signal is None]
    ts_frame = (
        base.select(*(f.expr(schema).alias(f.spec.name) for f in ts_feats)).collect()
        if ts_feats
        else None
    )
    columns: list[pl.Series] = []
    for f in feats:
        if f.signal is None:
            assert ts_frame is not None  # ts_feats is non-empty whenever a TS feature exists
            columns.append(ts_frame.get_column(f.spec.name))
        else:
            columns.append(evaluate(base, f, schema))
    return pl.DataFrame(columns)


def compute_lazy(
    frame: pl.DataFrame | pl.LazyFrame,
    *features: BoundFeature,
    schema: BarSchema,
) -> pl.LazyFrame:
    """Lazy ``select`` of time-series features (the fused path; used by the eager-vs-lazy gate).

    Cross-sectional features are inherently two-pass; pass them to ``compute`` instead.
    """
    if any(f.signal is not None for f in features):
        raise ValueError("compute_lazy supports time-series features only; use compute() for XS")
    assert_unique(f.spec.name for f in features)
    return frame.lazy().select(*(f.expr(schema) for f in features))


__all__ = [
    "Adjustment",
    "BarSchema",
    "BoundFeature",
    "CalendarRole",
    "Citation",
    "Cost",
    "DataTier",
    "Evidence",
    "FactorRole",
    "Family",
    "FeatureRef",
    "FeatureSetManifest",
    "FeatureSpec",
    "FrozenParams",
    "FrozenRegistryError",
    "Horizon",
    "InputRole",
    "NullPolicy",
    "PriceField",
    "PriceRole",
    "Recurrence",
    "Reference",
    "Registry",
    "SabiaValidationError",
    "SessionCalendar",
    "TransformRef",
    "Unit",
    "UtcCalendar",
    "ValidationMode",
    "VolumeField",
    "VolumeRole",
    "__version__",
    "assert_unique",
    "bind_feature",
    "compute",
    "compute_lazy",
    "cross_sectional",
    "distribution",
    "evaluate",
    "get_calendar",
    "mean_reversion",
    "momentum",
    "naming",
    "normalize",
    "returns",
    "seasonality",
    "trend",
    "validate",
    "volatility",
    "volume",
]
