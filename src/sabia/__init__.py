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
    adapters,
    cross_sectional,
    distribution,
    mean_reversion,
    microstructure,
    momentum,
    normalize,
    recipes,
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
)
from sabia.registry import (
    evaluate as _evaluate,
)  # internal two-pass helper; not a public export (L5)
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
from sabia.toolkit import (
    FeatureSet,
    describe,
    drop_warmup,
    max_min_history,
    required_columns,
    required_roles,
)
from sabia.typing import (
    Adjustment,
    CalendarRole,
    DepthRole,
    FactorRole,
    FeatureRef,
    FlowField,
    FlowRole,
    InputRole,
    PriceField,
    PriceRole,
    QuoteField,
    QuoteRole,
    VolumeField,
    VolumeRole,
)
from sabia.validate import (
    FrameAudit,
    SabiaValidationError,
    audit_frame,
    validate,
    validate_ticks,
)

# Kept in sync with pyproject.toml by tests/test_invariants.py::test_version_matches_pyproject.
__version__ = "0.4.0"


def compute(
    frame: pl.DataFrame | pl.LazyFrame,
    bound_feature: BoundFeature,
    *more: BoundFeature,
    schema: BarSchema,
    validation: ValidationMode = ValidationMode.STRICT,
    universe: Sequence[str] | None = None,
    membership: pl.DataFrame | pl.LazyFrame | None = None,
    include_keys: bool = False,
) -> pl.DataFrame:
    """Materialize bound features, resolving their roles against ``schema`` (FEATURES.md 4.2).

    Time-series features fuse into a single ``select``; cross-sectional features (which carry a
    per-symbol ``signal``) are evaluated two-pass each. A single ``compute`` rejects two same-named
    expressions (FEATURES.md 4.3). ``validation`` runs the input contract first (STRICT by default).
    ``universe`` (a static symbol set) / ``membership`` (an as-of ``(symbol, start, end)`` frame)
    declare the cross-sectional universe (FEATURES.md 2.5); pass one when any feature
    ``requires_universe`` so completeness is checked against the declared universe rather than
    inferred from the symbols that happen to be present.

    The result carries only the feature columns, aligned row-for-row with ``frame``. Set
    ``include_keys=True`` to prepend the identity columns (``schema.symbol_col`` when the frame is a
    panel, and ``schema.timestamp_col``) -- what a downstream pipeline usually wants.

    Cost note (FEATURES.md 10): all time-series features fuse into a single ``select`` -- one
    ``collect()``. Each cross-sectional feature, however, is materialized by its own eager
    ``evaluate`` because Polars cannot nest ``.over(symbol)`` inside ``.over(timestamp)`` in one
    expression, so the collect count grows linearly with the number of XS features (an N+1 pattern:
    one fused TS collect plus one per XS feature). Batch many XS features per ``compute`` only when
    the materialization cost is acceptable.
    """
    feats = (bound_feature, *more)
    assert_unique(f.spec.name for f in feats)
    if validation is not ValidationMode.OFF:
        required: frozenset[InputRole] = frozenset().union(*(f.spec.input_roles for f in feats))
        if universe is None and membership is None and any(f.spec.requires_universe for f in feats):
            raise ValueError(
                "a cross-sectional feature requires_universe=True; pass universe=... or "
                "membership=... to compute() (sabia asserts completeness against the declared "
                "universe, never infers it)"
            )
        validate(
            frame,
            schema=schema,
            required_roles=required,
            complete_panel=any(f.spec.requires_complete_panel for f in feats),
            universe=universe,
            membership=membership,
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
    if include_keys:
        columns.extend(_key_columns(base, schema))
    for f in feats:
        if f.signal is None:
            assert ts_frame is not None  # ts_feats is non-empty whenever a TS feature exists
            columns.append(ts_frame.get_column(f.spec.name))
        else:
            columns.append(_evaluate(base, f, schema))
    return pl.DataFrame(columns)


def _key_columns(base: pl.LazyFrame, schema: BarSchema) -> list[pl.Series]:
    """The identity columns to prepend under ``include_keys`` -- symbol (if a panel) + timestamp.

    Read from the input frame in its original row order, which the feature ``select``/``evaluate``
    paths both preserve, so the keys align row-for-row with the feature columns.
    """
    names = base.collect_schema().names()
    key_cols = [c for c in (schema.symbol_col, schema.timestamp_col) if c in names]
    return base.select(key_cols).collect().get_columns()


def compute_lazy(
    frame: pl.DataFrame | pl.LazyFrame,
    *features: BoundFeature,
    schema: BarSchema,
) -> pl.LazyFrame:
    """Lazy ``select`` of time-series features (the fused path; used by the eager-vs-lazy gate).

    Unlike ``compute``, this path does **not** validate the input contract -- it stays lazy and
    materializes nothing. Call ``sabia.validate(frame, schema=...)`` yourself first if the frame is
    untrusted, or use ``compute`` for the validated path. Cross-sectional features are inherently
    two-pass; pass them to ``compute`` instead.
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
    "DepthRole",
    "Evidence",
    "FactorRole",
    "Family",
    "FeatureRef",
    "FeatureSet",
    "FeatureSetManifest",
    "FeatureSpec",
    "FlowField",
    "FlowRole",
    "FrameAudit",
    "FrozenParams",
    "FrozenRegistryError",
    "Horizon",
    "InputRole",
    "NullPolicy",
    "PriceField",
    "PriceRole",
    "QuoteField",
    "QuoteRole",
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
    "adapters",
    "assert_unique",
    "audit_frame",
    "bind_feature",
    "compute",
    "compute_lazy",
    "cross_sectional",
    "describe",
    "distribution",
    "drop_warmup",
    "get_calendar",
    "max_min_history",
    "mean_reversion",
    "microstructure",
    "momentum",
    "naming",
    "normalize",
    "recipes",
    "required_columns",
    "required_roles",
    "returns",
    "seasonality",
    "trend",
    "validate",
    "validate_ticks",
    "volatility",
    "volume",
]
