# Ergonomic toolkit (audit follow-up): small, opinion-free helpers over BoundFeature collections --
# a FeatureSet container, a human-readable describe(), role/column introspection, and warmup
# trimming. No DAG, no scheduler, no trading opinions: these only compose the existing
# compute/manifest surface, so the library stays "TA-Lib for pipelines", not a framework.

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from sabia.registry import BoundFeature
from sabia.schema import BarSchema
from sabia.spec import ValidationMode
from sabia.typing import InputRole

if TYPE_CHECKING:
    from sabia.manifest import FeatureSetManifest
    from sabia.normalize import BoundTransform


def required_roles(features: Iterable[BoundFeature]) -> frozenset[InputRole]:
    """Union of every feature's input roles -- the roles a schema must resolve before computing."""
    return frozenset().union(*(f.spec.input_roles for f in features), frozenset())


def required_columns(features: Iterable[BoundFeature], schema: BarSchema) -> dict[str, str]:
    """Physical columns the features need, as ``{str(role): column}`` -- a pre-flight column set.

    Raises the same precise ``KeyError`` as ``schema.column`` if a required role is undeclared.
    """
    return {str(role): schema.column(role) for role in sorted(required_roles(features), key=str)}


def max_min_history(features: Iterable[BoundFeature]) -> int:
    """Largest ``min_history`` across the features -- the warmup before all of them have emitted."""
    return max((f.spec.min_history for f in features), default=0)


def drop_warmup(
    frame: pl.DataFrame, features: Iterable[BoundFeature], *, symbol_col: str | None = None
) -> pl.DataFrame:
    """Drop the leading warmup rows where some feature is still null.

    Removes the first ``max_min_history(features)`` rows -- a safe upper bound on the warmup, so no
    null-warmup row survives (it may drop one already-valid FINITE row). With ``symbol_col`` (a
    panel) the warmup is trimmed per symbol; otherwise globally. Row order is preserved.
    """
    n = max_min_history(features)
    if n <= 0:
        return frame
    if symbol_col is not None and symbol_col in frame.columns:
        return frame.filter(pl.int_range(pl.len()).over(symbol_col) >= n)
    return frame.slice(n)


def describe(feature: BoundFeature) -> str:
    """A readable one-feature card for notebooks / debugging, rendered from the feature's spec."""
    s = feature.spec
    roles = ", ".join(sorted(str(r) for r in s.input_roles)) or "(none)"
    lines = [
        s.name,
        f"family: {s.family.value}",
        f"roles: {roles}",
        f"min_history: {s.min_history}",
        f"effective_warmup: {s.effective_warmup}",
        f"recurrence: {s.recurrence.value}",
        f"unit: {s.output_unit.value}",
    ]
    if s.output_range is not None:
        lines.append(f"range: [{s.output_range[0]}, {s.output_range[1]}]")
    lines.append(f"citation: {s.citation}")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """A lightweight, immutable bundle of bound features -- a name for a fixed list, nothing more.

    Replaces threading 50 features through a variadic ``compute``: ``FeatureSet([...]).compute(df,
    schema=...)``. It delegates to the package-level ``compute`` and ``FeatureSetManifest.of``; it
    adds no scheduling, no DAG, no signal semantics.
    """

    features: tuple[BoundFeature, ...]

    def __init__(self, features: Iterable[BoundFeature]) -> None:
        object.__setattr__(self, "features", tuple(features))

    def compute(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        *,
        schema: BarSchema,
        include_keys: bool = False,
        validation: ValidationMode = ValidationMode.STRICT,
        universe: Sequence[str] | None = None,
        membership: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        """Materialize the set's features (delegates to ``sabia.compute``)."""
        if not self.features:
            raise ValueError("FeatureSet is empty; add at least one feature before computing")
        import sabia

        return sabia.compute(
            frame,
            *self.features,
            schema=schema,
            validation=validation,
            universe=universe,
            membership=membership,
            include_keys=include_keys,
        )

    def manifest(
        self, schema: BarSchema, *, transforms: Iterable[BoundTransform] = ()
    ) -> FeatureSetManifest:
        """Pin the set as a ``FeatureSetManifest`` (delegates to ``FeatureSetManifest.of``)."""
        import sabia
        from sabia.manifest import FeatureSetManifest

        return FeatureSetManifest.of(
            self.features, transforms, schema, sabia_version=sabia.__version__
        )

    def names(self) -> list[str]:
        return [f.spec.name for f in self.features]

    def required_roles(self) -> frozenset[InputRole]:
        return required_roles(self.features)

    def __iter__(self) -> Iterator[BoundFeature]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)


__all__ = [
    "FeatureSet",
    "describe",
    "drop_warmup",
    "max_min_history",
    "required_columns",
    "required_roles",
]
