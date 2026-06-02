# Feature-set manifest (FEATURES.md 4.4): pins everything needed to reproduce a feature set -- the
# bound features AND transforms (by name/version/fingerprint), the role mapping, calendar, the
# dependency DAG (as provenance, NOT a runtime scheduler -- 5), and tool versions. Round-trips so a
# stored dataset's exact feature definitions are provable, not assumed.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from sabia.schema import BarSchema
from sabia.typing import FeatureRef

if TYPE_CHECKING:
    from sabia.normalize import BoundTransform
    from sabia.spec import BoundFeature


@dataclass(frozen=True, slots=True)
class TransformRef:
    """A manifest pointer to a bound transform by identity (FEATURES.md 4.4, 5)."""

    name: str
    version: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class FeatureSetManifest:
    """Immutable, serializable pin of a feature bundle (FEATURES.md 4.4)."""

    features: tuple[FeatureRef, ...]
    transforms: tuple[TransformRef, ...]
    role_map: tuple[tuple[str, str], ...]  # (str(role), physical_column), sorted
    closed_col: str | None
    calendar: str
    dependency_edges: tuple[tuple[str, str], ...]  # (dependent_fp, dependency_fp), sorted
    polars_version: str
    sabia_version: str

    @classmethod
    def of(
        cls,
        features: Iterable[BoundFeature],
        transforms: Iterable[BoundTransform],
        schema: BarSchema,
        *,
        sabia_version: str,
        polars_version: str = pl.__version__,
    ) -> FeatureSetManifest:
        """Assemble a manifest from bound features + transforms + the resolving schema."""
        feats = tuple(FeatureRef(f.spec.name, f.spec.version, f.spec.fingerprint) for f in features)
        trans = tuple(
            TransformRef(t.spec.name, t.spec.version, t.spec.fingerprint) for t in transforms
        )
        role_map = tuple(sorted((str(role), col) for role, col in schema.roles.items()))
        edges = tuple(
            sorted(
                (f.spec.fingerprint, dep.fingerprint)
                for f in features
                for dep in f.spec.dependencies
            )
        )
        return cls(
            features=feats,
            transforms=trans,
            role_map=role_map,
            closed_col=schema.closed_col,
            calendar=schema.calendar,
            dependency_edges=edges,
            polars_version=polars_version,
            sabia_version=sabia_version,
        )


__all__ = ["FeatureSetManifest", "TransformRef"]
