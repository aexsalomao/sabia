# Feature-set manifest (FEATURES.md 4.4): pins everything needed to reproduce a feature set -- the
# bound features AND transforms (by name/version/fingerprint), the role mapping, calendar, the
# dependency DAG (as provenance, NOT a runtime scheduler -- 5), and tool versions. Round-trips so a
# stored dataset's exact feature definitions are provable, not assumed.

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from sabia.schema import BarSchema
from sabia.typing import FeatureRef

if TYPE_CHECKING:
    from sabia.normalize import BoundTransform
    from sabia.spec import BoundFeature

# Bumped when the serialized dict shape changes; ``from_dict`` rejects an unknown version so an old
# reader never silently misreads a newer manifest (FEATURES.md 4.4).
MANIFEST_SCHEMA_VERSION = 1


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

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready dict of the manifest -- the serialization the 4.4 round-trip rests on.

        Tagged with ``manifest_schema_version`` so a future shape change is detectable. Feature and
        transform refs become ``{name, version, fingerprint}`` dicts; the sorted pair tuples
        (``role_map``, ``dependency_edges``) become lists of two-element lists.
        """
        return {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "features": [_ref_to_dict(f) for f in self.features],
            "transforms": [_ref_to_dict(t) for t in self.transforms],
            "role_map": [list(pair) for pair in self.role_map],
            "closed_col": self.closed_col,
            "calendar": self.calendar,
            "dependency_edges": [list(pair) for pair in self.dependency_edges],
            "polars_version": self.polars_version,
            "sabia_version": self.sabia_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FeatureSetManifest:
        """Rebuild a manifest from ``to_dict`` output; raise on a malformed / unknown payload."""
        version = data.get("manifest_schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest_schema_version {version!r}; "
                f"this sabia reads version {MANIFEST_SCHEMA_VERSION}"
            )
        try:
            return cls(
                features=tuple(FeatureRef(*_ref_args(d)) for d in _as_list(data, "features")),
                transforms=tuple(TransformRef(*_ref_args(d)) for d in _as_list(data, "transforms")),
                role_map=tuple((a, b) for a, b in _as_list(data, "role_map")),
                closed_col=data["closed_col"],  # type: ignore[arg-type]
                calendar=data["calendar"],  # type: ignore[arg-type]
                dependency_edges=tuple((a, b) for a, b in _as_list(data, "dependency_edges")),
                polars_version=data["polars_version"],  # type: ignore[arg-type]
                sabia_version=data["sabia_version"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed manifest payload: {exc}") from exc

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write the manifest as JSON to ``path`` (an edge artifact -- I/O belongs here, 4.4)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> FeatureSetManifest:
        """Read a manifest previously written by ``to_json``."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _ref_to_dict(ref: FeatureRef | TransformRef) -> dict[str, object]:
    return {"name": ref.name, "version": ref.version, "fingerprint": ref.fingerprint}


def _ref_args(d: dict[str, object]) -> tuple[str, int, str]:
    # (name, version, fingerprint) checked against their declared types so a hand-edited payload
    # fails loudly here (wrapped into a ValueError by from_dict) rather than deep in a comparison.
    version = d["version"]
    if not isinstance(version, int):
        raise TypeError(f"ref 'version' must be int, got {type(version).__name__}")
    return str(d["name"]), version, str(d["fingerprint"])


def _as_list(data: dict[str, object], key: str) -> list:  # type: ignore[type-arg]
    value = data[key]
    if not isinstance(value, list):
        raise TypeError(f"manifest field {key!r} must be a list, got {type(value).__name__}")
    return value


__all__ = ["MANIFEST_SCHEMA_VERSION", "FeatureSetManifest", "TransformRef"]
