"""The feature-set manifest gate (FEATURES.md 3.4): formula identity is frozen once published."""

from __future__ import annotations

import pytest
from feature_manifest import MANIFEST
from synthetic import SCHEMA

import sabia
from sabia.manifest import MANIFEST_SCHEMA_VERSION, FeatureSetManifest


def _current() -> tuple[tuple[str, int, str], ...]:
    specs = sabia.Registry.default().specs()
    return tuple(sorted((s.name, s.version, s.fingerprint) for s in specs))


def test_registry_matches_manifest() -> None:
    current = _current()
    expected = tuple(sorted(MANIFEST))
    # A mismatch means a formula changed without a version bump, or a feature was added/removed
    # without updating the manifest. Resolve by bumping versions and updating feature_manifest.py.
    assert current == expected


def test_no_duplicate_fingerprints() -> None:
    fingerprints = [fp for (_, _, fp) in MANIFEST]
    assert len(fingerprints) == len(set(fingerprints)), "two features share a fingerprint"


def _default_manifest() -> FeatureSetManifest:
    return FeatureSetManifest.of(
        sabia.Registry.default().features(), (), SCHEMA, sabia_version=sabia.__version__
    )


def test_manifest_round_trips_through_dict() -> None:
    manifest = _default_manifest()
    assert FeatureSetManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_round_trips_through_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = _default_manifest()
    path = tmp_path / "manifest.json"
    manifest.to_json(path)
    assert FeatureSetManifest.from_json(path) == manifest


def test_manifest_from_dict_rejects_unknown_schema_version() -> None:
    payload = _default_manifest().to_dict()
    payload["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="manifest_schema_version"):
        FeatureSetManifest.from_dict(payload)


def test_manifest_from_dict_rejects_malformed_payload() -> None:
    payload = _default_manifest().to_dict()
    del payload["features"]
    with pytest.raises(ValueError, match="malformed manifest payload"):
        FeatureSetManifest.from_dict(payload)


def test_manifest_from_dict_rejects_non_int_ref_version() -> None:
    payload = _default_manifest().to_dict()
    payload["features"][0]["version"] = "1"  # type: ignore[index]
    with pytest.raises(ValueError, match="malformed manifest payload"):
        FeatureSetManifest.from_dict(payload)


def test_manifest_from_dict_rejects_non_list_field() -> None:
    payload = _default_manifest().to_dict()
    payload["role_map"] = {"close@tr": "close"}  # should be a list of pairs
    with pytest.raises(ValueError, match="malformed manifest payload"):
        FeatureSetManifest.from_dict(payload)


def test_public_api_exposes_every_family() -> None:
    families = (
        sabia.returns,
        sabia.trend,
        sabia.momentum,
        sabia.volatility,
        sabia.volume,
        sabia.distribution,
        sabia.mean_reversion,
        sabia.seasonality,
        sabia.cross_sectional,
    )
    assert all(hasattr(family, "FEATURES") for family in families)
