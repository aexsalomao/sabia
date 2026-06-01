"""The feature-set manifest gate (FEATURES.md 3.4): formula identity is frozen once published."""

from __future__ import annotations

from feature_manifest import MANIFEST

import sabia


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
