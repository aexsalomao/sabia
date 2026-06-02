"""07 — Manifests & reproducibility: pin a feature set so train == serve.

Every bound feature carries a content fingerprint over its canonical id, version, params, roles,
dependency fingerprints, and the pinned Polars version. A FeatureSetManifest snapshots the whole
bundle — features, transforms, the role map, calendar, and tool versions — so a stored dataset's
exact feature definitions are provable, not assumed. Rebuild the same factories tomorrow and the
fingerprints match; change a formula without bumping its version and they won't (CI catches it).

Run:  python examples/07_manifest.py
"""

from __future__ import annotations

from _data import default_schema

import sabia
from sabia import FeatureSetManifest


def main() -> None:
    schema = default_schema()

    features = [
        sabia.momentum.rsi(period=14),
        sabia.volatility.bb_pctb(window=20, n_std=2.0),
        sabia.cross_sectional.beta(window=252),
    ]
    transforms = [sabia.normalize.zscore(window=63)]

    manifest = FeatureSetManifest.of(features, transforms, schema, sabia_version=sabia.__version__)

    print("--- feature set manifest ---")
    for ref in manifest.features:
        print(f"  feature   {ref.name:<16} v{ref.version}  {ref.fingerprint}")
    for ref in manifest.transforms:
        print(f"  transform {ref.name:<16} v{ref.version}  {ref.fingerprint}")
    print(f"  calendar       : {manifest.calendar}")
    print(f"  polars_version : {manifest.polars_version}")
    print(f"  sabia_version  : {manifest.sabia_version}")
    print(f"  role_map rows  : {len(manifest.role_map)}")

    # Determinism: rebuilding the same factories yields byte-identical fingerprints.
    again = FeatureSetManifest.of(
        [sabia.momentum.rsi(period=14)], [], schema, sabia_version=sabia.__version__
    )
    rsi_now = again.features[0].fingerprint
    rsi_before = next(r.fingerprint for r in manifest.features if r.name == "rsi_14")
    print(f"\nrsi_14 fingerprint stable across rebuilds: {rsi_now == rsi_before}")

    # Different params -> different identity (rsi_14 vs rsi_21 are distinct contracts).
    rsi21 = sabia.momentum.rsi(period=21)
    print(f"rsi_14 vs rsi_21 fingerprints differ:      {rsi21.spec.fingerprint != rsi_before}")


if __name__ == "__main__":
    main()
