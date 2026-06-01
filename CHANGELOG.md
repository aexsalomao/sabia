# Changelog

All notable changes to `sabia` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Features now accept `symbol=None` to evaluate ungrouped on a bare single series (no `symbol`
  column), resolving the gap where `validate` accepted symbol-less frames but features then raised
  on `.over(symbol)`. Per-symbol grouping is centralized in `sabia._expr.grouped`.
- `feature_fingerprint` now hashes the transitive closure of first-party helpers a feature calls
  (e.g. `_rs_term`, `safe_div`, the cross-sectional reduction builder), not just the entry-point
  function. Previously a change to a shared helper — or to a cross-sectional feature's reduction —
  would not have tripped the manifest gate. Docstrings are also stripped before hashing, so a
  citation/docstring edit no longer forces a spurious version bump.
- All feature fingerprints regenerated in `tests/feature_manifest.py` to reflect the above (no
  formula math changed; this is a pre-consumption v0.1.x reset of the hashes).

### Fixed
- `normalize.zscore` now passes `min_samples=window` explicitly, matching every feature module and
  no longer relying on the implicit Polars default for the no-partial-window guarantee.

## [0.1.0] - 2026-06-01

### Added
- Initial scaffold: package layout, tooling (ruff, mypy strict, pytest), CI, docs.
- `spec.py`: `FeatureSpec` and the `Family` / `Horizon` / `DataTier` / `Recurrence` / `Cost` /
  `Column` enums.
- `validate.py`: the input contract (`sabia.validate`).
- `registry.py`: constructable feature registry.
- `normalize.py`: trailing/cross-sectional normalization transforms.
- Feature families (41 features): returns, volatility, momentum, trend, volume, distribution,
  mean_reversion, seasonality, cross_sectional. (`microstructure` is enum-only; the module ships
  later.)
- Public API: `sabia.compute` (eager materialization of `pl.Expr` features) and `sabia.evaluate`
  (two-pass evaluation for cross-sectional features).
- Cross-cutting invariant harness (causality, windowed-recompute parity, warmup, null-propagation,
  panel no-bleed, no-pandas guard) parametrized over the registry.
- Pinned feature-set manifest (`tests/feature_manifest.py`): a CI gate locking every feature's
  `(name, version, fingerprint)`.
- Calibration (integration) and benchmark (slow) test suites under their pytest markers.
