# Changelog

All notable changes to `sabia` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
