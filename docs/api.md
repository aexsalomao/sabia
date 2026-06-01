# API Reference

sabia exposes a flat, functional API: one module per feature family, plus a constructable registry
and a small set of shared types and helpers.

## Top level

| Symbol | Purpose |
|---|---|
| `sabia.compute(frame, *exprs)` | Materialize feature expressions into a `pl.DataFrame`. |
| `sabia.validate(frame)` | Check the input contract; raises `SabiaValidationError`. |
| `sabia.Registry` | Constructable registry; `Registry.default()`, `.where(pred)`, `.available(tier)`, `.specs()`. |
| `sabia.FeatureSpec` | Frozen metadata describing a feature. |
| `sabia.Family` / `Horizon` / `DataTier` / `Recurrence` / `Cost` / `Column` | Enums (see `FEATURES.md` §3). |

## Feature families

Each family is a module of pure functions returning `pl.Expr`:

`returns` · `trend` · `momentum` · `volatility` · `volume` · `distribution` ·
`mean_reversion` · `seasonality` · `cross_sectional`

```python
sabia.momentum.rsi(period=14)            # -> pl.Expr
sabia.volatility.vol_yz(window=21)       # -> pl.Expr (keyword-only params)
```

`microstructure` exists in the `Family` enum but ships in a later minor version (v1 is bars-only).

## Normalization

Trailing and cross-sectional transforms that preserve the causality invariant:

```python
sabia.normalize.zscore(expr, window=63)
sabia.normalize.xs_zscore(expr)
sabia.normalize.xs_rank(expr)
sabia.normalize.frac_diff(expr, d=0.4)
```

See `FEATURES.md` for the full contract and `CLAUDE.md` for the authoring rules.
