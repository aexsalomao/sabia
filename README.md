# sabia

> Polars-native technical features for trading pipelines — pure, point-in-time, online-ready.

`sabia` reads price/volume into features, grounded in the trading & finance literature. It is the
**features** brick in a layered stack:

```
marketgoblin (data in) → sabia (features) → quale (signals)
```

It depends on [`quando`](https://github.com/aexsalomao/quando) for sessions/calendar, and on nothing
else at runtime beyond `polars` and `numpy`. Risk/eval math lives in `ruin`; signals live in `quale`.
sabia computes features and nothing else.

## What it is

Pure functions over OHLCV bars that return Polars expressions (`pl.Expr`) — strictly trailing,
point-in-time correct, deterministic. **Batch-first, online-ready**: nothing streams in v1, but every
feature declares the history it needs and is covered by a windowed-recompute parity test, so a future
online engine is a thin wrapper rather than a rewrite.

## Install

```bash
uv sync --extra dev
```

## Quickstart

```python
import polars as pl
import sabia

frame = ...  # OHLCV LazyFrame/DataFrame; see sabia.validate for the input contract

# Features are pl.Expr — compose them lazily, or materialize eagerly:
df = sabia.compute(frame, sabia.momentum.rsi(14), sabia.volatility.vol_yz(21))

# Query the registry by horizon or data tier:
reg = sabia.Registry.default()
reg.where(lambda s: sabia.Horizon.MEDIUM in s.native_band)
reg.available(sabia.DataTier.DAILY)
```

## Invariants

Causality · point-in-time correctness · purity (no I/O, clocks, randomness) · Polars-native
(no pandas) · determinism within a declared tolerance. All enforced by tests, not convention. See
`FEATURES.md` for the full spec.

## License

MIT
