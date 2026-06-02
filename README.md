# sabia

> Polars-native technical features for trading pipelines — pure, point-in-time, online-ready.

`sabia` reads price/volume into features, grounded in the trading & finance literature. It is the
**features** brick in a layered stack:

```
marketgoblin (data in) → sabia (features) → quale (signals)
```

Runtime dependencies are just `polars` and `numpy`. (The stack's calendar brick
[`quando`](https://github.com/aexsalomao/quando) will be wired in when calendar-aware seasonality or
the microstructure tier needs it — v1 seasonality is pure vectorized datetime.) Risk/eval math lives
in `ruin`; signals live in `quale`. sabia computes features and nothing else.

## What it is

Pure factories over OHLCV bars. A factory binds its params and returns a `BoundFeature` — an immutable
`.spec` plus a `.expr(schema) -> pl.Expr` that resolves **column roles** (`close@tr`, `high@split`)
against a caller-supplied `BarSchema`. Features are strictly trailing, point-in-time correct, and
deterministic. **Batch-first, online-ready**: nothing streams in v1, but every feature declares the
history it needs and is covered by a windowed-recompute parity test, so a future online engine is a
thin wrapper rather than a rewrite.

## Install

```bash
uv sync --extra dev
```

## Quickstart

```python
import polars as pl
import sabia
from sabia import BarSchema, PriceRole, PriceField, Adjustment

frame = ...  # OHLCV LazyFrame/DataFrame; see sabia.validate for the input contract

# A BarSchema maps your physical column names to roles. sabia adjusts nothing — you declare which
# adjustment basis each column carries.
schema = BarSchema(roles={
    PriceRole(PriceField.OPEN,  Adjustment.SPLIT): "open",
    PriceRole(PriceField.HIGH,  Adjustment.SPLIT): "high",
    PriceRole(PriceField.LOW,   Adjustment.SPLIT): "low",
    PriceRole(PriceField.CLOSE, Adjustment.SPLIT): "close",
    PriceRole(PriceField.CLOSE, Adjustment.TR):    "close",
})

# Factories return BoundFeature objects; compute resolves roles against the schema and materializes:
df = sabia.compute(frame, sabia.momentum.rsi(period=14), sabia.volatility.vol_yz(window=21),
                   schema=schema)

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
