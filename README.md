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

A complete, copy-paste-runnable example — two symbols, 30 daily bars, three features:

```python
import math
import polars as pl
import sabia
from sabia import BarSchema

def bars(symbol, base):
    rets = [0.012 if i % 3 else -0.008 for i in range(30)]   # deterministic, varied
    close, c = [], base
    for r in rets:
        c *= math.exp(r); close.append(c)
    return pl.DataFrame({
        "timestamp": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 30),
                                       interval="1d", time_zone="UTC", eager=True),
        "symbol": [symbol] * 30,
        "open":  [x * 0.999 for x in close],
        "high":  [x * 1.004 for x in close],
        "low":   [x * 0.996 for x in close],
        "close": close,
        "volume": [1_000_000.0 + 1000 * i for i in range(30)],
    })

frame = pl.concat([bars("AAA", 100.0), bars("BBB", 50.0)]).sort("symbol", "timestamp")

# BarSchema maps your physical columns to roles. sabia adjusts nothing — you declare which
# adjustment basis each column carries. .ohlcv(...) is the shorthand for the common OHLCV case;
# for richer inputs (a separate total-return close, VWAP, a market factor) build BarSchema(roles=...).
schema = BarSchema.ohlcv()   # open/high/low/close/volume; close also backs close@tr

# Factories return BoundFeature objects; compute resolves their roles and materializes. include_keys
# prepends symbol/timestamp, aligned row-for-row, which is what a downstream pipeline wants.
features = sabia.compute(
    frame,
    sabia.returns.ret_log(period=1),
    sabia.momentum.roc(window=5),
    sabia.volatility.vol_cc(window=10),
    schema=schema,
    include_keys=True,
)
print(features.tail(5))
```

```
shape: (5, 5)
┌────────┬─────────────────────────┬───────────┬──────────┬───────────┐
│ symbol ┆ timestamp               ┆ ret_log_1 ┆ roc_5    ┆ vol_cc_10 │
│ ---    ┆ ---                     ┆ ---       ┆ ---      ┆ ---       │
│ str    ┆ datetime[μs, UTC]       ┆ f64       ┆ f64      ┆ f64       │
╞════════╪═════════════════════════╪═══════════╪══════════╪═══════════╡
│ BBB    ┆ 2024-01-26 00:00:00 UTC ┆ 0.012     ┆ 0.020201 ┆ 0.009661  │
│ BBB    ┆ 2024-01-27 00:00:00 UTC ┆ 0.012     ┆ 0.040811 ┆ 0.009661  │
│ BBB    ┆ 2024-01-28 00:00:00 UTC ┆ -0.008    ┆ 0.020201 ┆ 0.010328  │
│ BBB    ┆ 2024-01-29 00:00:00 UTC ┆ 0.012     ┆ 0.020201 ┆ 0.009661  │
│ BBB    ┆ 2024-01-30 00:00:00 UTC ┆ 0.012     ┆ 0.040811 ┆ 0.009661  │
└────────┴─────────────────────────┴───────────┴──────────┴───────────┘
```

Each feature emits `null` during its warm-up window (a rolling stat needs a full window first); use
`sabia.drop_warmup(...)` to trim those rows. Query the shipped catalog by horizon or data tier:

```python
reg = sabia.Registry.default()
reg.where(lambda s: sabia.Horizon.MEDIUM in s.native_band)
reg.available(sabia.DataTier.DAILY)
```

## Invariants

Causality · point-in-time correctness · purity (no I/O, clocks, randomness) · Polars-native
(no pandas) · determinism within a declared tolerance. All enforced by tests, not convention. See
`FEATURES.md` for the full spec.

## Reproducibility & versioning

`sabia` pins **`polars==1.39.3`** exactly and targets **Python ≥ 3.13** on purpose: the manifest's
feature `fingerprint` (§3.4) folds in the Polars version, so a stored dataset's exact feature
definitions stay provable across train and serve. These pins are deliberate, not an oversight — they
will relax toward a range once the fingerprint contract is settled past 1.0.

The fingerprint is a **best-effort reproducibility hash**, not a behavioral guarantee: it hashes each
feature's bound params, roles, dependencies, the Polars version, and the *normalized source* of its
expression (and first-party helpers). That catches the changes that matter in practice — a retuned
constant, a swapped role, an edited formula — but, being source-based, two mathematically equivalent
rewrites can still produce different fingerprints. Treat a fingerprint change as "prove this was
intended" (the CI manifest gate enforces exactly that), not as a proof of behavioral difference.
`FeatureSetManifest` serializes (`to_json` / `from_json`) so a dataset can carry its definitions.

## License

MIT
