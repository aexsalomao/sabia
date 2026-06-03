# Concepts & guide

This page is the mental model. Read it once and the rest of the API falls into place. It assumes the
[quickstart](index.md) has run; here we slow down and explain *why* each moving part exists.

## What sabia is (and is not)

`sabia` is a **pure technical-feature library**. You hand it canonicalized OHLCV bars; it hands back
[Polars](https://pola.rs/) expressions that are strictly trailing and point-in-time correct. It is
the *features* brick in a layered stack:

```
marketgoblin (data in) → sabia (features) → quale (signals)
```

It is deliberately small in scope. sabia **does**:

- compute ~66 technical features across 9 families, each grounded in the trading/finance literature,
- standardize and stationarize them (rolling z-score, cross-sectional rank, fractional differencing),
- describe every feature's metadata (warm-up, recurrence, unit, evidence tier, citation),
- pin a feature set's exact definitions for reproducibility (fingerprints + manifest).

sabia **does not** fetch data, adjust prices for splits/dividends, decide tradability, generate
signals, build portfolios, or compute risk/PnL. Those belong to neighbouring layers — `marketgoblin`
(data), `quale` (signals), `ruin` (risk/eval). sabia computes features and nothing else.

!!! note "The non-negotiables"
    Every feature is **pure** (no I/O, clocks, randomness, or logging), **causal** (uses only
    information at or before bar `t` — no window ever touches the future), **point-in-time correct**,
    and **deterministic** within a declared float tolerance. These are enforced by the test suite —
    causality tests, windowed-recompute parity tests, a no-pandas import guard — not by convention.

## The capabilities at a glance

| Family | What it measures | Examples |
|---|---|---|
| `returns` | log/simple returns, intraday/overnight splits, drawdown | `ret_log`, `ret_intraday`, `drawdown` |
| `trend` | moving averages, distance-to-trend, MACD, ADX, OLS slope | `sma_dist`, `ema_dist`, `macd_hist`, `adx` |
| `momentum` | rate of change, RSI, stochastics, formation-period momentum | `roc`, `rsi`, `stoch_k`, `mom`, `cci` |
| `volatility` | close-to-close + OHLC range estimators, EWMA, ATR | `vol_cc`, `vol_yz`, `vol_gk`, `vol_ewma`, `atr` |
| `volume` | liquidity, dollar volume, Amihud illiquidity, spreads, flow | `dollar_vol`, `amihud`, `mfi`, `roll_spread` |
| `distribution` | skew, kurtosis, downside/up-down dispersion | `skew`, `kurt`, `downside_dev` |
| `mean_reversion` | autocorrelation, variance ratio, half-life, z-distance | `autocorr`, `var_ratio`, `half_life`, `zscore_close` |
| `seasonality` | day-of-week, turn-of-month effects | `season_dow`, `season_tom` |
| `cross_sectional` | panel rank/z momentum, reversal, market beta, idio vol | `xs_rank_mom`, `beta`, `idio_vol` |
| `normalize` | trailing & cross-sectional standardizers, fractional diff | `zscore`, `xs_rank`, `frac_diff` |

The [feature catalog](catalog.md) lists every shipped feature with its params, roles, warm-up,
recurrence, unit, and evidence tier. `microstructure` exists in the `Family` enum but ships in a
later minor version — v1 is bars-only.

## Column roles & `BarSchema`

This is the one concept worth getting right before anything else.

A feature does not ask for a column named `"close"`. It asks for a **role** — say `close@tr` (the
close on a *total-return* basis) or `high@split` (the high on a *split-only* basis). A `BarSchema`
is the caller-supplied map from those roles to the physical column names in *your* frame.

```python
from sabia import BarSchema, PriceRole, PriceField, Adjustment

# Explicit form: declare exactly which physical column carries which role.
schema = BarSchema(roles={
    PriceRole(PriceField.CLOSE, Adjustment.TR):    "adj_close",   # dividend-adjusted close
    PriceRole(PriceField.CLOSE, Adjustment.SPLIT): "close",       # split-adjusted only
    PriceRole(PriceField.HIGH,  Adjustment.SPLIT): "high",
    # ...
})
```

Why two adjustment bases? Returns/momentum/trend features want a **total-return** series
(`@tr`) so a dividend drop is not mistaken for a real loss. Range estimators (Parkinson, Garman-Klass,
Yang-Zhang) want a **split-only** series (`@split`) where the high/low/open/close are mutually
consistent within the bar. sabia adjusts nothing — *you* declare which basis each column carries,
and the manifest pins that declaration.

For the common case, `BarSchema.ohlcv()` builds the whole map from plain column names:

```python
schema = BarSchema.ohlcv()                       # open/high/low/close/volume by name
schema = BarSchema.ohlcv(close="px", volume="vol")   # custom names
schema = BarSchema.ohlcv(tr_close="adj_close")       # separate total-return close
```

!!! warning "The `@tr` conflation in `ohlcv()`"
    When you omit `tr_close`, `close@tr` is backed by the **same split-only column** as
    `close@split`. There is no separate total-return series, so return/momentum/trend features
    silently run on split-only prices *labelled* total return. On dividend-paying instruments this is
    wrong — the dividend drop is treated as a real return. Pass `tr_close=` (e.g. an `adj_close`
    column) whenever a distinct total-return close exists.

`symbol` and `timestamp` are fixed canonical column names (override via `symbol_col=`/`timestamp_col=`
if yours differ); they are not role-tagged. Pass `closed_col=` to mark which bars are closed and
`calendar=` for the exchange code.

## Feature factories & `BoundFeature`

Every feature is a **factory** with keyword-only params. Calling it binds the params and returns a
`BoundFeature` — it does **not** return a `pl.Expr` directly:

```python
import sabia

bf = sabia.momentum.rsi(period=14)   # -> BoundFeature
bf.spec.name          # 'rsi_14'
bf.spec.min_history   # 249  (bars of warm-up before it is valid)
bf.expr(schema)       # -> pl.Expr, with roles resolved against `schema`
```

A `BoundFeature` is two things: an immutable `.spec` (all the metadata — see [the registry](#the-registry-featurespec))
and `.expr(schema)`, which resolves the feature's roles against your schema and returns the Polars
expression. You rarely call `.expr()` yourself — `compute()` does it for you.

## Computing features

`sabia.compute()` is the workhorse. It validates the frame, resolves every feature's roles, and
materializes a `pl.DataFrame`:

```python
features = sabia.compute(
    frame,
    sabia.returns.ret_log(period=1),
    sabia.momentum.roc(window=5),
    sabia.volatility.vol_cc(window=10),
    schema=schema,
    include_keys=True,   # prepend symbol + timestamp, aligned row-for-row
)
```

The result carries only the feature columns, aligned row-for-row with `frame`. `include_keys=True`
prepends the identity columns (`symbol` if it's a panel, plus `timestamp`) — what a downstream
pipeline usually wants.

- **`compute_lazy(frame, *features, schema=)`** returns a `pl.LazyFrame` and materializes nothing.
  It is time-series only (no cross-sectional features) and **does not validate** — call
  `sabia.validate(...)` yourself first if the frame is untrusted.
- Two features with the same name in one `compute` call are rejected (a naming collision is a bug).
- Time-series features fuse into a single `select` / one `collect()`. Cross-sectional features are
  inherently two-pass and each costs its own materialization (an N+1 pattern) — batch them
  deliberately.

### Warm-up, nulls, and degenerate inputs

A rolling statistic needs a full window before it can emit, so every feature produces `null` during
its **warm-up**. `spec.min_history` is the number of bars before a feature is valid. To trim the
leading warm-up rows from a result:

```python
clean = sabia.drop_warmup(features, [bf1, bf2], symbol_col="symbol")
n = sabia.max_min_history([bf1, bf2])   # the largest warm-up across the set
```

`drop_warmup` removes the first `max_min_history(...)` rows (per symbol on a panel) — a safe upper
bound, so no null-warm-up row survives.

Separately, **degenerate input** yields `null` by design: a flat price series gives an undefined
RSI, a zero-volume bar gives an undefined MFI, a log of a non-positive price is undefined. This is
documented domain behavior, not an error — `null` is the honest answer. Loud failure is reserved for
malformed *frames*, at `validate()`.

## Validation & the input contract

`sabia.validate(frame, schema=...)` enforces the contract every feature assumes: timestamps are
sorted, unique, and tz-aware UTC; each symbol is independently sorted; required columns exist with
the right dtypes; bars are closed; and — for cross-sectional features — the panel is complete across
the declared universe. It raises `SabiaValidationError` on the first hard violation.

`compute()` runs validation for you. Its strictness is controlled by `ValidationMode`:

| Mode | Behavior |
|---|---|
| `STRICT` (default) | raise on any contract violation |
| `RESEARCH` | warn on completeness/finalization; still raise on schema/dtype/role errors |
| `OFF` | skip validation entirely |

```python
sabia.compute(frame, bf, schema=schema, validation=sabia.ValidationMode.RESEARCH)
```

For a non-fatal health check — counts and ranges, never a raise — use `audit_frame`, which returns a
`FrameAudit` you can inspect before launching an expensive feature job:

```python
report = sabia.audit_frame(frame, schema=schema, features=[bf1, bf2])
```

## The registry & `FeatureSpec`

`Registry.default()` assembles every shipped feature. It is built by **explicit collection** — there
is no global singleton and no import-time registration side effect, so it is embeddable and
test-isolatable. Query it functionally:

```python
reg = sabia.Registry.default()

reg.specs()                                                    # all FeatureSpecs
reg.where(lambda s: sabia.Horizon.MEDIUM in s.native_band)     # by horizon band
reg.where(lambda s: s.family is sabia.Family.VOLATILITY)       # by family
reg.available(sabia.DataTier.DAILY)                            # computable on daily bars
```

`reg.available(tier)` returns features computable on input bars of that granularity — finer input
bars unlock strictly more features (`TICK` < `MINUTE` < `DAILY`).

Each feature carries a frozen `FeatureSpec`. The fields most useful to a consumer:

| Field | Meaning |
|---|---|
| `name` | unique snake-case id, e.g. `rsi_14` |
| `family` | one of the 9 `Family` values |
| `native_band` | the `Horizon` bands where the feature is primary |
| `min_history` | warm-up bars before the feature is valid |
| `effective_warmup` | burn-in for recursive features to converge within tolerance |
| `recurrence` | `FINITE` (exact tail-recompute) or `RECURSIVE_DECAY` (exact within tolerance) |
| `output_unit` | `Unit` of the output — `LOG_RETURN`, `ZSCORE`, `INDEX_0_100`, … |
| `data_tier` | minimum input granularity (`DataTier`) |
| `cost_class` | per-update online `Cost` hint (`O1` / `LINEAR` / `HEAVY`) |
| `input_roles` | the column roles a `BarSchema` must resolve |
| `evidence` | empirical standing *as constructed* — `formula_only`, `ta_canon`, `academic_single`, `academic_replicated` (not a predictability claim) |
| `citation` | the literature reference |

`sabia.describe(bf)` renders a readable one-feature card from the spec for notebooks and debugging.

## Normalization & transforms

Features are raw measurements; normalization makes them comparable. The `normalize` module ships
trailing and cross-sectional transforms that obey the **same causality invariant** as features. A
transform factory returns a `BoundTransform` — a pinned `.spec` plus `.apply(expr) -> expr` — so you
pipe a resolved feature expression through it:

```python
z   = sabia.normalize.zscore(window=63)         # trailing rolling z-score
xsz = sabia.normalize.xs_zscore(winsorize=3.0)  # cross-sectional z within each timestamp
xsr = sabia.normalize.xs_rank()                 # cross-sectional percentile rank in (0, 1]
fd  = sabia.normalize.frac_diff(d=0.4)          # fractional differencing: stationarity with memory

expr = z.apply(sabia.trend.ols_slope(window=63).expr(schema))
```

- **Trailing** transforms (`zscore`, `frac_diff`) look only backward over `window`/lag.
- **Cross-sectional** transforms (`xs_zscore`, `xs_rank`) compute within a single timestamp slice and
  never pool across time.
- Degenerate slices (zero dispersion) yield `null`, never `inf`.
- `frac_diff` (Lopez de Prado FFD) trades the minimum differencing needed for stationarity against
  retaining long memory; its weight count is surfaced as the transform's warm-up.

## Cross-sectional / panel features

Cross-sectional features rank or standardize *across symbols at each timestamp* (`xs_rank_mom`,
`xs_z_mom`, `rev_1m`), or regress a symbol against the market (`beta`, `idio_vol`). They impose two
extra contract requirements:

- **`requires_universe`** — rank/z/reversal features compare a symbol to its peers, so you must
  declare the universe explicitly. Pass `universe=[...]` (a static symbol set) or `membership=`
  (an as-of `(symbol, start, end)` frame) to `compute`. sabia asserts completeness against the
  declared universe; it never infers it from whichever symbols happen to be present.
- **`market_ret`** — `beta`/`idio_vol` are per-symbol but need a market-return column; declare it as
  a role in the schema.

```python
out = sabia.compute(
    panel,
    sabia.cross_sectional.xs_rank_mom(formation=252, skip=21),
    schema=schema,
    universe=["AAA", "BBB", "CCC"],
)
```

## Recipes & `FeatureSet`

A `FeatureSet` is an immutable, named bundle of bound features — a name for a fixed list, nothing
more. It saves you threading dozens of features through a variadic `compute`:

```python
fs = sabia.recipes.daily_core()          # returns + momentum + vol + trend + distribution
fs.compute(frame, schema=schema, include_keys=True)
fs.names()                               # the feature names in the set
fs.manifest(schema)                      # pin it (see below)
```

Shipped recipes: `daily_core()`, `volatility_core()`, `cross_sectional_core()`. They are *sensible
default panels*, **not** strategies — no signals, thresholds, or portfolio opinions. Build your own
with `sabia.FeatureSet([...])`.

## Reproducibility: fingerprints & manifests

Every feature carries a `fingerprint` — a best-effort reproducibility hash over its bound params,
roles, dependencies, the **pinned Polars version**, and the normalized source of its expression.
A `FeatureSetManifest` bundles the fingerprints of a whole feature set (plus any transforms and the
schema) and serializes so a stored dataset can carry its exact definitions across train and serve:

```python
manifest = sabia.FeatureSetManifest.of([bf1, bf2], (), schema, sabia_version=sabia.__version__)
blob = manifest.to_json()
restored = sabia.FeatureSetManifest.from_json(blob)
```

A fingerprint change means "prove this was intended" — a retuned constant, a swapped role, an edited
formula — and the CI manifest gate enforces exactly that. It is source-based, so two mathematically
equivalent rewrites *can* fingerprint differently; treat a change as a flag to review, not as a proof
of behavioral difference. This is why sabia pins `polars` exactly and targets a fixed Python — the
fingerprint folds in the Polars version, so a dataset's feature definitions stay provable.

## Where to go next

- **[Feature catalog](catalog.md)** — every shipped feature, its params, warm-up, and evidence tier.
- **[API reference](api.md)** — the complete public surface.
- **Runnable examples** — `examples/01_quickstart.py` … `07_manifest.py` in the repo, plus a Plotly
  notebook gallery under `examples/notebooks/`.
- **`FEATURES.md`** — the full specification and literature citations.
