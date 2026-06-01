# sabia — technical feature library

> **`sabia`** — reads price/volume into features, grounded in the trading & finance literature.
> *v2.1 — online-readiness contract (§7), clean dependency boundary (§2), §10 decisions resolved.*
>
> **Position in the stack:** `marketgoblin` (data in) → **sabia** (features) → `quale` (signals).
> Depends on `quando` (sessions/calendar). **Does not depend on `quant_features`** — the lower
> brick must not point up at the higher one; `quant_features` consumes sabia (§6), not the reverse.
> Risk/eval math stays in `ruin`. sabia computes features and nothing else.

---

## 1. Scope

**In scope.** Pure functions over OHLCV bars (and, behind a data tier, trade/quote data) that
produce point-in-time features: returns, trend, momentum, volatility/range, volume/liquidity,
microstructure, distribution moments, mean-reversion/memory, seasonality, cross-sectional.

**Explicitly batch-first, online-ready.** sabia is *not* a streaming engine. It is designed so a
future online/production engine is a thin wrapper around it (§7), never a rewrite.

**Out of scope (v1), deliberately.** Pattern recognition (head-and-shoulders, candlesticks),
Fibonacci, Elliott, Ichimoku — no robust out-of-sample evidence. Also out: signal generation,
portfolio construction, backtesting, risk metrics, the online engine itself.

---

## 2. Invariants (non-negotiable)

Enforced by tests (§8), not convention.

1. **Causality.** A value at `t` depends only on information available at or before `t`. Verified
   by the shift-invariance property test.
2. **PIT correctness.** Revisable inputs use as-of joins on the *knowledge* timestamp.
3. **Purity.** No global/mutable state, no I/O, no clocks, no randomness. `f(inputs, params)` is
   deterministic and referentially transparent. (This is what makes §7 possible.)
4. **Polars-native.** Computation is Polars expressions; `numpy` only where an expression genuinely
   can't express it. **No pandas, anywhere.**
5. **Determinism.** Same inputs + same `(name, version)` + pinned Polars → identical output, within
   a documented float tolerance (§8.5).
6. **Dependencies.** `polars`, `numpy`, `quando`. Nothing else at runtime. New deps require a
   changelog decision.

---

## 3. The feature contract

### 3.1 `FeatureSpec` metadata (sabia owns this type)

| Field              | Type                         | Purpose                                                          |
|--------------------|------------------------------|------------------------------------------------------------------|
| `name`             | `str`                        | Stable identifier, snake_case (§3.3).                            |
| `version`          | `int`                        | Bumped on any formula change; immutable once published (§3.4).   |
| `fingerprint`      | `str`                        | Content hash of formula source + params + Polars pin (§3.4).     |
| `family`           | `Family` (enum)              | Structural axis → module + namespace.                            |
| `native_band`      | `frozenset[Horizon]`         | Bands where the signal is primary (the ● cells).                 |
| `lookback`         | `int | None`                 | Nominal window in bars.                                          |
| `min_history`      | `int`                        | Bars required to emit a valid value at `t` (drives §7 buffering).|
| `recurrence`       | `Recurrence` (enum)          | `FINITE` or `RECURSIVE` (§7.2).                                  |
| `effective_warmup` | `int`                        | For `RECURSIVE`: burn-in bars to converge within tolerance.      |
| `cost_class`       | `Cost` (enum)                | `O1` / `LINEAR` / `HEAVY` — per-update online cost hint.         |
| `data_tier`        | `DataTier` (enum)            | `TICK` / `MINUTE` / `DAILY` minimum input granularity.           |
| `inputs`           | `frozenset[Column]`          | Required columns.                                                |
| `output_dtype`     | `pl.DataType`                | Declared, asserted at compute.                                   |
| `citation`         | `str`                        | Literature anchor.                                               |
| `params`           | `Mapping[str, Any]`          | Frozen parameterization.                                         |

`native_band` is the horizon model: one structural tree (by `family`); "split by horizon" is the
query `registry.where(lambda s: band in s.native_band)`, not a second tree.

Integration with `quant_features.TemporalPanel` lives in the consumer or an optional
`sabia.adapters.quant_features` shim — never as a runtime dependency of sabia.

### 3.2 Compute signature

A feature is a pure function returning a Polars expression — one canonical definition, used by
both batch and any future online path (§7):

```python
def rsi(close: str = "close", period: int = 14) -> pl.Expr:
    """Wilder's RSI. Strictly trailing. RECURSIVE. Citation: Wilder (1978)."""
    delta = pl.col(close).diff()
    gain  = delta.clip(lower_bound=0)
    loss  = (-delta).clip(lower_bound=0)
    avg_gain = gain.ewm_mean(alpha=1 / period, adjust=False, min_periods=period)
    avg_loss = loss.ewm_mean(alpha=1 / period, adjust=False, min_periods=period)
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).alias(f"rsi_{period}")
```

Panel-aware features use `.over(symbol)` so windows never bleed across symbols.

Default is lazy: features return `pl.Expr`, the single source of truth. An eager convenience —
`sabia.compute(frame, *exprs) -> pl.DataFrame` — materializes columns for callers who want them;
it's just `select` over the same expressions, never a second definition.

### 3.3 Naming

`{measure}_{param}`, lower snake_case: `rsi_14`, `ret_log_5`, `vol_yz_21`, `amihud_21`,
`xs_rank_mom_252`. Family is metadata, not in the name. Names are unique library-wide.

### 3.4 Versioning, immutability & fingerprint

Once a feature populates any stored dataset or backtest, its formula is frozen. To change behavior,
publish a new `version` and column suffix; never edit the old one.

- The registry pins `(name, version) → exact implementation`.
- `fingerprint` is a content hash over the normalized formula source, `params`, and the pinned
  Polars version. A prod system records it alongside outputs so train↔serve identity is *provable*,
  not assumed; CI fails if a fingerprint changes without a version bump.
- A **feature-set manifest** (`list[(name, version, fingerprint)]`) pins the whole bundle a strategy
  consumes — drift detection at the set level, not column by column.

### 3.5 Warmup / NaN / degenerate-input semantics (uniform)

- Emits `null` until `min_history` is satisfied. No partial-window values.
- `null` propagates; sabia never imputes.
- Degenerate inputs have a stated policy, tested on fixtures: division by zero (zero-volume halts,
  flat series → Amihud, Kyle's λ, RSI `rs`) yields `null`, not `inf`; `log` of non-positive yields
  `null`. No silent `inf`/`NaN` escaping into downstream math.

---

## 4. Normalization layer (separate from features)

Composable trailing transforms, same causality invariant, same tests:

- `zscore(expr, window)` — rolling, trailing only.
- `xs_zscore(expr)` / `xs_rank(expr)` — cross-sectional, computed within a single timestamp slice
  (`.over(timestamp)`), never pooling across time.
- `frac_diff(expr, d)` — fractional differentiation (López de Prado); stationarity with memory.

---

## 5. Family + horizon model

Families (`Family` enum, one module each):
`returns` · `trend` · `momentum` · `volatility` · `volume` ·
`distribution` · `mean_reversion` · `seasonality` · `cross_sectional`

`microstructure` is defined in the `Family` enum but **ships in a later minor version** — v1 is
bars-only (OHLCV). The tier machinery (§7.3 / §3.1 `data_tier`) is in place so it slots in without
touching the existing families.

Horizon bands (`Horizon` enum) with default lookback grids (trading bars):

| Band       | Range          | Default lookbacks   | Native data tier |
|------------|----------------|---------------------|------------------|
| `MICRO`    | ticks → min    | n/a (event windows) | `TICK`           |
| `INTRADAY` | min → session  | {12, 26, 78}        | `MINUTE`         |
| `SHORT`    | 1–10 days      | {3, 5, 10}          | `DAILY`          |
| `MEDIUM`   | 2wk – 6mo      | {21, 63, 126}       | `DAILY`          |
| `LONG`     | 6mo – 2y+      | {126, 252, 504}     | `DAILY`          |

`native_band` per family follows the agreed coverage matrix (`microstructure` = `{MICRO, INTRADAY}`
only; `trend` = `{MEDIUM, LONG}`; `volatility` spans `MICRO`→`MEDIUM`).

---

## 6. API & module layout

Flat functional API per family (mirrors `quando`), plus a *constructable* registry — not a global:

```
sabia/
  spec.py          # FeatureSpec, Family, Horizon, DataTier, Recurrence, Cost
  registry.py      # Registry(): (name,version)->spec+fn; .where(...); .available(tier)
  validate.py      # input-contract checks (§7.3)
  normalize.py
  adapters/        # optional: quant_features shim, etc.
  returns.py  trend.py  momentum.py  volatility.py  volume.py
  distribution.py  mean_reversion.py  seasonality.py  cross_sectional.py
  # microstructure.py — deferred to a later minor version (§5)
```

```python
import sabia
reg = sabia.Registry.default()                                   # explicit, embeddable
sabia.momentum.rsi(period=14)                                    # -> pl.Expr
reg.where(lambda s: Horizon.MEDIUM in s.native_band)             # by-horizon view
reg.available(DataTier.DAILY)                                    # computable on these bars
sabia.compute(df, sabia.momentum.rsi(14), sabia.volatility.vol_yz(21))  # eager → DataFrame
```

Registration is explicit/constructable so sabia embeds cleanly and tests in isolation — no
import-order-dependent global singleton.

---

## 7. Production readiness — batch-first, online-ready

sabia does not run live. It provides the seams so a future online engine is a thin wrapper.

### 7.1 The windowed-recompute guarantee (load-bearing)

For any feature, recomputing on the last `min_history` bars reproduces the full-history value at
`t`: **exactly** for `FINITE`, **within tolerance** for `RECURSIVE` after `effective_warmup`
burn-in. This is asserted over the whole registry (§8.2). Given it, the minimal online engine is
"buffer the declared history, recompute on each closed bar, take the last value" — provably equal
to the backtest, with zero new feature code.

### 7.2 Recurrence tiers

- `FINITE` — bounded window; tail-recompute is exact; online-trivial today.
- `RECURSIVE` — unbounded memory (Wilder, EWM); tail-recompute is exact within tolerance once fed
  `effective_warmup` bars. A future engine may carry state for speed, but only as an optimization
  validated against the canonical batch expr (§8.2).

`effective_warmup` is derived analytically where the decay is known — for the EWM family with
smoothing `alpha`, `effective_warmup = ceil(ln(tol) / ln(1 - alpha))` (≈14×period at `tol = 1e-6`) —
and measured empirically (smallest window where parity holds within `tol`) otherwise. §8.2 asserts
whatever value the spec declares.

### 7.3 Input contract (the engine enforces, sabia declares)

`sabia.validate(frame)` checks the preconditions every feature assumes: timestamps sorted, unique,
tz-aware UTC; per-symbol sorted; required columns present with expected dtypes; **bars closed**
(no in-progress bar); for cross-sectional features, the slice at `t` is the complete cross-section.
sabia states these as contracts and offers the validator; *policy* on violation (wait, drop, null)
is the engine's call, not sabia's.

---

## 8. Testing (production gates — extends `testing.md`)

All cross-cutting tests are **parametrized per feature** (`ids = spec.name`) — one pass/fail each,
never a loop in a test body. Floats are compared with `pytest.approx` against a declared tolerance.

1. **Causality (property test).** Append future bars, recompute, assert prior values unchanged.
2. **Windowed-recompute parity.** §7.1: tail-window recompute == full-history value at `t` (exact
   for `FINITE`, within tolerance for `RECURSIVE`). The same harness will validate any future online
   path — none ships in v1.
3. **Reference values.** Per-feature parametrized `(input, expected)` tables, hand-checked or vs a
   reference impl. **Not** auto-captured snapshots — those lock in implementation and get blindly
   regenerated on failure (see `testing.md`).
4. **Reproduction calibration.** Cross-sectional and TS-momentum features reproduce published Sharpe
   bands on a known *local* sample. `@pytest.mark.integration`, no network — not in the unit suite.
5. **Degenerate inputs.** §3.5 policy on zero-volume, flat series, log-domain breaches → `null`.
6. **Property-based (`hypothesis`).** Scale/shift behavior, `null` propagation, no-pandas guard,
   output dtype matches spec, `fingerprint` stability.
7. **Determinism.** Same inputs + version + pinned Polars → identical within tolerance (recursive
   float accumulation is not bit-identical across platforms; tolerance is the honest contract).
8. **Performance.** Benchmarks (`pytest-benchmark`), `slow`-marked — not hard timing asserts, which
   flake. Per-update cost tracked against `cost_class`.

Unit gates (every feature): 1, 2, 3, 5, 6, 7. Separate/marked: 4 (integration), 8 (slow).

---

## 9. Performance contract

Vectorized Polars expressions only; no `.map_elements`/per-row Python in hot paths. Features return
expressions so `LazyFrame` fuses them. Cross-sectional ops use `.over(...)`. No materialized
per-symbol intermediate frames.

---

## 10. Decisions (resolved for v1)

- **Name** — `sabia`.
- **Default eval** — lazy (`pl.Expr`); eager via `sabia.compute(frame, *exprs)` (§3.2).
- **`native_band` typing** — single `frozenset[Horizon]` (● only); the ○ "applicable" cells are
  derivable and not worth the metadata weight.
- **Microstructure** — in the `Family` enum, ships in a later minor version; v1 is bars-only (§5).
- **`effective_warmup`** — analytic where decay is known (EWM: `ceil(ln(tol)/ln(1-alpha))`),
  empirical otherwise; asserted by §8.2 (§7.2).
- **Online** — nothing ships in v1. The §7 contract (history metadata + windowed-recompute parity)
  is the entire "build-it-later" surface; no protocol, no engine.
