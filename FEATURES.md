# sabia — technical feature library

> **`sabia`** — reads price/volume into features, grounded in the trading & finance literature.
> *v5 — bound feature objects (factories return `BoundFeature`, `.expr(schema)` resolves roles; §4.2),
> adjustment-tagged roles `field@adjustment` (§2.2), `ValidationMode` (§8.3), integer warmups,
> evidence re-tiered, rolling-regression stack moved to Tier 1.1.*
>
> **Position in the stack:** `marketgoblin` (data in) → **sabia** (features) → `quale` (signals).
> Depends on `quando` (sessions/calendar) **as a future adapter only** — v1 ships a dependency-free
> internal calendar (§7 module map, §11). Does not depend on `quant_features` (it consumes sabia).
> Risk/eval math stays in `ruin`. sabia computes features and nothing else.
>
> **sabia does not decide whether a feature is tradable at the timestamp it is indexed by.** It
> computes *observation-time* features from closed, PIT-valid inputs. Signal construction and
> execution-time alignment belong downstream (§2.3).

---

## 1. Scope

**In scope (overall).** Pure factories over canonicalized market bars with explicit column *roles*
(§2). sabia never guesses whether a column is raw, adjusted, finalized, tradable, or PIT-valid.

**In scope for v1.** OHLCV bars only (+ an optional `market_ret` factor input). **No microstructure**
(needs tick/L1-L2; §6), no options (§13). **One calendar per frame/manifest** (§2.4).

**Batch-first, online-ready.** Not a streaming engine; a future online engine is a thin wrapper (§8).

**Out of scope (v1), deliberately.** Pattern recognition, Fibonacci, Elliott, Ichimoku; signal
generation, portfolio construction, backtesting, risk metrics, the online engine.

---

## 2. Data contract

sabia consumes canonicalized bars and validates them at the boundary (§8.3). It adjusts nothing and
infers nothing.

### 2.1 Canonical bar schema

| Column                | Dtype                  | Req. | Notes                                       |
|-----------------------|------------------------|:----:|---------------------------------------------|
| `symbol`              | `Utf8` / `Categorical` |  ●   | Panel key. Fixed canonical name, not role-tagged. |
| `timestamp`           | `Datetime[μs, UTC]`    |  ●   | Bar **observation** time (§2.3). Fixed canonical name. |
| OHLC, volume          | `Float64`/`Int64`      |  ●   | Role-tagged; physical names arbitrary (§2.2).|
| `vwap`,`trade_count`  | `Float64`/`Int64`      |  ○   |                                             |
| `is_final`            | `Boolean`              |  ○   | Bars-closed marker (§8.3); else upstream-certified. |
| `session_id`,`halted` | `Utf8`/`Boolean`       |  ○   | From `quando`; halt vs bad data (§2.4).     |

### 2.2 Column roles & adjustment policy

**Roles, not column strings.** Physical names are arbitrary; `BarSchema` maps them to roles, features
declare roles, the manifest records the mapping:

```python
BarSchema(roles={
    PriceRole(PriceField.CLOSE, Adjustment.TR):    "px_close_tradj",
    PriceRole(PriceField.HIGH,  Adjustment.SPLIT): "px_high_splitadj",
    VolumeRole(VolumeField.VOLUME, Adjustment.SPLIT): "vol_splitadj",
    FactorRole.MARKET_RET: "spx_ret",
}, closed_col="is_final", calendar="XNYS")
```

**Adjustment is part of the role identity.** A `PriceRole` is `(field, adjustment)` with
`Adjustment ∈ {TR, SPLIT, RAW}`, written `field@adjustment`. So `close@tr` and `high@split` are
distinct roles — no single `*_adj` doing two jobs. Role namespaces (`typing.py`): `PriceRole`
(`close/open/high/low @ tr|split|raw`, `vwap@split`), `VolumeRole` (`volume@split|raw`,
`dollar_volume@raw|split`), `FactorRole` (`market_ret`, …), `CalendarRole` (`session`).
`InputRole = PriceRole | VolumeRole | FactorRole | CalendarRole`. The market series is a `FactorRole`,
not jammed into OHLCV. `symbol` and `timestamp` are fixed canonical column names (`BarSchema.symbol_col`
/ `timestamp_col`), not role-tagged.

**sabia adjusts nothing.** Frozen policy (so features are comparable across vendors):
- **Returns / momentum / trend / oscillators on close** use `close@tr` (close-to-close = total return).
- **Range estimators** (Parkinson/GK/RS/YZ, ATR, stochastic/Williams/CCI/Bollinger) use **split-only**
  OHLC (`o/h/l/c@split`). Dividend-adjusted OHLC distorts historical ranges, so range vol on it
  measures adjustment artifacts. Adjusted OHLC must be **factor-consistent** (`validate()` checks
  ordering `low ≤ min(open,close) ≤ max(open,close) ≤ high` after resolution, §2.1).
- **Volume** uses `volume@split`. **Dollar volume**: `dollar_volume@raw` = `close@raw × volume@raw`
  (actual traded notional) and `dollar_volume@split`. **Amihud uses raw notional with TR returns** (§12).
- **VWAP** uses `vwap@split`, same basis as `close@split`; `vwap_dist_close` requires both on that basis.

### 2.3 Observation time vs. tradability

Three clocks: `event_time`, `available_time`, `knowledge_time`. **Rule:** features are indexed by
**observation timestamp, not tradability timestamp**; the consumer shifts per execution policy. sabia
never shifts. **`FactorRole` inputs are subject to the same contract** — for cross-market factors the
caller must align factor observations to the asset's session calendar before validation, or `mkt`
re-introduces timing ambiguity.

### 2.4 Calendar & bars

Lookbacks are **observed bars, not calendar days**, after `quando` resolves half-days/holidays.
A zero-volume **halt** (`halted`/`session_id`) ≠ zero-volume from bad data; sabia applies the
degenerate policy (§4.5) and uses the halt flag where present. **v1 assumes one calendar per
frame/manifest** (recorded in `BarSchema.calendar`); multi-calendar panels require partitioned compute
or a later calendar role — sabia does not pretend one calendar fits all. v1 ships an internal
`UtcCalendar` (§7); exchange calendars arrive via a future `quando` adapter without touching features.

### 2.5 Cross-sectional universe

Completeness is *relative to a declared universe*. Cross-sectional features declare
`requires_universe`/`requires_complete_panel`; the caller passes either a static universe
(`validate(frame, universe=[…])`) or as-of membership as a `(symbol, start, end)` frame
(`validate(frame, membership=df)`), under which the expected cross-section at each timestamp `t` is
`{symbol : start <= t < end}` (the point-in-time model for IPOs / delistings). sabia asserts
completeness against the declared universe, never infers it.

---

## 3. Invariants (non-negotiable)

Enforced by tests (§9), not convention.

1. **Causality** — value at `t` from info at/before `t`. Shift-invariance test.
2. **PIT correctness** — as-of joins on the *knowledge* timestamp (§2.3).
3. **Purity** — no global/mutable state, I/O, clocks, randomness, or logging in core. Schema is passed
   explicitly (§4.2), never global. Deterministic.
4. **Polars-native** — the resolved computation is Polars expressions; `numpy` only via the §10 escape
   hatch. **No pandas.**
5. **Determinism** — same inputs + `(name, version)` + pinned Polars → identical within tolerance (§9).
6. **Dependencies** — `polars`, `numpy`. (`quando` is a *future* adapter, not a v1 runtime dep — §7.)
   New deps are a changelog decision.

---

## 4. The feature contract

### 4.1 `FeatureSpec` — a parameterized factory binds to a concrete spec

A registered feature is a **parameterized factory**: binding params yields a `BoundFeature` carrying
an immutable `FeatureSpec` and a schema-resolving `.expr(schema)` (§4.2). The registry holds factories;
the manifest stores **bound** specs; the fingerprint is over bound params.

| Field | Type | Purpose |
|---|---|---|
| `name`,`version` | `str`,`int` | Bound id (§4.3); immutable once published (§4.4). |
| `fingerprint` | `str` | Canonical-id hash over bound params (§4.4). |
| `family`,`native_band` | `Family`,`frozenset[Horizon]` | Structural axis / primary bands. |
| `lookback`,`min_history` | `int|None`,`int` | Nominal window / emit-and-buffer threshold (§4.5). |
| `recurrence`,`effective_warmup` | `Recurrence`,`int` | §8.2; decay burn-in. |
| `cost_class`,`data_tier` | `Cost`,`DataTier` | `O1/LINEAR/HEAVY`; `TICK/MINUTE/DAILY`. |
| `input_roles` | `frozenset[InputRole]` | Declared roles incl. adjustment (§2.2). |
| `null_policy` | `NullPolicy` | §4.5. |
| `output_dtype`,`output_unit`,`output_range` | `pl.DataType`,`Unit`,`tuple|None` | `Unit`: `LOG_RETURN/RATIO/INDEX_0_100/UNITLESS/RETURN_STD_PER_BAR/PRICE_UNITS/RANK_0_1/ZSCORE`. |
| `evidence` | `Evidence` | Empirical standing of the feature *as constructed* (§4.1.1). |
| `dependencies` | `tuple[FeatureRef|TransformRef, ...]` | Manifest provenance (§4.4); **not** a runtime DAG (§5). |
| `requires_universe`,`requires_complete_panel` | `bool` | §2.5. |
| `citation`,`params` | `Citation`,`FrozenParams` | Structured (formula + optional empirical); immutable params. |

**4.1.1 `evidence`** describes the feature's empirical standing *as constructed* — **not** a guarantee
of return predictability. Pure-measurement features (returns, vol estimators) are `FORMULA_ONLY` or
the estimator's single-paper standing; replicated *anomalies* (momentum, Amihud, MAX, 52-week high)
earn `ACADEMIC_REPLICATED`. Formula vs. empirical references are separated in `citation`.

### 4.2 Compute signature — factories return `BoundFeature`, not `pl.Expr`

Role resolution needs a schema, so a feature factory returns a **`BoundFeature`** (owns `.spec`,
materializes `.expr(schema) -> pl.Expr`). It never returns a raw expression — that was the v4
contradiction (`role()` had no schema). Two-phase, purity intact:

```python
def rsi(*, period: int = 14, price: PriceRole = CLOSE_TR) -> BoundFeature:
    """Wilder's RSI. RECURSIVE_DECAY. evidence=TA_CANON. Wilder (1978)."""
    name = naming("rsi", period, role=price, default_adjustment=Adjustment.TR)
    def build(s: BarSchema) -> pl.Expr:
        d = pl.col(s.column(price)).diff()
        ag = d.clip(lower_bound=0).ewm_mean(alpha=1/period, adjust=False, min_samples=period)
        al = (-d).clip(lower_bound=0).ewm_mean(alpha=1/period, adjust=False, min_samples=period)
        return grouped(100 - 100 / (1 + ag/al), s.symbol_col).alias(name)
    return bind_feature(build, name=name, recurrence=Recurrence.RECURSIVE_DECAY,
                        min_history=187, effective_warmup=187, input_roles=(price,), …)
```

```python
b = sabia.momentum.rsi(period=14)             # BoundFeature
b.spec                                         # immutable metadata
b.expr(bar_schema)                             # -> pl.Expr (roles resolved)
sabia.compute(df, b, schema=bar_schema, validation=ValidationMode.STRICT)  # resolves + selects
```

Panel features use `.over(symbol)` (§9). Default lazy; eager via `compute(...)`.

### 4.3 Naming grammar

`{measure}_{params...}`, params in declared order. **Rule A — the name encodes the non-default role**:
`rsi_14` is `close@tr`; `rsi_raw_14` is `close@raw`. Multi-output groups suffix the role
(`di_plus_14`, `macd_signal_12_26_9`, `bb_pctb_20_2`). Momentum is `mom_{formation}_{skip}` — `mom_252_21`
is formation 252, skip 21 (not "252-day momentum over 21 days"). Annualized variants take `_ann` (§4.6).
Family-prefixed where a bare name ages badly (`season_dow`, `spread_corwin_schultz`, `vwap_dist_close`).
A single `compute(...)` rejects two same-named expressions.

### 4.4 Versioning, immutability, fingerprint & manifest

Bound specs are frozen once they populate a dataset/backtest — change behavior via a new `version`,
never an edit. `params` is a `FrozenParams` over hashable values. `fingerprint = hash(canonical_id +
version + bound params + input_roles + dependency fingerprints + dep pins)` — excludes
formatting/comments, includes helper and dependency fingerprints; never raw source. The
**`FeatureSetManifest`** pins bound features *and* transforms, plus the **role mapping**, schema,
calendar, adjustment policy, dependency DAG, and tool versions.

### 4.5 Warmup / null / degenerate semantics

`null` until `min_history`; no partial-window values. **`min_history` is the emit-and-buffer
threshold:** for `FINITE` it equals the window; for `RECURSIVE_DECAY` it equals `effective_warmup`
(the feature emits null until then, so windowed-recompute parity is exact-within-tolerance — the
statistical minimum `period` is documented but not the emit threshold). **Window-null policy**
(`null_policy`): `REQUIRE_FULL_WINDOW` (default) | `MIN_VALID_COUNT(min_valid)` | `SKIP_NULLS`. `null`
propagates; no imputation. Degenerate inputs → `null`, never `inf`/`NaN`.

### 4.6 Output conventions

Returns are log returns (`LOG_RETURN`) unless named otherwise. Volatility outputs are **per-bar**
unless suffixed `_ann`; annualization uses the calendar/session factor (`SessionCalendar.bars_per_year`),
never a hardcoded 252 in feature code. Cross-sectional `xs_rank`: percentile `(0,1]` (min `1/n`, max `1.0`,
from the rank/count implementation), ascending (so high
momentum ranks high), average ties, nulls stay null, declared `min_count`; `xs_zscore` optionally
winsorizes before standardizing.

---

## 5. Normalization & transforms (separate from features)

Composable trailing transforms with a **`TransformSpec`** (lookback, min_history, null_policy,
input/output dtype + unit, causality, fingerprint, dependencies), so the manifest pins transforms too.
A transform takes a bound feature / resolved expression and returns one (`BoundTransform.apply`).

- `zscore(window)` — rolling, trailing.
- `xs_zscore(winsorize=…)` / `xs_rank(ascending=…, min_count=…)` — within one timestamp
  slice (`.over(timestamp)`), never pooling across time (§4.6).
- `frac_diff(d)` — fractional differentiation; long memory → large `effective_warmup` (§8.2).

**Dependencies are declarative provenance** (§4.4). A factory **may compose lower-level expressions
internally** — e.g. `xs_rank_mom_252_21()` is a convenience bound feature that composes momentum + rank,
distinct from the generic `xs_rank(expr)` transform — but `compute` does not schedule a DAG unless a
future optimizer adds one.

---

## 6. Family + horizon model

Families (`Family` enum, one module each): `returns` · `trend` · `momentum` · `volatility` ·
`volume` · `distribution` · `mean_reversion` · `seasonality` · `cross_sectional`.

`microstructure` is in the enum but **ships in a later minor version** — v1 is bars-only; the tier
machinery (§8.3 / §4.1 `data_tier`) is ready so it slots in without touching v1 families.

Horizon bands (`Horizon`), default lookback grids (trading bars): `MICRO` ticks→min; `INTRADAY`
{12,26,78}; `SHORT` {3,5,10}; `MEDIUM` {21,63,126}; `LONG` {126,252,504}.

---

## 7. API & module layout

Flat functional API per family, plus an explicitly-constructed, **frozen** registry.

```
sabia/
  spec.py        # FeatureSpec, BoundFeature, Family, Horizon, DataTier, Recurrence, Cost, NullPolicy,
                 #   Evidence, Unit, ValidationMode, Adjustment (re-exported)
  typing.py      # InputRole = PriceRole | VolumeRole | FactorRole | CalendarRole; Adjustment; FeatureRef
  schema.py      # BarSchema (role→column, closed_col, calendar, symbol_col, timestamp_col), .column(role)
  params.py      # FrozenParams + validation
  references.py  # structured Citation
  naming.py      # naming() (Rule A) + assert_unique collision guard
  registry.py    # bind_feature; Registry.from_modules([...]).freeze(); .where(...); .available(tier)
  manifest.py    # FeatureSetManifest / FeatureRef / TransformRef (+ role map, dependency DAG)
  validate.py    # input contract, ValidationMode (§8.3)
  normalize.py   # transforms + TransformSpec + BoundTransform
  calendar.py    # SessionCalendar protocol + dependency-free UtcCalendar + get_calendar
  adapters/      # optional: quant_features shim; future quando_calendar adapter
  returns.py  trend.py  momentum.py  volatility.py  volume.py
  distribution.py  mean_reversion.py  seasonality.py  cross_sectional.py
  # microstructure.py — deferred (§6)
```

`Registry.default()` is pure: immutable, built from a deterministic static module list, frozen
immediately. Built-ins aren't overridable in place; a different impl is a different `(name, version)`.

---

## 8. Production readiness — batch-first, online-ready

The resolved `pl.Expr` is the canonical **batch** definition; any online implementation
(tail-recompute or stateful) is **validated against it** (§9).

### 8.1 Windowed-recompute guarantee
Recomputing on the last `min_history` bars reproduces the full-history value at `t` (for covered
classes, §8.2). Minimal online engine: buffer the declared history, recompute on each closed bar,
take the last value. Cross-sectional features need the full *panel* over the window.

### 8.2 Recurrence taxonomy
- **`FINITE`** — bounded window; tail-recompute exact (`min_history` = window).
- **`RECURSIVE_DECAY`** — decaying memory; exact within tolerance after `effective_warmup =
  ceil(ln(tol)/ln(1-alpha))` (`min_history` = `effective_warmup`, §4.5).
- **`PATH_DEPENDENT`** — resets/triggers (SAR, CUSUM); explicit state replay; parity is replay-based.
- **`EXPANDING`** — unbounded cumulative (raw OBV, A/D); **banned in v1**; ship differenced/bounded.

v1 ships only `FINITE` + `RECURSIVE_DECAY`, so the guarantee covers the entire v1 surface. The
registry rejects `PATH_DEPENDENT`/`EXPANDING` specs in v1.

### 8.3 Input contract & validation modes
`sabia.validate(frame, schema=…, universe=…, membership=…, mode=ValidationMode.STRICT)`:

- **`STRICT`** (default) — raises on any contract violation.
- **`RESEARCH`** — warns on completeness/finalization only; **still raises** on schema, dtype, role,
  ordering, duplicate-timestamp, or unsorted-panel violations.
- **`OFF`** — no validation.

`compute(..., validation=…)` uses the same enum (one vocabulary). **Bars-closed is checked via
`closed_col`/`is_final` or an upstream-certified frame flag — never wall-clock time** (preserves the
no-clocks invariant). Core default is always `STRICT`. `validate` returns warnings as a `list[str]`
(no logging in core); the caller decides policy.

---

## 9. Testing (production gates — extends `testing.md`)

Cross-cutting tests are **parametrized per bound feature** (`ids=spec.name`); floats via
`pytest.approx`. Per-feature gates: causality; windowed-recompute parity (by recurrence class);
reference values (hand-checked `(input, expected)` tables, **not** snapshots); degenerate inputs;
property-based (`hypothesis`: scale/shift, null propagation, no-pandas guard, dtype, **fingerprint
changes with bound params/roles**); determinism. Marked separately: reproduction calibration (`integration`,
local sample, no network); performance (`pytest-benchmark`, `slow`).

Data & invariance gates: symbol isolation (duplicate a symbol, perturb one, assert the other
unchanged; + shuffled order); input-order invariance; duplicate-timestamp & OHLC-ordering rejection;
role-misuse (`@tr` vs `@split` vs `@raw`); name collision; manifest round-trip (incl. role map +
dependency DAG); registry immutability; universe completeness; all-null/sparse-null rolling; integer
division/dtype; **eager-vs-lazy parity**; **chunked-vs-rechunked parity**; `ValidationMode` semantics
(RESEARCH warns on completeness but raises on schema/dtype/role/order).

---

## 10. Performance contract

Resolved expressions are vectorized Polars; `LazyFrame` fuses them; cross-sectional ops use
`.over(...)`; no materialized per-symbol frames. **Escape hatches** (for `HEAVY` features that can't
be pure expressions — rolling OLS, variance ratio, Roll/Corwin-Schultz, beta/idio-vol): NumPy kernels
over contiguous arrays, or `rolling_map`/`map_batches`, **only** for `HEAVY` features, **never**
per-row Python, and **must** carry eager-vs-lazy parity and benchmark coverage.

---

## 11. Decisions (resolved for v1)

- **Name** `sabia`. **Adjustment** part of role identity (`field@adjustment`); sabia adjusts nothing (§2.2).
- **Roles** schema maps columns → roles; signatures take `*Role`, not strings. **Bound objects**
  factories return `BoundFeature`; `.expr(schema)` resolves roles (§4.2) — purity preserved.
- **Indexing** observation-time, not tradability (§2.3). **Factor PIT** `market_ret` obeys the same
  contract (§2.3). **Bars-closed** via `closed_col`, never wall-clock (§8.3).
- **Naming** Rule A + `mom_{formation}_{skip}` + collision guard. **Recurrence** four classes; v1 =
  `FINITE` + `RECURSIVE_DECAY`. **`min_history`** = window (finite) / `effective_warmup` (decay).
- **Validation** `ValidationMode.STRICT|RESEARCH|OFF`, strict default. **Units** enum tag;
  **dependencies** manifest metadata, not a runtime DAG. **Calendar** one per frame/manifest in v1;
  internal `UtcCalendar` ships now, `quando` exchange-calendar adapter later (not a v1 runtime dep).
- **Microstructure / online / options** deferred.

---

## 12. v1 feature inventory

Bars-only (+ `market_ret`), `FINITE`/`RECURSIVE_DECAY`. **Tier 1.0** ships first (no rolling-regression
kernels); **1.1** adds estimator-heavy + the regression stack. `hurst`/`ou_halflife` **excluded** until
precisely specified. Roles use `field@adjustment` (`tr`=total-return, `split`=split-only, `raw`=raw);
`dvol@raw`=`dollar_volume@raw`; `mkt`=`market_ret`. min_history at `tol=1e-6` for decay.

> **The shipped library is a superset of this table.** It additionally ships drifted extras that
> predate the §12 contract and have no §12 equivalent — `ret_simple_1`, `adx_14`, `half_life_60`,
> `adv_21`, `signed_vol_21`, `dollar_vol`, and the price-level `sma_*`/`ema_*` — retained and
> documented as "beyond §12". They obey the same invariants and tests.

### returns — `FINITE`, unit `LOG_RETURN` (drawdown `RATIO`)
| Feature | Definition | Params | Roles | min_hist | Evidence | Tier |
|---|---|---|---|---|---|---|
| `ret_log_1/5/21/252` | `ln(close / close.shift(n))` | n | close@tr | n+1 | FORMULA_ONLY | 1.0 |
| `ret_overnight` | `ln(open / close.shift(1))` | – | open@tr, close@tr | 2 | FORMULA_ONLY | 1.0 |
| `ret_intraday` | `ln(close / open)` | – | open@tr, close@tr | 1 | FORMULA_ONLY | 1.0 |
| `drawdown_252` | `close / max(close,252) − 1` | 252 | close@tr | 252 | FORMULA_ONLY | 1.0 |

### trend
| Feature | Definition | Params | Roles | min_hist | Rec/Unit | Evidence | Tier |
|---|---|---|---|---|---|---|---|
| `sma_dist_50` | `close/SMA−1` | 50 | close@tr | 50 | FINITE/RATIO | TA_CANON | 1.0 |
| `ema_dist_50` | `close/EMA−1` | 50 | close@tr | 346 | DECAY/RATIO | TA_CANON | 1.0 |
| `dist_52w_high` | `close/max(close,252)−1` | 252 | close@tr | 252 | FINITE/RATIO | ACADEMIC_REPLICATED | 1.0 |
| `price_pctile_252` | percentile of close in window | 252 | close@tr | 252 | FINITE/RANK_0_1 | FORMULA_ONLY | 1.0 |
| `ols_slope_63` | OLS slope `ln close ~ t` (HEAVY) | 63 | close@tr | 63 | FINITE/LOG_RETURN | FORMULA_ONLY | 1.1 |
| `macd_12_26_9` (+signal/hist) | EMA spread (per-output warmup: line needs only the slow EMA to converge; signal/hist need slow + signal) | 12,26,9 | close@tr | 180 / 242 / 242 (line/signal/hist) | DECAY/LOG_RETURN | TA_CANON | 1.1 |

### momentum
| Feature | Definition | Params | Roles | min_hist | Rec/Unit | Evidence | Tier |
|---|---|---|---|---|---|---|---|
| `mom_252_21` | `ln(close.shift(21)/close.shift(252))` | 252,21 | close@tr | 253 | FINITE/LOG_RETURN | ACADEMIC_REPLICATED | 1.0 |
| `roc_21` | `close/close.shift(21)−1` | 21 | close@tr | 22 | FINITE/RATIO | TA_CANON | 1.0 |
| `rsi_14` | Wilder RSI | 14 | close@tr | 187 | DECAY/INDEX_0_100 | TA_CANON | 1.0 |
| `stoch_k_14`,`stoch_d_3` | %K (14) / %D (3-SMA of %K) | 14,3 | h/l/c@split | 14 / 16 | FINITE/INDEX_0_100 | TA_CANON | 1.1 |
| `williams_r_14` | Williams %R | 14 | h/l/c@split | 14 | FINITE/INDEX_0_100 | TA_CANON | 1.1 |
| `cci_20` | typical price vs MAD(20, same window) | 20 | h/l/c@split | 20 | FINITE/UNITLESS | TA_CANON | 1.1 |

### volatility — unit `RETURN_STD_PER_BAR` (`_ann` via §4.6); `atr` `PRICE_UNITS`
| Feature | Definition | Params | Roles | min_hist | Rec | Evidence | Tier |
|---|---|---|---|---|---|---|---|
| `vol_cc_21` | std of log returns | 21 | close@tr | 22 | FINITE | FORMULA_ONLY | 1.0 |
| `vol_ewma_0p94` | RiskMetrics EWMA (λ=0.94) | 0.94 | close@tr | 224 | DECAY | ACADEMIC_SINGLE | 1.0 |
| `vol_parkinson_21` | Parkinson | 21 | h/l@split | 21 | FINITE | ACADEMIC_SINGLE | 1.0 |
| `vol_gk_21` | Garman–Klass | 21 | o/h/l/c@split | 21 | FINITE | ACADEMIC_SINGLE | 1.0 |
| `vol_rs_21` | Rogers–Satchell | 21 | o/h/l/c@split | 21 | FINITE | ACADEMIC_SINGLE | 1.0 |
| `vol_yz_21` | Yang–Zhang | 21 | o/h/l/c@split | 22 | FINITE | ACADEMIC_SINGLE | 1.0 |
| `atr_14` | Average True Range | 14 | h/l/c@split | 187 | DECAY | TA_CANON | 1.0 |
| `semivar_down_21` | realized downside semivariance | 21 | close@tr | 22 | FINITE | ACADEMIC_SINGLE | 1.0 |
| `bb_pctb_20_2`,`bb_bw_20_2` | Bollinger %B / bandwidth | 20,2 | close@split | 20 | FINITE | TA_CANON | 1.1 |

### volume / liquidity — `FINITE`
| Feature | Definition | Params | Roles | min_hist | Unit | Evidence | Tier |
|---|---|---|---|---|---|---|---|
| `vol_z_21` | volume z-score | 21 | volume@split | 21 | ZSCORE | FORMULA_ONLY | 1.0 |
| `rel_volume_21` | `volume/SMA(volume,21)` | 21 | volume@split | 21 | RATIO | FORMULA_ONLY | 1.0 |
| `amihud_21` | `mean(|ret@tr| / dvol@raw)` | 21 | close@tr, dvol@raw | 22 | RATIO | ACADEMIC_REPLICATED | 1.0 |
| `vwap_dist_close` | `close/vwap − 1` (same basis) | – | close@split, vwap@split | 1 | RATIO | FORMULA_ONLY | 1.0 |
| `cmf_21` | Chaikin Money Flow | 21 | h/l/c@split, volume@split | 21 | UNITLESS | TA_CANON | 1.1 |
| `mfi_14` | Money Flow Index | 14 | h/l/c@split, volume@split | 15 | INDEX_0_100 | TA_CANON | 1.1 |
| `roll_spread_21` | Roll serial-cov spread (HEAVY) | 21 | close@tr | 22 | RATIO | ACADEMIC_SINGLE | 1.1 |
| `spread_corwin_schultz` | Corwin–Schultz (HEAVY) | 2 | h/l@split | 2 | RATIO | ACADEMIC_SINGLE | 1.1 |

### distribution — `FINITE`
| Feature | Definition | Params | Roles | min_hist | Unit | Evidence | Tier |
|---|---|---|---|---|---|---|---|
| `skew_21`,`kurt_21` | rolling moments of returns | 21 | close@tr | 22 | UNITLESS | FORMULA_ONLY | 1.0 |
| `downside_dev_21` | downside deviation | 21 | close@tr | 22 | RETURN_STD_PER_BAR | ACADEMIC_SINGLE | 1.0 |
| `up_down_vol_ratio_21` | upside/downside vol ratio | 21 | close@tr | 22 | RATIO | FORMULA_ONLY | 1.1 |

### mean_reversion (memory) — `FINITE`
| Feature | Definition | Params | Roles | min_hist | Unit | Evidence | Tier |
|---|---|---|---|---|---|---|---|
| `zscore_close_21` | distance from rolling mean | 21 | close@tr | 21 | ZSCORE | FORMULA_ONLY | 1.0 |
| `autocorr_1_21` | lag-1 return autocorrelation | 1,21 | close@tr | 23 | UNITLESS | FORMULA_ONLY | 1.0 |
| `var_ratio_2_21` | Lo–MacKinlay variance ratio (HEAVY) | 2,21 | close@tr | 23 | UNITLESS | ACADEMIC_REPLICATED | 1.1 |

*(`hurst_*`, `ou_halflife_*` excluded — estimator-ambiguous; OU only on spreads/demeaned series, with a pinned estimator.)*

### seasonality — `FINITE`, **exchange/session** calendar (`SessionCalendar`), not UTC wall-clock
| Feature | Definition | Params | Roles | min_hist | Unit | Tier |
|---|---|---|---|---|---|---|
| `season_dow` | session weekday, Monday=0 | – | session | 1 | UNITLESS | 1.0 |
| `season_tom_k` | last session of month + first k sessions | k | session | 1 | UNITLESS | 1.0 |

### cross_sectional — `requires_complete_panel`, `requires_universe`; ranks ascending (high mom → high rank)
| Feature | Definition | Params | Roles | min_hist | Unit | Evidence | Tier |
|---|---|---|---|---|---|---|---|
| `xs_rank_mom_252_21` | percentile rank of `mom_252_21` | 252, 21 | close@tr | 253 | RANK_0_1 | ACADEMIC_REPLICATED | 1.0 |
| `xs_z_mom_252_21` | cross-sectional z of momentum (winsorized) | 252, 21 | close@tr | 253 | ZSCORE | ACADEMIC_REPLICATED | 1.0 |
| `rev_1m_21` | short-term reversal (−`ret_log_21`, ranked) | 21 | close@tr | 22 | RANK_0_1 | ACADEMIC_REPLICATED | 1.0 |
| `beta_252` | OLS slope `ret_log_1 ~ mkt` (intercept; HEAVY) | 252 | close@tr, mkt | 253 | UNITLESS | ACADEMIC_REPLICATED | 1.1 |
| `idio_vol_252` | per-bar std of residuals vs `mkt` (HEAVY) | 252 | close@tr, mkt | 253 | RETURN_STD_PER_BAR | ACADEMIC_REPLICATED | 1.1 |

*`mkt` is a `FactorRole` (a PIT-valid, session-aligned return series, §2.3), not an OHLCV column.*

---

## 13. Roadmap — post-v1 add-ons

> Candidate features for a future version — **not v1 build targets**. By `Family`, each with a
> definition and literature anchor; on graduation each gets a bound `FeatureSpec`, the §9 tests, and
> §3 invariants. Many are `PATH_DEPENDENT`/`EXPANDING` (§8.2) needing replay/differenced forms.

### trend
- **ADX / DMI** — trend strength independent of direction. `RECURSIVE_DECAY`. *Wilder (1978).*  ·
  **Aroon** *(Chande 1995)*  · **Trend R²** `FINITE`  · **TRIX** `RECURSIVE_DECAY` *(Hutson 1983)*  ·
  **Parabolic SAR distance** `PATH_DEPENDENT` *(Wilder 1978)*.

### momentum
- **Ultimate Oscillator** *(Williams 1985)*  · **Momentum acceleration** `FINITE`  ·
  **Frog-in-the-pan** (information discreteness) *(Da, Gurun & Warachka 2014)*.

### volatility
- **Ulcer Index** *(Martin & McCann 1989)*  · **Chaikin Volatility** `RECURSIVE_DECAY`  ·
  **Realized range** (`MINUTE`/`INTRADAY`)  · **Vol term-structure ratio** `FINITE`.

### volume
- **Volume Profile / VPOC / value area** (`MINUTE`/`INTRADAY`)  · **A/D line** `EXPANDING` → ship
  **Chaikin oscillator** *(Williams)*  · **Force Index** `RECURSIVE_DECAY` *(Elder 1993)*  ·
  **Ease of Movement** *(Arms)*  · **Klinger Oscillator** `RECURSIVE_DECAY`.

### options (implied / derivatives) — blocked
> **Blocked — sabia does not ingest options data yet.** Waits on chain / IV-surface ingestion upstream.
- **ATM IV**, **variance risk premium** *(Bollerslev–Tauchen–Zhou 2009)*, **IV skew/smirk**
  *(Xing–Zhang–Zhao 2010)*, **risk reversal**, **IV term-structure slope**, **model-free implied
  skew/kurtosis** *(Bakshi–Kapadia–Madan 2003)*, **vol-of-vol**, **put–call ratio**, **implied
  correlation** (panel), **dealer gamma exposure** (`INTRADAY`), **IV rank/percentile**.

### distribution
- **Realized skewness** (`INTRADAY`) *(Amaya et al. 2015)*  · **Tail ratio**  · **Co-skewness /
  co-kurtosis** (needs `mkt`) *(Harvey & Siddique 2000)*.

### mean_reversion (memory & regime)
- **Choppiness Index** *(Dreiss)*  · **Sample / permutation entropy**  · **DFA** *(Peng et al. 1994)*  ·
  **CUSUM filter** `PATH_DEPENDENT` *(López de Prado 2018)*  · **SADF** bubble test `HEAVY`
  *(Phillips–Shi–Yu 2015)*  · **Average pairwise correlation** (panel)  · **Hurst / OU half-life**
  (precise estimator; OU on spreads only).

### cross_sectional
- **MAX (lottery)** *(Bali–Cakici–Whitelaw 2011)*  · **Downside beta / semibeta** (needs `mkt`)
  *(Ang–Chen–Xing 2006)*  · **Pastor–Stambaugh liquidity** *(Pastor & Stambaugh 2003)*.

### microstructure (later release; `TICK`/L1-L2)
- **Hasbrouck information share** *(1995)*  · **PIN** *(Easley et al.)*  · **order-book/quote
  imbalance**  · **trade-sign autocorrelation**  · **odd-lot ratio**.

### normalize (transform add-ons; extend §5)
- **Winsorization/clipping**  · **time-series self-percentile**  · **sector/industry neutralization**
  (group key)  · **factor residualization**.

### Out of scope even post-v1
- **VaR/CVaR/ES** → `ruin`  · **GARCH conditional vol** (stateful fitted model)  · **HMM/regime
  state** (stateful inference)  · **Turnover** (needs PIT shares-outstanding).
