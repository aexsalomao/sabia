# API reference

sabia exposes a flat, functional API: one module per feature family, plus a constructable registry,
a normalization layer, a small ergonomic toolkit, and the types and helpers that bind them together.
This page is the map; the [concepts guide](concepts.md) explains the model behind it.

## Compute

| Symbol | Signature | Purpose |
|---|---|---|
| `sabia.compute` | `compute(frame, *features, schema, validation=STRICT, universe=None, membership=None, include_keys=False) -> pl.DataFrame` | Validate, resolve roles, and materialize bound features into a DataFrame. |
| `sabia.compute_lazy` | `compute_lazy(frame, *features, schema) -> pl.LazyFrame` | Fused lazy `select` of time-series features. No validation, no XS features. |

```python
df = sabia.compute(frame, sabia.momentum.rsi(period=14), schema=schema, include_keys=True)
lf = sabia.compute_lazy(frame, sabia.returns.ret_log(period=1), schema=schema)
```

## Schema & roles

| Symbol | Purpose |
|---|---|
| `BarSchema` | Maps column **roles** to physical columns. `BarSchema.ohlcv(...)` is the shorthand; `BarSchema(roles={...})` is the explicit form. `.column(role)`, `.has(role)`. |
| `PriceRole(field, adjustment)` / `VolumeRole(...)` | Role keys. `field` is a `PriceField`/`VolumeField`; `adjustment` is an `Adjustment` (`TR` total-return / `SPLIT` split-only). |
| `PriceField` / `VolumeField` | Canonical fields — `OPEN`, `HIGH`, `LOW`, `CLOSE`; `VOLUME`. |
| `Adjustment` | Adjustment basis: `TR`, `SPLIT`, `RAW`. |
| `InputRole` / `CalendarRole` / `FactorRole` | Role protocol and the non-price role kinds (calendar, market factor). |

```python
from sabia import BarSchema, PriceRole, PriceField, Adjustment
schema = BarSchema.ohlcv(tr_close="adj_close")
schema = BarSchema(roles={PriceRole(PriceField.CLOSE, Adjustment.TR): "adj_close"})
```

## Feature families

Each family is a module of pure factories; each factory takes **keyword-only** params and returns a
`BoundFeature` (`.spec` + `.expr(schema) -> pl.Expr`).

`returns` · `trend` · `momentum` · `volatility` · `volume` · `distribution` · `mean_reversion` ·
`seasonality` · `cross_sectional`

```python
sabia.momentum.rsi(period=14)            # -> BoundFeature
sabia.volatility.vol_yz(window=21)       # -> BoundFeature
sabia.cross_sectional.beta(window=252)   # -> BoundFeature (needs a market_ret role + universe)
```

See the [feature catalog](catalog.md) for every shipped feature and its params. `microstructure`
exists in the `Family` enum but ships in a later minor version (v1 is bars-only).

## Normalization

Trailing and cross-sectional transforms that preserve the causality invariant. Each returns a
`BoundTransform` (`.spec` + `.apply(expr) -> expr`).

| Factory | Purpose |
|---|---|
| `normalize.zscore(window, *, over=None)` | Trailing rolling z-score; flat window → `null`. |
| `normalize.xs_zscore(*, winsorize=None, over="timestamp")` | Cross-sectional z within each slice; optional pre-winsorize. |
| `normalize.xs_rank(*, over="timestamp")` | Cross-sectional percentile rank in `(0, 1]`. |
| `normalize.frac_diff(d, *, threshold=1e-6, max_lag=100, over=None)` | Fixed-width fractional differencing (FFD). |

```python
z = sabia.normalize.zscore(window=63)
expr = z.apply(sabia.trend.ols_slope(window=63).expr(schema))
```

## Registry

| Symbol | Purpose |
|---|---|
| `Registry` | Constructable feature registry. `Registry.default()` assembles the shipped set. |
| `Registry.where(pred)` | Subset by a predicate over `FeatureSpec`. |
| `Registry.available(tier)` | Features computable on a given `DataTier`. |
| `Registry.specs()` | All `FeatureSpec`s. |
| `FeatureSpec` | Frozen per-feature metadata (see the [concepts guide](concepts.md#the-registry-featurespec)). |
| `BoundFeature` | A bound factory: `.spec` + `.expr(schema)`. |
| `bind_feature(...)` | Low-level binding helper used by family factories. |
| `FrozenRegistryError` | Raised on mutation of a frozen registry. |

```python
reg = sabia.Registry.default()
reg.where(lambda s: sabia.Horizon.MEDIUM in s.native_band)
reg.available(sabia.DataTier.DAILY)
```

## Toolkit & recipes

Ergonomic, opinion-free helpers over collections of bound features.

| Symbol | Purpose |
|---|---|
| `FeatureSet(features)` | Immutable bundle; `.compute(...)`, `.manifest(...)`, `.names()`, `.required_roles()`. |
| `describe(feature) -> str` | Readable one-feature card from the spec. |
| `drop_warmup(frame, features, *, symbol_col=None)` | Trim leading warm-up rows. |
| `max_min_history(features) -> int` | Largest `min_history` across a set. |
| `required_roles(features)` / `required_columns(features, schema)` | Pre-flight role / column sets. |
| `recipes.daily_core()` / `volatility_core()` / `cross_sectional_core()` | Preset `FeatureSet` bundles (panels, not strategies). |

## Validation

| Symbol | Purpose |
|---|---|
| `validate(frame, *, schema, ...)` | Enforce the input contract; raises `SabiaValidationError`. |
| `audit_frame(frame, *, schema, features=())` | Non-fatal health report → `FrameAudit`. |
| `ValidationMode` | `STRICT` (default) / `RESEARCH` / `OFF`. |
| `SabiaValidationError` | Raised on a hard contract violation. |
| `FrameAudit` | Counts-and-ranges report returned by `audit_frame`. |

## Reproducibility

| Symbol | Purpose |
|---|---|
| `FeatureSetManifest` | Pins a feature set's fingerprints + schema. `FeatureSetManifest.of(...)`, `.to_json()`, `.from_json()`. |
| `TransformRef` | A pinned reference to a normalization transform inside a manifest. |
| `Citation` / `Reference` | Literature references attached to features. |
| `naming` / `assert_unique` | Feature-naming grammar and uniqueness check. |
| `FrozenParams` | Immutable param container used in fingerprints. |

## Calendars

| Symbol | Purpose |
|---|---|
| `get_calendar(code)` | Resolve an exchange/session calendar by code. |
| `SessionCalendar` / `UtcCalendar` | Session calendar protocol and the default UTC implementation. |

## Enums

| Enum | Members |
|---|---|
| `Family` | the 9 feature families (+ `MICROSTRUCTURE`, enum-only in v1) |
| `Horizon` | `MICRO`, `INTRADAY`, `SHORT`, `MEDIUM`, `LONG` |
| `DataTier` | `TICK` < `MINUTE` < `DAILY` (finer input unlocks more features) |
| `Recurrence` | `FINITE`, `RECURSIVE_DECAY` (shipped in v1) |
| `Cost` | `O1`, `LINEAR`, `HEAVY` |
| `Unit` | `LOG_RETURN`, `RATIO`, `INDEX_0_100`, `UNITLESS`, `RETURN_STD_PER_BAR`, `PRICE_UNITS`, `RANK_0_1`, `ZSCORE` |
| `Evidence` | `FORMULA_ONLY`, `TA_CANON`, `ACADEMIC_SINGLE`, `ACADEMIC_REPLICATED` |
| `NullPolicy` | Window-null policy attached to a spec |

See `FEATURES.md` for the full contract and `CLAUDE.md` for the authoring rules.
