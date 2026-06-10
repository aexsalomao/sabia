# Changelog

All notable changes to `sabia` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-06-10

The post-v1 intraday microstructure tier (FEATURES.md §13): a pure tick→bar adapter edge layer and
the microstructure feature family that runs on the bars it produces. Additive — the daily features,
unchanged in formula, are untouched (a few fingerprints shift only because shared rolling-moment
helpers moved to `_math.py`; the math is byte-identical, so versions stay at 1).

### Added

- **`sabia.adapters.build_bars`** — a pure, lazy `build_bars(ticks, BarSpec)` transform that
  aggregates raw trade/quote ticks into intraday time / volume / dollar / tick bars, signing trades
  (tick rule / Lee–Ready) at aggregation time. Deterministic on tied timestamps (stable sort +
  tie-atomic bucketing) and point-in-time correct: each bar carries a `closed` marker so the
  in-progress trailing bar is never mistaken for final.
- **`microstructure` family** (MINUTE tier) — realized volatility (`rvar`, `bipower`, `jump_rj`,
  `rsemivar_up`/`dn`, `signed_jump`, `rskew`, `rkurt`), order flow (`trade_imbalance`,
  `sign_autocorr`, `vpin`), and liquidity (`quoted_spread`, `eff_spread`, `amihud_intraday`,
  `kyle_lambda`, `depth_imbalance`). `book_imbalance` (L2) ships as an unregistered factory.
- **`validate_ticks`** — the raw-tick input contract (non-decreasing timestamps allow ties,
  positive price, non-negative size, uncrossed quotes), run in one pass.
- **`QuoteRole` / `FlowRole` / `DepthRole`** input roles and **`BarSchema.trades()` / `.quotes()`**
  constructors mapping the adapter's output columns (including the `closed` marker by default).
- **`PATH_DEPENDENT`** recurrence is admitted (`EXPANDING` stays banned), with replay-based
  windowed-recompute parity coverage.

### Fixed

- `bipower` / `jump_rj`: bipower variation now sums `window - 1` adjacent products over the same
  `window` returns realized variance uses, so the jump statistic compares like for like.
- `quoted_spread`: a crossed quote (`bid > ask`) now yields null as documented, instead of leaking a
  negative spread into the rolling mean on the validation-off path.
- `build_bars`: fails loudly with `SabiaValidationError` on a symbol-less feed instead of a raw
  Polars error from inside the lazy plan.
- `sabia.__version__` is synced with `pyproject.toml` and locked by a test (was stale at `0.3.0`).

## [0.4.0] - 2026-06-02

Acting on an internal code review: correcting features whose formulas diverged from the FEATURES.md
§12 contract or their cited literature, plus closing the §9 production-gate test coverage. The
corrected formulas change output values and several feature fingerprints, so the pinned manifest is
regenerated in the same change (breaking, pre-1.0).

### Changed (breaking — corrected feature output & fingerprints)

- **`cci_20`** now uses the canonical single-window mean-absolute-deviation (was a trailing-mean
  variant); `min_history` 39 → 20 per §12.
- **`amihud_21`** uses the absolute log return (was a simple return), matching §4.6 and the
  library-wide `ret@tr` basis.
- **`var_ratio_2_21`** is now the Lo–MacKinlay bias-corrected estimator (matches the citation);
  `min_history` 22 → 23.
- **`cmf_21`** treats a flat (doji) bar as a zero money-flow contribution instead of nulling the
  whole window.
- **`season_tom_k`** flags the last session of the month plus the first `k` sessions (was a
  symmetric first-`k`/last-`k` window).
- **`xs_z_mom_252_21`** winsorizes within each timestamp slice (±3σ) before standardizing (§4.6).
- Float feature params are canonicalized via `float.hex()` for platform-stable fingerprints (§3.5),
  so `vol_ewma_0p94`, `bb_pctb_20_2`, `bb_bw_20_2` (and other float-param features) get new
  fingerprints. The pinned manifest is regenerated to match.
- `evaluate` is no longer re-exported at the top level (`sabia.evaluate`); it remains available as
  `sabia.registry.evaluate`.

### Added

- **`winsorize=`** parameter on the `normalize.xs_zscore` transform (default `None`, no behavior
  change for existing callers).
- §9 production-gate tests: eager-vs-lazy parity, chunked-vs-rechunked parity (both registry-
  parametrized, covering the HEAVY/`rolling_map` kernels), `ValidationMode` RESEARCH/OFF semantics,
  `validate()` OHLC-ordering rejection, symbol-isolation perturb-twin, input-order invariance,
  registry immutability, all-null inputs, float-param fingerprint stability, and a deterministic
  `future_bars=1` causality boundary.

### Fixed

- Cross-sectional momentum signals route through `safe_log` (no-NaN consistency, §4.5).
- `frac_diff` uses a tighter weight-truncation threshold for slow-decaying long memory.
- `BarSchema.ohlcv()` documents that, without `tr_close`, `close@tr`/`open@tr` are backed by the
  split-only columns (a silent basis conflation on dividend-paying instruments).
- `FEATURES.md` §12 reconciled with the code (macd per-output warmups 180/242/242, `var_ratio`
  `min_history` 23, `xs_rank` percentile range `(0,1]`).

## [0.3.0] - 2026-06-02

Acting on an external audit: closing gaps where the code did not honor its own contracts, plus
ergonomics. Breaking renames are pre-1.0 churn (the manifest lock is regenerated in the same change).

### Added

- **Manifest serialization.** `FeatureSetManifest.to_dict` / `from_dict` / `to_json` / `from_json`
  (tagged with `MANIFEST_SCHEMA_VERSION`) — the manifest now actually round-trips, as its docstring
  always claimed.
- **As-of universe membership.** `validate(..., membership=df)` / `compute(..., membership=df)` accept
  a `(symbol, start, end)` frame; completeness is then checked point-in-time
  (`{symbol : start <= t < end}`) instead of against a single static symbol count. Replaces the
  previously accepted-but-ignored `membership_asof`.
- **`compute(..., include_keys=True)`** prepends `symbol`/`timestamp`, aligned row-for-row.
- **`BarSchema.ohlcv(...)`** convenience constructor for the common plain-OHLCV case.
- **Parameter guards.** Factories now reject out-of-domain params at bind time (`period`/`window`/
  `span` ranges, `0 < lam < 1`, `skip < formation`, `fast < slow`, …) via `sabia._validate_params`.
- **Ergonomic toolkit** (`sabia.toolkit`, re-exported at top level): `FeatureSet`, `describe`,
  `required_roles` / `required_columns`, `max_min_history` / `drop_warmup`, and `audit_frame`
  (a non-raising `FrameAudit` report). New `sabia.recipes` (`daily_core`, `volatility_core`,
  `cross_sectional_core`) returning `FeatureSet` bundles — features, never strategies.
- **Docs.** Runnable README quickstart with real output; a generated `docs/catalog.md`; a
  Reproducibility note explaining the Polars pin / 3.13 floor and the best-effort fingerprint.

### Changed (breaking — feature renames)

- `xs_rank_mom_252` → `xs_rank_mom_252_21`, `xs_z_mom_252` → `xs_z_mom_252_21` (the `skip` param is
  now encoded in the name).
- `vol_ewma_94` → `vol_ewma_0p94` (a stable lambda token that never rounds two lambdas together).
- `ret_simple` → `ret_simple_1` (`ret_simple` gains a `period`, encoded like `ret_log`).
- `ret_overnight` / `ret_intraday` now encode a non-`@tr` adjustment basis in the name (defaults
  unchanged). `compute_lazy` documents that it does not validate.

## [0.2.0] - 2026-06-02

### Changed — v5 architecture (breaking; pre-1.0)

The library moves from the v2.1 contract (factories return `pl.Expr`, take string column names) to
the **v5** contract (`FeaturesSpec.md`). This is a ground-up redesign; nothing was published, so the
break is acceptable pre-1.0.

- **Roles, not columns.** Feature inputs are now `field@adjustment` roles (`close@tr`, `high@split`)
  resolved against a caller-supplied `BarSchema`. New `sabia.typing` (`Adjustment`, `PriceRole`,
  `VolumeRole`, `FactorRole`, `CalendarRole`, `InputRole`, `FeatureRef`) and `sabia.schema.BarSchema`.
- **Bound features.** A factory binds params and returns a `BoundFeature` owning an immutable `.spec`
  and `.expr(schema) -> pl.Expr`; it never returns a raw expression. `make_feature` →
  `registry.bind_feature`; `RegisteredFeature` → `BoundFeature`.
- **Richer `FeatureSpec`.** `inputs: frozenset[Column]` → `input_roles: frozenset[InputRole]`; adds
  `null_policy` (`NullPolicy`), `output_unit`/`output_range` (`Unit`), `evidence` (`Evidence`),
  `dependencies`, `requires_universe`/`requires_complete_panel`; `citation: str` → `Citation`
  (`sabia.references`); `params: Mapping` → `FrozenParams` (`sabia.params`). `Recurrence` gains
  `RECURSIVE_DECAY` (was `RECURSIVE`), `PATH_DEPENDENT`, `EXPANDING`; v1 ships only `FINITE` +
  `RECURSIVE_DECAY` (the registry rejects the other two).
- **Fingerprint algorithm change.** `feature_fingerprint` now hashes
  `canonical_id + version + bound params + input_roles + dependency fingerprints + polars pin` plus
  the transitive closure of the `build(schema)` helpers — so `rsi_14` (`close@tr`) and `rsi_raw_14`
  (`close@raw`) are provably distinct despite identical source. **All prior fingerprints are
  invalidated** (regenerated in `tests/feature_manifest.py`).
- **Validation modes.** `sabia.validate(frame, schema=…, mode=ValidationMode.STRICT|RESEARCH|OFF)`;
  adds an OHLC-ordering check after role resolution; returns warnings as a `list[str]` (no logging in
  core). `compute(..., validation=…)` shares the enum.
- **Transforms are pinned.** `sabia.normalize` transforms return a `BoundTransform` carrying a
  `TransformSpec` so the manifest pins transforms too.
- **Manifest.** New `sabia.manifest.FeatureSetManifest` / `TransformRef` pinning bound features +
  transforms + role map + calendar + dependency DAG + tool versions.
- **Calendar seam.** New dependency-free `sabia.calendar` (`SessionCalendar` protocol + `UtcCalendar`
  + `get_calendar`). Annualization uses `bars_per_year()`, never a hardcoded 252 in feature code.
  `quando` is a *future* adapter, not a v1 runtime dependency.
- **Naming grammar.** New `sabia.naming` (`naming()` Rule A + `assert_unique`).

### Added — inventory (66 bound features, the full §12 set plus documented extras)

- Migrated all prior families to the v5 contract and rounded out the §12 set: `ret_log_1/5/21/252`,
  `ret_simple`, `ret_overnight`, `ret_intraday`, `drawdown_252`; `sma_dist_50`, `ema_dist_50`,
  `dist_52w_high_252`, `price_pctile_252`, `ols_slope_63`, `macd_12_26_9` (+signal/hist);
  `mom_252_21`, `roc_10/21`, `rsi_14`, `williams_r_14`, `stoch_k_14`, `stoch_d_14_3`, `cci_20`;
  `vol_cc_21/63`, `vol_ewma_94`, `semivar_down_21`, `vol_parkinson_21`, `vol_gk_21`, `vol_rs_21`,
  `vol_yz_21`, `atr_14`, `bb_pctb_20_2`, `bb_bw_20_2`; `vol_z_21`, `rel_volume_21`, `amihud_21`,
  `vwap_dist_close`, `cmf_21`, `mfi_14`, `roll_spread_21`, `spread_corwin_schultz`; `skew_21`,
  `kurt_21`, `downside_dev_21`, `up_down_vol_ratio_21`; `zscore_close_21`, `autocorr_1_21`,
  `var_ratio_2_21`; `season_dow`, `season_tom_3`; `xs_rank_mom_252`, `xs_z_mom_252`, `rev_1m_21`,
  `beta_252`, `idio_vol_252`.
- Completed the Tier-1.1 set: `bb_pctb_20_2` / `bb_bw_20_2` (Bollinger %B / bandwidth, `close@split`)
  and the market-model features `beta_252` / `idio_vol_252` (rolling CAPM beta + idiosyncratic vol,
  closed-form from rolling moments). Wires the `MARKET_RET` `FactorRole` into the test schema and
  fixtures (incl. a `market_ret` column on the marketgoblin regression panel).
- Drifted extras with no §12 equivalent (`adx_14`, `half_life_60`, `adv_21`, `signed_vol_21`,
  `dollar_vol`, price-level `sma_*`/`ema_*`, `ret_simple`) are retained and documented as "beyond §12".

### Fixed — code-review hardening

- **Cross-sectional output naming.** `evaluate`/`compute` now alias every result to the feature name;
  previously a cross-sectional reduction returned a column named `__sabia_xs_signal__`/`literal`, and
  two rank-based XS features collided by name in `compute`.
- **No inf/NaN from returns.** New `_math.log_return` (= `safe_log(safe_div(...))`) is the single
  source for close-to-close returns across `returns`/`distribution`/`mean_reversion`/`volatility`;
  `autocorr`/`var_ratio` (bare `.log()`) could emit NaN on a non-positive ratio, and
  `safe_log(c/c.shift)` could emit inf on a zero base. Both now yield null (FEATURES.md 4.5).
- **`mfi` flat window.** A window with no flow now yields null (like RSI) instead of saturating at 100.
- **Fingerprint folds module constants.** `feature_fingerprint` now hashes the *values* of module-level
  constants a builder reads by name (e.g. `_CCI_SCALE`), so retuning a literal is provable at the
  manifest gate (FEATURES.md 4.4). `_normalized_source`/`_module_constants` are cached.
- **Decay parity guarantee.** `EWM_WARMUP_TOL` tightened to `1e-8` and `PARITY_RECURSIVE_TOLERANCE`
  *derived* from it (`= 1e-6`, 100× headroom) so the burn-in always out-converges what parity
  asserts; the two constants can no longer drift apart.
- **Universe enforcement.** `compute(..., universe=…, membership_asof=…)` forwards to `validate`;
  passing a cross-sectional feature without a `universe` now raises rather than inferring membership.

### Docs

- New runnable, offline `examples/` suite: quickstart, roles & adjustment, validation modes,
  registry queries, cross-sectional / factor-model features, normalization transforms, and manifest
  reproducibility (see `examples/README.md`).

### Tests

- Reference-value and degenerate tests for the new features (`bb_pctb`/`bb_bw`, `beta`/`idio_vol`).
- Real-data regression suite (`tests/test_marketgoblin_regression.py`) over a committed offline
  Yahoo panel fixture (`tests/data/marketgoblin_panel.parquet`, regenerated by
  `generate_marketgoblin_fixture.py`): every shipped feature must be finite/non-empty on real prices,
  XS columns must be named/collision-free, ranks form a per-date permutation, truncation is causal.
- Targeted unit tests: degenerate-input null contract (`test_degenerate.py`), role/adjustment
  resolution + universe enforcement (`test_roles.py`), and fingerprint constant-folding /
  tolerance-coupling (`test_spec.py`).

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
