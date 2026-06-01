# CLAUDE.md — building `sabia`

Implementation guide for Claude Code. **Read these first, in order:**

1. `FEATURES.md` — the spec: what to build, the contracts, the families, the §7 online-readiness contract.
2. `code-style.md` — how to write the code.
3. `testing.md` — how to test it.

This file is the **build order**, the **per-feature authoring loop**, and the **done gates**.
On conflict: `FEATURES.md` wins on *what*; `code-style.md` / `testing.md` win on *how*.

---

## What sabia is, in one paragraph

A pure, Polars-native technical-feature library. Functions take OHLCV (and later tick) frames and
return `pl.Expr` features that are strictly trailing and point-in-time correct. **Batch-first,
online-ready** (§7 of the spec): nothing online ships in v1, but the history-window metadata and the
parity test make a future engine a thin wrapper. Runtime deps: `polars`, `numpy`, `quando`. No pandas.

---

## Non-negotiables (spec invariants × your rules)

- **Pure.** No I/O, no clocks, no randomness, **no logging in core**, no module-level mutable state.
  Side effects live only at the edges (`sabia.adapters`, the consumer).
- **Causal + PIT.** Every feature uses only information at or before `t`. No window touches the future.
- **Typed end to end.** `mypy --strict` clean. Native types (`X | None`, `list`, `dict`). Structured
  data is a **frozen dataclass**, never a passed-around dict. Enums, never stringly-typed fields.
- **Polars expressions are the single source of truth.** `numpy` only where an expression genuinely
  can't express the math. **No pandas anywhere** — there's a `hypothesis`/import guard test for it.
- **Validate at the boundary, trust inside.** `sabia.validate(frame)` raises loudly on malformed
  input; feature bodies assume valid input and never re-check.

---

## Build order

Each step lands green (ruff + mypy + pytest) before the next.

1. **Scaffold.** `src/sabia` + mirrored `tests/`, `pyproject.toml`, ruff/mypy/pytest config,
   **pinned `polars` version** (the immutability guarantee depends on it — §3.4).
2. **`spec.py`.** `FeatureSpec` as `@dataclass(frozen=True, slots=True)`; enums `Family`, `Horizon`,
   `DataTier`, `Recurrence`, `Cost`; a `Column` enum/constants for canonical column names. No magic
   strings or numbers — lookback grids, tolerances, warmup multipliers are `SCREAMING_SNAKE` constants.
3. **`validate.py`.** The input contract (§7.3): sorted, unique, tz-aware-UTC timestamps; per-symbol
   sorted; required columns + dtypes; bars closed; complete cross-section for XS features. Raises a
   narrow, specific exception on violation. This is the only "fail loud" surface.
4. **`registry.py`.** `Registry` built by **explicit collection** — no import-time decorator that
   mutates a global. `Registry.default()` assembles the shipped features; `.where(pred)` and
   `.available(tier)` are the query surface. Embeddable and test-isolatable by construction.
5. **Cross-cutting test harness.** Causality (§8.1) and parity (§8.2) tests `parametrize` over
   `registry.specs()` with `ids=spec.name`, so every later feature is auto-covered with its own
   pass/fail. Wire this before writing features so feature #1 is born tested.
6. **`normalize.py`.** `zscore`, `xs_zscore`, `xs_rank`, `frac_diff` — same causality invariant,
   same harness. XS transforms compute within a single timestamp slice (`.over(timestamp)`).
7. **First feature end-to-end — the template.** Implement `rsi` (or `vol_yz`) fully: impl + populated
   `FeatureSpec` + reference-value table + degenerate case. **Stop and get it reviewed.** Every other
   feature copies this shape.
8. **Fan out families** in this order: `returns`, `volatility`, `momentum`, `trend`, `volume`, then
   `distribution`, `mean_reversion`, `seasonality`, `cross_sectional`. (`microstructure` is enum-only;
   it ships later.)
9. **Calibration** (§8.4) once `momentum` + `cross_sectional` exist — integration-marked, local sample.

---

## The per-feature authoring loop

A feature is **done** only when all of these hold:

- [ ] Pure function returning `pl.Expr`, strictly trailing; columns referenced via `Column` constants.
- [ ] File header comment (2–3 lines) + succinct docstring including the **literature citation**.
- [ ] `FeatureSpec` populated: `name`, `version=1`, `fingerprint`, `family`, `native_band`, `lookback`,
      `min_history`, `recurrence`, `effective_warmup` (analytic for the EWM family —
      `ceil(ln(tol)/ln(1-alpha))`; measured otherwise), `cost_class`, `data_tier`, `inputs`,
      `output_dtype`, `params`.
- [ ] **Reference-value test**: `@pytest.mark.parametrize` `(input, expected)` table, hand-checked or
      against a reference impl, floats via `pytest.approx`. **Never a snapshot file.**
- [ ] **Degenerate-input case** (parametrized): zero-volume / flat series / log-domain → `null`.
- [ ] Auto-covered by the registry-parametrized causality + parity tests (verify it appears).
- [ ] `ruff format`, `ruff check`, `mypy --strict`, `pytest` all green.

---

## Testing rules specific to sabia (on top of `testing.md`)

- **Parametrize, never loop.** Cross-cutting tests iterate features via `parametrize(ids=...)`, so a
  failure names the feature. No `for`/`if` in test bodies.
- **Floats → `pytest.approx`** with a single declared tolerance constant. Never `==`.
- **Causality test**: shift inputs forward in time, recompute, assert prior values unchanged.
- **Parity test**: recompute on the last `min_history` bars (FINITE → exact) or `effective_warmup`
  bars (RECURSIVE → approx) and assert equality with the full-history value at `t`.
- **Reference values are hand-verified tables**, not auto-captured golden files (your snapshot
  anti-pattern). They encode *what the math should produce*, decoupled from how it's computed.
- **Calibration + perf are out of the unit suite.** Calibration → `@pytest.mark.integration`, local
  fixture, no network. Perf → `pytest-benchmark`, `@pytest.mark.slow`; no hard timing asserts (flaky).
- **`hypothesis`** for invariants: `null` propagation, output dtype matches spec, scale/shift
  behavior, `fingerprint` stability, and a no-pandas import guard.

---

## Conventions resolved (so you don't rediscover them)

- **Columns** are a `Column` enum/constants; functions default to them (`close: str = Column.CLOSE`).
- **Registry** is constructed explicitly — no global singleton, no import-time registration side effects.
- **`null` on degenerate input is documented domain behavior**, not a silent fallback. Do not "fix" it
  into a raise. Loud failure belongs at `validate()`, for malformed frames only.
- **No logging in pure core.** If something is unexpected *inside* a feature, it's a bug — let it
  surface; don't swallow it.
- **`params`** is `Mapping[str, object]` on the generic spec; give a feature its own typed params
  dataclass when the params carry real structure. Avoid `Any`.
- **Smell-tests** from `code-style.md` apply: split a family module if it crosses ~300 lines; group
  >5 params into a dataclass.

---

## Out of scope for v1 — do NOT build

- Any online/streaming engine or `OnlineFeature` protocol. The §7 contract is **metadata + tests only**.
- The `microstructure` family module (it exists in the `Family` enum; the module ships later).
- Pattern recognition, Fibonacci, Elliott, Ichimoku.
- Signals, portfolio construction, backtesting, risk metrics — those are `quale` / `ruin`.
- `pandas`, in any form.
- A dependency on `quant_features` — the arrow points the other way (it consumes sabia).

---

## Definition of done

`ruff format` · `ruff check` · `mypy --strict` · `pytest` (incl. causality, parity, determinism, and
`hypothesis` gates) — all green. Calibration and benchmark suites green under their markers before a
release tag.
