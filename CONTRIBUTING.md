# Contributing to sabia

Thanks for your interest. sabia has a deliberately narrow scope (see `FEATURES.md` §1) and a strict
set of invariants (§2). Contributions are easiest to merge when they respect both.

## Setup

```bash
uv sync --extra dev
uv run pre-commit install
```

## The four gates

Every change must be green on all four before review:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/
uv run pytest
```

## Adding a feature

Follow the per-feature authoring loop in `CLAUDE.md`. In short, a feature is *done* only when:

1. It is a pure function returning `pl.Expr`, strictly trailing, panel-safe via `.over(symbol)`.
2. Its `FeatureSpec` is fully populated (including `fingerprint`, `min_history`, `recurrence`,
   `effective_warmup`, `cost_class`).
3. It has a **hand-checked** reference-value test (never a snapshot/golden file).
4. It has a degenerate-input test (zero-volume / flat series / log-domain → `null`).
5. It appears in the registry-parametrized causality + parity tests.

## Non-negotiables

No pandas. No I/O, clocks, or randomness in the core. No global mutable state. No dependency on
`quant_features` (the arrow points the other way). See `code-style.md` and `testing.md` for the rest.
