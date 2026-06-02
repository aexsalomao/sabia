# sabia examples

Runnable, self-contained scripts that walk through the library. Each generates its own deterministic
data (`_data.py`) — no network, no external files — so you can run any of them directly.

```bash
# from the repo root, using the project venv (or `uv run python examples/01_quickstart.py`)
python examples/01_quickstart.py
```

| Script | What it shows |
|---|---|
| `01_quickstart.py` | The core loop: `BarSchema` → bind feature factories → `compute()`. Eager and lazy. |
| `02_roles_and_adjustment.py` | Column **roles** vs names; why `close@tr` and `close@split` are distinct; precise role errors. |
| `03_validation.py` | The input contract and `ValidationMode` (STRICT / RESEARCH / OFF). |
| `04_registry.py` | `Registry.default()`; querying by horizon, family, tier, predicate; inspecting a `FeatureSpec`. |
| `05_cross_sectional.py` | Panel features: cross-sectional rank, market `beta`, `idio_vol`; the universe contract. |
| `06_normalize.py` | Normalization transforms: rolling `zscore`, cross-sectional `xs_rank`, `frac_diff`. |
| `07_manifest.py` | Fingerprints and `FeatureSetManifest` — pinning a feature set for reproducibility. |

## The mental model

sabia is a **pure feature library**. It takes canonicalized OHLCV bars and returns Polars
expressions that are strictly trailing and point-in-time correct. It does not fetch data, adjust
prices, decide tradability, or generate signals — those belong to neighbouring layers
(`marketgoblin` → **sabia** → `quale`).

A feature factory binds its parameters and returns a `BoundFeature`: an immutable `.spec` (all the
metadata) plus `.expr(schema)` that resolves column **roles** to your physical columns. `compute()`
validates the frame, resolves roles, and materializes a DataFrame; `compute_lazy()` keeps it lazy.
