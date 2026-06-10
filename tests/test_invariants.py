"""Cross-cutting invariant harness (FEATURES.md 9).

Every shipped feature is auto-covered here: the tests parametrize over ``Registry.default()`` with
``ids=spec.name``, so each feature gets its own pass/fail. There is no loop or branch in a test
body -- feature selection happens in the parametrize decorators.

The windowed-recompute parity test is the same harness that would validate a future online engine;
none ships in v1. Every feature's roles resolve against the suite's canonical ``SCHEMA``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest
from conftest import assert_series_close
from hypothesis import given, settings
from hypothesis import strategies as st
from synthetic import (
    ASK,
    ASK_SIZE,
    ASK_SIZE_L,
    BID,
    BID_SIZE,
    BID_SIZE_L,
    BUY_VOLUME,
    CLOSE,
    DEPTH_LEVELS,
    DOLLAR_VOLUME,
    HIGH,
    LOW,
    MID,
    OPEN,
    SCHEMA,
    SELL_VOLUME,
    SIGNED_DOLLAR,
    SIGNED_VOLUME,
    SYMBOL,
    TRADE_COUNT,
    VOLUME,
    VWAP,
    append_future,
)

import sabia
from sabia.microstructure import book_imbalance
from sabia.registry import BoundFeature, Registry, evaluate
from sabia.spec import (
    DEFAULT_FLOAT_TOLERANCE,
    PARITY_RECURSIVE_TOLERANCE,
    Family,
    Recurrence,
)

_FEATURES = Registry.default().features()
# Off-registry factories (book_imbalance needs L2 depth columns the default tiers lack) still run
# through every cross-cutting gate below: registration controls shipping, never test coverage.
_OFF_REGISTRY = (book_imbalance(levels=DEPTH_LEVELS),)
_TS = [f for f in _FEATURES if f.spec.family is not Family.CROSS_SECTIONAL] + list(_OFF_REGISTRY)
_XS = [f for f in _FEATURES if f.spec.family is Family.CROSS_SECTIONAL]

_TS_IDS = [f.spec.name for f in _TS]
_XS_IDS = [f.spec.name for f in _XS]

# For PATH_DEPENDENT parity: replay over every bar up to a cut this far from the end of the series
# must reproduce the full-history value at the cut (FEATURES.md 8.2 -- replay-based parity).
_REPLAY_TRUNCATION = 5

_VALUE_COLUMNS = {
    OPEN,
    HIGH,
    LOW,
    CLOSE,
    VOLUME,
    VWAP,
    DOLLAR_VOLUME,
    BID,
    ASK,
    MID,
    BID_SIZE,
    ASK_SIZE,
    SIGNED_VOLUME,
    BUY_VOLUME,
    SELL_VOLUME,
    SIGNED_DOLLAR,
    TRADE_COUNT,
    *BID_SIZE_L,
    *ASK_SIZE_L,
}


def _evaluate(feature: BoundFeature, frame: pl.DataFrame) -> pl.Series:
    # Delegates to the library evaluator: single-pass for time-series, two-pass for cross-sectional.
    return evaluate(frame, feature, SCHEMA)


def _value_inputs(feature: BoundFeature) -> list[str]:
    # Physical value columns the feature reads, resolved from its declared roles via the schema.
    cols = {SCHEMA.column(role) for role in feature.spec.input_roles}
    return sorted(cols & _VALUE_COLUMNS)


# Null-propagation only applies to features that consume value columns; timestamp-only features
# (seasonality) have no value input to poison.
_TS_VALUED = [f for f in _TS if _value_inputs(f)]
_TS_VALUED_IDS = [f.spec.name for f in _TS_VALUED]


# --- time-series families ----------------------------------------------------------------------


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_emits_null_until_min_history(feature: BoundFeature, series: pl.DataFrame) -> None:
    out = _evaluate(feature, series)
    min_history = feature.spec.min_history
    warmup = out.head(min_history - 1)
    assert warmup.null_count() == min_history - 1, "partial-window value emitted before min_history"
    # The feature must produce values on valid input. Some features are conditionally null by
    # design (e.g. an OU half-life only when mean-reverting), so we don't require the first
    # post-warmup value specifically -- only that the feature isn't all-null.
    assert out.slice(min_history - 1).drop_nulls().len() > 0, "no valid values after warmup"


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_output_dtype_matches_spec(feature: BoundFeature, series: pl.DataFrame) -> None:
    out = _evaluate(feature, series)
    assert out.dtype == feature.spec.output_dtype


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
@settings(max_examples=8, deadline=None)
@given(future_bars=st.integers(min_value=1, max_value=20))
def test_causality_future_does_not_change_past(
    feature: BoundFeature, series: pl.DataFrame, future_bars: int
) -> None:
    full = _evaluate(feature, series)
    extended = _evaluate(feature, append_future(series, future_bars)).head(series.height)
    # Appending future bars must not change any past value, exactly, for any trailing feature.
    assert_series_close(extended, full, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_causality_one_future_bar_does_not_change_past(
    feature: BoundFeature, series: pl.DataFrame
) -> None:
    # Deterministic companion to the hypothesis causality test: pin future_bars=1, the tightest
    # boundary (a single new bar at t+1 must leave every value at <= t exactly unchanged).
    full = _evaluate(feature, series)
    extended = _evaluate(feature, append_future(series, 1)).head(series.height)
    assert_series_close(extended, full, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


@pytest.mark.parametrize("feature", _TS_VALUED, ids=_TS_VALUED_IDS)
def test_all_null_input_yields_all_null_output(feature: BoundFeature, series: pl.DataFrame) -> None:
    # Null every value column the feature reads. Output must be entirely null -- never an imputed
    # value and never inf/NaN (FEATURES.md 4.5: degenerate input -> null, never inf/NaN).
    nulled = series.with_columns(
        pl.lit(None, dtype=series.schema[col]).alias(col) for col in _value_inputs(feature)
    )
    out = _evaluate(feature, nulled)
    assert out.null_count() == out.len(), "all-null input produced a non-null value"


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_windowed_recompute_parity(feature: BoundFeature, series: pl.DataFrame) -> None:
    spec = feature.spec
    full = _evaluate(feature, series)
    if spec.recurrence is Recurrence.FINITE:
        window = series.tail(spec.min_history)
        expected = full.tail(1)
        rtol = 0.0
    elif spec.recurrence is Recurrence.RECURSIVE_DECAY:
        window = series.tail(spec.effective_warmup)
        expected = full.tail(1)
        rtol = PARITY_RECURSIVE_TOLERANCE
    else:  # PATH_DEPENDENT: no fixed tail reproduces t -- parity is replay-based (8.2): replaying
        # every bar up to a strict prefix cut must reproduce the full-history value AT that cut,
        # exactly (an online engine replays the prefix; batch computes the full series).
        cut = series.height - _REPLAY_TRUNCATION
        window = series.head(cut)
        expected = full.slice(cut - 1, 1)
        rtol = 0.0
    window_last = _evaluate(feature, window).tail(1)
    assert_series_close(window_last, expected, rtol=rtol, atol=DEFAULT_FLOAT_TOLERANCE)


@pytest.mark.parametrize("feature", _TS_VALUED, ids=_TS_VALUED_IDS)
def test_interior_null_propagates(feature: BoundFeature, series: pl.DataFrame) -> None:
    # Poison the input bar one past the feature's warmup and assert at least one new null appears.
    # That bar's output has just emerged (so it is unmasked), and there is still room ahead for a
    # lag-only feature's affected output to land in-frame (e.g. mom reads close.shift(252)). Any
    # silent imputation would leave the null count unchanged -- the failure mode this guards for.
    poison_row = feature.spec.min_history
    row_index = pl.int_range(pl.len())
    poisoned = series.with_columns(
        pl.when(row_index == poison_row).then(None).otherwise(pl.col(col)).alias(col)
        for col in _value_inputs(feature)
    )
    clean_nulls = _evaluate(feature, series).null_count()
    poisoned_nulls = _evaluate(feature, poisoned).null_count()
    assert poisoned_nulls > clean_nulls, "null input did not propagate -- a value was imputed"


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_no_window_bleed_across_symbols(feature: BoundFeature, panel: pl.DataFrame) -> None:
    # BBB sits between AAA and CCC; a feature that forgot .over(symbol) would pull AAA's tail.
    full = panel.lazy().select(SYMBOL, feature.expr(SCHEMA)).collect()
    in_panel = full.filter(pl.col(SYMBOL) == "BBB").to_series(1)
    # Keep the symbol column: features use .over(symbol), so the lone-symbol frame still needs it.
    alone = _evaluate(feature, panel.filter(pl.col(SYMBOL) == "BBB"))
    assert_series_close(in_panel, alone, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


# --- cross-sectional family --------------------------------------------------------------------


def _xs_with_keys(feature: BoundFeature, frame: pl.DataFrame) -> pl.DataFrame:
    # Attach the evaluated value to its timestamp/symbol keys (evaluate preserves row order).
    return frame.select("timestamp", SYMBOL).with_columns(_value=_evaluate(feature, frame))


@pytest.mark.parametrize("feature", _XS, ids=_XS_IDS)
def test_xs_output_dtype_matches_spec(feature: BoundFeature, panel: pl.DataFrame) -> None:
    out = _evaluate(feature, panel)
    assert out.dtype == feature.spec.output_dtype
    # Guard against vacuous passes: an all-null output would satisfy every structural check.
    assert out.drop_nulls().len() > 0, "cross-sectional feature produced no values"


@pytest.mark.parametrize("feature", _XS, ids=_XS_IDS)
def test_xs_causality_future_does_not_change_past(
    feature: BoundFeature, panel: pl.DataFrame
) -> None:
    timestamps = panel.get_column("timestamp").unique().sort()
    cutoff = timestamps[len(timestamps) // 2]
    keep = pl.col("timestamp") <= cutoff
    full_past = _xs_with_keys(feature, panel).filter(keep).to_series(2)
    part = _xs_with_keys(feature, panel.filter(keep)).to_series(2)
    assert full_past.drop_nulls().len() > 0, "no values to compare"
    assert_series_close(part, full_past, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


@pytest.mark.parametrize("feature", _XS, ids=_XS_IDS)
def test_xs_windowed_recompute_parity(feature: BoundFeature, panel: pl.DataFrame) -> None:
    timestamps = panel.get_column("timestamp").unique().sort()
    window_start = timestamps[-feature.spec.min_history]
    last_ts = timestamps[-1]
    at_last = pl.col("timestamp") == last_ts
    full_last = _xs_with_keys(feature, panel).filter(at_last).sort(SYMBOL).to_series(2)
    windowed = panel.filter(pl.col("timestamp") >= window_start)
    part = _xs_with_keys(feature, windowed).filter(at_last).sort(SYMBOL).to_series(2)
    assert full_last.drop_nulls().len() > 0, "no values to compare"
    assert_series_close(part, full_last, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


# --- non-parametrized invariants ---------------------------------------------------------------


def _sabia_sources() -> list[Path]:
    return list(Path(sabia.__file__).parent.rglob("*.py"))


def test_version_matches_pyproject() -> None:
    # __version__ is hand-pinned in sabia/__init__.py; manifests stamp it as provenance, so a stale
    # value attests the wrong library version (train == serve depends on it).
    import tomllib

    pyproject = Path(sabia.__file__).parents[2] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert sabia.__version__ == declared


def test_no_sabia_module_imports_pandas() -> None:
    offenders: list[str] = []
    for path in _sabia_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(m == "pandas" or m.startswith("pandas.") for m in modules):
                offenders.append(path.name)
    assert not offenders, f"pandas imported in sabia: {offenders}"
