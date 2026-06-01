"""Cross-cutting invariant harness (FEATURES.md 8).

Every shipped feature is auto-covered here: the tests parametrize over ``Registry.default()`` with
``ids=spec.name``, so each feature gets its own pass/fail. There is no loop or branch in a test
body -- feature selection happens in the parametrize decorators.

The windowed-recompute parity test (test 2 in the spec) is the same harness that would validate a
future online engine; none ships in v1.
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest
from conftest import assert_series_close
from hypothesis import given, settings
from hypothesis import strategies as st
from synthetic import append_future

import sabia
from sabia.registry import RegisteredFeature, Registry
from sabia.spec import (
    DEFAULT_FLOAT_TOLERANCE,
    PARITY_RECURSIVE_TOLERANCE,
    Column,
    Family,
    Recurrence,
)

_FEATURES = Registry.default().features()
_TS = [f for f in _FEATURES if f.spec.family is not Family.CROSS_SECTIONAL]
_XS = [f for f in _FEATURES if f.spec.family is Family.CROSS_SECTIONAL]

_TS_IDS = [f.spec.name for f in _TS]
_XS_IDS = [f.spec.name for f in _XS]

_VALUE_COLUMNS = {Column.OPEN, Column.HIGH, Column.LOW, Column.CLOSE, Column.VOLUME}


def _evaluate(feature: RegisteredFeature, frame: pl.DataFrame) -> pl.Series:
    return frame.lazy().select(feature.build()).collect().to_series()


def _value_inputs(feature: RegisteredFeature) -> list[Column]:
    return [c for c in feature.spec.inputs if c in _VALUE_COLUMNS]


# --- time-series families ----------------------------------------------------------------------


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_emits_null_until_min_history(feature: RegisteredFeature, series: pl.DataFrame) -> None:
    out = _evaluate(feature, series)
    min_history = feature.spec.min_history
    warmup = out.head(min_history - 1)
    assert warmup.null_count() == min_history - 1, "partial-window value emitted before min_history"
    assert out[min_history - 1] is not None, "no valid value at min_history on clean input"


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_output_dtype_matches_spec(feature: RegisteredFeature, series: pl.DataFrame) -> None:
    out = _evaluate(feature, series)
    assert out.dtype == feature.spec.output_dtype


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
@settings(max_examples=8, deadline=None)
@given(future_bars=st.integers(min_value=1, max_value=20))
def test_causality_future_does_not_change_past(
    feature: RegisteredFeature, series: pl.DataFrame, future_bars: int
) -> None:
    full = _evaluate(feature, series)
    extended = _evaluate(feature, append_future(series, future_bars)).head(series.height)
    # Appending future bars must not change any past value, exactly, for any trailing feature.
    assert_series_close(extended, full, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_windowed_recompute_parity(feature: RegisteredFeature, series: pl.DataFrame) -> None:
    spec = feature.spec
    full_last = _evaluate(feature, series).tail(1)
    if spec.recurrence is Recurrence.FINITE:
        window = series.tail(spec.min_history)
        rtol, atol = 0.0, DEFAULT_FLOAT_TOLERANCE
    else:
        window = series.tail(spec.effective_warmup)
        rtol, atol = PARITY_RECURSIVE_TOLERANCE, DEFAULT_FLOAT_TOLERANCE
    window_last = _evaluate(feature, window).tail(1)
    assert_series_close(window_last, full_last, rtol=rtol, atol=atol)


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_interior_null_propagates(feature: RegisteredFeature, series: pl.DataFrame) -> None:
    last = series.height - 1
    row_index = pl.int_range(pl.len())
    poisoned = series.with_columns(
        pl.when(row_index == last).then(None).otherwise(pl.col(col)).alias(col)
        for col in _value_inputs(feature)
    )
    out = _evaluate(feature, poisoned)
    assert out[last] is None, "null input did not propagate -- a value was imputed"


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_no_window_bleed_across_symbols(feature: RegisteredFeature, panel: pl.DataFrame) -> None:
    # BBB sits between AAA and CCC; a feature that forgot .over(symbol) would pull AAA's tail.
    full = panel.lazy().select(Column.SYMBOL, feature.build()).collect()
    in_panel = full.filter(pl.col(Column.SYMBOL) == "BBB").to_series(1)
    # Keep the symbol column: features use .over(symbol), so the lone-symbol frame still needs it.
    alone = _evaluate(feature, panel.filter(pl.col(Column.SYMBOL) == "BBB"))
    assert_series_close(in_panel, alone, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


# --- cross-sectional family --------------------------------------------------------------------


@pytest.mark.parametrize("feature", _XS, ids=_XS_IDS)
def test_xs_output_dtype_matches_spec(feature: RegisteredFeature, panel: pl.DataFrame) -> None:
    out = panel.lazy().select(feature.build()).collect().to_series()
    assert out.dtype == feature.spec.output_dtype


@pytest.mark.parametrize("feature", _XS, ids=_XS_IDS)
def test_xs_causality_future_does_not_change_past(
    feature: RegisteredFeature, panel: pl.DataFrame
) -> None:
    timestamps = panel.get_column(Column.TIMESTAMP).unique().sort()
    cutoff = timestamps[len(timestamps) // 2]
    keep = pl.col(Column.TIMESTAMP) <= cutoff
    full = panel.lazy().select(Column.TIMESTAMP, Column.SYMBOL, feature.build()).collect()
    full_past = full.filter(keep).to_series(2)
    truncated = panel.filter(keep)
    part = truncated.lazy().select(feature.build()).collect().to_series()
    assert_series_close(part, full_past, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


@pytest.mark.parametrize("feature", _XS, ids=_XS_IDS)
def test_xs_windowed_recompute_parity(feature: RegisteredFeature, panel: pl.DataFrame) -> None:
    timestamps = panel.get_column(Column.TIMESTAMP).unique().sort()
    window_start = timestamps[-feature.spec.min_history]
    last_ts = timestamps[-1]
    full = panel.lazy().select(Column.TIMESTAMP, Column.SYMBOL, feature.build()).collect()
    full_last = full.filter(pl.col(Column.TIMESTAMP) == last_ts).sort(Column.SYMBOL).to_series(2)
    windowed = panel.filter(pl.col(Column.TIMESTAMP) >= window_start)
    part = (
        windowed.lazy()
        .select(Column.TIMESTAMP, Column.SYMBOL, feature.build())
        .collect()
        .filter(pl.col(Column.TIMESTAMP) == last_ts)
        .sort(Column.SYMBOL)
        .to_series(2)
    )
    assert_series_close(part, full_last, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


# --- non-parametrized invariants ---------------------------------------------------------------


def _sabia_sources() -> list[Path]:
    return list(Path(sabia.__file__).parent.rglob("*.py"))


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
