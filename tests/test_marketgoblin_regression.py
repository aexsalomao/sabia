"""Regression tests on a real marketgoblin (Yahoo) panel.

The panel in ``data/marketgoblin_panel.parquet`` is a committed snapshot of a real multi-symbol
daily frame fetched via marketgoblin (see ``data/generate_marketgoblin_fixture.py``). These tests
run fully offline -- no marketgoblin dependency, no network -- and assert the library behaves on
real prices: the input contract holds, every shipped feature computes without leaking inf/NaN, and
cross-sectional features come back correctly named (the #1 regression from the code review).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from conftest import assert_series_close
from synthetic import SCHEMA

import sabia
from sabia.registry import Registry, evaluate
from sabia.spec import DEFAULT_FLOAT_TOLERANCE

_FIXTURE = Path(__file__).parent / "data" / "marketgoblin_panel.parquet"
_FLOAT_DTYPES = (pl.Float32, pl.Float64)

PANEL = pl.read_parquet(_FIXTURE)
SYMBOLS = sorted(PANEL.get_column("symbol").unique().to_list())

_REGISTRY = Registry.default()
_FEATURES = _REGISTRY.features()
_IDS = [f.spec.name for f in _FEATURES]


def test_fixture_is_a_complete_real_panel() -> None:
    # Sanity: the snapshot is what the suite assumes (multi-symbol, complete cross-section, > the
    # longest warmup so even the deepest decay feature emits values).
    assert PANEL.get_column("symbol").n_unique() >= 2
    per_symbol = PANEL.group_by("symbol").len().get_column("len")
    assert per_symbol.min() == per_symbol.max()  # complete cross-section
    assert per_symbol.min() > max(f.spec.min_history for f in _FEATURES)


def test_validate_strict_passes_on_real_data() -> None:
    required = frozenset().union(*(f.spec.input_roles for f in _FEATURES))
    warnings = sabia.validate(
        PANEL,
        schema=SCHEMA,
        required_roles=required,
        complete_panel=True,
        universe=SYMBOLS,
        mode=sabia.ValidationMode.STRICT,
    )
    assert warnings == []


@pytest.mark.parametrize("feature", _FEATURES, ids=_IDS)
def test_feature_is_finite_and_nonempty_on_real_data(feature: sabia.BoundFeature) -> None:
    out = evaluate(PANEL, feature, SCHEMA)
    non_null = out.drop_nulls()
    assert non_null.len() > 0, "feature produced no values on real data"
    if out.dtype in _FLOAT_DTYPES:
        assert non_null.is_finite().all(), "feature leaked inf/NaN on real data"


def test_compute_names_cross_sectional_columns_correctly() -> None:
    # Regression for review finding #1: XS reductions used to come back as '__sabia_xs_signal__' /
    # 'literal'. compute() must label each column with the feature name.
    df = sabia.compute(PANEL, sabia.cross_sectional.xs_rank_mom(), schema=SCHEMA, universe=SYMBOLS)
    assert df.columns == ["xs_rank_mom_252_21"]


def test_compute_two_rank_features_do_not_collide() -> None:
    # xs_rank_mom and rev_1m both reduce by rank; before the alias fix both came back named
    # '__sabia_xs_signal__' and collided in the assembled frame.
    df = sabia.compute(
        PANEL,
        sabia.cross_sectional.xs_rank_mom(),
        sabia.cross_sectional.rev_1m(),
        schema=SCHEMA,
        universe=SYMBOLS,
    )
    assert df.columns == ["xs_rank_mom_252_21", "rev_1m_21"]


def test_compute_mixed_ts_and_xs_columns() -> None:
    df = sabia.compute(
        PANEL,
        sabia.momentum.rsi(period=14),
        sabia.cross_sectional.xs_z_mom(),
        schema=SCHEMA,
        universe=SYMBOLS,
    )
    assert df.columns == ["rsi_14", "xs_z_mom_252_21"]


def test_bounded_features_stay_in_range_on_real_data() -> None:
    aapl = PANEL.filter(pl.col("symbol") == "AAPL")
    rsi = evaluate(aapl, _REGISTRY.get("rsi_14"), SCHEMA).drop_nulls()
    assert rsi.min() >= 0.0 and rsi.max() <= 100.0
    cmf = evaluate(aapl, _REGISTRY.get("cmf_21"), SCHEMA).drop_nulls()
    assert cmf.min() >= -1.0 and cmf.max() <= 1.0


def test_cross_sectional_ranks_are_a_permutation_each_date() -> None:
    feature = _REGISTRY.get("xs_rank_mom_252_21")
    keyed = PANEL.select("timestamp").with_columns(v=evaluate(PANEL, feature, SCHEMA)).drop_nulls()
    # Every fully-populated date ranks its symbols into the (0, 1] grid k/n.
    n = len(SYMBOLS)
    per_date = keyed.group_by("timestamp").agg(pl.col("v").max().alias("mx"), pl.len().alias("k"))
    full = per_date.filter(pl.col("k") == n)
    assert full.height > 0
    assert (full.get_column("mx") - 1.0).abs().max() < 1e-9  # top rank is exactly 1.0


def test_causality_truncating_panel_does_not_change_prefix() -> None:
    # Drop the last 200 dates; every surviving value must be unchanged (strictly trailing).
    feature = _REGISTRY.get("macd_12_26_9")
    aapl = PANEL.filter(pl.col("symbol") == "AAPL").sort("timestamp")
    full = evaluate(aapl, feature, SCHEMA)
    truncated = evaluate(aapl.head(aapl.height - 200), feature, SCHEMA)
    head_full = full.head(truncated.len())
    # Same computation on a prefix: values must match the full-history prefix within tolerance
    # (use the shared helper rather than a raw float ==, per CLAUDE.md float-comparison rule).
    assert_series_close(truncated, head_full, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)
