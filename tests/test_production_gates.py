"""Production-readiness gates (FEATURES.md 9, 10): eager-vs-lazy and chunked-vs-rechunked parity,
symbol isolation, and input-order invariance.

Every gate is parametrized per bound feature (``ids=spec.name``) so a failure names the feature.
Feature selection / routing happens in module-level predicates and the parametrize decorators --
there is no math-bearing loop or branch in a test body. Time-series and cross-sectional features
route through the same library evaluator (``evaluate``: single-pass for TS, two-pass for XS), so a
single comparison covers both paths; the eager-vs-lazy gate additionally pins the explicit
``compute``/``compute_lazy`` surfaces for the time-series fan-out.

The HEAVY escape-hatch features (rolling OLS / variance ratio / Roll / Corwin-Schultz / beta /
idio-vol / percentile / CCI) carry NumPy or ``rolling_map`` kernels (FEATURES.md 10); the chunked
gate is the one that catches a kernel reading raw buffers across chunk boundaries, so its coverage
of those features is mandatory.
"""

from __future__ import annotations

import polars as pl
import pytest
from conftest import assert_series_close
from synthetic import SCHEMA, SYMBOL, make_series

import sabia
from sabia.registry import BoundFeature, Registry, evaluate
from sabia.spec import DEFAULT_FLOAT_TOLERANCE

_FEATURES = Registry.default().features()
_IDS = [f.spec.name for f in _FEATURES]

_TS = [f for f in _FEATURES if not f.spec.requires_complete_panel]
_TS_IDS = [f.spec.name for f in _TS]


# HEAVY escape-hatch features (NumPy / rolling_map kernels). Mandatory coverage for the parity gates
# below; declared here so the chunked gate provably exercises them (cross-checked in setup).
_HEAVY = frozenset(
    {
        "ols_slope_63",
        "roll_spread_21",
        "spread_corwin_schultz",
        "var_ratio_2_21",
        "beta_252",
        "idio_vol_252",
        "price_pctile_252",
        "cci_20",
    }
)


def test_heavy_escape_hatch_features_are_all_registered() -> None:
    # If a HEAVY feature is renamed/dropped, the parity gates below would silently stop covering it.
    assert set(_IDS) >= _HEAVY


# --- H5a: eager-vs-lazy parity (FEATURES.md 9, 10) ---------------------------------------------


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_eager_compute_equals_lazy_collect_for_time_series(
    feature: BoundFeature, series: pl.DataFrame
) -> None:
    eager = sabia.compute(series, feature, schema=SCHEMA, validation=sabia.ValidationMode.OFF)
    lazy = sabia.compute_lazy(series, feature, schema=SCHEMA).collect()
    assert_series_close(eager.to_series(), lazy.to_series(), rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


@pytest.mark.parametrize("feature", _FEATURES, ids=_IDS)
def test_evaluate_eager_equals_lazy_for_every_feature(
    feature: BoundFeature, panel: pl.DataFrame
) -> None:
    # ``evaluate`` is the path the eager assembler uses; routing it through a lazy frame must give
    # the same Series for both time-series (single-pass) and cross-sectional (two-pass) features.
    eager = evaluate(panel, feature, SCHEMA)
    lazy = evaluate(panel.lazy(), feature, SCHEMA)
    assert_series_close(eager, lazy, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


# --- H5b: chunked-vs-rechunked parity (FEATURES.md 9, 10) --------------------------------------


def _multi_chunk_panel() -> pl.DataFrame:
    # Per-symbol frames are each a single chunk and individually timestamp-sorted; concatenating
    # them in symbol order (without rechunking) is already globally sorted by (symbol, timestamp)
    # AND keeps one chunk per symbol -- the layout that trips a kernel reading a raw contiguous
    # buffer across chunk seams. A global ``.sort()`` here would rechunk to a single chunk and
    # silently defeat the gate (observed across Polars/Python combinations in CI).
    parts = [
        make_series(600, seed=i, symbol=s).sort("timestamp")
        for i, s in enumerate(("AAA", "BBB", "CCC"))
    ]
    return pl.concat(parts, rechunk=False)


_CHUNKED_PANEL = _multi_chunk_panel()


def test_chunked_panel_actually_has_multiple_chunks() -> None:
    # The chunked gate is meaningless if the frame is already a single chunk -- assert the setup.
    assert _CHUNKED_PANEL.n_chunks() > 1


@pytest.mark.parametrize("feature", _FEATURES, ids=_IDS)
def test_chunked_output_equals_rechunked(feature: BoundFeature) -> None:
    chunked = evaluate(_CHUNKED_PANEL, feature, SCHEMA)
    rechunked = evaluate(_CHUNKED_PANEL.rechunk(), feature, SCHEMA)
    assert_series_close(chunked, rechunked, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


# --- M8a: symbol isolation -- perturb-twin (FEATURES.md 9) -------------------------------------


def _twin_panel(perturb: bool) -> pl.DataFrame:
    # AAA plus a byte-identical twin AAB; optionally perturb the twin's close. AAA's output must be
    # unaffected by anything that happens to AAB (a feature that forgot .over(symbol) would leak).
    base = make_series(600, seed=0, symbol="AAA")
    twin = base.with_columns(pl.lit("AAB").alias(SYMBOL))
    if perturb:
        twin = twin.with_columns((pl.col("close") * 2.0 + 7.0).alias("close"))
    return pl.concat([base, twin]).sort(SYMBOL, "timestamp")


_TWIN_CLEAN = _twin_panel(perturb=False)
_TWIN_PERTURBED = _twin_panel(perturb=True)


def _aaa_values(feature: BoundFeature, frame: pl.DataFrame) -> pl.Series:
    keyed = frame.select("timestamp", SYMBOL).with_columns(_v=evaluate(frame, feature, SCHEMA))
    return keyed.filter(pl.col(SYMBOL) == "AAA").to_series(2)


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_perturbing_a_twin_symbol_does_not_change_the_other(feature: BoundFeature) -> None:
    clean = _aaa_values(feature, _TWIN_CLEAN)
    perturbed = _aaa_values(feature, _TWIN_PERTURBED)
    assert_series_close(perturbed, clean, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


@pytest.mark.parametrize("feature", _TS, ids=_TS_IDS)
def test_shuffling_symbol_order_does_not_change_output(feature: BoundFeature) -> None:
    # Re-order rows then re-sort by the canonical keys: a correctly grouped feature is invariant.
    shuffled = _TWIN_CLEAN.sample(fraction=1.0, shuffle=True, seed=7).sort(SYMBOL, "timestamp")
    base = _aaa_values(feature, _TWIN_CLEAN)
    reordered = _aaa_values(feature, shuffled)
    assert_series_close(reordered, base, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)


# --- M8b: input-order invariance (FEATURES.md 9) -----------------------------------------------


@pytest.mark.parametrize("feature", _FEATURES, ids=_IDS)
def test_output_is_invariant_to_input_row_order(feature: BoundFeature, panel: pl.DataFrame) -> None:
    # Shuffle (seeded) then re-sort by the sort keys: feature output must match the canonical frame.
    shuffled = panel.sample(fraction=1.0, shuffle=True, seed=13).sort(SYMBOL, "timestamp")
    base = evaluate(panel, feature, SCHEMA)
    reordered = evaluate(shuffled, feature, SCHEMA)
    assert_series_close(reordered, base, rtol=0.0, atol=DEFAULT_FLOAT_TOLERANCE)
