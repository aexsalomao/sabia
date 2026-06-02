"""Ergonomic toolkit: FeatureSet, describe, role/column introspection, drop_warmup, audit_frame."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import SCHEMA, make_panel, make_series

import sabia
from sabia.schema import BarSchema
from sabia.toolkit import (
    FeatureSet,
    describe,
    drop_warmup,
    max_min_history,
    required_columns,
    required_roles,
)
from sabia.typing import CLOSE_TR, HIGH_SPLIT, LOW_SPLIT


def test_required_roles_unions_inputs() -> None:
    feats = [sabia.momentum.rsi(period=14), sabia.volatility.atr(window=14)]
    roles = required_roles(feats)
    assert CLOSE_TR in roles
    assert HIGH_SPLIT in roles and LOW_SPLIT in roles


def test_required_columns_resolves_against_schema() -> None:
    cols = required_columns([sabia.momentum.rsi(period=14)], SCHEMA)
    assert cols == {"close@tr": "close"}


def test_max_min_history_takes_the_largest() -> None:
    feats = [sabia.returns.ret_log(period=1), sabia.returns.drawdown(window=252)]
    assert max_min_history(feats) == 252


def test_drop_warmup_trims_per_symbol() -> None:
    panel = make_panel(300)
    feats = [sabia.momentum.roc(window=21)]
    keyed = sabia.compute(panel, *feats, schema=SCHEMA, include_keys=True)
    trimmed = drop_warmup(keyed, feats, symbol_col="symbol")
    # 22 warmup rows dropped per symbol (min_history), so no remaining roc value is null.
    assert trimmed.get_column("roc_21").null_count() == 0
    assert trimmed.height == panel.height - 22 * panel.get_column("symbol").n_unique()


def test_describe_renders_key_spec_fields() -> None:
    card = describe(sabia.momentum.rsi(period=14))
    assert card.startswith("rsi_14")
    assert "family: momentum" in card
    assert "roles: close@tr" in card
    assert "range: [0.0, 100.0]" in card
    assert "Wilder (1978)" in card


def test_feature_set_computes_and_manifests() -> None:
    fs = FeatureSet(
        [
            sabia.returns.ret_log(period=1),
            sabia.momentum.rsi(period=14),
            sabia.volatility.vol_cc(window=10),
        ]
    )
    assert len(fs) == 3
    assert fs.names() == ["ret_log_1", "rsi_14", "vol_cc_10"]
    df = fs.compute(make_series(60), schema=SCHEMA, include_keys=True)
    assert df.columns == ["symbol", "timestamp", "ret_log_1", "rsi_14", "vol_cc_10"]
    manifest = fs.manifest(SCHEMA)
    assert {f.name for f in manifest.features} == set(fs.names())


def test_empty_feature_set_rejects_compute() -> None:
    with pytest.raises(ValueError, match="empty"):
        FeatureSet([]).compute(make_series(10), schema=SCHEMA)


def test_audit_frame_reports_clean_panel() -> None:
    panel = make_panel(50)
    report = sabia.audit_frame(panel, schema=SCHEMA, features=[sabia.momentum.rsi(period=14)])
    assert report.rows == panel.height
    assert report.symbols == 3
    assert report.timestamp_utc_ok is True
    assert report.missing_roles == ()
    assert report.ohlc_violations == 0
    assert report.duplicate_keys == 0
    assert report.completeness == pytest.approx(1.0)


def test_audit_frame_non_panel_reports_none_for_panel_fields() -> None:
    series = make_series(40).drop("symbol")
    report = sabia.audit_frame(series, schema=SCHEMA)
    assert report.symbols is None
    assert report.completeness is None
    assert report.rows == 40


def test_audit_frame_panel_missing_timestamp_does_not_raise() -> None:
    # A panel lacking its timestamp column is exactly the defect audit_frame should REPORT, not die.
    frame = pl.DataFrame({"symbol": ["A", "A"], "close": [1.0, 2.0]})
    report = sabia.audit_frame(frame, schema=BarSchema(roles={CLOSE_TR: "close"}))
    assert report.timestamp_utc_ok is False
    assert report.start is None and report.end is None
    assert report.completeness is None


def test_audit_frame_reports_missing_roles() -> None:
    panel = make_panel(40).drop("market_ret")  # beta needs market_ret, now absent
    report = sabia.audit_frame(panel, schema=SCHEMA, features=[sabia.cross_sectional.beta()])
    assert "market_ret" in report.missing_roles


def test_audit_frame_counts_non_final_bars() -> None:
    n = 5
    start = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n)],
            "symbol": ["A"] * n,
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [1.0] * n,
            "closed": [True, True, True, False, False],
        }
    )
    report = sabia.audit_frame(frame, schema=BarSchema.ohlcv(closed_col="closed"))
    assert report.non_final_bars == 2


def test_audit_frame_reports_partial_completeness() -> None:
    panel = make_panel(50)  # 3 symbols x 50 timestamps, complete
    # Drop CCC at the last timestamp so exactly one timestamp is incomplete.
    last_ts = panel.get_column("timestamp").max()
    incomplete = panel.filter(~((pl.col("symbol") == "CCC") & (pl.col("timestamp") == last_ts)))
    report = sabia.audit_frame(incomplete, schema=SCHEMA)
    assert report.completeness == pytest.approx(49 / 50)


def test_audit_frame_flags_problems() -> None:
    panel = make_panel(50)
    # Inject a duplicate key and break OHLC ordering on one row.
    broken = pl.concat([panel, panel.head(1)]).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("high") - 1000)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    report = sabia.audit_frame(broken, schema=SCHEMA)
    assert report.duplicate_keys == 1
    assert report.ohlc_violations >= 1


def test_recipes_return_feature_sets() -> None:
    assert isinstance(sabia.recipes.daily_core(), FeatureSet)
    assert len(sabia.recipes.volatility_core()) == 4
    xs = sabia.recipes.cross_sectional_core()
    assert len(xs) == 5
    assert sabia.cross_sectional.xs_rank_mom().spec.name in xs.names()
