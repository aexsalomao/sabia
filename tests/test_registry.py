"""Tests for the constructable feature registry."""

import polars as pl
import pytest

from sabia.registry import RegisteredFeature, Registry
from sabia.spec import (
    Column,
    Cost,
    DataTier,
    Family,
    FeatureSpec,
    Horizon,
    Recurrence,
)


def _feature(
    name: str,
    *,
    version: int = 1,
    band: Horizon = Horizon.MEDIUM,
    tier: DataTier = DataTier.DAILY,
) -> RegisteredFeature:
    spec = FeatureSpec(
        name=name,
        version=version,
        fingerprint="deadbeef",
        family=Family.MOMENTUM,
        native_band=frozenset({band}),
        lookback=14,
        min_history=14,
        recurrence=Recurrence.FINITE,
        effective_warmup=0,
        cost_class=Cost.LINEAR,
        data_tier=tier,
        inputs=frozenset({Column.CLOSE}),
        output_dtype=pl.Float64(),
        citation="test",
        params={},
    )
    return RegisteredFeature(spec=spec, build=lambda: pl.col(Column.CLOSE))


def test_default_registry_contains_shipped_features() -> None:
    assert "rsi_14" in Registry.default()


def test_add_and_get_roundtrips() -> None:
    reg = Registry([_feature("rsi_14")])
    assert reg.get("rsi_14").spec.name == "rsi_14"


def test_get_defaults_to_highest_version() -> None:
    reg = Registry([_feature("rsi_14", version=1), _feature("rsi_14", version=2)])
    assert reg.get("rsi_14").spec.version == 2


def test_get_specific_version() -> None:
    reg = Registry([_feature("rsi_14", version=1), _feature("rsi_14", version=2)])
    assert reg.get("rsi_14", version=1).spec.version == 1


def test_duplicate_name_version_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Registry([_feature("rsi_14"), _feature("rsi_14")])


def test_non_snake_case_name_rejected() -> None:
    with pytest.raises(ValueError, match="snake_case"):
        Registry([_feature("RSI14")])


def test_unknown_name_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="no feature named"):
        Registry().get("missing")


def test_where_filters_by_predicate() -> None:
    reg = Registry([_feature("a_1", band=Horizon.SHORT), _feature("b_1", band=Horizon.LONG)])
    long_only = reg.where(lambda s: Horizon.LONG in s.native_band)
    assert long_only.names() == ["b_1"]


def test_available_excludes_finer_tier_features() -> None:
    reg = Registry(
        [_feature("daily_1", tier=DataTier.DAILY), _feature("minute_1", tier=DataTier.MINUTE)]
    )
    # On daily bars, a minute-tier feature is not computable.
    assert reg.available(DataTier.DAILY).names() == ["daily_1"]


def test_available_on_finer_bars_unlocks_more() -> None:
    reg = Registry(
        [_feature("daily_1", tier=DataTier.DAILY), _feature("minute_1", tier=DataTier.MINUTE)]
    )
    assert set(reg.available(DataTier.MINUTE).names()) == {"daily_1", "minute_1"}


def test_contains_checks_name() -> None:
    reg = Registry([_feature("rsi_14")])
    assert "rsi_14" in reg
    assert "missing" not in reg


def test_build_returns_expression() -> None:
    feature = _feature("rsi_14")
    assert isinstance(feature.build(), pl.Expr)
