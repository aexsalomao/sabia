"""Tests for the constructable feature registry."""

import polars as pl
import pytest
from synthetic import SCHEMA

from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import BoundFeature, FrozenRegistryError, Registry, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, DataTier, Evidence, Family, Horizon, Recurrence, Unit
from sabia.typing import CLOSE_TR


def _feature(
    name: str,
    *,
    version: int = 1,
    band: Horizon = Horizon.MEDIUM,
    tier: DataTier = DataTier.DAILY,
    recurrence: Recurrence = Recurrence.FINITE,
) -> BoundFeature:
    def build(s: BarSchema) -> pl.Expr:
        return pl.col(s.column(CLOSE_TR)).alias(name)

    return bind_feature(
        build,
        name=name,
        version=version,
        family=Family.MOMENTUM,
        native_band=(band,),
        lookback=14,
        min_history=14,
        recurrence=recurrence,
        effective_warmup=14,
        cost_class=Cost.LINEAR,
        data_tier=tier,
        input_roles=(CLOSE_TR,),
        output_unit=Unit.UNITLESS,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=Reference("test", 2024)),
        params=FrozenParams(),
    )


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


def test_path_dependent_recurrence_is_admitted() -> None:
    # PATH_DEPENDENT is enabled for the microstructure family (VPIN buckets, CUSUM); replay-based
    # parity, not fixed-window (FEATURES.md 8.2, 13). It must register without error.
    reg = Registry([_feature("vpin_50", recurrence=Recurrence.PATH_DEPENDENT)])
    assert reg.get("vpin_50").spec.recurrence is Recurrence.PATH_DEPENDENT


def test_expanding_recurrence_still_rejected() -> None:
    # EXPANDING (unbounded cumulative) has no windowed-recompute guarantee and stays banned.
    with pytest.raises(ValueError, match="recurrence"):
        Registry([_feature("obv_raw", recurrence=Recurrence.EXPANDING)])


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


def test_expr_returns_expression() -> None:
    feature = _feature("rsi_14")
    assert isinstance(feature.expr(SCHEMA), pl.Expr)


# --- registry immutability (FEATURES.md 7) -----------------------------------------------------


def test_default_registry_rejects_in_place_add() -> None:
    # Registry.default() is frozen: built-ins are not overridable in place (a different impl is a
    # different (name, version)). add() on a frozen registry must raise.
    with pytest.raises(FrozenRegistryError, match="frozen"):
        Registry.default().add(_feature("brand_new_1"))


def test_default_registry_rejects_replacing_a_builtin() -> None:
    # Re-adding an existing built-in name is rejected the same way -- no in-place replacement.
    with pytest.raises(FrozenRegistryError, match="frozen"):
        Registry.default().add(_feature("rsi_14"))


def test_default_registry_instances_are_independent() -> None:
    # Two calls yield distinct objects that do not share mutable state: mutating an unfrozen copy of
    # one must not leak into the other.
    first = Registry.default()
    second = Registry.default()
    assert first is not second
    assert first._by_key is not second._by_key  # noqa: SLF001 -- asserting no shared mutable state
    unfrozen = Registry(first.features())  # an unfrozen copy
    unfrozen.add(_feature("brand_new_1"))
    assert "brand_new_1" not in second
