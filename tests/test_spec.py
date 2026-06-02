"""Tests for the feature contract types and helpers (spec.py)."""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sabia.cross_sectional import _xs_rank_reduce, _xs_zscore_reduce, xs_rank_mom, xs_z_mom
from sabia.momentum import cci
from sabia.params import FrozenParams
from sabia.schema import BarSchema
from sabia.spec import (
    EWM_WARMUP_TOL,
    HORIZON_LOOKBACKS,
    PARITY_RECURSIVE_TOLERANCE,
    Horizon,
    _module_constants,
    _transitive_sources,
    ewm_effective_warmup,
    feature_fingerprint,
)
from sabia.typing import CLOSE_RAW, CLOSE_TR
from sabia.volatility import _rs_term, vol_rs

# --- ewm_effective_warmup ----------------------------------------------------------------------


def test_ewm_warmup_matches_analytic_formula() -> None:
    alpha = 1 / 14
    assert ewm_effective_warmup(alpha) == math.ceil(math.log(EWM_WARMUP_TOL) / math.log(1 - alpha))


def test_ewm_warmup_grows_as_alpha_shrinks() -> None:
    assert ewm_effective_warmup(1 / 50) > ewm_effective_warmup(1 / 5)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_ewm_warmup_rejects_out_of_range_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        ewm_effective_warmup(alpha)


# --- fingerprint stability ---------------------------------------------------------------------


def _rsi_formatted(s: BarSchema) -> pl.Expr:
    delta = pl.col(s.column(CLOSE_TR)).diff()
    return delta.ewm_mean(alpha=1 / 14, adjust=False).alias("x")


def _rsi_reformatted(s: BarSchema) -> pl.Expr:
    # Same AST, different whitespace and an extra comment -- fingerprint must not change.
    delta = pl.col(s.column(CLOSE_TR)).diff()

    return delta.ewm_mean(alpha=1 / 14, adjust=False).alias("x")


def _rsi_changed(s: BarSchema) -> pl.Expr:
    delta = pl.col(s.column(CLOSE_TR)).diff()
    return delta.ewm_mean(alpha=2 / 14, adjust=False).alias("x")  # 2/14, not 1/14


def _fp(build, params, *, roles=(CLOSE_TR,)):  # type: ignore[no-untyped-def]
    return feature_fingerprint(
        canonical_id="rsi", version=1, params=params, input_roles=roles, build=build
    )


def test_fingerprint_is_deterministic() -> None:
    assert _fp(_rsi_formatted, FrozenParams(period=14)) == _fp(
        _rsi_formatted, FrozenParams(period=14)
    )


def test_fingerprint_ignores_formatting_and_comments() -> None:
    assert _fp(_rsi_formatted, FrozenParams(period=14)) == _fp(
        _rsi_reformatted, FrozenParams(period=14)
    )


def test_fingerprint_changes_with_formula() -> None:
    assert _fp(_rsi_formatted, FrozenParams(period=14)) != _fp(
        _rsi_changed, FrozenParams(period=14)
    )


def test_fingerprint_changes_with_params() -> None:
    assert _fp(_rsi_formatted, FrozenParams(period=14)) != _fp(
        _rsi_formatted, FrozenParams(period=21)
    )


def test_fingerprint_changes_with_roles() -> None:
    # Adjustment is part of role identity: rsi_14 (close@tr) and rsi_raw_14 (close@raw) must differ.
    assert _fp(_rsi_formatted, FrozenParams(period=14)) != _fp(
        _rsi_formatted, FrozenParams(period=14), roles=(CLOSE_RAW,)
    )


@given(period=st.integers(min_value=2, max_value=200))
def test_fingerprint_is_sixteen_hex_chars(period: int) -> None:
    fingerprint = _fp(_rsi_formatted, FrozenParams(period=period))
    assert len(fingerprint) == 16
    assert all(c in "0123456789abcdef" for c in fingerprint)


def test_fingerprint_covers_first_party_helpers() -> None:
    # The fingerprint must follow first-party calls: vol_rs delegates its math to _rs_term, so a
    # change to _rs_term must be inside vol_rs's hashed source -- otherwise the manifest gate would
    # miss a real formula change (the gap this guards against).
    sources = _transitive_sources([vol_rs])
    rs_term_source = _transitive_sources([_rs_term])[0]
    assert rs_term_source in sources


def test_fingerprint_covers_cross_sectional_reduction() -> None:
    # A cross-sectional feature's defining operation is its reduction (carried in `build`). The
    # rank and zscore reductions over the same momentum signal must yield different fingerprints.
    assert xs_rank_mom().spec.fingerprint != xs_z_mom().spec.fingerprint
    # And the reduction source itself is hashed (rank vs zscore differ).
    assert _transitive_sources([_xs_rank_reduce]) != _transitive_sources([_xs_zscore_reduce])


def test_fingerprint_folds_module_constants() -> None:
    # Review finding #4: a module constant used as a *value* (cci's _CCI_SCALE) must be folded into
    # the fingerprint by value, so retuning the literal is provable at the manifest gate even though
    # it is not a call _first_party_callees would follow.
    consts = _module_constants(cci().build)
    assert any(c.startswith("_CCI_SCALE=") for c in consts)
    # The value (not just the name) rides in the hashed payload, so editing 0.015 bumps the hash.
    payload = "".join(_transitive_sources([cci().build]))
    assert "_CCI_SCALE=0.015" in payload


def test_parity_tolerance_is_derived_with_headroom_over_warmup() -> None:
    # Review finding #5: the two tolerances must stay coupled so tightening one tightens the other,
    # and the burn-in must always be tighter than what parity asserts.
    assert EWM_WARMUP_TOL < PARITY_RECURSIVE_TOLERANCE
    assert pytest.approx(EWM_WARMUP_TOL * 100.0) == PARITY_RECURSIVE_TOLERANCE


# --- horizon grids -----------------------------------------------------------------------------


def test_every_band_has_a_lookback_grid() -> None:
    assert set(HORIZON_LOOKBACKS) == set(Horizon)
