"""Tests for the feature contract types and helpers (spec.py)."""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sabia.spec import (
    EWM_WARMUP_TOL,
    HORIZON_LOOKBACKS,
    Horizon,
    ewm_effective_warmup,
    feature_fingerprint,
)

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


def _rsi_formatted(close: str, period: int) -> pl.Expr:
    delta = pl.col(close).diff()
    return delta.ewm_mean(alpha=1 / period, adjust=False).alias("x")


def _rsi_reformatted(close: str, period: int) -> pl.Expr:
    # Same AST, different whitespace and an extra comment -- fingerprint must not change.
    delta = pl.col(close).diff()

    return delta.ewm_mean(alpha=1 / period, adjust=False).alias("x")


def _rsi_changed(close: str, period: int) -> pl.Expr:
    delta = pl.col(close).diff()
    return delta.ewm_mean(alpha=2 / period, adjust=False).alias("x")  # 2/period, not 1/period


def test_fingerprint_is_deterministic() -> None:
    assert feature_fingerprint(_rsi_formatted, {"period": 14}) == feature_fingerprint(
        _rsi_formatted, {"period": 14}
    )


def test_fingerprint_ignores_formatting_and_comments() -> None:
    assert feature_fingerprint(_rsi_formatted, {"period": 14}) == feature_fingerprint(
        _rsi_reformatted, {"period": 14}
    )


def test_fingerprint_changes_with_formula() -> None:
    assert feature_fingerprint(_rsi_formatted, {"period": 14}) != feature_fingerprint(
        _rsi_changed, {"period": 14}
    )


def test_fingerprint_changes_with_params() -> None:
    assert feature_fingerprint(_rsi_formatted, {"period": 14}) != feature_fingerprint(
        _rsi_formatted, {"period": 21}
    )


@given(period=st.integers(min_value=2, max_value=200))
def test_fingerprint_is_sixteen_hex_chars(period: int) -> None:
    fingerprint = feature_fingerprint(_rsi_formatted, {"period": period})
    assert len(fingerprint) == 16
    assert all(c in "0123456789abcdef" for c in fingerprint)


# --- horizon grids -----------------------------------------------------------------------------


def test_every_band_has_a_lookback_grid() -> None:
    assert set(HORIZON_LOOKBACKS) == set(Horizon)
