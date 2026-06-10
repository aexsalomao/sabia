"""Reference-value and degenerate-input tests for the microstructure family (FEATURES.md 13).

Hand-checked tables, not snapshots: they encode what the math should produce, decoupled from how.
The cross-cutting causality / parity / dtype gates auto-cover these features via the registry.
"""

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sabia.microstructure import (
    bipower,
    book_imbalance,
    eff_spread,
    jump_rj,
    quoted_spread,
    rsemivar_dn,
    rsemivar_up,
    rskew,
    rvar,
    signed_jump,
    trade_imbalance,
    vpin,
)
from sabia.schema import BarSchema
from sabia.typing import (
    ASK_RAW,
    BID_RAW,
    CLOSE_RAW,
    SIGNED_VOLUME_RAW,
    VOLUME_RAW,
    Adjustment,
    DepthRole,
    QuoteField,
)

_TOL = 1e-12
_SCHEMA = BarSchema(roles={SIGNED_VOLUME_RAW: "signed_volume", VOLUME_RAW: "volume"})
_CLOSE_SCHEMA = BarSchema(roles={CLOSE_RAW: "close"})
_QUOTE_SCHEMA = BarSchema(roles={CLOSE_RAW: "close", BID_RAW: "bid", ASK_RAW: "ask"})

_START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _frame(signed: list[float], volume: list[float]) -> pl.DataFrame:
    n = len(signed)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(n)],
            "symbol": ["AAA"] * n,
            "signed_volume": signed,
            "volume": volume,
        }
    )


def _imbalance(signed: list[float], volume: list[float], *, window: int = 2) -> list[float | None]:
    out = _frame(signed, volume).select(trade_imbalance(window=window).expr(_SCHEMA)).to_series()
    return out.to_list()


def test_trade_imbalance_is_windowed_signed_fraction() -> None:
    # window=2: bar0 null (warmup); bar1 = (3 + -1)/(10+10) = 0.1; bar2 = (-1 + 5)/(10+10) = 0.2.
    out = _imbalance([3.0, -1.0, 5.0], [10.0, 10.0, 10.0], window=2)
    assert out[0] is None
    assert out[1] == pytest.approx(0.1, abs=_TOL)
    assert out[2] == pytest.approx(0.2, abs=_TOL)


def test_all_buyer_initiated_saturates_at_one() -> None:
    # signed == volume every bar -> imbalance == +1 (pure buy pressure).
    out = _imbalance([10.0, 10.0, 10.0], [10.0, 10.0, 10.0], window=2)
    assert out[1] == pytest.approx(1.0, abs=_TOL)
    assert out[2] == pytest.approx(1.0, abs=_TOL)


def test_all_seller_initiated_saturates_at_minus_one() -> None:
    out = _imbalance([-4.0, -4.0], [4.0, 4.0], window=2)
    assert out[1] == pytest.approx(-1.0, abs=_TOL)


def test_balanced_flow_is_zero() -> None:
    out = _imbalance([5.0, -5.0, 5.0, -5.0], [10.0, 10.0, 10.0, 10.0], window=2)
    # each window sums signed to 0 -> imbalance 0.
    assert out[1] == pytest.approx(0.0, abs=_TOL)
    assert out[3] == pytest.approx(0.0, abs=_TOL)


def test_zero_volume_window_yields_null_never_inf() -> None:
    # A fully halted window (zero total volume) has no flow to normalize -> null, not inf/NaN.
    out = _imbalance([0.0, 0.0], [0.0, 0.0], window=2)
    assert out[1] is None


def test_vpin_is_mean_absolute_bucket_imbalance() -> None:
    # Per-bucket |signed|/volume: |3|/10=0.3, |-1|/10=0.1; n_buckets=2 mean = 0.2. Always in [0,1].
    out = (
        _frame([3.0, -1.0], [10.0, 10.0])
        .select(vpin(n_buckets=2).expr(_SCHEMA))
        .to_series()
        .to_list()
    )
    assert out[1] == pytest.approx(0.2, abs=_TOL)


def test_vpin_saturates_at_one_for_one_sided_flow() -> None:
    # Fully one-sided buckets (|signed| == volume) -> imbalance 1 every bucket -> VPIN == 1.
    out = (
        _frame([10.0, -10.0], [10.0, 10.0])
        .select(vpin(n_buckets=2).expr(_SCHEMA))
        .to_series()
        .to_list()
    )
    assert out[1] == pytest.approx(1.0, abs=_TOL)


# --- realized volatility -----------------------------------------------------------------------

# Closes built from known log returns: r = [_, 0.1, -0.2, 0.1] (first bar seeds, has no return).
_RETS = [0.0, 0.1, -0.2, 0.1]


def _close_frame(returns: list[float]) -> pl.DataFrame:
    closes = [100.0]
    for r in returns[1:]:
        closes.append(closes[-1] * math.exp(r))
    n = len(closes)
    return pl.DataFrame(
        {
            "timestamp": [_START + timedelta(minutes=i) for i in range(n)],
            "symbol": ["AAA"] * n,
            "close": closes,
        }
    )


def _last(feature, schema: BarSchema, frame: pl.DataFrame) -> float | None:
    return frame.select(feature.expr(schema)).to_series().to_list()[-1]


def test_rvar_is_sum_of_squared_returns() -> None:
    # window=2 at the last bar covers r=[-0.2, 0.1]: RV = 0.04 + 0.01 = 0.05.
    out = _last(rvar(window=2), _CLOSE_SCHEMA, _close_frame(_RETS))
    assert out == pytest.approx(0.05, abs=_TOL)


def test_semivariances_split_rv_by_sign() -> None:
    frame = _close_frame(_RETS)
    up = _last(rsemivar_up(window=2), _CLOSE_SCHEMA, frame)
    dn = _last(rsemivar_dn(window=2), _CLOSE_SCHEMA, frame)
    # last window r=[-0.2, 0.1]: up = 0.1^2 = 0.01; down = (-0.2)^2 = 0.04; up + down = RV.
    assert up == pytest.approx(0.01, abs=_TOL)
    assert dn == pytest.approx(0.04, abs=_TOL)
    assert up + dn == pytest.approx(0.05, abs=_TOL)


def test_signed_jump_is_upside_minus_downside() -> None:
    # sum r|r| over [-0.2, 0.1] = -0.04 + 0.01 = -0.03 = RS+ - RS- (0.01 - 0.04).
    out = _last(signed_jump(window=2), _CLOSE_SCHEMA, _close_frame(_RETS))
    assert out == pytest.approx(-0.03, abs=_TOL)


def test_bipower_pairs_adjacent_absolute_returns() -> None:
    # window=2 at the last bar covers the SAME returns as rvar, r=[-0.2, 0.1]: their window-1 = 1
    # adjacent product is |0.1||-0.2| = 0.02; bipower = (pi/2) * 0.02.
    out = _last(bipower(window=2), _CLOSE_SCHEMA, _close_frame(_RETS))
    assert out == pytest.approx((math.pi / 2.0) * 0.02, abs=_TOL)


def test_jump_rj_compares_rv_and_bv_over_the_same_returns() -> None:
    # window=2 over r=[-0.2, 0.1]: RV = 0.05, BV = (pi/2)*0.02, so RJ = 1 - (pi/2)*0.02/0.05 --
    # a like-for-like comparison (a return outside RV's window can no longer inflate BV).
    out = _last(jump_rj(window=2), _CLOSE_SCHEMA, _close_frame(_RETS))
    expected = 1.0 - (math.pi / 2.0) * 0.02 / 0.05
    assert out == pytest.approx(expected, abs=_TOL)
    assert 0.0 <= out <= 1.0


def test_rskew_is_null_on_a_flat_window() -> None:
    # Flat prices -> all returns 0 -> RV 0 -> skew undefined -> null (never inf/NaN).
    flat = _close_frame([0.0, 0.0, 0.0, 0.0])
    assert _last(rskew(window=2), _CLOSE_SCHEMA, flat) is None


# --- liquidity / spread ------------------------------------------------------------------------


def test_quoted_spread_is_mean_relative_spread() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [_START + timedelta(minutes=i) for i in range(2)],
            "symbol": ["AAA"] * 2,
            "close": [10.0, 10.0],
            "bid": [9.9, 9.8],
            "ask": [10.1, 10.2],
        }
    )
    # spreads: (10.1-9.9)/10 = 0.02; (10.2-9.8)/10 = 0.04; window=2 mean = 0.03.
    out = _last(quoted_spread(window=2), _QUOTE_SCHEMA, frame)
    assert out == pytest.approx(0.03, abs=_TOL)


def test_quoted_spread_crossed_quote_yields_null() -> None:
    # A crossed quote (bid > ask) is rejected by validate(); on the trusted (validation OFF) path
    # the expression itself nulls it -- never a negative spread leaking into the rolling mean.
    frame = pl.DataFrame(
        {
            "timestamp": [_START + timedelta(minutes=i) for i in range(2)],
            "symbol": ["AAA"] * 2,
            "close": [10.0, 10.0],
            "bid": [9.9, 10.2],  # second bar crossed: bid 10.2 > ask 10.0
            "ask": [10.1, 10.0],
        }
    )
    assert _last(quoted_spread(window=2), _QUOTE_SCHEMA, frame) is None


def test_eff_spread_measures_distance_from_mid() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [_START + timedelta(minutes=i) for i in range(2)],
            "symbol": ["AAA"] * 2,
            "close": [10.1, 9.9],  # mid is 10.0 both bars
            "bid": [9.9, 9.9],
            "ask": [10.1, 10.1],
        }
    )
    # 2*|close - mid|/mid: 2*0.1/10 = 0.02 each bar; window=2 mean = 0.02.
    out = _last(eff_spread(window=2), _QUOTE_SCHEMA, frame)
    assert out == pytest.approx(0.02, abs=_TOL)


# --- L2 book imbalance (DepthRole path, not in the default registry) ----------------------------


def _depth_role(side: QuoteField, level: int) -> DepthRole:
    return DepthRole(side, level, Adjustment.RAW)


def test_book_imbalance_sums_depth_across_levels() -> None:
    # Two book levels: bid sizes (3, 1), ask sizes (1, 1). Imbalance = (4 - 2)/(4 + 2) = 1/3.
    levels = 2
    roles = {
        _depth_role(QuoteField.BID_SIZE, 0): "bs0",
        _depth_role(QuoteField.BID_SIZE, 1): "bs1",
        _depth_role(QuoteField.ASK_SIZE, 0): "as0",
        _depth_role(QuoteField.ASK_SIZE, 1): "as1",
    }
    schema = BarSchema(roles=roles)
    frame = pl.DataFrame(
        {
            "timestamp": [_START],
            "symbol": ["AAA"],
            "bs0": [3.0],
            "bs1": [1.0],
            "as0": [1.0],
            "as1": [1.0],
        }
    )
    out = _last(book_imbalance(levels=levels), schema, frame)
    assert out == pytest.approx(1.0 / 3.0, abs=_TOL)
