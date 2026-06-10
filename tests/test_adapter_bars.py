"""Tick->bar aggregation (FEATURES.md 13): hand-checked boundaries, OHLCV/vwap/flow, causality.

The adapter is a pure transform, not a BoundFeature, so the registry-parametrized invariant suites
never see it -- these are its dedicated reference, causality, determinism, and degenerate gates.
Bars round-trip through ``sabia.validate`` so the output provably satisfies the input contract.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

import sabia
from sabia.adapters.bars import BarKind, BarSpec, build_bars
from sabia.schema import BarSchema

_TOL = 1e-9
_START = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _ticks(
    prices: list[float], sizes: list[float], *, symbol: str = "AAA", step_s: int = 1
) -> pl.DataFrame:
    n = len(prices)
    return pl.DataFrame(
        {
            "timestamp": [_START + timedelta(seconds=step_s * i) for i in range(n)],
            "symbol": [symbol] * n,
            "price": prices,
            "size": sizes,
        }
    )


def _bars(ticks: pl.DataFrame, spec: BarSpec) -> pl.DataFrame:
    return build_bars(ticks, spec).collect()


# --- volume bars: boundaries + OHLCV + vwap ----------------------------------------------------


def test_volume_bars_cut_on_exclusive_running_total() -> None:
    # sizes 4,4,4,4 threshold 10: prior totals 0,4,8,12 -> buckets 0,0,0,1. The boundary-crossing
    # tick (#2, which pushes cum to 12) completes bar 0; tick #3 opens bar 1.
    ticks = _ticks([10.0, 11.0, 12.0, 20.0], [4.0, 4.0, 4.0, 4.0])
    bars = _bars(ticks, BarSpec(kind=BarKind.VOLUME, threshold=10.0))
    assert bars.height == 2
    row0 = bars.row(0, named=True)
    assert (row0["open"], row0["high"], row0["low"], row0["close"]) == (10.0, 12.0, 10.0, 12.0)
    assert row0["volume"] == 12.0
    assert row0["trade_count"] == 3
    # vwap = (10*4 + 11*4 + 12*4) / 12 = 11.0
    assert row0["vwap"] == pytest.approx(11.0, abs=_TOL)
    # observation timestamp = last tick time in the bar (tick #2, +2s).
    assert row0["timestamp"] == _START + timedelta(seconds=2)
    row1 = bars.row(1, named=True)
    assert (row1["open"], row1["close"], row1["volume"]) == (20.0, 20.0, 4.0)


def test_dollar_bars_weight_by_notional() -> None:
    # weight = price*size: 10*1,10*1,10*1 -> prior 0,10,20 with threshold 15 -> buckets 0,0,1.
    ticks = _ticks([10.0, 10.0, 10.0], [1.0, 1.0, 1.0])
    bars = _bars(ticks, BarSpec(kind=BarKind.DOLLAR, threshold=15.0))
    assert bars.height == 2
    assert bars.row(0, named=True)["trade_count"] == 2
    assert bars.row(1, named=True)["trade_count"] == 1


def test_tick_bars_cut_every_n_trades() -> None:
    ticks = _ticks([10.0, 11.0, 12.0, 13.0, 14.0], [1.0, 1.0, 1.0, 1.0, 1.0])
    bars = _bars(ticks, BarSpec(kind=BarKind.TICK, n_ticks=2))
    # 5 ticks, 2 per bar -> bars of 2,2,1.
    assert bars.get_column("trade_count").to_list() == [2, 2, 1]
    assert bars.get_column("open").to_list() == [10.0, 12.0, 14.0]
    assert bars.get_column("close").to_list() == [11.0, 13.0, 14.0]


def test_time_bars_group_by_fixed_interval() -> None:
    # ticks every 30s; 2-minute bars -> 4 ticks per bar.
    ticks = _ticks([10.0, 11.0, 12.0, 13.0, 14.0], [1.0] * 5, step_s=30)
    bars = _bars(ticks, BarSpec(kind=BarKind.TIME, interval="2m"))
    assert bars.height == 2
    assert bars.row(0, named=True)["trade_count"] == 4
    # observation timestamp is the last tick within the bar (the 4th tick, +90s).
    assert bars.row(0, named=True)["timestamp"] == _START + timedelta(seconds=90)


# --- signed flow aggregates --------------------------------------------------------------------


def test_signed_flow_splits_buy_and_sell_volume() -> None:
    from sabia.adapters.bars import SignRule

    # tick rule: diffs -, +1, -1, +1 -> signs 0,+1,-1,+1. One volume bar (threshold huge).
    ticks = _ticks([10.0, 11.0, 10.0, 11.0], [2.0, 3.0, 5.0, 4.0])
    spec = BarSpec(kind=BarKind.VOLUME, threshold=1_000.0, sign_rule=SignRule.TICK_RULE)
    bar = _bars(ticks, spec).row(0, named=True)
    # buy = sizes where sign>0 = 3 + 4 = 7; sell = size where sign<0 = 5; first tick sign 0.
    assert bar["buy_volume"] == 7.0
    assert bar["sell_volume"] == 5.0
    # signed_volume = +3 -5 +4 = +2 (the sign-0 first tick contributes 0).
    assert bar["signed_volume"] == 2.0


def test_none_rule_yields_zero_signed_volume() -> None:
    ticks = _ticks([10.0, 11.0, 12.0], [1.0, 1.0, 1.0])
    bar = _bars(ticks, BarSpec(kind=BarKind.VOLUME, threshold=1_000.0)).row(0, named=True)
    assert bar["signed_volume"] == 0.0
    assert bar["buy_volume"] == 0.0
    assert bar["sell_volume"] == 0.0


# --- causality, isolation, determinism ---------------------------------------------------------


def test_appending_future_ticks_leaves_completed_bars_unchanged() -> None:
    # The bars formed entirely from a prefix of ticks must be byte-identical whether or not future
    # ticks follow -- the adapter never looks ahead (FEATURES.md 8.1).
    prefix = _ticks([10.0, 11.0, 12.0, 13.0], [4.0, 4.0, 4.0, 4.0])
    future = _ticks([20.0, 21.0], [4.0, 4.0]).with_columns(
        pl.col("timestamp") + timedelta(seconds=100)
    )
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0)
    prefix_bars = _bars(prefix, spec)
    full_bars = _bars(pl.concat([prefix, future]), spec)
    # Every bar the prefix marks closed is final: byte-identical in the full run. The prefix's
    # trailing open bar (closed == False) is the only one future ticks may revise.
    prefix_closed = prefix_bars.filter(pl.col("closed"))
    assert prefix_closed.height > 0
    assert prefix_closed.equals(full_bars.head(prefix_closed.height))


def test_volume_bars_do_not_bleed_across_symbols() -> None:
    a = _ticks([10.0, 11.0, 12.0], [4.0, 4.0, 4.0], symbol="AAA")
    b = _ticks([20.0, 21.0], [4.0, 4.0], symbol="BBB")
    bars = _bars(pl.concat([a, b]), BarSpec(kind=BarKind.VOLUME, threshold=10.0))
    counts = {
        s: bars.filter(pl.col("symbol") == s).get_column("volume").to_list() for s in ("AAA", "BBB")
    }
    # AAA: 12 then 0-left... 3 ticks*4=12 -> one full bar (12). BBB: 2 ticks*4=8 -> one partial bar.
    assert counts["AAA"] == [12.0]
    assert counts["BBB"] == [8.0]


def test_build_bars_is_deterministic_and_order_independent() -> None:
    ticks = _ticks([10.0, 11.0, 12.0, 13.0], [4.0, 4.0, 4.0, 4.0])
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0)
    first = _bars(ticks, spec)
    shuffled = ticks.sample(fraction=1.0, shuffle=True, seed=7)  # adapter sorts internally
    assert first.equals(_bars(shuffled, spec))


def test_tied_timestamps_are_deterministic_in_tape_order() -> None:
    # Simultaneous ticks are legal (validate_ticks allows ties); the stable sort keeps them in
    # input (tape) order, so repeated runs on the same frame are byte-identical.
    ticks = _ticks([10.0, 11.0, 12.0, 13.0], [1.0, 1.0, 1.0, 1.0], step_s=0)
    spec = BarSpec(kind=BarKind.TICK, n_ticks=2)
    first = _bars(ticks, spec)
    assert first.equals(_bars(ticks, spec))
    # Tape order defines open/close: all four ticks share one timestamp, so they form ONE bar
    # (tie-atomic buckets) whose open/close are the first/last tick in input order.
    assert first.height == 1
    row = first.row(0, named=True)
    assert (row["open"], row["close"]) == (10.0, 13.0)


def test_tied_timestamps_never_split_across_bars() -> None:
    # A bucket boundary inside a timestamp tie would emit duplicate (symbol, timestamp) bars --
    # which sabia.validate would then reject. Ties are atomic: the boundary moves to the next
    # distinct timestamp, keeping bar timestamps strictly increasing per symbol.
    # Sizes 6,6,6 at ONE timestamp: prior totals 0,6,12 would put the third tick in bucket 1,
    # splitting the tie into two bars at the same timestamp.
    tied = _ticks([10.0, 11.0, 12.0], [6.0, 6.0, 6.0], step_s=0)
    later = _ticks([13.0, 14.0], [4.0, 4.0]).with_columns(
        pl.col("timestamp") + timedelta(seconds=5)
    )
    bars = _bars(pl.concat([tied, later]), BarSpec(kind=BarKind.VOLUME, threshold=10.0))
    assert bars.get_column("timestamp").n_unique() == bars.height
    # The tied ticks (18 shares, crossing the 10-share threshold mid-tie) stay in one bar.
    assert bars.get_column("volume").to_list()[0] == 18.0


def test_last_bar_per_symbol_is_open_the_rest_closed() -> None:
    # A bar is provably complete only once a strictly later tick exists for its symbol (a future
    # tick may legally share the last bar's timestamp and join it), so exactly the final bar of
    # each symbol carries closed == False.
    a = _ticks([10.0, 11.0, 12.0], [4.0, 4.0, 4.0], symbol="AAA")
    b = _ticks([20.0, 21.0], [4.0, 4.0], symbol="BBB")
    bars = _bars(pl.concat([a, b]), BarSpec(kind=BarKind.TICK, n_ticks=2))
    closed = {
        s: bars.filter(pl.col("symbol") == s).get_column("closed").to_list() for s in ("AAA", "BBB")
    }
    # AAA: bars of 2,1 ticks -> [closed, open]; BBB: one (partial) bar -> [open].
    assert closed["AAA"] == [True, False]
    assert closed["BBB"] == [False]


def test_build_bars_requires_a_symbol_column() -> None:
    from sabia.validate import SabiaValidationError

    # validate_ticks accepts a symbol-less feed, but aggregation groups by symbol -- the precise
    # boundary error, never a raw Polars ColumnNotFoundError from inside the lazy plan.
    no_symbol = pl.DataFrame(
        {
            "timestamp": [_START, _START + timedelta(seconds=1)],
            "price": [10.0, 11.0],
            "size": [1.0, 1.0],
        }
    )
    with pytest.raises(SabiaValidationError, match="symbol"):
        build_bars(no_symbol, BarSpec(kind=BarKind.VOLUME, threshold=10.0))


# --- degenerate inputs -------------------------------------------------------------------------


def test_empty_ticks_yield_empty_bars_with_canonical_columns() -> None:
    empty = _ticks([], []).clear()
    bars = _bars(empty, BarSpec(kind=BarKind.VOLUME, threshold=10.0))
    assert bars.height == 0
    assert "open" in bars.columns and "signed_volume" in bars.columns and "closed" in bars.columns


def test_zero_volume_bar_yields_null_vwap_never_inf() -> None:
    # All-zero sizes -> bar volume 0 -> vwap is null (safe_div), never inf/NaN.
    ticks = _ticks([10.0, 11.0], [0.0, 0.0])
    bars = _bars(ticks, BarSpec(kind=BarKind.TICK, n_ticks=2))
    assert bars.row(0, named=True)["vwap"] is None


# --- contract round-trip -----------------------------------------------------------------------


def test_bars_satisfy_the_input_contract() -> None:
    from sabia.adapters.bars import SignRule
    from sabia.validate import SabiaValidationError

    ticks = _ticks([10.0, 11.0, 12.0, 11.0, 13.0, 12.0], [4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0, sign_rule=SignRule.TICK_RULE)
    bars = build_bars(ticks, spec).collect()
    schema = BarSchema.trades(
        signed_volume="signed_volume",
        buy_volume="buy_volume",
        sell_volume="sell_volume",
        trade_count="trade_count",
        vwap="vwap",
    )
    # The trailing in-progress bar (closed == False) is caught by the finalization gate that
    # trades() arms by default -- the raw adapter output is NOT point-in-time safe as-is.
    with pytest.raises(SabiaValidationError, match="not final"):
        sabia.validate(bars, schema=schema)
    # Filtering to completed bars satisfies the full contract; a clean return is the assertion.
    assert sabia.validate(bars.filter(pl.col("closed")), schema=schema) == []


def test_build_bars_validate_strict_rejects_bad_ticks() -> None:
    from sabia.spec import ValidationMode
    from sabia.validate import SabiaValidationError

    # Out-of-order timestamps: build_bars(validate=STRICT) runs the tick contract before building.
    bad = pl.DataFrame(
        {
            "timestamp": [_START + timedelta(seconds=2), _START + timedelta(seconds=1)],
            "symbol": ["AAA", "AAA"],
            "price": [10.0, 11.0],
            "size": [4.0, 4.0],
        }
    )
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0)
    with pytest.raises(SabiaValidationError, match="non-decreasing"):
        build_bars(bad, spec, validate=ValidationMode.STRICT).collect()
