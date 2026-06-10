"""Trade-signing rules for the tick->bar adapter (FEATURES.md 13).

Hand-checked reference tables for the tick rule (zero-tick forward-fill) and Lee-Ready (vs the quote
midpoint, tick-rule fallback at the mid). Signing is strictly causal: a trade's sign depends only on
itself and earlier trades.
"""

from datetime import UTC, datetime, timedelta

import polars as pl

from sabia.adapters.bars import BarKind, BarSpec, SignRule
from sabia.adapters.signing import SIGN_COLUMN, sign_ticks

_VOL = BarSpec(kind=BarKind.VOLUME, threshold=10.0)


def _ticks(prices: list[float], *, bids=None, asks=None, symbol="AAA") -> pl.DataFrame:
    n = len(prices)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    data = {
        "timestamp": [start + timedelta(seconds=i) for i in range(n)],
        "symbol": [symbol] * n,
        "price": prices,
        "size": [1.0] * n,
    }
    if bids is not None:
        data["bid"] = bids
        data["ask"] = asks
    return pl.DataFrame(data)


def _signs(df: pl.DataFrame, spec: BarSpec) -> list[int]:
    out = sign_ticks(df.lazy(), spec).select(SIGN_COLUMN).collect().to_series()
    return out.to_list()


def test_none_rule_signs_every_tick_zero() -> None:
    df = _ticks([10.0, 11.0, 9.0])
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0, sign_rule=SignRule.NONE)
    assert _signs(df, spec) == [0, 0, 0]


def test_tick_rule_zero_tick_carries_prior_sign() -> None:
    # diffs: -, +1, 0, -1, +2 -> raw: null,+1,null,-1,+1 -> ffill+fill0: 0,+1,+1,-1,+1.
    df = _ticks([10.0, 11.0, 11.0, 10.0, 12.0])
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0, sign_rule=SignRule.TICK_RULE)
    assert _signs(df, spec) == [0, 1, 1, -1, 1]


def test_tick_rule_first_tick_defaults_to_zero() -> None:
    df = _ticks([10.0, 9.0])  # first has no prior -> 0; second is a downtick -> -1.
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0, sign_rule=SignRule.TICK_RULE)
    assert _signs(df, spec) == [0, -1]


def test_lee_ready_classifies_against_the_midpoint() -> None:
    # mids = 10. price>mid -> +1; price<mid -> -1; price==mid -> tick-rule fallback.
    # prices: 11(>),9(<),10(==, fallback: diff +1 from 9 -> +1),10(==, diff 0 -> carry +1).
    df = _ticks(
        [11.0, 9.0, 10.0, 10.0],
        bids=[9.5, 9.5, 9.5, 9.5],
        asks=[10.5, 10.5, 10.5, 10.5],
    )
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0, sign_rule=SignRule.LEE_READY)
    assert _signs(df, spec) == [1, -1, 1, 1]


def test_signing_is_per_symbol_no_bleed() -> None:
    # Two symbols interleaved: each symbol's tick rule must use only its own prior trades.
    a = _ticks([10.0, 11.0], symbol="AAA")
    b = _ticks([20.0, 19.0], symbol="BBB")
    df = pl.concat([a, b])
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10.0, sign_rule=SignRule.TICK_RULE)
    signed = sign_ticks(df.lazy(), spec).sort("symbol", "timestamp").collect()
    by_symbol = {
        s: signed.filter(pl.col("symbol") == s).get_column(SIGN_COLUMN).to_list()
        for s in ("AAA", "BBB")
    }
    assert by_symbol == {"AAA": [0, 1], "BBB": [0, -1]}
