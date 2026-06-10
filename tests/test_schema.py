"""BarSchema convenience constructors: ohlcv (2.2) and trades/quotes (FEATURES.md 13)."""

from datetime import UTC, datetime, timedelta

import polars as pl

import sabia
from sabia.schema import BarSchema
from sabia.typing import (
    ASK_RAW,
    ASK_SIZE_RAW,
    BID_RAW,
    BID_SIZE_RAW,
    CLOSE_RAW,
    CLOSE_SPLIT,
    CLOSE_TR,
    HIGH_SPLIT,
    LOW_SPLIT,
    MID_RAW,
    OPEN_SPLIT,
    OPEN_TR,
    SIGNED_VOLUME_RAW,
    TRADE_COUNT_RAW,
    VOLUME_RAW,
    VOLUME_SPLIT,
    Adjustment,
    PriceField,
    PriceRole,
)


def test_ohlcv_maps_default_columns() -> None:
    s = BarSchema.ohlcv()
    assert s.column(OPEN_SPLIT) == "open"
    assert s.column(HIGH_SPLIT) == "high"
    assert s.column(LOW_SPLIT) == "low"
    assert s.column(CLOSE_SPLIT) == "close"
    assert s.column(VOLUME_SPLIT) == "volume"
    # close also backs the @tr return roles when no separate tr_close is given.
    assert s.column(CLOSE_TR) == "close"
    assert s.column(OPEN_TR) == "open"


def test_ohlcv_separate_tr_close() -> None:
    s = BarSchema.ohlcv(close="px", tr_close="adj_close")
    assert s.column(CLOSE_SPLIT) == "px"
    assert s.column(CLOSE_TR) == "adj_close"


def test_ohlcv_passes_through_identity_and_calendar() -> None:
    s = BarSchema.ohlcv(symbol_col="sym", timestamp_col="ts", closed_col="closed", calendar="XNYS")
    assert s.symbol_col == "sym"
    assert s.timestamp_col == "ts"
    assert s.closed_col == "closed"
    assert s.calendar == "XNYS"


def test_ohlcv_resolves_default_features_end_to_end() -> None:
    n = 30
    start = datetime(2024, 1, 1, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n)],
            "symbol": ["A"] * n,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0] * n,
        }
    )
    out = sabia.compute(
        df,
        sabia.momentum.rsi(period=14),
        sabia.volatility.vol_yz(window=5),
        schema=BarSchema.ohlcv(),
        include_keys=True,
    )
    assert out.columns == ["symbol", "timestamp", "rsi_14", "vol_yz_5"]
    assert out.height == n


# --- trades() / quotes() (intraday microstructure tier, FEATURES.md 13) ------------------------


def test_trades_maps_raw_ohlcv_and_skips_unsupplied_optionals() -> None:
    s = BarSchema.trades()
    # OHLCV resolve on the RAW basis (not split, unlike ohlcv()).
    assert s.column(PriceRole(PriceField.OPEN, Adjustment.RAW)) == "open"
    assert s.column(CLOSE_RAW) == "close"
    assert s.column(VOLUME_RAW) == "volume"
    # Optional flow columns are absent until supplied -- the role simply does not resolve.
    assert not s.has(SIGNED_VOLUME_RAW)
    assert not s.has(TRADE_COUNT_RAW)


def test_trades_maps_supplied_flow_columns() -> None:
    s = BarSchema.trades(
        close="px", signed_volume="sv", buy_volume="bv", sell_volume="xv", trade_count="n"
    )
    assert s.column(CLOSE_RAW) == "px"
    assert s.column(SIGNED_VOLUME_RAW) == "sv"
    assert s.column(TRADE_COUNT_RAW) == "n"


def test_quotes_adds_l1_on_top_of_trades() -> None:
    s = BarSchema.quotes(
        bid="b", ask="a", bid_size="bs", ask_size="as", mid="m", signed_volume="sv"
    )
    assert s.column(BID_RAW) == "b"
    assert s.column(ASK_RAW) == "a"
    assert s.column(BID_SIZE_RAW) == "bs"
    assert s.column(ASK_SIZE_RAW) == "as"
    assert s.column(MID_RAW) == "m"
    # the trade-side roles still resolve
    assert s.column(CLOSE_RAW) == "close"
    assert s.column(SIGNED_VOLUME_RAW) == "sv"


def test_quotes_skips_unsupplied_sizes_and_mid() -> None:
    s = BarSchema.quotes()
    assert s.column(BID_RAW) == "bid"
    assert s.column(ASK_RAW) == "ask"
    assert not s.has(BID_SIZE_RAW)
    assert not s.has(MID_RAW)


def test_trades_passes_through_identity_and_calendar() -> None:
    s = BarSchema.trades(symbol_col="sym", timestamp_col="ts", closed_col="closed", calendar="XNYS")
    assert s.symbol_col == "sym"
    assert s.timestamp_col == "ts"
    assert s.closed_col == "closed"
    assert s.calendar == "XNYS"
