"""BarSchema.ohlcv convenience constructor (FEATURES.md 2.2)."""

from datetime import UTC, datetime, timedelta

import polars as pl

import sabia
from sabia.schema import BarSchema
from sabia.typing import (
    CLOSE_SPLIT,
    CLOSE_TR,
    HIGH_SPLIT,
    LOW_SPLIT,
    OPEN_SPLIT,
    OPEN_TR,
    VOLUME_SPLIT,
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
