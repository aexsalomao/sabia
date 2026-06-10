"""Deterministic OHLCV generators + the canonical test schema (seeded; no runtime randomness).

A single-symbol series and a multi-symbol panel sharing one timestamp vector (so the panel is a
complete cross-section). All frames satisfy the input contract: sorted, unique, tz-aware-UTC
timestamps, low <= open/close <= high, positive volume. ``SCHEMA`` maps every role the shipped
features declare onto these physical columns (the same physical column backs several adjustment
roles, since the synthetic data carries one basis).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from sabia.schema import BarSchema
from sabia.typing import (
    ASK_RAW,
    ASK_SIZE_RAW,
    BID_RAW,
    BID_SIZE_RAW,
    BUY_VOLUME_RAW,
    CLOSE_RAW,
    CLOSE_SPLIT,
    CLOSE_TR,
    DVOL_RAW,
    HIGH_SPLIT,
    LOW_SPLIT,
    MARKET_RET,
    MID_RAW,
    OPEN_SPLIT,
    OPEN_TR,
    SELL_VOLUME_RAW,
    SIGNED_DOLLAR_RAW,
    SIGNED_VOLUME_RAW,
    TRADE_COUNT_RAW,
    VOLUME_RAW,
    VOLUME_SPLIT,
    VWAP_SPLIT,
    Adjustment,
    DepthRole,
    QuoteField,
)

# Physical column names (the canonical timestamp/symbol names plus arbitrary OHLCV names).
TIMESTAMP = "timestamp"
SYMBOL = "symbol"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"
VWAP = "vwap"
DOLLAR_VOLUME = "dollar_volume"
MARKET = "market_ret"
# Intraday microstructure columns (L1 quote + adapter-derived flow aggregates, FEATURES.md 13).
BID = "bid"
ASK = "ask"
MID = "mid"
BID_SIZE = "bid_size"
ASK_SIZE = "ask_size"
SIGNED_VOLUME = "signed_volume"
BUY_VOLUME = "buy_volume"
SELL_VOLUME = "sell_volume"
SIGNED_DOLLAR = "signed_dollar"
TRADE_COUNT = "trade_count"
# L2 per-level book depth (DepthRole), so the off-registry ``book_imbalance`` factory runs through
# the same cross-cutting harness as the registered features (tests/test_invariants.py).
DEPTH_LEVELS = 2
BID_SIZE_L = tuple(f"bid_size_l{level}" for level in range(DEPTH_LEVELS))
ASK_SIZE_L = tuple(f"ask_size_l{level}" for level in range(DEPTH_LEVELS))

# The market factor is common across symbols (it is one series the whole universe shares), so it is
# generated from a fixed seed independent of the per-symbol OHLCV seed -- identical for every symbol
# at a given bar, and stable across calls of the same length.
_MARKET_SEED = 12345

# One schema for the whole suite: every role resolves to its physical column. The synthetic frames
# carry a single adjustment basis, so tr / split / raw of a field all map to the same column.
SCHEMA = BarSchema(
    roles={
        CLOSE_TR: CLOSE,
        CLOSE_SPLIT: CLOSE,
        CLOSE_RAW: CLOSE,
        OPEN_TR: OPEN,
        OPEN_SPLIT: OPEN,
        HIGH_SPLIT: HIGH,
        LOW_SPLIT: LOW,
        VWAP_SPLIT: VWAP,
        VOLUME_SPLIT: VOLUME,
        VOLUME_RAW: VOLUME,
        DVOL_RAW: DOLLAR_VOLUME,
        MARKET_RET: MARKET,
        BID_RAW: BID,
        ASK_RAW: ASK,
        MID_RAW: MID,
        BID_SIZE_RAW: BID_SIZE,
        ASK_SIZE_RAW: ASK_SIZE,
        SIGNED_VOLUME_RAW: SIGNED_VOLUME,
        BUY_VOLUME_RAW: BUY_VOLUME,
        SELL_VOLUME_RAW: SELL_VOLUME,
        SIGNED_DOLLAR_RAW: SIGNED_DOLLAR,
        TRADE_COUNT_RAW: TRADE_COUNT,
        **{
            DepthRole(QuoteField.BID_SIZE, level, Adjustment.RAW): col
            for level, col in enumerate(BID_SIZE_L)
        },
        **{
            DepthRole(QuoteField.ASK_SIZE, level, Adjustment.RAW): col
            for level, col in enumerate(ASK_SIZE_L)
        },
    }
)

_START = datetime(2020, 1, 1, tzinfo=UTC)


def _timestamps(n: int, *, offset: int = 0) -> list[datetime]:
    return [_START + timedelta(days=offset + i) for i in range(n)]


def _ohlcv_columns(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    hi_base = np.maximum(open_, close)
    lo_base = np.minimum(open_, close)
    high = hi_base * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = lo_base * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.integers(100_000, 1_000_000, n).astype(np.float64)
    market_ret = np.random.default_rng(_MARKET_SEED).normal(0.0, 0.01, n)
    # Intraday L1 quote + flow aggregates. Drawn AFTER the OHLCV columns so adding them leaves the
    # existing series byte-identical (reference tests unaffected). Contract-valid by construction:
    # half_spread > 0 so bid < ask and both > 0; sizes/counts non-negative; buy+sell == volume so
    # signed_volume == buy - sell is consistent with the bar's total volume.
    half_spread = close * 0.0005 * (1.0 + np.abs(rng.normal(0.0, 0.5, n)))
    bid = close - half_spread
    ask = close + half_spread
    bid_size = rng.integers(1, 5_000, n).astype(np.float64)
    ask_size = rng.integers(1, 5_000, n).astype(np.float64)
    buy_frac = rng.uniform(0.2, 0.8, n)
    buy_volume = volume * buy_frac
    sell_volume = volume - buy_volume
    trade_count = rng.integers(1, 500, n).astype(np.float64)
    depth = {
        col: rng.integers(1, 5_000, n).astype(np.float64) for col in (*BID_SIZE_L, *ASK_SIZE_L)
    }
    return {
        OPEN: open_,
        HIGH: high,
        LOW: low,
        CLOSE: close,
        VOLUME: volume,
        VWAP: (high + low + close) / 3.0,
        DOLLAR_VOLUME: close * volume,
        MARKET: market_ret,
        BID: bid,
        ASK: ask,
        MID: (bid + ask) / 2.0,
        BID_SIZE: bid_size,
        ASK_SIZE: ask_size,
        SIGNED_VOLUME: buy_volume - sell_volume,
        BUY_VOLUME: buy_volume,
        SELL_VOLUME: sell_volume,
        SIGNED_DOLLAR: (buy_volume - sell_volume) * close,
        TRADE_COUNT: trade_count,
        **depth,
    }


def make_series(n: int, *, seed: int = 0, offset: int = 0, symbol: str = "AAA") -> pl.DataFrame:
    """A single-symbol OHLCV frame of ``n`` daily bars.

    Carries a constant ``symbol`` column: the canonical frame is a panel, and a single series is
    just a one-symbol panel, so every feature's ``.over(symbol)`` works uniformly.
    """
    return pl.DataFrame(
        {
            TIMESTAMP: _timestamps(n, offset=offset),
            SYMBOL: [symbol] * n,
            **_ohlcv_columns(n, seed),
        }
    )


def make_panel(
    n: int, *, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC"), seed: int = 0
) -> pl.DataFrame:
    """A multi-symbol panel; all symbols share one ``n``-length timestamp vector (complete XS)."""
    timestamps = _timestamps(n)
    frames = [
        pl.DataFrame(
            {
                TIMESTAMP: timestamps,
                SYMBOL: [symbol] * n,
                **_ohlcv_columns(n, seed + i),
            }
        )
        for i, symbol in enumerate(symbols)
    ]
    return pl.concat(frames).sort(SYMBOL, TIMESTAMP)


def append_future(base: pl.DataFrame, m: int, *, seed: int = 99) -> pl.DataFrame:
    """Append ``m`` future bars to a single-symbol frame (for the causality property test)."""
    future = make_series(m, seed=seed, offset=base.height)
    return pl.concat([base, future])
