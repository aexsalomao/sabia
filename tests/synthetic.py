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
    CLOSE_RAW,
    CLOSE_SPLIT,
    CLOSE_TR,
    DVOL_RAW,
    HIGH_SPLIT,
    LOW_SPLIT,
    MARKET_RET,
    OPEN_SPLIT,
    OPEN_TR,
    VOLUME_RAW,
    VOLUME_SPLIT,
    VWAP_SPLIT,
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
    return {
        OPEN: open_,
        HIGH: high,
        LOW: low,
        CLOSE: close,
        VOLUME: volume,
        VWAP: (high + low + close) / 3.0,
        DOLLAR_VOLUME: close * volume,
        MARKET: market_ret,
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
