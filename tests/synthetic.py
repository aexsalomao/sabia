"""Deterministic OHLCV generators for the test suite (seeded; no runtime randomness in sabia).

A single-symbol series and a multi-symbol panel sharing one timestamp vector (so the panel is a
complete cross-section). All frames satisfy the input contract: sorted, unique, tz-aware-UTC
timestamps, low <= open/close <= high, positive volume.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from sabia.spec import Column

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
    return {
        Column.OPEN: open_,
        Column.HIGH: high,
        Column.LOW: low,
        Column.CLOSE: close,
        Column.VOLUME: volume,
    }


def make_series(n: int, *, seed: int = 0, offset: int = 0, symbol: str = "AAA") -> pl.DataFrame:
    """A single-symbol OHLCV frame of ``n`` daily bars.

    Carries a constant ``symbol`` column: the canonical frame is a panel, and a single series is
    just a one-symbol panel, so every feature's ``.over(symbol)`` works uniformly.
    """
    return pl.DataFrame(
        {
            Column.TIMESTAMP: _timestamps(n, offset=offset),
            Column.SYMBOL: [symbol] * n,
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
                Column.TIMESTAMP: timestamps,
                Column.SYMBOL: [symbol] * n,
                **_ohlcv_columns(n, seed + i),
            }
        )
        for i, symbol in enumerate(symbols)
    ]
    return pl.concat(frames).sort(Column.SYMBOL, Column.TIMESTAMP)


def append_future(base: pl.DataFrame, m: int, *, seed: int = 99) -> pl.DataFrame:
    """Append ``m`` future bars to a single-symbol frame (for the causality property test)."""
    future = make_series(m, seed=seed, offset=base.height)
    return pl.concat([base, future])
