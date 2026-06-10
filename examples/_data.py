"""Shared synthetic market data for the sabia examples — deterministic, no network.

Every frame here already satisfies sabia's input contract (see ``03_validation.py``): sorted and
unique tz-aware UTC timestamps, per-symbol ordering, OHLC bounds, positive volume, and — for the
panel — a complete cross-section (all symbols present at every timestamp). A common market-return
factor rides along so the market-model features (``beta``, ``idio_vol``) have something to regress.

These helpers exist only to make the examples runnable on their own. In a real pipeline the frame
comes from your data layer (e.g. ``marketgoblin``); sabia neither fetches nor adjusts anything.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

# The examples print Polars tables, whose box-drawing glyphs need a UTF-8 stdout. Windows consoles
# default to cp1252 and would raise UnicodeEncodeError, so make stdout UTF-8 here. This is an
# example-only convenience — sabia itself never touches stdout.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sabia import (
    Adjustment,
    BarSchema,
    FactorRole,
    PriceField,
    PriceRole,
    VolumeField,
    VolumeRole,
)

_START = datetime(2021, 1, 1, tzinfo=UTC)
# The market factor is one series the whole universe shares, so it is generated from a fixed seed
# independent of the per-symbol price seed: identical for every symbol at a given bar.
_MARKET_SEED = 12345


def _timestamps(n: int) -> list[datetime]:
    return [_START + timedelta(days=i) for i in range(n)]


def _ohlcv(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(np.float64)
    market_ret = np.random.default_rng(_MARKET_SEED).normal(0.0003, 0.009, n)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": (high + low + close) / 3.0,
        "dollar_volume": close * volume,
        "market_ret": market_ret,
    }


def make_ohlcv(n: int = 400, *, seed: int = 0, symbol: str = "AAA") -> pl.DataFrame:
    """A single-symbol daily OHLCV frame of ``n`` bars."""
    return pl.DataFrame({"timestamp": _timestamps(n), "symbol": [symbol] * n, **_ohlcv(n, seed)})


def make_panel(
    n: int = 400, *, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"), seed: int = 0
) -> pl.DataFrame:
    """A multi-symbol panel; all symbols share one timestamp vector (a complete cross-section)."""
    timestamps = _timestamps(n)
    frames = [
        pl.DataFrame({"timestamp": timestamps, "symbol": [sym] * n, **_ohlcv(n, seed + i)})
        for i, sym in enumerate(symbols)
    ]
    return pl.concat(frames).sort("symbol", "timestamp")


def make_trades(n: int = 5_000, *, seed: int = 0, symbol: str = "AAA") -> pl.DataFrame:
    """A stream of ``n`` raw trade ticks (price + size) for one symbol -- the adapter's input.

    Ticks arrive at irregular sub-second times (many can share a microsecond, as on a real tape) and
    follow a random walk in price. This is what ``sabia.adapters.build_bars`` aggregates into the
    intraday bars the microstructure family consumes. Deterministic; no network.
    """
    rng = np.random.default_rng(seed)
    # Irregular arrivals: cumulative integer microsecond gaps (0 gaps -> simultaneous ticks).
    gaps_us = rng.integers(0, 500_000, n).cumsum()
    timestamps = [_START + timedelta(microseconds=int(g)) for g in gaps_us]
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0002, n)))
    size = rng.integers(1, 500, n).astype(np.float64)
    return pl.DataFrame(
        {"timestamp": timestamps, "symbol": [symbol] * n, "price": price, "size": size}
    )


def default_schema() -> BarSchema:
    """A BarSchema mapping every role the shipped features need onto our physical columns.

    The synthetic data carries a single adjustment basis, so ``@tr`` / ``@split`` / ``@raw`` of a
    field all point at the same physical column. ``02_roles_and_adjustment.py`` shows the realistic
    case where those bases are *different* columns.
    """
    return BarSchema(
        roles={
            PriceRole(PriceField.OPEN, Adjustment.TR): "open",
            PriceRole(PriceField.OPEN, Adjustment.SPLIT): "open",
            PriceRole(PriceField.HIGH, Adjustment.SPLIT): "high",
            PriceRole(PriceField.LOW, Adjustment.SPLIT): "low",
            PriceRole(PriceField.CLOSE, Adjustment.TR): "close",
            PriceRole(PriceField.CLOSE, Adjustment.SPLIT): "close",
            PriceRole(PriceField.CLOSE, Adjustment.RAW): "close",
            PriceRole(PriceField.VWAP, Adjustment.SPLIT): "vwap",
            VolumeRole(VolumeField.VOLUME, Adjustment.SPLIT): "volume",
            VolumeRole(VolumeField.VOLUME, Adjustment.RAW): "volume",
            VolumeRole(VolumeField.DOLLAR_VOLUME, Adjustment.RAW): "dollar_volume",
            FactorRole.MARKET_RET: "market_ret",
        }
    )
