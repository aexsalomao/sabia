"""Synthetic price processes that *bake in* a stylized fact, for the notebook gallery.

``examples/_data.py`` produces a near-random walk — fine for an API tour, but on a random walk
volatility does not cluster, returns are thin-tailed, and there is no momentum. To *show* those
well-known effects honestly, each generator here simulates a return process that contains the effect
on purpose (GARCH variance, Student-t innovations, an autoregressive trend), then lets a sabia
feature reveal it. The simulation is the teaching device; in a real pipeline the frame comes from
your data layer and sabia neither fetches nor adjusts anything.

Every frame returned here satisfies sabia's STRICT input contract: sorted, unique, tz-aware UTC
timestamps; per-symbol ordering; OHLC bounds; positive volume; and — for the panel — a complete
cross-section with one shared market-return factor so ``beta`` / ``idio_vol`` have something to
regress. Everything is seeded and offline.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Reuse the one schema the shipped features expect. _data.py sits one level up by the scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _data import default_schema  # noqa: E402  (re-exported for the notebooks' convenience)

__all__ = [
    "default_schema",
    "fat_tailed_ohlcv",
    "garch_ohlcv",
    "momentum_panel",
    "seasonal_ohlcv",
]

_START = datetime(2021, 1, 1, tzinfo=UTC)
# The market factor is one series the whole universe shares (FEATURES.md market model), generated
# from a fixed seed independent of any per-symbol seed so it is identical at every bar across names.
_MARKET_SEED = 12345


def _timestamps(n: int) -> list[datetime]:
    return [_START + timedelta(days=i) for i in range(n)]


def _market_ret(n: int) -> np.ndarray:
    return np.random.default_rng(_MARKET_SEED).normal(0.0003, 0.009, n)


def _wrap_close(
    close: np.ndarray, rng: np.random.Generator, market_ret: np.ndarray
) -> dict[str, np.ndarray]:
    """Build a contract-valid OHLCV record around a close path (mirrors ``_data._ohlcv``)."""
    n = close.shape[0]
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(np.float64)
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


def _frame(close: np.ndarray, rng: np.random.Generator, *, symbol: str) -> pl.DataFrame:
    n = close.shape[0]
    record = _wrap_close(close, rng, _market_ret(n))
    return pl.DataFrame({"timestamp": _timestamps(n), "symbol": [symbol] * n, **record})


def _close_from_returns(returns: np.ndarray, *, start: float = 100.0) -> np.ndarray:
    path: np.ndarray = start * np.exp(np.cumsum(returns))
    return path


def garch_ohlcv(
    n: int = 750,
    *,
    omega: float = 2.0e-6,
    alpha: float = 0.10,
    beta: float = 0.88,
    leverage: float = 0.06,
    seed: int = 7,
    symbol: str = "GRCH",
) -> pl.DataFrame:
    """A GJR-GARCH(1,1) close path: variance clusters and reacts more to down-shocks (leverage).

    ``sigma2_t = omega + (alpha + leverage*1[eps_{t-1}<0]) * eps_{t-1}^2 + beta*sigma2_{t-1}``.
    ``alpha + beta`` near 1 gives the slow-decaying volatility persistence seen in real returns
    (Bollerslev 1986); ``leverage > 0`` makes downside moves raise tomorrow's variance more than
    upside moves of the same size (Glosten-Jagannathan-Runkle 1993), so downside vol estimators run
    hot relative to upside.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    eps = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-6)
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        shock = eps[t - 1] ** 2
        lev = leverage if eps[t - 1] < 0.0 else 0.0
        sigma2[t] = omega + (alpha + lev) * shock + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    returns = 0.0003 + eps
    return _frame(_close_from_returns(returns), rng, symbol=symbol)


def fat_tailed_ohlcv(
    n: int = 750,
    *,
    df: float = 5.0,
    daily_vol: float = 0.012,
    seed: int = 11,
    symbol: str = "FATT",
) -> pl.DataFrame:
    """A close path driven by Student-t innovations — leptokurtic, with heavier tails than Gaussian.

    Innovations are scaled to a unit-variance Student-t with ``df`` degrees of freedom, so excess
    kurtosis is ``6 / (df - 4)`` for ``df > 4`` (Fama 1965, Mandelbrot 1963: real returns are fat
    tailed). Lower ``df`` -> fatter tails.
    """
    rng = np.random.default_rng(seed)
    t = rng.standard_t(df, n)
    z = t / np.sqrt(df / (df - 2.0))  # rescale to unit variance
    returns = 0.0003 + daily_vol * z
    return _frame(_close_from_returns(returns), rng, symbol=symbol)


def momentum_panel(
    n: int = 600,
    *,
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"),
    trend_strength: float = 0.16,
    seed: int = 3,
) -> pl.DataFrame:
    """A complete panel ordered winner -> loser, sharing one market factor.

    Symbols are ordered from strongest to weakest. Each name's idiosyncratic return is
    ``r_t = mu + beta * market_t + s_t`` with an AR(1) signal ``s_t = phi * s_{t-1} + noise``. Two
    independent gradients run across the universe:

    - ``mu`` (permanent drift) from ``+0.0010``/bar down to ``-0.0010``/bar — genuine
      cross-sectional winners vs losers, so ranking on past return picks names that keep
      outperforming (Jegadeesh-Titman 1993 momentum).
    - ``phi`` from ``+trend_strength`` (positive return autocorrelation -> variance ratio > 1)
      down to ``-trend_strength`` (mean-reversion, VR < 1) — the Lo-MacKinlay (1988) variance-ratio
      spectrum.

    Names also carry a spread of market betas so the factor-model features (``beta``, ``idio_vol``)
    vary across the cross-section.
    """
    market = _market_ret(n)
    phis = np.linspace(trend_strength, -trend_strength, len(symbols))
    drifts = np.linspace(0.0010, -0.0010, len(symbols))
    betas = np.linspace(0.6, 1.4, len(symbols))
    frames = []
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + i)
        phi = float(phis[i])
        noise = rng.normal(0.0, 0.010, n)
        signal = np.empty(n)
        signal[0] = noise[0]
        for t in range(1, n):
            signal[t] = phi * signal[t - 1] + noise[t]
        returns = drifts[i] + betas[i] * market + signal
        close = _close_from_returns(returns)
        record = _wrap_close(close, rng, market)
        frames.append(pl.DataFrame({"timestamp": _timestamps(n), "symbol": [sym] * n, **record}))
    return pl.concat(frames).sort("symbol", "timestamp")


def seasonal_ohlcv(
    n: int = 750,
    *,
    dow_effect: tuple[float, ...] = (-0.0015, 0.0002, 0.0004, 0.0004, 0.0012, 0.0, 0.0),
    daily_vol: float = 0.010,
    seed: int = 5,
    symbol: str = "SEAS",
) -> pl.DataFrame:
    """A close path with a day-of-week drift baked in (Monday=0 .. Sunday=6).

    ``dow_effect[w]`` is added to the mean return on weekday ``w`` — the calendar anomalies surveyed
    by French (1980) / Lakonishok-Smidt (1988). ``season_dow`` recovers the weekday so the effect is
    visible as a difference in mean return across weekdays.
    """
    rng = np.random.default_rng(seed)
    timestamps = _timestamps(n)
    weekday = np.array([ts.weekday() for ts in timestamps])
    drift = np.array(dow_effect)[weekday]
    returns = drift + rng.normal(0.0, daily_vol, n)
    close = _close_from_returns(returns)
    record = _wrap_close(close, rng, _market_ret(n))
    return pl.DataFrame({"timestamp": timestamps, "symbol": [symbol] * n, **record})
