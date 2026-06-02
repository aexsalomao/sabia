"""Reference-value and contract tests for the cross-sectional family.

Cross-sectional features are two-pass (per-symbol signal -> cross-sectional reduction), so they are
evaluated through ``registry.evaluate`` rather than a bare ``select``.
"""

from datetime import UTC, datetime, timedelta
from math import exp, sqrt

import numpy as np
import polars as pl
import pytest
from synthetic import CLOSE, MARKET, SCHEMA, SYMBOL, TIMESTAMP

from sabia.cross_sectional import beta, idio_vol
from sabia.registry import Registry, evaluate


def _panel(symbol_closes: dict[str, list[float]]) -> pl.DataFrame:
    frames = []
    for symbol, closes in symbol_closes.items():
        n = len(closes)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        frames.append(
            pl.DataFrame(
                {
                    TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
                    SYMBOL: [symbol] * n,
                    CLOSE: closes,
                }
            )
        )
    return pl.concat(frames).sort(SYMBOL, TIMESTAMP)


def _feature(name: str):  # type: ignore[no-untyped-def]
    return Registry.default().get(name)


def test_xs_z_mom_is_centered_across_universe() -> None:
    # 253 bars so 252/21 momentum is defined on the last bar; A up, B flat, C down.
    n = 253
    panel = _panel(
        {
            "A": [100.0 + i for i in range(n)],
            "B": [100.0] * n,
            "C": [100.0 - i * 0.2 for i in range(n)],
        }
    )
    feature = _feature("xs_z_mom_252")
    keyed = panel.select(TIMESTAMP, SYMBOL).with_columns(v=evaluate(panel, feature, SCHEMA))
    last_ts = panel.get_column(TIMESTAMP).max()
    z = keyed.filter(pl.col(TIMESTAMP) == last_ts).get_column("v")
    assert z.drop_nulls().len() == 3
    assert z.mean() == pytest.approx(0.0, abs=1e-9)


def test_xs_rank_mom_ranks_in_zero_one() -> None:
    n = 253
    panel = _panel(
        {
            "A": [100.0 + i for i in range(n)],
            "B": [100.0 + i * 0.5 for i in range(n)],
            "C": [100.0 - i * 0.2 for i in range(n)],
        }
    )
    feature = _feature("xs_rank_mom_252")
    keyed = panel.select(TIMESTAMP, SYMBOL).with_columns(v=evaluate(panel, feature, SCHEMA))
    last_ts = panel.get_column(TIMESTAMP).max()
    ranked = keyed.filter(pl.col(TIMESTAMP) == last_ts).sort(SYMBOL).get_column("v")
    # A has the strongest momentum, C the weakest; ranks are in (0, 1].
    assert ranked[0] > ranked[1] > ranked[2]
    assert ranked.min() > 0.0 and ranked.max() <= 1.0


def test_rev_1m_ranks_recent_losers_high() -> None:
    n = 22
    panel = _panel(
        {
            "A": [100.0 + i for i in range(n)],  # winner -> low reversal rank
            "B": [100.0] * n,
            "C": [100.0 - i * 0.5 for i in range(n)],  # loser -> high reversal rank
        }
    )
    feature = _feature("rev_1m_21")
    keyed = panel.select(TIMESTAMP, SYMBOL).with_columns(v=evaluate(panel, feature, SCHEMA))
    last_ts = panel.get_column(TIMESTAMP).max()
    ranked = keyed.filter(pl.col(TIMESTAMP) == last_ts).sort(SYMBOL).get_column("v")
    # Reversal = -return ranked ascending: loser C ranks above winner A.
    assert ranked[2] > ranked[0]
    assert ranked.min() > 0.0 and ranked.max() <= 1.0


# --- single-factor market model (beta / idiosyncratic vol) -------------------------------------


def _market_frame(closes: list[float], market: list[float]) -> pl.DataFrame:
    n = len(closes)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            SYMBOL: ["A"] * n,
            CLOSE: closes,
            MARKET: market,
        }
    )


def _closes_from_returns(returns: list[float], *, start: float = 100.0) -> list[float]:
    # Build a close path whose one-bar log returns equal ``returns`` (returns[0] is unused -- the
    # first bar has no prior close). Lets a test specify exact returns and read back exact features.
    closes = [start]
    for r in returns[1:]:
        closes.append(closes[-1] * exp(r))
    return closes


def test_beta_recovers_known_slope() -> None:
    # Asset returns are exactly 2x the market every bar, so the rolling OLS slope must be 2.0.
    market = [0.0, 0.01, -0.02, 0.015, 0.005, -0.01, 0.02]
    closes = _closes_from_returns([2.0 * m for m in market])
    out = _market_frame(closes, market).select(beta(window=5).expr(SCHEMA)).to_series()
    assert out[-1] == pytest.approx(2.0, abs=1e-9)


def test_idio_vol_matches_residual_std() -> None:
    # Returns are NOT proportional to the market, so the residual variance is genuinely positive;
    # compare against the population residual std computed independently over the last window.
    market = [0.0, 0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.015]
    returns = [0.0, 0.02, -0.01, 0.05, 0.0, 0.01, -0.02, 0.03]
    window = 5
    closes = _closes_from_returns(returns)
    out = _market_frame(closes, market).select(idio_vol(window=window).expr(SCHEMA)).to_series()
    r = np.array(returns[-window:])
    m = np.array(market[-window:])
    cov = float(np.mean(r * m) - np.mean(r) * np.mean(m))
    var_m = float(np.mean(m * m) - np.mean(m) ** 2)
    var_r = float(np.mean(r * r) - np.mean(r) ** 2)
    expected = sqrt(var_r - cov * cov / var_m)
    assert out[-1] == pytest.approx(expected, abs=1e-9)


def test_beta_and_idio_null_on_flat_market() -> None:
    # A flat market has zero variance -> the regression is undefined -> both features null, not inf.
    n = 10
    market = [0.0] * n
    closes = [100.0 * (1.01**i) for i in range(n)]
    frame = _market_frame(closes, market)
    assert frame.select(beta(window=5).expr(SCHEMA)).to_series()[-1] is None
    assert frame.select(idio_vol(window=5).expr(SCHEMA)).to_series()[-1] is None
