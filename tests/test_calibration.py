"""Calibration (FEATURES.md 9): the cross-sectional momentum feature must detect the effect it is
designed for on a known local sample. Integration-marked, deterministic, no network.

We build a panel with an embedded persistent drift per symbol, so past momentum predicts future
return. A working ``xs_rank_mom`` should then have a positive information coefficient (the Spearman
correlation between today's rank and next period's return), averaged over time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from synthetic import CLOSE, SCHEMA, SYMBOL, TIMESTAMP

from sabia.registry import Registry, evaluate

pytestmark = pytest.mark.integration

_N_SYMBOLS = 24
_N_DAYS = 400


def _panel_with_embedded_momentum() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    # Strong persistent per-symbol drift relative to noise, so momentum is detectable in next-day
    # returns; this is a calibration sample, not a realistic market.
    drifts = np.linspace(-0.003, 0.003, _N_SYMBOLS)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(days=i) for i in range(_N_DAYS)]
    frames = []
    for sym_id, drift in enumerate(drifts):
        noise = rng.normal(0.0, 0.005, _N_DAYS)
        close = 100.0 * np.exp(np.cumsum(drift + noise))
        frames.append(
            pl.DataFrame(
                {
                    TIMESTAMP: timestamps,
                    SYMBOL: [f"S{sym_id:02d}"] * _N_DAYS,
                    CLOSE: close,
                }
            )
        )
    return pl.concat(frames).sort(SYMBOL, TIMESTAMP)


def _information_coefficient(panel: pl.DataFrame, rank: pl.Series) -> float:
    keyed = panel.select(TIMESTAMP, SYMBOL, CLOSE).with_columns(rank=rank)
    # Next-period return per symbol (lookahead is fine here: this is evaluation, not a feature).
    keyed = keyed.with_columns(
        fwd_ret=(pl.col(CLOSE).shift(-1) / pl.col(CLOSE) - 1).over(SYMBOL)
    ).drop_nulls(["rank", "fwd_ret"])
    daily_ic = (
        keyed.group_by(TIMESTAMP)
        .agg(pl.corr("rank", "fwd_ret", method="spearman").alias("ic"))
        .get_column("ic")
        .drop_nulls()
    )
    return float(daily_ic.mean())


def test_xs_momentum_has_positive_information_coefficient() -> None:
    panel = _panel_with_embedded_momentum()
    feature = Registry.default().get("xs_rank_mom_252")
    rank = evaluate(panel, feature, SCHEMA)
    assert rank.len() == panel.height
    ic = _information_coefficient(panel, rank)
    assert ic > 0.05, f"cross-sectional momentum IC too low: {ic:.3f}"
