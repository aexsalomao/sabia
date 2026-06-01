"""Calibration (FEATURES.md 8.4): the cross-sectional momentum feature must detect the effect it
is designed for on a known local sample. Integration-marked, deterministic, no network.

We build a panel with an embedded persistent drift per symbol, so past momentum predicts future
return. A working xs_rank_mom should then have a positive information coefficient (the Spearman
correlation between today's rank and next period's return), averaged over time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from sabia.cross_sectional import momentum_signal
from sabia.normalize import xs_rank
from sabia.registry import XS_SIGNAL_COLUMN, Registry, evaluate
from sabia.spec import Column

pytestmark = pytest.mark.integration

_N_SYMBOLS = 24
_N_DAYS = 400
_MOMENTUM_WINDOW = 126  # use a registered-style window short enough to leave evaluation history


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
                    Column.TIMESTAMP: timestamps,
                    Column.SYMBOL: [f"S{sym_id:02d}"] * _N_DAYS,
                    Column.CLOSE: close,
                }
            )
        )
    return pl.concat(frames).sort(Column.SYMBOL, Column.TIMESTAMP)


def _information_coefficient(panel: pl.DataFrame, rank: pl.Series) -> float:
    keyed = panel.select(Column.TIMESTAMP, Column.SYMBOL, Column.CLOSE).with_columns(rank=rank)
    # Next-period return per symbol (lookahead is fine here: this is evaluation, not a feature).
    keyed = keyed.with_columns(
        fwd_ret=(pl.col(Column.CLOSE).shift(-1) / pl.col(Column.CLOSE) - 1).over(Column.SYMBOL)
    ).drop_nulls(["rank", "fwd_ret"])
    daily_ic = (
        keyed.group_by(Column.TIMESTAMP)
        .agg(pl.corr("rank", "fwd_ret", method="spearman").alias("ic"))
        .get_column("ic")
        .drop_nulls()
    )
    return float(daily_ic.mean())


def test_xs_momentum_has_positive_information_coefficient() -> None:
    panel = _panel_with_embedded_momentum()
    # Cross-sectional rank of momentum at the calibration window (two-pass: signal, then rank).
    rank = (
        panel.lazy()
        .with_columns(momentum_signal(window=_MOMENTUM_WINDOW).alias(XS_SIGNAL_COLUMN))
        .select(xs_rank(pl.col(XS_SIGNAL_COLUMN)))
        .collect()
        .to_series()
    )
    ic = _information_coefficient(panel, rank)
    assert ic > 0.05, f"cross-sectional momentum IC too low: {ic:.3f}"
    # Sanity: the shipped feature evaluates on this panel without error.
    feature = Registry.default().get("xs_rank_mom_252")
    assert evaluate(panel, feature).len() == panel.height
