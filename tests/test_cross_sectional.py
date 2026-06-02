"""Reference-value and contract tests for the cross-sectional family.

Cross-sectional features are two-pass (per-symbol signal -> cross-sectional reduction), so they are
evaluated through ``registry.evaluate`` rather than a bare ``select``.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import CLOSE, SCHEMA, SYMBOL, TIMESTAMP

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
