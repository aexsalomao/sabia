"""Role-resolution and universe-contract tests (FEATURES.md 2.2, 2.5, 4.3).

These guard the headline of the v5 redesign -- "roles, not columns" -- which the shared single-basis
SCHEMA cannot exercise (it maps every adjustment to one column). Here @tr and @split resolve to
DIFFERENT physical columns, so a feature reading the wrong adjustment is caught. Also covers
review findings #6 (compute must enforce a declared universe) and #1 (XS output naming).
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import SCHEMA, make_panel

import sabia
from sabia._math import log_return
from sabia.schema import BarSchema
from sabia.typing import (
    CLOSE_SPLIT,
    CLOSE_TR,
    HIGH_SPLIT,
    LOW_SPLIT,
    OPEN_SPLIT,
    VOLUME_SPLIT,
)
from sabia.validate import SabiaValidationError

# close@tr and close@split resolve to DISTINCT physical columns carrying different values.
_SPLIT_SCHEMA = BarSchema(
    roles={
        CLOSE_TR: "c_tr",
        CLOSE_SPLIT: "c_split",
        OPEN_SPLIT: "o",
        HIGH_SPLIT: "h",
        LOW_SPLIT: "low",
        VOLUME_SPLIT: "v",
    }
)


def _dual_basis_frame(n: int = 30) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    tr = [100.0 + i for i in range(n)]
    split = [50.0 + 2 * i for i in range(n)]  # deliberately different series
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n)],
            "symbol": ["A"] * n,
            "c_tr": tr,
            "c_split": split,
            "o": split,
            "h": [s + 1 for s in split],
            "low": [s - 1 for s in split],
            "v": [1000.0] * n,
        }
    )


def test_tr_feature_reads_the_tr_column_not_split() -> None:
    # ret_log declares close@tr; it must read c_tr and ignore c_split.
    df = _dual_basis_frame()
    out = df.select(sabia.returns.ret_log(period=1).expr(_SPLIT_SCHEMA)).to_series()
    expect_tr = df.select(log_return(pl.col("c_tr"), pl.col("c_tr").shift(1))).to_series()
    wrong_split = df.select(log_return(pl.col("c_split"), pl.col("c_split").shift(1))).to_series()
    assert out.to_list() == pytest.approx(expect_tr.to_list(), nan_ok=True)
    # Sanity: the two bases genuinely differ, so this is a real discrimination test.
    assert out.drop_nulls().to_list() != pytest.approx(wrong_split.drop_nulls().to_list())


def test_split_feature_reads_the_split_column_not_tr() -> None:
    # dollar_vol declares close@split * volume@split; it must read c_split, never c_tr.
    df = _dual_basis_frame()
    out = df.select(sabia.volume.dollar_vol().expr(_SPLIT_SCHEMA)).to_series()
    expect_split = df.select(pl.col("c_split") * pl.col("v")).to_series()
    wrong_tr = df.select(pl.col("c_tr") * pl.col("v")).to_series()
    assert out.to_list() == pytest.approx(expect_split.to_list())
    assert out.to_list() != pytest.approx(wrong_tr.to_list())


def test_unmapped_role_raises_precise_keyerror() -> None:
    # A schema missing close@tr must fail loudly when a tr feature resolves it.
    schema = BarSchema(roles={CLOSE_SPLIT: "c_split"})
    with pytest.raises(KeyError, match="close@tr"):
        sabia.returns.ret_log(period=1).expr(schema)


# --- universe enforcement (#6) -----------------------------------------------------------------


def test_compute_requires_universe_for_cross_sectional() -> None:
    panel = make_panel(300)
    with pytest.raises(ValueError, match="requires_universe"):
        sabia.compute(panel, sabia.cross_sectional.xs_rank_mom(), schema=SCHEMA)


def test_compute_rejects_universe_with_an_always_missing_symbol() -> None:
    panel = make_panel(300)  # symbols AAA, BBB, CCC
    with pytest.raises(SabiaValidationError, match="missing symbols"):
        sabia.compute(
            panel,
            sabia.cross_sectional.xs_rank_mom(),
            schema=SCHEMA,
            universe=["AAA", "BBB", "CCC", "ZZZ"],  # ZZZ is never present
        )


# --- cross-sectional output naming (#1) --------------------------------------------------------


def test_compute_names_xs_column_after_the_feature() -> None:
    panel = make_panel(300)
    df = sabia.compute(
        panel,
        sabia.cross_sectional.xs_rank_mom(),
        sabia.cross_sectional.rev_1m(),
        schema=SCHEMA,
        universe=["AAA", "BBB", "CCC"],
    )
    assert df.columns == ["xs_rank_mom_252_21", "rev_1m_21"]


# --- include_keys (identity columns aligned to feature rows) -----------------------------------


def test_compute_include_keys_prepends_aligned_identity() -> None:
    # Mixed TS + XS compute with include_keys must prepend symbol/timestamp aligned row-for-row with
    # the input -- both the TS select and the XS evaluate preserve input row order.
    panel = make_panel(300)
    df = sabia.compute(
        panel,
        sabia.momentum.rsi(period=14),
        sabia.cross_sectional.xs_rank_mom(),
        schema=SCHEMA,
        universe=["AAA", "BBB", "CCC"],
        include_keys=True,
    )
    assert df.columns == ["symbol", "timestamp", "rsi_14", "xs_rank_mom_252_21"]
    assert df.height == panel.height
    assert df.select("symbol", "timestamp").equals(panel.select("symbol", "timestamp"))
