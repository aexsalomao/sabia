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
from sabia.naming import naming
from sabia.schema import BarSchema
from sabia.typing import (
    ASK_RAW,
    BID_RAW,
    BID_SIZE_RAW,
    CLOSE_SPLIT,
    CLOSE_TR,
    HIGH_SPLIT,
    LOW_SPLIT,
    OPEN_SPLIT,
    SIGNED_VOLUME_RAW,
    VOLUME_SPLIT,
    Adjustment,
    DepthRole,
    FlowField,
    FlowRole,
    QuoteField,
    QuoteRole,
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


# --- intraday microstructure roles (FEATURES.md 13) --------------------------------------------


@pytest.mark.parametrize(
    ("role", "rendered"),
    [
        (BID_RAW, "bid@raw"),
        (ASK_RAW, "ask@raw"),
        (BID_SIZE_RAW, "bid_size@raw"),
        (SIGNED_VOLUME_RAW, "signed_volume@raw"),
        (DepthRole(QuoteField.BID, 1, Adjustment.RAW), "bid_l1@raw"),
        (DepthRole(QuoteField.ASK_SIZE, 0, Adjustment.RAW), "ask_size_l0@raw"),
    ],
    ids=["bid", "ask", "bid_size", "signed_vol", "depth_bid_l1", "depth_ask_size_l0"],
)
def test_micro_role_renders_unique_identity(role: object, rendered: str) -> None:
    # Each role's __str__ is its fingerprint identity (spec.feature_fingerprint folds str(role)).
    assert str(role) == rendered


def test_quote_role_is_price_classifies_price_vs_size() -> None:
    assert QuoteRole(QuoteField.BID, Adjustment.RAW).is_price
    assert QuoteRole(QuoteField.MID, Adjustment.RAW).is_price
    assert not QuoteRole(QuoteField.BID_SIZE, Adjustment.RAW).is_price


def test_micro_roles_are_hashable_and_distinct_in_a_set() -> None:
    # Frozen + hashable so they sit in frozenset[InputRole] and fold order-independently.
    roles = {BID_RAW, ASK_RAW, BID_RAW, SIGNED_VOLUME_RAW}
    assert len(roles) == 3
    assert BID_RAW != ASK_RAW
    assert FlowRole(FlowField.SIGNED_VOLUME, Adjustment.RAW) == SIGNED_VOLUME_RAW


def test_depth_role_rejects_mid_side_and_negative_level() -> None:
    with pytest.raises(ValueError, match="side"):
        DepthRole(QuoteField.MID, 0, Adjustment.RAW)
    with pytest.raises(ValueError, match="level"):
        DepthRole(QuoteField.BID, -1, Adjustment.RAW)


def test_naming_emits_adjustment_token_for_micro_roles() -> None:
    # Rule A: a quote role whose adjustment deviates from the default emits the token.
    split_bid = QuoteRole(QuoteField.BID, Adjustment.SPLIT)
    assert (
        naming("qspread", 20, role=split_bid, default_adjustment=Adjustment.RAW)
        == "qspread_split_20"
    )
    assert naming("qspread", 20, role=BID_RAW, default_adjustment=Adjustment.RAW) == "qspread_20"


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
