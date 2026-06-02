"""Tests for the input contract (sabia.validate)."""

from datetime import UTC, datetime

import polars as pl
import pytest

from sabia.schema import BarSchema
from sabia.typing import (
    CLOSE_SPLIT,
    CLOSE_TR,
    HIGH_SPLIT,
    LOW_SPLIT,
    OPEN_SPLIT,
    VOLUME_SPLIT,
    VWAP_SPLIT,
)
from sabia.validate import SabiaValidationError, validate

# A schema over the single-series OHLCV columns. VWAP_SPLIT maps to an absent column on purpose, to
# exercise the "role maps to an absent column" path.
OHLCV_SCHEMA = BarSchema(
    roles={
        OPEN_SPLIT: "open",
        HIGH_SPLIT: "high",
        LOW_SPLIT: "low",
        CLOSE_SPLIT: "close",
        CLOSE_TR: "close",
        VOLUME_SPLIT: "volume",
        VWAP_SPLIT: "vwap",
    }
)

# A minimal close-only schema for the panel fixtures (no OHLC ordering to resolve).
CLOSE_SCHEMA = BarSchema(roles={CLOSE_TR: "close"})


def _ts(*hours: int) -> list[datetime]:
    return [datetime(2024, 1, 1, h, tzinfo=UTC) for h in hours]


@pytest.fixture
def single_series() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": _ts(0, 1, 2, 3),
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.2, 2.2, 3.2, 4.2],
            "volume": [100, 200, 300, 400],
        }
    )


@pytest.fixture
def panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": _ts(0, 1, 0, 1),
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "close": [1.0, 1.1, 2.0, 2.1],
        }
    )


def test_valid_single_series_passes(single_series: pl.DataFrame) -> None:
    validate(single_series, schema=OHLCV_SCHEMA, required_roles=[CLOSE_TR, VOLUME_SPLIT])


def test_valid_panel_passes(panel: pl.DataFrame) -> None:
    validate(panel, schema=CLOSE_SCHEMA, required_roles=[CLOSE_TR], complete_panel=True)


def test_lazyframe_is_accepted(single_series: pl.DataFrame) -> None:
    validate(single_series.lazy(), schema=OHLCV_SCHEMA, required_roles=[CLOSE_TR])


def test_missing_timestamp_rejected(single_series: pl.DataFrame) -> None:
    with pytest.raises(SabiaValidationError, match="timestamp"):
        validate(single_series.drop("timestamp"), schema=OHLCV_SCHEMA)


def test_naive_timestamp_rejected(single_series: pl.DataFrame) -> None:
    naive = single_series.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    with pytest.raises(SabiaValidationError, match="tz-aware UTC"):
        validate(naive, schema=OHLCV_SCHEMA)


def test_non_utc_timestamp_rejected(single_series: pl.DataFrame) -> None:
    eastern = single_series.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York")
    )
    with pytest.raises(SabiaValidationError, match="UTC"):
        validate(eastern, schema=OHLCV_SCHEMA)


def test_unsorted_timestamps_rejected(single_series: pl.DataFrame) -> None:
    shuffled = single_series.reverse()
    with pytest.raises(SabiaValidationError, match="strictly increasing"):
        validate(shuffled, schema=OHLCV_SCHEMA)


def test_duplicate_timestamps_rejected(single_series: pl.DataFrame) -> None:
    dupe = pl.concat([single_series.head(1), single_series])
    with pytest.raises(SabiaValidationError, match="strictly increasing"):
        validate(dupe.sort("timestamp"), schema=OHLCV_SCHEMA)


def test_missing_required_column_rejected(single_series: pl.DataFrame) -> None:
    # VWAP_SPLIT resolves to the absent "vwap" column.
    with pytest.raises(SabiaValidationError, match="absent column"):
        validate(single_series, schema=OHLCV_SCHEMA, required_roles=[VWAP_SPLIT])


def test_wrong_price_dtype_rejected(single_series: pl.DataFrame) -> None:
    int_close = single_series.with_columns(pl.col("close").cast(pl.Int64))
    with pytest.raises(SabiaValidationError, match="float"):
        validate(int_close, schema=OHLCV_SCHEMA, required_roles=[CLOSE_TR])


def test_panel_unsorted_within_symbol_rejected(panel: pl.DataFrame) -> None:
    bad = panel.sort("symbol", "timestamp", descending=[False, True])
    with pytest.raises(SabiaValidationError, match="within each symbol"):
        validate(bad, schema=CLOSE_SCHEMA)


def test_incomplete_cross_section_rejected(panel: pl.DataFrame) -> None:
    # Drop BBB's first timestamp so t=0 has only one symbol.
    incomplete = panel.filter(~((pl.col("symbol") == "BBB") & (pl.col("timestamp") == _ts(0)[0])))
    with pytest.raises(SabiaValidationError, match="missing symbols"):
        validate(incomplete, schema=CLOSE_SCHEMA, complete_panel=True)


def test_cross_sectional_on_non_panel_rejected(single_series: pl.DataFrame) -> None:
    with pytest.raises(SabiaValidationError, match="requires a 'symbol'"):
        validate(single_series, schema=OHLCV_SCHEMA, complete_panel=True)


# --- as-of membership (FEATURES.md 2.5) ---------------------------------------------------------


@pytest.fixture
def ipo_panel() -> pl.DataFrame:
    # AAA trades at both t=0 and t=1; BBB lists only at t=1. A complete cross-section under a static
    # universe would (wrongly) flag t=0 as missing BBB -- as-of membership knows BBB isn't a member.
    return pl.DataFrame(
        {
            "timestamp": _ts(0, 1, 1),
            "symbol": ["AAA", "AAA", "BBB"],
            "close": [1.0, 1.1, 2.0],
        }
    )


def _membership(rows: list[tuple[str, int, int | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [s for s, _, _ in rows],
            "start": [_ts(start)[0] for _, start, _ in rows],
            "end": [_ts(end)[0] if end is not None else None for _, _, end in rows],
        },
        schema={
            "symbol": pl.String,
            "start": pl.Datetime("us", "UTC"),
            "end": pl.Datetime("us", "UTC"),
        },
    )


def test_membership_allows_ipo_midwindow(ipo_panel: pl.DataFrame) -> None:
    # BBB is a member only from t=1 onward, so its absence at t=0 is not a completeness violation.
    membership = _membership([("AAA", 0, None), ("BBB", 1, None)])
    validate(ipo_panel, schema=CLOSE_SCHEMA, complete_panel=True, membership=membership)


def test_membership_flags_genuinely_missing_symbol(ipo_panel: pl.DataFrame) -> None:
    # BBB is declared a member from t=0, but the panel has no BBB row at t=0 -> a real gap.
    membership = _membership([("AAA", 0, None), ("BBB", 0, None)])
    with pytest.raises(SabiaValidationError, match="missing symbols"):
        validate(ipo_panel, schema=CLOSE_SCHEMA, complete_panel=True, membership=membership)


@pytest.fixture
def delisting_panel() -> pl.DataFrame:
    # AAA trades the whole window (t=0,1,2); DEL trades t=0,1 then is ABSENT at t=2.
    return pl.DataFrame(
        {
            "timestamp": _ts(0, 1, 2, 0, 1),
            "symbol": ["AAA", "AAA", "AAA", "DEL", "DEL"],
            "close": [1.0, 1.1, 1.2, 2.0, 2.1],
        }
    )


def test_membership_end_is_exclusive_at_the_boundary(delisting_panel: pl.DataFrame) -> None:
    # DEL's window ends at t=2 (exclusive [0,2)): it is NOT a member at t=2, so its absence there is
    # not a gap. This passes under exclusive end but would FAIL under inclusive (t<=end) semantics.
    membership = _membership([("AAA", 0, None), ("DEL", 0, 2)])
    validate(delisting_panel, schema=CLOSE_SCHEMA, complete_panel=True, membership=membership)


def test_membership_flags_member_absent_at_boundary(delisting_panel: pl.DataFrame) -> None:
    # DEL is still a member at t=2 (window [0,3)) but absent there -> a genuine completeness gap.
    membership = _membership([("AAA", 0, None), ("DEL", 0, 3)])
    with pytest.raises(SabiaValidationError, match="missing symbols"):
        validate(delisting_panel, schema=CLOSE_SCHEMA, complete_panel=True, membership=membership)


def test_membership_column_names_collide_safely_with_timestamp_col() -> None:
    # A panel whose timestamp_col is literally "start" must not silently break the membership join.
    schema = BarSchema(roles={CLOSE_TR: "close"}, timestamp_col="start")
    panel = pl.DataFrame(
        {"start": _ts(0, 1, 1), "symbol": ["AAA", "AAA", "BBB"], "close": [1.0, 1.1, 2.0]}
    )
    membership = _membership([("AAA", 0, None), ("BBB", 1, None)])
    validate(
        panel, schema=schema, complete_panel=True, membership=membership
    )  # no collision, no raise


def test_malformed_membership_frame_rejected(ipo_panel: pl.DataFrame) -> None:
    bad = pl.DataFrame({"symbol": ["AAA"], "start": _ts(0)})  # missing 'end'
    with pytest.raises(SabiaValidationError, match="missing column"):
        validate(ipo_panel, schema=CLOSE_SCHEMA, complete_panel=True, membership=bad)
