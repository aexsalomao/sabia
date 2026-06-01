"""Tests for the input contract (sabia.validate)."""

from datetime import UTC, datetime

import polars as pl
import pytest

from sabia.spec import Column
from sabia.validate import SabiaValidationError, validate


def _ts(*hours: int) -> list[datetime]:
    return [datetime(2024, 1, 1, h, tzinfo=UTC) for h in hours]


@pytest.fixture
def single_series() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.TIMESTAMP: _ts(0, 1, 2, 3),
            Column.OPEN: [1.0, 2.0, 3.0, 4.0],
            Column.HIGH: [1.5, 2.5, 3.5, 4.5],
            Column.LOW: [0.5, 1.5, 2.5, 3.5],
            Column.CLOSE: [1.2, 2.2, 3.2, 4.2],
            Column.VOLUME: [100, 200, 300, 400],
        }
    )


@pytest.fixture
def panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            Column.TIMESTAMP: _ts(0, 1, 0, 1),
            Column.SYMBOL: ["AAA", "AAA", "BBB", "BBB"],
            Column.CLOSE: [1.0, 1.1, 2.0, 2.1],
        }
    )


def test_valid_single_series_passes(single_series: pl.DataFrame) -> None:
    validate(single_series, required=[Column.CLOSE, Column.VOLUME])


def test_valid_panel_passes(panel: pl.DataFrame) -> None:
    validate(panel, required=[Column.CLOSE], cross_sectional=True)


def test_lazyframe_is_accepted(single_series: pl.DataFrame) -> None:
    validate(single_series.lazy(), required=[Column.CLOSE])


def test_missing_timestamp_rejected(single_series: pl.DataFrame) -> None:
    with pytest.raises(SabiaValidationError, match="timestamp"):
        validate(single_series.drop(Column.TIMESTAMP))


def test_naive_timestamp_rejected(single_series: pl.DataFrame) -> None:
    naive = single_series.with_columns(pl.col(Column.TIMESTAMP).dt.replace_time_zone(None))
    with pytest.raises(SabiaValidationError, match="tz-aware UTC"):
        validate(naive)


def test_non_utc_timestamp_rejected(single_series: pl.DataFrame) -> None:
    eastern = single_series.with_columns(
        pl.col(Column.TIMESTAMP).dt.convert_time_zone("America/New_York")
    )
    with pytest.raises(SabiaValidationError, match="UTC"):
        validate(eastern)


def test_unsorted_timestamps_rejected(single_series: pl.DataFrame) -> None:
    shuffled = single_series.reverse()
    with pytest.raises(SabiaValidationError, match="strictly increasing"):
        validate(shuffled)


def test_duplicate_timestamps_rejected(single_series: pl.DataFrame) -> None:
    dupe = pl.concat([single_series.head(1), single_series])
    with pytest.raises(SabiaValidationError, match="strictly increasing"):
        validate(dupe.sort(Column.TIMESTAMP))


def test_missing_required_column_rejected(single_series: pl.DataFrame) -> None:
    with pytest.raises(SabiaValidationError, match="symbol"):
        validate(single_series, required=[Column.SYMBOL])


def test_wrong_price_dtype_rejected(single_series: pl.DataFrame) -> None:
    int_close = single_series.with_columns(pl.col(Column.CLOSE).cast(pl.Int64))
    with pytest.raises(SabiaValidationError, match="float"):
        validate(int_close, required=[Column.CLOSE])


def test_panel_unsorted_within_symbol_rejected(panel: pl.DataFrame) -> None:
    bad = panel.sort(Column.SYMBOL, Column.TIMESTAMP, descending=[False, True])
    with pytest.raises(SabiaValidationError, match="within each symbol"):
        validate(bad)


def test_incomplete_cross_section_rejected(panel: pl.DataFrame) -> None:
    # Drop BBB's first timestamp so t=0 has only one symbol.
    incomplete = panel.filter(
        ~((pl.col(Column.SYMBOL) == "BBB") & (pl.col(Column.TIMESTAMP) == _ts(0)[0]))
    )
    with pytest.raises(SabiaValidationError, match="missing symbols"):
        validate(incomplete, cross_sectional=True)


def test_cross_sectional_on_non_panel_rejected(single_series: pl.DataFrame) -> None:
    with pytest.raises(SabiaValidationError, match="requires a 'symbol'"):
        validate(single_series, cross_sectional=True)
