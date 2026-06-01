# The input contract (FEATURES.md 7.3): the ONLY fail-loud surface in sabia. validate() raises
# SabiaValidationError on a malformed frame; feature bodies trust valid input and never re-check.
# Operates on the lazy schema and small aggregates -- it never materializes the full frame.

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from sabia.spec import Column

# Acceptable dtypes per canonical column. Prices are floating point; volume may be int or float;
# symbols are string-like. Timestamps get the dedicated tz check below.
_FLOAT_DTYPES = (pl.Float32, pl.Float64)
_INT_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)
_STRING_DTYPES = (pl.String, pl.Categorical, pl.Enum)

_PRICE_COLUMNS = frozenset({Column.OPEN, Column.HIGH, Column.LOW, Column.CLOSE})
_ACCEPTED_DTYPES: dict[Column, tuple[type[pl.DataType], ...]] = {
    Column.OPEN: _FLOAT_DTYPES,
    Column.HIGH: _FLOAT_DTYPES,
    Column.LOW: _FLOAT_DTYPES,
    Column.CLOSE: _FLOAT_DTYPES,
    Column.VOLUME: _FLOAT_DTYPES + _INT_DTYPES,
    Column.SYMBOL: _STRING_DTYPES,
}

_UTC = "UTC"


class SabiaValidationError(ValueError):
    """Raised when an input frame violates the contract every feature assumes (FEATURES.md 7.3)."""


def validate(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    required: Iterable[Column] = (),
    cross_sectional: bool = False,
) -> None:
    """Check the input contract; raise ``SabiaValidationError`` on the first violation.

    Args:
        frame: OHLCV frame, eager or lazy. A ``symbol`` column makes it a panel; ordering and
            uniqueness are then enforced per symbol.
        required: Columns a downstream feature needs. Each must be present with an accepted dtype.
        cross_sectional: When ``True`` (and the frame is a panel), assert every timestamp carries
            the complete cross-section -- the same set of symbols at every ``t``.

    Note:
        A ``symbol`` column is optional here: a symbol-less frame is a valid single series.
        Time-series features group their trailing windows per symbol, so they expect that column
        by default; to run them on a bare single series, pass ``symbol=None`` to the feature.

        "Bars closed" (no in-progress bar) is part of the contract but is not structurally
        checkable without a clock, which purity forbids (FEATURES.md 2.3). It is the engine's
        responsibility to pass only closed bars; sabia declares the precondition and trusts it.
    """
    lf = frame.lazy()
    schema = lf.collect_schema()
    is_panel = Column.SYMBOL in schema.names()

    _check_timestamp(schema)
    _check_required_columns(schema, required)
    if is_panel:
        _check_symbol_dtype(schema)
    _check_ordering_and_uniqueness(lf, is_panel)
    if cross_sectional:
        _require_panel(is_panel)
        _check_complete_cross_section(lf)


def _check_timestamp(schema: pl.Schema) -> None:
    if Column.TIMESTAMP not in schema.names():
        raise SabiaValidationError(f"missing required column '{Column.TIMESTAMP}'")
    dtype = schema[Column.TIMESTAMP]
    if not isinstance(dtype, pl.Datetime):
        raise SabiaValidationError(f"'{Column.TIMESTAMP}' must be Datetime, got {dtype}")
    if dtype.time_zone != _UTC:
        raise SabiaValidationError(
            f"'{Column.TIMESTAMP}' must be tz-aware UTC, got time_zone={dtype.time_zone!r}"
        )


def _check_required_columns(schema: pl.Schema, required: Iterable[Column]) -> None:
    names = schema.names()
    for column in required:
        if column not in names:
            raise SabiaValidationError(f"missing required column '{column}'")
        accepted = _ACCEPTED_DTYPES.get(column)
        if accepted is not None and schema[column] not in accepted:
            kind = "float" if column in _PRICE_COLUMNS else "expected"
            raise SabiaValidationError(
                f"column '{column}' has dtype {schema[column]}, {kind} one of {accepted}"
            )


def _check_symbol_dtype(schema: pl.Schema) -> None:
    if schema[Column.SYMBOL] not in _STRING_DTYPES:
        raise SabiaValidationError(
            f"'{Column.SYMBOL}' must be string-like, got {schema[Column.SYMBOL]}"
        )


def _check_ordering_and_uniqueness(lf: pl.LazyFrame, is_panel: bool) -> None:
    # Strictly increasing timestamps == sorted AND unique in one check. On a panel the check is
    # per symbol (timestamps repeat across symbols, so they are unique only within a symbol).
    epoch = pl.col(Column.TIMESTAMP).dt.epoch(time_unit="ns")
    step = epoch.diff().over(Column.SYMBOL) if is_panel else epoch.diff()
    # diff()'s first element per group is null; a non-positive step is out-of-order or a duplicate.
    violation = lf.select((step <= 0).any().alias("bad")).collect().item()
    if violation:
        scope = "within each symbol" if is_panel else "globally"
        raise SabiaValidationError(
            f"timestamps must be strictly increasing {scope} (sorted and unique)"
        )


def _check_complete_cross_section(lf: pl.LazyFrame) -> None:
    counts = lf.group_by(Column.TIMESTAMP).agg(pl.col(Column.SYMBOL).n_unique().alias("k"))
    distinct = lf.select(pl.col(Column.SYMBOL).n_unique().alias("total")).collect().item()
    incomplete = counts.select((pl.col("k") != distinct).any().alias("bad")).collect().item()
    if incomplete:
        raise SabiaValidationError(
            "cross-sectional contract violated: some timestamps are missing symbols "
            f"(expected {distinct} symbols at every timestamp)"
        )


def _require_panel(is_panel: bool) -> None:
    if not is_panel:
        raise SabiaValidationError(
            f"cross_sectional=True requires a '{Column.SYMBOL}' column (a panel frame)"
        )


__all__ = ["SabiaValidationError", "validate"]
