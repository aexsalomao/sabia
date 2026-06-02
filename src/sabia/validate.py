# The input contract (FEATURES.md 8.3): the fail-loud surface. validate() raises
# SabiaValidationError on a malformed frame; feature bodies trust valid input and never re-check.
# Operates on the lazy schema and small aggregates -- it never materializes the full frame.
# ValidationMode tunes strictness: STRICT raises on everything; RESEARCH warns on
# completeness/finalization only but still raises on schema/dtype/role/order; OFF skips. No logging
# in core -- warnings are returned.

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from sabia.schema import BarSchema
from sabia.spec import ValidationMode
from sabia.typing import (
    Adjustment,
    FactorRole,
    InputRole,
    PriceField,
    PriceRole,
    VolumeRole,
)

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
_UTC = "UTC"


class SabiaValidationError(ValueError):
    """Raised when an input frame violates the contract every feature assumes (FEATURES.md 8.3)."""


def validate(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    schema: BarSchema,
    required_roles: Iterable[InputRole] = (),
    complete_panel: bool = False,
    universe: Sequence[str] | None = None,
    membership_asof: object = None,
    mode: ValidationMode = ValidationMode.STRICT,
) -> list[str]:
    """Check the input contract; return non-fatal warnings, raise on hard violations.

    Args:
        frame: OHLCV frame, eager or lazy. ``schema.symbol_col`` makes it a panel; ordering and
            uniqueness are then enforced per symbol.
        schema: maps roles to physical columns; ``required_roles`` are resolved against it.
        required_roles: roles a downstream feature needs; each must resolve to a present, typed col.
        complete_panel: assert every timestamp carries the full cross-section (FEATURES.md 2.5).
        universe / membership_asof: declared universe + as-of membership (FEATURES.md 2.5). sabia
            asserts completeness against the universe; it never infers membership.
        mode: STRICT raises on any violation; RESEARCH warns on completeness/finalization but still
            raises on schema/dtype/role/order; OFF skips all checks.

    Returns:
        A list of warning strings (RESEARCH mode). STRICT returns an empty list or raises. No
        logging happens in core -- the caller decides what to do with warnings.

    Note:
        "Bars closed" is checked via ``schema.closed_col`` / ``is_final`` -- never wall-clock time
        (purity forbids it, FEATURES.md 8.3). With no ``closed_col`` the frame is upstream-trusted.
    """
    if mode is ValidationMode.OFF:
        return []

    warnings: list[str] = []
    lf = frame.lazy()
    names = lf.collect_schema().names()
    is_panel = schema.symbol_col in names

    _check_timestamp(lf, schema)
    _check_required_roles(lf, schema, required_roles, names)
    if is_panel:
        _check_symbol_dtype(lf, schema)
    _check_ordering_and_uniqueness(lf, schema, is_panel)
    _check_ohlc_ordering(lf, schema)

    _soft_check(
        _finalization_violation(lf, schema, names),
        "some bars are not final (closed_col has False values)",
        mode,
        warnings,
    )
    if complete_panel:
        _soft_check(
            _incomplete_cross_section(lf, schema, is_panel, universe),
            "cross-sectional contract violated: some timestamps are missing symbols",
            mode,
            warnings,
        )
    return warnings


def _soft_check(violated: bool, message: str, mode: ValidationMode, warnings: list[str]) -> None:
    # Completeness/finalization: RESEARCH warns, STRICT raises (FEATURES.md 8.3).
    if not violated:
        return
    if mode is ValidationMode.RESEARCH:
        warnings.append(message)
    else:
        raise SabiaValidationError(message)


def _schema_dtype(lf: pl.LazyFrame) -> pl.Schema:
    return lf.collect_schema()


def _check_timestamp(lf: pl.LazyFrame, schema: BarSchema) -> None:
    sch = _schema_dtype(lf)
    col = schema.timestamp_col
    if col not in sch.names():
        raise SabiaValidationError(f"missing required column {col!r}")
    dtype = sch[col]
    if not isinstance(dtype, pl.Datetime):
        raise SabiaValidationError(f"{col!r} must be Datetime, got {dtype}")
    if dtype.time_zone != _UTC:
        raise SabiaValidationError(
            f"{col!r} must be tz-aware UTC, got time_zone={dtype.time_zone!r}"
        )


def _check_required_roles(
    lf: pl.LazyFrame, schema: BarSchema, required_roles: Iterable[InputRole], names: list[str]
) -> None:
    sch = _schema_dtype(lf)
    for role in required_roles:
        try:
            col = schema.column(role)
        except KeyError as exc:
            raise SabiaValidationError(str(exc)) from None
        if col not in names:
            raise SabiaValidationError(f"role {role} maps to absent column {col!r}")
        _check_role_dtype(role, sch[col], col)


def _check_role_dtype(role: InputRole, dtype: pl.DataType, col: str) -> None:
    if isinstance(role, PriceRole | FactorRole):
        accepted: tuple[type[pl.DataType], ...] = _FLOAT_DTYPES
        kind = "float"
    elif isinstance(role, VolumeRole):
        accepted = _FLOAT_DTYPES + _INT_DTYPES
        kind = "numeric"
    else:  # CalendarRole: a session label column, string-like
        accepted = _STRING_DTYPES
        kind = "string-like"
    if dtype not in accepted:
        raise SabiaValidationError(
            f"role {role} (column {col!r}) has dtype {dtype}, expected {kind} one of {accepted}"
        )


def _check_symbol_dtype(lf: pl.LazyFrame, schema: BarSchema) -> None:
    sch = _schema_dtype(lf)
    if sch[schema.symbol_col] not in _STRING_DTYPES:
        raise SabiaValidationError(
            f"{schema.symbol_col!r} must be string-like, got {sch[schema.symbol_col]}"
        )


def _check_ordering_and_uniqueness(lf: pl.LazyFrame, schema: BarSchema, is_panel: bool) -> None:
    # Strictly increasing timestamps == sorted AND unique in one check. On a panel the check is per
    # symbol (timestamps repeat across symbols, so they are unique only within a symbol).
    epoch = pl.col(schema.timestamp_col).dt.epoch(time_unit="ns")
    step = epoch.diff().over(schema.symbol_col) if is_panel else epoch.diff()
    violation = lf.select((step <= 0).any().alias("bad")).collect().item()
    if violation:
        scope = "within each symbol" if is_panel else "globally"
        raise SabiaValidationError(
            f"timestamps must be strictly increasing {scope} (sorted and unique)"
        )


def _check_ohlc_ordering(lf: pl.LazyFrame, schema: BarSchema) -> None:
    # For every adjustment basis with a coherent OHLC set, assert low <= min(o,c) and max(o,c) <=
    # high after role resolution (FEATURES.md 2.2). Skip a basis that doesn't resolve all four.
    for adjustment in Adjustment:
        cols = _resolve_ohlc(schema, adjustment)
        if cols is None:
            continue
        o, h, low, c = (pl.col(name) for name in cols)
        bad = (
            lf.select(
                ((low > pl.min_horizontal(o, c)) | (pl.max_horizontal(o, c) > h)).any().alias("bad")
            )
            .collect()
            .item()
        )
        if bad:
            raise SabiaValidationError(
                f"OHLC ordering violated on @{adjustment.value}: "
                "require low <= min(open, close) <= max(open, close) <= high"
            )


def _resolve_ohlc(schema: BarSchema, adjustment: Adjustment) -> tuple[str, str, str, str] | None:
    roles = (
        PriceRole(PriceField.OPEN, adjustment),
        PriceRole(PriceField.HIGH, adjustment),
        PriceRole(PriceField.LOW, adjustment),
        PriceRole(PriceField.CLOSE, adjustment),
    )
    if not all(schema.has(role) for role in roles):
        return None
    o, h, low, c = (schema.column(role) for role in roles)
    return o, h, low, c


def _finalization_violation(lf: pl.LazyFrame, schema: BarSchema, names: list[str]) -> bool:
    col = schema.closed_col
    if col is None or col not in names:
        return False
    return bool(lf.select((~pl.col(col)).any().alias("bad")).collect().item())


def _incomplete_cross_section(
    lf: pl.LazyFrame, schema: BarSchema, is_panel: bool, universe: Sequence[str] | None
) -> bool:
    if not is_panel:
        raise SabiaValidationError(
            f"complete_panel=True requires a {schema.symbol_col!r} column (a panel frame)"
        )
    expected = (
        len(set(universe))
        if universe is not None
        else lf.select(pl.col(schema.symbol_col).n_unique()).collect().item()
    )
    counts = lf.group_by(schema.timestamp_col).agg(pl.col(schema.symbol_col).n_unique().alias("k"))
    return bool(counts.select((pl.col("k") != expected).any().alias("bad")).collect().item())


__all__ = ["SabiaValidationError", "validate"]
