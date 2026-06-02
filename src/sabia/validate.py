# The input contract (FEATURES.md 8.3): the fail-loud surface. validate() raises
# SabiaValidationError on a malformed frame; feature bodies trust valid input and never re-check.
# Operates on the lazy schema and small aggregates -- it never materializes the full frame.
# ValidationMode tunes strictness: STRICT raises on everything; RESEARCH warns on
# completeness/finalization only but still raises on schema/dtype/role/order; OFF skips. No logging
# in core -- warnings are returned.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from sabia.spec import BoundFeature

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
    membership: pl.DataFrame | pl.LazyFrame | None = None,
    mode: ValidationMode = ValidationMode.STRICT,
) -> list[str]:
    """Check the input contract; return non-fatal warnings, raise on hard violations.

    Args:
        frame: OHLCV frame, eager or lazy. ``schema.symbol_col`` makes it a panel; ordering and
            uniqueness are then enforced per symbol.
        schema: maps roles to physical columns; ``required_roles`` are resolved against it.
        required_roles: roles a downstream feature needs; each must resolve to a present, typed col.
        complete_panel: assert every timestamp carries the full cross-section (FEATURES.md 2.5).
        universe: the declared static universe -- the fixed set of symbols expected at *every*
            timestamp. sabia asserts completeness against it; it never infers membership.
        membership: as-of universe membership as a frame of ``(symbol, start, end)`` rows (UTC
            datetimes; ``end`` null = still a member). When given, the expected cross-section at
            each timestamp ``t`` is ``{symbol : start <= t < end}`` -- the point-in-time model for
            panels with IPOs / delistings (FEATURES.md 2.5). Takes precedence over ``universe``.
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
            _incomplete_cross_section(lf, schema, is_panel, universe, membership),
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
    lf: pl.LazyFrame,
    schema: BarSchema,
    is_panel: bool,
    universe: Sequence[str] | None,
    membership: pl.DataFrame | pl.LazyFrame | None,
) -> bool:
    if not is_panel:
        raise SabiaValidationError(
            f"complete_panel=True requires a {schema.symbol_col!r} column (a panel frame)"
        )
    if membership is not None:
        return _incomplete_vs_membership(lf, schema, membership)
    # Static universe (or, absent one, the present symbols): every timestamp carries the full set.
    expected = (
        len(set(universe))
        if universe is not None
        else lf.select(pl.col(schema.symbol_col).n_unique()).collect().item()
    )
    counts = lf.group_by(schema.timestamp_col).agg(pl.col(schema.symbol_col).n_unique().alias("k"))
    return bool(counts.select((pl.col("k") != expected).any().alias("bad")).collect().item())


# Membership frame contract (FEATURES.md 2.5): the as-of universe is three columns.
_MEMBERSHIP_SYMBOL = "symbol"
_MEMBERSHIP_START = "start"
_MEMBERSHIP_END = "end"
_MEMBERSHIP_COLS = (_MEMBERSHIP_SYMBOL, _MEMBERSHIP_START, _MEMBERSHIP_END)
# Internal projection names for the membership join. Dunder-prefixed so the cross-join can never
# collide with a caller's timestamp_col / symbol_col even if it happens to be named "start"/"end".
_MBR_SYMBOL = "__sabia_mbr_symbol"
_MBR_START = "__sabia_mbr_start"
_MBR_END = "__sabia_mbr_end"


def _incomplete_vs_membership(
    lf: pl.LazyFrame, schema: BarSchema, membership: pl.DataFrame | pl.LazyFrame
) -> bool:
    # Point-in-time completeness: at each timestamp t the expected cross-section is the set of
    # symbols whose membership interval [start, end) covers t. A symbol that the membership says
    # should trade at t but is absent from the panel is the violation we flag (extra symbols outside
    # their window are a separate data-quality concern, not a completeness gap). Compared on epoch
    # nanoseconds so the check is robust to the frame's Datetime time-unit.
    ml = _validated_membership(membership)
    ts_col, sym_col = schema.timestamp_col, schema.symbol_col
    t_epoch = pl.col(ts_col).dt.epoch(time_unit="ns")
    timestamps = lf.select(ts_col).unique()
    expected = (
        timestamps.join(ml, how="cross")
        .filter(
            (t_epoch >= pl.col(_MBR_START))
            & (pl.col(_MBR_END).is_null() | (t_epoch < pl.col(_MBR_END)))
        )
        .select(ts_col, pl.col(_MBR_SYMBOL).alias(sym_col))
    )
    present = lf.select(ts_col, sym_col).unique()
    missing = expected.join(present, on=[ts_col, sym_col], how="anti")
    return bool(missing.select(pl.len()).collect().item() > 0)


def _validated_membership(membership: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    # Validate the membership frame's shape, then project it to collision-proof internal names
    # (symbol, start_epoch, end_epoch) so the downstream cross-join is independent of the panel's
    # configured timestamp_col / symbol_col.
    ml = membership.lazy()
    sch = ml.collect_schema()
    missing_cols = [c for c in _MEMBERSHIP_COLS if c not in sch.names()]
    if missing_cols:
        raise SabiaValidationError(
            f"membership frame is missing column(s) {missing_cols}; "
            f"it must carry {list(_MEMBERSHIP_COLS)}"
        )
    if sch[_MEMBERSHIP_SYMBOL] not in _STRING_DTYPES:
        raise SabiaValidationError(
            f"membership {_MEMBERSHIP_SYMBOL!r} must be string-like, got {sch[_MEMBERSHIP_SYMBOL]}"
        )
    for col in (_MEMBERSHIP_START, _MEMBERSHIP_END):
        dtype = sch[col]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != _UTC:
            raise SabiaValidationError(
                f"membership {col!r} must be tz-aware UTC Datetime, got {dtype}"
            )
    return ml.select(
        pl.col(_MEMBERSHIP_SYMBOL).alias(_MBR_SYMBOL),
        pl.col(_MEMBERSHIP_START).dt.epoch(time_unit="ns").alias(_MBR_START),
        pl.col(_MEMBERSHIP_END).dt.epoch(time_unit="ns").alias(_MBR_END),
    )


# --- non-raising audit (the report counterpart of validate) ------------------------------------


@dataclass(frozen=True, slots=True)
class FrameAudit:
    """A non-fatal health report for a frame (the audit counterpart of ``validate``).

    Where ``validate`` raises on the first hard violation, ``audit_frame`` runs the same lazy
    aggregates and *reports* them, so a caller can inspect a frame before an expensive feature job.
    """

    rows: int
    symbols: int | None  # None when the frame is not a panel
    start: datetime | None
    end: datetime | None
    timestamp_utc_ok: bool
    missing_roles: tuple[str, ...]  # required roles that do not resolve to a present column
    ohlc_violations: int  # rows breaking low <= min(o,c) <= max(o,c) <= high, summed over bases
    duplicate_keys: int  # rows beyond the first per (symbol, timestamp) / per timestamp
    non_final_bars: int  # bars with closed_col == False (0 when there is no closed_col)
    completeness: (
        float | None
    )  # fraction of timestamps carrying the full cross-section (panel only)


def audit_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    schema: BarSchema,
    features: Iterable[BoundFeature] = (),
) -> FrameAudit:
    """Inspect a frame and return a ``FrameAudit`` -- counts and ranges, never a raise.

    Aligned with the library philosophy: it prevents bad feature computation by surfacing problems
    (missing roles, OHLC breaks, duplicate keys, incomplete cross-sections) up front. Pure lazy
    aggregates -- it never materializes the full frame. ``features`` declares which roles to check.
    """
    lf = frame.lazy()
    sch = lf.collect_schema()
    names = sch.names()
    ts, sym = schema.timestamp_col, schema.symbol_col
    is_panel = sym in names

    ts_dtype = sch[ts] if ts in names else None
    ts_ok = isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone == _UTC
    rows = int(lf.select(pl.len()).collect().item())
    if ts_ok:
        start, end = (
            lf.select(pl.col(ts).min().alias("start"), pl.col(ts).max().alias("end"))
            .collect()
            .row(0)
        )
    else:
        start, end = None, None
    symbols = int(lf.select(pl.col(sym).n_unique()).collect().item()) if is_panel else None

    roles: frozenset[InputRole] = frozenset().union(
        *(f.spec.input_roles for f in features), frozenset()
    )
    missing = tuple(
        sorted(str(r) for r in roles if not schema.has(r) or schema.column(r) not in names)
    )

    key = [sym, ts] if is_panel else [ts]
    distinct_keys = int(lf.select(pl.struct(key).n_unique()).collect().item()) if ts in names else 0
    duplicate_keys = rows - distinct_keys if ts in names else 0

    return FrameAudit(
        rows=rows,
        symbols=symbols,
        start=start,
        end=end,
        timestamp_utc_ok=ts_ok,
        missing_roles=missing,
        ohlc_violations=_count_ohlc_violations(lf, schema, names),
        duplicate_keys=duplicate_keys,
        non_final_bars=_count_non_final(lf, schema, names),
        completeness=_completeness(lf, schema) if is_panel and ts in names else None,
    )


def _count_ohlc_violations(lf: pl.LazyFrame, schema: BarSchema, names: list[str]) -> int:
    # audit_frame never raises, so skip a basis whose resolved columns are not all present (validate
    # is the surface that turns a missing OHLC column into a hard error).
    total = 0
    for adjustment in Adjustment:
        cols = _resolve_ohlc(schema, adjustment)
        if cols is None or any(name not in names for name in cols):
            continue
        o, h, low, c = (pl.col(name) for name in cols)
        bad = (low > pl.min_horizontal(o, c)) | (pl.max_horizontal(o, c) > h)
        total += int(lf.select(bad.sum().alias("n")).collect().item() or 0)
    return total


def _count_non_final(lf: pl.LazyFrame, schema: BarSchema, names: list[str]) -> int:
    col = schema.closed_col
    if col is None or col not in names:
        return 0
    return int(lf.select((~pl.col(col)).sum().alias("n")).collect().item() or 0)


def _completeness(lf: pl.LazyFrame, schema: BarSchema) -> float:
    expected = lf.select(pl.col(schema.symbol_col).n_unique()).collect().item()
    if not expected:
        return 1.0
    counts = lf.group_by(schema.timestamp_col).agg(pl.col(schema.symbol_col).n_unique().alias("k"))
    full, total = (
        counts.select((pl.col("k") == expected).sum().alias("full"), pl.len().alias("total"))
        .collect()
        .row(0)
    )
    return float(full) / float(total) if total else 1.0


__all__ = ["FrameAudit", "SabiaValidationError", "audit_frame", "validate"]
