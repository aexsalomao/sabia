# Calendar seam (FEATURES.md 2.4, 4.6, 7): a SessionCalendar resolves session-relative quantities
# (weekday, within-month position, annualization factor) so seasonality and `_ann` variants never
# hardcode a 252 in feature code. v1 ships a dependency-free `UtcCalendar` (calendar-day
# approximation, pure Polars `dt` expressions). Exchange calendars arrive later as a `quando`
# adapter implementing this same protocol -- with zero changes to feature code.

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

# The UTC default's annualization factor. The one labeled place a "252" lives; feature code reads it
# via `bars_per_year()`, never inline (FEATURES.md 4.6).
_DEFAULT_BARS_PER_YEAR = 252.0


@runtime_checkable
class SessionCalendar(Protocol):
    """Session-relative expression fragments + the annualization factor (FEATURES.md 4.6).

    All methods return per-row Polars expressions (group-agnostic); the consuming feature applies
    any `.over(...)` grouping. This keeps the calendar a pure expression provider and the causal
    grouping a feature concern.
    """

    code: str

    def bars_per_year(self) -> float: ...

    def session_weekday(self, ts: pl.Expr) -> pl.Expr:
        """Session weekday, Monday=0 .. Sunday=6 (FEATURES.md 12 `season_dow`)."""
        ...

    def month_key(self, ts: pl.Expr) -> pl.Expr:
        """A within-frame integer key identifying the (year, month) a session falls in."""
        ...

    def day_of_month(self, ts: pl.Expr) -> pl.Expr: ...

    def days_in_month(self, ts: pl.Expr) -> pl.Expr: ...


class UtcCalendar:
    """Dependency-free calendar: calendar-day approximation over UTC timestamps.

    Good enough for v1 seasonality and annualization; an exchange-accurate `SessionCalendar`
    (sessions after half-days/holidays) is a future `quando` adapter implementing the same protocol.
    """

    code = "UTC"

    def bars_per_year(self) -> float:
        return _DEFAULT_BARS_PER_YEAR

    def session_weekday(self, ts: pl.Expr) -> pl.Expr:
        # Polars `dt.weekday()` is ISO (Monday=1); shift to Monday=0 per the spec.
        return ts.dt.weekday() - 1

    def month_key(self, ts: pl.Expr) -> pl.Expr:
        return ts.dt.year() * 12 + ts.dt.month()

    def day_of_month(self, ts: pl.Expr) -> pl.Expr:
        return ts.dt.day()

    def days_in_month(self, ts: pl.Expr) -> pl.Expr:
        return ts.dt.month_end().dt.day()


def get_calendar(code: str) -> SessionCalendar:
    """Resolve a calendar code to a ``SessionCalendar``. Only ``"UTC"`` ships in v1."""
    if code == UtcCalendar.code:
        return UtcCalendar()
    raise KeyError(f"no calendar {code!r} in v1; install the quando adapter for exchange calendars")


__all__ = ["SessionCalendar", "UtcCalendar", "get_calendar"]
