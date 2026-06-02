# Parameter guards for feature factories: raise loudly at BIND time on out-of-domain params, so a
# bad period / window / lambda fails where the caller wrote it -- the same "validate at the
# boundary" discipline validate() applies to frames (FEATURES.md 8.3), here for the factory call.
# Pure: the only side effect is a ValueError. ``bool`` is rejected for int params (it is an int
# subclass, and a True/False window is always a mistake). Guards run before naming(), so they never
# affect a valid feature's fingerprint.

from __future__ import annotations

import math


def positive_int(name: str, value: int) -> None:
    """Require ``value`` to be an int strictly greater than zero."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def non_negative_int(name: str, value: int) -> None:
    """Require ``value`` to be an int >= 0."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")


def int_at_least(name: str, value: int, minimum: int) -> None:
    """Require ``value`` to be an int >= ``minimum`` (e.g. an EWM span needs >= 2 for alpha < 1)."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")


def positive(name: str, value: float) -> None:
    """Require ``value`` to be a finite, strictly positive real (a band width / threshold)."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def in_open_interval(name: str, value: float, lower: float, upper: float) -> None:
    """Require a finite ``lower < value < upper`` (e.g. a RiskMetrics lambda in (0, 1))."""
    if not math.isfinite(value) or not lower < value < upper:
        raise ValueError(f"{name} must be in ({lower}, {upper}), got {value!r}")


def less_than(low_name: str, low: int, high_name: str, high: int) -> None:
    """Require ``low < high`` for a pair of related params (e.g. ``skip < formation``)."""
    if not low < high:
        raise ValueError(f"{low_name} ({low}) must be less than {high_name} ({high})")


__all__ = [
    "in_open_interval",
    "int_at_least",
    "less_than",
    "non_negative_int",
    "positive",
    "positive_int",
]
