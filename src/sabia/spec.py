# The feature contract: FeatureSpec metadata, the structural/horizon enums, the canonical
# column names, the shared numeric constants, and the fingerprint helper. Everything else in
# sabia references the types defined here. See FEATURES.md sections 3 and 5.

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import textwrap
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum
from math import ceil, log

import polars as pl

# --- canonical columns -------------------------------------------------------------------------

# StrEnum so members are usable directly as column names: pl.col(Column.CLOSE) just works, and
# functions can default to them -- e.g. ``def rsi(close: str = Column.CLOSE)``.


class Column(StrEnum):
    """Canonical OHLCV column names. The input contract (validate.py) expects these."""

    TIMESTAMP = "timestamp"
    SYMBOL = "symbol"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"


# --- structural / horizon axes -----------------------------------------------------------------


class Family(Enum):
    """Structural axis -- one module per family.

    ``MICROSTRUCTURE`` is defined here but its module ships in a later minor version (v1 is
    bars-only); the tier machinery is in place so it slots in without touching other families.
    """

    RETURNS = "returns"
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    DISTRIBUTION = "distribution"
    MEAN_REVERSION = "mean_reversion"
    SEASONALITY = "seasonality"
    CROSS_SECTIONAL = "cross_sectional"
    MICROSTRUCTURE = "microstructure"


class Horizon(Enum):
    """Horizon bands. A feature's ``native_band`` lists the bands where it is primary."""

    MICRO = "micro"
    INTRADAY = "intraday"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class DataTier(IntEnum):
    """Minimum input granularity, finest to coarsest. Ordering drives ``Registry.available``.

    A feature is computable on input bars of tier ``t`` iff ``t`` is at least as fine as the
    feature's declared tier -- i.e. finer input bars unlock strictly more features.
    """

    TICK = 0
    MINUTE = 1
    DAILY = 2


class Recurrence(Enum):
    """Tail-recompute behavior (FEATURES.md 7.2).

    FINITE: bounded window; tail-recompute is exact.
    RECURSIVE: unbounded memory (Wilder, EWM, cumulative); exact within tolerance after warmup.
    """

    FINITE = "finite"
    RECURSIVE = "recursive"


class Cost(Enum):
    """Per-update online cost hint."""

    O1 = "o1"
    LINEAR = "linear"
    HEAVY = "heavy"


# --- numeric constants (no magic numbers anywhere else) ----------------------------------------

# Default lookback grids per band, in trading bars (FEATURES.md 5). MICRO uses event windows,
# so it has no fixed grid.
HORIZON_LOOKBACKS: dict[Horizon, tuple[int, ...]] = {
    Horizon.MICRO: (),
    Horizon.INTRADAY: (12, 26, 78),
    Horizon.SHORT: (3, 5, 10),
    Horizon.MEDIUM: (21, 63, 126),
    Horizon.LONG: (126, 252, 504),
}

# Single declared tolerance for reference-value comparisons and FINITE parity (FEATURES.md 8.5).
# Recursive float accumulation is not bit-identical across platforms -- tolerance is the honest
# contract, never ``==`` (testing.md).
DEFAULT_FLOAT_TOLERANCE = 1e-9

# Convergence tolerance for RECURSIVE windowed-recompute parity, after effective_warmup burn-in.
PARITY_RECURSIVE_TOLERANCE = 1e-6

# Tolerance target used to derive analytic EWM warmup.
EWM_WARMUP_TOL = 1e-6

# Feature names are ``{measure}_{param}`` snake_case, unique library-wide (FEATURES.md 3.3).
NAME_PATTERN = r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"


def ewm_effective_warmup(alpha: float, tol: float = EWM_WARMUP_TOL) -> int:
    """Burn-in bars for an EWM with smoothing ``alpha`` to converge within ``tol``.

    Derived analytically (FEATURES.md 7.2): the residual weight on pre-window history decays as
    ``(1 - alpha)**n``, so ``n = ceil(ln(tol) / ln(1 - alpha))``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return ceil(log(tol) / log(1.0 - alpha))


# --- fingerprint -------------------------------------------------------------------------------


def _strip_docstring(node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    # A docstring survives ast.unparse as a string-literal statement, so a docstring- or
    # citation-only edit would otherwise change the fingerprint even though the formula did not.
    # Drop the leading string-literal expression so only the expression itself is hashed.
    body = node.body
    first = body[0] if body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        del body[0]


def _normalized_source(fn: Callable[..., object]) -> str:
    """Canonical, formatting-, comment- and docstring-independent source for ``fn``.

    Round-tripping through the AST drops comments and normalizes whitespace, so reformatting a
    feature (e.g. a ``ruff format`` pass) does not change its fingerprint -- only a real change to
    the expression does. The function name, decorators, and docstring are neutralized too: renaming,
    re-decorating, or editing the docstring/citation of a feature whose formula is unchanged must
    not bump its fingerprint. This is what makes the 3.4 immutability guarantee enforceable in CI.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    _strip_docstring(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            node.name = "_"
            node.decorator_list = []
            _strip_docstring(node)
    return ast.unparse(tree)


def _unwrap(fn: object) -> Callable[..., object] | None:
    """The underlying function behind a ``functools.partial`` (or ``fn`` itself), if callable."""
    if isinstance(fn, functools.partial):
        fn = fn.func
    return fn if callable(fn) else None


def _first_party_callees(fn: Callable[..., object]) -> list[Callable[..., object]]:
    """First-party (``sabia.*``) functions called by name within ``fn``'s body.

    Calls via attribute access (``pl.col``, ``expr.over``) are skipped -- those are third-party and
    pinned by the Polars version already in the payload. Only bare-name calls that resolve, through
    ``fn``'s globals, to a callable defined in a ``sabia`` module are followed.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    namespace = getattr(fn, "__globals__", {})
    callees: list[Callable[..., object]] = []
    for name in names:
        obj = namespace.get(name)
        if callable(obj) and getattr(obj, "__module__", "").startswith("sabia"):
            callees.append(obj)
    return callees


def _transitive_sources(roots: Iterable[object]) -> list[str]:
    """Normalized source of every ``root`` and, transitively, every first-party helper it calls.

    Hashing only the top-level function (the old behavior) left helper bodies -- ``safe_div``, the
    Rogers-Satchell ``_rs_term``, the cross-sectional reduction -- outside the fingerprint, so a
    change to their math would not trip the manifest gate. Following first-party callees closes that
    gap: the fingerprint covers the whole formula, not just its entry point. Sorted by qualified
    name so the payload is order-independent.
    """
    sources: dict[str, str] = {}
    stack = [fn for fn in (_unwrap(root) for root in roots) if fn is not None]
    while stack:
        fn = stack.pop()
        key = f"{fn.__module__}.{fn.__qualname__}"
        if key in sources:
            continue
        sources[key] = _normalized_source(fn)
        stack.extend(_first_party_callees(fn))
    return [source for _, source in sorted(sources.items())]


def feature_fingerprint(
    fn: Callable[..., pl.Expr],
    params: Mapping[str, object],
    *extra_fns: object,
    polars_version: str = pl.__version__,
) -> str:
    """Content hash over normalized formula source + params + the pinned Polars version (3.4).

    The hash covers ``fn`` and any ``extra_fns`` (e.g. a cross-sectional feature's reduction
    builder) with the transitive closure of first-party helpers each calls, so the whole formula is
    fingerprinted -- not just its entry point. Recorded alongside stored outputs so train-vs-serve
    identity is provable, not assumed. CI fails if a fingerprint changes without a version bump.
    """
    payload = " ".join(
        (
            *_transitive_sources((fn, *extra_fns)),
            json.dumps(params, sort_keys=True, default=str),
            polars_version,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- the spec ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable metadata describing one feature (FEATURES.md 3.1).

    Populated per feature in the family modules and collected by ``Registry.default``. The
    cross-cutting test harness reads these fields to drive causality, parity, dtype, and
    fingerprint-stability checks.
    """

    name: str
    version: int
    fingerprint: str
    family: Family
    native_band: frozenset[Horizon]
    lookback: int | None
    min_history: int
    recurrence: Recurrence
    effective_warmup: int
    cost_class: Cost
    data_tier: DataTier
    inputs: frozenset[Column]
    output_dtype: pl.DataType
    citation: str
    params: Mapping[str, object]


__all__ = [
    "DEFAULT_FLOAT_TOLERANCE",
    "EWM_WARMUP_TOL",
    "HORIZON_LOOKBACKS",
    "NAME_PATTERN",
    "PARITY_RECURSIVE_TOLERANCE",
    "Column",
    "Cost",
    "DataTier",
    "Family",
    "FeatureSpec",
    "Horizon",
    "Recurrence",
    "ewm_effective_warmup",
    "feature_fingerprint",
]
