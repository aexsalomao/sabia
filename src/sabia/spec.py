# The feature contract: FeatureSpec metadata, the structural/horizon enums, the canonical
# column names, the shared numeric constants, and the fingerprint helper. Everything else in
# sabia references the types defined here. See FEATURES.md sections 3 and 5.

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from collections.abc import Callable, Mapping
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


def _normalized_source(fn: Callable[..., pl.Expr]) -> str:
    """Canonical, formatting- and comment-independent source for ``fn``.

    Round-tripping through the AST drops comments and normalizes whitespace, so reformatting a
    feature (e.g. a ``ruff format`` pass) does not change its fingerprint -- only a real change to
    the expression does. The function name and decorators are neutralized too: renaming or
    re-decorating a feature whose formula is unchanged must not bump its fingerprint. This is what
    makes the 3.4 immutability guarantee enforceable in CI.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            node.name = "_"
            node.decorator_list = []
    return ast.unparse(tree)


def feature_fingerprint(
    fn: Callable[..., pl.Expr],
    params: Mapping[str, object],
    polars_version: str = pl.__version__,
) -> str:
    """Content hash over normalized formula source + params + the pinned Polars version (3.4).

    Recorded alongside stored outputs so train-vs-serve identity is provable, not assumed. CI
    fails if a fingerprint changes without a version bump.
    """
    payload = " ".join(
        (
            _normalized_source(fn),
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
