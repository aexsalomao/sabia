# The feature contract: FeatureSpec metadata, BoundFeature, the structural/horizon/recurrence enums,
# the output/evidence/null-policy/validation enums, the shared numeric constants, and the
# fingerprint helper. Other modules reference the types defined here. See FEATURES.md sections 4, 8.

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import ceil, log
from typing import TYPE_CHECKING

import polars as pl

from sabia.params import FrozenParams
from sabia.references import Citation
from sabia.schema import BarSchema
from sabia.typing import Adjustment, FeatureRef, InputRole

if TYPE_CHECKING:
    from sabia.manifest import TransformRef

# Adjustment is imported and re-exported here (in __all__) per the FEATURES.md 7 module map:
# spec.py owns the public enum surface, even though Adjustment is defined in typing.py.

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
    """Tail-recompute behavior (FEATURES.md 8.2).

    FINITE: bounded window; tail-recompute is exact.
    RECURSIVE_DECAY: decaying memory (Wilder, EWM); exact within tolerance after effective_warmup.
    PATH_DEPENDENT: resets/triggers (SAR, CUSUM); parity is replay-based -- banned in v1.
    EXPANDING: unbounded cumulative (raw OBV, A/D) -- banned in v1; ship differenced/bounded.
    """

    FINITE = "finite"
    RECURSIVE_DECAY = "recursive_decay"
    PATH_DEPENDENT = "path_dependent"
    EXPANDING = "expanding"


# The recurrence classes v1 ships; the registry rejects the rest (FEATURES.md 8.2).
V1_RECURRENCES: frozenset[Recurrence] = frozenset({Recurrence.FINITE, Recurrence.RECURSIVE_DECAY})


class Cost(Enum):
    """Per-update online cost hint."""

    O1 = "o1"
    LINEAR = "linear"
    HEAVY = "heavy"


class Unit(Enum):
    """The unit of a feature's output (FEATURES.md 4.1, 4.6)."""

    LOG_RETURN = "log_return"
    RATIO = "ratio"
    INDEX_0_100 = "index_0_100"
    UNITLESS = "unitless"
    RETURN_STD_PER_BAR = "return_std_per_bar"
    PRICE_UNITS = "price_units"
    RANK_0_1 = "rank_0_1"
    ZSCORE = "zscore"


class Evidence(Enum):
    """Empirical standing of the feature *as constructed*; not a predictability claim (4.1.1)."""

    FORMULA_ONLY = "formula_only"
    TA_CANON = "ta_canon"
    ACADEMIC_SINGLE = "academic_single"
    ACADEMIC_REPLICATED = "academic_replicated"


class ValidationMode(Enum):
    """Boundary-validation strictness (FEATURES.md 8.3). One vocabulary for validate + compute."""

    STRICT = "strict"  # raise on any contract violation
    RESEARCH = "research"  # warn on completeness/finalization; still raise on schema/dtype/role
    OFF = "off"  # no validation


# --- null policy (FEATURES.md 4.5) -------------------------------------------------------------


class NullKind(Enum):
    """How a rolling window treats nulls within it."""

    REQUIRE_FULL_WINDOW = "require_full_window"
    MIN_VALID_COUNT = "min_valid_count"
    SKIP_NULLS = "skip_nulls"


@dataclass(frozen=True, slots=True)
class NullPolicy:
    """Window-null policy (FEATURES.md 4.5). ``min_valid`` is set iff kind is MIN_VALID_COUNT."""

    kind: NullKind
    min_valid: int | None = None

    def __post_init__(self) -> None:
        if (self.kind is NullKind.MIN_VALID_COUNT) != (self.min_valid is not None):
            raise ValueError("min_valid must be set iff kind is MIN_VALID_COUNT")


REQUIRE_FULL_WINDOW = NullPolicy(NullKind.REQUIRE_FULL_WINDOW)
SKIP_NULLS = NullPolicy(NullKind.SKIP_NULLS)


def min_valid_count(n: int) -> NullPolicy:
    """A NullPolicy that emits once at least ``n`` non-null values are in the window."""
    return NullPolicy(NullKind.MIN_VALID_COUNT, n)


# --- numeric constants (no magic numbers anywhere else) ----------------------------------------

# Default lookback grids per band, in trading bars (FEATURES.md 6). MICRO uses event windows, so it
# has no fixed grid.
HORIZON_LOOKBACKS: dict[Horizon, tuple[int, ...]] = {
    Horizon.MICRO: (),
    Horizon.INTRADAY: (12, 26, 78),
    Horizon.SHORT: (3, 5, 10),
    Horizon.MEDIUM: (21, 63, 126),
    Horizon.LONG: (126, 252, 504),
}

# Single declared tolerance for reference-value comparisons and FINITE parity (FEATURES.md 9).
# Recursive float accumulation is not bit-identical across platforms -- tolerance is the honest
# contract, never ``==`` (testing.md).
DEFAULT_FLOAT_TOLERANCE = 1e-9

# Tolerance target used to derive the analytic EWM warmup: the residual weight on pre-window history
# decays as (1-alpha)**n, so the burn-in nulls everything until that weight is below this. Set well
# below the parity tolerance below so the recomputed value converges with headroom -- the diff() and
# gain/loss ratio inside Wilder-style features amplify the raw weight residual by a small factor, so
# burning to the parity tolerance itself would leave no margin (the v4 contract was one decay-bar
# short here; deriving the two together restores the guarantee).
EWM_WARMUP_TOL = 1e-8

# Convergence tolerance for RECURSIVE_DECAY windowed-recompute parity, after effective_warmup.
# DERIVED from the warmup target with a fixed margin so the two cannot drift apart: tightening the
# burn-in automatically tightens what parity asserts, and the headroom (100x) absorbs the recursive
# amplification noted above. Never bit-equality -- recursive float accumulation is platform-bound.
_PARITY_WARMUP_MARGIN = 100.0
PARITY_RECURSIVE_TOLERANCE = EWM_WARMUP_TOL * _PARITY_WARMUP_MARGIN

# Feature names are ``{measure}_{params}`` snake_case, unique library-wide (FEATURES.md 4.3).
NAME_PATTERN = r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"


def ewm_effective_warmup(alpha: float, tol: float = EWM_WARMUP_TOL) -> int:
    """Burn-in bars for an EWM with smoothing ``alpha`` to converge within ``tol``.

    Derived analytically (FEATURES.md 8.2): the residual weight on pre-window history decays as
    ``(1 - alpha)**n``, so ``n = ceil(ln(tol) / ln(1 - alpha))``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return ceil(log(tol) / log(1.0 - alpha))


# --- fingerprint -------------------------------------------------------------------------------

# Unit separator: joins fingerprint payload segments without colliding with any textual content.
_PAYLOAD_SEP = "␟"


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


# Deterministic memoization, not banned mutable state (FEATURES.md 3.3): the cache maps a function's
# identity to the pure, immutable result of hashing its source. The output depends only on ``fn``'s
# code, never on call order or external state, so it is purely a build-time speedup -- repeatability
# is unaffected. (3.3 bans state that changes a feature's *output*; this changes nothing visible.)
@functools.cache
def _normalized_source(fn: Callable[..., object]) -> str:
    """Canonical, formatting-, comment- and docstring-independent source for ``fn``.

    Round-tripping through the AST drops comments and normalizes whitespace, so reformatting a
    feature (e.g. a ``ruff format`` pass) does not change its fingerprint -- only a real change to
    the expression does. The function name, decorators, and docstring are neutralized too: renaming,
    re-decorating, or editing the docstring/citation of a feature whose formula is unchanged must
    not bump its fingerprint. This is what makes the 4.4 immutability guarantee enforceable in CI.

    Cached by function identity: the shared helpers (``safe_div``, ``grouped`` ...) would be
    re-hashed for every feature that calls them at import time, so memoizing the AST work makes
    fingerprinting O(unique functions) instead of O(features x helpers).
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    _strip_docstring(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            node.name = "_"
            node.decorator_list = []
            _strip_docstring(node)
    return ast.unparse(tree)


# Constant types folded into the fingerprint by value (a tuned literal must bump the hash).
_CONSTANT_TYPES = (int, float, str, bool, tuple)


# Deterministic memoization, not banned mutable state (FEATURES.md 3.3): like
# ``_normalized_source``, this is a pure function-identity -> result mapping (the constants ``fn``
# reads are fixed by its source). Caching only avoids recomputing the same AST walk; it never
# affects what is produced.
@functools.cache
def _module_constants(fn: Callable[..., object]) -> tuple[str, ...]:
    """``name=repr(value)`` for every module-level scalar/tuple constant ``fn`` reads by name.

    ``_first_party_callees`` follows only bare-name *calls*, so a module constant used as a *value*
    (``_CCI_SCALE * mad``, ``2.0 * _LN2``, ``_CS_DENOM``) would otherwise sit outside the hashed
    source -- retuning its value would silently change every output without bumping the fingerprint,
    defeating the 4.4 train-vs-serve guarantee. Resolving the Load-context names against ``fn``'s
    globals and folding the resolved values in closes that gap; the values are hashed, so an edit to
    the literal is provable at the manifest gate. Callables, modules, and types are skipped (those
    are pinned via the source closure / Polars version already).
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    namespace = getattr(fn, "__globals__", {})
    refs: list[str] = []
    for name in names:
        if name not in namespace:
            continue
        obj = namespace[name]
        if isinstance(obj, _CONSTANT_TYPES) and not callable(obj):
            refs.append(f"{name}={obj!r}")
    return tuple(sorted(refs))


def _unwrap(fn: object) -> Callable[..., object] | None:
    """The underlying function behind a ``functools.partial`` (or ``fn`` itself), if callable."""
    if isinstance(fn, functools.partial):
        fn = fn.func
    return fn if callable(fn) else None


def _first_party_callees(fn: Callable[..., object]) -> list[Callable[..., object]]:
    """First-party (``sabia.*``) functions called by name within ``fn``'s body.

    Calls via attribute access (``pl.col``, ``schema.column``) are skipped -- those are third-party
    or method calls and pinned elsewhere. Only bare-name calls that resolve, through ``fn``'s
    globals, to a callable defined in a ``sabia`` module are followed.
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

    Hashing only the top-level builder would leave helper bodies -- ``safe_div``, the
    Rogers-Satchell term, the cross-sectional reduction -- outside the fingerprint, so a change to
    their math would not trip the manifest gate. Following first-party callees closes that gap: the
    fingerprint covers the whole formula, not just its entry point. Sorted by name, order-stable.
    """
    sources: dict[str, str] = {}
    stack = [fn for fn in (_unwrap(root) for root in roots) if fn is not None]
    while stack:
        fn = stack.pop()
        key = f"{fn.__module__}.{fn.__qualname__}"
        if key in sources:
            continue
        # Hash the normalized body AND the values of any module constants it reads by name, so a
        # tuned literal (e.g. _CCI_SCALE) bumps the fingerprint just like an edited expression.
        sources[key] = "".join((_normalized_source(fn), *_module_constants(fn)))
        stack.extend(_first_party_callees(fn))
    return [source for _, source in sorted(sources.items())]


def feature_fingerprint(
    *,
    canonical_id: str,
    version: int,
    params: FrozenParams,
    input_roles: Iterable[InputRole],
    build: Callable[[BarSchema], pl.Expr],
    signal: Callable[[BarSchema], pl.Expr] | None = None,
    dependencies: Iterable[FeatureRef | TransformRef] = (),
    polars_version: str = pl.__version__,
) -> str:
    """Content hash over the bound identity (FEATURES.md 4.4).

    Folds ``canonical_id + version + bound params + input_roles + dependency fingerprints + the
    pinned Polars version`` together with the normalized source of ``build`` (and ``signal``) and
    the transitive closure of first-party helpers each calls. Because the role set folds in,
    ``rsi_14`` (``close@tr``) and ``rsi_raw_14`` (``close@raw``) are distinct despite equal source.
    Recorded alongside stored outputs so train-vs-serve identity is provable; CI fails if a
    fingerprint changes without a version bump.
    """
    builders: tuple[object, ...] = (build,) if signal is None else (build, signal)
    payload = _PAYLOAD_SEP.join(
        (
            canonical_id,
            str(version),
            params.canonical(),
            "|".join(sorted(str(role) for role in input_roles)),
            "|".join(sorted(dep.fingerprint for dep in dependencies)),
            polars_version,
            *_transitive_sources(builders),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def transform_fingerprint(
    *,
    canonical_id: str,
    version: int,
    params: FrozenParams,
    apply: Callable[[pl.Expr], pl.Expr],
    polars_version: str = pl.__version__,
) -> str:
    """Content hash for a normalization transform (FEATURES.md 5), so the manifest pins it too.

    Same machinery as ``feature_fingerprint`` minus roles: hashes the canonical id, version, bound
    params, the pinned Polars version, and the transitive source of the ``apply`` closure.
    """
    payload = _PAYLOAD_SEP.join(
        (
            canonical_id,
            str(version),
            params.canonical(),
            polars_version,
            *_transitive_sources((apply,)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- the spec ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable metadata describing one bound feature (FEATURES.md 4.1).

    Populated by ``registry.bind_feature`` and exposed on each ``BoundFeature``. The cross-cutting
    test harness reads these fields to drive causality, parity, dtype, role-misuse, and
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
    input_roles: frozenset[InputRole]
    null_policy: NullPolicy
    output_dtype: pl.DataType
    output_unit: Unit
    output_range: tuple[float, float] | None
    evidence: Evidence
    dependencies: tuple[FeatureRef | TransformRef, ...]
    requires_universe: bool
    requires_complete_panel: bool
    citation: Citation
    params: FrozenParams


@dataclass(frozen=True, slots=True)
class BoundFeature:
    """A feature with its params bound: an immutable spec + a schema-resolving expression (4.2).

    ``build`` and ``signal`` take a ``BarSchema`` so roles resolve to physical columns at build time
    -- purity intact, because the schema is an explicit argument, never global. ``signal`` is set
    only for cross-sectional features: it builds the per-symbol pre-pass that the evaluator
    materializes before ``build`` reduces across the cross-section.
    """

    spec: FeatureSpec
    build: Callable[[BarSchema], pl.Expr]
    signal: Callable[[BarSchema], pl.Expr] | None = field(default=None)

    def expr(self, schema: BarSchema) -> pl.Expr:
        """Resolve roles against ``schema`` and return the canonical expression."""
        return self.build(schema)


__all__ = [
    "DEFAULT_FLOAT_TOLERANCE",
    "EWM_WARMUP_TOL",
    "HORIZON_LOOKBACKS",
    "NAME_PATTERN",
    "PARITY_RECURSIVE_TOLERANCE",
    "REQUIRE_FULL_WINDOW",
    "SKIP_NULLS",
    "V1_RECURRENCES",
    "Adjustment",
    "BoundFeature",
    "Cost",
    "DataTier",
    "Evidence",
    "Family",
    "FeatureSpec",
    "Horizon",
    "NullKind",
    "NullPolicy",
    "Recurrence",
    "Unit",
    "ValidationMode",
    "ewm_effective_warmup",
    "feature_fingerprint",
    "min_valid_count",
    "transform_fingerprint",
]
