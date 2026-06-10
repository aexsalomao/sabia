# The feature registry: a constructable, freezable catalog of bound features. Built by EXPLICIT
# collection (FEATURES.md 7) -- no import-time decorator mutates a global, so the registry is
# embeddable and test-isolatable by construction. `bind_feature` is the single construction point
# for a shipped feature: it builds the spec (incl. the fingerprint) and wraps it in a BoundFeature.

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterable, Iterator

import polars as pl

from sabia.params import FrozenParams
from sabia.references import Citation
from sabia.schema import BarSchema
from sabia.spec import (
    ALLOWED_RECURRENCES,
    NAME_PATTERN,
    REQUIRE_FULL_WINDOW,
    BoundFeature,
    Cost,
    DataTier,
    Evidence,
    Family,
    FeatureSpec,
    Horizon,
    NullPolicy,
    Recurrence,
    Unit,
    feature_fingerprint,
)
from sabia.typing import FeatureRef, InputRole

# Family modules whose FEATURES tuples make up Registry.default(). Appended explicitly so the
# default catalog grows by intent -- never via import side effects.
_FAMILY_MODULES: tuple[str, ...] = (
    "sabia.returns",
    "sabia.volatility",
    "sabia.momentum",
    "sabia.trend",
    "sabia.volume",
    "sabia.distribution",
    "sabia.mean_reversion",
    "sabia.seasonality",
    "sabia.cross_sectional",
    "sabia.microstructure",
)

_NAME_RE = re.compile(NAME_PATTERN)

_DEFAULT_OUTPUT_DTYPE: pl.DataType = pl.Float64()

# Intermediate column holding a cross-sectional feature's per-symbol signal during two-pass
# evaluation (Polars cannot nest .over(symbol) inside .over(timestamp) in one expression).
XS_SIGNAL_COLUMN = "__sabia_xs_signal__"


class FrozenRegistryError(RuntimeError):
    """Raised when mutating a frozen registry (FEATURES.md 7 -- built-ins are not overridable)."""


class Registry:
    """A catalog of bound features queryable by horizon, data tier, or arbitrary predicate.

    Construct one explicitly from ``BoundFeature`` objects, or use ``Registry.default`` for the
    shipped set (assembled then frozen). ``where`` / ``available`` return new (sub-)registries, so
    queries compose. A frozen registry rejects further ``add``.
    """

    def __init__(self, features: Iterable[BoundFeature] = (), *, frozen: bool = False) -> None:
        self._by_key: dict[tuple[str, int], BoundFeature] = {}
        self._frozen = False
        for feature in features:
            self.add(feature)
        self._frozen = frozen

    def add(self, feature: BoundFeature) -> None:
        """Register a feature. Raises on bad name, dup key, banned recurrence, or frozen reg."""
        if self._frozen:
            raise FrozenRegistryError("cannot add to a frozen registry")
        spec = feature.spec
        if not _NAME_RE.match(spec.name):
            raise ValueError(
                f"feature name {spec.name!r} is not snake_case (pattern {NAME_PATTERN})"
            )
        if spec.recurrence not in ALLOWED_RECURRENCES:
            raise ValueError(
                f"feature {spec.name!r} has recurrence {spec.recurrence.value}, not admitted "
                f"(only {sorted(r.value for r in ALLOWED_RECURRENCES)})"
            )
        key = (spec.name, spec.version)
        if key in self._by_key:
            raise ValueError(f"duplicate feature {spec.name!r} version {spec.version}")
        self._by_key[key] = feature

    def freeze(self) -> Registry:
        """Return a new frozen registry over the same features (FEATURES.md 7)."""
        return Registry(self._by_key.values(), frozen=True)

    def get(self, name: str, version: int | None = None) -> BoundFeature:
        """Look up a feature by name; defaults to the highest registered version."""
        versions = sorted(v for (n, v) in self._by_key if n == name)
        if not versions:
            raise KeyError(f"no feature named {name!r}")
        resolved = versions[-1] if version is None else version
        try:
            return self._by_key[(name, resolved)]
        except KeyError:
            raise KeyError(f"feature {name!r} has no version {resolved}") from None

    def where(self, predicate: Callable[[FeatureSpec], bool]) -> Registry:
        """A new registry of features whose spec satisfies ``predicate`` (e.g. by band)."""
        return Registry(f for f in self._by_key.values() if predicate(f.spec))

    def available(self, tier: DataTier) -> Registry:
        """Features computable on input bars at ``tier``.

        A feature is computable iff the bars are at least as fine as it requires. Finer enum values
        are coarser tiers, so the bars qualify when ``tier <= spec.data_tier``.
        """
        return self.where(lambda s: tier <= s.data_tier)

    def specs(self) -> list[FeatureSpec]:
        """All specs, in registration order -- the harness parametrizes over this."""
        return [f.spec for f in self._by_key.values()]

    def features(self) -> list[BoundFeature]:
        return list(self._by_key.values())

    def names(self) -> list[str]:
        return [spec.name for spec in self.specs()]

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[BoundFeature]:
        return iter(self._by_key.values())

    def __contains__(self, name: object) -> bool:
        return any(n == name for (n, _) in self._by_key)

    @classmethod
    def from_modules(cls, module_names: Iterable[str]) -> Registry:
        """Assemble a registry by importing each module's ``FEATURES`` tuple (not yet frozen)."""
        features: list[BoundFeature] = []
        for module_name in module_names:
            module = importlib.import_module(module_name)
            features.extend(module.FEATURES)
        return cls(features)

    @classmethod
    def default(cls) -> Registry:
        """The shipped feature set, assembled from the family modules and frozen (FEATURES.md 7)."""
        return cls.from_modules(_FAMILY_MODULES).freeze()


def bind_feature(
    build: Callable[[BarSchema], pl.Expr],
    *,
    name: str,
    family: Family,
    native_band: Iterable[Horizon],
    lookback: int | None,
    min_history: int,
    recurrence: Recurrence,
    effective_warmup: int,
    cost_class: Cost,
    input_roles: Iterable[InputRole],
    output_unit: Unit,
    evidence: Evidence,
    citation: Citation,
    params: FrozenParams,
    output_dtype: pl.DataType = _DEFAULT_OUTPUT_DTYPE,
    output_range: tuple[float, float] | None = None,
    null_policy: NullPolicy = REQUIRE_FULL_WINDOW,
    data_tier: DataTier = DataTier.DAILY,
    dependencies: Iterable[FeatureRef] = (),
    requires_universe: bool = False,
    requires_complete_panel: bool = False,
    version: int = 1,
    signal: Callable[[BarSchema], pl.Expr] | None = None,
) -> BoundFeature:
    """Construct a ``BoundFeature``: the single construction point for shipped features (4.1, 4.2).

    ``build`` is the schema-resolving formula closure (and the fingerprint subject); ``signal`` is
    the per-symbol pre-pass for cross-sectional features. The fingerprint folds the params, roles,
    and dependency fingerprints together with the transitive source of ``build``/``signal`` -- so a
    role swap or a helper-math change is provable at the manifest gate (FEATURES.md 4.4).
    """
    roles = frozenset(input_roles)
    deps = tuple(dependencies)
    spec = FeatureSpec(
        name=name,
        version=version,
        fingerprint=feature_fingerprint(
            canonical_id=name,
            version=version,
            params=params,
            input_roles=roles,
            build=build,
            signal=signal,
            dependencies=deps,
        ),
        family=family,
        native_band=frozenset(native_band),
        lookback=lookback,
        min_history=min_history,
        recurrence=recurrence,
        effective_warmup=effective_warmup,
        cost_class=cost_class,
        data_tier=data_tier,
        input_roles=roles,
        null_policy=null_policy,
        output_dtype=output_dtype,
        output_unit=output_unit,
        output_range=output_range,
        evidence=evidence,
        dependencies=deps,
        requires_universe=requires_universe,
        requires_complete_panel=requires_complete_panel,
        citation=citation,
        params=params,
    )
    return BoundFeature(spec=spec, build=build, signal=signal)


def evaluate(
    frame: pl.DataFrame | pl.LazyFrame, feature: BoundFeature, schema: BarSchema
) -> pl.Series:
    """Evaluate one feature to a Series, two-pass for cross-sectional features.

    A cross-sectional feature's per-symbol ``signal`` is materialized first (Polars cannot nest
    ``.over(symbol)`` inside ``.over(timestamp)`` in a single expression); time-series features
    evaluate in a single ``select``.
    """
    lf = frame.lazy()
    if feature.signal is not None:
        lf = lf.with_columns(feature.signal(schema).alias(XS_SIGNAL_COLUMN))
    # Alias to the feature name unconditionally: a time-series ``build`` already aliases itself, but
    # a cross-sectional reduction returns a bare (possibly unnamed) expression, so without this the
    # output Series would carry an incidental name (``__sabia_xs_signal__`` / ``literal``) and two
    # such features would collide by name in ``compute`` (FEATURES.md 4.3).
    return lf.select(feature.expr(schema).alias(feature.spec.name)).collect().to_series()


__all__ = [
    "XS_SIGNAL_COLUMN",
    "BoundFeature",
    "FrozenRegistryError",
    "Registry",
    "bind_feature",
    "evaluate",
]
