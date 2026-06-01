# The feature registry: a constructable catalog mapping (name, version) -> spec + bound builder.
# Built by EXPLICIT collection (FEATURES.md 6) -- there is no import-time decorator mutating a
# global singleton, so the registry is embeddable and test-isolatable by construction.

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass

import polars as pl

from sabia.spec import (
    NAME_PATTERN,
    Column,
    Cost,
    DataTier,
    Family,
    FeatureSpec,
    Horizon,
    Recurrence,
    feature_fingerprint,
)

# Family modules whose FEATURES tuples make up Registry.default(). Appended as each family lands so
# the default catalog grows explicitly -- never via import side effects.
_FAMILY_MODULES: tuple[str, ...] = (
    "sabia.returns",
    "sabia.volatility",
    "sabia.momentum",
)

_NAME_RE = re.compile(NAME_PATTERN)

# Most features output Float64; a module-level singleton avoids a call in argument defaults.
_DEFAULT_OUTPUT_DTYPE: pl.DataType = pl.Float64()


@dataclass(frozen=True, slots=True)
class RegisteredFeature:
    """One concrete, fully-parameterized feature: its spec plus a builder for the canonical expr.

    ``build`` is zero-arg because the parameterization (period, window, ...) is frozen into the
    spec; it returns the expression over the canonical OHLCV columns. Callers who need custom
    column names call the family function directly.
    """

    spec: FeatureSpec
    build: Callable[[], pl.Expr]


class Registry:
    """A catalog of features queryable by horizon, data tier, or arbitrary predicate.

    Construct one explicitly from ``RegisteredFeature`` objects, or use ``Registry.default`` for the
    shipped set. ``where`` / ``available`` return new (sub-)registries, so queries compose.
    """

    def __init__(self, features: Iterable[RegisteredFeature] = ()) -> None:
        self._by_key: dict[tuple[str, int], RegisteredFeature] = {}
        for feature in features:
            self.add(feature)

    def add(self, feature: RegisteredFeature) -> None:
        """Register a feature. Raises on a malformed name or a duplicate ``(name, version)``."""
        spec = feature.spec
        if not _NAME_RE.match(spec.name):
            raise ValueError(
                f"feature name {spec.name!r} is not snake_case (pattern {NAME_PATTERN})"
            )
        key = (spec.name, spec.version)
        if key in self._by_key:
            raise ValueError(f"duplicate feature {spec.name!r} version {spec.version}")
        self._by_key[key] = feature

    def get(self, name: str, version: int | None = None) -> RegisteredFeature:
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

    def features(self) -> list[RegisteredFeature]:
        return list(self._by_key.values())

    def names(self) -> list[str]:
        return [spec.name for spec in self.specs()]

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[RegisteredFeature]:
        return iter(self._by_key.values())

    def __contains__(self, name: object) -> bool:
        return any(n == name for (n, _) in self._by_key)

    @classmethod
    def default(cls) -> Registry:
        """Assemble the shipped feature set by importing each family module's ``FEATURES`` tuple."""
        features: list[RegisteredFeature] = []
        for module_name in _FAMILY_MODULES:
            module = importlib.import_module(module_name)
            features.extend(module.FEATURES)
        return cls(features)


def make_feature(
    fn: Callable[..., pl.Expr],
    build: Callable[[], pl.Expr],
    *,
    name: str,
    family: Family,
    native_band: Iterable[Horizon],
    lookback: int | None,
    min_history: int,
    recurrence: Recurrence,
    effective_warmup: int,
    cost_class: Cost,
    inputs: Iterable[Column],
    citation: str,
    params: Mapping[str, object],
    output_dtype: pl.DataType = _DEFAULT_OUTPUT_DTYPE,
    data_tier: DataTier = DataTier.DAILY,
    version: int = 1,
) -> RegisteredFeature:
    """Build a ``RegisteredFeature``: the single construction point for shipped features.

    ``fn`` is the formula function (used for the fingerprint); ``build`` is the zero-arg closure
    producing the canonical expression. The fingerprint is derived from ``fn`` and ``params`` so
    train-vs-serve identity is provable (FEATURES.md 3.4).
    """
    spec = FeatureSpec(
        name=name,
        version=version,
        fingerprint=feature_fingerprint(fn, params),
        family=family,
        native_band=frozenset(native_band),
        lookback=lookback,
        min_history=min_history,
        recurrence=recurrence,
        effective_warmup=effective_warmup,
        cost_class=cost_class,
        data_tier=data_tier,
        inputs=frozenset(inputs),
        output_dtype=output_dtype,
        citation=citation,
        params=dict(params),
    )
    return RegisteredFeature(spec=spec, build=build)


__all__ = ["RegisteredFeature", "Registry", "make_feature"]
