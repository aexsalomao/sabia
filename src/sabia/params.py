# FrozenParams (FEATURES.md 4.4): a feature's bound parameterization over hashable scalar values.
# Immutable and hashable so it can live inside a frozen FeatureSpec and fold deterministically into
# the fingerprint -- a parameter change must change the hash, making train/serve identity provable.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# v1 params are scalars (periods, spans, EWMA lambda, k). Widen here (and the recursive check) if a
# future param needs a tuple.
ParamValue = int | float | str | bool | None
_SCALAR_TYPES = (int, float, str, bool, type(None))


def _canonical_float(value: float) -> str:
    """Exact, platform-independent canonical form for a float (the fingerprint input, 4.4).

    ``repr`` is fragile for this: it is precision-tuned for round-tripping, but ``2.0`` vs ``2``
    differ and ``-0.0`` vs ``0.0`` differ even though they compare equal, so two equal floats could
    canonicalize -- and thus fingerprint -- differently. ``float.hex`` is the exact, round-trippable
    representation of the underlying IEEE-754 value, so equal floats always map to the same string.
    NaN/inf have no ``hex`` round-trip subtlety here (``float.hex`` renders them deterministically),
    and ``-0.0`` is normalized to ``+0.0`` so it agrees with ``0.0 == -0.0``.
    """
    if value == 0.0:  # collapse -0.0 and 0.0, which are equal but hex-distinct
        value = 0.0
    return value.hex()


def _canonical_scalar(value: ParamValue) -> str:
    """Canonical rendering of one param value. Floats go through ``_canonical_float``; every other
    scalar keeps its ``repr`` so int/str/bool/None canonical strings are byte-identical to before.

    ``bool`` is checked before ``float`` only incidentally -- ``bool`` is an ``int`` subclass, not a
    ``float``, so it never reaches the float branch; the explicit guard documents that intent.
    """
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, float):
        return _canonical_float(value)
    return repr(value)


@dataclass(frozen=True, slots=True)
class FrozenParams:
    """An immutable, hashable mapping of bound parameter names to scalar values.

    Validates at construction that every value is a hashable scalar (FEATURES.md 4.4) and stores a
    sorted, immutable view, so ``canonical()`` -- the fingerprint contribution -- is deterministic.
    """

    _data: Mapping[str, ParamValue] = field(default_factory=dict)

    def __init__(self, **kwargs: ParamValue) -> None:
        for key, value in kwargs.items():
            if not isinstance(value, _SCALAR_TYPES):
                raise TypeError(f"param {key!r} value {value!r} is not a hashable scalar")
        object.__setattr__(self, "_data", MappingProxyType(dict(sorted(kwargs.items()))))

    def __getitem__(self, key: str) -> ParamValue:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def items(self) -> list[tuple[str, ParamValue]]:
        return list(self._data.items())

    def as_dict(self) -> dict[str, ParamValue]:
        return dict(self._data)

    def canonical(self) -> str:
        """Deterministic sorted rendering of the bound params: the fingerprint input (4.4).

        Floats are rendered via ``_canonical_float`` (``float.hex``) rather than ``repr`` so the
        canonical string is exact and platform-independent -- two ``float`` values that compare
        equal always render identically, keeping ``canonical()`` consistent with
        ``__eq__``/``__hash__`` and the fingerprint stable across machines (determinism hardening,
        FEATURES.md 3.5). Non-float scalars keep their ``repr`` so int/str/bool/None params hash
        exactly as before.
        """
        return ";".join(f"{k}={_canonical_scalar(v)}" for k, v in self._data.items())

    def __hash__(self) -> int:
        return hash(tuple(self._data.items()))


__all__ = ["FrozenParams", "ParamValue"]
