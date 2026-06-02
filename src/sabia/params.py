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
        """Deterministic sorted repr-stable rendering: the fingerprint input (FEATURES.md 4.4)."""
        return ";".join(f"{k}={v!r}" for k, v in self._data.items())

    def __hash__(self) -> int:
        return hash(tuple(self._data.items()))


__all__ = ["FrozenParams", "ParamValue"]
