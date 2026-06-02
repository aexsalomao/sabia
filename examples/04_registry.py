"""04 — The registry: discover and query the shipped feature set.

Registry.default() is the frozen catalog of every shipped feature, assembled by explicit collection
(no import-time magic). Query it by horizon band, data tier, family, or any predicate over the
FeatureSpec — each query returns a new sub-registry, so they compose.

Run:  python examples/04_registry.py
"""

from __future__ import annotations

from _data import default_schema, make_ohlcv

import sabia
from sabia import DataTier, Family, Horizon


def main() -> None:
    reg = sabia.Registry.default()
    print(f"shipped features: {len(reg)}")

    # Filter by horizon band (where a feature is "primary"):
    medium = reg.where(lambda s: Horizon.MEDIUM in s.native_band)
    print(f"\nMEDIUM-band features ({len(medium)}):", medium.names()[:8], "...")

    # Filter by family:
    vol = reg.where(lambda s: s.family is Family.VOLATILITY)
    print(f"volatility family ({len(vol)}):", vol.names())

    # Filter by data tier — finer input bars unlock strictly more features. Everything in v1 is
    # computable on DAILY bars:
    print("computable on DAILY bars:", len(reg.available(DataTier.DAILY)))

    # Predicates compose; .specs() / .names() expose the contents.
    cheap_decay = reg.where(
        lambda s: s.recurrence is sabia.Recurrence.RECURSIVE_DECAY and s.cost_class is sabia.Cost.O1
    )
    print("\nO(1) recursive-decay features:", cheap_decay.names())

    # Look up one feature and inspect its full contract metadata:
    spec = reg.get("rsi_14").spec
    print("\n--- rsi_14 spec ---")
    print(f"  family         : {spec.family.value}")
    print(f"  recurrence     : {spec.recurrence.value}")
    print(f"  min_history    : {spec.min_history}  (emits null before this)")
    print(f"  effective_warmup: {spec.effective_warmup}")
    print(f"  input_roles    : {sorted(str(r) for r in spec.input_roles)}")
    print(f"  output_unit    : {spec.output_unit.value}")
    print(f"  evidence       : {spec.evidence.value}")
    print(f"  citation       : {spec.citation}")
    print(f"  fingerprint    : {spec.fingerprint}")

    # A sub-registry is just features — bind nothing extra, compute straight from it:
    feature = reg.get("vol_yz_21")
    out = sabia.compute(make_ohlcv(n=120), feature, schema=default_schema())
    print("\nvol_yz_21 last value:", out.tail(1).item())


if __name__ == "__main__":
    main()
