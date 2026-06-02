"""03 — Validation: the input contract is the one fail-loud surface.

sabia.validate() checks a frame against the contract every feature assumes (sorted/unique tz-aware
UTC timestamps, required role columns + dtypes, OHLC ordering, complete cross-section, bars closed).
Feature bodies then trust the input and never re-check. Three modes:

    STRICT   (default) — raise on any violation
    RESEARCH           — warn on completeness/finalization, still raise on schema/dtype/role/order
    OFF                — skip validation entirely

compute(..., validation=...) uses the same enum, so the vocabulary is shared.

Run:  python examples/03_validation.py
"""

from __future__ import annotations

from _data import default_schema, make_ohlcv

import sabia
from sabia import ValidationMode


def main() -> None:
    schema = default_schema()
    good = make_ohlcv(n=60)

    # A clean frame returns an empty warning list under STRICT.
    warnings = sabia.validate(good, schema=schema, mode=ValidationMode.STRICT)
    print("clean frame, STRICT warnings:", warnings)

    # Break the OHLC ordering (low above high) -> STRICT raises a precise error.
    broken = good.with_columns((good["high"] + 1.0).alias("low"))
    try:
        sabia.validate(broken, schema=schema, mode=ValidationMode.STRICT)
    except sabia.SabiaValidationError as exc:
        print("\nOHLC violation raises:", exc)

    # Duplicate a timestamp -> not strictly increasing -> raises in every non-OFF mode.
    dupes = good.head(1).vstack(good)
    try:
        sabia.validate(dupes, schema=schema, mode=ValidationMode.STRICT)
    except sabia.SabiaValidationError as exc:
        print("duplicate timestamp raises:", exc)

    # Finalization is a *soft* check: mark a bar not-final and point the schema at the flag column.
    # STRICT raises; RESEARCH downgrades it to a returned warning (the caller decides what to do).
    soft_schema = sabia.BarSchema(roles=dict(schema.roles), closed_col="is_final")
    partial = good.with_columns(
        (good["timestamp"] != good["timestamp"].max()).alias("is_final")  # last bar still forming
    )
    research = sabia.validate(partial, schema=soft_schema, mode=ValidationMode.RESEARCH)
    print("\nRESEARCH mode warnings:", research)
    try:
        sabia.validate(partial, schema=soft_schema, mode=ValidationMode.STRICT)
    except sabia.SabiaValidationError as exc:
        print("same frame under STRICT raises:", exc)

    # OFF skips all checks (use only when the frame is already trusted upstream).
    print("\nOFF warnings:", sabia.validate(broken, schema=schema, mode=ValidationMode.OFF))


if __name__ == "__main__":
    main()
