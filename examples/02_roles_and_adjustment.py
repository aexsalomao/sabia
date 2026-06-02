"""02 — Roles & adjustment: why sabia takes column *roles*, not column names.

sabia adjusts nothing and infers nothing. You declare which adjustment basis each column carries,
and each feature declares which basis it needs:

    * returns / momentum / trend / oscillators use close@tr  (total return, incl. dividends)
    * range estimators (Yang-Zhang, ATR, Bollinger ...) use o/h/l/c@split (split-only)

`close@tr` and `close@split` are DISTINCT roles. If they live in different physical columns, a
feature reading the wrong basis is impossible — the schema routes each role to the right column.

Run:  python examples/02_roles_and_adjustment.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import polars as pl

import sabia
from sabia import Adjustment, BarSchema, PriceField, PriceRole

# This example is self-contained (it does not import the shared _data helper), so make stdout UTF-8
# here too: Polars' table glyphs would otherwise raise UnicodeEncodeError on a cp1252 console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _dual_basis_frame(n: int = 60) -> pl.DataFrame:
    # A frame where total-return close and split-only close are genuinely different series.
    start = datetime(2021, 1, 1, tzinfo=UTC)
    tr = [100.0 * (1.001**i) for i in range(n)]  # dividends reinvested -> drifts up faster
    split = [80.0 * (1.0005**i) for i in range(n)]  # split-only -> lower level, milder drift
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n)],
            "symbol": ["AAA"] * n,
            "close_tr": tr,
            "close_split": split,
            "high_split": [s * 1.01 for s in split],
            "low_split": [s * 0.99 for s in split],
            "open_split": split,
        }
    )


def main() -> None:
    frame = _dual_basis_frame()

    # The same field (CLOSE) appears twice with different adjustments -> different columns.
    schema = BarSchema(
        roles={
            PriceRole(PriceField.CLOSE, Adjustment.TR): "close_tr",
            PriceRole(PriceField.CLOSE, Adjustment.SPLIT): "close_split",
            PriceRole(PriceField.HIGH, Adjustment.SPLIT): "high_split",
            PriceRole(PriceField.LOW, Adjustment.SPLIT): "low_split",
            PriceRole(PriceField.OPEN, Adjustment.SPLIT): "open_split",
        }
    )

    # ret_log declares close@tr -> reads close_tr; vol_yz declares o/h/l/c@split -> reads the split
    # columns. Neither can accidentally touch the other basis.
    ret = sabia.returns.ret_log(period=1)  # close@tr
    yz = sabia.volatility.vol_yz(window=10)  # o/h/l/c@split

    out = sabia.compute(frame, ret, yz, schema=schema)
    print(frame.select("timestamp").hstack(out).tail(4))

    # A schema resolves a role to its physical column on demand:
    print("\nclose@tr    ->", schema.column(PriceRole(PriceField.CLOSE, Adjustment.TR)))
    print("close@split ->", schema.column(PriceRole(PriceField.CLOSE, Adjustment.SPLIT)))

    # Ask for a role the schema does not declare and you get a precise error, not a silent fallback.
    try:
        schema.column(PriceRole(PriceField.VWAP, Adjustment.SPLIT))
    except KeyError as exc:
        print("\nmissing role raises:", str(exc).split(";")[0])


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    main()
