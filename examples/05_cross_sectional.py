"""05 — Cross-sectional & factor-model features on a panel.

Cross-sectional features rank / standardize each name against the universe at each timestamp, so
they need a complete panel and a declared universe. The market-model features (beta, idio_vol)
regress each name's returns on the market_ret factor. All are point-in-time: a value at t uses only
information at or before t.

Run:  python examples/05_cross_sectional.py
"""

from __future__ import annotations

import polars as pl
from _data import default_schema, make_panel

import sabia


def main() -> None:
    panel = make_panel(n=400, symbols=("AAA", "BBB", "CCC", "DDD"))
    schema = default_schema()
    universe = ["AAA", "BBB", "CCC", "DDD"]

    xs_rank = sabia.cross_sectional.xs_rank_mom(formation=252, skip=21)
    beta = sabia.cross_sectional.beta(window=252)  # market beta vs market_ret
    idio = sabia.cross_sectional.idio_vol(window=252)  # residual (idiosyncratic) vol

    # Cross-sectional features require_universe: pass the universe so completeness is checked
    # against it, never inferred from whichever symbols happen to be present.
    out = sabia.compute(panel, xs_rank, beta, idio, schema=schema, universe=universe)
    keyed = panel.select("timestamp", "symbol").hstack(out)

    last_ts = panel.get_column("timestamp").max()
    print("cross-section on the final date:")
    print(keyed.filter(pl.col("timestamp") == last_ts).sort("xs_rank_mom_252", descending=True))

    # Forget the universe on a cross-sectional feature and compute() refuses, loudly:
    try:
        sabia.compute(panel, xs_rank, schema=schema)
    except ValueError as exc:
        print("\nmissing universe raises:", str(exc).split(";")[0])

    # A single cross-sectional feature can also be evaluated straight to a Series (two-pass under
    # the hood: per-symbol signal -> within-timestamp reduction):
    series = sabia.evaluate(panel, beta, schema)
    print("\nbeta_252 — non-null values:", series.drop_nulls().len(), "of", series.len())


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    main()
