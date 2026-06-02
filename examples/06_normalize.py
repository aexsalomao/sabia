"""06 — Normalization transforms: compose trailing / cross-sectional standardizers.

Transforms are separate from features (FEATURES.md 5) but obey the same causality invariant. A
transform factory returns a BoundTransform whose .apply(expr) -> expr you pipe onto a feature
expression (or a plain column). They carry their own pinned spec, so the manifest pins them too.

The robust pattern when grouping is involved: materialize the feature column first, then apply the
transform to that column. That keeps a single .over(...) grouping per expression (Polars cannot
nest .over(symbol) inside .over(timestamp)).

Run:  python examples/06_normalize.py
"""

from __future__ import annotations

import polars as pl
from _data import default_schema, make_ohlcv, make_panel

import sabia
from sabia import Adjustment, PriceField, PriceRole


def main() -> None:
    schema = default_schema()

    # --- time-series: rolling z-score of RSI on a single symbol (no grouping needed) ---
    series = make_ohlcv(n=200)
    rsi = sabia.momentum.rsi(period=14)
    z = sabia.normalize.zscore(window=63)
    ts = series.select(
        "timestamp",
        rsi.expr(schema),
        z.apply(rsi.expr(schema)).alias("rsi_z63"),
    )
    print("rolling z-score of RSI:")
    print(ts.tail(3))

    # --- cross-sectional: rank a momentum signal within each date across the universe ---
    panel = make_panel(n=300)
    mom = sabia.momentum.mom(formation=126, skip=21)
    # 1) materialize the per-symbol signal, 2) rank it within each timestamp slice.
    with_signal = panel.lazy().with_columns(mom.expr(schema)).collect()
    xr = sabia.normalize.xs_rank(over="timestamp")
    ranked = with_signal.select(
        "timestamp",
        "symbol",
        mom.spec.name,
        xr.apply(pl.col(mom.spec.name)).alias("mom_xs_rank"),
    )
    last_ts = panel.get_column("timestamp").max()
    print("\ncross-sectional rank of momentum on the final date:")
    print(ranked.filter(pl.col("timestamp") == last_ts).sort("mom_xs_rank", descending=True))

    # --- fractional differentiation: stationarity with memory, applied to a price column ---
    close_col = schema.column(PriceRole(PriceField.CLOSE, Adjustment.TR))
    ffd = sabia.normalize.frac_diff(0.4, over="symbol")  # lag within each symbol on a panel
    diffed = panel.select(
        "timestamp", "symbol", ffd.apply(pl.col(close_col)).alias("close_ffd_0p4")
    )
    print("\nfractionally differenced close (d=0.4):")
    print(diffed.filter(pl.col("symbol") == "AAA").tail(3))
    print("transform spec:", ffd.spec.name, "| fingerprint:", ffd.spec.fingerprint)


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    main()
