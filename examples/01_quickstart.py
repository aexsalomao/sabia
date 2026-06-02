"""01 — Quickstart: compute a few features on one symbol.

The whole sabia loop in three steps:
    1. describe your columns with a BarSchema (which physical column carries which *role*),
    2. pick feature factories and bind their params -> BoundFeature objects,
    3. compute() resolves the roles against the schema and materializes a DataFrame.

Run:  python examples/01_quickstart.py
"""

from __future__ import annotations

import polars as pl
from _data import default_schema, make_ohlcv

import sabia


def main() -> None:
    frame = make_ohlcv(n=400)  # a DataFrame or LazyFrame of OHLCV bars
    schema = default_schema()

    # Each factory returns a BoundFeature: an immutable .spec plus an .expr(schema) closure.
    rsi = sabia.momentum.rsi(period=14)
    vol = sabia.volatility.vol_yz(window=21)
    macd = sabia.trend.macd(fast=12, slow=26, signal=9)

    # compute() validates the input (STRICT by default), resolves roles, and returns one column per
    # feature, each named by its canonical id. Features are strictly trailing, so early rows stay
    # null until each feature's min_history is reached.
    features = sabia.compute(frame, rsi, vol, macd, schema=schema)

    print("feature columns:", features.columns)
    out = frame.select("timestamp").hstack(features)
    print(out.tail(5))

    # Same computation, kept lazy for fusion into a larger query plan (time-series features only):
    lazy = sabia.compute_lazy(frame, rsi, vol, schema=schema)
    print("\nlazy plan returns:", lazy.collect_schema().names())


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    main()
