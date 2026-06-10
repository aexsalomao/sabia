"""08 — Intraday microstructure: raw ticks -> bars -> features.

The full edge->core path for an intraday strategy:
    1. aggregate raw trade ticks into information-driven bars with sabia.adapters.build_bars
       (here: volume bars -- equal-volume buckets, the right clock for flow/toxicity signals),
    2. describe those bars with BarSchema.trades() (which adapter column carries which role),
    3. compute() the microstructure family -- order-flow imbalance, VPIN toxicity, realized vol.

The adapter signs each trade (the tick rule) at aggregation time, so the bars carry signed_volume /
buy_volume / sell_volume and the features never re-derive a trade direction. Everything is pure and
point-in-time: each bar is stamped with its last tick time, so trailing features stay causal.

Run:  python examples/08_intraday.py
"""

from __future__ import annotations

import polars as pl
from _data import make_trades

import sabia
from sabia.adapters import BarKind, BarSpec, SignRule, build_bars


def main() -> None:
    ticks = make_trades(n=5_000)  # raw trade ticks: timestamp, symbol, price, size
    print(f"raw ticks: {ticks.height}  ({ticks.columns})")

    # 1. Aggregate into volume bars of 10k shares each, signing trades with the tick rule. validate
    #    runs the raw-tick contract first (monotonic timestamps, price > 0, size >= 0).
    spec = BarSpec(kind=BarKind.VOLUME, threshold=10_000.0, sign_rule=SignRule.TICK_RULE)
    bars = build_bars(ticks, spec, validate=sabia.ValidationMode.STRICT).collect()
    # The symbol's final bar is still in progress (closed == False): a later tick would revise it,
    # so it is not point-in-time safe. Keep only finalized bars -- the validate() gate below
    # (BarSchema.trades() maps the closed marker by default) rejects open bars otherwise.
    bars = bars.filter(pl.col("closed"))
    print(f"\nvolume bars (closed): {bars.height}")
    print(bars.tail(3))

    # 2. The adapter's output resolves against BarSchema.trades() -- OHLCV on the raw basis plus the
    #    signed-flow aggregates the features need.
    schema = sabia.BarSchema.trades(
        signed_volume="signed_volume", volume="volume", vwap="vwap", trade_count="trade_count"
    )

    # 3. Compute the microstructure family on the bars (MINUTE-tier features).
    features = sabia.compute(
        bars,
        sabia.microstructure.trade_imbalance(window=12),
        sabia.microstructure.vpin(n_buckets=20),
        sabia.microstructure.rvar(window=20),
        sabia.microstructure.signed_jump(window=20),
        schema=schema,
        include_keys=True,
    )
    print("\nintraday features:", [c for c in features.columns if c not in ("symbol", "timestamp")])
    print(features.tail(5))


if __name__ == "__main__":
    pl.Config.set_tbl_cols(-1)
    main()
