# Trade signing for the tick->bar adapter (FEATURES.md 13). Lives at the EDGE: it consumes raw,
# un-canonical ticks and tags each trade with a buy(+1)/sell(-1) direction so the bar builder can
# sum signed flow. Pure Polars, strictly causal -- every rule looks only at the current trade and
# the ones before it, so a feature reading the resulting aggregate stays point-in-time correct.

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from sabia.adapters.bars import BarSpec

# Internal per-tick sign column (dunder-prefixed so it never collides with a caller column). The
# bar builder consumes it and it never reaches the canonical output.
SIGN_COLUMN = "__sabia_sign__"
_RAW_SIGN = "__sabia_raw_sign__"


def _raw_tick_sign(price: str, symbol: str) -> pl.Expr:
    # Lee-Ready (1991) tick rule, before zero-tick fill: +1 on an uptick, -1 on a downtick, null on
    # a zero-tick or the first trade. diff() over symbol looks only backward, so this is causal.
    diff = pl.col(price).diff()
    return pl.when(diff > 0).then(1).when(diff < 0).then(-1).otherwise(None).over(symbol)


def sign_ticks(ticks: pl.LazyFrame, spec: BarSpec) -> pl.LazyFrame:
    """Tag each tick with a buy(+1)/sell(-1)/unknown(0) direction per ``spec.sign_rule``.

    NONE -> every tick 0 (no signing; signed aggregates are all zero). TICK_RULE -> the Lee-Ready
    tick rule (zero-ticks carry the prior sign). LEE_READY -> trade vs quote midpoint, with the tick
    rule as the at-the-mid fallback (needs bid/ask columns; tier L1). The sign column is internal
    (``SIGN_COLUMN``); the bar builder sums it into signed_volume / buy_volume / sell_volume.

    LEE_READY input contract (Lee & Ready 1991): the bid/ask on each trade row must be the quote
    **prevailing** when the trade printed -- the pre-trade L1, as-of-joined backward from the quote
    stream. The rule classifies against that midpoint on the same row; it cannot detect a feed that
    stamps the post-trade (already-updated) quote, which inverts the classification of every
    quote-moving trade. If your tape carries post-trade quotes, lag them before aggregation.
    """
    from sabia.adapters.bars import SignRule  # local import: break the bars<->signing cycle

    symbol, price = spec.symbol_col, spec.price_col

    if spec.sign_rule is SignRule.NONE:
        return ticks.with_columns(pl.lit(0, dtype=pl.Int8).alias(SIGN_COLUMN))

    # Materialize the raw tick-rule sign, then forward-fill the zero-ticks' nulls within each symbol
    # (a separate step avoids nesting .over inside .over).
    raw = ticks.with_columns(_raw_tick_sign(price, symbol).alias(_RAW_SIGN))
    tick_sign = pl.col(_RAW_SIGN).forward_fill().over(symbol).fill_null(0).cast(pl.Int8)

    if spec.sign_rule is SignRule.TICK_RULE:
        return raw.with_columns(tick_sign.alias(SIGN_COLUMN)).drop(_RAW_SIGN)

    # LEE_READY: classify against the prevailing midpoint; fall back to the tick rule at the mid.
    mid = (pl.col(spec.bid_col) + pl.col(spec.ask_col)) / 2.0
    lee_ready = (
        pl.when(pl.col(price) > mid)
        .then(1)
        .when(pl.col(price) < mid)
        .then(-1)
        .otherwise(tick_sign)
        .cast(pl.Int8)
    )
    return raw.with_columns(lee_ready.alias(SIGN_COLUMN)).drop(_RAW_SIGN)


__all__ = ["SIGN_COLUMN", "sign_ticks"]
