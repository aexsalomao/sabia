"""Regenerate the committed marketgoblin real-data fixture (manual, not run by pytest).

The sabia test suite must run offline with no marketgoblin dependency, so we snapshot a real
Yahoo panel once and commit it as ``marketgoblin_panel.parquet``. The regression suite
(``test_marketgoblin_regression.py``) loads that parquet directly with Polars.

Run from marketgoblin's venv with sabia on the path::

    cd C:/Users/axsal/dev/marketgoblin
    PYTHONPATH=C:/Users/axsal/dev/sabia/src .venv/Scripts/python.exe \
        C:/Users/axsal/dev/sabia/tests/data/generate_marketgoblin_fixture.py

The output frame is already sabia-ready: timestamp (Datetime[us, UTC]), symbol, open/high/low/close
/volume/vwap/dollar_volume (the adjusted Yahoo slice), restricted to timestamps present for every
symbol so the cross-section is complete.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
from marketgoblin import MarketGoblin

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
START, END = "2021-01-01", "2024-03-31"
OUT = Path(__file__).with_name("marketgoblin_panel.parquet")


def main() -> None:
    goblin = MarketGoblin(provider="yahoo", save_path=tempfile.mkdtemp())
    results = goblin.fetch_many(SYMBOLS, START, END, parse_dates=True)
    frames = [lf.collect().filter(pl.col("is_adjusted")).sort("date") for lf in results.values()]
    panel = (
        pl.concat(frames)
        .with_columns(
            pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("timestamp"),
            *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")),
        )
        .with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("vwap"),
            (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
        )
    )
    n = len(SYMBOLS)
    complete = (
        panel.group_by("timestamp")
        .agg(pl.col("symbol").n_unique().alias("k"))
        .filter(pl.col("k") == n)
        .select("timestamp")
    )
    panel = (
        panel.join(complete, on="timestamp")
        .select(
            "timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "dollar_volume"
        )
        .sort("symbol", "timestamp")
    )
    panel.write_parquet(OUT)
    print(
        f"wrote {OUT} : {panel.height} rows, {panel['symbol'].n_unique()} symbols, "
        f"{panel['timestamp'].dt.date().min()} -> {panel['timestamp'].dt.date().max()}"
    )


if __name__ == "__main__":
    main()
