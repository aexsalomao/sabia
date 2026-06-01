"""Performance benchmarks (FEATURES.md 8.8): track per-feature cost, slow-marked. No hard timing
asserts -- those flake; the benchmark records the time and we only assert correctness of the run.
"""

from __future__ import annotations

import polars as pl
import pytest
from synthetic import make_panel

from sabia.momentum import rsi
from sabia.registry import Registry, evaluate
from sabia.volatility import vol_yz

pytestmark = pytest.mark.slow

_N_BARS = 2000
_N_SYMBOLS = 50


@pytest.fixture(scope="module")
def large_panel() -> pl.DataFrame:
    symbols = tuple(f"S{i:02d}" for i in range(_N_SYMBOLS))
    return make_panel(_N_BARS, symbols=symbols, seed=0)


def test_benchmark_rsi_over_large_panel(benchmark, large_panel: pl.DataFrame) -> None:  # type: ignore[no-untyped-def]
    result = benchmark(lambda: large_panel.lazy().select(rsi(period=14)).collect())
    assert result.height == large_panel.height


def test_benchmark_vol_yz_over_large_panel(benchmark, large_panel: pl.DataFrame) -> None:  # type: ignore[no-untyped-def]
    result = benchmark(lambda: large_panel.lazy().select(vol_yz(window=21)).collect())
    assert result.height == large_panel.height


def test_benchmark_full_registry(benchmark, large_panel: pl.DataFrame) -> None:  # type: ignore[no-untyped-def]
    reg = Registry.default()

    def _evaluate_all() -> int:
        return sum(evaluate(large_panel, feature).len() for feature in reg)

    total = benchmark(_evaluate_all)
    assert total == len(reg) * large_panel.height
