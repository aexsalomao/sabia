"""Reference-value and degenerate-input tests for the returns family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sabia.returns import ret_log, ret_simple
from sabia.spec import Column

TOL = 1e-12


def _frame(closes: list[float | None]) -> pl.DataFrame:
    n = len(closes)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            Column.TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            Column.SYMBOL: ["AAA"] * n,
            Column.CLOSE: closes,
        }
    )


def test_ret_simple_matches_hand_value() -> None:
    out = _frame([100.0, 110.0, 99.0]).select(ret_simple()).to_series()
    assert out.to_list() == pytest.approx([None, 0.10, -0.10], abs=TOL, nan_ok=True)


def test_ret_log_one_bar_matches_hand_value() -> None:
    import math

    out = _frame([100.0, 200.0, 100.0]).select(ret_log(period=1)).to_series()
    assert out.to_list()[1:] == pytest.approx([math.log(2), math.log(0.5)], abs=TOL)


def test_ret_log_k_spans_k_bars() -> None:
    import math

    out = _frame([10.0, 20.0, 40.0, 80.0]).select(ret_log(period=2)).to_series()
    # At index 2: ln(40/10) = ln(4); index 3: ln(80/20) = ln(4).
    assert out.to_list()[2:] == pytest.approx([math.log(4), math.log(4)], abs=TOL)


def test_ret_simple_zero_base_is_null() -> None:
    out = _frame([0.0, 100.0]).select(ret_simple()).to_series()
    assert out[1] is None


def test_ret_log_nonpositive_is_null_not_nan() -> None:
    out = _frame([100.0, -50.0, 50.0]).select(ret_log(period=1)).to_series()
    assert out[1] is None  # ratio negative -> null, not NaN


def test_ret_log_leading_value_is_null() -> None:
    out = _frame([100.0, 110.0]).select(ret_log(period=1)).to_series()
    assert out[0] is None
