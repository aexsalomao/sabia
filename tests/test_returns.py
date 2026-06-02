"""Reference-value and degenerate-input tests for the returns family."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from synthetic import CLOSE, SCHEMA, SYMBOL, TIMESTAMP, make_series

import sabia
from sabia.returns import ret_intraday, ret_log, ret_overnight, ret_simple
from sabia.typing import CLOSE_SPLIT, CLOSE_TR, OPEN_SPLIT, OPEN_TR

TOL = 1e-12


def _frame(closes: list[float | None]) -> pl.DataFrame:
    n = len(closes)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            TIMESTAMP: [start + timedelta(days=i) for i in range(n)],
            SYMBOL: ["AAA"] * n,
            CLOSE: closes,
        }
    )


def test_ret_simple_matches_hand_value() -> None:
    out = _frame([100.0, 110.0, 99.0]).select(ret_simple().expr(SCHEMA)).to_series()
    assert out.to_list() == pytest.approx([None, 0.10, -0.10], abs=TOL, nan_ok=True)


def test_ret_log_one_bar_matches_hand_value() -> None:
    import math

    out = _frame([100.0, 200.0, 100.0]).select(ret_log(period=1).expr(SCHEMA)).to_series()
    assert out.to_list()[1:] == pytest.approx([math.log(2), math.log(0.5)], abs=TOL)


def test_ret_log_k_spans_k_bars() -> None:
    import math

    out = _frame([10.0, 20.0, 40.0, 80.0]).select(ret_log(period=2).expr(SCHEMA)).to_series()
    # At index 2: ln(40/10) = ln(4); index 3: ln(80/20) = ln(4).
    assert out.to_list()[2:] == pytest.approx([math.log(4), math.log(4)], abs=TOL)


def test_ret_simple_period_spans_k_bars() -> None:
    out = _frame([100.0, 50.0, 200.0]).select(ret_simple(period=2).expr(SCHEMA)).to_series()
    # At index 2: 200/100 - 1 = 1.0; the first two bars have no 2-bar base -> null.
    assert out.to_list() == pytest.approx([None, None, 1.0], abs=TOL, nan_ok=True)
    spec = ret_simple(period=2).spec
    assert (spec.lookback, spec.min_history, spec.effective_warmup) == (2, 3, 3)


@pytest.mark.parametrize(
    ("factory", "open_role", "close_role", "expected"),
    [
        ("overnight_tr", OPEN_TR, CLOSE_TR, "ret_overnight"),
        ("overnight_split", OPEN_SPLIT, CLOSE_SPLIT, "ret_overnight_split"),
        ("overnight_mixed", OPEN_SPLIT, CLOSE_TR, "ret_overnight_split_tr"),
        ("intraday_tr", OPEN_TR, CLOSE_TR, "ret_intraday"),
        ("intraday_split", OPEN_SPLIT, CLOSE_SPLIT, "ret_intraday_split"),
        ("intraday_mixed", OPEN_SPLIT, CLOSE_TR, "ret_intraday_split_tr"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_basis_token_encodes_non_tr_roles_in_name(
    factory: str, open_role: object, close_role: object, expected: str
) -> None:
    fn = ret_overnight if factory.startswith("overnight") else ret_intraday
    assert fn(open_=open_role, close=close_role).spec.name == expected  # type: ignore[arg-type]


def test_rebound_roles_do_not_collide_in_one_compute() -> None:
    # Two rebindings of the same decomposition get distinct names (the _basis_token contract), so a
    # single compute() does not trip the duplicate-name guard.
    df = sabia.compute(
        make_series(5),
        ret_intraday(open_=OPEN_TR, close=CLOSE_TR),
        ret_intraday(open_=OPEN_SPLIT, close=CLOSE_SPLIT),
        schema=SCHEMA,
    )
    assert df.columns == ["ret_intraday", "ret_intraday_split"]


def test_ret_simple_zero_base_is_null() -> None:
    out = _frame([0.0, 100.0]).select(ret_simple().expr(SCHEMA)).to_series()
    assert out[1] is None


def test_ret_log_nonpositive_is_null_not_nan() -> None:
    out = _frame([100.0, -50.0, 50.0]).select(ret_log(period=1).expr(SCHEMA)).to_series()
    assert out[1] is None  # ratio negative -> null, not NaN


def test_ret_log_leading_value_is_null() -> None:
    out = _frame([100.0, 110.0]).select(ret_log(period=1).expr(SCHEMA)).to_series()
    assert out[0] is None
