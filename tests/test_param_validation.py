"""Factory parameter guards (sabia._validate_params): out-of-domain params raise at BIND time.

Parametrized, never looped (CLAUDE.md): a failure names the exact bad-param case. Each bad case
matches the GUARD's message so it fails for the right reason (not, say, an incidental naming error).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sabia import (
    cross_sectional,
    distribution,
    mean_reversion,
    momentum,
    normalize,
    returns,
    seasonality,
    trend,
    volatility,
    volume,
)
from sabia.registry import BoundFeature

# (id, factory call, expected-message regex). The regex pins which guard fired.
_BAD: list[tuple[str, Callable[[], object], str]] = [
    ("rsi_period_below_2", lambda: momentum.rsi(period=1), "must be an int >="),
    ("rsi_period_zero", lambda: momentum.rsi(period=0), "must be an int >="),
    ("rsi_period_float", lambda: momentum.rsi(period=14.0), "must be an int >="),  # type: ignore[arg-type]
    ("roc_window_zero", lambda: momentum.roc(window=0), "must be a positive int"),
    ("mom_skip_equals_formation", lambda: momentum.mom(formation=21, skip=21), "must be less than"),
    (
        "mom_skip_exceeds_formation",
        lambda: momentum.mom(formation=21, skip=30),
        "must be less than",
    ),
    ("mom_skip_negative", lambda: momentum.mom(formation=21, skip=-1), "non-negative int"),
    ("cci_window_below_2", lambda: momentum.cci(window=1), "must be an int >="),
    ("williams_r_window_one", lambda: momentum.williams_r(window=1), "must be an int >="),
    ("stoch_k_window_one", lambda: momentum.stoch_k(window=1), "must be an int >="),
    ("stoch_d_window_one", lambda: momentum.stoch_d(window=1), "must be an int >="),
    (
        "stoch_d_smooth_zero",
        lambda: momentum.stoch_d(window=14, smooth=0),
        "must be a positive int",
    ),
    ("vol_ewma_lambda_one", lambda: volatility.vol_ewma(lam=1.0), "must be in"),
    ("vol_ewma_lambda_zero", lambda: volatility.vol_ewma(lam=0.0), "must be in"),
    ("vol_ewma_lambda_nan", lambda: volatility.vol_ewma(lam=float("nan")), "must be in"),
    ("vol_yz_window_one", lambda: volatility.vol_yz(window=1), "must be an int >="),
    ("vol_cc_window_one", lambda: volatility.vol_cc(window=1), "must be an int >="),
    ("bb_pctb_nstd_zero", lambda: volatility.bb_pctb(n_std=0.0), "finite positive number"),
    ("bb_pctb_nstd_nan", lambda: volatility.bb_pctb(n_std=float("nan")), "finite positive number"),
    ("atr_window_one", lambda: volatility.atr(window=1), "must be an int >="),
    ("ret_log_period_zero", lambda: returns.ret_log(period=0), "must be a positive int"),
    ("ret_simple_period_zero", lambda: returns.ret_simple(period=0), "must be a positive int"),
    ("drawdown_window_one", lambda: returns.drawdown(window=1), "must be an int >="),
    ("macd_fast_ge_slow", lambda: trend.macd(fast=26, slow=12), "must be less than"),
    ("macd_signal_below_2", lambda: trend.macd(signal=1), "must be an int >="),
    ("ema_span_below_2", lambda: trend.ema(span=1), "must be an int >="),
    ("ols_slope_window_one", lambda: trend.ols_slope(window=1), "must be an int >="),
    ("var_ratio_q_below_2", lambda: mean_reversion.var_ratio(q=1), "must be an int >="),
    ("autocorr_lag_zero", lambda: mean_reversion.autocorr(lag=0), "must be a positive int"),
    ("skew_window_one", lambda: distribution.skew(window=1), "must be an int >="),
    ("vol_z_window_one", lambda: volume.vol_z(window=1), "must be an int >="),
    ("season_tom_k_zero", lambda: seasonality.season_tom(k=0), "must be a positive int"),
    (
        "xs_rank_mom_skip_eq_formation",
        lambda: cross_sectional.xs_rank_mom(formation=5, skip=5),
        "less than",
    ),
    ("beta_window_one", lambda: cross_sectional.beta(window=1), "must be an int >="),
    (
        "frac_diff_threshold_zero",
        lambda: normalize.frac_diff(0.5, threshold=0.0),
        "finite positive",
    ),
    (
        "frac_diff_threshold_nan",
        lambda: normalize.frac_diff(0.5, threshold=float("nan")),
        "finite positive",
    ),
    (
        "frac_diff_max_lag_zero",
        lambda: normalize.frac_diff(0.5, max_lag=0),
        "must be a positive int",
    ),
    ("zscore_window_one", lambda: normalize.zscore(1), "must be an int >="),
]


@pytest.mark.parametrize(
    ("call", "match"), [(c, m) for _, c, m in _BAD], ids=[i for i, _, _ in _BAD]
)
def test_factory_rejects_bad_params(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


# A representative valid call from each guarded family binds AND yields its expected name -- so a
# guard that perturbed naming/binding (it must be fingerprint-neutral) would be caught here too.
_GOOD: list[tuple[str, Callable[[], BoundFeature]]] = [
    ("rsi_14", lambda: momentum.rsi(period=14)),
    ("vol_ewma_0p97", lambda: volatility.vol_ewma(lam=0.97)),
    ("vol_ewma_0p9", lambda: volatility.vol_ewma(lam=0.9)),
    ("mom_120_5", lambda: momentum.mom(formation=120, skip=5)),
    ("ret_simple_1", lambda: returns.ret_simple(period=1)),
    ("ret_simple_2", lambda: returns.ret_simple(period=2)),
    ("macd_12_26_9", lambda: trend.macd(fast=12, slow=26, signal=9)),
    ("xs_rank_mom_252_21", lambda: cross_sectional.xs_rank_mom(formation=252, skip=21)),
]


@pytest.mark.parametrize(("name", "call"), _GOOD, ids=[i for i, _ in _GOOD])
def test_factory_accepts_valid_params_and_names(
    name: str, call: Callable[[], BoundFeature]
) -> None:
    feature = call()
    assert isinstance(feature, BoundFeature)
    assert feature.spec.name == name


def test_vol_ewma_small_lambda_yields_valid_name() -> None:
    # A scientific-notation-prone lambda must still yield a snake_case name (not 'vol_ewma_1e-05').
    assert volatility.vol_ewma(lam=1e-05).spec.name == "vol_ewma_0p00001"
