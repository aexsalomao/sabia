# Mean-reversion (memory) family: how far price sits from its recent center, how persistent its
# returns are, and how fast it pulls back. All FINITE, strictly trailing, panel-safe via
# .over(symbol). Regression-style measures (autocorrelation, variance ratio, OU half-life) are built
# entirely from rolling moments -- no per-row Python (FEATURES.md 10). Close-based, close@tr.

from __future__ import annotations

from math import log

import polars as pl

from sabia._expr import emit_after, grouped
from sabia._math import log_return, safe_div, safe_sqrt
from sabia._validate_params import int_at_least, positive_int
from sabia.naming import naming
from sabia.params import FrozenParams
from sabia.references import Citation, Reference
from sabia.registry import BoundFeature, bind_feature
from sabia.schema import BarSchema
from sabia.spec import Cost, Evidence, Family, Horizon, Recurrence, Unit
from sabia.typing import CLOSE_TR, PriceRole

_LN2 = log(2.0)
_BANDS = (Horizon.SHORT, Horizon.MEDIUM)
_CLM = Reference("Campbell, Lo & MacKinlay", 1997)


def _log_return(s: BarSchema, close: PriceRole) -> pl.Expr:
    c = pl.col(s.column(close))
    return log_return(c, c.shift(1))


def zscore_close(*, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Rolling z-distance of close from its own ``window``-bar mean: the reversion signal. FINITE.

    ``(close - mean) / std`` over the window; a flat window (zero std) yields null, never inf.
    Citation: Campbell, Lo & MacKinlay (1997).
    """
    int_at_least("window", window, 2)
    name = naming("zscore_close", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        mean = c.rolling_mean(window, min_samples=window)
        std = c.rolling_std(window, min_samples=window)
        value = pl.when(std == 0).then(None).otherwise((c - mean) / std)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MEAN_REVERSION,
        native_band=_BANDS,
        lookback=window,
        min_history=window,
        recurrence=Recurrence.FINITE,
        effective_warmup=window,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=Unit.ZSCORE,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=_CLM),
        params=FrozenParams(window=window),
    )


def autocorr(*, lag: int = 1, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Rolling autocorrelation of log returns at ``lag`` over ``window`` bars. FINITE, UNITLESS.

    Pearson correlation of the return with its ``lag``-bar-lagged self, from rolling moments. A flat
    window (zero variance) yields null. Citation: Campbell, Lo & MacKinlay (1997).
    """
    positive_int("lag", lag)
    int_at_least("window", window, 2)
    name = naming("autocorr", lag, window)

    def build(s: BarSchema) -> pl.Expr:
        r = _log_return(s, close)
        r_lag = r.shift(lag)
        value = _rolling_corr(r, r_lag, window)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MEAN_REVERSION,
        native_band=_BANDS,
        lookback=window,
        min_history=window + lag + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + lag + 1,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=Unit.UNITLESS,
        evidence=Evidence.FORMULA_ONLY,
        citation=Citation(formula=_CLM),
        params=FrozenParams(lag=lag, window=window),
    )


def var_ratio(*, q: int = 2, window: int = 21, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Lo-MacKinlay (1988) bias-corrected variance ratio ``VR(q)`` over ``window`` bars. HEAVY.

    The plain ratio ``Var(rq) / (q * Var(r1))`` is biased -- worst exactly at a small window and
    small ``q``. This is the Lo & MacKinlay (1988) unbiased estimator with the overlapping-sample
    bias correction: the 1-bar variance uses the unbiased divisor ``T-1`` and the q-period variance
    uses the overlap-corrected divisor ``m = q * (T-q+1) * (1 - q/T)`` (``T`` = number of one-bar
    returns in the window). VR is 1 under a random walk, <1 under mean reversion, >1 under momentum.
    Zero one-bar variance (a flat window) yields null. FINITE, UNITLESS. Citation: Lo & MacKinlay
    (1988).
    """
    int_at_least("q", q, 2)  # VR(q) aggregates q-bar returns; q >= 2 by construction
    int_at_least("window", window, 2)
    name = naming("var_ratio", q, window)
    t = float(window)  # T = number of one-bar returns in the rolling window
    nq = window - q + 1  # number of overlapping q-period returns fully inside the same window
    m = q * (t - q + 1) * (1.0 - q / t)  # Lo-MacKinlay overlap-bias correction factor

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        r1 = log_return(c, c.shift(1))
        rq = log_return(c, c.shift(q))
        # Closed-form rolling moments. Each window carries its OWN mean via rolling sums (a separate
        # rolling_mean per row would bleed the leading null of r1 into adjacent windows). var_a uses
        # all T one-bar returns; var_c uses the T-q+1 overlapping q-period returns ending at the
        # same bar (Lo-MacKinlay's unbiased sigma_a^2 / sigma_c^2).
        s1 = r1.rolling_sum(window, min_samples=window)
        s1_sq = (r1 * r1).rolling_sum(window, min_samples=window)
        sq = rq.rolling_sum(nq, min_samples=nq)
        sq_sq = (rq * rq).rolling_sum(nq, min_samples=nq)
        mean1 = s1 / t  # window mean of the one-bar returns
        # Unbiased 1-bar variance: sum((r1 - mean1)^2) / (T - 1).
        var_a = (s1_sq - s1 * mean1) / (t - 1.0)
        # Overlap-corrected q-period variance: sum((rq - q*mean1)^2) / m, where the q-bar return has
        # mean q*mean1 (it is the sum of q consecutive one-bar returns). Expand the square over the
        # T-q+1 overlapping returns: sum(rq^2) - 2*q*mean1*sum(rq) + (T-q+1)*(q*mean1)^2. m already
        # carries the q normalization, so VR = var_c / var_a (not var_c / (q*var_a)).
        qm = q * mean1
        var_c = (sq_sq - 2.0 * qm * sq + nq * qm * qm) / m
        value = safe_div(var_c, var_a)
        # var_c's T-q+1-wide window fills q-1 bars sooner than the full T-bar one-bar window, so
        # gate to the spec's emit-and-buffer threshold (min_history = window + q): null until then.
        gated = emit_after(value, window + q, s.symbol_col)
        return grouped(gated, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MEAN_REVERSION,
        native_band=_BANDS,
        lookback=window,
        min_history=window + q,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + q,
        cost_class=Cost.HEAVY,
        input_roles=(close,),
        output_unit=Unit.UNITLESS,
        evidence=Evidence.ACADEMIC_REPLICATED,
        citation=Citation(formula=Reference("Lo & MacKinlay", 1988)),
        params=FrozenParams(q=q, window=window),
    )


# --- extras beyond §12 (no §12 equivalent): OU mean-reversion half-life --------------------------


def half_life(*, window: int = 60, close: PriceRole = CLOSE_TR) -> BoundFeature:
    """Ornstein-Uhlenbeck mean-reversion half-life (in bars), from a rolling OLS slope. FINITE.

    Regresses the one-bar change on the prior level over ``window`` bars; the slope ``beta`` gives a
    half-life ``-ln(2) / ln(1 + beta)`` only when ``-1 < beta < 0`` (genuinely mean-reverting),
    otherwise null. Built from rolling moments, so it stays a vectorized expression. (Extra.)
    """
    int_at_least("window", window, 2)
    name = naming("half_life", window)

    def build(s: BarSchema) -> pl.Expr:
        c = pl.col(s.column(close))
        level = c.shift(1)
        change = c.diff()
        beta = _rolling_slope(level, change, window)
        reverting = (beta > -1) & (beta < 0)
        value = pl.when(reverting).then(-_LN2 / (1 + beta).log()).otherwise(None)
        return grouped(value, s.symbol_col).alias(name)

    return bind_feature(
        build,
        name=name,
        family=Family.MEAN_REVERSION,
        native_band=(Horizon.MEDIUM,),
        lookback=window,
        min_history=window + 1,
        recurrence=Recurrence.FINITE,
        effective_warmup=window + 1,
        cost_class=Cost.LINEAR,
        input_roles=(close,),
        output_unit=Unit.UNITLESS,
        evidence=Evidence.ACADEMIC_SINGLE,
        citation=Citation(formula=Reference("Ornstein & Uhlenbeck", 1930)),
        params=FrozenParams(window=window),
    )


def _rolling_slope(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    # OLS slope of y on x over the window, from population moments: cov(x, y) / var(x).
    mean_x = x.rolling_mean(window, min_samples=window)
    mean_y = y.rolling_mean(window, min_samples=window)
    mean_xy = (x * y).rolling_mean(window, min_samples=window)
    mean_xx = (x * x).rolling_mean(window, min_samples=window)
    return safe_div(mean_xy - mean_x * mean_y, mean_xx - mean_x * mean_x)


def _rolling_corr(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    # Pearson correlation over the window, from population moments: cov / (std_x * std_y).
    mean_x = x.rolling_mean(window, min_samples=window)
    mean_y = y.rolling_mean(window, min_samples=window)
    cov = (x * y).rolling_mean(window, min_samples=window) - mean_x * mean_y
    var_x = (x * x).rolling_mean(window, min_samples=window) - mean_x * mean_x
    var_y = (y * y).rolling_mean(window, min_samples=window) - mean_y * mean_y
    return safe_div(cov, safe_sqrt(var_x * var_y))


FEATURES: tuple[BoundFeature, ...] = (
    zscore_close(window=21),
    autocorr(lag=1, window=21),
    var_ratio(q=2, window=21),
    half_life(window=60),
)


__all__ = [
    "FEATURES",
    "autocorr",
    "half_life",
    "var_ratio",
    "zscore_close",
]
