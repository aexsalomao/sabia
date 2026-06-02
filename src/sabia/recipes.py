# Recipe presets (audit follow-up): named bundles of shipped features, NOT strategies. Each returns
# a FeatureSet so a caller computes a sensible default panel in one line -- no signals, no portfolio
# opinions, no thresholds. The point is to reduce ceremony, not to encode a view of the market.

from __future__ import annotations

from sabia.cross_sectional import beta, idio_vol, rev_1m, xs_rank_mom, xs_z_mom
from sabia.distribution import downside_dev, kurt, skew
from sabia.mean_reversion import autocorr, zscore_close
from sabia.momentum import mom, roc, rsi
from sabia.returns import drawdown, ret_log
from sabia.toolkit import FeatureSet
from sabia.trend import macd_hist, sma_dist
from sabia.volatility import atr, vol_cc, vol_yz


def daily_core() -> FeatureSet:
    """A broad single-name daily panel: returns, momentum, volatility, trend, distribution."""
    return FeatureSet(
        (
            ret_log(period=1),
            ret_log(period=21),
            roc(window=21),
            rsi(period=14),
            mom(formation=252, skip=21),
            vol_cc(window=21),
            vol_yz(window=21),
            atr(window=14),
            sma_dist(window=50),
            macd_hist(),
            zscore_close(window=21),
            autocorr(lag=1, window=21),
            skew(window=21),
            kurt(window=21),
            downside_dev(window=21),
            drawdown(window=252),
        )
    )


def volatility_core() -> FeatureSet:
    """The volatility / range-estimator family at the canonical 21-bar window (plus ATR)."""
    return FeatureSet((vol_cc(window=21), vol_cc(window=63), vol_yz(window=21), atr(window=14)))


def cross_sectional_core() -> FeatureSet:
    """The cross-sectional factor panel: momentum rank / z, short-term reversal, beta, idio vol.

    The rank / z / reversal features ``require_universe`` -- pass ``universe=`` or ``membership=``
    to ``compute``; ``beta`` / ``idio_vol`` are per-symbol and instead need a ``market_ret`` column.
    """
    return FeatureSet(
        (
            xs_rank_mom(formation=252, skip=21),
            xs_z_mom(formation=252, skip=21),
            rev_1m(window=21),
            beta(window=252),
            idio_vol(window=252),
        )
    )


__all__ = ["cross_sectional_core", "daily_core", "volatility_core"]
