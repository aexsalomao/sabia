"""
sabia -- Polars-native technical features for trading pipelines.

Pure functions over OHLCV bars that return strictly-trailing, point-in-time-correct ``pl.Expr``
features. Batch-first, online-ready. See ``FEATURES.md`` for the full specification.

    import polars as pl
    import sabia

    sabia.validate(frame, required=[sabia.Column.CLOSE])
    df = sabia.compute(frame, sabia.momentum.rsi(period=14), sabia.volatility.vol_yz(window=21))

    reg = sabia.Registry.default()
    reg.where(lambda s: sabia.Horizon.MEDIUM in s.native_band)
    reg.available(sabia.DataTier.DAILY)

Time-series features compose as ``pl.Expr`` via ``compute``. Cross-sectional features are two-pass;
evaluate them through ``sabia.evaluate(frame, feature)`` or the registry.
"""

from __future__ import annotations

import polars as pl

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
from sabia.registry import RegisteredFeature, Registry, evaluate, make_feature
from sabia.spec import (
    Column,
    Cost,
    DataTier,
    Family,
    FeatureSpec,
    Horizon,
    Recurrence,
)
from sabia.validate import SabiaValidationError, validate

__version__ = "0.1.0"


def compute(frame: pl.DataFrame | pl.LazyFrame, *exprs: pl.Expr) -> pl.DataFrame:
    """Materialize feature expressions into a DataFrame -- a ``select`` over the same expressions.

    For time-series and normalization features (``pl.Expr``). Cross-sectional features are two-pass;
    use ``evaluate`` (or the registry) for those.
    """
    return frame.lazy().select(*exprs).collect()


__all__ = [
    "Column",
    "Cost",
    "DataTier",
    "Family",
    "FeatureSpec",
    "Horizon",
    "RegisteredFeature",
    "Registry",
    "Recurrence",
    "SabiaValidationError",
    "__version__",
    "compute",
    "cross_sectional",
    "distribution",
    "evaluate",
    "make_feature",
    "mean_reversion",
    "momentum",
    "normalize",
    "returns",
    "seasonality",
    "trend",
    "validate",
    "volatility",
    "volume",
]
