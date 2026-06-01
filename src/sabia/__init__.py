"""
sabia — Polars-native technical features for trading pipelines.

Pure functions over OHLCV bars that return strictly-trailing, point-in-time-correct ``pl.Expr``
features. Batch-first, online-ready. See ``FEATURES.md`` for the full specification.

The public API is assembled as the families land; until then this module exposes only the version.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
