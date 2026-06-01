"""Pinned feature-set manifest (FEATURES.md 3.4): the (name, version, fingerprint) of every shipped
feature. This is a deliberate lock on formula identity, not a snapshot of computed output -- it is
what makes the immutability guarantee enforceable. If a formula changes, bump the feature's version
and update this manifest in the same commit; a fingerprint change without a version bump fails CI.
"""

from __future__ import annotations

MANIFEST: tuple[tuple[str, int, str], ...] = (
    ("adv_21", 1, "536673e0e7ad8e9e"),
    ("adx_14", 1, "19c1b930361540eb"),
    ("amihud_21", 1, "f197aab74b80442d"),
    ("atr_14", 1, "a53934af309a1d79"),
    ("bollinger_pctb_20", 1, "7c730e3177bc7bf7"),
    ("cmf_21", 1, "a2541496703f3ee9"),
    ("day_of_week", 1, "e7de3fa1e5ba9cde"),
    ("dist_ma_50", 1, "663b2acaf5c93b78"),
    ("dollar_vol", 1, "d8ffb299b54334a4"),
    ("downside_dev_21", 1, "c61354130d0e8da2"),
    ("ema_12", 1, "d96e18cee156470a"),
    ("ema_26", 1, "4a680156e872c990"),
    ("half_life_60", 1, "7e4425b0f436e38c"),
    ("kurtosis_63", 1, "24cf221e13f8e4ac"),
    ("macd_12_26", 1, "0f55558b6d6f5fc4"),
    ("month_of_year", 1, "c2f8b68e19f2dba2"),
    ("ret_log_1", 1, "0ea60f27abb2ef94"),
    ("ret_log_21", 1, "2c8aca6c43ea1090"),
    ("ret_log_5", 1, "3c36f2209ee22389"),
    ("ret_simple", 1, "9d2fc7bd638339ec"),
    ("roc_10", 1, "0a34f79e72b45bbb"),
    ("rsi_14", 1, "132c1c45c770c38d"),
    ("signed_vol_21", 1, "819cc55bb3e9291e"),
    ("skew_63", 1, "1b67e029029295c1"),
    ("sma_200", 1, "21a8e71fdf40ddf7"),
    ("sma_50", 1, "55e9298d921550ce"),
    ("stoch_d_14_3", 1, "b541d7d1557441e9"),
    ("stoch_k_14", 1, "ec85ab697a803818"),
    ("turn_of_month", 1, "206958fb2529c926"),
    ("vol_close_21", 1, "26e14e87996bf390"),
    ("vol_close_63", 1, "04b4ed39bef9b9f9"),
    ("vol_gk_21", 1, "060fb0b5cd57bd74"),
    ("vol_parkinson_21", 1, "ab88fbd25fca0f31"),
    ("vol_rs_21", 1, "087b442814ad4551"),
    ("vol_yz_21", 1, "3356480f0f797865"),
    ("vol_zscore_21", 1, "0bd8d3892d92d97f"),
    ("williams_r_14", 1, "266904b9fbb47bdc"),
    ("xs_rank_mom_252", 1, "e86684bb31877dda"),
    ("xs_rank_vol_63", 1, "bc74178dfe6a924f"),
    ("xs_zscore_ret_21", 1, "113926a7f3124b5b"),
    ("zdist_20", 1, "2539ecdd0ccc8bc7"),
)
