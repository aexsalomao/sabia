"""Pinned feature-set manifest (FEATURES.md 3.4): the (name, version, fingerprint) of every shipped
feature. This is a deliberate lock on formula identity, not a snapshot of computed output -- it is
what makes the immutability guarantee enforceable. If a formula changes, bump the feature's version
and update this manifest in the same commit; a fingerprint change without a version bump fails CI.
"""

from __future__ import annotations

MANIFEST: tuple[tuple[str, int, str], ...] = (
    ("adv_21", 1, "6662157c77b074a6"),
    ("adx_14", 1, "3c1f877bca64e6f9"),
    ("amihud_21", 1, "ba779d726fbe9d9e"),
    ("atr_14", 1, "d4f0a532ceb77915"),
    ("bollinger_pctb_20", 1, "76dbc906bfd1547f"),
    ("cmf_21", 1, "4d819f932ec60adc"),
    ("day_of_week", 1, "975c6e5eb8478173"),
    ("dist_ma_50", 1, "f38511ac61fface2"),
    ("dollar_vol", 1, "623c62d048ca7cfe"),
    ("downside_dev_21", 1, "3746ad57f5f1cc5c"),
    ("ema_12", 1, "9bd47fd5da760542"),
    ("ema_26", 1, "423f2033d1136958"),
    ("half_life_60", 1, "2dd35c1b110574a8"),
    ("kurtosis_63", 1, "94701b2f48095d54"),
    ("macd_12_26", 1, "ee45e2b162ae44bd"),
    ("month_of_year", 1, "6ea2b621acc97cc5"),
    ("ret_log_1", 1, "fdacd1aa35b1bcee"),
    ("ret_log_21", 1, "d02ffdfa40bfd65b"),
    ("ret_log_5", 1, "4e8dddf03ed47013"),
    ("ret_simple", 1, "d2da8ba5ff1d9525"),
    ("roc_10", 1, "3ae789c57da1c97b"),
    ("rsi_14", 1, "92010f23e7a676e2"),
    ("signed_vol_21", 1, "86d73e48df05baec"),
    ("skew_63", 1, "79df23d5913de6bd"),
    ("sma_200", 1, "18046735d018cc4d"),
    ("sma_50", 1, "fcfc85b6fdac57f8"),
    ("stoch_d_14_3", 1, "889ef46259be49c2"),
    ("stoch_k_14", 1, "b67546ac86e6116a"),
    ("turn_of_month", 1, "9c90ad7db0be60a4"),
    ("vol_close_21", 1, "f66ad4581c6bfd37"),
    ("vol_close_63", 1, "fb5f68b6268ddc14"),
    ("vol_gk_21", 1, "2d95769fa6811029"),
    ("vol_parkinson_21", 1, "aee7571871f03aaa"),
    ("vol_rs_21", 1, "c408a8669cbbcaab"),
    ("vol_yz_21", 1, "91e20fafb109044c"),
    ("vol_zscore_21", 1, "701514b9163486ba"),
    ("williams_r_14", 1, "74ad3e71b5c84a99"),
    ("xs_rank_mom_252", 1, "d63cd1d9740649fb"),
    ("xs_rank_vol_63", 1, "4e944e477affe63c"),
    ("xs_zscore_ret_21", 1, "3ddce41188d77b81"),
    ("zdist_20", 1, "bbb2d92a5c63546f"),
)
