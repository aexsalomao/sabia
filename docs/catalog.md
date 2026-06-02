# Feature catalog

The 66 features shipped by `sabia.Registry.default()`, grouped by family. Generated from the registry — `name`, params, the roles each needs, warm-up (`min_history`), recurrence, unit, and evidence tier. See `FEATURES.md` for definitions and citations.

## cross_sectional

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `beta_252` | window=252 | close@tr, market_ret | 253 | finite | unitless | academic_replicated |
| `idio_vol_252` | window=252 | close@tr, market_ret | 253 | finite | return_std_per_bar | academic_replicated |
| `rev_1m_21` | window=21 | close@tr | 22 | finite | rank_0_1 | academic_replicated |
| `xs_rank_mom_252_21` | formation=252, skip=21 | close@tr | 253 | finite | rank_0_1 | academic_replicated |
| `xs_z_mom_252_21` | formation=252, skip=21 | close@tr | 253 | finite | zscore | academic_replicated |

## distribution

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `downside_dev_21` | window=21 | close@tr | 22 | finite | return_std_per_bar | academic_single |
| `kurt_21` | window=21 | close@tr | 22 | finite | unitless | formula_only |
| `skew_21` | window=21 | close@tr | 22 | finite | unitless | formula_only |
| `up_down_vol_ratio_21` | window=21 | close@tr | 22 | finite | ratio | formula_only |

## mean_reversion

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `autocorr_1_21` | lag=1, window=21 | close@tr | 23 | finite | unitless | formula_only |
| `half_life_60` | window=60 | close@tr | 61 | finite | unitless | academic_single |
| `var_ratio_2_21` | q=2, window=21 | close@tr | 23 | finite | unitless | academic_replicated |
| `zscore_close_21` | window=21 | close@tr | 21 | finite | zscore | formula_only |

## momentum

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `cci_20` | window=20 | close@split, high@split, low@split | 39 | finite | unitless | ta_canon |
| `mom_252_21` | formation=252, skip=21 | close@tr | 253 | finite | log_return | academic_replicated |
| `roc_10` | window=10 | close@tr | 11 | finite | ratio | ta_canon |
| `roc_21` | window=21 | close@tr | 22 | finite | ratio | ta_canon |
| `rsi_14` | period=14 | close@tr | 249 | recursive_decay | index_0_100 | ta_canon |
| `stoch_d_14_3` | smooth=3, window=14 | close@split, high@split, low@split | 16 | finite | index_0_100 | ta_canon |
| `stoch_k_14` | window=14 | close@split, high@split, low@split | 14 | finite | index_0_100 | ta_canon |
| `williams_r_14` | window=14 | close@split, high@split, low@split | 14 | finite | index_0_100 | ta_canon |

## returns

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `drawdown_252` | window=252 | close@tr | 252 | finite | ratio | formula_only |
| `ret_intraday` | — | close@tr, open@tr | 1 | finite | log_return | formula_only |
| `ret_log_1` | period=1 | close@tr | 2 | finite | log_return | formula_only |
| `ret_log_21` | period=21 | close@tr | 22 | finite | log_return | formula_only |
| `ret_log_252` | period=252 | close@tr | 253 | finite | log_return | formula_only |
| `ret_log_5` | period=5 | close@tr | 6 | finite | log_return | formula_only |
| `ret_overnight` | — | close@tr, open@tr | 2 | finite | log_return | formula_only |
| `ret_simple_1` | period=1 | close@tr | 2 | finite | ratio | formula_only |

## seasonality

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `season_dow` | — | — | 1 | finite | unitless | academic_single |
| `season_tom_3` | k=3 | — | 1 | finite | unitless | academic_single |

## trend

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `adx_14` | window=14 | close@split, high@split, low@split | 526 | recursive_decay | index_0_100 | ta_canon |
| `dist_52w_high_252` | window=252 | close@tr | 252 | finite | ratio | academic_replicated |
| `ema_12` | span=12 | close@tr | 111 | recursive_decay | price_units | ta_canon |
| `ema_26` | span=26 | close@tr | 240 | recursive_decay | price_units | ta_canon |
| `ema_dist_50` | span=50 | close@tr | 461 | recursive_decay | ratio | ta_canon |
| `macd_12_26_9` | fast=12, signal=9, slow=26 | close@tr | 240 | recursive_decay | log_return | ta_canon |
| `macd_hist_12_26_9` | fast=12, signal=9, slow=26 | close@tr | 323 | recursive_decay | log_return | ta_canon |
| `macd_signal_12_26_9` | fast=12, signal=9, slow=26 | close@tr | 323 | recursive_decay | log_return | ta_canon |
| `ols_slope_63` | window=63 | close@tr | 63 | finite | log_return | formula_only |
| `price_pctile_252` | window=252 | close@tr | 252 | finite | rank_0_1 | formula_only |
| `sma_200` | window=200 | close@tr | 200 | finite | price_units | ta_canon |
| `sma_50` | window=50 | close@tr | 50 | finite | price_units | ta_canon |
| `sma_dist_50` | window=50 | close@tr | 50 | finite | ratio | ta_canon |

## volatility

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `atr_14` | window=14 | close@split, high@split, low@split | 249 | recursive_decay | price_units | ta_canon |
| `bb_bw_20_2` | n_std=2.0, window=20 | close@split | 20 | finite | ratio | ta_canon |
| `bb_pctb_20_2` | n_std=2.0, window=20 | close@split | 20 | finite | unitless | ta_canon |
| `semivar_down_21` | window=21 | close@tr | 22 | finite | return_std_per_bar | academic_single |
| `vol_cc_21` | window=21 | close@tr | 22 | finite | return_std_per_bar | formula_only |
| `vol_cc_63` | window=63 | close@tr | 64 | finite | return_std_per_bar | formula_only |
| `vol_ewma_0p94` | lam=0.94 | close@tr | 298 | recursive_decay | return_std_per_bar | academic_single |
| `vol_gk_21` | window=21 | close@split, high@split, low@split, open@split | 21 | finite | return_std_per_bar | academic_single |
| `vol_parkinson_21` | window=21 | high@split, low@split | 21 | finite | return_std_per_bar | academic_single |
| `vol_rs_21` | window=21 | close@split, high@split, low@split, open@split | 21 | finite | return_std_per_bar | academic_single |
| `vol_yz_21` | window=21 | close@split, high@split, low@split, open@split | 22 | finite | return_std_per_bar | academic_single |

## volume

| Feature | Params | Roles | min_history | Recurrence | Unit | Evidence |
|---|---|---|---|---|---|---|
| `adv_21` | window=21 | close@split, volume@split | 21 | finite | price_units | formula_only |
| `amihud_21` | window=21 | close@tr, dollar_volume@raw | 22 | finite | ratio | academic_replicated |
| `cmf_21` | window=21 | close@split, high@split, low@split, volume@split | 21 | finite | unitless | ta_canon |
| `dollar_vol` | — | close@split, volume@split | 1 | finite | price_units | formula_only |
| `mfi_14` | window=14 | close@split, high@split, low@split, volume@split | 15 | finite | index_0_100 | ta_canon |
| `rel_volume_21` | window=21 | volume@split | 21 | finite | ratio | formula_only |
| `roll_spread_21` | window=21 | close@tr | 23 | finite | ratio | academic_single |
| `signed_vol_21` | window=21 | close@tr, volume@split | 22 | finite | unitless | formula_only |
| `spread_corwin_schultz` | — | high@split, low@split | 2 | finite | ratio | academic_single |
| `vol_z_21` | window=21 | volume@split | 21 | finite | zscore | formula_only |
| `vwap_dist_close` | — | close@split, vwap@split | 1 | finite | ratio | formula_only |
