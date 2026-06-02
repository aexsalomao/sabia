# sabia notebooks — data-viz gallery

Plotly notebooks that *show* what sabia's features measure. Each one is built around a **well-known
market concept** and uses a sabia feature to reveal it on a chart — not just a table.

```bash
# from this folder, with the project installed (e.g. `uv pip install -e ..` or your venv)
pip install plotly nbformat jupyterlab
jupyter lab            # then open any notebook
```

The committed `.ipynb` files include rendered figures, so they display in JupyterLab, VS Code, and
[nbviewer](https://nbviewer.org/) without re-running. (GitHub's own viewer does not render Plotly —
open them in one of those, or re-run locally.) Plotly, Jupyter, and nbformat are **not** sabia
dependencies; install them yourself as above.

| Notebook | Concept / hypothesis | Key sabia features |
|---|---|---|
| `01_price_trend_oscillators.ipynb` | The anatomy of a technical chart: trend distance, overbought/oversold, crossovers | `sma_dist`, `ema_dist`, `rsi`, `macd`/`macd_signal`/`macd_hist`, `stoch_k`/`stoch_d`, `cci` |
| `02_volatility_clustering_and_estimators.ipynb` | Volatility clusters; range/OHLC estimators are more efficient; the leverage effect | `vol_cc`, `vol_parkinson`, `vol_gk`, `vol_rs`, `vol_yz`, `vol_ewma`, `atr`, `up_down_vol_ratio` |
| `03_distribution_fat_tails.ipynb` | Returns are leptokurtic — fatter tails than a normal allows | `kurt`, `skew`, `downside_dev` |
| `04_momentum_meanreversion_crosssection.ipynb` | Variance ratios (trend vs mean-reversion); cross-sectional momentum; market-model risk | `var_ratio`, `xs_rank_mom`, `beta`, `idio_vol` |
| `05_normalization_and_stationarity.ipynb` | Fractional differentiation: stationarity *with* memory; trailing & cross-sectional standardizers | `normalize.frac_diff`, `normalize.zscore`, `normalize.xs_rank` |

## A note on the data

Notebook 01 uses the neutral random walk from `examples/_data.py` — there is no edge, the point is to
read the indicators. The other notebooks **simulate data that bakes in the stylized fact on purpose**
(`_simulate.py`): a GJR-GARCH process so volatility clusters, Student-t innovations so tails are fat,
an AR(1)+drift panel so momentum and mean-reversion are real. This is a teaching device — it lets a
feature *demonstrably* recover the effect. In a real pipeline the frame comes from your data layer
(e.g. `marketgoblin`); sabia neither fetches nor adjusts anything.

Helper modules (not notebooks):

- `_simulate.py` — the stylized-fact price simulators (all emit contract-valid OHLCV) and a re-export
  of `default_schema`.
- `_stats.py` — small numpy helpers used in the charts (`acf`, `normal_pdf`, `adf_tstat`, `fit_line`).

For the API itself — `BarSchema`, `compute`, the registry, manifests — see the runnable scripts in
[`../`](../README.md).
