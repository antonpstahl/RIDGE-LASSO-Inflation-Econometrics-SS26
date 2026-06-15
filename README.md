# LASSO & Ridge Regression for Inflation Forecasting

**Seminar paper · Current Topics in Econometrics**
Technische Universität Dresden · Supervisor: Prof. Bernhard Schipp

Forecasting the German HICP inflation rate from macroeconomic indicators using
**regularisation (Ridge, LASSO, Elastic Net)** — benchmarked against **naive baselines
(Random Walk, AR)**.

**Research question:** Do macroeconomic predictors with Ridge/LASSO beat pure inflation
persistence (Random Walk)?
**Key finding:** Regularisation fixes the severe OLS overfitting (test R² −0.40 → 0.77, Adaptive LASSO),
**but does not beat the Random Walk** — the marginal macroeconomic contribution beyond
persistence is near zero. The analysis demonstrates regularisation and variable selection
with many highly collinear predictors *and* honestly benchmarks their forecast value
against the naive baseline.

---

## Project structure

```
RIDGE-LASSO-Inflation-Econometrics-SS26/
├── README.md                    This file (English)
├── README_DE.md                 German version
├── requirements.txt             Pinned dependencies
├── docs/
│   └── Vorgehensplan_Seminararbeit_Oekonometrie.pdf  Original project plan (German)
├── notebooks/
│   └── LASSO_Ridge_Inflationsprognose.ipynb   Main analysis notebook (with outputs)
├── src/                         Python package — reusable pipeline modules
│   ├── __init__.py
│   ├── config.py                Paths, seeds, hyperparameter grids, CV objects
│   ├── data_preparation.py      Data download (ECB/Eurostat API + CSV cache)
│   ├── data_preprocessing.py    YoY transformation, lag feature engineering
│   ├── evaluation.py            OOS evaluation, rolling-origin, DM test, horizon analysis
│   ├── models.py                Model definitions (OLS, Ridge, LASSO, Elastic Net, AR)
│   ├── pipeline.py              End-to-end orchestration via run_all()
│   ├── reporting.py             Figure generation (fig_01–fig_13) + table export
│   └── training.py              Model fitting and cross-validation
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_evaluation.py
│   └── test_models.py
├── data/
│   ├── raw/data_raw.csv         Raw data (index/rate values, cached)
│   └── processed/data_yoy.csv   YoY-transformed data
└── results/
    ├── results_table.csv/.tex   Model comparison (MSE/RMSE/R², incl. benchmarks)
    ├── horizons_table.csv/.tex  RMSE by forecast horizon h ∈ {1,3,6,12}
    ├── sources_table.csv/.tex   Data sources (variable → ECB/Eurostat code)
    └── figures/                 fig_01_hvpi_zeitreihe.png … fig_13_horizonte_rmse.png
```

## Data sources

| Role | Source | Series |
|------|--------|--------|
| Target variable | ECB SDW | German HICP `ICP/M.DE.N.000000.4.INX` |
| Predictors | Eurostat | Industrial production, business surveys, producer prices, unemployment, labour cost index |

33 predictor series → 165 lag features (5 lags × 33 series), after NaN filter **155 features** (1-month forecast horizon).

> **Sample window:** The raw cache (`data/raw/data_raw.csv`) extends to **2026-05**. IP and PPI
> series were rebased to I21 (2021=100); growth rates are materially identical. Shortest predictor
> end: `BS_Produktionserwart` 2024-09 → feature matrix extends to ca. **2024-10**; `dropna` trims
> to the common observation window.

> **Data source note:** The original plan used Deutsche Bundesbank (SDMX). Their API was
> unreachable from the working environment, so ECB + Eurostat are used instead (EU-harmonised,
> materially equivalent). This deviation is noted in the paper.

## Reproduction

Data are cached in `data/raw/data_raw.csv`; only on the first run (or with
`use_cache=False`) will data be fetched from ECB + Eurostat.

```bash
pip install -r requirements.txt

# Option A — run the full pipeline as a Python script
python -c "from src.pipeline import run_all; run_all()"

# Option B — execute notebook (writes figures to results/figures/)
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/LASSO_Ridge_Inflationsprognose.ipynb

# Run tests
pytest tests/
```

Or open the notebook interactively in Jupyter / VS Code and run all cells.
The committed notebook already contains the outputs of the last run; figures are also
available as PNGs in `results/figures/`.

## Results overview (last run)

<!-- RESULTS:BEGIN -->
Dataset: **261 observations** (2002-01 – 2024-10), of which **225 training / 36 test**
(test window 2021-06 – 2024-10), **155 features**.

**Test window (fixed chronological split), RMSE in percentage points of the inflation rate,
sorted by performance (DM = Diebold-Mariano test vs. random walk; n.s. = not significant):**

| Model | λ | Test RMSE | RMSE/RW | Test R² | DM | Coeff. ≠ 0 |
|-------|----------:|----------:|--------:|--------:|---:|-----------:|
| **Random Walk** | –        | **0.94** | **1.00** | 0.89 | – | – |
| Lag model (ADL) | –      | 1.05 | 1.12 | 0.87 |  n.s. | 5 |
| Adaptive LASSO | 0.00032 | 1.38 | 1.47 | 0.77 |  * | 50 / 155 |
| LASSO + HICP lags | 0.064  | 1.47 | 1.57 | 0.74 |  ** | 7 / 160 |
| LASSO | 0.030              | 1.83 | 1.95 | 0.59 |  ** | 29 / 155 |
| Elastic Net | 0.039        | 1.85 | 1.96 | 0.59 |  ** | 34 / 155 |
| Ridge | 54.8               | 1.96 | 2.08 | 0.54 |  ** | 155 / 155 |
| OLS | –                    | 3.40 | 3.62 | −0.40 |  ** | 155 / 155 |

DM test (HLN-corrected, T=36): no model beats the RW significantly (low power at T=36). Block-bootstrap CIs per model: see `results/inference_table.csv`.

**Robustness check (Rolling-Origin, expanding window):** RW 0.94 · AR 0.95 · LASSO+HICP 0.95 ·
LASSO 1.09 · Elastic Net 1.09 · Ridge 1.16 · OLS 2.34. The adaptive models (AR, LASSO+HICP)
nearly match the RW here, but do not beat it significantly (Diebold-Mariano n.s.).
<!-- RESULTS:END -->

### Key findings

1. **Regularisation fixes OLS overfitting.** OLS is unusable at p/n ≈ 0.69 with strong
   multicollinearity (test R² −0.40); Ridge/LASSO/Elastic Net/Adaptive LASSO stabilise estimation
   substantially (R² up to 0.77, Adaptive LASSO; 0.74 with LASSO+HICP lags), with plain LASSO
   selecting only 29 of 155 features.
2. **No model beats the naive Random Walk.** Across all horizons (h ∈ {1,3,6,12}),
   `ŷ_t = y_{t-1}` is the hardest benchmark — macroeconomic models remain above it.
3. **Macroeconomic marginal value ≈ 0.** Only with the HICP own-lags (LASSO+HICP) is the RW
   *matched*, not beaten. Pure macro models are structurally disadvantaged because they lack
   the best individual predictor — the last inflation rate.

This is consistent with the inflation forecasting literature (Atkeson & Ohanian 2001; Stock &
Watson 2007): structural models generally do not beat the naive benchmark. The Diebold-Mariano
test (HLN correction, T=36) confirms: no model significantly beats the RW at the 5% level.
