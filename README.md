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
├── src/                         Python package — reusable analysis functions
│   ├── __init__.py
│   ├── config.py                Paths, seeds, hyperparameter grids, CV objects
│   ├── data_preparation.py      Data download (ECB/Eurostat API + CSV cache)
│   ├── data_preprocessing.py    YoY transformation, lag feature engineering
│   ├── evaluation.py            OOS evaluation, rolling-origin, DM test, horizon analysis
│   ├── models.py                Model definitions (OLS, Ridge, LASSO, Elastic Net, AR)
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

# 1) Run the test suite (covers the reusable functions in src/)
pytest tests/

# 2) Re-run the notebook end-to-end (regenerates all results/ artefacts + figures)
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/LASSO_Ridge_Inflationsprognose.ipynb
```

Or open the notebook interactively in Jupyter / VS Code, restart the kernel and "Run All".
The notebook is the single, central analysis; `src/` holds only the reusable functions and
`tests/` covers them. The committed notebook already contains the outputs of the last run;
figures are also available as PNGs in `results/figures/`.

## Results overview (last run)

<!-- RESULTS:BEGIN -->
Dataset: **261 observations** (2002-01 – 2024-10), of which **225 training / 36 test**
(test window 2021-06 – 2024-10), **155 features**.

**Test window (fixed chronological split), RMSE in percentage points of the inflation rate.**
Test = DM (non-nested) or CW (nested, Clark & West 2007); n.s. = not significant.

| Model | λ | Test RMSE | RMSE/RW | Test R² | Test | Coeff. ≠ 0 |
|-------|----------:|----------:|--------:|--------:|-----:|-----------:|
| *— Benchmark —* | | | | | | |
| **Random Walk** | – | **0.94** | **1.00** | 0.89 | – | – |
| Lag model (ADL) | – | 1.05 | 1.12 | 0.87 | CW  * | 5 |
| *— Central comparison: own lags + macro (economically clean, ceteris paribus) —* | | | | | | |
| LASSO + HICP lags | 0.064 | 1.47 | 1.57 | 0.74 | CW  n.s. | 7 / 160 |
| *— Didactic: macro only, no own lags (structurally disadvantaged) —* | | | | | | |
| Adaptive LASSO | 0.00032 | 1.38 | 1.47 | 0.77 | DM  * | 50 / 155 |
| LASSO | 0.030 | 1.83 | 1.95 | 0.59 | DM  ** | 29 / 155 |
| Elastic Net | 0.039 | 1.85 | 1.96 | 0.59 | DM  ** | 34 / 155 |
| Ridge | 54.8 | 1.96 | 2.08 | 0.54 | DM  ** | 155 / 155 |
| OLS | – | 3.40 | 3.62 | −0.40 | DM  ** | 155 / 155 |

**Central finding:** Lag model (ADL, own lags only) RMSE/RW = 1.12 · LASSO+HICP (own lags + macro) RMSE/RW = 1.57 → macro value-added beyond persistence ≈ 0 (ceteris paribus).
The pure macro models (didactic group) lack the strongest single predictor (HICP lag) — their performance (RMSE/RW ≥ 1.47) illustrates regularization vs. OLS overfitting but is **not a fair race against the RW**.

Inference tests (T=36): DM = Diebold-Mariano (HLN-corrected, two-sided) for pure macro models; CW = Clark-West (2007, one-sided) for lag model and LASSO+HICP (nested within RW). No model beats the RW significantly (low power at T=36). Block-bootstrap CIs: `results/inference_table.csv`.
*Note: The RW R² reflects the persistence (autocorrelation) of the YoY series (ŷ_t = y_{t−1}); it is not comparable to the model R².*

**Robustness check (Rolling-Origin, expanding window):** RW 0.94 · AR 0.95 · LASSO+HICP 0.95 ·
LASSO 1.09 · Elastic Net 1.09 · Ridge 1.16 · OLS 2.34. The nested models (AR, LASSO+HICP)
nearly match the RW here, but do not beat it significantly (Clark-West test n.s.).

**Sample-extension robustness (AP32):** Dropping the single binding series (`BS_Produktionserwart`, ends 2024-09) extends the OOS window to **2025-12** (+14 months; post-shock segment 2023-04–2025-12, n=28, was 14). In the calmer post-shock regime **AR, LASSO+HVPI** beat the RW significantly (DM/CW p<0.10; best LASSO+HVPI, RMSE/RW=0.95). → The *RW-unbeatable* claim is thus tested out-of-sample outside the energy price shock for the first time. Table: `results/robustness_extended.csv`.
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

4. **Contrast to Medeiros et al. (2021).** Medeiros, Vasconcelos, Veiga & Zilberman (JBES 2021,
   "Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods")
   find robust ML forecast gains for US-CPI. Four explanations — each directly linked to the
   results here — for the absent ML advantage in the German HICP setting:
   (a) The 2021–2024 test window is dominated by the largest energy-price shock in the sample,
   making the near-I(1) Random Walk mechanically hard to beat (confirmed by the regime analysis
   §4.5.2 and the Giacomini-Rossi test §4.5.3);
   (b) the YoY target embeds a strong persistence-driven autocorrelation (12-month cumulation),
   producing an exceptionally strong RW benchmark — partly an artefact of the target definition;
   (c) the analysis uses linear shrinkage only (Ridge/LASSO/EN), whereas Medeiros et al. exploit
   Random Forests that capture non-linearities and regime interactions;
   (d) the EU sample (2002–2024, 261 obs.) is shorter and dominated by a low-variance ZLB era
   (2015–2021), weakening the training signal of macro predictors.
   The null finding here *complements* Medeiros et al.: it shows that the ML advantage is
   context-dependent and does not replicate in this linear, European, shock-dominated setting.
