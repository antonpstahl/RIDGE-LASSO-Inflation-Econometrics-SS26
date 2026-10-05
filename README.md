# LASSO & Ridge Regression for Inflation Forecasting

[Deutsche Version](README_DE.md)

**Seminar paper, Current Topics in Econometrics**
Technische Universität Dresden, summer term 2026. Author: Anton Stahl. Supervisor: Prof. Bernhard Schipp

Forecasting the German HICP inflation rate from macroeconomic indicators using
**regularisation (Ridge, LASSO, Elastic Net)**, benchmarked against **naive baselines
(Random Walk, AR)**.

**Research question:** Do macroeconomic predictors with Ridge/LASSO beat pure inflation
persistence (Random Walk)?
**Key finding:** Regularisation fixes the severe OLS overfitting (test R² -0.40 to 0.77, Adaptive LASSO),
**but it does not beat the Random Walk**. The marginal macroeconomic contribution beyond
persistence is near zero. The analysis demonstrates regularisation and variable selection
with many highly collinear predictors *and* honestly benchmarks their forecast value
against the naive baseline.

![Forecast vs. actual HICP inflation rate in the test window 2021-06 to 2024-10](results/figures/fig_04_prognose.png)

---

## Project structure

```
RIDGE-LASSO-Inflation-Econometrics-SS26/
├── README.md                    This file (English)
├── README_DE.md                 German version
├── LICENSE                      MIT license (code)
├── requirements.txt             Pinned dependencies (tested with Python 3.13)
├── notebooks/
│   └── LASSO_Ridge_Inflationsprognose.ipynb   Main analysis notebook (with outputs)
├── src/                         Python package, reusable analysis functions
│   ├── config.py                Paths, seeds, hyperparameter grids, CV objects, series definitions
│   ├── data_preparation.py      Data download (ECB/Eurostat API + CSV cache)
│   ├── data_preprocessing.py    YoY/MoM transformation, lag feature engineering, train/test split
│   ├── models.py                Adaptive LASSO estimator (Ridge-initialised)
│   ├── training.py              Model fitting and cross-validation (OLS, Ridge, LASSO, Elastic Net,
│   │                            Adaptive LASSO, AR, LASSO + HICP lags)
│   ├── evaluation.py            Rolling-origin OOS, DM/CW tests, block bootstrap, regime analysis,
│   │                            Giacomini-Rossi test, horizon analysis, robustness checks
│   └── reporting.py             Figures (fig_01 to fig_14), CSV/LaTeX tables, README sync
├── tests/                       pytest suite for the functions in src/
├── data/
│   ├── raw/data_raw.csv         Raw data (index/rate values, cached)
│   └── processed/data_yoy.csv   YoY-transformed data
└── results/
    ├── results_table.csv/.tex         Model comparison on the fixed split (incl. benchmarks)
    ├── inference_table.csv/.tex       DM/CW tests, block-bootstrap CIs, Bonferroni correction
    ├── compare_oos_table.csv          Rolling-origin RMSE (fixed vs. adaptive λ)
    ├── regime_table.csv/.tex          RMSE in the shock vs. disinflation regime
    ├── gr_table.csv                   Giacomini-Rossi fluctuation test (rolling statistic)
    ├── horizons_table.csv/.tex        RMSE by forecast horizon h ∈ {1,3,6,12}
    ├── robustness_extended.csv/.tex   Sample extension to 2025-12 (post-shock OOS test)
    ├── robustness_mom_table.csv/.tex  MoM target + Atkeson-Ohanian benchmark
    ├── selection_economic.csv/.tex    LASSO selection grouped by economic channel
    ├── stationarity_table.csv/.tex    ADF/KPSS stationarity tests
    ├── sources_table.csv/.tex         Data sources (variable → ECB/Eurostat code)
    └── figures/                       fig_01_hvpi_zeitreihe.png ... fig_14_giacomini_rossi.png
```

## Data sources

| Role | Source | Series |
|------|--------|--------|
| Target variable | ECB Data Portal | German HICP `ICP/M.DE.N.000000.4.INX` |
| Predictors | Eurostat | Industrial production, business surveys, producer prices, unemployment, labour cost index |

33 predictor series → 165 lag features (5 lags × 33 series), after NaN filter **155 features** (1-month forecast horizon).
The full list of series and dataset codes is in `results/sources_table.csv`.

> **Sample window:** The raw cache (`data/raw/data_raw.csv`) extends to **2026-05**. IP and PPI
> series were rebased to I21 (2021=100), and growth rates stay materially identical. The shortest predictor
> ends at `BS_Produktionserwart` 2024-09, so the feature matrix extends to ca. **2024-10**. The `dropna` step then trims
> to the common observation window.

> **Data source note:** The original plan used Deutsche Bundesbank (SDMX). Their API was
> unreachable from the working environment, so ECB + Eurostat are used instead (EU-harmonised,
> materially equivalent). This deviation is noted in the paper.

## Reproduction

Requires Python 3.13 (the versions in `requirements.txt` are the tested environment).
Data are cached in `data/raw/data_raw.csv`. Only on the first run (or with
`use_cache=False`) is data fetched from ECB + Eurostat.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Run the test suite (covers the reusable functions in src/)
pytest tests/

# 2) Re-run the notebook end-to-end (regenerates all results/ artefacts + figures)
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/LASSO_Ridge_Inflationsprognose.ipynb
```

Or open the notebook interactively in Jupyter / VS Code, restart the kernel and "Run All".
The optional adaptive rolling-origin run (`RUN_ADAPTIVE_RO` in section 4.5) takes about
10 to 20 minutes. Set it to `False` for a faster run.
The notebook is the single, central analysis. `src/` holds only the reusable functions and
`tests/` covers them. The committed notebook already contains the outputs of the last run, and the
figures are also available as PNGs in `results/figures/`.

## Results overview (last run)

<!-- RESULTS:BEGIN -->
Dataset: **261 observations** (2002-01 - 2024-10), of which **225 training / 36 test**
(test window 2021-06 - 2024-10), **155 features**.

**Test window (fixed chronological split), RMSE in percentage points of the inflation rate.**
Test = DM (non-nested, two-sided) or CW (nested, Clark & West 2007, one-sided). \* p<0.10, \*\* p<0.05 (unadjusted), n.s. = not significant.

| Model | λ | Test RMSE | RMSE/RW | Test R² | Test | Coeff. ≠ 0 |
|-------|----------:|----------:|--------:|--------:|-----:|-----------:|
| *- Benchmark -* | | | | | | |
| **Random Walk** | - | **0.94** | **1.00** | 0.89 | - | - |
| Lag model (AR) | - | 1.05 | 1.12 | 0.87 | CW  * | 5 |
| *- Central comparison: own lags + macro (economically clean, ceteris paribus) -* | | | | | | |
| LASSO + HICP lags | 0.064 | 1.47 | 1.57 | 0.74 | CW  n.s. | 7 / 160 |
| *- Didactic: macro only, no own lags (structurally disadvantaged) -* | | | | | | |
| Adaptive LASSO | 0.00032 | 1.38 | 1.47 | 0.77 | DM  * (RW better) | 50 / 155 |
| LASSO | 0.030 | 1.83 | 1.95 | 0.59 | DM  ** (RW better) | 29 / 155 |
| Elastic Net | 0.039 | 1.85 | 1.96 | 0.59 | DM  ** (RW better) | 34 / 155 |
| Ridge | 54.8 | 1.96 | 2.08 | 0.54 | DM  ** (RW better) | 155 / 155 |
| OLS | - | 3.40 | 3.62 | -0.40 | DM  ** (RW better) | 155 / 155 |

**Central finding:** Lag model (AR, own lags only) RMSE/RW = 1.12 - LASSO+HICP (own lags + macro) RMSE/RW = 1.57, so macro value-added beyond persistence ≈ 0 (ceteris paribus).
The pure macro models (didactic group) lack the strongest single predictor (HICP lag) - their performance (RMSE/RW ≥ 1.47) illustrates regularization vs. OLS overfitting but is **not a fair race against the RW**.

Inference tests (T=36): DM = Diebold-Mariano (HLN-corrected, two-sided) for pure macro models, CW = Clark-West (2007, one-sided) for lag model and LASSO+HICP (nested within RW). After Bonferroni correction no model beats the RW significantly (low power at T=36). Block-bootstrap CIs: `results/inference_table.csv`.
*Note: The RW R² reflects the persistence (autocorrelation) of the YoY series (ŷ_t = y_{t-1}), it is not comparable to the model R².*

**Robustness check (rolling-origin, expanding window):** RW 0.94 - AR 0.95 - LASSO+HICP 0.95 -
LASSO 1.09 - Elastic Net 1.09 - Ridge 1.16 - OLS 2.34. The nested models (AR, LASSO+HICP)
nearly match the RW here, but do not beat it significantly (Clark-West test n.s.).

**Sample-extension robustness:** Dropping the single binding series (`BS_Produktionserwart`, ends 2024-09) extends the OOS window to **2025-12** (+14 months, post-shock segment 2023-04-2025-12, n=28, was 14). In the calmer post-shock regime **AR, LASSO+HICP** beat the RW significantly (DM/CW p<0.10 unadjusted, n.s. after Bonferroni correction; best model LASSO+HICP, RMSE/RW=0.95). This is the first time the *RW-unbeatable* claim is tested out-of-sample outside the energy price shock. Table: `results/robustness_extended.csv`.
<!-- RESULTS:END -->

### Key findings

1. **Regularisation fixes OLS overfitting.** OLS is unusable at p/n ≈ 0.69 with strong
   multicollinearity (test R² -0.40). Ridge, LASSO, Elastic Net and Adaptive LASSO stabilise estimation
   substantially (R² up to 0.77 with Adaptive LASSO, 0.74 with LASSO+HICP lags), and plain LASSO
   selects only 29 of 155 features.
2. **No model beats the naive Random Walk at short horizons.** At h ∈ {1, 3, 6} the Random Walk
   is the hardest benchmark, and the macroeconomic models remain above it. Only at h = 12 do
   LASSO/Elastic Net (RMSE/RW ≈ 0.90) and Ridge (≈ 0.96) undercut the weak 12-month Random Walk
   `ŷ_t = y_{t-12}` in point terms (the horizon analysis has no significance test).
3. **Macroeconomic marginal value ≈ 0.** Even with the HICP own lags, LASSO+HICP only *matches*
   the RW in the rolling-origin design (RMSE/RW ≈ 1.01, the same as the pure AR model) and does not
   beat it. Pure macro models are structurally disadvantaged because they lack the best individual
   predictor, the last inflation rate.

This is consistent with the inflation forecasting literature (Atkeson & Ohanian 2001, Stock &
Watson 2007), where structural models generally do not beat the naive benchmark. The Diebold-Mariano
and Clark-West tests (T=36) confirm that no model significantly beats the RW at the 5% level.

4. **Contrast to Medeiros et al. (2021).** Medeiros, Vasconcelos, Veiga & Zilberman (JBES 2021,
   "Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods")
   find robust ML forecast gains for US-CPI. There are four explanations for the absent ML advantage
   in the German HICP setting, and each one ties directly back to the results here.
   (a) The 2021 to 2024 test window is dominated by the largest energy-price shock in the sample,
   which makes the near-I(1) Random Walk mechanically hard to beat (confirmed by the regime analysis
   and the Giacomini-Rossi test, notebook sections 4.5.2 and 4.5.3).
   (b) The YoY target embeds a strong persistence-driven autocorrelation (12-month cumulation),
   which produces an exceptionally strong RW benchmark that is partly an artefact of the target definition.
   (c) The analysis uses linear shrinkage only (Ridge/LASSO/EN), whereas Medeiros et al. exploit
   Random Forests that capture non-linearities and regime interactions.
   (d) The EU sample (2002 to 2024, 261 obs.) is shorter and dominated by a low-variance ZLB era
   (2015 to 2021), which weakens the training signal of macro predictors.
   The null finding here *complements* Medeiros et al.: it shows that the ML advantage is
   context-dependent and does not replicate in this linear, European, shock-dominated setting.

## License

The code in this repository is released under the [MIT License](LICENSE).
The data in `data/` were retrieved from the ECB Data Portal and from Eurostat
(© European Union) and remain subject to their reuse policies, which allow free reuse
provided the source is acknowledged. They are not covered by the MIT License.
