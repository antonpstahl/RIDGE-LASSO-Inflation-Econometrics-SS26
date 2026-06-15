# LASSO & Ridge Regression zur Inflationsprognose

**Seminararbeit · Aktuelle Fragen der Ökonometrie**
Technische Universität Dresden · Betreuer: Prof. Bernhard Schipp

Prognose der deutschen HVPI-Inflationsrate aus makroökonomischen Indikatoren mit
**Regularisierung (Ridge, LASSO, Elastic Net)** — gemessen **gegen naive Benchmarks
(Random Walk, AR)**.

**Forschungsfrage:** Schlagen makroökonomische Prädiktoren mit Ridge/LASSO die reine
Inflationspersistenz (Random Walk)?
**Kernbefund:** Regularisierung behebt das massive Overfitting von OLS (Test-R² −1,41 → 0,75),
**schlägt den Random Walk aber nicht** — der Makro-Mehrwert über die Persistenz hinaus ist
nahe null. Die Analyse demonstriert damit Regularisierung und Variablenselektion bei vielen,
stark kollinearen Prädiktoren *und* ordnet ihren Prognosewert ehrlich gegen den naiven
Benchmark ein.

---

## Projektstruktur

```
RIDGE_LASSO_Inflation_Econometrics_SS26/
├── README.md                  Diese Datei
├── requirements.txt           Gepinnte Abhängigkeiten
├── notebooks/
│   └── LASSO_Ridge_Inflationsprognose.ipynb   Eigenständige Hauptanalyse (mit Outputs)
├── IMPLEMENTIERUNGSPLAN.md    Methodik-, Korrektur- und Schreibfahrplan
├── data/
│   ├── raw/data_raw.csv        Rohdaten (Index-/Quotenwerte)
│   └── processed/data_yoy.csv  YoY-transformierte Daten
├── results/
│   ├── results_table.csv       Modellvergleich (MSE/RMSE/R², inkl. Benchmarks)
│   ├── horizons_table.csv      RMSE je Prognose-Horizont h ∈ {1,3,6,12}
│   ├── sources_table.csv       Datenquellen (Variable → ECB/Eurostat-Code)
│   └── figures/                fig_01 … fig_13 (PNG)
└── docs/
    └── Vorgehensplan_Seminararbeit_Oekonometrie.pdf
```

## Datenquellen

| Rolle | Quelle | Reihe(n) |
|-------|--------|----------|
| Zielvariable | ECB SDW | HVPI Deutschland `ICP/M.DE.N.000000.4.INX` |
| Prädiktoren | Eurostat | Industrieproduktion, Business Surveys, Produzentenpreise, Arbeitslosigkeit, Lohnkostenindex |

33 Prädiktor-Reihen → **165 Features** mit Lags `[1, 2, 3, 6, 12]` (Prognose-Horizont 1 Monat).

> **Hinweis zum Stichprobenfenster:** Der Roh-Cache (`data/raw/data_raw.csv`) reicht
> bis **2026-04**, die Modellierungsstichprobe endet jedoch bei **2024-01**. Grund: Die
> Eurostat-Industrieproduktions- und Produzentenpreis-Reihen enden im verwendeten Cache
> bei 2023-12; der >20 %-NaN-Filter und das abschließende `dropna` schneiden auf das
> gemeinsame Fenster aller Reihen zu (Horizont +1 → letztes Ziel 2024-01).

> **Hinweis zur Datenquelle:** Der ursprüngliche Vorgehensplan sah die Deutsche
> Bundesbank (SDMX) vor. Deren API war aus der Arbeitsumgebung nicht erreichbar,
> daher wird auf ECB + Eurostat zurückgegriffen (EU-harmonisiert, inhaltlich
> gleichwertig). Diese Abweichung ist in der Arbeit zu erwähnen.

## Reproduktion

Das Notebook ist **eigenständig** – Datenabruf, YoY-Transformation und Lag-Features
sind direkt enthalten (kein separates Python-Modul nötig). Daten werden aus `data/raw/data_raw.csv`
gecacht; nur beim ersten Lauf (oder mit `get_raw_data(use_cache=False)`) wird von
ECB + Eurostat geladen.

```bash
pip install -r requirements.txt

# Notebook ausführen (nutzt den Daten-Cache, schreibt Abbildungen nach results/figures/)
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/LASSO_Ridge_Inflationsprognose.ipynb
```

Oder einfach interaktiv in Jupyter / VS Code öffnen und alle Zellen ausführen.
Das eingecheckte Notebook enthält bereits die Outputs des letzten Laufs; die
Abbildungen liegen zusätzlich als PNG in `results/figures/`.

## Ergebnis-Überblick (letzter Lauf)

Datensatz: **254 Beobachtungen** (2002-01 – 2024-01), davon **218 Training / 36 Test**
(Testfenster 2020-11 – 2024-01), **165 Features**.

**Testfenster (fester chronologischer Split), RMSE in Prozentpunkten der Inflationsrate,
sortiert nach Güte:**

| Modell | λ | Test-RMSE | RMSE/RW | Test-R² | Koeff. ≠ 0 |
|--------|----------:|----------:|--------:|--------:|-----------:|
| **Random Walk** | –        | **0.99** | **1.00** | 0.91 | – |
| AR (Lags 1,2,3,6,12) | –   | 1.01 | 1.02 | 0.90 | 5 |
| LASSO + HVPI-Lags | 0.066  | 1.44 | 1.45 | 0.80 | 9 / 170 |
| LASSO | 0.034              | 1.62 | 1.63 | 0.75 | 27 / 165 |
| Elastic Net | 0.038        | 1.63 | 1.64 | 0.75 | 26 / 165 |
| Ridge | 403.7              | 2.94 | 2.96 | 0.17 | 165 / 165 |
| OLS | –                    | 5.03 | 5.06 | −1.41 | 165 / 165 |

**Robustheitscheck (Rolling-Origin, Expanding Window):** RW 0.99 · AR 0.96 · LASSO+HVPI 0.98 ·
LASSO 1.05 · Elastic Net 1.05 · Ridge 1.52 · OLS 3.61. Die adaptiven Modelle (AR, LASSO+HVPI)
erreichen den RW hier knapp, schlagen ihn aber nicht nachweisbar.

### Kernbefunde

1. **Regularisierung behebt OLS-Overfitting.** OLS ist bei p/n ≈ 0,76 und starker
   Multikollinearität unbrauchbar (Test-R² −1,41); Ridge/LASSO/Elastic Net stabilisieren die
   Schätzung deutlich (R² bis 0,75), LASSO selektiert dabei nur 27 von 165 Features.
2. **Kein Modell schlägt den naiven Random Walk.** Über alle Horizonte (h ∈ {1,3,6,12}) ist
   `ŷ_t = y_{t-1}` die härteste Messlatte — die makroökonomischen Modelle liegen darüber.
3. **Makro-Mehrwert ≈ 0.** Erst mit den HVPI-Eigen-Lags (LASSO+HVPI) wird der RW *erreicht*,
   nicht geschlagen. Die reinen Makro-Modelle sind strukturell benachteiligt, weil ihnen der
   beste Einzelprädiktor — die letzte Inflationsrate — fehlt.

Das deckt sich mit der Literatur zur Inflationsprognose (Atkeson & Ohanian 2001; Stock &
Watson 2007): strukturelle Modelle schlagen den naiven Benchmark in der Regel nicht. Eine
formale Absicherung (Diebold-Mariano-Test) sowie die Verlängerung der Stichprobe um die
Disinflation 2024–25 sind im [Implementierungsplan](IMPLEMENTIERUNGSPLAN.md) (Phase B) vorgesehen.
