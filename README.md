# LASSO & Ridge Regression zur Inflationsprognose

**Seminararbeit · Aktuelle Fragen der Ökonometrie**
Technische Universität Dresden · Betreuer: Prof. Bernhard Schipp

Prognose der deutschen HVPI-Inflationsrate aus makroökonomischen Indikatoren mit
**Regularisierung (Ridge, LASSO, Elastic Net)** — gemessen **gegen naive Benchmarks
(Random Walk, AR)**.

**Forschungsfrage:** Schlagen makroökonomische Prädiktoren mit Ridge/LASSO die reine
Inflationspersistenz (Random Walk)?
**Kernbefund:** Regularisierung behebt das massive Overfitting von OLS (Test-R² −0,40 → 0,74),
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

33 Prädiktor-Reihen → 165 Lag-Features (5 Lags × 33 Reihen), nach NaN-Filter **155 Features** (Prognose-Horizont 1 Monat).

> **Hinweis zum Stichprobenfenster:** Der Roh-Cache (`data/raw/data_raw.csv`) reicht
> bis **2026-05**. IP- und PPI-Reihen wurden auf Basisjahr I21 (2021=100) umgestellt
> (I15 endete bei 2023-12; Wachstumsraten inhaltlich identisch). Kürzestes Prädiktorende:
> `BS_Produktionserwart` 2024-09 → Feature-Matrix reicht bis ca. **2024-10**; das
> `dropna` schneidet auf das gemeinsame Beobachtungsfenster zu.

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

<!-- RESULTS:BEGIN -->
Datensatz: **261 Beobachtungen** (2002-01 – 2024-10), davon **225 Training / 36 Test**
(Testfenster 2021-06 – 2024-10), **155 Features**.

**Testfenster (fester chronologischer Split), RMSE in Prozentpunkten der Inflationsrate,
sortiert nach Güte:**

| Modell | λ | Test-RMSE | RMSE/RW | Test-R² | Koeff. ≠ 0 |
|--------|----------:|----------:|--------:|--------:|-----------:|
| **Random Walk** | –        | **0.94** | **1.00** | 0.89 | – |
| Lag-Modell (ADL) | –      | 1.05 | 1.12 | 0.87 | 5 |
| LASSO + HVPI-Lags | 0.064  | 1.47 | 1.57 | 0.74 | 7 / 160 |
| LASSO | 0.030              | 1.83 | 1.95 | 0.59 | 29 / 155 |
| Elastic Net | 0.039        | 1.85 | 1.96 | 0.59 | 34 / 155 |
| Ridge | 54.8               | 1.96 | 2.08 | 0.54 | 155 / 155 |
| OLS | –                    | 3.40 | 3.62 | −0.40 | 155 / 155 |

**Robustheitscheck (Rolling-Origin, Expanding Window):** RW 0.94 · AR 0.95 · LASSO+HVPI 0.95 ·
LASSO 1.09 · Elastic Net 1.09 · Ridge 1.16 · OLS 2.34. Die adaptiven Modelle (AR, LASSO+HVPI)
erreichen den RW hier knapp, schlagen ihn aber nicht nachweisbar (Diebold-Mariano n.s.).
<!-- RESULTS:END -->

### Kernbefunde

1. **Regularisierung behebt OLS-Overfitting.** OLS ist bei p/n ≈ 0,69 und starker
   Multikollinearität unbrauchbar (Test-R² −0,40); Ridge/LASSO/Elastic Net stabilisieren die
   Schätzung deutlich (R² bis 0,74 mit LASSO+HVPI), LASSO selektiert dabei nur 29 von 155 Features.
2. **Kein Modell schlägt den naiven Random Walk.** Über alle Horizonte (h ∈ {1,3,6,12}) ist
   `ŷ_t = y_{t-1}` die härteste Messlatte — die makroökonomischen Modelle liegen darüber.
3. **Makro-Mehrwert ≈ 0.** Erst mit den HVPI-Eigen-Lags (LASSO+HVPI) wird der RW *erreicht*,
   nicht geschlagen. Die reinen Makro-Modelle sind strukturell benachteiligt, weil ihnen der
   beste Einzelprädiktor — die letzte Inflationsrate — fehlt.

Das deckt sich mit der Literatur zur Inflationsprognose (Atkeson & Ohanian 2001; Stock &
Watson 2007): strukturelle Modelle schlagen den naiven Benchmark in der Regel nicht. Der
Diebold-Mariano-Test (HLN-Korrektur, T=36) bestätigt: kein Modell schlägt den RW
nachweisbar auf dem 5-%-Niveau.
