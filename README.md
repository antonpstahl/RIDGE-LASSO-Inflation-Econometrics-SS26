# LASSO & Ridge Regression zur Inflationsprognose

**Seminararbeit · Aktuelle Fragen der Ökonometrie**
Technische Universität Dresden · Betreuer: Prof. Bernhard Schipp

Empirischer Vergleich von **OLS, Ridge und LASSO** bei der Prognose der deutschen
HVPI-Inflationsrate aus makroökonomischen Indikatoren. Demonstriert Regularisierung
und Variablenselektion in einer Situation mit vielen, stark kollinearen Prädiktoren.

---

## Projektstruktur

```
RIDGE_LASSO_Inflation_Econometrics_SS26/
├── README.md                  Diese Datei
├── requirements.txt           Gepinnte Abhängigkeiten
├── notebooks/
│   └── LASSO_Ridge_Inflationsprognose.ipynb   Eigenständige Hauptanalyse (mit Outputs)
├── src/                       Optional/Legacy – vom Notebook nicht mehr benötigt
│   ├── data_loader.py         Datenabruf-Funktionen (jetzt im Notebook inline)
│   └── create_notebook.py     Früherer Notebook-Generator (obsolet)
├── data/
│   ├── raw/data_raw.csv        Rohdaten (Index-/Quotenwerte)
│   └── processed/data_yoy.csv  YoY-transformierte Daten
├── results/
│   ├── results_table.csv       Modellvergleich (MSE/RMSE/R²)
│   └── figures/                fig_01 … fig_10 (PNG)
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
sind direkt enthalten (kein Import aus `src/`). Daten werden aus `data/raw/data_raw.csv`
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

| Modell | λ | Train MSE | Test MSE | Test RMSE | Test R² | Koeff. ≠ 0 |
|--------|----------:|----------:|---------:|----------:|--------:|-----------:|
| OLS    | –         | 0.0131    | 27.79    | 5.27 %    | −1.65   | 165 / 165  |
| Ridge  | 464.16    | 0.1685    | 9.31     | 3.05 %    | 0.11    | 165 / 165  |
| LASSO  | 0.0343    | 0.1193    | **2.63** | **1.62 %**| **0.75**| 29 / 165   |

LASSO ist klar am besten und selektiert 29 von 165 Features. OLS überanpasst stark
(hohes p/n-Verhältnis, Multikollinearität).
