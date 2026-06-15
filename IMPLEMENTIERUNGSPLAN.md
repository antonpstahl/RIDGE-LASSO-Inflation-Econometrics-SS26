# Implementierungsplan — LASSO & Ridge zur Inflationsprognose

**Seminararbeit · Aktuelle Fragen der Ökonometrie · TU Dresden · Prof. Bernhard Schipp**
Stand: 2026-06-14 · Status: **Phase A (Empirie) abgeschlossen** · internes Gutachten deckt
zentralen Interpretationsdefekt auf → **Phase B (Korrektur & Schreiben) offen**

Dieser Plan beschreibt (a) die methodischen und strukturellen Verbesserungen der
Implementierung, (b) den Schreibfahrplan, der Code-Ergebnisse auf die Gliederung
der Arbeit abbildet, und (c) — neu — den **Korrekturplan aus dem Gutachten** (Phase B, §1b
und AP7–AP12). Er ersetzt den groben PDF-Vorgehensplan (`docs/`) auf der Umsetzungsebene.

---

## 0. Zweck & Leitlinie

**Korrigierte Leitfrage (nach Gutachten):** *Schlagen makroökonomische Prädiktoren mit
Ridge/LASSO die reine Inflationspersistenz (Random-Walk-Benchmark)?* Die empirische
Antwort der vorliegenden Daten lautet **nein** — der naive Random Walk ist über alle
Horizonte die härteste Messlatte (siehe G1).

Daraus folgt die belastbare, ehrliche These der Arbeit:

> **Regularisierung rettet OLS vor katastrophalem Overfitting** (Test-R²: −1,41 → 0,75),
> **liefert aber keinen Prognosemehrwert über die Inflationspersistenz hinaus.** Sobald die
> HVPI-Eigen-Lags zugelassen werden (AR, LASSO+HVPI), wird der Random Walk gerade erreicht,
> nicht geschlagen ⇒ der Makro-Mehrwert ist ≈ 0.

Die ursprüngliche Leitlinie („zeigen, dass Regularisierung OLS schlägt und LASSO am besten
ist") bleibt als *didaktischer Teilaspekt* erhalten (OLS vs. Ridge/LASSO), ist aber **nicht**
die wissenschaftliche Kernaussage — der Vergleich nur gegen OLS wäre ein Strohmann. Die
Einordnung erfolgt im Lichte von Atkeson & Ohanian (2001) und Stock & Watson (2007):
Strukturmodelle schlagen den naiven Benchmark in der Inflationsprognose typischerweise nicht.

---

## 1. Aktueller Stand — Kurzbewertung

**Stärken (beibehalten):**
- Korrekte Zeitreihen-Hygiene: chronologischer Split, `TimeSeriesSplit`-CV,
  `StandardScaler` nur auf Train gefittet, verlagerte Prädiktoren.
- Theoriegeleitete Prädiktorauswahl (Phillips-Kurve, Cost-Push, Angebotsseite, Erwartungen).
- Reproduzierbar (Notebook aus Skript generiert, Daten gecacht) und vollständig visualisiert.
- **Die richtigen Benchmarks wurden gebaut** (RW, AR, LASSO+HVPI, relatives RMSE) — das
  Auswertungsgerüst trägt bereits die korrekte Aussage; sie muss nur gezogen werden.

**Schwachpunkte Phase A (S1–S7) — in dieser Sitzung adressiert (Stand: umgesetzt, siehe §7):**

| # | Befund | Konsequenz |
|---|--------|-----------|
| S1 | **„n < p"-Erzählung falsch.** Tatsächlich n_train = 218, p = 165 (Notebook druckt `n < p: NEIN`); Kommentar in `create_notebook.py` behauptet das Gegenteil. | Sachfehler, fällt einem Prüfer auf. OLS scheitert an **p/n ≈ 0,76 + Multikollinearität**, nicht an Unteridentifikation. |
| S2 | **Kein naiver Benchmark**; HVPI-Eigen-Lags als Prädiktoren ausgeschlossen. | Nicht belegbar, ob Modelle die Inflationspersistenz schlagen. **Größte inhaltliche Lücke.** |
| S3 | **Look-ahead-Leakage** bei Lohnkosten (lineare Q→M-Interpolation nutzt Zukunftswerte). | Verzerrt Prognose minimal zugunsten der Modelle. |
| S4 | **Einzelnes Testfenster** (2020-11 – 2024-01, volatiles Regime). | Drei Kennzahlen, regimeabhängig, keine Streuungsaussage. |
| S5 | **27 Monate verworfen** (Rohdaten bis 2026-04, Feature-Matrix endet 2024-01 wg. kürzester Reihe). | Aktuelle Daten ungenutzt; Ursache undokumentiert. |
| S6 | NaN-Spaltenfilter auf **Gesamtstichprobe** statt nur Train. | Geringfügiger Leak (Selektion nach Missingness). |
| S7 | Gemischte Saisonbereinigung (IP/PPI = NSA, Surveys/ALQ = SA). | Begründungsbedürftig. |

**Struktur (bereits erledigt in dieser Sitzung):** professionelle Ordnerstruktur,
Pfade angepasst, `README.md`, `.gitignore`, gepinnte `requirements.txt`,
`__pycache__` entfernt.

### 1b. Gutachten-Befunde (G1–G8, nach Phase A) — **offen, Phase B**

Verifiziert am ausgeführten Notebook und an `results/results_table.csv` /
`results/horizons_table.csv`:

| # | Befund | Beleg | Konsequenz |
|---|--------|-------|-----------|
| **G1** | **KERNDEFEKT: Random Walk schlägt jedes Makro-Modell.** Test-RMSE RW = 0,99; LASSO = 1,62 (+63 %); RW gewinnt auch bei h = 1/3/6/12. Notebook druckt selbst *„Bestes Modell: Random Walk"* (Cell 45). **README behauptet dennoch „LASSO ist klar am besten" und lässt RW/AR/EN/LASSO+HVPI aus der Headline-Tabelle.** | Cell 27/45, README | Zentrale Aussage der Arbeit derzeit **falsch dargestellt**. → AP7 |
| **G2** | **Einzelfenster ↔ Rolling-Origin widersprechen sich.** AR/LASSO+HVPI *verlieren* im Einzelfenster (1,01 / 1,44), *gewinnen* aber rolling knapp gegen RW (0,96 / 0,98). Unkommentiert. | Cell 27 vs. Cell 32 | Rangfolge nicht belastbar; Prüfer-Angriffsfläche. → AP8 |
| **G3** | **Kein Signifikanztest.** 36 Testpunkte, RMSE 0,96 vs. 0,99 ⇒ Unterschiede sind höchstwahrscheinlich Rauschen. Diebold-Mariano (in AP3 vorgesehen) fehlt. | — | Jede „A schlägt B"-Aussage **unbelegt**. → AP9 |
| **G4** | **h = 12 degeneriert:** LASSO **und** Elastic Net selektieren 0 Variablen (reiner Intercept, identischer RMSE 5,316). Unkommentiert. | horizons_table.csv | Wirkt wie Bug; ist interpretierbar (kein Signal auf Jahreshorizont). → AP10 |
| **G5** | **Numerische Instabilität:** `divide by zero / overflow / invalid value in matmul` in LASSO+HVPI. Nicht vom `ConvergenceWarning`-Filter abgedeckt, ungeklärt. | Cell 21 Output | Verdacht auf Near-Singularität/Skalierungsproblem. → AP10 |
| **G6** | **27 Monate Zieldaten verworfen** (HVPI bis 2025-12 gültig; Modell endet 2024-01 wegen IP/PPI-Cache bis 2023-12). Disinflation 2024–25 ungenutzt. | Cell 8, Rohcache | Verschärfung von S5: Ursache klar, Behebung offen; bestes OOS-Fenster fehlt. → AP11 |
| **G7** | **„AR(p)" terminologisch falsch** — verwendet Lags {1,2,3,6,12}, nicht 1…p. | Cell 20 | Eigentlich restringierter AR / ADL. → AP10 |
| **G8** | **Repo-/Doku-Inkonsistenz:** README beschreibt nicht existierendes `src/`; README-Zahlen ≠ Notebook-Zahlen (veralteter Lauf); kein `random_state`. | README, git ls-files | Schlampige Außendarstellung. → AP7/AP12 |

---

## 2. Zielbild der finalen Empirie

Eine Auswertung, die folgendes vergleicht — **leakage-frei** und über **mehrere Horizonte**:

- **Benchmarks:** Random Walk, AR/ADL auf der Inflationsrate
- **Regularisierung:** OLS, Ridge, LASSO, Elastic Net
- **Validierung:** Rolling-Origin-Out-of-Sample (pseudo-echtzeit) statt Einzelfenster
- **Inferenz:** Signifikanz der Prognoseunterschiede (Diebold-Mariano) statt nur Punkt-RMSE
- **Ausgabe:** Gütetabelle (RMSE, R², relativ zum RW), Prognoseplots, Koeffizientenpfade,
  Selektionsstabilität
- **Aussage:** **benchmark-zentriert** — der RW ist die Messlatte; jede Tabelle/Abbildung
  führt ihn mit; das Fazit bewertet den Makro-Mehrwert *über die Persistenz hinaus*.

---

## 3. Arbeitspakete (priorisiert)

> Reihenfolge ist bewusst: erst **korrigieren** (AP1), dann **einordnen** (AP2 Benchmark),
> dann **absichern** (AP3 OOS), dann **erweitern** (AP4/AP5).
> Phase B (AP7–AP12) korrigiert anschließend die im Gutachten gefundenen Defekte.

### AP1 — Methodische Korrekturen *(Priorität: hoch, Aufwand: gering)*

**1.1 `n>p`-Framing korrigieren.** In `src/create_notebook.py` (OLS-Zelle) und in der
schriftlichen Arbeit: Motivation umformulieren auf *hohes p/n-Verhältnis (165/218) +
Multikollinearität → Overfitting*. Den irreführenden Kommentar
„Bei n < p ist OLS nicht identifiziert" entfernen.
→ *Bezug: Kapitel 2.1 (Grenzen von OLS), 3 (Daten).*

**1.2 Look-ahead bei Lohnkosten beheben.** In `src/data_loader.py`,
`load_all_data()` (LCI-Block):
```python
# vorher (nutzt den späteren Quartalswert für Zwischenmonate → Leak):
s_m = s_q.resample("MS").interpolate(method="linear")
# nachher (Treppenfunktion, nur Vergangenheit):
s_m = s_q.resample("MS").ffill()
```
→ *Bezug: Kapitel 3 (Methodik, „keine Datenleckage").*

**1.3 NaN-Filter nur auf Train.** `build_feature_matrix()` so erweitern, dass die
>20%-NaN-Regel nur auf dem Trainingsteil bestimmt wird (z. B. `train_end`-Argument
übergeben; `nan_frac` auf `X.iloc[:train_end]` berechnen).
→ *Bezug: Kapitel 3.*

**1.4 Konvergenz statt Pauschal-Ignore.** `warnings.filterwarnings("ignore")` entfernen
oder gezielt setzen; prüfen, ob `Lasso`/`LassoCV` mit `max_iter=10000` konvergieren
(sonst erhöhen). Verhindert verdeckte Nicht-Konvergenz.

**1.5 Stichproben-Truncation behandeln.** Die kürzeste Prädiktorreihe identifizieren
(sie kappt die Matrix auf 2024-01). Optionen dokumentieren/umsetzen: Reihe aktualisieren,
ausschließen, oder Enddatum bewusst als Stichprobengrenze begründen.
→ *Bezug: Kapitel 3 (Datensatzbeschreibung, Stichprobenzeitraum).*

### AP2 — Benchmarks: Random Walk & AR *(Priorität: hoch, Aufwand: gering–mittel)*

Ohne diese Modelle ist keine Aussage „die Modelle prognostizieren gut" möglich.

- **Random Walk** auf der YoY-Rate: `ŷ_t = y_{t-1}` (für Horizont h: `y.shift(h)`).
- **AR(p)** auf der Inflationsrate (eigene Lags). Empfehlung: `statsmodels.AutoReg`
  (dann `statsmodels` zu `requirements.txt`), alternativ HVPI-Lags als Features in
  `LinearRegression`.
- **„Makro-Mehrwert"-Modell:** LASSO **inklusive** HVPI-Eigen-Lags. Zeigt, ob die
  Makro-Prädiktoren *über die Persistenz hinaus* etwas beitragen — das eigentlich
  interessante Ergebnis.
- **Relative Güte:** Theil's U bzw. `RMSE_Modell / RMSE_RW` in die Ergebnistabelle.

In Ergebnistabelle (`results/results_table.csv`) und Prognoseplot (`fig_04`) integrieren.
→ *Bezug: Kapitel 4 (Empirische Ergebnisse), neue Benchmark-Spalte.*

### AP3 — Rolling-Origin-Out-of-Sample *(Priorität: hoch, Aufwand: mittel)*

Einzelfenster (S4) durch fortlaufende Pseudo-Echtzeit-Prognose ersetzen:

```python
def rolling_origin(model_factory, X, y, start, horizon=1):
    preds, idx = [], []
    for t in range(start, len(y) - horizon + 1):
        Xtr, ytr = X.iloc[:t], y.iloc[:t]
        sc = StandardScaler().fit(Xtr)
        m  = model_factory().fit(sc.transform(Xtr), ytr)
        preds.append(m.predict(sc.transform(X.iloc[[t + horizon - 1]]))[0])
        idx.append(y.index[t + horizon - 1])
    return pd.Series(preds, index=idx)
```

- **Expanding window** (alternativ rolling fixed). λ pro Origin via `LassoCV`/`RidgeCV`
  neu wählen (sauber, langsamer) **oder** periodisch (z. B. jährlich) — Trade-off
  dokumentieren.
- Ausgabe: Fehlerreihe → mittlerer RMSE + gleitender RMSE-Plot; optional
  **Diebold-Mariano-Test** auf Signifikanz der Prognoseunterschiede.
→ *Bezug: Kapitel 3 (Evaluationsdesign) + 4 (robuste Ergebnisse). Neue Abbildung „fig_11".*

### AP4 — Elastic Net & mehrere Horizonte *(Priorität: mittel, Aufwand: mittel)*

- **Elastic Net** als L1/L2-Kombination:
  ```python
  ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,.99,1],
               alphas=np.logspace(-3,1,100), cv=tscv, max_iter=10000)
  ```
  In Modellvergleich + Shrinkage-/Pfad-Plots aufnehmen.
- **Horizonte** h ∈ {1, 3, 6, 12}: Schleife über
  `build_feature_matrix(df_yoy, forecast_horizon=h)`; Güte & #selektierte Variablen je
  Horizont tabellieren.
- Ausgabe: Tabelle *RMSE × Horizont × Modell*; Abbildung „fig_12" (RMSE über Horizont).
→ *Bezug: Kapitel 4 + Fazit/Ausblick (Elastic Net als Brücke zu Double ML).*

### AP5 — Diagnostik & Robustheit *(Priorität: mittel, Aufwand: gering–mittel)*

- **Multikollinearität sichtbar machen** (motiviert Ridge): Korrelations-Heatmap *unter
  den Prädiktoren* (aktuell zeigt `fig_02` nur Korrelation *mit y*) bzw. Konditionszahl
  von `XᵀX`. → *Kapitel 2.2 / 3.*
- **Selektionsstabilität:** über die Rolling-Windows zählen, wie häufig jede Variable von
  LASSO gewählt wird → Robustheit der „Top-Prädiktoren". → *Kapitel 4 (Variablenselektion).*

### AP6 — Aufbereitung für die Arbeit *(Priorität: niedrig, Aufwand: gering)*

- Abbildungen in Endqualität exportieren (einheitlicher Stil, ausreichend DPI; ggf. PDF
  für LaTeX). Tabellen nach LaTeX/Markdown exportieren.
- Quellen-/Reihen-Tabelle (Variable → Eurostat/ECB-Code) für den Anhang generieren.

---

## 3b. Phase B — Korrekturen & Schärfung nach Gutachten (AP7–AP12)

> Phase A hat die Empirie *gebaut*; Phase B macht ihre **Aussage korrekt und belastbar**.
> Hebel-Reihenfolge: erst die Kernaussage retten (AP7), dann absichern (AP9, AP8),
> dann Defekte bereinigen (AP10, AP11), zuletzt Robustheit (AP12).

### AP7 — These & Ergebnis-Framing korrigieren *(Priorität: **KRITISCH**, Aufwand: gering)*

Behebt **G1/G8** — der mit Abstand größte inhaltliche Hebel (~1 h, rettet die Kernaussage).

- **Leitfrage neu** (siehe §0): „Schlagen Makro-Prädiktoren + Regularisierung die
  Inflationspersistenz (RW)?" — Antwort der Daten: **nein**.
- **RW (und AR) in jede Ergebnis-Darstellung** als Referenz; README-Headline-Tabelle um
  RW/AR/Elastic Net/LASSO+HVPI ergänzen. Satz **„LASSO ist klar am besten" streichen.**
- **Hauptbefund formulieren:** Regularisierung rettet OLS vor Overfitting (R² −1,41 → 0,75),
  ohne Mehrwert über die Persistenz; **LASSO+HVPI ≈ RW ⇒ Makro-Mehrwert ≈ 0**. Zusätzlich
  betonen: OLS/Ridge/LASSO sind *ohne* HVPI-Eigen-Lags strukturell benachteiligt — der
  beste Einzelprädiktor (die letzte YoY-Rate) fehlt ihnen.
- **README ↔ Notebook synchronisieren** (kanonischer letzter Lauf); **`src/`-Referenz
  entfernen** (Verzeichnis existiert nicht).
→ *Bezug: README; Kapitel 1 (Forschungsfrage), 4 (Ergebnis), 5 (Fazit).*

### AP8 — Einzelfenster vs. Rolling-Origin auflösen *(Priorität: hoch, Aufwand: gering)*

Behebt **G2**.

- Beide Designs nebeneinander darstellen und die Diskrepanz **mechanistisch erklären:**
  das Einzelfenster friert Koeffizienten/λ auf dem Vor-2020-Regime ein und wendet sie auf
  den Energiepreisschock an; Rolling-Origin (Expanding Window) re-schätzt monatlich →
  adaptiert. Daher AR/LASSO+HVPI rolling konkurrenzfähig, im Einzelfenster nicht.
- **Rolling-Origin als Hauptergebnis** führen; Einzelfenster nur als Illustration der
  Regimeabhängigkeit (nicht als Headline-Tabelle).
→ *Bezug: Kapitel 4 (robuste Ergebnisse).*

### AP9 — Signifikanz der Prognoseunterschiede *(Priorität: hoch, Aufwand: mittel)*

Behebt **G3** — ohne diesen Test ist keine Rangfolge zulässig.

- **Diebold-Mariano-Test**, RW als Referenz, je Horizont; quadratische Verlustfunktion;
  Small-Sample-Korrektur nach Harvey-Leybourne-Newbold. `statsmodels` aktivieren (in
  `requirements.txt` bereits auskommentiert) **oder** sklearn-only selbst implementieren:
  ```python
  def diebold_mariano(e1, e2, h=1):
      d = e1**2 - e2**2                      # Verlustdifferenz (RW vs. Modell)
      d_bar = d.mean(); T = len(d)
      gamma = [np.cov(d[:-k], d[k:])[0, 1] for k in range(1, h)] if h > 1 else []
      var = (d.var(ddof=0) + 2*sum(gamma)) / T
      dm = d_bar / np.sqrt(var)
      hln = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)   # HLN-Korrektur
      return dm * hln                         # ~ t_{T-1}
  ```
- **Erwartetes Ergebnis offen kommunizieren:** RW vs. AR/LASSO+HVPI voraussichtlich **nicht
  signifikant** → genau so schreiben („kein nachweisbarer Unterschied zum Random Walk").
→ *Bezug: Kapitel 3 (Evaluationsdesign), 4. requirements.txt: `statsmodels` einschalten.*

### AP10 — Numerische & terminologische Defekte *(Priorität: mittel, Aufwand: gering–mittel)*

Behebt **G4/G5/G7**.

- **G5 Overflow** (Cell 21, LASSO+HVPI) reproduzieren und beheben: Ursache prüfen
  (Near-Singularität durch HVPI-Lags? Spalte mit ~0 Train-Std?), ggf. degenerierte Spalten
  vor dem Skalieren droppen; Warnungen **gezielt** statt pauschal behandeln.
- **G4 h=12-Degeneration:** λ-Gitter prüfen (`np.logspace` ggf. nach unten erweitern). Wenn
  0 selektierte Variablen ökonomisch korrekt sind, als **Befund** schreiben: „kein
  ausnutzbares Makro-Signal auf 12-Monats-Horizont".
- **G7 Begriff:** „AR(p)" → „autoregressives Lag-Modell / ADL (Lags {1,2,3,6,12})".
→ *Bezug: Kapitel 3 (Methodik) / 4 (Ergebnisse).*

### AP11 — Stichprobe verlängern oder Truncation begründen *(Priorität: mittel, Aufwand: gering)*

Behebt **G6** (verschärftes S5).

- `get_raw_data(use_cache=False)` ausführen; IP-/PPI-Reihen frisch laden. Liefert Eurostat
  aktuelle Werte, reicht das Ziel-Sample bis ~2025-12 → **echtes OOS über die Disinflation
  2024–25** (Test auf Generalisierung außerhalb des Trainingsregimes).
- Falls die API kappt: Truncation **explizit begründen** und Konsequenz benennen (Testregime
  = nahezu ausschließlich Energiepreis-Auf-/Abschwung → eingeschränkte externe Validität).
→ *Bezug: Kapitel 3 (Stichprobe), 5 (Limitation).*

### AP12 — Reproduzierbarkeit & Robustheit *(Priorität: niedrig, Aufwand: gering)*

Behebt Rest von **G8**.

- `random_state` dokumentieren/setzen (Schätzer hier deterministisch — explizit erwähnen);
  **einen kanonischen Lauf** festlegen, README/Notebook-Zahlen konsistent halten.
- Optionaler Robustheitscheck: λ je Origin neu wählen (statt eingefroren) → zeigt
  Sensitivität der OOS-Güte gegenüber der λ-Wahl.
→ *Bezug: Kapitel 3 (Methodik/Reproduzierbarkeit).*

---

## 4. Sequenzierung & Meilensteine

**Phase A (Empirie) — abgeschlossen:**

| Reihenfolge | Paket | Liefert | Abhängig von |
|---|---|---|---|
| 1 | AP1 Korrekturen | leakage-freie, korrekt gerahmte Basis | – |
| 2 | AP2 Benchmarks | Einordnung (RW/AR), Makro-Mehrwert | AP1 |
| 3 | AP3 Rolling-Origin | robuste OOS-Güte | AP1, AP2 |
| 4 | AP4 Elastic Net/Horizonte | breiterer Methoden-/Horizontvergleich | AP1 |
| 5 | AP5 Diagnostik | Multikollinearität, Selektionsstabilität | AP3 |
| 6 | AP6 Aufbereitung | finale Tabellen/Abbildungen | alle |

**Phase B (Gutachten-Korrekturen) — offen:**

| Reihenfolge | Paket | Liefert | Abhängig von |
|---|---|---|---|
| 7 | **AP7 Framing/These** | **korrekte Kernaussage, RW-zentriert, README synchron** | Phase A |
| 8 | AP9 Signifikanz (DM) | belastbare Rangfolge bzw. „kein Unterschied" | AP2, AP3 |
| 9 | AP8 Design-Abgleich | Einzelfenster vs. rolling aufgelöst | AP3 |
| 10 | AP10 Defekte | Overflow/h=12/Begriff bereinigt | – |
| 11 | AP11 Stichprobe | verlängertes oder begründetes Sample | – |
| 12 | AP12 Reproduzierbarkeit | konsistente, robuste Zahlen | alle |

Empfehlung Phase A: AP1+AP2 zuerst und zügig (kleiner Aufwand, größter inhaltlicher Hebel),
dann AP3 als Kernstück der „robusten Ergebnisse", AP4/AP5 nach verfügbarer Zeit.

**Empfehlung Phase B: AP7 SOFORT** (kritisch, ~1 h, rettet die Kernaussage), dann **AP9 + AP8**
(machen die Aussage belastbar), danach AP10/AP11, AP12 nach Zeitbudget.

---

## 5. Schreibfahrplan (Code → Arbeit)

Gliederung gemäß `docs/Vorgehensplan…pdf`. Empfohlene **Schreibreihenfolge**:
Kapitel 2 (Theorie) parallel zum Coden → 3 (Daten/Methodik) → 4 (Ergebnisse) →
1 (Einleitung) und 5 (Fazit) zuletzt.

| Kapitel | Inhalt | Material aus |
|---|---|---|
| 1 Einleitung | Motivation (Overfitting, Multikollinearität, p≈n); **Forschungsfrage = Schlagen Makro-Prädiktoren + Regularisierung die Inflationspersistenz (RW)?** | AP7 |
| 2 Theorie | OLS-Grenzen, Ridge (L2), LASSO (L1), λ-Wahl via CV, ML-Bezug | Literatur; `fig_06/07/10` |
| 3 Daten & Methodik | HVPI + Prädiktoren, YoY-Stationarisierung, Lags, Standardisierung, TS-CV, **Evaluationsdesign (Rolling-Origin) + Signifikanz (Diebold-Mariano)**, Leckage-Vermeidung, Quellen-Abweichung Bundesbank→ECB/Eurostat, **Stichproben-Truncation** | `data_loader.py`, `fig_01/02/03`, AP1/AP3/AP9/AP11 |
| 4 Empirische Ergebnisse | **Rolling-Origin als Hauptergebnis, RW/AR als Referenz**; Einzelfenster-vs-rolling-Abgleich; **DM-Signifikanz**; Persistenz-/Makro-Mehrwert-Befund (LASSO+HVPI ≈ RW); Variablenselektion; Horizonte | `results_table.csv`, `horizons_table.csv`, `fig_04/05/08/09/11/12/13`, AP7/AP8/AP9 |
| 5 Fazit | **Korrigierte Kernaussage:** Regularisierung rettet OLS, schlägt aber den RW nicht; Makro-Mehrwert ≈ 0; Einordnung in Atkeson & Ohanian (2001) / Stock & Watson (2007); Limitationen; Ausblick (Elastic Net, Double ML) | AP7/AP9/AP11, Diskussion |
| 6 Literatur | Tibshirani 1996, Hoerl & Kennard 1970, James et al. 2021, Stock & Watson 2007, **Atkeson & Ohanian 2001, Diebold & Mariano 1995** | – |

**Zu adressierende Limitationen (Kapitel 5):** revidierte statt Echtzeit-Datenvintages;
Testregime stark von Energiepreisschock geprägt; Quartals-Lohnkosten als Treppenfunktion;
**keine statistisch signifikanten Prognoseunterschiede zum RW (G3)**; **27 verworfene Monate /
Disinflation 2024–25 ungenutzt (G6)**; **reine Makro-Modelle ohne HVPI-Eigen-Lags strukturell
benachteiligt (G1)**.

---

## 6. Risiken & offene Entscheidungen

- **Narrativ-Falle (kritisch):** Die Arbeit darf **nicht** um die durch die eigenen Daten
  widerlegte These „LASSO gewinnt" geschrieben werden. Verbindliche Kernaussage =
  benchmark-zentriert (RW als Messlatte). Siehe AP7 — vor Beginn des Schreibens umsetzen.
- **Datenvintage:** Es werden revidierte Endstände genutzt (kein Echtzeit-Datensatz) —
  als Limitation benennen, kein Blocker für eine Seminararbeit.
- **Rechenzeit:** Rolling-Origin × 4 Horizonte × λ-Reselektion kann minutenlang laufen.
  Ggf. λ periodisch statt je Origin neu wählen.
- **AR-/DM-Implementierung:** `statsmodels` aufnehmen (AR + Diebold-Mariano) oder bei reinem
  sklearn-Stack bleiben (HVPI-Lags als Features, DM selbst implementiert — Skizze in AP9).
- **Umfang Horizonte:** {1,3,6,12} vs. nur {1,12} — abhängig vom Zeitbudget bis Abgabe.

---

## 7. Definition of Done

**Phase A (Empirie) — abgeschlossen:**
- [x] AP1: kein Leakage, korrektes p/n-Framing, Konvergenz geprüft
- [x] AP2: RW + AR in Tabelle & Plot; relative RMSE; Makro-Mehrwert-Modell
- [x] AP3: Rolling-Origin-RMSE für alle Modelle; gleitender RMSE-Plot
- [x] AP4: Elastic Net + Horizont-Tabelle
- [x] AP5: Kollinearitäts- & Selektionsstabilitäts-Abbildung
- [x] AP6: finale Tabellen/Abbildungen exportiert

**Phase B (Gutachten-Korrekturen) — offen:**
- [x] AP7: These/Framing korrigiert (README-Headline mit allen Modellen + RW/AR, Kernbefunde umgeschrieben, „LASSO am besten" entfernt); Notebook-Abschnitt 6 um schriftliche Interpretation ergänzt; `src/`-Referenz entfernt (2026-06-15)
- [ ] AP8: Einzelfenster vs. Rolling-Origin erklärt; rolling als Hauptergebnis geführt
- [ ] AP9: Diebold-Mariano-Test umgesetzt; (Nicht-)Signifikanz explizit benannt
- [ ] AP10: matmul-Overflow geklärt; h=12-Degeneration dokumentiert; AR-Begriff korrigiert
- [ ] AP11: Stichprobe verlängert oder Truncation belastbar begründet
- [ ] AP12: random_state/Reproduzierbarkeit; README↔Notebook-Zahlen konsistent

**Schreiben:**
- [ ] Text: Kapitel 1–5 mit **korrigierter, benchmark-zentrierter Kernaussage** geschrieben, Abbildungen referenziert, Limitationen benannt
