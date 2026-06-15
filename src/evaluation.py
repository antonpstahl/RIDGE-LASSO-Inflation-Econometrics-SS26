"""Stage 4: Rolling-Origin OOS, Diebold-Mariano, Selektion, Horizonte, Stationaritaet."""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import (
    ALPHAS_LASSO, ALPHAS_LASSO_INNER, ALPHAS_RIDGE, ALPHAS_RIDGE_INNER,
    COLORS_OOS, HORIZONS, L1_RATIOS_ENET, L1_RATIOS_ENET_INNER,
    LAGS, TEST_MONTHS, TSCV, TSCV_INNER, WINDOW_ROLLING_RMSE,
)
from .data_preprocessing import build_feature_matrix
from .models import AdaptiveLasso


# ── Rolling-Origin ────────────────────────────────────────────────────────────

def rolling_origin(model_factory, X, y, start):
    """Expanding-Window Rolling-Origin Prognose.

    Parameters
    ----------
    model_factory : callable, () → sklearn estimator
    X, y          : vollstaendige Feature-Matrix / Zielvariable
    start         : erster OOS-Index (trainiert auf [0:start], prognostiziert [start])
    """
    preds, idx = [], []
    for t in range(start, len(y)):
        Xtr, ytr = X.iloc[:t], y.iloc[:t]
        sc = StandardScaler().fit(Xtr)
        m  = model_factory().fit(sc.transform(Xtr), ytr)
        preds.append(m.predict(sc.transform(X.iloc[[t]]))[0])
        idx.append(y.index[t])
    return pd.Series(preds, index=idx)


# ── Diebold-Mariano ───────────────────────────────────────────────────────────

def diebold_mariano(e_rw, e_mod, h=1):
    """DM-Test (quadr. Verlust), HLN-korrigiert; zweiseitiger p-Wert via t(T-1).

    Verlustdifferenz d_t = e_RW^2 - e_M^2 (positiv → Modell M schlägt RW).
    """
    d     = np.asarray(e_rw) ** 2 - np.asarray(e_mod) ** 2
    T     = len(d)
    d_bar = d.mean()
    var_d = d.var(ddof=0)
    for k in range(1, h):
        var_d += 2 * np.cov(d[:-k], d[k:], ddof=0)[0, 1]
    var_d  = max(var_d / T, 1e-15)
    dm_raw = d_bar / np.sqrt(var_d)
    hln    = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm_raw * hln
    p_val  = 2 * (1 - sp_stats.t.cdf(abs(dm_hln), df=T - 1))
    return dm_hln, p_val


# ── OOS-Prognosen (festes λ) ──────────────────────────────────────────────────

def compute_oos_predictions(models_ctx, splits, X, y, train_end):
    """Berechnet Rolling-Origin-Prognosen mit festen Hyperparametern (schnell)."""
    lambda_lasso  = models_ctx["lambda_lasso"]
    lambda_ridge  = models_ctx["lambda_ridge"]
    lambda_enet   = models_ctx["lambda_enet"]
    l1_ratio_enet = models_ctx["l1_ratio_enet"]
    lasso_plus_alpha = models_ctx["lasso_plus_cv"].alpha_

    X_ar     = splits["X_ar"];     y_ar     = splits["y_ar"]
    X_plus   = splits["X_plus"];   y_plus   = splits["y_plus"]
    start_ar   = splits["start_ar"]
    start_plus = splits["start_plus"]
    y_test     = splits["y_test"]

    # Random Walk
    oos_rw = y.shift(1).iloc[train_end:].rename("RW")

    # Lag-Modell (ADL)
    oos_ar = rolling_origin(
        lambda: LinearRegression(), X_ar, y_ar, start_ar
    ).rename("AR")

    # OLS
    oos_ols = rolling_origin(
        lambda: LinearRegression(), X, y, train_end
    ).rename("OLS")

    # Ridge (festes λ)
    oos_ridge = rolling_origin(
        lambda: Ridge(alpha=lambda_ridge), X, y, train_end
    ).rename("Ridge")

    # LASSO (festes λ)
    oos_lasso = rolling_origin(
        lambda: Lasso(alpha=lambda_lasso, max_iter=10000), X, y, train_end
    ).rename("LASSO")

    # Elastic Net (feste Hyperparameter)
    oos_enet = rolling_origin(
        lambda: ElasticNet(alpha=lambda_enet, l1_ratio=l1_ratio_enet, max_iter=10000),
        X, y, train_end,
    ).rename("Elastic Net")

    # LASSO+HVPI (festes λ)
    oos_lasso_plus = rolling_origin(
        lambda: Lasso(alpha=lasso_plus_alpha, max_iter=10000),
        X_plus, y_plus, start_plus,
    ).rename("LASSO+HVPI")

    print("Rolling-Origin-Prognosen berechnet (alle Modelle inkl. Elastic Net).")

    oos_df    = pd.concat(
        [oos_rw, oos_ar, oos_ols, oos_ridge, oos_lasso, oos_enet, oos_lasso_plus], axis=1
    )
    y_oos_ref = y.loc[y_test.index]

    oos_rmse = {}
    print("Rolling-Origin RMSE (Expanding Window, h=1, λ fest aus initialem CV):")
    print("-" * 65)
    for col in oos_df.columns:
        preds_col  = oos_df[col].reindex(y_oos_ref.index).dropna()
        actual_col = y_oos_ref.loc[preds_col.index]
        oos_rmse[col] = np.sqrt(mean_squared_error(actual_col, preds_col))

    rw_rmse = oos_rmse["RW"]
    for col, rmse in oos_rmse.items():
        rel = "1.000 (Ref)" if col == "RW" else f"{rmse/rw_rmse:.3f}"
        print(f"  {col:<14}: RMSE = {rmse:.4f}   RMSE/RW = {rel}")

    return dict(
        oos_rw=oos_rw, oos_ar=oos_ar, oos_ols=oos_ols, oos_ridge=oos_ridge,
        oos_lasso=oos_lasso, oos_enet=oos_enet, oos_lasso_plus=oos_lasso_plus,
        oos_df=oos_df, y_oos_ref=y_oos_ref, oos_rmse=oos_rmse,
    )


# ── Adaptive Rolling-Origin (λ je Origin neu via CV) ─────────────────────────

def compute_adaptive_oos(X, y, splits, train_end, tscv_inner=None):
    """Adaptive Rolling-Origin: λ wird je Origin neu per CV bestimmt (~10-20 min)."""
    if tscv_inner is None:
        tscv_inner = TSCV_INNER

    X_plus   = splits["X_plus"]
    y_plus   = splits["y_plus"]
    start_plus = splits["start_plus"]

    print("Starte adaptive Rolling-Origin (λ je Origin via CV) …")
    print("(Laufzeit ~10–20 min — Fortschritt wird nicht angezeigt)")

    oos_lasso_adap = rolling_origin(
        lambda: LassoCV(
            alphas=ALPHAS_LASSO_INNER, cv=tscv_inner, max_iter=10000, n_jobs=-1
        ), X, y, train_end
    ).rename("LASSO (adapt.)")

    oos_ridge_adap = rolling_origin(
        lambda: RidgeCV(alphas=ALPHAS_RIDGE_INNER, cv=tscv_inner),
        X, y, train_end,
    ).rename("Ridge (adapt.)")

    oos_enet_adap = rolling_origin(
        lambda: ElasticNetCV(
            l1_ratio=L1_RATIOS_ENET_INNER,
            alphas=ALPHAS_LASSO_INNER,
            cv=tscv_inner, max_iter=10000, n_jobs=-1,
        ), X, y, train_end
    ).rename("Elastic Net (adapt.)")

    oos_lasso_plus_adap = rolling_origin(
        lambda: LassoCV(
            alphas=ALPHAS_LASSO_INNER, cv=tscv_inner, max_iter=10000, n_jobs=-1
        ), X_plus, y_plus, start_plus
    ).rename("LASSO+HVPI (adapt.)")

    oos_alasso_adap = rolling_origin(
        lambda: AdaptiveLasso(
            alphas=ALPHAS_LASSO_INNER, cv=tscv_inner, max_iter=10000
        ), X, y, train_end
    ).rename("Adaptive LASSO (adapt.)")

    print("Fertig.")
    return dict(
        oos_lasso_adap=oos_lasso_adap,
        oos_ridge_adap=oos_ridge_adap,
        oos_enet_adap=oos_enet_adap,
        oos_lasso_plus_adap=oos_lasso_plus_adap,
        oos_alasso_adap=oos_alasso_adap,
    )


def compute_compare_oos(oos_ctx, adap_ctx, y_oos_ref):
    """Zusammenfassung: festes λ vs. adaptives λ."""
    compare_oos = pd.concat([
        oos_ctx["oos_rw"],           oos_ctx["oos_ar"],
        oos_ctx["oos_lasso"],        adap_ctx["oos_lasso_adap"],
        oos_ctx["oos_ridge"],        adap_ctx["oos_ridge_adap"],
        oos_ctx["oos_enet"],         adap_ctx["oos_enet_adap"],
        oos_ctx["oos_lasso_plus"],   adap_ctx["oos_lasso_plus_adap"],
        adap_ctx["oos_alasso_adap"],
    ], axis=1)

    rw_rmse_ref = oos_ctx["oos_rmse"]["RW"]
    adap_rmse = {}
    print("\nRolling-Origin RMSE: festes λ vs. adaptives λ je Origin")
    print(f"{'Modell':<25} {'RMSE':>7}  {'RMSE/RW':>8}")
    print("-" * 45)
    for col in compare_oos.columns:
        p    = compare_oos[col].reindex(y_oos_ref.index).dropna()
        a    = y_oos_ref.loc[p.index]
        rmse = np.sqrt(mean_squared_error(a, p))
        adap_rmse[col] = rmse
        marker = " ◀ adapt." if "(adapt.)" in col else ""
        print(f"  {col:<23} {rmse:>7.4f}  {rmse/rw_rmse_ref:>8.4f}{marker}")

    return dict(compare_oos=compare_oos, adap_rmse=adap_rmse)


# ── Diebold-Mariano-Tests ─────────────────────────────────────────────────────

def compute_dm_tests(oos_ctx):
    """DM-Test je Modell gegen Random Walk (HLN-korrigiert, T=36)."""
    oos_df    = oos_ctx["oos_df"]
    y_oos_ref = oos_ctx["y_oos_ref"]

    y_ref     = y_oos_ref.loc[oos_df.index.intersection(y_oos_ref.index)]
    e_rw_ro   = (oos_ctx["oos_rw"].reindex(y_ref.index) - y_ref).dropna()

    dm_records = []
    print("Diebold-Mariano-Test (Referenz: Random Walk, h=1, HLN-Korrektur, T=36)")
    print(f"{'Modell':<15} {'DM-Stat':>9} {'p-Wert':>9} {'Signifikanz':>12}")
    print("-" * 50)
    for col in ["AR", "LASSO+HVPI", "LASSO", "Elastic Net", "Ridge", "OLS"]:
        preds  = oos_df[col].reindex(y_ref.index).dropna()
        e_mod  = (preds - y_ref.loc[preds.index]).dropna()
        e_rw_a = e_rw_ro.loc[e_mod.index]
        dm, pv = diebold_mariano(e_rw_a.values, e_mod.values, h=1)
        sig    = "**" if pv < 0.05 else ("*" if pv < 0.10 else "n.s.")
        dm_records.append({
            "Modell": col, "DM-Stat": round(dm, 3),
            "p-Wert": round(pv, 4), "Sig.": sig,
        })
        print(f"  {col:<13} {dm:>+9.3f} {pv:>9.4f} {sig:>12}")

    print("-" * 50)
    print("DM > 0: Modell schlägt Random Walk (niedr. Verlust)  | * p<0.10  ** p<0.05")
    dm_df = pd.DataFrame(dm_records).set_index("Modell")
    return {"dm_df": dm_df}


# ── Selektionsstabilität ──────────────────────────────────────────────────────

def compute_selection_stability(X, y, train_end, lambda_lasso):
    """Zaehlt, wie oft LASSO je Variable ueber alle Rolling-Windows selektiert."""
    from sklearn.linear_model import Lasso

    selection_counts = np.zeros(X.shape[1])
    for t in range(train_end, len(y)):
        Xtr = X.iloc[:t]
        sc  = StandardScaler().fit(Xtr)
        m   = Lasso(alpha=lambda_lasso, max_iter=10000).fit(sc.transform(Xtr), y.iloc[:t])
        selection_counts += (m.coef_ != 0).astype(int)

    n_windows = len(y) - train_end
    sel_freq  = pd.Series(selection_counts / n_windows, index=X.columns)
    sel_freq  = sel_freq[sel_freq > 0].sort_values(ascending=False)

    print(f"Variablen selektiert in ≥1 Fenster:    {len(sel_freq)}")
    print(f"Variablen selektiert in ≥50 % Fenster: {(sel_freq >= 0.5).sum()}")
    print(f"\nTop-15 nach Auswahlhäufigkeit:")
    print(sel_freq.head(15).to_string())

    return {"sel_freq": sel_freq, "n_windows": n_windows}


# ── Horizont-Analyse ──────────────────────────────────────────────────────────

def compute_horizon_analysis(df_yoy, tscv=None):
    """RMSE je Horizont h ∈ {1, 3, 6, 12} (neue CV je Horizont)."""
    if tscv is None:
        tscv = TSCV

    from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV

    horizon_records = []
    print(f"{'h':>3}  {'RW':>7}  {'OLS':>7}  {'Ridge':>7}  {'LASSO':>7} {'(sel)':>5}"
          f"  {'EN':>7} {'(sel)':>5}")
    print("-" * 65)

    for h in HORIZONS:
        Xh, yh = build_feature_matrix(
            df_yoy, lags=LAGS, forecast_horizon=h, test_months=TEST_MONTHS
        )
        te_h            = len(yh) - TEST_MONTHS
        Xtr_h, Xte_h   = Xh.iloc[:te_h], Xh.iloc[te_h:]
        ytr_h, yte_h   = yh.iloc[:te_h], yh.iloc[te_h:]
        sc_h            = StandardScaler().fit(Xtr_h)
        Xtr_hs = sc_h.transform(Xtr_h)
        Xte_hs = sc_h.transform(Xte_h)

        # Random Walk (h-Schritt)
        y_rw_h    = yh.shift(h).reindex(yte_h.index).dropna()
        rmse_rw_h = np.sqrt(mean_squared_error(yte_h.loc[y_rw_h.index], y_rw_h))

        # OLS
        ols_h      = LinearRegression().fit(Xtr_hs, ytr_h)
        rmse_ols_h = np.sqrt(mean_squared_error(yte_h, ols_h.predict(Xte_hs)))

        # Ridge
        ridge_h      = RidgeCV(alphas=ALPHAS_RIDGE, cv=tscv).fit(Xtr_hs, ytr_h)
        rmse_ridge_h = np.sqrt(mean_squared_error(yte_h, ridge_h.predict(Xte_hs)))

        # LASSO
        lasso_h = LassoCV(
            alphas=ALPHAS_LASSO, cv=tscv, max_iter=10000, n_jobs=-1
        ).fit(Xtr_hs, ytr_h)
        rmse_lasso_h = np.sqrt(mean_squared_error(yte_h, lasso_h.predict(Xte_hs)))
        nsel_lasso_h = int(np.sum(lasso_h.coef_ != 0))

        # Elastic Net
        enet_h = ElasticNetCV(
            l1_ratio=L1_RATIOS_ENET_INNER, alphas=ALPHAS_LASSO,
            cv=tscv, max_iter=10000, n_jobs=-1,
        ).fit(Xtr_hs, ytr_h)
        rmse_enet_h = np.sqrt(mean_squared_error(yte_h, enet_h.predict(Xte_hs)))
        nsel_enet_h = int(np.sum(enet_h.coef_ != 0))

        horizon_records.append({
            "Horizont h": h,
            "RW": rmse_rw_h,    "OLS": rmse_ols_h,
            "Ridge": rmse_ridge_h,
            "LASSO": rmse_lasso_h, "LASSO Sel.": nsel_lasso_h,
            "Elastic Net": rmse_enet_h, "EN Sel.": nsel_enet_h,
        })
        print(f"h={h:2d}: RW={rmse_rw_h:.3f}  OLS={rmse_ols_h:.3f}  "
              f"Ridge={rmse_ridge_h:.3f}  LASSO={rmse_lasso_h:.3f} "
              f"({nsel_lasso_h:3d})  EN={rmse_enet_h:.3f} ({nsel_enet_h:3d})")

    df_horizons = pd.DataFrame(horizon_records).set_index("Horizont h")

    # h=12-Degeneration dokumentieren (G4)
    for rec in horizon_records:
        if rec["LASSO Sel."] == 0 or rec["EN Sel."] == 0:
            h_deg = rec["Horizont h"]
            print(f"\nBEFUND (G4): Bei h={h_deg} selektiert LASSO {rec['LASSO Sel.']} und "
                  f"Elastic Net {rec['EN Sel.']} Variablen (reiner Intercept).")
            print("Interpretation: Kein ausnutzbares Makro-Signal auf Jahreshorizont "
                  "(λ-Pfad bevorzugt Nulllösung). RMSE identisch → Befund, kein Bug.")

    print(df_horizons.to_string())
    df_horizons.to_csv("results/horizons_table.csv")
    print("\nHorizont-Tabelle gespeichert: results/horizons_table.csv")

    return {"df_horizons": df_horizons}


# ── Stationaritätstests (ADF + KPSS) ─────────────────────────────────────────

# Repräsentative Prädiktoren je Gruppe (Niveau-Spaltenname im Rohdata-Frame)
_STATIONARITY_SERIES = {
    "HVPI":                    "HVPI",
    "IP (Verarb. Gew.)":       "IP_Verarbeitendes_Gew",
    "PPI (Gesamt)":            "PPI_Gesamt",
    "BS (Konjunkturklima)":    "BS_Konjunkturklima",
    "ALQ (Gesamt)":            "ALQ_Gesamt",
    "LCI (Lohnkosten BN)":     "LCI_Lohnkosten_BN",
}


def compute_stationarity_tests(df_raw, df_yoy):
    """ADF- und KPSS-Test auf Niveau- und YoY-Reihen (Stufe 4 – Diagnostik).

    Testet für jede Reihe in _STATIONARITY_SERIES sowohl das Niveau als auch
    die YoY-Transformierte und gibt einen kompakten DataFrame zurück.

    ADF  H0: Einheitswurzel (nicht-stationär) → Verwerfung belegt Stationarität.
    KPSS H0: Stationarität              → Nicht-Verwerfung belegt Stationarität.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    records = []
    for label, col in _STATIONARITY_SERIES.items():
        for transform, series_src in [("Niveau", df_raw), ("YoY (%)", df_yoy)]:
            if col not in series_src.columns:
                continue
            s = series_src[col].dropna()
            if len(s) < 20:
                continue

            # ADF (maxlag=None → Schwert-Formel; regression='c' = Konstante)
            adf_stat, adf_p, _, _, adf_crit, _ = adfuller(s, regression="c", autolag="AIC")
            adf_reject = bool(adf_p < 0.05)

            # KPSS (regression='c' = Level-Stationarität; nlags='auto')
            # InterpolationWarning bei Randwerten (p<0.01 oder p>0.10) ist erwartet.
            try:
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    kpss_stat, kpss_p, _, kpss_crit = kpss(s, regression="c", nlags="auto")
                kpss_reject = bool(kpss_p < 0.05)
            except Exception:
                kpss_stat, kpss_p, kpss_reject = np.nan, np.nan, None

            # Gemeinsames Urteil: stationär wenn ADF verwirft UND KPSS nicht verwirft
            if adf_reject and (kpss_reject is False):
                verdict = "stationär"
            elif (not adf_reject) and (kpss_reject is True):
                verdict = "nicht-stationär"
            else:
                verdict = "unklar/persistent"

            records.append({
                "Reihe":       label,
                "Transform.":  transform,
                "ADF-Stat.":   round(adf_stat, 3),
                "ADF p-Wert":  round(float(adf_p), 4),
                "ADF Urteil":  "I(0)" if adf_reject else "I(1)?",
                "KPSS-Stat.":  round(float(kpss_stat), 3) if not np.isnan(kpss_stat) else "–",
                "KPSS p-Wert": round(float(kpss_p), 4)   if not np.isnan(kpss_p)   else "–",
                "KPSS Urteil": "I(0)" if (kpss_reject is False) else ("I(1)?" if kpss_reject else "–"),
                "Gesamt":      verdict,
            })

    df_stat = pd.DataFrame(records)
    print("\nStationaritätstests (ADF & KPSS)")
    print("=" * 75)
    print(df_stat.to_string(index=False))
    print()
    print("ADF: H0 = Einheitswurzel; Verwerfung (p<0.05) → stationär.")
    print("KPSS: H0 = Stationarität; Nicht-Verwerfung (p≥0.05) → stationär.")
    print()
    n_stat = (df_stat["Gesamt"] == "stationär").sum()
    n_ni   = (df_stat["Gesamt"] == "nicht-stationär").sum()
    n_unk  = df_stat["Gesamt"].str.startswith("unklar").sum()
    print(f"Urteil: {n_stat} stationär, {n_ni} nicht-stationär, {n_unk} unklar/persistent.")
    print("Hinweis: HVPI-YoY zeigt hohe Persistenz (nahe I(1)) — konsistent mit")
    print("der Literatur zur Inflationsdynamik (Stock & Watson 2007). Die YoY-")
    print("Transformation verringert die Persistenz gegenüber dem Niveau klar,")
    print("ist aber bei kurzen OOS-Fenstern kein Garant für vollständige Stationarität.")
    return {"df_stationarity": df_stat}
