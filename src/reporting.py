"""Abbildungen, LaTeX/CSV-Export, README-Auto-Sync (querschnittlich)."""
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # noqa: F401 (used via plt.subplots)
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

from .config import (
    AR_LAGS, COLORS, COLORS_OOS, FIGURES_DIR, TEST_MONTHS, TOP_N_STABILITY,
    WINDOW_ROLLING_RMSE,
)


# ── fig_01: HVPI Zeitreihe ────────────────────────────────────────────────────

def fig_01_hvpi(df_raw, df_yoy):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    hvpi_raw = df_raw["HVPI"].dropna()
    axes[0].plot(hvpi_raw.index, hvpi_raw.values, color="#2196F3", linewidth=1.5)
    axes[0].set_title("HVPI Deutschland – Indexniveau (2015 = 100)")
    axes[0].set_ylabel("Index")
    axes[0].axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    hvpi_yoy = df_yoy["HVPI"].dropna()
    axes[1].plot(hvpi_yoy.index, hvpi_yoy.values, color="#E91E63", linewidth=1.5)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].axhline(2, color="#4CAF50", linestyle=":", linewidth=1.2, alpha=0.8,
                    label="EZB-Ziel (2%)")
    axes[1].fill_between(hvpi_yoy.index, hvpi_yoy.values, 0,
                         where=hvpi_yoy.values > 0, alpha=0.15, color="#E91E63")
    axes[1].set_title("HVPI-Inflationsrate Deutschland (YoY, %)")
    axes[1].set_ylabel("Veränderung ggü. Vorjahr (%)")
    axes[1].legend()

    plt.tight_layout()
    _save("fig_01_hvpi_zeitreihe.png")
    plt.show()
    print("Abbildung gespeichert: fig_01_hvpi_zeitreihe.png")


# ── fig_02: Korrelation Prädiktoren mit y ─────────────────────────────────────

def fig_02_correlation(X, y):
    pred_cols_l1 = [c for c in X.columns if c.endswith("_L1")]
    corr_with_y  = (
        X[pred_cols_l1].corrwith(y)
        .rename(lambda c: c.replace("_L1", ""))
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    colors  = ["#E91E63" if v < 0 else "#2196F3" for v in corr_with_y.values]
    ax.barh(corr_with_y.index, corr_with_y.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Korrelation der Prädiktoren (Lag 1) mit HVPI-Inflationsrate")
    ax.set_xlabel("Pearson-Korrelation")
    plt.tight_layout()
    _save("fig_02_korrelation.png")
    plt.show()
    print("Abbildung gespeichert: fig_02_korrelation.png")


# ── fig_02b: Korrelations-Heatmap + Konditionszahl ───────────────────────────

def fig_02b_heatmap(X, train_end):
    _n_test = TEST_MONTHS
    _Xtr    = X.iloc[:-_n_test]
    pred_l1 = _Xtr[[c for c in _Xtr.columns if c.endswith("_L1")]]

    fig, ax = plt.subplots(figsize=(13, 11))
    corr_pred = pred_l1.corr()
    mask      = np.triu(np.ones_like(corr_pred, dtype=bool))
    sns.heatmap(
        corr_pred, mask=mask, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.3, ax=ax,
        xticklabels=[c.replace("_L1", "") for c in pred_l1.columns],
        yticklabels=[c.replace("_L1", "") for c in pred_l1.columns],
    )
    ax.set_title("Korrelationsmatrix der Prädiktoren (Lag 1) — Trainingsset")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    _save("fig_02b_korr_heatmap.png")
    plt.show()
    print("Abbildung gespeichert: fig_02b_korr_heatmap.png")

    _Xtr_s   = StandardScaler().fit_transform(_Xtr)
    cond_XtX = np.linalg.cond(_Xtr_s.T @ _Xtr_s)
    print(f"\nKonditionszahl von X'X (standardisiert): {cond_XtX:.2e}")
    print("→ Werte >> 1 bestätigen starke Multikollinearität und erklären OLS-Instabilität.")


# ── fig_03: TimeSeriesSplit Visualisierung ────────────────────────────────────

def fig_03_tscv(X_train_s, tscv):
    fig, ax = plt.subplots(figsize=(12, 4))
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train_s)):
        ax.scatter(tr_idx, [fold] * len(tr_idx), s=3, color="#2196F3", alpha=0.4)
        ax.scatter(te_idx, [fold] * len(te_idx), s=3, color="#E91E63", alpha=0.8)

    ax.set_xlabel("Beobachtungsindex (Trainingsset)")
    ax.set_ylabel("Fold")
    ax.set_title("TimeSeriesSplit Cross-Validation (Blau=Training, Rot=Test)")
    ax.legend(handles=[
        Patch(color="#2196F3", alpha=0.7, label="Training"),
        Patch(color="#E91E63", alpha=0.9, label="Validation"),
    ])
    plt.tight_layout()
    _save("fig_03_tscv.png")
    plt.show()


# ── fig_04: Prognose-Plot ─────────────────────────────────────────────────────

def fig_04_forecast(ctx):
    y_test              = ctx["y_test"]
    y_pred_rw_test      = ctx["y_pred_rw_test"]
    y_pred_ar_test      = ctx["y_pred_ar_test"]
    y_pred_ols_test     = ctx["y_pred_ols_test"]
    y_pred_ridge_test   = ctx["y_pred_ridge_test"]
    y_pred_lasso_test   = ctx["y_pred_lasso_test"]
    y_pred_enet_test    = ctx["y_pred_enet_test"]
    y_pred_lasso_plus   = ctx["y_pred_lasso_plus_test"]
    rmse_rw_test        = ctx["rmse_rw_test"]
    rmse_ar_test        = ctx["rmse_ar_test"]
    mse_ols_test        = ctx["mse_ols_test"]
    mse_ridge_test      = ctx["mse_ridge_test"]
    mse_lasso_test      = ctx["mse_lasso_test"]
    mse_enet_test       = ctx["mse_enet_test"]
    rmse_lasso_plus_test = ctx["rmse_lasso_plus_test"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.index, y_test.values, "k-", linewidth=2, label="Tatsächlich", zorder=5)
    ax.plot(y_test.index, y_pred_rw_test.values,
            ":", color="#9E9E9E", linewidth=1.8,
            label=f"Random Walk (RMSE={rmse_rw_test:.3f})", alpha=0.9)
    ax.plot(y_pred_ar_test.index, y_pred_ar_test.values,
            "-.", color="#795548", linewidth=1.5,
            label=f"Lag-Modell ADL (RMSE={rmse_ar_test:.3f})", alpha=0.9)
    ax.plot(y_test.index, y_pred_ols_test,
            "--", color=COLORS["OLS"], linewidth=1.3,
            label=f"OLS (RMSE={np.sqrt(mse_ols_test):.3f})", alpha=0.7)
    ax.plot(y_test.index, y_pred_ridge_test,
            "--", color=COLORS["Ridge"], linewidth=1.5,
            label=f"Ridge (RMSE={np.sqrt(mse_ridge_test):.3f})", alpha=0.85)
    ax.plot(y_test.index, y_pred_lasso_test,
            "--", color=COLORS["LASSO"], linewidth=1.5,
            label=f"LASSO (RMSE={np.sqrt(mse_lasso_test):.3f})", alpha=0.85)
    ax.plot(y_test.index, y_pred_enet_test,
            "--", color=COLORS["ElasticNet"], linewidth=1.5,
            label=f"Elastic Net (RMSE={np.sqrt(mse_enet_test):.3f})", alpha=0.85)
    ax.plot(y_pred_lasso_plus.index, y_pred_lasso_plus.values,
            "--", color="#9C27B0", linewidth=1.5,
            label=f"LASSO+HVPI (RMSE={rmse_lasso_plus_test:.3f})", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_title("Prognose vs. tatsächliche HVPI-Inflationsrate (Testset)")
    ax.set_ylabel("HVPI-Inflationsrate (YoY, %)")
    ax.set_xlabel("Datum")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    _save("fig_04_prognose.png")
    plt.show()
    print("Abbildung gespeichert: fig_04_prognose.png")


# ── fig_05: MSE/RMSE-Vergleich ────────────────────────────────────────────────

def fig_05_mse_comparison(ctx):
    mse_rw_test    = ctx["mse_rw_test"]
    mse_ar_test    = ctx["mse_ar_test"]
    mse_ols_test   = ctx["mse_ols_test"]
    mse_ridge_test = ctx["mse_ridge_test"]
    mse_lasso_test = ctx["mse_lasso_test"]
    mse_enet_test  = ctx["mse_enet_test"]
    mse_lasso_plus = ctx["mse_lasso_plus_test"]

    all_models = ["RW", "ADL", "OLS", "Ridge", "LASSO", "Elastic Net", "LASSO+HVPI"]
    mse_vals   = [mse_rw_test, mse_ar_test, mse_ols_test, mse_ridge_test,
                  mse_lasso_test, mse_enet_test, mse_lasso_plus]
    rmse_vals  = [np.sqrt(v) for v in mse_vals]
    colors_bar = ["#9E9E9E", "#795548", COLORS["OLS"], COLORS["Ridge"],
                  COLORS["LASSO"], COLORS["ElasticNet"], "#9C27B0"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(all_models)); width = 0.5

    axes[0].bar(x, mse_vals, width, color=colors_bar, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_models, rotation=20, ha="right")
    axes[0].set_title("Test MSE: Benchmarks vs. Regularisierungsmodelle")
    axes[0].set_ylabel("Mean Squared Error")

    bars = axes[1].bar(x, rmse_vals, width, color=colors_bar, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_models, rotation=20, ha="right")
    axes[1].set_title("Test-RMSE (Prozentpunkte Inflationsrate)")
    axes[1].set_ylabel("RMSE (%)")
    for bar, val in zip(bars, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    _save("fig_05_mse_vergleich.png")
    plt.show()
    print("Abbildung gespeichert: fig_05_mse_vergleich.png")


# ── fig_06: LASSO-Pfad ───────────────────────────────────────────────────────

def fig_06_lasso_path(X_train_s, y_train, lasso_cv, feat_names):
    alphas_path = np.logspace(-3, 1, 80)
    alphas_lasso_path, coefs_lasso, _ = lasso_path(
        X_train_s, y_train, alphas=alphas_path, max_iter=10000
    )
    top_idx    = np.argsort(np.abs(lasso_cv.coef_))[::-1][:15]
    lambda_lasso = lasso_cv.alpha_

    fig, ax = plt.subplots(figsize=(13, 7))
    for i in top_idx:
        ax.semilogx(alphas_lasso_path, coefs_lasso[i, :], linewidth=1.5,
                    label=feat_names[i].replace("_L", " (L").replace("L", "L") + ")")
    ax.axvline(lambda_lasso, color="red", linestyle="--", linewidth=1.5,
               label=f"Opt. λ = {lambda_lasso:.5f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("LASSO-Koeffizientenpfade (Top-15 Features)")
    ax.set_xlabel("Regularisierungsparameter λ (log-Skala)")
    ax.set_ylabel("Koeffizient")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    _save("fig_06_lasso_path.png")
    plt.show()
    print("Abbildung gespeichert: fig_06_lasso_path.png")
    return top_idx


# ── fig_07: Ridge-Pfad ───────────────────────────────────────────────────────

def fig_07_ridge_path(X_train_s, y_train, ridge_cv, top_idx, feat_names):
    from sklearn.linear_model import Ridge as _Ridge

    alphas_ridge_path = np.logspace(-2, 4, 80)
    coefs_ridge_path  = np.array([
        _Ridge(alpha=a).fit(X_train_s, y_train).coef_
        for a in alphas_ridge_path
    ])
    lambda_ridge = ridge_cv.alpha_

    fig, ax = plt.subplots(figsize=(13, 7))
    for i in top_idx:
        ax.semilogx(alphas_ridge_path, coefs_ridge_path[:, i], linewidth=1.5,
                    label=feat_names[i].replace("_L", " (L") + ")")
    ax.axvline(lambda_ridge, color="orange", linestyle="--", linewidth=1.5,
               label=f"Opt. λ = {lambda_ridge:.2f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Ridge-Koeffizientenpfade (Top-15 Features nach LASSO-Rang)")
    ax.set_xlabel("Regularisierungsparameter λ (log-Skala)")
    ax.set_ylabel("Koeffizient")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    _save("fig_07_ridge_path.png")
    plt.show()
    print("Abbildung gespeichert: fig_07_ridge_path.png")


# ── fig_08: LASSO-Selektion ──────────────────────────────────────────────────

def fig_08_lasso_selection(lasso_cv, X):
    lasso_coefs = pd.Series(lasso_cv.coef_, index=X.columns)
    selected    = lasso_coefs[lasso_coefs != 0].sort_values(key=np.abs, ascending=False)
    lambda_lasso = lasso_cv.alpha_

    print(f"LASSO selektiert {len(selected)} von {len(lasso_coefs)} Features:\n")
    print(selected.to_string())

    fig, ax = plt.subplots(figsize=(12, max(5, len(selected) * 0.35)))
    colors_sel = ["#E91E63" if v < 0 else "#2196F3" for v in selected.values]
    ax.barh(range(len(selected)), selected.values, color=colors_sel, alpha=0.85)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"LASSO-Koeffizienten bei opt. λ = {lambda_lasso:.5f} "
                 f"({len(selected)} selektierte Features)")
    ax.set_xlabel("Standardisierter Koeffizient")
    plt.tight_layout()
    _save("fig_08_lasso_selektion.png")
    plt.show()
    print("\nAbbildung gespeichert: fig_08_lasso_selektion.png")


# ── fig_09: LASSO CV-Pfad ────────────────────────────────────────────────────

def fig_09_lasso_cv_path(lasso_cv):
    cv_mses  = np.mean(lasso_cv.mse_path_, axis=1)
    cv_stds  = np.std(lasso_cv.mse_path_, axis=1)
    lambda_lasso = lasso_cv.alpha_

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(lasso_cv.alphas_, cv_mses, color="#4CAF50", linewidth=2,
                label="Mittlerer CV-MSE")
    ax.fill_between(lasso_cv.alphas_, cv_mses - cv_stds, cv_mses + cv_stds,
                    alpha=0.2, color="#4CAF50", label="±1 Std.-Abw.")
    ax.axvline(lambda_lasso, color="red", linestyle="--", linewidth=1.5,
               label=f"Min. λ = {lambda_lasso:.5f}")
    ax.set_title("LASSO Cross-Validation: MSE als Funktion von λ")
    ax.set_xlabel("Regularisierungsparameter λ (log-Skala)")
    ax.set_ylabel("Mittlerer CV-MSE")
    ax.legend()
    plt.tight_layout()
    _save("fig_09_lasso_cv_path.png")
    plt.show()
    print("Abbildung gespeichert: fig_09_lasso_cv_path.png")


# ── fig_10: Shrinkage-Vergleich ──────────────────────────────────────────────

def fig_10_shrinkage(ols, ridge_cv, lasso_cv, enet_cv):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, coef), col in zip(
        axes,
        [("OLS",         ols.coef_),
         ("Ridge",       ridge_cv.coef_),
         ("LASSO",       lasso_cv.coef_),
         ("Elastic Net", enet_cv.coef_)],
        [COLORS["OLS"], COLORS["Ridge"], COLORS["LASSO"], COLORS["ElasticNet"]],
    ):
        ax.scatter(ols.coef_, coef, alpha=0.5, s=15, color=col)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
        lim = max(np.abs(ols.coef_).max(), np.abs(coef).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"Shrinkage: OLS → {name}")
        ax.set_xlabel("OLS-Koeffizient")
        ax.set_ylabel(f"{name}-Koeffizient")
    plt.suptitle("Shrinkage der Koeffizienten (Diagonale = kein Shrinkage)", y=1.02)
    plt.tight_layout()
    _save("fig_10_shrinkage.png")
    plt.show()
    print("Abbildung gespeichert: fig_10_shrinkage.png")


# ── fig_11: Rolling RMSE ─────────────────────────────────────────────────────

def fig_11_rolling_rmse(oos_df, y_oos_ref, oos_rmse):
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in oos_df.columns:
        preds_col = oos_df[col].reindex(y_oos_ref.index).dropna()
        sq_err    = (preds_col - y_oos_ref.loc[preds_col.index]) ** 2
        roll_rmse = sq_err.rolling(WINDOW_ROLLING_RMSE).mean().apply(np.sqrt)
        ax.plot(roll_rmse.index, roll_rmse.values,
                label=f"{col} (Ø {oos_rmse[col]:.3f})",
                color=COLORS_OOS.get(col, "black"), linewidth=1.5)
    ax.set_title(
        f"Gleitender RMSE ({WINDOW_ROLLING_RMSE}-Monats-Fenster) – Rolling-Origin Out-of-Sample"
    )
    ax.set_ylabel("RMSE (Prozentpunkte)")
    ax.set_xlabel("Datum")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("fig_11_rolling_rmse.png")
    plt.show()
    print("Abbildung gespeichert: fig_11_rolling_rmse.png")


# ── fig_12: Selektionsstabilität ─────────────────────────────────────────────

def fig_12_selection_stability(sel_freq, n_windows, lambda_lasso):
    top_vars   = sel_freq.head(TOP_N_STABILITY)
    colors_stab = [
        "#4CAF50" if f >= 0.5 else "#FFC107" if f >= 0.25 else "#9E9E9E"
        for f in top_vars.values
    ]

    fig, ax = plt.subplots(figsize=(12, max(5, TOP_N_STABILITY * 0.35)))
    ax.barh(range(len(top_vars)), top_vars.values, color=colors_stab, alpha=0.85)
    ax.set_yticks(range(len(top_vars)))
    ax.set_yticklabels(top_vars.index, fontsize=9)
    ax.axvline(0.5,  color="#4CAF50", linestyle="--", linewidth=1.2, label="≥50 % (robust)")
    ax.axvline(0.25, color="#FFC107", linestyle="--", linewidth=1.0, alpha=0.8,
               label="≥25 % (gelegentlich)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Auswahlhäufigkeit (Anteil der Rolling-Windows)")
    ax.set_title(
        f"LASSO-Selektionsstabilität: Top-{TOP_N_STABILITY} Variablen\n"
        f"({n_windows} Rolling-Windows, λ = {lambda_lasso:.5f})"
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save("fig_12_selektionsstabilitaet.png")
    plt.show()
    print("Abbildung gespeichert: fig_12_selektionsstabilitaet.png")


# ── fig_13: Horizonte RMSE ───────────────────────────────────────────────────

def fig_13_horizons(df_horizons):
    from .config import HORIZONS

    rmse_cols  = ["RW", "OLS", "Ridge", "LASSO", "Elastic Net"]
    col_colors = {
        "RW": "#9E9E9E", "OLS": COLORS["OLS"], "Ridge": COLORS["Ridge"],
        "LASSO": COLORS["LASSO"], "Elastic Net": COLORS["ElasticNet"],
    }
    markers = {"RW": "o", "OLS": "s", "Ridge": "^", "LASSO": "D", "Elastic Net": "P"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in rmse_cols:
        ax.plot(df_horizons.index, df_horizons[col],
                marker=markers[col], linewidth=1.8, markersize=7,
                color=col_colors[col], label=col)
    ax.set_title("RMSE nach Prognose-Horizont h (Monate voraus)")
    ax.set_xlabel("Prognose-Horizont h (Monate)")
    ax.set_ylabel("RMSE (Prozentpunkte)")
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save("fig_13_horizonte_rmse.png")
    plt.show()
    print("Abbildung gespeichert: fig_13_horizonte_rmse.png")


# ── Tabellen-Exporte ─────────────────────────────────────────────────────────

def export_results_table(results, y_test):
    _MON_DE = {1: "Jan.", 2: "Feb.", 3: "Mär.", 4: "Apr.", 5: "Mai", 6: "Jun.",
               7: "Jul.", 8: "Aug.", 9: "Sep.", 10: "Okt.", 11: "Nov.", 12: "Dez."}
    _t0_tex = f"{_MON_DE[y_test.index[0].month]}~{y_test.index[0].year}"
    _t1_tex = f"{_MON_DE[y_test.index[-1].month]}~{y_test.index[-1].year}"

    results_export = results.rename(columns={
        "Test R²":           r"Test $R^2$",
        "Nicht-Null-Koeff.": r"Koeff.$\neq$0",
    })
    latex_results = results_export.to_latex(
        float_format="%.4f", escape=False,
        caption=(
            "Prognosemodelle im Vergleich: mittlerer RMSE, relatives RMSE (RMSE/RW) "
            f"und $R^2$ im Testset ({_t0_tex}--{_t1_tex})."
        ),
        label="tab:ergebnisse",
    )
    with open("results/results_table.tex", "w") as f:
        f.write(latex_results)
    print("results/results_table.tex gespeichert.")
    print(latex_results)


def export_horizons_table(df_horizons):
    df_hz_export = df_horizons.rename(columns={
        "LASSO Sel.": r"LASSO $\hat{k}$",
        "EN Sel.":    r"EN $\hat{k}$",
    })
    with open("results/horizons_table.tex", "w") as f:
        f.write(df_hz_export.to_latex(
            float_format="%.3f", escape=False,
            caption=(
                r"RMSE je Prognose-Horizont $h \in \{1,3,6,12\}$ Monate. "
                r"$\hat{k}$ = Anzahl nicht-null Koeffizienten."
            ),
            label="tab:horizonte",
        ))
    print("Horizont-Tabelle als LaTeX gespeichert: results/horizons_table.tex")


def export_sources_table():
    from .config import (
        BS_INDICATORS, LCI_SERIES, PROD_SECTORS, PPI_SECTORS, UNEMP_GROUPS,
    )

    rows = [
        {"Variable": "HVPI", "Quelle": "ECB SDW",
         "Datensatz": "ICP/M.DE.N.000000.4.INX",
         "Code / Filter": "DE, 000000 (gesamt), Index 2015=100", "Freq.": "M", "SA": "–"},
    ]
    for name, nace in PROD_SECTORS.items():
        rows.append({"Variable": name, "Quelle": "Eurostat",
                     "Datensatz": "sts_inpr_m",
                     "Code / Filter": f"nace_r2={nace}, unit=I15, geo=DE",
                     "Freq.": "M", "SA": "NSA"})
    for name, indic in BS_INDICATORS.items():
        rows.append({"Variable": name, "Quelle": "Eurostat",
                     "Datensatz": "ei_bsin_m_r2",
                     "Code / Filter": f"indic={indic}, geo=DE",
                     "Freq.": "M", "SA": "SA"})
    for name, nace in PPI_SECTORS.items():
        rows.append({"Variable": name, "Quelle": "Eurostat",
                     "Datensatz": "sts_inppd_m",
                     "Code / Filter": f"nace_r2={nace}, unit=I15, geo=DE",
                     "Freq.": "M", "SA": "NSA"})
    for name, grp in UNEMP_GROUPS.items():
        rows.append({"Variable": name, "Quelle": "Eurostat",
                     "Datensatz": "une_rt_m",
                     "Code / Filter": f"sex={grp['sex']}, age={grp['age']}, unit=PC_ACT, geo=DE",
                     "Freq.": "M", "SA": "SA"})
    for name, flt in LCI_SERIES.items():
        rows.append({"Variable": name, "Quelle": "Eurostat",
                     "Datensatz": "lc_lci_r2_q",
                     "Code / Filter": (
                         f"nace_r2={flt['nace_r2']}, lcstruct={flt['lcstruct']}, "
                         f"unit=I20, geo=DE; Q→M via ffill"
                     ),
                     "Freq.": "Q→M", "SA": "NSA"})

    df_sources = pd.DataFrame(rows)
    df_sources.to_csv("results/sources_table.csv", index=False)
    with open("results/sources_table.tex", "w") as f:
        f.write(df_sources.to_latex(
            index=False, escape=True,
            caption=(
                "Datenquellen: Variable, Quelle, Datensatz-Code und "
                "Saisonbereinigung (SA = saisonbereinigt, NSA = nicht bereinigt)."
            ),
            label="tab:quellen",
        ))
    print(f"Quellentabelle: {len(df_sources)} Reihen → results/sources_table.csv + .tex")
    print(df_sources.to_string(index=False))


# ── README Auto-Sync ─────────────────────────────────────────────────────────

def update_readmes(ctx):
    """Regeneriert den <!-- RESULTS:BEGIN/END --> Block in README.md und README_DE.md."""
    from .config import ROOT, TEST_MONTHS

    def _neg(v, d=2):
        return f"{v:.{d}f}".replace("-", "−")

    y      = ctx["y"]
    y_test = ctx["y_test"]
    X      = ctx["X"]
    X_plus = ctx["X_plus"]

    train_end = ctx["train_end"]
    ro        = ctx["oos_rmse"]

    _n_total = len(y)
    _n_train = train_end
    _n_feat  = X.shape[1]
    _n_plus  = X_plus.shape[1]
    _d0 = y.index[0].strftime("%Y-%m")
    _d1 = y.index[-1].strftime("%Y-%m")
    _t0 = y_test.index[0].strftime("%Y-%m")
    _t1 = y_test.index[-1].strftime("%Y-%m")

    _rw  = ctx["rmse_rw_test"]
    _ar  = ctx["rmse_ar_test"]
    _lp  = ctx["rmse_lasso_plus_test"]
    _las = float(np.sqrt(ctx["mse_lasso_test"]))
    _en  = float(np.sqrt(ctx["mse_enet_test"]))
    _ri  = float(np.sqrt(ctx["mse_ridge_test"]))
    _ols = float(np.sqrt(ctx["mse_ols_test"]))
    _nz_l = ctx["n_nonzero"]

    lasso_plus_alpha = ctx["lasso_plus_cv"].alpha_
    lambda_lasso     = ctx["lambda_lasso"]
    lambda_enet      = ctx["lambda_enet"]
    lambda_ridge     = ctx["lambda_ridge"]
    n_nonzero_plus   = ctx["n_nonzero_plus"]
    n_nonzero_enet   = ctx["n_nonzero_enet"]
    r2_rw_test       = ctx["r2_rw_test"]
    r2_ar_test       = ctx["r2_ar_test"]
    r2_lasso_plus    = ctx["r2_lasso_plus_test"]
    r2_lasso_test    = ctx["r2_lasso_test"]
    r2_enet_test     = ctx["r2_enet_test"]
    r2_ridge_test    = ctx["r2_ridge_test"]
    r2_ols_test      = ctx["r2_ols_test"]

    block_de = (
        f"Datensatz: **{_n_total} Beobachtungen** ({_d0} – {_d1}), "
        f"davon **{_n_train} Training / {TEST_MONTHS} Test**\n"
        f"(Testfenster {_t0} – {_t1}), **{_n_feat} Features**.\n\n"
        f"**Testfenster (fester chronologischer Split), RMSE in Prozentpunkten der Inflationsrate,\n"
        f"sortiert nach Güte:**\n\n"
        f"| Modell | λ | Test-RMSE | RMSE/RW | Test-R² | Koeff. ≠ 0 |\n"
        f"|--------|----------:|----------:|--------:|--------:|-----------:|\n"
        f"| **Random Walk** | –        | **{_rw:.2f}** | **1.00** | {r2_rw_test:.2f} | – |\n"
        f"| Lag-Modell (ADL) | –      | {_ar:.2f} | {_ar/_rw:.2f} | {r2_ar_test:.2f} | {len(AR_LAGS)} |\n"
        f"| LASSO + HVPI-Lags | {lasso_plus_alpha:.3f}  | {_lp:.2f} | {_lp/_rw:.2f} | {r2_lasso_plus:.2f} | {n_nonzero_plus} / {_n_plus} |\n"
        f"| LASSO | {lambda_lasso:.3f}              | {_las:.2f} | {_las/_rw:.2f} | {r2_lasso_test:.2f} | {_nz_l} / {_n_feat} |\n"
        f"| Elastic Net | {lambda_enet:.3f}        | {_en:.2f} | {_en/_rw:.2f} | {r2_enet_test:.2f} | {n_nonzero_enet} / {_n_feat} |\n"
        f"| Ridge | {lambda_ridge:.1f}               | {_ri:.2f} | {_ri/_rw:.2f} | {r2_ridge_test:.2f} | {_n_feat} / {_n_feat} |\n"
        f"| OLS | –                    | {_ols:.2f} | {_ols/_rw:.2f} | {_neg(r2_ols_test)} | {_n_feat} / {_n_feat} |\n\n"
        f"**Robustheitscheck (Rolling-Origin, Expanding Window):** "
        f"RW {ro['RW']:.2f} · AR {ro['AR']:.2f} · LASSO+HVPI {ro['LASSO+HVPI']:.2f} ·\n"
        f"LASSO {ro['LASSO']:.2f} · Elastic Net {ro['Elastic Net']:.2f} · "
        f"Ridge {ro['Ridge']:.2f} · OLS {ro['OLS']:.2f}. "
        f"Die adaptiven Modelle (AR, LASSO+HVPI)\n"
        f"erreichen den RW hier knapp, schlagen ihn aber nicht nachweisbar "
        f"(Diebold-Mariano n.s.)."
    )

    block_en = (
        f"Dataset: **{_n_total} observations** ({_d0} – {_d1}), "
        f"of which **{_n_train} training / {TEST_MONTHS} test**\n"
        f"(test window {_t0} – {_t1}), **{_n_feat} features**.\n\n"
        f"**Test window (fixed chronological split), RMSE in percentage points of the inflation rate,\n"
        f"sorted by performance:**\n\n"
        f"| Model | λ | Test RMSE | RMSE/RW | Test R² | Coeff. ≠ 0 |\n"
        f"|-------|----------:|----------:|--------:|--------:|-----------:|\n"
        f"| **Random Walk** | –        | **{_rw:.2f}** | **1.00** | {r2_rw_test:.2f} | – |\n"
        f"| Lag model (ADL) | –      | {_ar:.2f} | {_ar/_rw:.2f} | {r2_ar_test:.2f} | {len(AR_LAGS)} |\n"
        f"| LASSO + HICP lags | {lasso_plus_alpha:.3f}  | {_lp:.2f} | {_lp/_rw:.2f} | {r2_lasso_plus:.2f} | {n_nonzero_plus} / {_n_plus} |\n"
        f"| LASSO | {lambda_lasso:.3f}              | {_las:.2f} | {_las/_rw:.2f} | {r2_lasso_test:.2f} | {_nz_l} / {_n_feat} |\n"
        f"| Elastic Net | {lambda_enet:.3f}        | {_en:.2f} | {_en/_rw:.2f} | {r2_enet_test:.2f} | {n_nonzero_enet} / {_n_feat} |\n"
        f"| Ridge | {lambda_ridge:.1f}               | {_ri:.2f} | {_ri/_rw:.2f} | {r2_ridge_test:.2f} | {_n_feat} / {_n_feat} |\n"
        f"| OLS | –                    | {_ols:.2f} | {_ols/_rw:.2f} | {_neg(r2_ols_test)} | {_n_feat} / {_n_feat} |\n\n"
        f"**Robustness check (Rolling-Origin, expanding window):** "
        f"RW {ro['RW']:.2f} · AR {ro['AR']:.2f} · LASSO+HICP {ro['LASSO+HVPI']:.2f} ·\n"
        f"LASSO {ro['LASSO']:.2f} · Elastic Net {ro['Elastic Net']:.2f} · "
        f"Ridge {ro['Ridge']:.2f} · OLS {ro['OLS']:.2f}. "
        f"The adaptive models (AR, LASSO+HICP)\n"
        f"nearly match the RW here, but do not beat it significantly (Diebold-Mariano n.s.)."
    )

    _MARKER = re.compile(
        r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", re.DOTALL
    )
    for fpath, block in [
        (ROOT / "README_DE.md", block_de),
        (ROOT / "README.md",    block_en),
    ]:
        txt = fpath.read_text(encoding="utf-8")
        new = _MARKER.sub(
            f"<!-- RESULTS:BEGIN -->\n{block}\n<!-- RESULTS:END -->", txt
        )
        assert new != txt or block in txt, f"{fpath}: Marker nicht gefunden"
        fpath.write_text(new, encoding="utf-8")
        print(f"{fpath} ✓")

    print("\nREADME auto-sync abgeschlossen.")
    print(f"  {_n_total} Beob. ({_d0}–{_d1}), {_n_feat} Features, "
          f"Test {_t0}–{_t1}")
    print(f"  RW {_rw:.4f}  |  LASSO {_las:.4f}  |  OLS {_ols:.4f}")


# ── Zusammenfassung ──────────────────────────────────────────────────────────

def print_summary(ctx):
    """Druckt die Zusammenfassung (entspricht Cell 51 im Originalnotebook)."""
    X       = ctx["X"]
    y       = ctx["y"]
    y_test  = ctx["y_test"]
    results = ctx["results"]
    selected = ctx["selected"]

    print("=" * 75)
    print("ZUSAMMENFASSUNG DER ERGEBNISSE")
    print("=" * 75)
    print(f"Datensatz:  {X.shape[0]} Monate, {X.shape[1]} Features")
    print(f"Zeitraum:   {y.index[0].strftime('%Y-%m')} – {y.index[-1].strftime('%Y-%m')}")
    print(f"Test-Split: {len(y_test)} Monate (chronologisch)")
    print()
    print(f"{'Modell':<14} {'Test-RMSE':>10} {'RMSE/RW':>9} {'Test-R²':>10} {'Koeff.≠0':>10}")
    print("-" * 58)

    rmse_rw = ctx["rmse_rw_test"]
    for name, rmse_val, r2_val, nz in [
        ("Random Walk",  ctx["rmse_rw_test"],              ctx["r2_rw_test"],    "-"),
        ("ADL",          ctx["rmse_ar_test"],              ctx["r2_ar_test"],    str(len(AR_LAGS))),
        ("OLS",          np.sqrt(ctx["mse_ols_test"]),     ctx["r2_ols_test"],   str(int(np.sum(ctx["ols"].coef_ != 0)))),
        ("Ridge",        np.sqrt(ctx["mse_ridge_test"]),   ctx["r2_ridge_test"], str(len(ctx["ridge_cv"].coef_))),
        ("LASSO",        np.sqrt(ctx["mse_lasso_test"]),   ctx["r2_lasso_test"], str(ctx["n_nonzero"])),
        ("Elastic Net",  np.sqrt(ctx["mse_enet_test"]),    ctx["r2_enet_test"],  str(ctx["n_nonzero_enet"])),
        ("LASSO+HVPI",   ctx["rmse_lasso_plus_test"],      ctx["r2_lasso_plus_test"], str(ctx["n_nonzero_plus"])),
        ("Adapt. LASSO", ctx["rmse_alasso_test"],          ctx["r2_alasso_test"], str(ctx["n_nonzero_alasso"])),
    ]:
        rel = "1.000 (Ref)" if name == "Random Walk" else f"{rmse_val/rmse_rw:.3f}"
        print(f"{name:<14} {rmse_val:>10.4f} {rel:>9} {r2_val:>10.4f} {nz:>10}")
    print("=" * 75)

    best = results["Test RMSE"].astype(float).idxmin()
    print(f"\nBestes Modell nach Test-RMSE: {best}")
    print()
    print("Selektierte Variablengruppen (LASSO):")
    for grp, prefix in [
        ("Industrieproduktion", "IP_"),
        ("Business Surveys",    "BS_"),
        ("Produzentenpreise",   "PPI_"),
        ("Arbeitsmarkt",        "ALQ_"),
        ("Lohnkosten",          "LCI_"),
    ]:
        sel_grp = [c for c in selected.index if prefix in c]
        if sel_grp:
            print(f"  {grp}: {len(sel_grp)} Feature(s) – {sel_grp}")


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _save(filename):
    plt.savefig(FIGURES_DIR / filename, bbox_inches="tight")
