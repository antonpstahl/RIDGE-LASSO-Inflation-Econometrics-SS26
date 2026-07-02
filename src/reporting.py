"""Figures, LaTeX/CSV export, README auto-sync (cross-cutting)."""
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


# --- fig_01: HICP time series ---

def fig_01_hvpi(df_raw, df_yoy):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    hvpi_raw = df_raw["HVPI"].dropna()
    axes[0].plot(hvpi_raw.index, hvpi_raw.values, color="#2196F3", linewidth=1.5)
    axes[0].set_title("HICP Germany - index level (2015 = 100)")
    axes[0].set_ylabel("Index")
    axes[0].axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    hvpi_yoy = df_yoy["HVPI"].dropna()
    axes[1].plot(hvpi_yoy.index, hvpi_yoy.values, color="#E91E63", linewidth=1.5)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].axhline(2, color="#4CAF50", linestyle=":", linewidth=1.2, alpha=0.8,
                    label="ECB target (2%)")
    axes[1].fill_between(hvpi_yoy.index, hvpi_yoy.values, 0,
                         where=hvpi_yoy.values > 0, alpha=0.15, color="#E91E63")
    axes[1].set_title("HICP inflation rate Germany (YoY, %)")
    axes[1].set_ylabel("Year-over-year change (%)")
    axes[1].legend()

    plt.tight_layout()
    _save("fig_01_hvpi_zeitreihe.png")
    plt.show()
    print("Figure saved: fig_01_hvpi_zeitreihe.png")


# --- fig_02: correlation of predictors with y ---

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
    ax.set_title("Correlation of predictors (lag 1) with HICP inflation rate")
    ax.set_xlabel("Pearson correlation")
    plt.tight_layout()
    _save("fig_02_korrelation.png")
    plt.show()
    print("Figure saved: fig_02_korrelation.png")


# --- fig_02b: correlation heatmap + condition number ---

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
    ax.set_title("Correlation matrix of predictors (lag 1) - training set")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    _save("fig_02b_korr_heatmap.png")
    plt.show()
    print("Figure saved: fig_02b_korr_heatmap.png")

    _Xtr_s   = StandardScaler().fit_transform(_Xtr)
    cond_XtX = np.linalg.cond(_Xtr_s.T @ _Xtr_s)
    print(f"\nCondition number of X'X (standardised): {cond_XtX:.2e}")
    print("-> Values >> 1 confirm strong multicollinearity and explain OLS instability.")


# --- fig_03: TimeSeriesSplit visualisation ---

def fig_03_tscv(X_train_s, tscv):
    fig, ax = plt.subplots(figsize=(12, 4))
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train_s)):
        ax.scatter(tr_idx, [fold] * len(tr_idx), s=3, color="#2196F3", alpha=0.4)
        ax.scatter(te_idx, [fold] * len(te_idx), s=3, color="#E91E63", alpha=0.8)

    ax.set_xlabel("Observation index (training set)")
    ax.set_ylabel("Fold")
    ax.set_title("TimeSeriesSplit cross-validation (blue=training, red=test)")
    ax.legend(handles=[
        Patch(color="#2196F3", alpha=0.7, label="Training"),
        Patch(color="#E91E63", alpha=0.9, label="Validation"),
    ])
    plt.tight_layout()
    _save("fig_03_tscv.png")
    plt.show()


# --- fig_04: forecast plot ---

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
    ax.plot(y_test.index, y_test.values, "k-", linewidth=2, label="Actual", zorder=5)
    ax.plot(y_test.index, y_pred_rw_test.values,
            ":", color="#9E9E9E", linewidth=1.8,
            label=f"Random Walk (RMSE={rmse_rw_test:.3f})", alpha=0.9)
    ax.plot(y_pred_ar_test.index, y_pred_ar_test.values,
            "-.", color="#795548", linewidth=1.5,
            label=f"Lag model ADL (RMSE={rmse_ar_test:.3f})", alpha=0.9)
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
    ax.set_title("Forecast vs. actual HICP inflation rate (test set)")
    ax.set_ylabel("HICP inflation rate (YoY, %)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    _save("fig_04_prognose.png")
    plt.show()
    print("Figure saved: fig_04_prognose.png")


# --- fig_05: MSE/RMSE comparison ---

def fig_05_mse_comparison(ctx):
    mse_rw_test    = ctx["mse_rw_test"]
    mse_ar_test    = ctx["mse_ar_test"]
    mse_ols_test   = ctx["mse_ols_test"]
    mse_ridge_test = ctx["mse_ridge_test"]
    mse_lasso_test = ctx["mse_lasso_test"]
    mse_enet_test  = ctx["mse_enet_test"]
    mse_lasso_plus = ctx["mse_lasso_plus_test"]

    # Order mirrors the grouping: benchmark -> with own lags -> illustrative
    all_models = ["RW", "ADL", "LASSO+HVPI", "OLS", "Ridge", "LASSO", "Elastic Net"]
    mse_vals   = [mse_rw_test, mse_ar_test, mse_lasso_plus,
                  mse_ols_test, mse_ridge_test, mse_lasso_test, mse_enet_test]
    rmse_vals  = [np.sqrt(v) for v in mse_vals]
    colors_bar = ["#9E9E9E", "#795548", "#9C27B0",
                  COLORS["OLS"], COLORS["Ridge"], COLORS["LASSO"], COLORS["ElasticNet"]]
    # Illustrative (indices 3-6): hatched bars, slightly transparent
    hatches    = ["", "", "", "///", "///", "///", "///"]
    alphas     = [0.9, 0.9, 0.9, 0.55, 0.55, 0.55, 0.55]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(all_models)); width = 0.5

    for ax, vals, title, ylabel in [
        (axes[0], mse_vals,  "Test MSE: groups compared",                  "Mean Squared Error"),
        (axes[1], rmse_vals, "Test RMSE (percentage points of inflation)",  "RMSE (%)"),
    ]:
        for xi, (val, col, hatch, alpha) in enumerate(
            zip(vals, colors_bar, hatches, alphas)
        ):
            ax.bar(xi, val, width, color=col, alpha=alpha, hatch=hatch,
                   edgecolor="white" if hatch == "" else col)
        # Divider between LASSO+HVPI and OLS (after index 2)
        ax.axvline(2.5, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(all_models, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    # RMSE value label only on the right panel
    for xi, val in enumerate(rmse_vals):
        axes[1].text(xi, val + 0.05, f"{val:.3f}", ha="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch as _Patch
    legend_handles = [
        _Patch(color="#9E9E9E",          label="Benchmark (RW, ADL)"),
        _Patch(color="#9C27B0",          label="With own lags / core comparison (LASSO+HVPI)"),
        _Patch(color="gray", alpha=0.55, hatch="///",
               label="Illustrative - macro only, no own lags (OLS, Ridge, LASSO, EN)"),
    ]
    axes[1].legend(handles=legend_handles, fontsize=8, loc="upper right")

    plt.suptitle(
        "Model comparison by group - divider: benchmark/core comparison | illustrative",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    _save("fig_05_mse_vergleich.png")
    plt.show()
    print("Figure saved: fig_05_mse_vergleich.png")


# --- fig_06: LASSO path ---

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
    ax.set_title("LASSO coefficient paths (top-15 features)")
    ax.set_xlabel("Regularisation parameter λ (log scale)")
    ax.set_ylabel("Coefficient")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    _save("fig_06_lasso_path.png")
    plt.show()
    print("Figure saved: fig_06_lasso_path.png")
    return top_idx


# --- fig_07: Ridge path ---

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
    ax.set_title("Ridge coefficient paths (top-15 features by LASSO rank)")
    ax.set_xlabel("Regularisation parameter λ (log scale)")
    ax.set_ylabel("Coefficient")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    _save("fig_07_ridge_path.png")
    plt.show()
    print("Figure saved: fig_07_ridge_path.png")


# --- fig_08: LASSO selection ---

def fig_08_lasso_selection(lasso_cv, X):
    lasso_coefs = pd.Series(lasso_cv.coef_, index=X.columns)
    selected    = lasso_coefs[lasso_coefs != 0].sort_values(key=np.abs, ascending=False)
    lambda_lasso = lasso_cv.alpha_

    print(f"LASSO selects {len(selected)} of {len(lasso_coefs)} features:\n")
    print(selected.to_string())

    fig, ax = plt.subplots(figsize=(12, max(5, len(selected) * 0.35)))
    colors_sel = ["#E91E63" if v < 0 else "#2196F3" for v in selected.values]
    ax.barh(range(len(selected)), selected.values, color=colors_sel, alpha=0.85)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"LASSO coefficients at opt. λ = {lambda_lasso:.5f} "
                 f"({len(selected)} selected features)")
    ax.set_xlabel("Standardised coefficient")
    plt.tight_layout()
    _save("fig_08_lasso_selektion.png")
    plt.show()
    print("\nFigure saved: fig_08_lasso_selektion.png")


# --- fig_09: LASSO CV path ---

def fig_09_lasso_cv_path(lasso_cv):
    cv_mses  = np.mean(lasso_cv.mse_path_, axis=1)
    cv_stds  = np.std(lasso_cv.mse_path_, axis=1)
    lambda_lasso = lasso_cv.alpha_

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(lasso_cv.alphas_, cv_mses, color="#4CAF50", linewidth=2,
                label="Mean CV MSE")
    ax.fill_between(lasso_cv.alphas_, cv_mses - cv_stds, cv_mses + cv_stds,
                    alpha=0.2, color="#4CAF50", label="±1 std. dev.")
    ax.axvline(lambda_lasso, color="red", linestyle="--", linewidth=1.5,
               label=f"Min. λ = {lambda_lasso:.5f}")
    ax.set_title("LASSO cross-validation: MSE as a function of λ")
    ax.set_xlabel("Regularisation parameter λ (log scale)")
    ax.set_ylabel("Mean CV MSE")
    ax.legend()
    plt.tight_layout()
    _save("fig_09_lasso_cv_path.png")
    plt.show()
    print("Figure saved: fig_09_lasso_cv_path.png")


# --- fig_10: shrinkage comparison ---

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
        ax.set_title(f"Shrinkage: OLS -> {name}")
        ax.set_xlabel("OLS coefficient")
        ax.set_ylabel(f"{name} coefficient")
    plt.suptitle("Shrinkage of coefficients (diagonal = no shrinkage)", y=1.02)
    plt.tight_layout()
    _save("fig_10_shrinkage.png")
    plt.show()
    print("Figure saved: fig_10_shrinkage.png")


# --- fig_11: rolling RMSE ---

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
        f"Rolling RMSE ({WINDOW_ROLLING_RMSE}-month window) - rolling-origin out-of-sample"
    )
    ax.set_ylabel("RMSE (percentage points)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save("fig_11_rolling_rmse.png")
    plt.show()
    print("Figure saved: fig_11_rolling_rmse.png")


# --- fig_12: selection stability ---

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
               label="≥25 % (occasional)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selection frequency (share of rolling windows)")
    ax.set_title(
        f"LASSO selection stability: top-{TOP_N_STABILITY} variables\n"
        f"({n_windows} rolling windows, λ = {lambda_lasso:.5f})"
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save("fig_12_selektionsstabilitaet.png")
    plt.show()
    print("Figure saved: fig_12_selektionsstabilitaet.png")


# --- fig_13: horizons RMSE ---

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
    ax.set_title("RMSE by forecast horizon h (months ahead)")
    ax.set_xlabel("Forecast horizon h (months)")
    ax.set_ylabel("RMSE (percentage points)")
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save("fig_13_horizonte_rmse.png")
    plt.show()
    print("Figure saved: fig_13_horizonte_rmse.png")


# --- Table exports ---

def export_results_table(results, y_test):
    _MON_EN = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    _t0_tex = f"{_MON_EN[y_test.index[0].month]}~{y_test.index[0].year}"
    _t1_tex = f"{_MON_EN[y_test.index[-1].month]}~{y_test.index[-1].year}"

    # Drop Group column for LaTeX body (grouping shown via \midrule separators)
    results_tex = results.drop(columns=["Group"], errors="ignore").rename(columns={
        "Test R²":         r"Test $R^2$",
        "Non-zero coeff.": r"Coeff.$\neq$0",
    })
    latex_results = results_tex.to_latex(
        float_format="%.4f", escape=False,
        caption=(
            r"Forecast models compared: mean RMSE, relative RMSE (RMSE/RW) "
            f"and $R^2$ on the test set ({_t0_tex}--{_t1_tex}). "
            r"\emph{Benchmark}: random walk and lag model (ADL, HICP own lags only). "
            r"\emph{Core comparison}: LASSO+HVPI (own lags + macro) - "
            r"economically sound, as \emph{ceteris paribus} with respect to own lags. "
            r"\emph{Illustrative}: OLS-Adaptive LASSO without own lags "
            r"(structurally disadvantaged, show regularisation vs.\ OLS overfitting)."
        ),
        label="tab:ergebnisse",
    )
    # Insert \midrule between benchmark / with own lags / illustrative
    latex_results = _insert_group_midrules(latex_results, after_rows={1, 2})

    with open("results/results_table.tex", "w") as f:
        f.write(latex_results)
    print("results/results_table.tex saved.")
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
                r"RMSE per forecast horizon $h \in \{1,3,6,12\}$ months. "
                r"$\hat{k}$ = number of non-zero coefficients."
            ),
            label="tab:horizonte",
        ))
    print("Horizons table saved as LaTeX: results/horizons_table.tex")


def export_stationarity_table(df_stationarity):
    """Export stationarity table to results/stationarity_table.{csv,tex}."""
    df_stationarity.to_csv("results/stationarity_table.csv", index=False)
    print("results/stationarity_table.csv saved.")

    col_rename = {
        "Series":       "Series",
        "Transform.":   "Transform.",
        "ADF-Stat.":    "ADF stat.",
        "ADF p-value":  "ADF $p$",
        "ADF verdict":  "ADF",
        "KPSS-Stat.":   "KPSS stat.",
        "KPSS p-value": "KPSS $p$",
        "KPSS verdict": "KPSS",
        "Overall":      "Verdict",
    }
    df_tex = df_stationarity.rename(columns=col_rename)
    latex_str = df_tex.to_latex(
        index=False, escape=False,
        caption=(
            "ADF and KPSS stationarity tests on level and YoY-transformed series. "
            "ADF H\\textsubscript{0}: unit root. KPSS H\\textsubscript{0}: stationarity. "
            "Verdict \\emph{stationary}: ADF rejects ($p<0.05$) and KPSS does not reject ($p\\geq0.05$)."
        ),
        label="tab:stationaritaet",
    )
    with open("results/stationarity_table.tex", "w") as f:
        f.write(latex_str)
    print("results/stationarity_table.tex saved.")


def export_inference_table(df_inference, y_test=None):
    """Export single-split inference table (block-bootstrap CI + DM/CW) as CSV + LaTeX."""
    df_inference.to_csv("results/inference_table.csv")
    print("results/inference_table.csv saved.")

    # Merge CI column for LaTeX output
    df_tex = df_inference.copy()
    df_tex["95%-KI"] = df_tex.apply(
        lambda r: f"[{r['CI 2.5%']:.3f}, {r['CI 97.5%']:.3f}]", axis=1
    )
    df_tex = df_tex.drop(columns=["CI 2.5%", "CI 97.5%"])
    # Stat. column: either "Stat." (post AP22) or "DM-Stat" (backward compatibility)
    stat_col = "Stat." if "Stat." in df_tex.columns else "DM-Stat"
    has_test_col = "Test" in df_tex.columns
    has_bonf_col = "p adj. (Bonf.)" in df_tex.columns
    if has_test_col and has_bonf_col:
        df_tex = df_tex[["Test RMSE", "95%-KI", "Test", stat_col, "p-value",
                          "p adj. (Bonf.)", "Sig. adj."]]
        df_tex.columns = [
            "Test RMSE", r"95\% CI (bootstrap)", "Test", "Stat.", "$p$-value",
            r"$p_{\mathrm{adj}}$ (Bonf.)", "Sig. adj.",
        ]
    elif has_test_col:
        df_tex = df_tex[["Test RMSE", "95%-KI", "Test", stat_col, "p-value", "Sig."]]
        df_tex.columns = [
            "Test RMSE", r"95\% CI (bootstrap)", "Test", "Stat.", "$p$-value", "Sig.",
        ]
    else:
        df_tex = df_tex[["Test RMSE", "95%-KI", stat_col, "p-value", "Sig."]]
        df_tex.columns = [
            "Test RMSE", r"95\% CI (bootstrap)", "Stat.", "$p$-value", "Sig.",
        ]

    t_note = ""
    if y_test is not None:
        t_note = f", T={len(y_test)} test points"

    bonf_note = (
        r" $p_{\mathrm{adj}}$: Bonferroni correction for multiple testing "
        r"($p_{\mathrm{adj},i}=\min(n\cdot p_i,1)$, $n$ = number of tests)."
        if has_bonf_col else ""
    )

    latex_str = df_tex.to_latex(
        escape=False,
        float_format="%.4f",
        na_rep="-",
        caption=(
            r"Single-window inference: block-bootstrap 95\% confidence interval for "
            r"the RMSE (circular block bootstrap, $l=6\approx\sqrt{T}$, $B=2\,000$) "
            r"and inference test per model against the "
            f"random walk{t_note}. "
            r"Non-nested models: Diebold-Mariano test (HLN correction, $h=1$, "
            r"two-sided). Nested models (lag model, LASSO+HVPI): "
            r"Clark-West test (2007, one-sided, $H_1$: model more accurate than RW). "
            + bonf_note +
            r" Stat.\ $> 0$: model beats RW. Sig.: * $p<0.10$, ** $p<0.05$."
        ),
        label="tab:inferenz",
    )
    with open("results/inference_table.tex", "w") as f:
        f.write(latex_str)
    print("results/inference_table.tex saved.")
    print(df_inference.to_string())


def export_sources_table():
    from .config import (
        BS_INDICATORS, LCI_SERIES, PROD_SECTORS, PPI_SECTORS, UNEMP_GROUPS,
    )

    rows = [
        {"Variable": "HVPI", "Source": "ECB SDW",
         "Dataset": "ICP/M.DE.N.000000.4.INX",
         "Code / Filter": "DE, 000000 (total), Index 2015=100", "Freq.": "M", "SA": "-"},
    ]
    for name, nace in PROD_SECTORS.items():
        rows.append({"Variable": name, "Source": "Eurostat",
                     "Dataset": "sts_inpr_m",
                     "Code / Filter": f"nace_r2={nace}, unit=I15, geo=DE",
                     "Freq.": "M", "SA": "NSA"})
    for name, indic in BS_INDICATORS.items():
        rows.append({"Variable": name, "Source": "Eurostat",
                     "Dataset": "ei_bsin_m_r2",
                     "Code / Filter": f"indic={indic}, geo=DE",
                     "Freq.": "M", "SA": "SA"})
    for name, nace in PPI_SECTORS.items():
        rows.append({"Variable": name, "Source": "Eurostat",
                     "Dataset": "sts_inppd_m",
                     "Code / Filter": f"nace_r2={nace}, unit=I15, geo=DE",
                     "Freq.": "M", "SA": "NSA"})
    for name, grp in UNEMP_GROUPS.items():
        rows.append({"Variable": name, "Source": "Eurostat",
                     "Dataset": "une_rt_m",
                     "Code / Filter": f"sex={grp['sex']}, age={grp['age']}, unit=PC_ACT, geo=DE",
                     "Freq.": "M", "SA": "SA"})
    for name, flt in LCI_SERIES.items():
        rows.append({"Variable": name, "Source": "Eurostat",
                     "Dataset": "lc_lci_r2_q",
                     "Code / Filter": (
                         f"nace_r2={flt['nace_r2']}, lcstruct={flt['lcstruct']}, "
                         f"unit=I20, geo=DE; Q->M via ffill"
                     ),
                     "Freq.": "Q->M", "SA": "NSA"})

    df_sources = pd.DataFrame(rows)
    df_sources.to_csv("results/sources_table.csv", index=False)
    with open("results/sources_table.tex", "w") as f:
        f.write(df_sources.to_latex(
            index=False, escape=True,
            caption=(
                "Data sources: variable, source, dataset code and "
                "seasonal adjustment (SA = seasonally adjusted, NSA = not adjusted)."
            ),
            label="tab:quellen",
        ))
    print(f"Sources table: {len(df_sources)} rows -> results/sources_table.csv + .tex")
    print(df_sources.to_string(index=False))


def export_robustness_table(df_robustness_mom):
    """Export MoM robustness table (AP29) to results/robustness_mom_table.{csv,tex}."""
    df_robustness_mom.to_csv("results/robustness_mom_table.csv")
    print("results/robustness_mom_table.csv saved.")

    df_tex = df_robustness_mom.rename(columns={
        "Test RMSE (MoM)": "Test RMSE",
        "RMSE/RW":         r"RMSE/RW",
        "RMSE/AO":         r"RMSE/AO",
    })
    latex_str = df_tex.to_latex(
        float_format="%.4f",
        escape=False,
        caption=(
            r"Robustness check MoM specification (AP29): rolling-origin RMSE "
            r"($h=1$, fixed $\lambda$) for the HICP monthly rate (Δ\,\%) "
            r"instead of the annual rate (YoY). "
            r"AO: Atkeson-Ohanian benchmark - rolling 12-month mean "
            r"of the MoM rates (Atkeson \& Ohanian 2001, \emph{AER} 91). "
            r"RMSE/RW\,$<1$: model undercuts the RW in RMSE. "
            r"RMSE/AO\,$<1$: model undercuts the AO benchmark in RMSE. "
            r"In the MoM specification the RW is a weak benchmark that "
            r"all macro models undercut. The decisive benchmark is the AO benchmark "
            r"(pure persistence), which no model undercuts. "
            r"The finding \emph{no macro value-added beyond persistence} "
            r"is thus robust to the choice of YoY\,vs.\,MoM as the target."
        ),
        label="tab:robustheit_mom",
    )
    with open("results/robustness_mom_table.tex", "w") as f:
        f.write(latex_str)
    print("results/robustness_mom_table.tex saved.")
    print(df_robustness_mom.to_string())


def export_robustness_extended_table(ext_ctx):
    """Export sample-extension robustness (AP32) to results/robustness_extended.{csv,tex}."""
    df  = ext_ctx["df_robustness_extended"]
    df.to_csv("results/robustness_extended.csv")
    print("results/robustness_extended.csv saved.")

    dropped   = ", ".join(ext_ctx["dropped"]) or "-"
    post_start = (pd.Timestamp(ext_ctx["shock_end"])
                  + pd.DateOffset(months=1)).strftime("%Y-%m")
    n_post_tests = int(df["p Post"].notna().sum())
    tex_cols = ["RMSE Shock", "RMSE/RW Shock", "RMSE Post", "RMSE/RW Post",
                "RMSE Total", "RMSE/RW Total", "Test", "Stat Post",
                "p Post", "p adj. Post", "Sig Post adj."]
    df_tex = df[tex_cols].rename(columns={
        "RMSE/RW Shock": r"RMSE/RW$_{S}$", "RMSE/RW Post": r"RMSE/RW$_{P}$",
        "RMSE/RW Total": r"RMSE/RW$_{G}$", "Stat Post": "Stat$_P$",
        "p Post": r"$p_P$", "p adj. Post": r"$p_P^{\mathrm{adj}}$",
        "Sig Post adj.": "Sig$_P$",
    })
    latex_str = df_tex.to_latex(
        float_format="%.4f",
        escape=False,
        na_rep="-",
        caption=(
            r"Sample-extension robustness (AP32): rolling-origin RMSE ($h=1$, "
            r"fixed $\lambda$) after removing the binding series \texttt{"
            + dropped.replace("_", r"\_") + r"}. The OOS window extends from "
            + ext_ctx["orig_end"].strftime("%Y-%m") + r" to "
            + ext_ctx["ext_end"].strftime("%Y-%m") + r" (+"
            + str(ext_ctx["months_gained"]) + r" months). The post-shock segment ("
            + post_start + r"--" + ext_ctx["ext_end"].strftime("%Y-%m")
            + r", $n_P=" + str(ext_ctx["n_post"]) + r"$) tests the claim "
            r"\emph{RW unbeatable} for the first time out-of-sample in the non-shock regime. "
            r"Index $S$: shock, $P$: post-shock, $G$: total. "
            r"Post-test: DM (HLN, two-sided) or CW (2007, one-sided, nested) "
            r"vs. RW. $p_P^{\mathrm{adj}}$: Bonferroni correction over the "
            + str(n_post_tests) + r" parallel post-tests, consistent with the "
            r"single-window inference (\cref{tab:inferenz}). Sig$_P$ refers to "
            r"$p_P^{\mathrm{adj}}$ (* $p<0.10$, ** $p<0.05$). Unadjusted, "
            r"AR and LASSO+HVPI reach $p_P<0.05$, but after correction "
            r"no model is significant at the 5\,\% level any more."
        ),
        label="tab:robustheit_extended",
    )
    with open("results/robustness_extended.tex", "w") as f:
        f.write(latex_str)
    print("results/robustness_extended.tex saved.")
    print(df.to_string())


def fig_14_giacomini_rossi(gr_ctx, shock_end=None):
    """Giacomini-Rossi fluctuation plot: rolling GR_t(m) statistic vs. time.

    Shows, for key models, *when* the relative predictive ability vs. RW is positive
    (model better) or negative (model worse) and whether the critical band
    is exceeded. Adds the time dimension to the pooled DM/CW test.

    Source: Giacomini & Rossi (2010), Tab. 1. Giacomini & White (2006).
    """
    from .config import REGIME_SHOCK_END

    gr_df  = gr_ctx["gr_df"]
    cv_05  = gr_ctx["cv_05"]
    cv_10  = gr_ctx["cv_10"]
    m      = gr_ctx["m"]
    mu     = gr_ctx["mu"]
    if shock_end is None:
        shock_end = REGIME_SHOCK_END

    # Priority: AR and LASSO+HVPI first (nested borderline cases), then others
    priority = ["AR", "LASSO+HVPI", "Adaptive LASSO", "LASSO", "Ridge"]
    cols     = [c for c in priority if c in gr_df.columns][:4]

    _col_map = {
        "AR":             "#795548",
        "LASSO+HVPI":     "#9C27B0",
        "Adaptive LASSO": "#FF5722",
        "LASSO":          "#4CAF50",
        "Ridge":          "#FF9800",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    # Critical bands
    ax.axhline( cv_05, color="red",    linestyle="--", linewidth=1.3, alpha=0.85,
                label=f"+{cv_05:.3f} / -{cv_05:.3f}  (5% crit., GR 2010, μ={mu:.2f})")
    ax.axhline(-cv_05, color="red",    linestyle="--", linewidth=1.3, alpha=0.85)
    ax.axhline( cv_10, color="darkorange", linestyle=":",  linewidth=1.0, alpha=0.7,
                label=f"+{cv_10:.3f} / -{cv_10:.3f}  (10% crit.)")
    ax.axhline(-cv_10, color="darkorange", linestyle=":",  linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.9, linestyle="-", alpha=0.4)

    # Regime split
    shock_ts = pd.Timestamp(shock_end)
    ax.axvline(shock_ts, color="#555", linestyle="--", linewidth=1.1, alpha=0.7,
               label=f"Regime split ({shock_end}): shock -> disinflation")

    # GR statistics per model
    for col in cols:
        series = gr_df[col].dropna()
        ax.plot(series.index, series.values,
                label=col, color=_col_map.get(col, "black"),
                linewidth=1.8, marker="o", markersize=3.5, alpha=0.9)

    ax.set_title(
        f"Giacomini-Rossi fluctuation test: time-varying predictive ability vs. random walk\n"
        f"(rolling DM statistic, window m={m}, μ={mu:.2f}, "
        f"GR_t>0: model better than RW, crit. values: Giacomini & Rossi 2010, Tab. 1)"
    )
    ax.set_xlabel("Date (end of the rolling window)")
    ax.set_ylabel("GR statistic  GR_t(m)")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    _save("fig_14_giacomini_rossi.png")
    plt.show()
    print("Figure saved: fig_14_giacomini_rossi.png")


def export_gr_table(gr_ctx):
    """Export GR fluctuation statistics per model as CSV."""
    gr_df = gr_ctx["gr_df"]
    gr_df.to_csv("results/gr_table.csv")
    print(f"results/gr_table.csv saved. "
          f"({len(gr_df)} time points, m={gr_ctx['m']}, μ={gr_ctx['mu']:.3f})")
    print(f"  cv_5%={gr_ctx['cv_05']:.3f}, cv_10%={gr_ctx['cv_10']:.3f} "
          f"(Giacomini & Rossi 2010, Tab. 1)")


def export_regime_table(df_regime, shock_end="2023-03", n_shock=None, n_disfl=None):
    """Export regime table (shock vs. disinflation) to results/regime_table.{csv,tex}."""
    df_regime.to_csv("results/regime_table.csv")
    print("results/regime_table.csv saved.")

    disfl_start = (pd.Timestamp(shock_end) + pd.DateOffset(months=1)).strftime("%Y-%m")
    n_info = (
        f"$n_{{\\text{{Shock}}}}={n_shock}$, $n_{{\\text{{Disinfl.}}}}={n_disfl}$. "
        if n_shock is not None else ""
    )
    latex_str = df_regime.to_latex(
        float_format="%.4f",
        escape=False,
        caption=(
            r"Rolling-origin RMSE and RMSE/RW per inflation regime ($h=1$, fixed $\lambda$). "
            r"Shock: energy price shock (rising/peak, 2021-06--" + shock_end + r"). "
            r"Disinflation: " + disfl_start + r"--2024-10. "
            + n_info
            + r"RMSE/RW $< 1$: model undercuts the RW in RMSE. "
            r"RMSE/RW $= 1.00$: reference (RW). "
            r"No model beats the RW \emph{significantly} in either regime. "
            r"In the disinflation regime AR and LASSO+HVPI do undercut it in "
            r"point RMSE ($0.92$ each), but without statistical significance. "
            r"The statement is not limited to the energy price shock regime."
        ),
        label="tab:regime",
    )
    with open("results/regime_table.tex", "w") as f:
        f.write(latex_str)
    print("results/regime_table.tex saved.")
    print(df_regime.to_string())


# --- Selection interpretation (AP30) ---

def export_selection_economic(sel_regime_ctx):
    """Export regime-dependent selection frequency per economic group."""
    df  = sel_regime_ctx["df_sel_groups"]
    n_s = sel_regime_ctx["n_shock_sel"]
    n_d = sel_regime_ctx["n_disfl_sel"]

    df.to_csv("results/selection_economic.csv")
    print("results/selection_economic.csv saved.")

    latex_str = df.to_latex(
        float_format="%.3f",
        escape=False,
        caption=(
            r"LASSO selection frequency per economic variable group "
            r"(mean share of rolling windows in which ≥\,1 variable of the group "
            r"was selected). "
            r"Shock: energy price shock regime (2021-06--2023-03, "
            f"$n_{{\\text{{Shock}}}}={n_s}$). "
            r"Disinflation: 2023-04--2024-10 "
            f"($n_{{\\text{{Disinfl.}}}}={n_d}$). "
            r"Cost-push hypothesis (cost-push/Phillips curve): "
            r"PPI and LCI variables should be selected more often in the shock regime."
        ),
        label="tab:selection_economic",
    )
    with open("results/selection_economic.tex", "w") as f:
        f.write(latex_str)
    print("results/selection_economic.tex saved.")
    print(df.to_string())


# --- README auto-sync ---

def update_readmes(ctx):
    """Regenerate the <!-- RESULTS:BEGIN/END --> block in README.md and README_DE.md."""
    from .config import ROOT, TEST_MONTHS

    def _neg(v, d=2):
        return f"{v:.{d}f}"

    y      = ctx["y"]
    y_test = ctx["y_test"]
    X      = ctx["X"]
    X_plus = ctx["X_plus"]

    train_end = ctx["train_end"]
    ro        = ctx["oos_rmse"]
    ext       = ctx.get("robustness_extended")

    _n_total = len(y)
    _n_train = train_end
    _n_feat  = X.shape[1]
    _n_plus  = X_plus.shape[1]
    _d0 = y.index[0].strftime("%Y-%m")
    _d1 = y.index[-1].strftime("%Y-%m")
    _t0 = y_test.index[0].strftime("%Y-%m")
    _t1 = y_test.index[-1].strftime("%Y-%m")

    _rw    = ctx["rmse_rw_test"]
    _ar    = ctx["rmse_ar_test"]
    _alasso = ctx["rmse_alasso_test"]
    _lp    = ctx["rmse_lasso_plus_test"]
    _las   = float(np.sqrt(ctx["mse_lasso_test"]))
    _en    = float(np.sqrt(ctx["mse_enet_test"]))
    _ri    = float(np.sqrt(ctx["mse_ridge_test"]))
    _ols   = float(np.sqrt(ctx["mse_ols_test"]))
    _nz_l  = ctx["n_nonzero"]

    lasso_plus_alpha  = ctx["lasso_plus_cv"].alpha_
    lambda_alasso     = ctx["alasso"].alpha_
    lambda_lasso      = ctx["lambda_lasso"]
    lambda_enet       = ctx["lambda_enet"]
    lambda_ridge      = ctx["lambda_ridge"]
    n_nonzero_plus    = ctx["n_nonzero_plus"]
    n_nonzero_enet    = ctx["n_nonzero_enet"]
    n_nonzero_alasso  = ctx["n_nonzero_alasso"]
    r2_rw_test        = ctx["r2_rw_test"]
    r2_ar_test        = ctx["r2_ar_test"]
    r2_alasso_test    = ctx["r2_alasso_test"]
    r2_lasso_plus     = ctx["r2_lasso_plus_test"]
    r2_lasso_test     = ctx["r2_lasso_test"]
    r2_enet_test      = ctx["r2_enet_test"]
    r2_ridge_test     = ctx["r2_ridge_test"]
    r2_ols_test       = ctx["r2_ols_test"]

    # DM significance markers from df_inference (if present)
    df_inf = ctx.get("df_inference")
    def _sig(model_name):
        if df_inf is None or model_name not in df_inf.index:
            return ""
        s = df_inf.loc[model_name, "Sig."]
        return f" {s}" if s not in ("-", "–", "") else ""

    block_de = (
        f"Datensatz: **{_n_total} Beobachtungen** ({_d0} - {_d1}), "
        f"davon **{_n_train} Training / {TEST_MONTHS} Test**\n"
        f"(Testfenster {_t0} - {_t1}), **{_n_feat} Features**.\n\n"
        f"**Testfenster (fester chronologischer Split), RMSE in Prozentpunkten der Inflationsrate.**\n"
        f"Test = DM (nicht-geschachtelt) oder CW (geschachtelt, Clark & West 2007). n.s. = nicht signifikant.\n\n"
        f"| Modell | λ | Test-RMSE | RMSE/RW | Test-R² | Test | Koeff. ≠ 0 |\n"
        f"|--------|----------:|----------:|--------:|--------:|-----:|-----------:|\n"
        f"| *- Benchmark -* | | | | | | |\n"
        f"| **Random Walk** | - | **{_rw:.2f}** | **1.00** | {r2_rw_test:.2f} | - | - |\n"
        f"| Lag-Modell (ADL) | - | {_ar:.2f} | {_ar/_rw:.2f} | {r2_ar_test:.2f} | CW {_sig('Lag model (ADL)') or 'n.s.'} | {len(AR_LAGS)} |\n"
        f"| *- Zentraler Vergleich: Eigen-Lags + Makro (ökonomisch sauber, ceteris paribus) -* | | | | | | |\n"
        f"| LASSO + HVPI-Lags | {lasso_plus_alpha:.3f} | {_lp:.2f} | {_lp/_rw:.2f} | {r2_lasso_plus:.2f} | CW {_sig('LASSO+HVPI') or 'n.s.'} | {n_nonzero_plus} / {_n_plus} |\n"
        f"| *- Didaktisch: nur Makro, ohne Eigen-Lags (strukturell benachteiligt) -* | | | | | | |\n"
        f"| Adaptive LASSO | {lambda_alasso:.5f} | {_alasso:.2f} | {_alasso/_rw:.2f} | {r2_alasso_test:.2f} | DM {_sig('Adaptive LASSO') or 'n.s.'} | {n_nonzero_alasso} / {_n_feat} |\n"
        f"| LASSO | {lambda_lasso:.3f} | {_las:.2f} | {_las/_rw:.2f} | {r2_lasso_test:.2f} | DM {_sig('LASSO') or 'n.s.'} | {_nz_l} / {_n_feat} |\n"
        f"| Elastic Net | {lambda_enet:.3f} | {_en:.2f} | {_en/_rw:.2f} | {r2_enet_test:.2f} | DM {_sig('Elastic Net') or 'n.s.'} | {n_nonzero_enet} / {_n_feat} |\n"
        f"| Ridge | {lambda_ridge:.1f} | {_ri:.2f} | {_ri/_rw:.2f} | {r2_ridge_test:.2f} | DM {_sig('Ridge') or 'n.s.'} | {_n_feat} / {_n_feat} |\n"
        f"| OLS | - | {_ols:.2f} | {_ols/_rw:.2f} | {_neg(r2_ols_test)} | DM {_sig('OLS') or 'n.s.'} | {_n_feat} / {_n_feat} |\n\n"
        f"**Zentraler Befund:** Lag-Modell (ADL, nur Eigen-Lags) RMSE/RW = {_ar/_rw:.2f} und "
        f"LASSO+HVPI (Eigen-Lags + Makro) RMSE/RW = {_lp/_rw:.2f}, also "
        f"Makro-Mehrwert über die Persistenz hinaus etwa 0 (ceteris paribus).\n"
        f"Den reinen Makro-Modellen (didaktischer Teil) fehlt der stärkste Einzelprädiktor (HVPI-Lag). "
        f"Ihr Abschneiden (RMSE/RW ≥ {min(_alasso/_rw, _las/_rw, _en/_rw, _ri/_rw):.2f}) "
        f"illustriert den Nutzen von Regularisierung gegenüber OLS-Overfitting, "
        f"ist aber **kein fairer Vergleich gegen den RW**.\n\n"
        f"Inferenztests (T={TEST_MONTHS}): DM = Diebold-Mariano (HLN-korr., zweiseitig) für reine Makro-Modelle. "
        f"CW = Clark-West (2007, einseitig) für Lag-Modell und LASSO+HVPI (geschachtelt in RW). "
        f"Kein Modell schlägt den RW signifikant (geringe Power bei T={TEST_MONTHS}). Block-Bootstrap-KI: `results/inference_table.csv`.\n"
        f"*Hinweis: Der RW-R² spiegelt die Persistenz der YoY-Rate wider (ŷ_t = y_{{t-1}} erklärt die "
        f"Autokorrelation). Er ist nicht mit dem Modell-R² gleichzusetzen.*\n\n"
        f"**Robustheitscheck (Rolling-Origin, Expanding Window):** "
        f"RW {ro['RW']:.2f} - AR {ro['AR']:.2f} - LASSO+HVPI {ro['LASSO+HVPI']:.2f} -\n"
        f"LASSO {ro['LASSO']:.2f} - Elastic Net {ro['Elastic Net']:.2f} - "
        f"Ridge {ro['Ridge']:.2f} - OLS {ro['OLS']:.2f}. "
        f"Die geschachtelten Modelle (AR, LASSO+HVPI)\n"
        f"erreichen den RW hier knapp, schlagen ihn aber nicht nachweisbar "
        f"(Clark-West-Test n.s.)."
    )

    block_en = (
        f"Dataset: **{_n_total} observations** ({_d0} - {_d1}), "
        f"of which **{_n_train} training / {TEST_MONTHS} test**\n"
        f"(test window {_t0} - {_t1}), **{_n_feat} features**.\n\n"
        f"**Test window (fixed chronological split), RMSE in percentage points of the inflation rate.**\n"
        f"Test = DM (non-nested) or CW (nested, Clark & West 2007), n.s. = not significant.\n\n"
        f"| Model | λ | Test RMSE | RMSE/RW | Test R² | Test | Coeff. ≠ 0 |\n"
        f"|-------|----------:|----------:|--------:|--------:|-----:|-----------:|\n"
        f"| *- Benchmark -* | | | | | | |\n"
        f"| **Random Walk** | - | **{_rw:.2f}** | **1.00** | {r2_rw_test:.2f} | - | - |\n"
        f"| Lag model (ADL) | - | {_ar:.2f} | {_ar/_rw:.2f} | {r2_ar_test:.2f} | CW {_sig('Lag model (ADL)') or 'n.s.'} | {len(AR_LAGS)} |\n"
        f"| *- Central comparison: own lags + macro (economically clean, ceteris paribus) -* | | | | | | |\n"
        f"| LASSO + HICP lags | {lasso_plus_alpha:.3f} | {_lp:.2f} | {_lp/_rw:.2f} | {r2_lasso_plus:.2f} | CW {_sig('LASSO+HVPI') or 'n.s.'} | {n_nonzero_plus} / {_n_plus} |\n"
        f"| *- Didactic: macro only, no own lags (structurally disadvantaged) -* | | | | | | |\n"
        f"| Adaptive LASSO | {lambda_alasso:.5f} | {_alasso:.2f} | {_alasso/_rw:.2f} | {r2_alasso_test:.2f} | DM {_sig('Adaptive LASSO') or 'n.s.'} | {n_nonzero_alasso} / {_n_feat} |\n"
        f"| LASSO | {lambda_lasso:.3f} | {_las:.2f} | {_las/_rw:.2f} | {r2_lasso_test:.2f} | DM {_sig('LASSO') or 'n.s.'} | {_nz_l} / {_n_feat} |\n"
        f"| Elastic Net | {lambda_enet:.3f} | {_en:.2f} | {_en/_rw:.2f} | {r2_enet_test:.2f} | DM {_sig('Elastic Net') or 'n.s.'} | {n_nonzero_enet} / {_n_feat} |\n"
        f"| Ridge | {lambda_ridge:.1f} | {_ri:.2f} | {_ri/_rw:.2f} | {r2_ridge_test:.2f} | DM {_sig('Ridge') or 'n.s.'} | {_n_feat} / {_n_feat} |\n"
        f"| OLS | - | {_ols:.2f} | {_ols/_rw:.2f} | {_neg(r2_ols_test)} | DM {_sig('OLS') or 'n.s.'} | {_n_feat} / {_n_feat} |\n\n"
        f"**Central finding:** Lag model (ADL, own lags only) RMSE/RW = {_ar/_rw:.2f} - "
        f"LASSO+HICP (own lags + macro) RMSE/RW = {_lp/_rw:.2f}, so "
        f"macro value-added beyond persistence ≈ 0 (ceteris paribus).\n"
        f"The pure macro models (didactic group) lack the strongest single predictor (HICP lag) - "
        f"their performance (RMSE/RW ≥ {min(_alasso/_rw, _las/_rw, _en/_rw, _ri/_rw):.2f}) "
        f"illustrates regularization vs. OLS overfitting but is **not a fair race against the RW**.\n\n"
        f"Inference tests (T={TEST_MONTHS}): DM = Diebold-Mariano (HLN-corrected, two-sided) for pure macro models, "
        f"CW = Clark-West (2007, one-sided) for lag model and LASSO+HICP (nested within RW). "
        f"No model beats the RW significantly (low power at T={TEST_MONTHS}). Block-bootstrap CIs: `results/inference_table.csv`.\n"
        f"*Note: The RW R² reflects the persistence (autocorrelation) of the YoY series "
        f"(ŷ_t = y_{{t-1}}), it is not comparable to the model R².*\n\n"
        f"**Robustness check (rolling-origin, expanding window):** "
        f"RW {ro['RW']:.2f} - AR {ro['AR']:.2f} - LASSO+HICP {ro['LASSO+HVPI']:.2f} -\n"
        f"LASSO {ro['LASSO']:.2f} - Elastic Net {ro['Elastic Net']:.2f} - "
        f"Ridge {ro['Ridge']:.2f} - OLS {ro['OLS']:.2f}. "
        f"The nested models (AR, LASSO+HICP)\n"
        f"nearly match the RW here, but do not beat it significantly "
        f"(Clark-West test n.s.)."
    )

    # --- Sample-extension robustness (AP32) - optional additional paragraph ---
    if ext:
        df_e   = ext["df_robustness_extended"]
        non_rw = df_e.drop("RW", errors="ignore")
        best_p = non_rw["RMSE/RW Post"].idxmin()
        val_p  = non_rw.loc[best_p, "RMSE/RW Post"]
        post0  = (pd.Timestamp(ext["shock_end"]) + pd.DateOffset(months=1)).strftime("%Y-%m")
        post1  = ext["ext_end"].strftime("%Y-%m")
        sig_win = non_rw[(non_rw["Stat Post"] > 0) & (non_rw["Sig Post"].isin(["*", "**"]))]
        drop_str = ", ".join(ext["dropped"]) or "-"

        if len(sig_win) > 0:
            verdict_de = (f"schlagen **{', '.join(sig_win.index)}** den RW signifikant "
                          f"(DM/CW p<0,10, bestes {best_p}, RMSE/RW={val_p:.2f})")
            verdict_en = (f"**{', '.join(sig_win.index)}** beat the RW significantly "
                          f"(DM/CW p<0.10, best {best_p}, RMSE/RW={val_p:.2f})")
        elif val_p < 1.0:
            verdict_de = (f"unterbietet das beste Modell ({best_p}) den RW in der "
                          f"Punktschätzung (RMSE/RW={val_p:.2f}), aber **nicht signifikant** (DM/CW n.s.)")
            verdict_en = (f"the best model ({best_p}) edges below the RW in point terms "
                          f"(RMSE/RW={val_p:.2f}), but **not significantly** (DM/CW n.s.)")
        else:
            verdict_de = (f"schlägt **weiterhin kein Modell** den RW (bestes {best_p}, "
                          f"RMSE/RW={val_p:.2f})")
            verdict_en = (f"**still no model** beats the RW (best {best_p}, RMSE/RW={val_p:.2f})")

        ext_de = (
            f"\n\n**Robustheit Sample-Verlängerung (AP32):** Entfernt man die einzige "
            f"bindende Reihe (`{drop_str}`, endet 2024-09), reicht das OOS-Fenster bis "
            f"**{post1}** (+{ext['months_gained']} Monate, Post-Schock-Segment "
            f"{post0}-{post1}, n={ext['n_post']}, vorher 14). Im ruhigeren Post-Schock-"
            f"Regime {verdict_de}. Damit ist die These *RW unschlagbar* erstmals "
            f"out-of-sample außerhalb des Energiepreisschocks geprüft. "
            f"Tabelle: `results/robustness_extended.csv`."
        )
        ext_en = (
            f"\n\n**Sample-extension robustness (AP32):** Dropping the single binding "
            f"series (`{drop_str}`, ends 2024-09) extends the OOS window to **{post1}** "
            f"(+{ext['months_gained']} months, post-shock segment {post0}-{post1}, "
            f"n={ext['n_post']}, was 14). In the calmer post-shock regime {verdict_en}. "
            f"This is the first time the *RW-unbeatable* claim is tested out-of-sample "
            f"outside the energy price shock. Table: `results/robustness_extended.csv`."
        )
        block_de += ext_de
        block_en += ext_en

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
        assert new != txt or block in txt, f"{fpath}: marker not found"
        fpath.write_text(new, encoding="utf-8")
        print(f"{fpath} OK")

    print("\nREADME auto-sync complete.")
    print(f"  {_n_total} obs. ({_d0}-{_d1}), {_n_feat} features, "
          f"Test {_t0}-{_t1}")
    print(f"  RW {_rw:.4f}  |  LASSO {_las:.4f}  |  OLS {_ols:.4f}")


# --- Summary ---

def print_summary(ctx):
    """Print the summary (corresponds to cell 51 in the original notebook)."""
    X       = ctx["X"]
    y       = ctx["y"]
    y_test  = ctx["y_test"]
    results = ctx["results"]
    selected = ctx["selected"]

    print("=" * 75)
    print("SUMMARY OF RESULTS")
    print("=" * 75)
    print(f"Dataset:    {X.shape[0]} months, {X.shape[1]} features")
    print(f"Period:     {y.index[0].strftime('%Y-%m')} - {y.index[-1].strftime('%Y-%m')}")
    print(f"Test split: {len(y_test)} months (chronological)")
    print()
    print(f"{'Model':<16} {'Test-RMSE':>10} {'RMSE/RW':>9} {'Test-R²':>10} {'Coeff.≠0':>10}")

    rmse_rw = ctx["rmse_rw_test"]

    _groups = [
        ("-- Benchmark ------------------------------------------------------", [
            ("Random Walk",  ctx["rmse_rw_test"],            ctx["r2_rw_test"],        "-"),
            ("ADL",          ctx["rmse_ar_test"],            ctx["r2_ar_test"],        str(len(AR_LAGS))),
        ]),
        ("-- Core comparison: own lags + macro (ceteris paribus) ----------", [
            ("LASSO+HVPI",   ctx["rmse_lasso_plus_test"],    ctx["r2_lasso_plus_test"], str(ctx["n_nonzero_plus"])),
        ]),
        ("-- Illustrative: macro only, no own lags (structurally disadv.) --", [
            ("OLS",          np.sqrt(ctx["mse_ols_test"]),   ctx["r2_ols_test"],       str(int(np.sum(ctx["ols"].coef_ != 0)))),
            ("Ridge",        np.sqrt(ctx["mse_ridge_test"]), ctx["r2_ridge_test"],     str(len(ctx["ridge_cv"].coef_))),
            ("LASSO",        np.sqrt(ctx["mse_lasso_test"]), ctx["r2_lasso_test"],     str(ctx["n_nonzero"])),
            ("Elastic Net",  np.sqrt(ctx["mse_enet_test"]),  ctx["r2_enet_test"],      str(ctx["n_nonzero_enet"])),
            ("Adapt. LASSO", ctx["rmse_alasso_test"],        ctx["r2_alasso_test"],    str(ctx["n_nonzero_alasso"])),
        ]),
    ]

    for group_label, rows in _groups:
        print(f"\n{group_label}")
        print("-" * 75)
        for name, rmse_val, r2_val, nz in rows:
            rel = "1.000 (Ref)" if name == "Random Walk" else f"{rmse_val/rmse_rw:.3f}"
            print(f"{name:<16} {rmse_val:>10.4f} {rel:>9} {r2_val:>10.4f} {nz:>10}")

    print("\n" + "=" * 75)
    _ar  = ctx["rmse_ar_test"]
    _lp  = ctx["rmse_lasso_plus_test"]
    print(f"\nCore finding (ceteris paribus):")
    print(f"  ADL (own lags only):       RMSE/RW = {_ar/rmse_rw:.3f}")
    print(f"  LASSO+HVPI (own lags + macro): RMSE/RW = {_lp/rmse_rw:.3f}")
    print(f"  → macro value-added beyond persistence ≈ {(_lp - _ar)/rmse_rw:+.3f} (RMSE/RW)")
    best = results["Test RMSE"].astype(float).idxmin()
    print(f"\nBest model by test RMSE: {best}")
    print()
    print("Selected variable groups (LASSO):")
    for grp, prefix in [
        ("Industrial production", "IP_"),
        ("Business surveys",      "BS_"),
        ("Producer prices",       "PPI_"),
        ("Labour market",         "ALQ_"),
        ("Labour costs",          "LCI_"),
    ]:
        sel_grp = [c for c in selected.index if prefix in c]
        if sel_grp:
            print(f"  {grp}: {len(sel_grp)} feature(s) - {sel_grp}")


# --- Helper functions ---

def _insert_group_midrules(latex_str: str, after_rows: set) -> str:
    """Insert \\midrule after specific (0-indexed) data rows into a
    pandas LaTeX string - for visual group separation in the table.

    after_rows: set of 0-indexed row numbers after which a
    \\midrule should be inserted.
    """
    lines = latex_str.split("\n")
    result = []
    past_header_midrule = False
    data_row_idx = -1

    for line in lines:
        stripped = line.strip()
        # The first \midrule separates the header from the data
        if stripped == r"\midrule" and not past_header_midrule:
            past_header_midrule = True
            result.append(line)
            continue
        # Data rows end with \\  and are not LaTeX command lines
        if past_header_midrule and stripped.endswith(r"\\") and not stripped.startswith("\\"):
            result.append(line)
            data_row_idx += 1
            if data_row_idx in after_rows:
                result.append(r"\midrule")
            continue
        result.append(line)

    return "\n".join(result)


def _save(filename):
    plt.savefig(FIGURES_DIR / filename, bbox_inches="tight")
