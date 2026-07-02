"""Stage 3b: Single-window model fitting and evaluation."""
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNetCV, LassoCV, LinearRegression, RidgeCV,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import (
    ALPHAS_LASSO, ALPHAS_RIDGE, AR_LAGS, L1_RATIOS_ENET, TSCV,
)
from .models import AdaptiveLasso


def fit_all_models(X, y, splits, tscv=None):
    """Fits all models on the training split, returns models + metrics.

    Parameters
    ----------
    X, y        : full feature matrix and target variable
    splits      : ctx dict from data_preprocessing.prepare_splits
    tscv        : TimeSeriesSplit object (default: config.TSCV)

    Returns
    -------
    dict with all model objects, hyperparameters, forecasts and metrics.
    """
    if tscv is None:
        tscv = TSCV

    X_train    = splits["X_train"]
    X_test     = splits["X_test"]
    y_train    = splits["y_train"]
    y_test     = splits["y_test"]
    X_train_s  = splits["X_train_s"]
    X_test_s   = splits["X_test_s"]
    X_ar_train = splits["X_ar_train"]
    X_ar_test  = splits["X_ar_test"]
    y_ar_train = splits["y_ar_train"]
    sc_ar      = splits["sc_ar"]
    X_plus_train_s = splits["X_plus_train_s"]
    X_plus_test_s  = splits["X_plus_test_s"]
    X_plus_train   = splits["X_plus_train"]
    y_plus_train   = splits["y_plus_train"]
    X_plus_test    = splits["X_plus_test"]

    ctx = {}

    # --- Random Walk ---
    y_pred_rw_test     = y.shift(1).loc[y_test.index]
    mse_rw_test        = mean_squared_error(y_test, y_pred_rw_test)
    r2_rw_test         = r2_score(y_test, y_pred_rw_test)
    rmse_rw_test       = np.sqrt(mse_rw_test)
    print(f"Random Walk - Test MSE: {mse_rw_test:.4f}  |  RMSE: {rmse_rw_test:.4f}"
          f"  |  R²: {r2_rw_test:.4f}")
    ctx.update(dict(
        y_pred_rw_test=y_pred_rw_test,
        mse_rw_test=mse_rw_test, rmse_rw_test=rmse_rw_test, r2_rw_test=r2_rw_test,
    ))

    # --- Lag model (ADL) ---
    ar_model = LinearRegression()
    ar_model.fit(sc_ar.fit_transform(X_ar_train), y_ar_train)
    y_pred_ar_test = pd.Series(
        ar_model.predict(sc_ar.transform(X_ar_test)), index=X_ar_test.index
    )
    mse_ar_test  = mean_squared_error(y_test, y_pred_ar_test)
    r2_ar_test   = r2_score(y_test, y_pred_ar_test)
    rmse_ar_test = np.sqrt(mse_ar_test)
    print(f"Lag model (ADL)       - Test MSE: {mse_ar_test:.4f}  |  "
          f"RMSE: {rmse_ar_test:.4f}  |  R²: {r2_ar_test:.4f}")
    ctx.update(dict(
        ar_model=ar_model,
        y_pred_ar_test=y_pred_ar_test,
        mse_ar_test=mse_ar_test, rmse_ar_test=rmse_ar_test, r2_ar_test=r2_ar_test,
    ))

    # --- LASSO + HVPI own lags (macro value-added) ---
    lasso_plus_cv = LassoCV(
        alphas=ALPHAS_LASSO, cv=tscv, max_iter=10000, n_jobs=-1
    )
    # Coordinate descent on the extended feature matrix triggers benign FP exceptions
    # under near-singularity (matmul overflow/invalid), suppressed locally.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        lasso_plus_cv.fit(X_plus_train_s, y_plus_train)
    y_pred_lasso_plus_test = pd.Series(
        lasso_plus_cv.predict(X_plus_test_s), index=X_plus_test.index
    )
    mse_lasso_plus_test  = mean_squared_error(y_test, y_pred_lasso_plus_test)
    r2_lasso_plus_test   = r2_score(y_test, y_pred_lasso_plus_test)
    rmse_lasso_plus_test = np.sqrt(mse_lasso_plus_test)
    n_nonzero_plus       = int(np.sum(lasso_plus_cv.coef_ != 0))
    print(f"LASSO+HVPI - λ={lasso_plus_cv.alpha_:.5f}, "
          f"MSE={mse_lasso_plus_test:.4f}, RMSE={rmse_lasso_plus_test:.4f}, "
          f"R²={r2_lasso_plus_test:.4f}, Coeff.≠0: {n_nonzero_plus}/{X_plus_train.shape[1]}")
    ctx.update(dict(
        lasso_plus_cv=lasso_plus_cv,
        y_pred_lasso_plus_test=y_pred_lasso_plus_test,
        mse_lasso_plus_test=mse_lasso_plus_test,
        rmse_lasso_plus_test=rmse_lasso_plus_test,
        r2_lasso_plus_test=r2_lasso_plus_test,
        n_nonzero_plus=n_nonzero_plus,
    ))

    # --- OLS ---
    ols = LinearRegression()
    ols.fit(X_train_s, y_train)
    y_pred_ols_train = ols.predict(X_train_s)
    y_pred_ols_test  = ols.predict(X_test_s)
    mse_ols_train = mean_squared_error(y_train, y_pred_ols_train)
    mse_ols_test  = mean_squared_error(y_test,  y_pred_ols_test)
    r2_ols_test   = r2_score(y_test, y_pred_ols_test)
    print(f"OLS - Train MSE: {mse_ols_train:.4f}  |  Test MSE: {mse_ols_test:.4f}")
    print(f"OLS - Test R²:   {r2_ols_test:.4f}")
    print(f"OLS - Non-zero coefficients: {np.sum(ols.coef_ != 0)}/{len(ols.coef_)}")
    ctx.update(dict(
        ols=ols,
        y_pred_ols_train=y_pred_ols_train, y_pred_ols_test=y_pred_ols_test,
        mse_ols_train=mse_ols_train, mse_ols_test=mse_ols_test, r2_ols_test=r2_ols_test,
    ))

    # --- Ridge ---
    ridge_cv = RidgeCV(
        alphas=ALPHAS_RIDGE, cv=tscv, scoring="neg_mean_squared_error"
    )
    ridge_cv.fit(X_train_s, y_train)
    lambda_ridge       = ridge_cv.alpha_
    y_pred_ridge_train = ridge_cv.predict(X_train_s)
    y_pred_ridge_test  = ridge_cv.predict(X_test_s)
    mse_ridge_train = mean_squared_error(y_train, y_pred_ridge_train)
    mse_ridge_test  = mean_squared_error(y_test,  y_pred_ridge_test)
    r2_ridge_test   = r2_score(y_test, y_pred_ridge_test)
    print(f"Ridge - Optimal λ (cross-validation): {lambda_ridge:.4f}")
    print(f"Ridge - Train MSE: {mse_ridge_train:.4f}  |  Test MSE: {mse_ridge_test:.4f}")
    print(f"Ridge - Test R²:   {r2_ridge_test:.4f}")
    ctx.update(dict(
        ridge_cv=ridge_cv, lambda_ridge=lambda_ridge,
        y_pred_ridge_train=y_pred_ridge_train, y_pred_ridge_test=y_pred_ridge_test,
        mse_ridge_train=mse_ridge_train, mse_ridge_test=mse_ridge_test,
        r2_ridge_test=r2_ridge_test,
    ))

    # --- LASSO ---
    lasso_cv = LassoCV(
        alphas=ALPHAS_LASSO, cv=tscv, max_iter=10000, n_jobs=-1,
    )
    lasso_cv.fit(X_train_s, y_train)
    lambda_lasso       = lasso_cv.alpha_
    y_pred_lasso_train = lasso_cv.predict(X_train_s)
    y_pred_lasso_test  = lasso_cv.predict(X_test_s)
    mse_lasso_train = mean_squared_error(y_train, y_pred_lasso_train)
    mse_lasso_test  = mean_squared_error(y_test,  y_pred_lasso_test)
    r2_lasso_test   = r2_score(y_test, y_pred_lasso_test)
    n_nonzero       = int(np.sum(lasso_cv.coef_ != 0))
    print(f"LASSO - Optimal λ (cross-validation): {lambda_lasso:.6f}")
    print(f"LASSO - Train MSE: {mse_lasso_train:.4f}  |  Test MSE: {mse_lasso_test:.4f}")
    print(f"LASSO - Test R²:   {r2_lasso_test:.4f}")
    print(f"LASSO - Selected variables: {n_nonzero}/{len(lasso_cv.coef_)} "
          f"({n_nonzero/len(lasso_cv.coef_)*100:.1f}%)")
    ctx.update(dict(
        lasso_cv=lasso_cv, lambda_lasso=lambda_lasso,
        y_pred_lasso_train=y_pred_lasso_train, y_pred_lasso_test=y_pred_lasso_test,
        mse_lasso_train=mse_lasso_train, mse_lasso_test=mse_lasso_test,
        r2_lasso_test=r2_lasso_test, n_nonzero=n_nonzero,
    ))

    # --- Elastic Net ---
    enet_cv = ElasticNetCV(
        l1_ratio=L1_RATIOS_ENET,
        alphas=ALPHAS_LASSO,
        cv=tscv, max_iter=10000, n_jobs=-1,
    )
    enet_cv.fit(X_train_s, y_train)
    lambda_enet       = enet_cv.alpha_
    l1_ratio_enet     = enet_cv.l1_ratio_
    y_pred_enet_train = enet_cv.predict(X_train_s)
    y_pred_enet_test  = enet_cv.predict(X_test_s)
    mse_enet_train  = mean_squared_error(y_train, y_pred_enet_train)
    mse_enet_test   = mean_squared_error(y_test,  y_pred_enet_test)
    r2_enet_test    = r2_score(y_test, y_pred_enet_test)
    n_nonzero_enet  = int(np.sum(enet_cv.coef_ != 0))
    print(f"Elastic Net - Optimal α: {lambda_enet:.6f}, l1_ratio: {l1_ratio_enet:.2f}")
    print(f"Elastic Net - Train MSE: {mse_enet_train:.4f}  |  Test MSE: {mse_enet_test:.4f}")
    print(f"Elastic Net - Test R²:   {r2_enet_test:.4f}")
    ctx.update(dict(
        enet_cv=enet_cv, lambda_enet=lambda_enet, l1_ratio_enet=l1_ratio_enet,
        y_pred_enet_train=y_pred_enet_train, y_pred_enet_test=y_pred_enet_test,
        mse_enet_train=mse_enet_train, mse_enet_test=mse_enet_test,
        r2_enet_test=r2_enet_test, n_nonzero_enet=n_nonzero_enet,
    ))

    # --- Adaptive LASSO ---
    alasso = AdaptiveLasso(cv=tscv, max_iter=10000).fit(X_train_s, y_train)
    y_pred_alasso_test  = pd.Series(alasso.predict(X_test_s),  index=y_test.index)
    y_pred_alasso_train = pd.Series(alasso.predict(X_train_s), index=y_train.index)
    mse_alasso_test    = mean_squared_error(y_test,  y_pred_alasso_test)
    mse_alasso_train   = mean_squared_error(y_train, y_pred_alasso_train)
    rmse_alasso_test   = np.sqrt(mse_alasso_test)
    r2_alasso_test     = r2_score(y_test, y_pred_alasso_test)
    n_nonzero_alasso   = int(np.sum(alasso.coef_ != 0))
    print(f"Adaptive LASSO - λ={alasso.alpha_:.5f}, "
          f"RMSE={rmse_alasso_test:.4f}, RMSE/RW={rmse_alasso_test/rmse_rw_test:.4f}, "
          f"R²={r2_alasso_test:.4f}, Coeff.≠0: {n_nonzero_alasso}/{X_train_s.shape[1]}")
    ctx.update(dict(
        alasso=alasso,
        y_pred_alasso_test=y_pred_alasso_test, y_pred_alasso_train=y_pred_alasso_train,
        mse_alasso_test=mse_alasso_test, mse_alasso_train=mse_alasso_train,
        rmse_alasso_test=rmse_alasso_test, r2_alasso_test=r2_alasso_test,
        n_nonzero_alasso=n_nonzero_alasso,
    ))

    # --- Results table ---
    results = _build_results_table(ctx, X.shape[1], splits["X_plus_train"].shape[1])
    results.to_csv("results/results_table.csv")
    print("\nResults table saved: results_table.csv")

    # LASSO coefficients for reporting
    lasso_coefs = pd.Series(lasso_cv.coef_, index=X.columns)
    selected    = lasso_coefs[lasso_coefs != 0].sort_values(key=np.abs, ascending=False)
    top_idx     = np.argsort(np.abs(lasso_cv.coef_))[::-1][:15]

    ctx.update(dict(results=results, selected=selected, top_idx=top_idx))
    return ctx


def _build_results_table(ctx, n_feat, n_plus):
    """Builds the results table (corresponds to Cell 29 in the original notebook).

    Order: Benchmark -> With own lags (central comparison) -> illustrative.
    The 'Gruppe' column makes the assignment explicit and is used by reporting.py
    for LaTeX separator lines and README grouping.
    """
    c = ctx
    # Order: [Benchmark] RW, ADL -> [With own lags] LASSO+HVPI -> [illustrative] OLS, Ridge, LASSO, EN, Adaptive LASSO
    results = pd.DataFrame({
        "Model": [
            "Random Walk", "Lag model (ADL)", "LASSO+HVPI",
            "OLS", "Ridge", "LASSO", "Elastic Net", "Adaptive LASSO",
        ],
        "λ": [
            "-", "-", f"{c['lasso_plus_cv'].alpha_:.5f}",
            "-", f"{c['lambda_ridge']:.3f}", f"{c['lambda_lasso']:.5f}",
            f"{c['lambda_enet']:.5f}", f"{c['alasso'].alpha_:.5f}",
        ],
        "Train MSE": [
            "-", "-", "-",
            round(c["mse_ols_train"], 4),   round(c["mse_ridge_train"], 4),
            round(c["mse_lasso_train"], 4), round(c["mse_enet_train"], 4),
            round(c["mse_alasso_train"], 4),
        ],
        "Test MSE": [
            c["mse_rw_test"],       c["mse_ar_test"],
            c["mse_lasso_plus_test"],
            c["mse_ols_test"],      c["mse_ridge_test"],
            c["mse_lasso_test"],    c["mse_enet_test"],
            c["mse_alasso_test"],
        ],
        "Test RMSE": [
            c["rmse_rw_test"],          c["rmse_ar_test"],
            c["rmse_lasso_plus_test"],
            np.sqrt(c["mse_ols_test"]), np.sqrt(c["mse_ridge_test"]),
            np.sqrt(c["mse_lasso_test"]), np.sqrt(c["mse_enet_test"]),
            c["rmse_alasso_test"],
        ],
        "RMSE/RW": [
            1.0,
            c["rmse_ar_test"]              / c["rmse_rw_test"],
            c["rmse_lasso_plus_test"]      / c["rmse_rw_test"],
            np.sqrt(c["mse_ols_test"])     / c["rmse_rw_test"],
            np.sqrt(c["mse_ridge_test"])   / c["rmse_rw_test"],
            np.sqrt(c["mse_lasso_test"])   / c["rmse_rw_test"],
            np.sqrt(c["mse_enet_test"])    / c["rmse_rw_test"],
            c["rmse_alasso_test"]          / c["rmse_rw_test"],
        ],
        "Test R²": [
            c["r2_rw_test"],       c["r2_ar_test"],
            c["r2_lasso_plus_test"],
            c["r2_ols_test"],      c["r2_ridge_test"],
            c["r2_lasso_test"],    c["r2_enet_test"],
            c["r2_alasso_test"],
        ],
        "Non-zero coeff.": [
            "-", str(len(AR_LAGS)),
            str(c["n_nonzero_plus"]) + f" / {n_plus}",
            str(int(np.sum(c["ols"].coef_ != 0))),
            str(len(c["ridge_cv"].coef_)),
            str(c["n_nonzero"]),
            str(c["n_nonzero_enet"]),
            str(c["n_nonzero_alasso"]),
        ],
    }).set_index("Model")

    for col in ["Test MSE", "Test RMSE", "RMSE/RW", "Test R²"]:
        results[col] = results[col].round(4)

    # Group identifier: Benchmark / With own lags (central comparison) / Illustrative
    results.insert(0, "Group", [
        "Benchmark", "Benchmark", "With own lags",
        "Illustrative", "Illustrative", "Illustrative", "Illustrative", "Illustrative",
    ])
    return results
