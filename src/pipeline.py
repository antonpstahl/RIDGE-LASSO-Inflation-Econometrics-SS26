"""End-to-End-Orchestrierung: run_all() reproduziert alle results/-Artefakte."""
from . import config
from .data_preparation import get_raw_data, print_truncation_info
from .data_preprocessing import build_feature_matrix, prepare_splits, transform_to_yoy
from .evaluation import (
    compute_adaptive_oos,
    compute_compare_oos,
    compute_dm_tests,
    compute_giacomini_rossi,
    compute_horizon_analysis,
    compute_oos_predictions,
    compute_regime_analysis,
    compute_robustness_mom,
    compute_selection_stability,
    compute_single_split_inference,
)
from .reporting import (
    export_gr_table,
    export_horizons_table,
    export_inference_table,
    export_regime_table,
    export_results_table,
    export_robustness_table,
    export_sources_table,
    fig_01_hvpi,
    fig_02_correlation,
    fig_02b_heatmap,
    fig_03_tscv,
    fig_04_forecast,
    fig_05_mse_comparison,
    fig_06_lasso_path,
    fig_07_ridge_path,
    fig_08_lasso_selection,
    fig_09_lasso_cv_path,
    fig_10_shrinkage,
    fig_11_rolling_rmse,
    fig_12_selection_stability,
    fig_13_horizons,
    fig_14_giacomini_rossi,
    print_summary,
    update_readmes,
)
from .training import fit_all_models


def run_all(use_cache=True, verbose=True, adaptive_rolling=True,
            generate_figures=True):
    """Fuehrt die vollstaendige Pipeline aus und gibt einen Kontext-Dict zurueck.

    Parameters
    ----------
    use_cache         : bool  — Rohdaten aus CSV-Cache laden (True) oder API (False)
    verbose           : bool  — Status-Ausgaben aktivieren
    adaptive_rolling  : bool  — Adaptive Rolling-Origin (λ je Origin, ~10-20 min)
    generate_figures  : bool  — Abbildungen erzeugen und speichern

    Returns
    -------
    ctx : dict mit allen Zwischenergebnissen (Daten, Modelle, Metriken, Tabellen)
    """
    config.setup_environment()

    # ── Stage 1: Datenbeschaffung ─────────────────────────────────────────────
    df_raw = get_raw_data(use_cache=use_cache, verbose=verbose)

    # ── Stage 2: Preprocessing ───────────────────────────────────────────────
    df_yoy = transform_to_yoy(df_raw)
    df_yoy.to_csv(config.DATA_PROCESSED)

    X, y = build_feature_matrix(df_yoy, lags=config.LAGS, forecast_horizon=1,
                                 test_months=config.TEST_MONTHS)
    if verbose:
        print(f"Feature-Matrix:  {X.shape[0]} Beobachtungen x {X.shape[1]} Features")
        print(f"Zielvariable:    {len(y)} Beobachtungen")
        print(f"Zeitraum:        {y.index[0]:%Y-%m} - {y.index[-1]:%Y-%m}")
        print(f"\nAnteil NaN in X: {X.isna().mean().mean():.1%}")

    train_end = len(y) - config.TEST_MONTHS
    splits    = prepare_splits(X, y, train_end, ar_lags=config.AR_LAGS)

    if verbose:
        y_train = splits["y_train"]; y_test = splits["y_test"]
        X_train = splits["X_train"]
        print(f"\nTrainingsdaten: {len(y_train)} Monate "
              f"({y_train.index[0]:%Y-%m} – {y_train.index[-1]:%Y-%m})")
        print(f"Testdaten:      {len(y_test)} Monate "
              f"({y_test.index[0]:%Y-%m} – {y_test.index[-1]:%Y-%m})")
        print(f"\nDimensionen: {X_train.shape[0]} Train × {X_train.shape[1]} Features")
        print(f"n < p: {'JA (hochdimensional)' if X_train.shape[0] < X_train.shape[1] else 'NEIN'}")

    # ── Stage 3: Modellschätzung ──────────────────────────────────────────────
    models_ctx = fit_all_models(X, y, splits, tscv=config.TSCV)

    # ── Stage 4: Evaluation ───────────────────────────────────────────────────
    inf_ctx = compute_single_split_inference(models_ctx, splits)
    oos_ctx = compute_oos_predictions(models_ctx, splits, X, y, train_end)

    if adaptive_rolling:
        adap_ctx    = compute_adaptive_oos(X, y, splits, train_end, config.TSCV_INNER)
        compare_ctx = compute_compare_oos(oos_ctx, adap_ctx, oos_ctx["y_oos_ref"])
    else:
        adap_ctx    = {}
        compare_ctx = {"compare_oos": oos_ctx["oos_df"], "adap_rmse": oos_ctx["oos_rmse"]}

    reg_ctx = compute_regime_analysis(oos_ctx)
    dm_ctx  = compute_dm_tests(oos_ctx)
    gr_ctx  = compute_giacomini_rossi(
        oos_ctx,
        adap_ctx=adap_ctx if adaptive_rolling else None,
    )
    sel_ctx = compute_selection_stability(X, y, train_end, models_ctx["lambda_lasso"])
    hor_ctx = compute_horizon_analysis(df_yoy, tscv=config.TSCV)
    rob_ctx = compute_robustness_mom(df_raw, test_months=config.TEST_MONTHS)

    # ── Kontext zusammenführen ────────────────────────────────────────────────
    ctx = {
        "df_raw": df_raw, "df_yoy": df_yoy,
        "X": X, "y": y, "train_end": train_end,
        **splits,
        **models_ctx,
        **inf_ctx,
        **oos_ctx,
        **adap_ctx,
        **compare_ctx,
        **reg_ctx,
        **dm_ctx,
        **gr_ctx,
        **sel_ctx,
        **hor_ctx,
        **rob_ctx,
    }

    # ── Reporting ─────────────────────────────────────────────────────────────
    if generate_figures:
        fig_01_hvpi(df_raw, df_yoy)
        fig_02_correlation(X, y)
        fig_02b_heatmap(X, train_end)
        fig_03_tscv(splits["X_train_s"], config.TSCV)
        fig_04_forecast(ctx)
        fig_05_mse_comparison(ctx)

        feat_names = X.columns.tolist()
        top_idx    = models_ctx["top_idx"]
        fig_06_lasso_path(splits["X_train_s"], splits["y_train"],
                          models_ctx["lasso_cv"], feat_names)
        fig_07_ridge_path(splits["X_train_s"], splits["y_train"],
                          models_ctx["ridge_cv"], top_idx, feat_names)
        fig_08_lasso_selection(models_ctx["lasso_cv"], X)
        fig_09_lasso_cv_path(models_ctx["lasso_cv"])
        fig_10_shrinkage(models_ctx["ols"], models_ctx["ridge_cv"],
                         models_ctx["lasso_cv"], models_ctx["enet_cv"])
        fig_11_rolling_rmse(oos_ctx["oos_df"], oos_ctx["y_oos_ref"], oos_ctx["oos_rmse"])
        fig_12_selection_stability(sel_ctx["sel_freq"], sel_ctx["n_windows"],
                                   models_ctx["lambda_lasso"])
        fig_13_horizons(hor_ctx["df_horizons"])
        fig_14_giacomini_rossi(gr_ctx)

    print_summary(ctx)
    export_results_table(models_ctx["results"], splits["y_test"])
    export_inference_table(inf_ctx["df_inference"], splits["y_test"])
    export_horizons_table(hor_ctx["df_horizons"])
    export_regime_table(
        reg_ctx["df_regime"],
        shock_end=reg_ctx["shock_end"],
        n_shock=reg_ctx["n_shock"],
        n_disfl=reg_ctx["n_disfl"],
    )
    export_sources_table()
    export_gr_table(gr_ctx)
    export_robustness_table(rob_ctx["df_robustness_mom"])
    update_readmes(ctx)

    return ctx
