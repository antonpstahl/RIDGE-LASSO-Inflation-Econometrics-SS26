"""Smoke tests: DM test and RMSE sanity for evaluation."""
import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sklearn.model_selection import TimeSeriesSplit

from src.evaluation import (
    bonferroni_correct, clark_west, compute_giacomini_rossi,
    compute_regime_analysis, compute_robustness_extended_oos,
    compute_robustness_mom, compute_selection_by_regime,
    diebold_mariano, rolling_origin,
)


# --- Diebold-Mariano ---

def test_dm_perfect_model():
    """Perfect model (e_mod=0) should have a positive DM statistic."""
    np.random.seed(0)
    e_rw  = np.random.randn(36)
    e_mod = np.zeros(36)
    dm, pv = diebold_mariano(e_rw, e_mod, h=1)
    assert dm > 0, "DM stat should be positive when model is perfect"
    assert 0 <= pv <= 1, f"p-value outside [0,1]: {pv}"


def test_dm_rw_is_rw():
    """Identical errors (model = RW) should give DM ≈ 0."""
    np.random.seed(1)
    e_rw = np.random.randn(36)
    dm, pv = diebold_mariano(e_rw, e_rw, h=1)
    assert abs(dm) < 1e-10, f"DM should be 0 for equal errors, got {dm}"


def test_dm_worse_model():
    """Worse model (larger errors) should have a negative DM statistic."""
    np.random.seed(2)
    e_rw  = np.random.randn(36) * 1.0
    e_mod = np.random.randn(36) * 2.0   # larger errors
    dm, pv = diebold_mariano(e_rw, e_mod, h=1)
    assert dm < 0, "DM stat should be negative when model is worse than RW"
    assert 0 <= pv <= 1


def test_dm_p_value_range():
    """p-value must lie in [0,1]."""
    e1 = np.linspace(-1, 1, 36)
    e2 = np.linspace(-2, 2, 36)
    _, pv = diebold_mariano(e1, e2, h=1)
    assert 0.0 <= pv <= 1.0


# --- Clark-West ---

def test_cw_perfect_nested_model():
    """Perfect nested model (e_mod=0) → CW stat positive, p-value in [0,1]."""
    np.random.seed(10)
    e_rw  = np.random.randn(36)
    e_mod = np.zeros(36)
    cw, pv = clark_west(e_rw, e_mod, h=1)
    assert cw > 0, f"CW stat should be positive for a perfect model, got {cw}"
    assert 0.0 <= pv <= 1.0, f"p-value outside [0,1]: {pv}"


def test_cw_equal_errors_zero_stat():
    """If nested model = RW (e_mod = e_rw), then f_t = 0 for all t → CW ≈ 0."""
    np.random.seed(11)
    e_rw = np.random.randn(36)
    cw, pv = clark_west(e_rw, e_rw, h=1)
    assert abs(cw) < 1e-10, f"CW should be 0 for equal errors, got {cw}"


def test_cw_p_value_one_sided():
    """CW p-value is one-sided: for a positive CW stat, p < 0.5."""
    np.random.seed(12)
    e_rw  = np.random.randn(100) * 2.0   # larger RW errors
    e_mod = np.random.randn(100) * 0.5   # smaller model errors
    cw, pv = clark_west(e_rw, e_mod, h=1)
    assert cw > 0, "CW stat should be positive when model is clearly better"
    assert 0.0 <= pv < 0.5, f"One-sided p-value for positive CW should be < 0.5, got {pv}"


def test_cw_adjustment_exceeds_dm_numerator():
    """CW numerator (mean f_t^CW) is >= DM numerator (mean f_t^DM).

    f_t^CW = f_t^DM + (e_mod - e_rw)^2  => mean(f^CW) >= mean(f^DM).
    """
    np.random.seed(13)
    e_rw  = np.random.randn(50)
    e_mod = np.random.randn(50)
    f_dm  = e_rw**2 - e_mod**2
    f_cw  = e_rw**2 - (e_mod**2 - (e_mod - e_rw)**2)
    assert f_cw.mean() >= f_dm.mean() - 1e-12, (
        f"CW numerator {f_cw.mean():.6f} should be >= DM numerator {f_dm.mean():.6f}"
    )


# --- Bonferroni correction ---

def test_bonferroni_correct_basic():
    """p_adj_i = min(n * p_i, 1.0) for all non-NaN entries."""
    p_values = [0.05, 0.10, 0.02]   # n = 3
    p_adj = bonferroni_correct(p_values)
    expected = [min(3 * p, 1.0) for p in p_values]
    np.testing.assert_allclose(p_adj, expected)


def test_bonferroni_correct_nan_passthrough():
    """NaN entries (reference model without test) are passed through unchanged."""
    p_values = [np.nan, 0.05, 0.10]   # n_valid = 2
    p_adj = bonferroni_correct(p_values)
    assert np.isnan(p_adj[0]), "NaN entry should remain unchanged"
    np.testing.assert_allclose(p_adj[1], min(2 * 0.05, 1.0))
    np.testing.assert_allclose(p_adj[2], min(2 * 0.10, 1.0))


def test_bonferroni_correct_clamp():
    """Corrected p-values are capped at 1.0."""
    p_values = [0.8, 0.9]   # 2 * 0.8 = 1.6 > 1.0
    p_adj = bonferroni_correct(p_values)
    assert all(v <= 1.0 for v in p_adj), "No p_adj may exceed 1.0"
    np.testing.assert_allclose(p_adj, [1.0, 1.0])


# --- Rolling-Origin ---

def test_rolling_origin_length():
    """Rolling-Origin should return len(y) - start forecasts."""
    np.random.seed(3)
    n   = 50; start = 30
    X   = pd.DataFrame(np.random.randn(n, 3))
    y   = pd.Series(np.random.randn(n))
    from sklearn.linear_model import LinearRegression
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    assert len(preds) == n - start, \
        f"Expected {n - start} forecasts, got {len(preds)}"


def test_rolling_origin_index_alignment():
    """Forecast index should match the y index from start onwards."""
    np.random.seed(4)
    idx = pd.date_range("2020-01", periods=40, freq="MS")
    X   = pd.DataFrame(np.random.randn(40, 2), index=idx)
    y   = pd.Series(np.random.randn(40), index=idx)
    from sklearn.linear_model import LinearRegression
    start = 25
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    expected_idx = idx[start:]
    pd.testing.assert_index_equal(preds.index, expected_idx)


# --- Regime analysis ---

def test_regime_analysis_rw_reference():
    """RW has RMSE/RW = 1.0 in both regimes (self-reference)."""
    idx = pd.date_range("2021-06", periods=40, freq="MS")
    rng = np.random.default_rng(42)
    n   = len(idx)
    y   = pd.Series(rng.standard_normal(n), index=idx)
    rw  = pd.Series(rng.standard_normal(n), index=idx)
    ar  = pd.Series(rng.standard_normal(n), index=idx)
    oos_ctx = {
        "oos_df":    pd.DataFrame({"RW": rw, "AR": ar}),
        "y_oos_ref": y,
    }
    result = compute_regime_analysis(oos_ctx, shock_end="2023-03")
    df     = result["df_regime"]
    assert np.isclose(df.loc["RW", "RMSE/RW Shock"],   1.0, atol=1e-10), \
        "RW RMSE/RW Shock should be exactly 1.0"
    assert np.isclose(df.loc["RW", "RMSE/RW Disinfl."], 1.0, atol=1e-10), \
        "RW RMSE/RW Disinfl. should be exactly 1.0"
    assert np.isclose(df.loc["RW", "RMSE/RW Total"],   1.0, atol=1e-10), \
        "RW RMSE/RW Total should be exactly 1.0"


def test_regime_analysis_disjoint_split():
    """n_shock + n_disfl == n (regime split is disjoint and complete)."""
    idx = pd.date_range("2021-06", periods=40, freq="MS")
    rng = np.random.default_rng(7)
    n   = len(idx)
    y   = pd.Series(rng.standard_normal(n), index=idx)
    oos_ctx = {
        "oos_df": pd.DataFrame({
            "RW": rng.standard_normal(n),
            "AR": rng.standard_normal(n),
        }, index=idx),
        "y_oos_ref": y,
    }
    result = compute_regime_analysis(oos_ctx, shock_end="2023-03")
    assert result["n_shock"] + result["n_disfl"] == n, \
        "n_shock + n_disfl must equal total n"
    assert result["n_shock"] > 0, "Shock regime must not be empty"
    assert result["n_disfl"] > 0, "Disinflation regime must not be empty"


# --- Giacomini-Rossi fluctuation test ---

def _make_gr_ctx(y_vals, rw_vals, model_vals, model_name="M",
                 start="2021-06", freq="MS"):
    """Helper function: builds a minimal oos_ctx for compute_giacomini_rossi."""
    T   = len(y_vals)
    idx = pd.date_range(start, periods=T, freq=freq)
    return {
        "oos_df": pd.DataFrame(
            {"RW": rw_vals, model_name: model_vals}, index=idx
        ),
        "y_oos_ref": pd.Series(y_vals, index=idx),
    }


def test_gr_constant_advantage_always_positive():
    """Constant model advantage (d_t > 0 throughout) → all GR statistics positive.

    DoD sanity 1: 'constant advantage ⇒ flat statistic' (flat and positive).
    """
    rng = np.random.default_rng(42)
    T   = 36
    y   = np.zeros(T)
    rw  = np.ones(T) * 1.0   # RW error = 1
    mod = np.ones(T) * 0.3   # model error = 0.3 (clearly better)
    # Small noise so that the HAC variance > 0
    noise = rng.normal(0, 0.02, T)
    rw  = rw + noise
    mod = mod + noise * 0.5

    oos_ctx = _make_gr_ctx(y, rw, mod, model_name="AR")
    gr_ctx  = compute_giacomini_rossi(oos_ctx, m=12)
    gr_ar   = gr_ctx["gr_df"]["AR"]

    assert (gr_ar > 0).all(), (
        f"All GR statistics should be positive under a constant advantage; "
        f"min={gr_ar.min():.3f}"
    )


def test_gr_regime_break_sign_change():
    """Sign change at a constructed regime break → GR_t changes sign.

    DoD sanity 2: 'sign change at a constructed regime break ⇒ band exceedance'.
    We check at least that the GR series contains both positive and negative
    values - a sign change requires an advantage/disadvantage switch.
    """
    T   = 36
    y   = np.zeros(T)
    rw  = np.ones(T)
    # First half: model worse (d_t < 0); second half: model better (d_t > 0)
    mod = np.empty(T)
    mod[:T // 2]  = 0.1   # model almost equal to RW (but worse in the square)
    mod[T // 2:]  = 2.0   # model much worse → d_t negative in 2nd half

    # We flip: d_t = e_RW^2 - e_M^2. If RW=1 and mod=0.1 → d=1-0.01=0.99 (positive).
    # If RW=1 and mod=2 → d=1-4=-3 (negative). So the first half has d>0, the second d<0.
    oos_ctx = _make_gr_ctx(y, rw, mod, model_name="LASSO+HVPI")
    gr_ctx  = compute_giacomini_rossi(oos_ctx, m=10)
    gr_vals = gr_ctx["gr_df"]["LASSO+HVPI"].values

    has_positive = bool((gr_vals > 0).any())
    has_negative = bool((gr_vals < 0).any())
    assert has_positive and has_negative, (
        f"GR series should change sign at a regime break; "
        f"min={gr_vals.min():.3f}, max={gr_vals.max():.3f}"
    )


def test_rolling_origin_no_lookahead():
    """Forecast at time t may only use data up to t-1 (no look-ahead).

    We set y[0:start] = 0, y[start:] = 1. A correct expanding-window model
    trains on [0:start] and predicts a value near 0 for t=start,
    not the true value 1.
    """
    np.random.seed(5)
    n = 60; start = 40
    X = pd.DataFrame(np.random.randn(n, 1))
    y = pd.Series([0.0] * start + [1.0] * (n - start))
    from sklearn.linear_model import LinearRegression
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    # First forecast (trained only on zeros) should be near 0, not 1
    first_pred = preds.iloc[0]
    assert abs(first_pred) < 0.5, \
        f"Look-ahead suspicion: first forecast = {first_pred:.3f} (expected ≈ 0)"


# --- MoM robustness check (AP29) ---

def _make_raw_df(T=220, n_pred=2, seed=99):
    """Synthetic df_raw: HVPI + n_pred predictor price-level series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2004-01", periods=T, freq="MS")
    data = {"HVPI": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.003, T)))}
    for i in range(n_pred):
        data[f"P{i}"] = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.004, T)))
    return pd.DataFrame(data, index=idx)


def test_robustness_mom_output_structure():
    """compute_robustness_mom returns dict with DataFrame (structure smoke test)."""
    df_raw = _make_raw_df()
    result = compute_robustness_mom(df_raw, test_months=20)

    assert "df_robustness_mom" in result, "Key 'df_robustness_mom' missing"
    df = result["df_robustness_mom"]
    assert isinstance(df, pd.DataFrame), "df_robustness_mom must be a DataFrame"


def test_robustness_mom_expected_models():
    """All six models (RW, AO, AR, Ridge, LASSO, LASSO+HVPI) are present."""
    df_raw = _make_raw_df()
    df = compute_robustness_mom(df_raw, test_months=20)["df_robustness_mom"]

    expected = {"RW", "AO (Atkeson-Ohanian)", "AR", "Ridge", "LASSO", "LASSO+HVPI"}
    assert set(df.index) == expected, (
        f"Missing/unexpected models: {set(df.index).symmetric_difference(expected)}"
    )


def test_robustness_mom_rw_self_reference():
    """RW has RMSE/RW = 1.0 (self-reference)."""
    df_raw = _make_raw_df()
    df = compute_robustness_mom(df_raw, test_months=20)["df_robustness_mom"]
    assert np.isclose(df.loc["RW", "RMSE/RW"], 1.0, atol=1e-10), (
        f"RW RMSE/RW should be exactly 1.0, got {df.loc['RW', 'RMSE/RW']}"
    )


def test_robustness_mom_ao_self_reference():
    """AO has RMSE/AO = 1.0 (self-reference)."""
    df_raw = _make_raw_df()
    df = compute_robustness_mom(df_raw, test_months=20)["df_robustness_mom"]
    assert np.isclose(df.loc["AO (Atkeson-Ohanian)", "RMSE/AO"], 1.0, atol=1e-10), (
        f"AO RMSE/AO should be exactly 1.0, got {df.loc['AO (Atkeson-Ohanian)', 'RMSE/AO']}"
    )


def test_robustness_mom_positive_rmse():
    """All RMSE values are finite and positive."""
    df_raw = _make_raw_df()
    df = compute_robustness_mom(df_raw, test_months=20)["df_robustness_mom"]
    rmse_col = df["Test RMSE (MoM)"]
    assert rmse_col.notna().all(), "No NaN values expected in RMSE column"
    assert (rmse_col > 0).all(), f"All RMSE values must be positive:\n{rmse_col}"


# --- compute_selection_by_regime (AP30) ---

def _make_selection_data():
    """Minimal synthetic dataset: PPI/ALQ/IP groups, 50 points."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    X = pd.DataFrame({
        "PPI_Test_L1":  np.random.randn(n),
        "PPI_Test_L2":  np.random.randn(n),
        "ALQ_Test_L1":  np.random.randn(n),
        "IP_Test_L1":   np.random.randn(n),
        "LCI_Test_L1":  np.random.randn(n),
        "BS_Test_L1":   np.random.randn(n),
    }, index=dates)
    y = pd.Series(np.random.randn(n), index=dates)
    return X, y


def test_selection_by_regime_partition():
    """n_shock_sel + n_disfl_sel must equal the OOS windows."""
    X, y = _make_selection_data()
    train_end = 36
    ctx = compute_selection_by_regime(X, y, train_end, lambda_lasso=0.1,
                                      shock_end="2021-06")
    assert ctx["n_shock_sel"] + ctx["n_disfl_sel"] == len(y) - train_end


def test_selection_by_regime_groups_present():
    """df_sel_groups must contain all present group labels."""
    X, y = _make_selection_data()
    train_end = 36
    ctx = compute_selection_by_regime(X, y, train_end, lambda_lasso=0.1,
                                      shock_end="2021-06")
    df = ctx["df_sel_groups"]
    assert set(["Total", "Shock", "Disinflation"]) == set(df.columns), (
        f"Expected columns: Total/Shock/Disinflation, got: {list(df.columns)}"
    )
    expected_groups = {
        "PPI (producer prices/cost-push)", "ALQ (labour market/Phillips)",
        "IP (industrial production)", "LCI (labour cost/cost-push)",
        "BS (business expectations)",
    }
    assert expected_groups.issubset(set(df.index)), (
        f"Expected groups missing: {expected_groups - set(df.index)}"
    )


def test_selection_by_regime_freq_in_unit_interval():
    """All selection frequencies must lie in [0, 1]."""
    X, y = _make_selection_data()
    train_end = 36
    ctx = compute_selection_by_regime(X, y, train_end, lambda_lasso=0.1,
                                      shock_end="2021-06")
    df = ctx["df_sel_groups"]
    assert (df >= 0).all().all() and (df <= 1).all().all(), (
        f"Frequencies outside [0,1]:\n{df}"
    )


# --- Sample extension / post-shock OOS (AP32) ---

def _make_yoy_df(end="2025-08", early_end="2024-06", n_pred=3, seed=7):
    """Synthetic YoY DataFrame with an early-ending (binding) series."""
    idx = pd.date_range("2006-01-01", end, freq="MS")
    rng = np.random.default_rng(seed)
    T   = len(idx)
    hvpi = 2.0 + np.cumsum(rng.normal(0, 0.05, T)) + rng.normal(0, 0.3, T)
    data = {"HVPI": hvpi}
    for i in range(n_pred):
        data[f"P{i}"] = 0.4 * hvpi + rng.normal(0, 1.2, T)
    df = pd.DataFrame(data, index=idx)
    df["BS_early"] = rng.normal(0, 1.0, T)
    df.loc[df.index > pd.Timestamp(early_end), "BS_early"] = np.nan  # binding series
    return df


# Fast CV split for the tests (instead of config.TSCV with n_splits=10)
_FAST_TSCV = TimeSeriesSplit(n_splits=3, test_size=6)


def test_extended_oos_output_structure():
    """compute_robustness_extended_oos returns DataFrame with all key models."""
    df = _make_yoy_df()
    res = compute_robustness_extended_oos(
        df, drop_cols=("BS_early",), test_months=24, tscv=_FAST_TSCV,
    )
    assert "df_robustness_extended" in res
    out = res["df_robustness_extended"]
    assert isinstance(out, pd.DataFrame)
    expected = {"RW", "AR", "OLS", "Ridge", "LASSO", "Elastic Net", "LASSO+HVPI"}
    assert expected.issubset(set(out.index)), (
        f"Missing models: {expected - set(out.index)}"
    )


def test_extended_oos_extends_sample():
    """Dropping the binding series extends the sample (months_gained > 0)."""
    df = _make_yoy_df(end="2025-08", early_end="2024-06")
    res = compute_robustness_extended_oos(
        df, drop_cols=("BS_early",), test_months=24, tscv=_FAST_TSCV,
    )
    assert res["months_gained"] > 0, "Sample should extend"
    assert res["ext_end"] > res["orig_end"], "ext_end must lie after orig_end"
    assert res["dropped"] == ["BS_early"]


def test_extended_oos_rw_self_reference():
    """RW has RMSE/RW = 1.0 in both regimes and overall (self-reference)."""
    df = _make_yoy_df()
    out = compute_robustness_extended_oos(
        df, drop_cols=("BS_early",), test_months=24, tscv=_FAST_TSCV,
    )["df_robustness_extended"]
    for col in ["RMSE/RW Shock", "RMSE/RW Post", "RMSE/RW Total"]:
        assert np.isclose(out.loc["RW", col], 1.0, atol=1e-10), (
            f"RW {col} should be exactly 1.0, got {out.loc['RW', col]}"
        )


def test_extended_oos_regime_partition_nonempty():
    """Both regime segments (shock, post-shock) are non-empty in the test window."""
    df = _make_yoy_df()
    res = compute_robustness_extended_oos(
        df, drop_cols=("BS_early",), test_months=24, tscv=_FAST_TSCV,
    )
    assert res["n_shock"] > 0, "Shock segment must not be empty"
    assert res["n_post"] > 0, "Post-shock segment must not be empty"
