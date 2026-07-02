"""Stage 2: YoY transformation, feature matrix, train/test split, scaling."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import AR_LAGS, LAGS, TEST_MONTHS


def transform_to_yoy(df):
    """Transforms all columns into YoY change rates (%) -> stationary."""
    df_yoy = df.pct_change(12) * 100
    return df_yoy.replace([np.inf, -np.inf], np.nan)


def transform_to_mom(df):
    """Transforms all columns into MoM change rates (%) - alternative to YoY.

    Robustness specification (AP29): checks whether the finding 'RW unbeatable'
    is an artefact of the YoY choice (G31 / Atkeson & Ohanian 2001).
    """
    df_mom = df.pct_change(1) * 100
    return df_mom.replace([np.inf, -np.inf], np.nan)


def build_feature_matrix(df_yoy, lags=None, target_col="HVPI",
                         forecast_horizon=1, test_months=TEST_MONTHS):
    """Builds a leakage-free feature matrix X (lagged predictors) and target y."""
    if lags is None:
        lags = LAGS
    predictor_cols = [c for c in df_yoy.columns if c != target_col]
    frames = []
    for lag in lags:
        lagged = df_yoy[predictor_cols].shift(lag + forecast_horizon - 1)
        lagged.columns = [f"{c}_L{lag}" for c in predictor_cols]
        frames.append(lagged)
    X = pd.concat(frames, axis=1)
    y = df_yoy[target_col]

    # NaN filter: exclude columns with >20% NaN in the training window.
    # The test boundary is determined in the post-dropna index space (not in the
    # pre-dropna index space), because the pre-dropna frame contains NaN rows at the
    # end (predictors without the most recent data), which would shift
    # len(X) - test_months by several months into the actual test window. This way
    # the filter window contains 0 test months.
    combined_pre = pd.concat([X, y], axis=1).dropna()
    if len(combined_pre) > test_months:
        train_cutoff = combined_pre.index[-test_months - 1]
        nan_frac = X.loc[X.index <= train_cutoff].isna().mean()
    else:
        nan_frac = X.isna().mean()
    X = X.loc[:, nan_frac <= 0.20]

    combined = pd.concat([X, y], axis=1).dropna()
    return combined.drop(columns=[target_col]), combined[target_col]


def prepare_splits(X, y, train_end, ar_lags=None):
    """Prepares all data splits and scalings, returns a ctx dict.

    Contains:
    - main split (X_train/X_test/y_train/y_test, scaled)
    - ADL benchmark (X_ar, sc_ar, ar_model inputs)
    - LASSO+HVPI macro value-added (X_plus, sc_plus)
    """
    if ar_lags is None:
        ar_lags = AR_LAGS

    # --- Main split ---
    X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # --- ADL (HVPI own lags) ---
    X_ar = pd.DataFrame({f"HVPI_L{l}": y.shift(l) for l in ar_lags})
    X_ar = X_ar.loc[y.index].dropna()
    y_ar = y.loc[X_ar.index]

    X_ar_train = X_ar.loc[X_ar.index <= y_train.index[-1]]
    X_ar_test  = X_ar.loc[X_ar.index.isin(y_test.index)]
    y_ar_train = y_ar.loc[X_ar_train.index]

    sc_ar = StandardScaler()

    # --- LASSO + HVPI own lags (macro value-added) ---
    X_plus = X.copy()
    for l in ar_lags:
        X_plus[f"HVPI_L{l}"] = y.shift(l)
    X_plus = X_plus.loc[y.index].dropna()
    y_plus = y.loc[X_plus.index]

    X_plus_train = X_plus.loc[X_plus.index <= y_train.index[-1]]
    X_plus_test  = X_plus.loc[X_plus.index.isin(y_test.index)]
    y_plus_train = y_plus.loc[X_plus_train.index]

    sc_plus        = StandardScaler()
    X_plus_train_s = sc_plus.fit_transform(X_plus_train)
    X_plus_test_s  = sc_plus.transform(X_plus_test)

    # Start indices for rolling origin (first OOS period)
    test_start = y_test.index[0]
    start_ar   = int((X_ar.index >= test_start).argmax())
    start_plus = int((X_plus.index >= test_start).argmax())

    return {
        # Main
        "X_train":    X_train,   "X_test":    X_test,
        "y_train":    y_train,   "y_test":    y_test,
        "X_train_s":  X_train_s, "X_test_s":  X_test_s,
        "scaler":     scaler,
        # ADL
        "X_ar":       X_ar,      "y_ar":      y_ar,
        "X_ar_train": X_ar_train,"X_ar_test": X_ar_test,
        "y_ar_train": y_ar_train,"sc_ar":     sc_ar,
        "start_ar":   start_ar,
        # LASSO+HVPI
        "X_plus":        X_plus,       "y_plus":        y_plus,
        "X_plus_train":  X_plus_train, "X_plus_test":   X_plus_test,
        "X_plus_train_s":X_plus_train_s,"X_plus_test_s":X_plus_test_s,
        "y_plus_train":  y_plus_train, "sc_plus":       sc_plus,
        "start_plus":    start_plus,
    }
