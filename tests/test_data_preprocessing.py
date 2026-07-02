"""Smoke tests: leakage and shape invariants for data_preprocessing."""
import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data_preprocessing import (
    build_feature_matrix,
    prepare_splits,
    transform_to_yoy,
)


# --- Fixtures ---

@pytest.fixture
def synthetic_df():
    """Simple synthetic dataset (noise, monthly 2000-2020)."""
    np.random.seed(0)
    idx = pd.date_range("2000-01", periods=252, freq="MS")
    df  = pd.DataFrame(
        np.random.randn(252, 5),
        index=idx,
        columns=["HVPI", "IP_A", "BS_B", "PPI_C", "ALQ_D"],
    )
    return df


@pytest.fixture
def yoy_and_features(synthetic_df):
    df_yoy = transform_to_yoy(synthetic_df)
    X, y   = build_feature_matrix(df_yoy, lags=[1, 2], forecast_horizon=1, test_months=24)
    return X, y


# --- Tests ---

def test_yoy_no_inf(synthetic_df):
    df_yoy = transform_to_yoy(synthetic_df)
    assert not np.isinf(df_yoy.values).any(), "YoY matrix contains Inf values"


def test_feature_matrix_no_nan(yoy_and_features):
    X, y = yoy_and_features
    assert not X.isna().any().any(), "Feature matrix X contains NaN after dropna"
    assert not y.isna().any(),       "Target variable y contains NaN after dropna"


def test_feature_matrix_shape(yoy_and_features):
    X, y = yoy_and_features
    assert X.shape[0] == len(y), "X and y have different row counts"
    assert X.shape[1] > 0,       "Feature matrix is empty"


def test_split_disjoint(yoy_and_features):
    X, y       = yoy_and_features
    train_end  = len(y) - 24
    splits     = prepare_splits(X, y, train_end)
    X_train    = splits["X_train"]
    X_test     = splits["X_test"]
    assert X_train.index[-1] < X_test.index[0], \
        "Train end does not lie before test start (time-series leak)"
    assert len(set(X_train.index) & set(X_test.index)) == 0, \
        "Train and test indices overlap"


def test_split_sizes(yoy_and_features):
    X, y      = yoy_and_features
    train_end = len(y) - 24
    splits    = prepare_splits(X, y, train_end)
    assert len(splits["y_train"]) == train_end, "Training set has wrong length"
    assert len(splits["y_test"])  == 24,        "Test set has wrong length"


def test_scaler_train_std(yoy_and_features):
    X, y      = yoy_and_features
    train_end = len(y) - 24
    splits    = prepare_splits(X, y, train_end)
    X_train_s = splits["X_train_s"]
    # Column std in the scaled training set should be ≈ 1
    std_max = np.abs(X_train_s.std(axis=0) - 1).max()
    assert std_max < 1e-10, f"Scaling error: max |std-1| = {std_max:.2e}"


def test_nan_filter_only_on_train():
    """NaN filter must not use test months as a missingness criterion.

    Constructs a column B (canary) that has NaN only in the actual test window
    (post-dropna last test_months rows).  The pre-dropna frame contains NaN rows
    at the end (column A ends earlier) => len(X)-test_months lands too late =>
    the canary NaN block falls inside the old filter window.

    Buggy:  B_L1 NaN share in the pre-dropna window ≈ 21 % > 20 % => B is
            filtered out; assert fails.
    Correct: B_L1 NaN share in the actual training window ≈ 4 % < 20 % => B stays.
    """
    np.random.seed(1)
    n            = 60
    test_months_t = 12
    idx          = pd.date_range("2000-01", periods=n, freq="MS")

    # A: NaN for last 10 rows → pre-dropna frame has 10 trailing NaN rows,
    #    so len(X)-12 = 48 includes rows 38-47, which overlap the canary NaN block.
    a_vals = np.concatenate([np.random.randn(50), np.full(10, np.nan)])

    # B (canary): NaN for rows 38-49 (12 rows = the actual post-dropna test period).
    # In old filter (rows 0-47): rows 0+39..47 → 10 NaN / 48 rows ≈ 20.8 % → excluded.
    # In new filter (rows 0-26, true training): row 0 only → 1/27 ≈  3.7 % → kept.
    b_vals = np.concatenate([
        np.random.randn(38),
        np.full(12, np.nan),
        np.random.randn(10),
    ])

    df = pd.DataFrame(
        {"HVPI": np.random.randn(n), "A": a_vals, "B": b_vals}, index=idx
    )

    X, y = build_feature_matrix(df, lags=[1], forecast_horizon=1,
                                test_months=test_months_t)

    assert any("B" in col for col in X.columns), (
        "Column B must not be filtered out: its NaN lies only in the test period. "
        "The NaN-filter must use the post-dropna training boundary, not len(X)-test_months."
    )
