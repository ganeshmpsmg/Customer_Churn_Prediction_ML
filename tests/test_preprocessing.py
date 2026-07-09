import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from preprocessing import clean_data, validate_data, get_feature_columns, build_preprocessor


@pytest.fixture
def sample_df():
    n_normal = 20
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "customerID": [f"A{i}" for i in range(n_normal)],
        "gender": rng.choice(["Male", "Female"], n_normal).tolist(),
        "SeniorCitizen": rng.choice([0, 1], n_normal).tolist(),
        "tenure": rng.integers(1, 60, n_normal).tolist(),
        "MonthlyCharges": rng.normal(65, 5, n_normal).round(2).tolist(),
        "TotalCharges": [str(x) for x in rng.normal(1500, 300, n_normal).round(2)],
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n_normal).tolist(),
        "Churn": rng.choice(["Yes", "No"], n_normal).tolist(),
    })
    # Inject one clear outlier and one duplicate + one missing value
    df.loc[0, "MonthlyCharges"] = 500.0  # far outside the ~65±5 distribution
    df.loc[len(df)] = df.loc[1]  # exact duplicate row
    df.loc[2, "TotalCharges"] = np.nan
    return df


def test_validate_data_passes_with_required_columns(sample_df):
    validate_data(sample_df)  # should not raise


def test_validate_data_raises_on_missing_columns():
    bad_df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(ValueError):
        validate_data(bad_df)


def test_clean_data_removes_duplicates(sample_df):
    before = len(sample_df)
    cleaned = clean_data(sample_df)
    assert len(cleaned) == before - 1  # one duplicate row removed


def test_clean_data_imputes_missing_total_charges(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["TotalCharges"].isnull().sum() == 0


def test_clean_data_encodes_target_as_binary(sample_df):
    cleaned = clean_data(sample_df)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})


def test_clean_data_caps_outliers(sample_df):
    cleaned = clean_data(sample_df)
    # The 500 outlier in MonthlyCharges should be capped below its original value
    assert cleaned["MonthlyCharges"].max() < 500.0


def test_get_feature_columns_splits_numeric_and_categorical(sample_df):
    cleaned = clean_data(sample_df)
    numeric_cols, categorical_cols = get_feature_columns(cleaned)
    assert "MonthlyCharges" in numeric_cols
    assert "Contract" in categorical_cols
    assert "Churn" not in numeric_cols and "Churn" not in categorical_cols
    assert "customerID" not in numeric_cols and "customerID" not in categorical_cols


def test_build_preprocessor_transforms_shapes(sample_df):
    cleaned = clean_data(sample_df)
    numeric_cols, categorical_cols = get_feature_columns(cleaned)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X = cleaned.drop(columns=["Churn", "customerID"])
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(cleaned)
    assert transformed.shape[1] > 0
