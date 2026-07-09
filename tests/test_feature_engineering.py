import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from feature_engineering import (
    add_tenure_group, add_spend_features, add_service_usage_score,
    add_engagement_score, add_risk_score, engineer_features,
)


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "tenure": [0, 6, 15, 50],
        "MonthlyCharges": [50.0, 60.0, 70.0, 80.0],
        "TotalCharges": [0.0, 360.0, 1050.0, 4000.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PhoneService": ["Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No", "No internet service", "No"],
        "OnlineBackup": ["No", "No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No", "No internet service", "Yes"],
        "TechSupport": ["Yes", "No", "No internet service", "No"],
        "StreamingTV": ["No", "Yes", "No internet service", "Yes"],
        "StreamingMovies": ["No", "Yes", "No internet service", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)", "Electronic check"],
    })


def test_add_tenure_group_creates_expected_buckets(base_df):
    df = add_tenure_group(base_df.copy())
    assert "TenureGroup" in df.columns
    assert df.loc[0, "TenureGroup"] == "0-6 Months"
    assert df.loc[3, "TenureGroup"] == "4+ Years"


def test_add_spend_features_no_nans(base_df):
    df = add_spend_features(base_df.copy())
    assert not df["AvgMonthlySpend"].isnull().any()
    assert "SpendTrend" in df.columns
    assert "EstimatedFutureValue" in df.columns


def test_add_service_usage_score_within_range(base_df):
    df = add_service_usage_score(base_df.copy())
    assert df["ServiceUsageScore"].between(0, 9).all()


def test_add_engagement_score_within_range(base_df):
    df = add_service_usage_score(base_df.copy())
    df = add_engagement_score(df)
    assert (df["EngagementScore"] >= 0).all()


def test_add_risk_score_within_bounds(base_df):
    df = add_risk_score(base_df.copy())
    assert df["RiskScore"].between(0, 100).all()


def test_month_to_month_has_higher_risk_than_two_year(base_df):
    df = add_risk_score(base_df.copy())
    mtm_risk = df.loc[df["Contract"] == "Month-to-month", "RiskScore"].mean()
    two_year_risk = df.loc[df["Contract"] == "Two year", "RiskScore"].mean()
    assert mtm_risk > two_year_risk


def test_engineer_features_end_to_end_adds_all_columns(base_df):
    df = engineer_features(base_df.copy())
    expected_cols = {
        "TenureGroup", "AvgMonthlySpend", "SpendTrend", "RevenuePerTenure",
        "EstimatedFutureValue", "ServiceUsageScore", "EngagementScore", "RiskScore",
    }
    assert expected_cols.issubset(set(df.columns))
