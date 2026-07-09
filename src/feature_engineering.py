"""
feature_engineering.py
-----------------------
Derives business-meaningful features from the cleaned churn dataset:
tenure groups, spend trends, engagement/risk scores, service usage, etc.
"""
import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger(__name__)

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely maps customer tenure into categorical groups.
    Handles potential 'Tenure' vs 'tenure' casing differences.
    """
    if "tenure" not in df.columns:
        if "Tenure" in df.columns:
            df["tenure"] = df["Tenure"]
        else:
            raise ValueError(
                "Input data is missing the required 'tenure' column."
            )

    # Maintained your script's specific binning intervals and business logic
    bins = [-1, 6, 12, 24, 48, np.inf]
    labels = ["0-6 Months", "6-12 Months", "1-2 Years", "2-4 Years", "4+ Years"]
    
    df["TenureGroup"] = pd.cut(
        df["tenure"], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )
    return df


def add_spend_features(df: pd.DataFrame) -> pd.DataFrame:
    # Average historical monthly spend implied by total charges vs tenure
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, np.nan), df["MonthlyCharges"]
    )
    df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(df["MonthlyCharges"])

    # Spend trend: is the customer currently paying more or less than their historical average?
    df["SpendTrend"] = df["MonthlyCharges"] - df["AvgMonthlySpend"]

    # Revenue-per-tenure-month efficiency feature
    df["RevenuePerTenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    # CLV proxy: total revenue extrapolated by contract length commitment
    contract_multiplier = df.get("Contract", pd.Series(["Month-to-month"] * len(df))).map(
        {"Month-to-month": 1, "One year": 12, "Two year": 24}
    ).fillna(1)
    df["EstimatedFutureValue"] = df["MonthlyCharges"] * contract_multiplier
    return df


def add_service_usage_score(df: pd.DataFrame) -> pd.DataFrame:
    """Counts how many 'add-on' services a customer subscribes to (0-9)."""
    present_cols = [c for c in SERVICE_COLS if c in df.columns]
    df["ServiceUsageScore"] = df[present_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    return df


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    A composite 0-100 engagement score: rewards tenure, add-on services,
    and longer contracts; penalizes paperless/electronic-check-only
    relationships which correlate with lower engagement in telco data.
    """
    tenure_component = (df["tenure"] / df["tenure"].max()).fillna(0) * 40
    service_component = (df["ServiceUsageScore"] / 9) * 30
    contract_component = df.get("Contract", pd.Series(["Month-to-month"] * len(df))).map(
        {"Month-to-month": 0, "One year": 15, "Two year": 30}
    ).fillna(0)
    df["EngagementScore"] = (tenure_component + service_component + contract_component).round(1)
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    A rule-based composite risk score (0-100) used as an additional model
    feature AND as a human-interpretable business metric independent of
    the ML model's own probability output.
    """
    risk = np.zeros(len(df))
    if "Contract" in df.columns:
        risk += df["Contract"].map({"Month-to-month": 35, "One year": 12, "Two year": 0}).fillna(20)
    if "PaymentMethod" in df.columns:
        risk += (df["PaymentMethod"] == "Electronic check") * 15
    if "InternetService" in df.columns:
        risk += (df["InternetService"] == "Fiber optic") * 12
    if "TechSupport" in df.columns:
        risk += (df["TechSupport"] == "No") * 10
    if "OnlineSecurity" in df.columns:
        risk += (df["OnlineSecurity"] == "No") * 8
    risk += np.clip((24 - df["tenure"]) / 24 * 20, 0, 20)  # short tenure = risk
    df["RiskScore"] = np.clip(risk, 0, 100).round(1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features...")
    df = df.copy()
    df = add_tenure_group(df)
    df = add_spend_features(df)
    df = add_service_usage_score(df)
    df = add_engagement_score(df)
    df = add_risk_score(df)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df
