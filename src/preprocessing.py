"""
preprocessing.py
----------------
Data loading, validation, cleaning, and the sklearn ColumnTransformer used
to encode/scale features before modeling.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import get_logger, TARGET_COL, ID_COL

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# 1. Load & validate
# ----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    validate_data(df)
    return df


def validate_data(df: pd.DataFrame):
    """Basic data-quality checks; raises if something is fundamentally broken."""
    required_cols = {ID_COL, TARGET_COL, "tenure", "MonthlyCharges"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    n_dupes = df.duplicated().sum()
    n_missing = df.isnull().sum().sum()
    logger.info(f"Validation: {len(df)} rows, {n_dupes} duplicate rows, "
                f"{n_missing} total missing values.")


# ----------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------
def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Drop exact duplicates (skip at inference time to keep row alignment with caller's IDs)
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before - len(df)} duplicate rows.")

    # TotalCharges sometimes arrives as a blank string / needs numeric coercion
    if "TotalCharges" not in df.columns:
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
        else:
            df["TotalCharges"] = 0

    if not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute numeric missing values with median
    for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

    # Standardize SeniorCitizen to Yes/No-style int (already 0/1 typically)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Normalize target to binary int (handles object/string/category dtypes)
    if TARGET_COL in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        mapped = df[TARGET_COL].astype(str).map({"Yes": 1, "No": 0})
        df[TARGET_COL] = mapped.fillna(df[TARGET_COL])

    # Outlier handling on continuous billing columns via IQR capping (winsorize)
    for col in ["MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower, upper)
            if n_outliers:
                logger.info(f"Capped {n_outliers} outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")

    return df


# ----------------------------------------------------------------------
# 3. Column typing helpers
# ----------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame):
    """Split feature columns into numeric and categorical, excluding ID/target."""
    exclude = {ID_COL, TARGET_COL}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    categorical_cols = [c for c in df.select_dtypes(include=["object", "string", "category"]).columns
                         if c not in exclude]
    return numeric_cols, categorical_cols


# ----------------------------------------------------------------------
# 4. Build preprocessing pipeline (encode + scale)
# ----------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer):
    """Return human-readable feature names after a ColumnTransformer.transform()."""
    return list(preprocessor.get_feature_names_out())
