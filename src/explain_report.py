"""
explain_report.py
------------------
Generates global SHAP explainability plots (summary plot, feature
importance bar chart) from the saved model/preprocessor/explainer and
a sample of the training data. Run after train.py.

Usage:
    python src/explain_report.py --data data/raw/telco_churn.csv --sample 300
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, clean_data, get_feature_columns
from feature_engineering import engineer_features
from explainability import plot_summary, plot_feature_importance_bar
from utils import (
    get_logger, load_object, TARGET_COL, ID_COL,
    PREPROCESSOR_PATH, FEATURE_NAMES_PATH, SHAP_EXPLAINER_PATH, REPORTS_DIR,
)

logger = get_logger(__name__)


def main(data_path: str, sample_size: int):
    preprocessor = load_object(PREPROCESSOR_PATH)
    feature_names = load_object(FEATURE_NAMES_PATH)
    explainer = load_object(SHAP_EXPLAINER_PATH)

    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    if "TenureGroup" in df.columns:
        df["TenureGroup"] = df["TenureGroup"].astype(str)

    X_raw = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    sample = X_raw.sample(min(sample_size, len(X_raw)), random_state=42)
    X = preprocessor.transform(sample)

    logger.info("Generating global SHAP summary plot...")
    summary_path = plot_summary(explainer, X, feature_names, os.path.join(REPORTS_DIR, "shap_summary.png"))
    logger.info(f"Saved: {summary_path}")

    logger.info("Generating global SHAP feature importance bar chart...")
    imp_path = plot_feature_importance_bar(explainer, X, feature_names,
                                            os.path.join(REPORTS_DIR, "shap_feature_importance.png"))
    logger.info(f"Saved: {imp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()
    main(args.data, args.sample)
