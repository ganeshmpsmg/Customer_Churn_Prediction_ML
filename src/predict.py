"""
predict.py
----------
Loads trained artifacts (model, preprocessor, SHAP explainer) and exposes
functions for single-customer and batch predictions, complete with
business recommendations.

Usage:
    python src/predict.py --input data/raw/telco_churn.csv --output reports/predictions.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_data, get_feature_columns
from feature_engineering import engineer_features
from explainability import get_top_churn_reasons
from recommendation_engine import build_customer_report, rank_customers_by_priority
from utils import (
    get_logger, load_object, TARGET_COL, ID_COL,
    BEST_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, SHAP_EXPLAINER_PATH,
)

logger = get_logger(__name__)


class ChurnPredictor:
    """A convenience wrapper bundling all trained artifacts for inference."""

    def __init__(self):
        self.model = load_object(BEST_MODEL_PATH)
        self.preprocessor = load_object(PREPROCESSOR_PATH)
        self.feature_names = load_object(FEATURE_NAMES_PATH)
        try:
            self.explainer = load_object(SHAP_EXPLAINER_PATH)
        except FileNotFoundError:
            logger.warning("SHAP explainer artifact not found; explanations will be skipped.")
            self.explainer = None

        # KernelExplainer (used for non-tree models like Logistic Regression) is
        # orders of magnitude slower than TreeExplainer. Cap how many rows we
        # explain per batch call so the dashboard/API stay responsive.
        import shap as _shap
        self.explainer_is_fast = isinstance(self.explainer, _shap.TreeExplainer)
        self.max_explained_rows = 5000 if self.explainer_is_fast else 25

    def _prepare_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy().reset_index(drop=True)
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        # add a dummy target column temporarily for reuse of clean_data (harmless no-op if absent)
        # drop_duplicates=False: inference must preserve row alignment with the caller's IDs
        df = clean_data(df.assign(**{TARGET_COL: 0}), drop_duplicates=False)
        df = df.drop(columns=[TARGET_COL])
        df = engineer_features(df)
        if "TenureGroup" in df.columns:
            df["TenureGroup"] = df["TenureGroup"].astype(str)
        return df

    def predict_batch(self, raw_df: pd.DataFrame, explain: bool = True) -> pd.DataFrame:
        raw_df = raw_df.reset_index(drop=True)
        ids = raw_df[ID_COL] if ID_COL in raw_df.columns else pd.Series(range(len(raw_df)), name=ID_COL)
        features_df = self._prepare_features(raw_df)
        X = self.preprocessor.transform(features_df.drop(columns=[ID_COL], errors="ignore"))

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        n_to_explain = min(len(raw_df), self.max_explained_rows) if (explain and self.explainer is not None) else 0
        if explain and n_to_explain < len(raw_df):
            logger.info(f"Explainer is slow (KernelExplainer); explaining only the first "
                        f"{n_to_explain}/{len(raw_df)} rows. Prediction/risk scores are unaffected.")

        reports = []
        for i in range(len(raw_df)):
            top_reasons = []
            if i < n_to_explain:
                try:
                    top_reasons = get_top_churn_reasons(self.explainer, X[i:i + 1], self.feature_names, top_n=3)
                except Exception as e:
                    logger.warning(f"SHAP explanation failed for row {i}: {e}")

            report = build_customer_report(
                customer_id=ids.iloc[i],
                churn_probability=probs[i],
                monthly_charges=float(raw_df.iloc[i].get("MonthlyCharges", 0) or 0),
                top_reasons=top_reasons,
                tenure=int(raw_df.iloc[i].get("tenure", 0) or 0) if "tenure" in raw_df.columns else None,
            )
            report["prediction"] = "Churn" if preds[i] == 1 else "No Churn"
            reports.append(report)

        result_df = pd.DataFrame(reports)
        return result_df

    def predict_single(self, customer_dict: dict) -> dict:
        raw_df = pd.DataFrame([customer_dict])
        result = self.predict_batch(raw_df)
        return result.iloc[0].to_dict()


def main(input_path: str, output_path: str):
    logger.info(f"Loading predictor artifacts...")
    predictor = ChurnPredictor()

    raw_df = pd.read_csv(input_path)
    logger.info(f"Running batch prediction on {len(raw_df)} customers...")
    results = predictor.predict_batch(raw_df)
    ranked = rank_customers_by_priority(results.to_dict("records"))
    ranked_df = pd.DataFrame(ranked)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ranked_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    logger.info(f"Risk level distribution:\n{ranked_df['risk_level'].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="reports/predictions.csv")
    args = parser.parse_args()
    main(args.input, args.output)
