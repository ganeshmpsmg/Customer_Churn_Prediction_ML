"""
segmentation.py
----------------
Bonus feature: unsupervised customer segmentation using KMeans clustering
on engineered behavioral/billing features. Complements the supervised
churn model by grouping customers into interpretable personas (e.g.
"high-value loyal", "at-risk high-spender", "new low-engagement") for
targeted retention campaigns.

Usage:
    python src/segmentation.py --data data/raw/telco_churn.csv --k 4
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, clean_data
from feature_engineering import engineer_features
from utils import get_logger, save_object, MODELS_DIR, REPORTS_DIR

logger = get_logger(__name__)

SEGMENTATION_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "ServiceUsageScore",
    "EngagementScore", "RiskScore",
]

def find_best_k(X_scaled, k_range=range(2, 7)):
    best_k, best_score = 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        score = silhouette_score(X_scaled, km.labels_)
        logger.info(f"k={k} silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _assign_persona_labels(df: pd.DataFrame, features: list) -> pd.Series:
    """
    KMeans cluster indices are arbitrary, so we derive a human-readable,
    data-driven label per cluster from its actual tenure/spend/risk
    profile relative to the other clusters, rather than guessing a fixed
    persona name that might not match the data.
    """
    stats = df.groupby("Segment")[features].mean(numeric_only=True)
    tenure_median = stats["tenure"].median()
    spend_median = stats["MonthlyCharges"].median()
    risk_median = stats["RiskScore"].median() if "RiskScore" in stats.columns else stats["MonthlyCharges"].median()

    labels = {}
    for seg, row in stats.iterrows():
        tenure_desc = "New" if row["tenure"] < tenure_median else "Long-tenure"
        spend_desc = "High-spend" if row["MonthlyCharges"] >= spend_median else "Low-spend"
        risk_desc = "High-risk" if row.get("RiskScore", 0) >= risk_median else "Low-risk"
        labels[seg] = f"{tenure_desc}, {spend_desc}, {risk_desc}"
    return df["Segment"].map(labels)


def segment_customers(df: pd.DataFrame, k: int = None):
    features = [c for c in SEGMENTATION_FEATURES if c in df.columns]
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if k is None:
        k = find_best_k(X_scaled)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["Segment"] = labels
    df["SegmentLabel"] = _assign_persona_labels(df, features)

    save_object(scaler, os.path.join(MODELS_DIR, "segmentation_scaler.joblib"))
    save_object(kmeans, os.path.join(MODELS_DIR, "segmentation_kmeans.joblib"))

    summary = df.groupby("SegmentLabel")[features + ["Churn"]].mean(numeric_only=True) \
        if "Churn" in df.columns else df.groupby("SegmentLabel")[features].mean()
    logger.info(f"Segment summary:\n{summary}")
    return df, summary


def main(data_path: str, k: int):
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    df, summary = segment_customers(df, k=k if k > 0 else None)

    out_path = os.path.join(REPORTS_DIR, "customer_segments.csv")
    df.to_csv(out_path, index=False)
    summary_path = os.path.join(REPORTS_DIR, "segment_summary.csv")
    summary.to_csv(summary_path)
    logger.info(f"Saved segmented customers to {out_path} and summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters (0 = auto-select via silhouette score)")
    args = parser.parse_args()
    main(args.data, args.k)
