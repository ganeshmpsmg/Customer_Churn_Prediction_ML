"""
explainability.py
------------------
SHAP-based explainability: global feature importance, summary plots,
waterfall plots for individual predictions.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import get_logger, REPORTS_DIR

logger = get_logger(__name__)


def build_explainer(model, X_background: np.ndarray):
    """
    Builds a SHAP explainer appropriate for tree-based models
    (XGBoost/LightGBM/CatBoost/RandomForest/DecisionTree) or falls back
    to a model-agnostic explainer (e.g. for Logistic Regression).
    """
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("Using shap.TreeExplainer")
    except Exception as e:
        logger.info(f"TreeExplainer unavailable ({e}); falling back to KernelExplainer sample")
        background_sample = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.Explainer(model.predict_proba, background_sample)
    return explainer


def compute_shap_values(explainer, X):
    shap_values = explainer(X) if hasattr(explainer, "__call__") else explainer.shap_values(X)
    return shap_values


def plot_summary(explainer, X, feature_names, save_path=None, max_display=15):
    shap_values = explainer(X)
    # For binary classifiers, take the class-1 (churn) contributions if 3D
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(values, X, feature_names=feature_names, max_display=max_display, show=False)
    save_path = save_path or os.path.join(REPORTS_DIR, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_importance_bar(explainer, X, feature_names, save_path=None, max_display=15):
    shap_values = explainer(X)
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]
    mean_abs_shap = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:max_display]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[order][::-1], mean_abs_shap[order][::-1], color="#2563eb")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance")
    fig.tight_layout()
    save_path = save_path or os.path.join(REPORTS_DIR, "shap_feature_importance.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_waterfall_for_customer(explainer, X_row, feature_names, save_path=None, max_display=12):
    """X_row: a single-row 2D array (1, n_features)."""
    shap_values = explainer(X_row)
    single = shap_values[0]
    if single.values.ndim == 2:  # (n_features, n_classes)
        # select churn class (index 1)
        single = shap.Explanation(
            values=single.values[:, 1],
            base_values=single.base_values[1] if np.ndim(single.base_values) else single.base_values,
            data=single.data,
            feature_names=feature_names,
        )
    else:
        single.feature_names = feature_names

    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(single, max_display=max_display, show=False)
    save_path = save_path or os.path.join(REPORTS_DIR, "shap_waterfall_customer.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def get_top_churn_reasons(explainer, X_row, feature_names, top_n=3):
    """
    Returns the top-N features pushing THIS customer's prediction toward
    churn, in human-readable form, for the recommendation engine.
    """
    shap_values = explainer(X_row)
    single = shap_values[0]
    values = single.values
    if values.ndim == 2:
        values = values[:, 1]

    df = pd.DataFrame({"feature": feature_names, "shap_value": values})
    df["abs_value"] = df["shap_value"].abs()
    top_pushing_churn = df[df["shap_value"] > 0].sort_values("abs_value", ascending=False).head(top_n)
    reasons = [
        {"feature": clean_feature_name(row.feature), "impact": round(float(row.shap_value), 4)}
        for row in top_pushing_churn.itertuples()
    ]
    return reasons


def clean_feature_name(name: str) -> str:
    """Turn 'cat__Contract_Month-to-month' into 'Contract: Month-to-month'."""
    name = name.replace("num__", "").replace("cat__", "")
    if "_" in name:
        parts = name.split("_", 1)
        return f"{parts[0]}: {parts[1]}" if len(parts) > 1 else name
    return name
