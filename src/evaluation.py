"""
evaluation.py
-------------
Model evaluation metrics and plots: confusion matrix, ROC, PR curve,
classification report.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
)

from utils import get_logger, REPORTS_DIR

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return metrics


def get_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"]); ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    save_path = save_path or os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_roc_curve(y_true, y_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", color="#2563eb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    save_path = save_path or os.path.join(REPORTS_DIR, "roc_curve.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_precision_recall_curve(y_true, y_proba, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#059669")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    save_path = save_path or os.path.join(REPORTS_DIR, "pr_curve.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def evaluate_model(model, X_test, y_test, model_name="model", save_plots=True) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    report = get_classification_report(y_test, y_pred)
    logger.info(f"[{model_name}] Metrics: {metrics}")
    logger.info(f"[{model_name}] Classification report:\n{report}")

    if save_plots:
        plot_confusion_matrix(y_test, y_pred,
                               os.path.join(REPORTS_DIR, f"{model_name}_confusion_matrix.png"))
        plot_roc_curve(y_test, y_proba,
                       os.path.join(REPORTS_DIR, f"{model_name}_roc_curve.png"))
        plot_precision_recall_curve(y_test, y_proba,
                                     os.path.join(REPORTS_DIR, f"{model_name}_pr_curve.png"))

    metrics["classification_report"] = report
    return metrics
