"""
utils.py
--------
Shared configuration, logging setup, and small helper functions used
across the churn prediction pipeline.
"""
import logging
import os
import sys
import joblib

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "telco_churn.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "churn_features.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.joblib")
SHAP_EXPLAINER_PATH = os.path.join(MODELS_DIR, "shap_explainer.joblib")

TARGET_COL = "Churn"
ID_COL = "customerID"

RANDOM_STATE = 42

for d in [DATA_DIR, os.path.join(DATA_DIR, "raw"), os.path.join(DATA_DIR, "processed"),
          MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        file_handler = logging.FileHandler(os.path.join(REPORTS_DIR, "pipeline.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------
def save_object(obj, path):
    joblib.dump(obj, path)


def load_object(path):
    return joblib.load(path)


def risk_level_from_probability(prob: float) -> str:
    """Bucket a churn probability into a business-friendly risk tier."""
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"
