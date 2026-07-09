"""
train.py
--------
End-to-end training pipeline:
  1. Load + validate + clean data
  2. Feature engineering
  3. Preprocess (encode/scale) via ColumnTransformer
  4. Train & compare 6 model families
  5. Hyperparameter tune the top candidates (RandomizedSearchCV + CV)
  6. Evaluate all models, auto-select the best by ROC-AUC
  7. Fit a SHAP explainer on the winner
  8. Persist model, preprocessor, feature names, metrics, explainer
  9. Log everything to MLflow

Usage:
    python src/train.py --data data/raw/telco_churn.csv --tune
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, clean_data, get_feature_columns, build_preprocessor, get_output_feature_names
from feature_engineering import engineer_features
from evaluation import evaluate_model
from explainability import build_explainer
from utils import (
    get_logger, save_object, RANDOM_STATE, TARGET_COL, ID_COL,
    BEST_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, METRICS_PATH, SHAP_EXPLAINER_PATH,
    RAW_DATA_PATH,
)

logger = get_logger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


MODEL_ZOO = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
        {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 10, 20]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        {"n_estimators": [200, 400, 600], "max_depth": [5, 10, 15, None], "min_samples_leaf": [1, 3, 5]},
    ),
    "XGBoost": (
        xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],
         "subsample": [0.7, 0.9, 1.0]},
    ),
    "LightGBM": (
        lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        {"n_estimators": [200, 400], "max_depth": [-1, 5, 10], "learning_rate": [0.01, 0.05, 0.1],
         "num_leaves": [15, 31, 63]},
    ),
    "CatBoost": (
        CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
        {"iterations": [200, 400], "depth": [4, 6, 8], "learning_rate": [0.01, 0.05, 0.1]},
    ),
}


def get_class_weight_ratio(y):
    neg, pos = np.bincount(y)
    return neg / max(pos, 1)


def train_and_compare(X_train, y_train, X_test, y_test, tune: bool = False, quick: bool = False):
    """Trains every model in MODEL_ZOO, optionally hyperparameter-tunes it,
    evaluates on the test set, and returns the results + fitted estimators."""
    results = {}
    fitted_models = {}
    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=RANDOM_STATE)

    for name, (base_model, param_grid) in MODEL_ZOO.items():
        start = time.time()
        model = base_model

        if name in ("XGBoost",) and hasattr(model, "set_params"):
            model.set_params(scale_pos_weight=get_class_weight_ratio(y_train))

        if tune:
            logger.info(f"Tuning {name} with RandomizedSearchCV...")
            search = RandomizedSearchCV(
                model, param_distributions=param_grid,
                n_iter=6 if quick else 12, scoring="roc_auc", cv=cv,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info(f"{name} best params: {search.best_params_}")
        else:
            model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, model_name=name, save_plots=True)
        elapsed = time.time() - start
        metrics["train_time_sec"] = round(elapsed, 2)
        results[name] = metrics
        fitted_models[name] = model
        logger.info(f"{name} done in {elapsed:.1f}s | ROC-AUC={metrics['roc_auc']:.4f} | "
                    f"F1={metrics['f1_score']:.4f}")

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({k: v for k, v in getattr(model, "get_params", lambda: {})().items()
                                    if isinstance(v, (int, float, str, bool)) or v is None})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    return results, fitted_models


def select_best_model(results: dict, fitted_models: dict, metric="roc_auc"):
    best_name = max(results, key=lambda n: results[n][metric])
    return best_name, fitted_models[best_name], results[best_name]


def main(data_path: str, tune: bool, quick: bool):
    logger.info("=" * 70)
    logger.info("CHURN PREDICTION TRAINING PIPELINE STARTED")
    logger.info("=" * 70)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("customer_churn_prediction")
        mlflow.start_run(run_name="pipeline_run")

    # 1. Load & clean
    df = load_data(data_path)
    df = clean_data(df)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Split features / target
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")

    numeric_cols, categorical_cols = get_feature_columns(df)
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    # TenureGroup is categorical (category dtype) - ensure captured
    if "TenureGroup" in X.columns and "TenureGroup" not in categorical_cols:
        categorical_cols.append("TenureGroup")
        X["TenureGroup"] = X["TenureGroup"].astype(str)

    logger.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Preprocess
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = get_output_feature_names(preprocessor)

    logger.info(f"Processed train shape: {X_train.shape}, test shape: {X_test.shape}")

    # 5 & 6. Train, tune, compare
    results, fitted_models = train_and_compare(X_train, y_train, X_test, y_test, tune=tune, quick=quick)

    # 7. Select best
    best_name, best_model, best_metrics = select_best_model(results, fitted_models)
    logger.info("=" * 70)
    logger.info(f"BEST MODEL: {best_name} | ROC-AUC={best_metrics['roc_auc']:.4f} | "
                f"F1={best_metrics['f1_score']:.4f} | Accuracy={best_metrics['accuracy']:.4f}")
    logger.info("=" * 70)

    # 8. Build SHAP explainer on best model
    logger.info("Building SHAP explainer for best model...")
    try:
        explainer = build_explainer(best_model, X_train)
        save_object(explainer, SHAP_EXPLAINER_PATH)
    except Exception as e:
        logger.warning(f"Could not build SHAP explainer: {e}")

    # 9. Persist artifacts
    save_object(best_model, BEST_MODEL_PATH)
    save_object(preprocessor, PREPROCESSOR_PATH)
    save_object(feature_names, FEATURE_NAMES_PATH)
    save_object({"best_model_name": best_name, "all_results": results}, METRICS_PATH)

    logger.info(f"Artifacts saved to models/ directory.")

    if MLFLOW_AVAILABLE:
        mlflow.log_param("best_model", best_name)
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.end_run()

    logger.info("TRAINING PIPELINE COMPLETE.")
    return best_name, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=RAW_DATA_PATH)
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--quick", action="store_true", help="Fewer CV folds / iterations for fast runs")
    args = parser.parse_args()
    main(args.data, args.tune, args.quick)
