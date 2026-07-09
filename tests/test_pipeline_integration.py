"""
End-to-end integration test: trains a tiny model on synthetic data and
verifies the full predict pipeline (including SHAP + recommendations)
returns a well-formed result. Uses a temp directory for models/data so it
never overwrites the user's real trained artifacts.
"""
import os
import sys
import importlib

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(TESTS_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("churn_test")

    # Point utils' module-level paths to a scratch directory before importing
    # anything else that reads them at import time.
    os.environ["CHURN_TEST_MODE"] = "1"

    import utils
    utils.MODELS_DIR = str(tmp_dir / "models")
    utils.REPORTS_DIR = str(tmp_dir / "reports")
    utils.BEST_MODEL_PATH = os.path.join(utils.MODELS_DIR, "best_model.joblib")
    utils.PREPROCESSOR_PATH = os.path.join(utils.MODELS_DIR, "preprocessor.joblib")
    utils.FEATURE_NAMES_PATH = os.path.join(utils.MODELS_DIR, "feature_names.joblib")
    utils.METRICS_PATH = os.path.join(utils.MODELS_DIR, "metrics.joblib")
    utils.SHAP_EXPLAINER_PATH = os.path.join(utils.MODELS_DIR, "shap_explainer.joblib")
    os.makedirs(utils.MODELS_DIR, exist_ok=True)
    os.makedirs(utils.REPORTS_DIR, exist_ok=True)

    sys.path.insert(0, os.path.join(TESTS_DIR, "..", "data"))
    from generate_sample_data import generate
    data_path = str(tmp_dir / "sample.csv")
    df = generate(n=400, seed=1)
    df.to_csv(data_path, index=False)

    import train as train_module
    importlib.reload(train_module)
    train_module.BEST_MODEL_PATH = utils.BEST_MODEL_PATH
    train_module.PREPROCESSOR_PATH = utils.PREPROCESSOR_PATH
    train_module.FEATURE_NAMES_PATH = utils.FEATURE_NAMES_PATH
    train_module.METRICS_PATH = utils.METRICS_PATH
    train_module.SHAP_EXPLAINER_PATH = utils.SHAP_EXPLAINER_PATH

    best_name, best_metrics = train_module.main(data_path, tune=False, quick=True)
    return {
        "utils": utils, "data_path": data_path, "best_name": best_name, "best_metrics": best_metrics,
    }


def test_training_produces_reasonable_auc(trained_artifacts):
    assert trained_artifacts["best_metrics"]["roc_auc"] > 0.55  # better than a coin flip


def test_training_saves_all_artifacts(trained_artifacts):
    utils = trained_artifacts["utils"]
    for path in [utils.BEST_MODEL_PATH, utils.PREPROCESSOR_PATH,
                 utils.FEATURE_NAMES_PATH, utils.METRICS_PATH]:
        assert os.path.exists(path), f"Missing artifact: {path}"


def test_predict_single_customer_end_to_end(trained_artifacts):
    import predict as predict_module
    importlib.reload(predict_module)
    utils = trained_artifacts["utils"]
    predict_module.BEST_MODEL_PATH = utils.BEST_MODEL_PATH
    predict_module.PREPROCESSOR_PATH = utils.PREPROCESSOR_PATH
    predict_module.FEATURE_NAMES_PATH = utils.FEATURE_NAMES_PATH
    predict_module.SHAP_EXPLAINER_PATH = utils.SHAP_EXPLAINER_PATH

    predictor = predict_module.ChurnPredictor()
    import pandas as pd
    df = pd.read_csv(trained_artifacts["data_path"])
    result = predictor.predict_single(df.iloc[0].to_dict())

    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert result["prediction"] in {"Churn", "No Churn"}
    assert isinstance(result["recommendations"], list) and len(result["recommendations"]) > 0
