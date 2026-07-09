import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from evaluation import compute_metrics, get_classification_report
from recommendation_engine import recommend_action, build_customer_report, rank_customers_by_priority
from utils import risk_level_from_probability


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_worst_case():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.9, 0.1, 0.1])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 0.0
    assert metrics["roc_auc"] == 0.0


def test_classification_report_contains_class_names():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = get_classification_report(y_true, y_pred)
    assert "No Churn" in report
    assert "Churn" in report


@pytest.mark.parametrize("prob,expected", [(0.9, "High"), (0.5, "Medium"), (0.1, "Low")])
def test_risk_level_from_probability(prob, expected):
    assert risk_level_from_probability(prob) == expected


def test_recommend_action_matches_contract_keyword():
    action = recommend_action("Contract: Month-to-month")
    assert "contract" in action.lower()


def test_recommend_action_falls_back_for_unknown_feature():
    action = recommend_action("SomeRandomFeature")
    assert isinstance(action, str) and len(action) > 0


def test_build_customer_report_structure():
    reasons = [{"feature": "Contract: Month-to-month", "impact": 0.3}]
    report = build_customer_report("CUST-1", 0.82, monthly_charges=70.0, top_reasons=reasons, tenure=3)
    assert report["risk_level"] == "High"
    assert report["customer_id"] == "CUST-1"
    assert report["estimated_annual_revenue_at_risk"] == round(0.82 * 70.0 * 12, 2)
    assert len(report["recommendations"]) >= 1


def test_rank_customers_by_priority_orders_high_risk_first():
    reports = [
        {"risk_level": "Low", "estimated_annual_revenue_at_risk": 1000},
        {"risk_level": "High", "estimated_annual_revenue_at_risk": 200},
        {"risk_level": "Medium", "estimated_annual_revenue_at_risk": 500},
    ]
    ranked = rank_customers_by_priority(reports)
    assert ranked[0]["risk_level"] == "High"
    assert ranked[-1]["risk_level"] == "Low"


def test_rank_customers_by_priority_orders_by_revenue_within_same_risk():
    reports = [
        {"risk_level": "High", "estimated_annual_revenue_at_risk": 100},
        {"risk_level": "High", "estimated_annual_revenue_at_risk": 900},
    ]
    ranked = rank_customers_by_priority(reports)
    assert ranked[0]["estimated_annual_revenue_at_risk"] == 900
