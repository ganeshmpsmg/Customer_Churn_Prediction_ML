"""
recommendation_engine.py
-------------------------
Translates a raw churn probability + SHAP explanation into a business
recommendation: risk level, top reasons, retention actions, and
estimated revenue at risk.
"""
from utils import risk_level_from_probability, get_logger

logger = get_logger(__name__)

# Maps a "reason" feature keyword -> retention recommendation
RETENTION_PLAYBOOK = {
    "Contract": "Offer an incentive to upgrade to a 1- or 2-year contract (discount or bundled service).",
    "PaymentMethod": "Encourage a switch to automatic bank transfer / credit card billing with a small credit.",
    "InternetService": "Proactively reach out about fiber service reliability/pricing; offer a loyalty discount.",
    "TechSupport": "Offer a free trial of premium tech support to reduce service friction.",
    "OnlineSecurity": "Bundle online security add-on at a discounted rate for the next 3 months.",
    "MonthlyCharges": "Offer a personalized discount or right-size their plan to reduce bill shock.",
    "tenure": "Assign a loyalty specialist / welcome call program for newer customers.",
    "TotalCharges": "Review lifetime value and offer a loyalty reward or milestone discount.",
    "PaperlessBilling": "Send a personalized billing satisfaction survey and support outreach.",
    "SeniorCitizen": "Offer simplified plans and dedicated senior support line.",
    "RiskScore": "Flag for proactive retention team outreach within 48 hours.",
    "ServiceUsageScore": "Introduce underused add-on services with a free trial.",
    "EngagementScore": "Send a re-engagement campaign highlighting unused benefits.",
}


def recommend_action(reason_feature: str) -> str:
    for key, action in RETENTION_PLAYBOOK.items():
        if key.lower() in reason_feature.lower():
            return action
    return "Schedule a personalized retention call to understand customer needs."


def build_customer_report(customer_id, churn_probability, monthly_charges,
                           top_reasons, tenure=None) -> dict:
    """
    Builds the full business-facing report for a single customer.
    top_reasons: list of dicts [{feature, impact}, ...] from explainability.get_top_churn_reasons
    """
    risk_level = risk_level_from_probability(churn_probability)
    recommendations = [recommend_action(r["feature"]) for r in top_reasons] or \
        ["No dominant risk driver identified; monitor account."]

    # Simple revenue-at-risk estimate: probability-weighted annualized monthly charges
    estimated_revenue_loss = round(churn_probability * monthly_charges * 12, 2)

    report = {
        "customer_id": customer_id,
        "churn_probability": round(float(churn_probability), 4),
        "risk_level": risk_level,
        "top_churn_reasons": top_reasons,
        "recommendations": list(dict.fromkeys(recommendations)),  # de-dupe, preserve order
        "estimated_annual_revenue_at_risk": estimated_revenue_loss,
        "tenure_months": tenure,
    }
    return report


def rank_customers_by_priority(reports: list) -> list:
    """
    Sorts a list of customer reports by (risk_level, revenue_at_risk) to
    produce a prioritized outreach list for the retention team.
    """
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(
        reports,
        key=lambda r: (risk_order.get(r["risk_level"], 3), -r["estimated_annual_revenue_at_risk"]),
    )
