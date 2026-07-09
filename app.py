"""
app.py
------
Streamlit dashboard for the Customer Churn Prediction System.

Run:
    streamlit run app.py
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import (
    BEST_MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH, load_object, REPORTS_DIR,
)

st.set_page_config(page_title="Customer Churn Prediction", layout="wide", page_icon="📉")

MODELS_TRAINED = os.path.exists(BEST_MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)


@st.cache_resource
def get_predictor():
    from predict import ChurnPredictor
    return ChurnPredictor()


@st.cache_data
def get_metrics():
    return load_object(METRICS_PATH)


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("📉 Churn Prediction System")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔮 Single Prediction", "📁 Batch Prediction",
     "📊 Model Performance", "🧠 Explainability (SHAP)", "💰 Revenue & Risk Dashboard"],
)

if not MODELS_TRAINED:
    st.error(
        "No trained model found. Please run the training pipeline first:\n\n"
        "```\npython data/generate_sample_data.py\npython src/train.py --tune\n```"
    )
    st.stop()

# ==================================================================
# OVERVIEW
# ==================================================================
if page == "🏠 Overview":
    st.title("Customer Churn Prediction System")
    st.markdown(
        "An end-to-end ML system that predicts customer churn probability, "
        "explains *why* using SHAP, and generates retention recommendations."
    )

    metrics_bundle = get_metrics()
    best_name = metrics_bundle["best_model_name"]
    best_metrics = metrics_bundle["all_results"][best_name]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_name)
    c2.metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
    c3.metric("F1-Score", f"{best_metrics['f1_score']:.3f}")
    c4.metric("Accuracy", f"{best_metrics['accuracy']:.3f}")

    st.subheader("Model Comparison")
    comp_df = pd.DataFrame({
        name: {k: v for k, v in m.items() if k != "classification_report"}
        for name, m in metrics_bundle["all_results"].items()
    }).T.reset_index().rename(columns={"index": "Model"})
    fig = px.bar(comp_df, x="Model", y="roc_auc", color="Model", title="ROC-AUC by Model",
                 text_auto=".3f")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp_df.style.format({c: "{:.4f}" for c in comp_df.columns if c != "Model"}),
                 use_container_width=True)

    st.info("Use the sidebar to explore single/batch predictions, model performance, "
            "SHAP explainability, and the revenue-at-risk dashboard.")

# ==================================================================
# SINGLE PREDICTION
# ==================================================================
elif page == "🔮 Single Prediction":
    st.title("🔮 Predict Churn for a Single Customer")

    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        with c3:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 65.0)

        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        customer = {
            "customerID": "MANUAL-ENTRY", "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }
        predictor = get_predictor()
        result = predictor.predict_single(customer)

        prob = result["churn_probability"]
        risk = result["risk_level"]
        color = {"High": "red", "Medium": "orange", "Low": "green"}[risk]

        col_gauge, col_info = st.columns([1, 1.3])
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#d1fae5"},
                        {"range": [40, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.metric("Prediction", result["prediction"])
            st.metric("Risk Level", risk)
            st.metric("Estimated Annual Revenue at Risk", f"${result['estimated_annual_revenue_at_risk']:,.2f}")

            st.subheader("Top Churn Drivers")
            reasons = result.get("top_churn_reasons", [])
            if reasons:
                for r in reasons:
                    st.write(f"- **{r['feature']}** (impact: {r['impact']:+.3f})")
            else:
                st.write("No dominant SHAP driver identified for this customer.")

            st.subheader("Retention Recommendations")
            for rec in result.get("recommendations", []):
                st.write(f"✅ {rec}")

# ==================================================================
# BATCH PREDICTION
# ==================================================================
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Churn Prediction")
    st.write("Upload a CSV with the same schema as the training data (customerID, demographics, "
             "services, billing columns).")

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(raw_df)} customers.")
        st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", use_container_width=True):
            with st.spinner("Scoring customers..."):
                predictor = get_predictor()
                results = predictor.predict_batch(raw_df)
                from recommendation_engine import rank_customers_by_priority
                ranked = pd.DataFrame(rank_customers_by_priority(results.to_dict("records")))

            st.success(f"Scored {len(ranked)} customers.")

            c1, c2, c3 = st.columns(3)
            c1.metric("High Risk", int((ranked["risk_level"] == "High").sum()))
            c2.metric("Medium Risk", int((ranked["risk_level"] == "Medium").sum()))
            c3.metric("Low Risk", int((ranked["risk_level"] == "Low").sum()))

            fig = px.pie(ranked, names="risk_level", title="Risk Level Distribution",
                         color="risk_level",
                         color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"})
            st.plotly_chart(fig, use_container_width=True)

            search = st.text_input("🔍 Search by customer ID")
            display_df = ranked
            if search:
                display_df = ranked[ranked["customer_id"].astype(str).str.contains(search, case=False)]

            st.dataframe(display_df, use_container_width=True, height=400)

            csv_bytes = ranked.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Full Prediction Report (CSV)", csv_bytes,
                                "churn_predictions.csv", "text/csv", use_container_width=True)

# ==================================================================
# MODEL PERFORMANCE
# ==================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    metrics_bundle = get_metrics()
    best_name = metrics_bundle["best_model_name"]
    model_names = list(metrics_bundle["all_results"].keys())
    selected = st.selectbox("Select model", model_names, index=model_names.index(best_name))

    m = metrics_bundle["all_results"][selected]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4.metric("F1-Score", f"{m['f1_score']:.3f}")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.text("Classification Report")
    st.code(m["classification_report"])

    img_cols = st.columns(3)
    plot_files = {
        "Confusion Matrix": f"{selected}_confusion_matrix.png",
        "ROC Curve": f"{selected}_roc_curve.png",
        "Precision-Recall Curve": f"{selected}_pr_curve.png",
    }
    for col, (title, fname) in zip(img_cols, plot_files.items()):
        path = os.path.join(REPORTS_DIR, fname)
        if os.path.exists(path):
            col.image(path, caption=title, use_container_width=True)

# ==================================================================
# EXPLAINABILITY
# ==================================================================
elif page == "🧠 Explainability (SHAP)":
    st.title("🧠 Explainable AI (SHAP)")
    st.markdown("Global and local explanations for the best model's predictions.")

    st.subheader("Global Feature Importance")
    imp_path = os.path.join(REPORTS_DIR, "shap_feature_importance.png")
    summ_path = os.path.join(REPORTS_DIR, "shap_summary.png")
    c1, c2 = st.columns(2)
    if os.path.exists(imp_path):
        c1.image(imp_path, caption="Mean |SHAP value| per feature", use_container_width=True)
    else:
        c1.info("Run `python src/explain_report.py` to generate global SHAP plots.")
    if os.path.exists(summ_path):
        c2.image(summ_path, caption="SHAP Summary Plot", use_container_width=True)

    st.subheader("Explain an Individual Customer")
    predictor = get_predictor()
    sample_path = "data/raw/telco_churn.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        cust_id = st.selectbox("Pick a customer ID", df["customerID"].head(200).tolist())
        if st.button("Explain this customer"):
            row = df[df["customerID"] == cust_id].iloc[0].to_dict()
            with st.spinner("Computing SHAP explanation..."):
                result = predictor.predict_single(row)
            st.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
            st.metric("Risk Level", result["risk_level"])
            reasons = result.get("top_churn_reasons", [])
            if reasons:
                reason_df = pd.DataFrame(reasons)
                fig = px.bar(reason_df, x="impact", y="feature", orientation="h",
                             title=f"Top Churn Drivers for {cust_id}",
                             color="impact", color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No individual SHAP explanation available for this customer.")

# ==================================================================
# REVENUE & RISK DASHBOARD
# ==================================================================
elif page == "💰 Revenue & Risk Dashboard":
    st.title("💰 Revenue Impact & Risk Dashboard")
    pred_path = os.path.join(REPORTS_DIR, "predictions.csv")
    if not os.path.exists(pred_path):
        st.warning("No batch predictions found yet. Run a batch prediction first, or run "
                   "`python src/predict.py --input data/raw/telco_churn.csv --output reports/predictions.csv`.")
    else:
        df = pd.read_csv(pred_path)
        total_at_risk = df["estimated_annual_revenue_at_risk"].sum()
        high_risk_customers = (df["risk_level"] == "High").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Annual Revenue at Risk", f"${total_at_risk:,.0f}")
        c2.metric("High-Risk Customers", int(high_risk_customers))
        c3.metric("Total Customers Scored", len(df))

        fig1 = px.histogram(df, x="churn_probability", nbins=30, color="risk_level",
                            title="Churn Probability Distribution",
                            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"})
        st.plotly_chart(fig1, use_container_width=True)

        top20 = df.sort_values("estimated_annual_revenue_at_risk", ascending=False).head(20)
        fig2 = px.bar(top20, x="customer_id", y="estimated_annual_revenue_at_risk", color="risk_level",
                      title="Top 20 Customers by Revenue at Risk",
                      color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"})
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Priority Outreach List (Top 50)")
        st.dataframe(df.sort_values(["risk_level", "estimated_annual_revenue_at_risk"],
                                     ascending=[True, False]).head(50), use_container_width=True)
