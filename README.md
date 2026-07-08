# 📉 Customer Churn Prediction System

A production-ready, end-to-end Machine Learning system that predicts customer churn,
explains *why* each customer is at risk using SHAP, and generates business
recommendations to drive retention — complete with a training pipeline, REST API,
interactive Streamlit dashboard, and Docker deployment.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)

---

## ✨ Features

- **Data pipeline**: validation, cleaning, outlier capping, encoding, scaling
- **Feature engineering**: tenure groups, spend trends, service usage score, engagement score, risk score
- **6-model comparison**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost — auto-selects the best by ROC-AUC
- **Hyperparameter tuning**: `RandomizedSearchCV` + Stratified K-Fold cross-validation
- **Explainable AI**: SHAP global feature importance, summary plots, and per-customer waterfall explanations
- **Full evaluation suite**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC & PR curves
- **Business recommendation engine**: churn probability → risk tier → top drivers → retention playbook → revenue-at-risk estimate → priority ranking
- **Interactive Streamlit dashboard**: single & batch prediction, SHAP explorer, model comparison, revenue dashboard
- **REST API** (FastAPI) with API-key auth, single & batch endpoints, OpenAPI docs
- **MLflow** experiment tracking
- **Docker / docker-compose** deployment
- **Bonus**: customer segmentation (KMeans), high-risk email alerting, GitHub Actions CI
- **29 passing unit + integration tests**

---

## 🗂️ Project Structure

```
CustomerChurnPrediction/
├── app.py                     # Streamlit dashboard
├── api.py                     # FastAPI REST API
├── data/
│   ├── generate_sample_data.py   # Synthetic Telco-schema dataset generator
│   └── raw/telco_churn.csv       # Generated sample data (or your own dataset)
├── src/
│   ├── utils.py                  # Shared config, logging, paths
│   ├── preprocessing.py          # Load, validate, clean, encode, scale
│   ├── feature_engineering.py    # Business feature derivation
│   ├── train.py                  # Full training + tuning + model selection pipeline
│   ├── predict.py                # ChurnPredictor: single & batch inference
│   ├── evaluation.py             # Metrics + plots
│   ├── explainability.py         # SHAP explanations
│   ├── explain_report.py         # Generates global SHAP plots for the dashboard
│   ├── recommendation_engine.py  # Business recommendations
│   ├── segmentation.py           # (Bonus) KMeans customer segmentation
│   └── monitoring.py             # (Bonus) High-risk scan + email alerts
├── tests/                     # 29 unit + integration tests (pytest)
├── models/                    # Saved model, preprocessor, explainer artifacts
├── reports/                   # Generated plots, logs, prediction CSVs
├── notebooks/                 # EDA notebook
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml   # CI: tests + docker build
└── README.md
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Tested on **Python 3.12.10**.

### 2. Generate (or bring your own) data

This repo ships with a synthetic data generator that mirrors the schema of the
**IBM Telco Customer Churn** dataset, so the whole pipeline runs out-of-the-box
without any external download:

```bash
python data/generate_sample_data.py --n 7043 --out data/raw/telco_churn.csv
```

To use a real dataset instead (e.g. Kaggle's `blastchar/telco-customer-churn`,
IBM's Telco Churn dataset, or a bank-churn dataset), download it and place it at
`data/raw/telco_churn.csv` with matching column names (`customerID`, `Churn`,
`tenure`, `MonthlyCharges`, `TotalCharges`, service/contract columns, etc.), or
point `--data` at your file directly.

### 3. Train the models

```bash
# Fast run (no tuning) — good for iterating
python src/train.py --data data/raw/telco_churn.csv --quick

# Full run: hyperparameter tuning across all 6 models
python src/train.py --data data/raw/telco_churn.csv --tune
```

This will:
1. Clean & validate the data
2. Engineer features
3. Train & compare 6 model families
4. Tune the top candidates (if `--tune`)
5. Auto-select the best model by ROC-AUC
6. Fit a SHAP explainer
7. Save everything to `models/` and evaluation plots to `reports/`

### 4. Generate global SHAP plots (for the dashboard)

```bash
python src/explain_report.py --data data/raw/telco_churn.csv --sample 300
```

### 5. Run batch predictions from the CLI

```bash
python src/predict.py --input data/raw/telco_churn.csv --output reports/predictions.csv
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501`. The dashboard includes:
- **Overview** — model comparison & headline metrics
- **Single Prediction** — form-based single-customer scoring with a probability gauge
- **Batch Prediction** — CSV upload, risk breakdown, searchable table, CSV download
- **Model Performance** — confusion matrix, ROC curve, PR curve per model
- **Explainability (SHAP)** — global importance + per-customer driver charts
- **Revenue & Risk Dashboard** — revenue-at-risk, priority outreach list

### 7. Launch the REST API

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "CUST-000001", "gender": "Female", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5, "TotalCharges": 425.0
  }'
```

Set `CHURN_API_KEY` as an environment variable to override the default dev key
in production.

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

29 tests covering preprocessing, feature engineering, evaluation metrics,
the recommendation engine, and a full end-to-end train→predict integration test.

---

## 🐳 Docker Deployment

### Single container (dashboard)

```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

### Dashboard + API together

```bash
docker compose up --build
```
- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000/docs`

The image trains a default model at build time so it's immediately usable; mount
your own `models/` volume to use pre-trained artifacts instead (see `docker-compose.yml`).

---

## ☁️ Other Deployment Options

- **Streamlit Community Cloud**: push this repo to GitHub, connect it at
  [share.streamlit.io](https://share.streamlit.io), set the main file to `app.py`.
  Add a build step or pre-commit trained `models/*.joblib` artifacts (Streamlit
  Cloud doesn't run `train.py` for you).
- **Render**: create a Web Service pointing at this repo. Build command:
  `pip install -r requirements.txt && python data/generate_sample_data.py && python src/train.py --quick`.
  Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` (dashboard)
  or `uvicorn api:app --host 0.0.0.0 --port $PORT` (API).
- **Any container platform** (Fly.io, Railway, ECS, Cloud Run): use the provided `Dockerfile`.

---

## 📊 MLflow Experiment Tracking

`train.py` automatically logs each model's parameters and metrics to MLflow if
it's installed (it's in `requirements.txt`). View the UI with:

```bash
mlflow ui
```

Then open `http://localhost:5000` to compare runs across models and hyperparameter
configurations.

---

## 🎁 Bonus Features

**Customer segmentation (KMeans):**
```bash
python src/segmentation.py --data data/raw/telco_churn.csv --k 4
```
Outputs `reports/customer_segments.csv` with a data-driven persona label per
customer (e.g. "New, High-spend, High-risk") for targeted campaigns.

**Real-time monitoring & email alerts:**
```bash
python src/monitoring.py --input data/raw/telco_churn.csv --threshold 0.8
```
Flags customers above a churn-probability threshold and (optionally) emails a
summary if `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, and `ALERT_EMAIL_TO` are
set as environment variables. Wire this into a scheduler (cron / GitHub Actions
schedule / Airflow) for continuous monitoring.

**CI/CD:** `.github/workflows/ci.yml` runs the full test suite and a Docker
build on every push/PR to `main`.

**REST API authentication:** header-based `X-API-Key` check (swap for OAuth2/JWT
in real production use — see `api.py::verify_api_key`).

---

## 🧠 How Model Selection Works

`train.py` trains all 6 models on an 80/20 stratified split, evaluates each on
held-out test data, and automatically selects the model with the highest
**ROC-AUC** (a good primary metric for imbalanced churn classification since it's
threshold-independent). Precision/recall/F1/accuracy for every model are still
logged and shown in the dashboard for a fuller picture — in production, you'd
typically pick the operating threshold (and possibly the metric to optimize) based
on the actual cost of a false positive (unnecessary retention offer) vs. a false
negative (lost customer).

---

## 📄 Data Schema

The system expects a CSV with (at minimum) these columns — matching the IBM Telco
Customer Churn dataset:

| Column | Type | Description |
|---|---|---|
| `customerID` | string | Unique customer identifier |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | categorical | Demographics |
| `tenure` | int | Months as a customer |
| `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categorical | Services subscribed |
| `Contract`, `PaperlessBilling`, `PaymentMethod` | categorical | Billing/account details |
| `MonthlyCharges`, `TotalCharges` | float | Billing amounts |
| `Churn` | Yes/No | Target (only required for training) |

---

## ⚠️ Note on the Sample Dataset

Since this environment cannot download from Kaggle/IBM directly, `data/generate_sample_data.py`
generates a **synthetic but statistically realistic** dataset with the same schema and
a churn rate (~27%) matching the real IBM Telco dataset, with realistic churn drivers
baked in (month-to-month contracts, electronic check payment, fiber internet, low
tenure, high monthly charges, etc.). Swap in the real dataset at any time — the
entire pipeline is schema-compatible.

---

## 📜 License

MIT — free to use for learning, portfolio, and internship interview purposes.
"# Customer_Churn_Prediction_ML" 
