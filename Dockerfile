# ---------------------------------------------------------------------
# Customer Churn Prediction System — Production Dockerfile
# ---------------------------------------------------------------------
FROM python:3.12.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by lightgbm/xgboost/catboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate sample data + train a default model at build time so the image
# is immediately usable. Skip this (and mount your own /app/models) if you
# want to bring pre-trained artifacts instead.
RUN python data/generate_sample_data.py --n 5000 --out data/raw/telco_churn.csv \
    && python src/train.py --data data/raw/telco_churn.csv --quick \
    && python src/explain_report.py --data data/raw/telco_churn.csv --sample 200

EXPOSE 8501 8000

# Default: launch the Streamlit dashboard.
# Override CMD to run the FastAPI service instead:
#   docker run -p 8000:8000 churn-app uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
