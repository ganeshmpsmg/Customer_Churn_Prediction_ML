"""
api.py
------
FastAPI REST API for the Customer Churn Prediction System.

Run locally:
    uvicorn api:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""
import os
import sys
import time
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import get_logger, BEST_MODEL_PATH, PREPROCESSOR_PATH

logger = get_logger("api")

API_KEY = os.environ.get("CHURN_API_KEY", "dev-secret-key")  # override in production via env var

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability and returns business recommendations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from predict import ChurnPredictor
        _predictor = ChurnPredictor()
    return _predictor


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Simple header-based API key auth. Swap for OAuth2/JWT in real production use."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return True


class CustomerRecord(BaseModel):
    customerID: str = Field(default="UNKNOWN")
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "CUST-000001",
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 425.0,
            }
        }


class BatchRequest(BaseModel):
    customers: List[CustomerRecord]


class PredictionResponse(BaseModel):
    customer_id: str
    prediction: str
    churn_probability: float
    risk_level: str
    top_churn_reasons: list
    recommendations: list
    estimated_annual_revenue_at_risk: float


@app.get("/", tags=["Health"])
def root():
    return {"service": "Customer Churn Prediction API", "status": "ok"}


@app.get("/health", tags=["Health"])
def health():
    ready = os.path.exists(BEST_MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)
    return {"status": "healthy" if ready else "model_not_trained", "model_ready": ready}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerRecord, authorized: bool = Depends(verify_api_key)):
    start = time.time()
    predictor = get_predictor()
    result = predictor.predict_single(customer.model_dump())
    logger.info(f"Single prediction for {customer.customerID} in {time.time()-start:.3f}s")
    return {
        "customer_id": str(result["customer_id"]),
        "prediction": result["prediction"],
        "churn_probability": result["churn_probability"],
        "risk_level": result["risk_level"],
        "top_churn_reasons": result.get("top_churn_reasons", []),
        "recommendations": result.get("recommendations", []),
        "estimated_annual_revenue_at_risk": result["estimated_annual_revenue_at_risk"],
    }


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
def predict_batch(request: BatchRequest, authorized: bool = Depends(verify_api_key)):
    if not request.customers:
        raise HTTPException(status_code=400, detail="No customers provided.")
    start = time.time()
    predictor = get_predictor()
    raw_df = pd.DataFrame([c.model_dump() for c in request.customers])
    results = predictor.predict_batch(raw_df)
    logger.info(f"Batch prediction for {len(request.customers)} customers in {time.time()-start:.3f}s")

    return [
        {
            "customer_id": str(r["customer_id"]),
            "prediction": r["prediction"],
            "churn_probability": r["churn_probability"],
            "risk_level": r["risk_level"],
            "top_churn_reasons": r.get("top_churn_reasons", []),
            "recommendations": r.get("recommendations", []),
            "estimated_annual_revenue_at_risk": r["estimated_annual_revenue_at_risk"],
        }
        for r in results.to_dict("records")
    ]
