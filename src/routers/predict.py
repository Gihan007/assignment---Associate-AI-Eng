import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")

router = APIRouter()


class PredictRequest(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


class PredictResponse(BaseModel):
    prediction: int
    probability: float


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


@router.on_event("startup")
def _load_artifacts():
    global model
    model = load_model()


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([payload.dict()])

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else 0.0
    return PredictResponse(prediction=int(pred), probability=float(proba))
