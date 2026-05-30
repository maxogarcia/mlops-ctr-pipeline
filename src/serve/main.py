import os
import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time

# Config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "ctr_xgboost")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# Prometheus metrics
PREDICTION_COUNTER = Counter("predictions_total", "Total predictions made")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
POSITIVE_COUNTER = Counter("positive_predictions_total", "Total positive CTR predictions")

# Column names
INT_COLS = [f'int_{i}' for i in range(1, 14)]
CAT_COLS = [f'cat_{i}' for i in range(1, 27)]

app = FastAPI(title="CTR Prediction API", version="1.0.0")

# Load model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"Loading model from {model_uri}...")
    model = mlflow.xgboost.load_model(model_uri)
    print("Model loaded successfully")

class PredictionRequest(BaseModel):
    int_1: Optional[float] = 0
    int_2: Optional[float] = 0
    int_3: Optional[float] = 0
    int_4: Optional[float] = 0
    int_5: Optional[float] = 0
    int_6: Optional[float] = 0
    int_7: Optional[float] = 0
    int_8: Optional[float] = 0
    int_9: Optional[float] = 0
    int_10: Optional[float] = 0
    int_11: Optional[float] = 0
    int_12: Optional[float] = 0
    int_13: Optional[float] = 0
    cat_1: Optional[float] = 0
    cat_2: Optional[float] = 0
    cat_3: Optional[float] = 0
    cat_4: Optional[float] = 0
    cat_5: Optional[float] = 0
    cat_6: Optional[float] = 0
    cat_7: Optional[float] = 0
    cat_8: Optional[float] = 0
    cat_9: Optional[float] = 0
    cat_10: Optional[float] = 0
    cat_11: Optional[float] = 0
    cat_12: Optional[float] = 0
    cat_13: Optional[float] = 0
    cat_14: Optional[float] = 0
    cat_15: Optional[float] = 0
    cat_16: Optional[float] = 0
    cat_17: Optional[float] = 0
    cat_18: Optional[float] = 0
    cat_19: Optional[float] = 0
    cat_20: Optional[float] = 0
    cat_21: Optional[float] = 0
    cat_22: Optional[float] = 0
    cat_23: Optional[float] = 0
    cat_24: Optional[float] = 0
    cat_25: Optional[float] = 0
    cat_26: Optional[float] = 0

class PredictionResponse(BaseModel):
    click_probability: float
    will_click: bool
    model_version: str

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "version": MODEL_VERSION}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    data = pd.DataFrame([request.dict()])
    prob = float(model.predict_proba(data)[:, 1][0])

    latency = time.time() - start
    PREDICTION_COUNTER.inc()
    PREDICTION_LATENCY.observe(latency)
    if prob >= 0.5:
        POSITIVE_COUNTER.inc()

    return PredictionResponse(
        click_probability=round(prob, 4),
        will_click=prob >= 0.5,
        model_version=MODEL_VERSION
    )

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
