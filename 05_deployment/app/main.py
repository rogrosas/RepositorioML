
# -*- coding: utf-8 -*-
"""
API de Riesgo de Crédito (FastAPI)
Endpoints:
- GET /health
- GET /schema
- POST /evaluate_risk
- POST /batch_evaluate_risk
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import Record, BatchRecords, Prediction, BatchPredictions
from .service import predict_one, predict_batch, health_report, get_feature_names

API_VERSION = os.getenv("API_VERSION", "1.0.0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Credit Risk API", version=API_VERSION)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return health_report()

@app.get("/schema")
def schema():
    return {"expected_features": get_feature_names()}

@app.post("/evaluate_risk", response_model=Prediction)
def evaluate_risk(record: Record):
    try:
        result = predict_one(record.features)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_evaluate_risk", response_model=BatchPredictions)
def batch_evaluate_risk(batch: BatchRecords):
    try:
        records = [r.features for r in batch.records]
        result = predict_batch(records)
        return JSONResponse(content={"predictions": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
