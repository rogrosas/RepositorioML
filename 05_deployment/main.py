
# -*- coding: utf-8 -*-
"""
FastAPI para exponer el modelo de riesgo de crédito.

Endpoints:
- GET /health
- POST /evaluate_risk
- POST /batch_evaluate_risk
"""
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import Record, BatchRecords, Prediction, BatchPredictions
from .service import predict_one, predict_batch

app = FastAPI(title="Credit Risk API", version=os.getenv("API_VERSION", "1.0.0"))

# CORS opcional (ajusta orígenes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get('/health')
def health():
    return {"status": "ok"}

@app.post('/evaluate_risk', response_model=Prediction)
def evaluate_risk(record: Record):
    try:
        result = predict_one(record.features)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post('/batch_evaluate_risk', response_model=BatchPredictions)
def batch_evaluate_risk(batch: BatchRecords):
    try:
        records = [r.features for r in batch.records]
        result = predict_batch(records)
        return JSONResponse(content={"predictions": result})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e})
                            
        