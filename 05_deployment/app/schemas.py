# -*- coding: utf-8 -*-
"""Esquemas Pydantic de entrada/salida."""
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

Numeric = Union[int, float]
FeatureValue = Union[Numeric, str, None]

class Record(BaseModel):
    features: Dict[str, FeatureValue] = Field(..., description="Mapa feature→valor")

    # Evita warnings por nombres con 'model_*'
    model_config = {
        "protected_namespaces": ()
    }

class BatchRecords(BaseModel):
    records: List[Record]

    model_config = {
        "protected_namespaces": ()
    }

class Prediction(BaseModel):
    probability: float
    decision: str
    model_version: Optional[str] = None
    warnings: Optional[List[str]] = None

    model_config = {
        "protected_namespaces": ()
    }

class BatchPredictions(BaseModel):
    predictions: List[Prediction]

    model_config = {
        "protected_namespaces": ()
    }

