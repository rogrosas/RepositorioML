
# -*- coding: utf-8 -*-
"""
Servicio de inferencia: carga artefactos y ejecuta predicciones con diagnóstico claro.
"""
from typing import Dict, Any, List
import os, json, numpy as np, pandas as pd
from joblib import load as joblib_load
import pickle

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MODEL_PATH    = os.getenv("MODEL_PATH", os.path.join(ARTIFACTS_DIR, "model.pkl"))  # usa tu archivo real
PREP_PATH     = os.getenv("PREP_PATH",  os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
FEATS_PATH    = os.getenv("FEATS_PATH", os.path.join(ARTIFACTS_DIR, "feature_names.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

_model = None
_prep  = None
_feature_names: List[str] = []
_load_errors: List[str] = []


def _safe_load_any(path: str):
    if not os.path.exists(path):
        _load_errors.append(f"Archivo no encontrado: {path}")
        return None
    # 1) joblib
    try:
        return joblib_load(path)
    except Exception as e_job:
        # 2) pickle
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e_pkl:
            _load_errors.append(f"No se pudo cargar {path} con joblib ({e_job}) ni pickle ({e_pkl})")
            return None


def _safe_load_features(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "feature_names" in data:
                return list(data["feature_names"])
            return list(data)
    except Exception as e:
        _load_errors.append(f"Error al cargar feature_names desde {path}: {e}")
        return []


# Cargar artefactos al importar
_model = _safe_load_any(MODEL_PATH)
_prep  = _safe_load_any(PREP_PATH)  # si no existe, quedará None (no fatal)
_feature_names = _safe_load_features(FEATS_PATH)


def health_report() -> Dict[str, Any]:
    return {
        "status": "ok" if _model is not None else "error",
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "preprocessor_loaded": _prep is not None,
        "preprocessor_path": PREP_PATH,
        "n_features_expected": len(_feature_names) if _feature_names else None,
        "load_errors": _load_errors,
        "version": MODEL_VERSION,
    }


def get_feature_names() -> List[str]:
    return _feature_names


def _to_dataframe(features: Dict[str, Any]) -> pd.DataFrame:
    if _feature_names:
        row = {k: features.get(k) for k in _feature_names}
        return pd.DataFrame([row])
    return pd.DataFrame([features])


def _predict_proba_estimator(X):
    # sklearn estimators / pipelines
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)
        # Seleccionar la columna correcta según el orden real de las clases
        classes_ = getattr(_model, "classes_", None)
        if classes_ is not None:
            try:
                idx_rechazo = list(classes_).index(1)  # asumiendo que y=1 es "rechazo"
            except ValueError:
                # Si la etiqueta 1 no existe, usar la segunda columna si hay binario
                idx_rechazo = 1 if proba.shape[1] > 1 else 0
        else:
            # Fallback si no hay classes_
            idx_rechazo = 1 if proba.shape[1] > 1 else 0

    # XGBoost Booster
    try:
        import xgboost as xgb
        if isinstance(_model, xgb.Booster):
            dm = xgb.DMatrix(X)
            proba = _model.predict(dm)
            return float(proba[0])
    except Exception:
        pass

    # LightGBM Booster
    try:
        import lightgbm as lgb
        if isinstance(_model, lgb.Booster):
            proba = _model.predict(X)
            return float(proba[0])
    except Exception:
        pass

    # Fallback: sólo predict 0/1
    if hasattr(_model, "predict"):
        pred = _model.predict(X)
        return float(pred[0])

    raise RuntimeError("El modelo cargado no expone predict_proba, decision_function ni predict compatible.")


def predict_one(features: Dict[str, Any], low: float = 0.40, high: float = 0.60) -> Dict[str, Any]:
    if _model is None:
        raise RuntimeError(f"Modelo no cargado. Revisa MODEL_PATH='{MODEL_PATH}'. Errores: {_load_errors}")

    df = _to_dataframe(features)

    # Si hay preprocesador separado, usarlo; si tu modelo es un Pipeline, no necesitas _prep.
    X = _prep.transform(df) if _prep is not None else df

    prob = _predict_proba_estimator(X)

    # Decisión por umbrales
    if prob >= high:
        decision = "RECHAZAR"
    elif prob < low:
        decision = "APROBAR"
    else:
        decision = "REVISIÓN MANUAL"

    return {"probability": prob, "decision": decision, "model_version": MODEL_VERSION}


def predict_batch(records: List[Dict[str, Any]], low: float = 0.40, high: float = 0.60):
    return [predict_one(r, low=low, high=high) for r in records]
