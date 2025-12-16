
# 05_deployment – Credit Risk API (lista para `model.pkl`)

## 1) Qué incluye
- FastAPI con endpoints: `/health`, `/schema`, `/evaluate_risk`, `/batch_evaluate_risk`.
- Loader robusto que carga `artifacts/model.pkl` con **joblib o pickle**.
- Soporte para modelos sklearn (`predict_proba`/`predict`) y boosters de XGBoost/LightGBM.
- Diagnóstico en `/health` (ruta, estado de carga, errores).
- Dockerfile + requirements.

## 2) Copia tus artefactos reales
Coloca en `./artifacts`:
- `model.pkl`  ← **tu modelo campeón** (ideal: `sklearn.Pipeline`).
- `feature_names.json` ← **orden y nombres** de columnas del entrenamiento.
- `preprocessor.joblib` (opcional) ← si tu preprocesamiento está fuera del modelo.

## 3) Ejecutar local
```bash
# Opción A: entra al directorio de deployment
cd 05_deployment
uvicorn app.main:app --reload --port 8000

# Opción B: ejecutar desde la raíz del repo
uvicorn --app-dir 05_deployment app.main:app --reload --port 8000
```

## 4) Probar en Swagger
Visita `http://127.0.0.1:8000/docs` y usa `POST /evaluate_risk`.
Ejemplo de cuerpo (ajusta a tus features reales):
```json
{
  "features": {
    "DAYS_BIRTH": -18000,
    "AMT_CREDIT": 500000,
    "AMT_INCOME_TOTAL": 250000,
    "NUM_CREDIT_BUREAU_INQUIRIES": 2
  }
}
```

## 5) Docker
```bash
# Construir
docker build -t credit-risk-api -f 05_deployment/Dockerfile .

# Ejecutar
docker run -p 8000:8000 credit-risk-api
```

## 6) Solución de problemas
- `model_loaded: false` en `/health` → revisa que `artifacts/model.pkl` exista y se pueda cargar.
- Error de columnas/shape → tu `feature_names.json` no coincide con el entrenamiento.
- Modelo sin `predict_proba` → el loader usa `decision_function`; si no aplica, cae a `predict` (0/1).

## 7) Variables de entorno (si deseas cambiar rutas sin editar código)
```bash
# Windows PowerShell
$env:MODEL_PATH = "artifacts/model.pkl"
$env:FEATS_PATH = "artifacts/feature_names.json"
uvicorn --app-dir 05_deployment app.main:app --reload --port 8000
```
