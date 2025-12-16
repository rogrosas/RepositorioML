
FROM python:3.11-slim
WORKDIR /app

COPY 05_deployment/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY 05_deployment/app ./app
COPY artifacts ./artifacts

ENV ARTIFACTS_DIR=/app/artifacts     MODEL_PATH=/app/artifacts/model.pkl     PREP_PATH=/app/artifacts/preprocessor.joblib     FEATS_PATH=/app/artifacts/feature_names.json     MODEL_VERSION=v1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
