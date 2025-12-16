# make_feature_names.py
# -*- coding: utf-8 -*-
"""
Genera artifacts/feature_names.json a partir de artifacts/final_dataset.parquet.
- Excluye la columna objetivo 'TARGET' si existe.
- Mantiene el orden exacto de columnas del dataset final de entrenamiento.
"""

import os
import json
import pandas as pd

PARQUET_PATH = os.path.join("artifacts", "final_dataset.parquet")
JSON_PATH = os.path.join("artifacts", "feature_names.json")

def main():
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"No se encontró {PARQUET_PATH}. "
            f"Ejecuta este script en la raíz del repo y verifica la ruta/archivo."
        )

    # Cargar dataset final (todas las features ya integradas/ingenierizadas)
    df = pd.read_parquet(PARQUET_PATH)

    # Excluir la columna objetivo si existe (case-insensitive)
    cols = list(df.columns)
    feature_cols = [c for c in cols if c.upper() != "TARGET"]

    # Guardar JSON con el orden exacto
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    print(f"✅ feature_names.json creado con {len(feature_cols)} columnas en: {JSON_PATH}")

if __name__ == "__main__":
    main()
