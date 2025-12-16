
"""
02_data_preparation/build_dataset.py
Construye el dataset final y guarda artifacts/final_dataset.parquet
"""
import pandas as pd
from pathlib import Path

from cleaning import basic_clean_application, downcast_numeric
from merge_bureau import agregar_bureau
from merge_previous import agregar_previous_features
from feature_engineering import build_features


def build_dataset(data_dir: Path, artifacts_dir: Path) -> Path:
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Carga parquet
    application = pd.read_parquet(data_dir / "application_.parquet")
    bureau = pd.read_parquet(data_dir / "bureau.parquet")
    bureau_balance = pd.read_parquet(data_dir / "bureau_balance.parquet")
    previous = pd.read_parquet(data_dir / "previous_application.parquet")
    pos_cash = pd.read_parquet(data_dir / "POS_CASH_balance.parquet")
    installments = pd.read_parquet(data_dir / "installments_payments.parquet")
    credit_card = pd.read_parquet(data_dir / "credit_card_balance.parquet")

    # Limpieza base
    application_clean = basic_clean_application(application)

    # Agregaciones
    bureau_features = agregar_bureau(bureau, bureau_balance)
    prev_features = agregar_previous_features(previous, pos_cash, installments, credit_card)

    # Unión principal por SK_ID_CURR
    merged = application_clean.set_index('SK_ID_CURR').join(bureau_features, how='left').join(prev_features, how='left')

    # Features derivadas + downcast
    final = build_features(merged)
    final = downcast_numeric(final)

    out_path = artifacts_dir / "final_dataset.parquet"
    final.to_parquet(out_path)
    return out_path


if __name__ == '__main__':
    # Este bloque sirve para pruebas directas, pero el runner es el recomendado
    DATA_DIR = Path(r"..\datos_examen")
    ARTIFACTS_DIR = Path("artifacts")
    out = build_dataset(DATA_DIR, ARTIFACTS_DIR)
    print(f"Dataset final guardado en: {out}")
