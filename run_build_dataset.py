
"""
run_build_dataset.py
Runner para construir el dataset final.
Añade 02_data_preparation al sys.path y usa DATA_DIR=Path(r"..\datos_examen").
"""
from pathlib import Path
import sys

# Añade carpeta 02_data_preparation al sys.path
sys.path.append(str(Path('02_data_preparation').resolve()))

from build_dataset import build_dataset

if __name__ == '__main__':
    DATA_DIR = Path(r".\datos_examen")  # ← según tu petición
    ARTIFACTS_DIR = Path("artifacts")

    print("CWD:", Path.cwd())
    print("DATA_DIR (resolve):", DATA_DIR.resolve())

    # Verificación de archivos esperados
    required = [
        "application_.parquet",
        "bureau.parquet",
        "bureau_balance.parquet",
        "previous_application.parquet",
        "POS_CASH_balance.parquet",
        "installments_payments.parquet",
        "credit_card_balance.parquet",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        print("❌ Faltan archivos en DATA_DIR:", missing)
        raise SystemExit("Revisa la ruta y los nombres de los .parquet")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = build_dataset(DATA_DIR, ARTIFACTS_DIR)
    print(f"✅ Dataset final guardado en: {out}")
