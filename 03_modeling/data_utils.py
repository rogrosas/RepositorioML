
from pathlib import Path
import pandas as pd
from .config import TARGET_COL, ID_COL

def load_final_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])
    return X, y
