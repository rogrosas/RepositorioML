
from pathlib import Path
TARGET_COL = 'TARGET'
ID_COL = 'SK_ID_CURR'
DATASET_PARQUET = Path('artifacts/final_dataset.parquet')
ARTIFACTS_DIR = Path('artifacts')
METRICS_JSON = ARTIFACTS_DIR / 'metrics.json'
MODEL_PKL = ARTIFACTS_DIR / 'model.pkl'
FEATURES_CSV = ARTIFACTS_DIR / 'feature_importance.csv'
ROC_PNG = ARTIFACTS_DIR / 'roc_curve.png'
CM_PNG = ARTIFACTS_DIR / 'confusion_matrix.png'
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5
