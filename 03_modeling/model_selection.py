
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

def get_models() -> Dict[str, object]:
    models = {
        'logreg': LogisticRegression(max_iter=2000, n_jobs=-1),
        'rf': RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    }
    if XGBClassifier is not None:
        models['xgb'] = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
    if LGBMClassifier is not None:
        models['lgbm'] = LGBMClassifier(
            n_estimators=800, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
    return models
