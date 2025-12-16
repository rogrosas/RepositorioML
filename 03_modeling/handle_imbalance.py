
from typing import Optional, Dict
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

def get_class_weights(y) -> Optional[Dict[int, float]]:
    classes = np.unique(y)
    try:
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        return {int(k): float(v) for k, v in zip(classes, weights)}
    except Exception:
        return None

def get_smote(random_state: int = 42):
    return SMOTE(random_state=random_state) if SMOTE is not None else None
