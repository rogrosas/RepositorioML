
from typing import Dict, Any
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    return {
        'auc': float(roc_auc_score(y_true, y_proba)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'roc_curve': {k: list(map(float, v)) for k, v in zip(['fpr','tpr','thresholds'], roc_curve(y_true, y_proba))}
    }

def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
