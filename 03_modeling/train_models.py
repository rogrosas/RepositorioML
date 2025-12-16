
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .config import DATASET_PARQUET, METRICS_JSON, MODEL_PKL, FEATURES_CSV, TEST_SIZE, RANDOM_STATE, N_SPLITS
from .data_utils import load_final_dataset, split_features_target
from .model_selection import get_models
from .handle_imbalance import get_class_weights
from .metrics import compute_metrics, save_metrics
from .plots import plot_roc, plot_confusion_matrix
from .save_model import save_model


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=['number']).columns.tolist()
    cat = X.select_dtypes(include=['object','category']).columns.tolist()
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('num', num_pipe, num), ('cat', cat_pipe, cat)])


def evaluate_and_select_model(X_train, y_train, pre, models):
    best = None; best_score = -np.inf
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for name, clf in models.items():
        pipe = Pipeline([('pre', pre), ('clf', clf)])
        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            score = float(np.mean(scores))
        except Exception:
            score = -np.inf
        print(f"Modelo {name} -> AUC CV: {score:.4f}")
        if score > best_score:
            best_score = score; best = name
    assert best is not None, 'No se pudo evaluar ningún modelo'
    print(f"Mejor modelo: {best} (AUC CV={best_score:.4f})")
    return best


def get_feature_names(preprocessor, X):
    num = X.select_dtypes(include=['number']).columns.tolist()
    cat = X.select_dtypes(include=['object','category']).columns.tolist()
    try:
        oh = preprocessor.named_transformers_['cat'].named_steps['oh']
        oh_cols = oh.get_feature_names_out(cat).tolist()
    except Exception:
        oh_cols = cat
    return num + oh_cols


def compute_feature_importance(pipe, X, out_path: Path):
    try:
        clf = pipe.named_steps['clf']; pre = pipe.named_steps['pre']; pre.fit(X)
        names = get_feature_names(pre, X)
        if hasattr(clf, 'feature_importances_'): imps = clf.feature_importances_
        elif hasattr(clf, 'coef_'): imps = np.abs(clf.coef_).ravel()
        else: return False
        import pandas as pd
        pd.DataFrame({'feature': names[:len(imps)], 'importance': imps}).sort_values('importance', ascending=False).to_csv(out_path, index=False)
        return True
    except Exception:
        return False


def train_main():
    df = load_final_dataset(DATASET_PARQUET)
    X, y = split_features_target(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    pre = build_preprocessor(Xtr)
    models = get_models()
    cw = get_class_weights(ytr)
    if cw is not None:
        if 'logreg' in models: models['logreg'].set_params(class_weight=cw)
        if 'rf' in models: models['rf'].set_params(class_weight=cw)
        if 'xgb' in models:
            w0 = cw.get(0, 1.0); w1 = cw.get(1, 1.0)
            models['xgb'].set_params(scale_pos_weight=w1/(w0 if w0 else 1.0))
        if 'lgbm' in models: models['lgbm'].set_params(class_weight=cw)
    best = evaluate_and_select_model(Xtr, ytr, pre, models)
    final = Pipeline([('pre', pre), ('clf', models[best])])
    final.fit(Xtr, ytr)
    ypred = final.predict(Xte)
    try:
        ypro = final.predict_proba(Xte)[:,1]
    except Exception:
        scores = final.decision_function(Xte)
        ypro = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    m = compute_metrics(yte, ypred, ypro); save_metrics(m, METRICS_JSON)
    from numpy import array
    plot_roc(m['roc_curve']['fpr'], m['roc_curve']['tpr'], out_path=Path('artifacts/roc_curve.png'))
    plot_confusion_matrix(array(m['confusion_matrix']), out_path=Path('artifacts/confusion_matrix.png'))
    _ = compute_feature_importance(final, Xtr, out_path=FEATURES_CSV)
    save_model(final, MODEL_PKL)
    print('Entrenamiento completo. Modelo y métricas guardados en artifacts/')

if __name__ == '__main__':
    train_main()
