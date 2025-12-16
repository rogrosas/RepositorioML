
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_roc(fpr, tpr, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0,1],[0,1],'k--', label='Azar')
    plt.xlabel('FPR (Falsos Positivos)')
    plt.ylabel('TPR (Verdaderos Positivos)')
    plt.title('Curva ROC')
    plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_confusion_matrix(cm, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap='Blues')
    plt.title('Matriz de Confusión'); plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['No Incumple (0)','Incumple (1)'])
    plt.yticks(ticks, ['No Incumple (0)','Incumple (1)'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.xlabel('Predicción'); plt.ylabel('Real')
    plt.tight_layout(); plt.savefig(out_path); plt.close()
