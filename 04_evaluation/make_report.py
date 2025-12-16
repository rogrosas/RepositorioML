from pathlib import Path
import json
import base64
from datetime import datetime

TEMPLATE = """<!DOCTYPE html>
<html lang=\"es\">
<head>
<meta charset=\"utf-8\" />
<title>Reporte de Evaluación - Modelo Riesgo Crédito</title>
<style>
body{font-family: Arial, sans-serif; margin: 24px; line-height: 1.5;}
h1,h2{color:#222}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;}
.card{border:1px solid #ddd;border-radius:8px;padding:16px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:8px;text-align:left}
th{background:#f5f5f5}
.code{font-family: Consolas, monospace; background:#f9f9f9; padding:6px 8px; border-radius:6px}
img{max-width:100%; height:auto; border:1px solid #eee}
.footer{margin-top:32px;color:#666;font-size:12px}
</style>
</head>
<body>
<h1>Reporte de Evaluación – Modelo Campeón</h1>
<p>Este reporte resume las métricas principales y visualizaciones generadas durante la evaluación del modelo campeón. Los artefactos provienen de la carpeta <span class=\"code\">artifacts/</span>.</p>
<h2>Métricas Clave</h2>
<table>
<tr><th>Métrica</th><th>Valor</th></tr>
<tr><td>AUC</td><td>{auc:.4f}</td></tr>
<tr><td>Precision</td><td>{precision:.4f}</td></tr>
<tr><td>Recall</td><td>{recall:.4f}</td></tr>
<tr><td>F1</td><td>{f1:.4f}</td></tr>
</table>
<h2>Matriz de Confusión</h2>
<pre class=\"code\">{cm}</pre>
<div class=\"grid\">
<div class=\"card\"><h2>Curva ROC</h2><img src=\"data:image/png;base64,{roc_b64}\" alt=\"ROC\" /></div>
<div class=\"card\"><h2>Matriz de Confusión (Gráfico)</h2><img src=\"data:image/png;base64,{cm_b64}\" alt=\"Confusion Matrix\" /></div>
</div>
<h2>Top Features (si disponible)</h2>
{feat_table}
<div class=\"footer\"><p>Generado automáticamente. Fecha: {date}.</p></div>
</body>
</html>"""

def read_metrics(metrics_path: Path):
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def img_to_b64(img_path: Path) -> str:
    if not img_path.exists():
        return ''
    return base64.b64encode(img_path.read_bytes()).decode('ascii')


def build_features_table(csv_path: Path) -> str:
    if not csv_path.exists():
        return '<p>No disponible.</p>'
    import pandas as pd
    df = pd.read_csv(csv_path).sort_values('importance', ascending=False).head(15)
    rows = '\n'.join(f"<tr><td>{r['feature']}</td><td>{r['importance']:.6f}</td></tr>" for _, r in df.iterrows())
    return f"<table><tr><th>Feature</th><th>Importance</th></tr>{rows}</table>"


def make_report(artifacts_dir: Path, out_html: Path):
    m = read_metrics(artifacts_dir / 'metrics.json')
    roc_b64 = img_to_b64(artifacts_dir / 'roc_curve.png')
    cm_b64 = img_to_b64(artifacts_dir / 'confusion_matrix.png')
    feat_table = build_features_table(artifacts_dir / 'feature_importance.csv')
    html = TEMPLATE.format(
        auc=m.get('auc', 0.0), precision=m.get('precision', 0.0),
        recall=m.get('recall', 0.0), f1=m.get('f1', 0.0),
        cm=m.get('confusion_matrix', []), roc_b64=roc_b64, cm_b64=cm_b64,
        feat_table=feat_table, date=datetime.now().strftime('%Y-%m-%d %H:%M')
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding='utf-8')
    return out_html

if __name__ == '__main__':
    ARTS = Path('artifacts'); OUT = Path('04_evaluation/report.html')
    print('Reporte HTML generado en:', make_report(ARTS, OUT))
