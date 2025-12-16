
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de plantilla de payload para /evaluate_risk
Lee artifacts/feature_names.json y crea payload_template.json con todas las columnas.

Uso:
    python generate_payload_template.py [--out payload_template.json] [--defaults nan|zero]

- --defaults nan: llena con null (NaN) para que el preprocesador impute (recomendado)
- --defaults zero: llena con 0 para numéricas; cadenas vacías para categóricas (si las conoces)
"""
import json, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='payload_template.json')
parser.add_argument('--defaults', choices=['nan','zero'], default='nan')
args = parser.parse_args()

feat_path = os.path.join('artifacts','feature_names.json')
if not os.path.exists(feat_path):
    raise FileNotFoundError(f"No se encontró {feat_path}. Asegúrate de ejecutarlo en la raíz del repo.")

with open(feat_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    if isinstance(data, dict) and 'feature_names' in data:
        feature_names = list(data['feature_names'])
    else:
        feature_names = list(data)

features = {}
for name in feature_names:
    if args.defaults == 'nan':
        features[name] = None
    else:
        # zero defaults; si algunas son categóricas, tendrás que reemplazar manualmente
        features[name] = 0

payload = {"features": features}

with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print(f"Plantilla creada: {args.out} (con {len(feature_names)} columnas)")
