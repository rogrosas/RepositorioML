
# Herramienta para generar payload completo

1) Copia `generate_payload_template.py` a la **raíz de tu repositorio** (donde está `artifacts/feature_names.json`).
2) Ejecuta:
   ```bash
   python generate_payload_template.py --out payload_template.json --defaults nan
   ```
3) Se creará `payload_template.json` con todas las columnas esperadas por el modelo.
4) Abre el archivo y rellena solo algunas columnas con valores reales; el resto puede quedar `null` si tu preprocesador/pipeline imputa.
5) En Swagger, pega el contenido del JSON dentro del cuerpo de `POST /evaluate_risk`.

> Si tu pipeline **no imputa** y requiere valores obligatorios, usa `--defaults zero` y rellena manualmente los campos clave.
