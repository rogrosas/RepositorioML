# credit-risk-prediction
Proyecto Examen Machine Learning.

Estructura de Microservicios

/project_root

/01_data_understanding: Scripts o notebooks para el Análisis Exploratorio de Datos (EDA).

/02_data_preparation: Scripts (‘.py‘) para la limpieza, preprocesamiento y ingeniería de características.

/03_modeling: Scripts para el entrenamiento, ajuste de hiperparámetros y validación de modelos.

/04_evaluation: Scripts para la evaluación final del modelo campeón y la generación de reportes/visua-lizaciones.

/05_deployment: Código de la API (ej. app.py) y archivos asociados.

/artifacts: Para almacenar salidas como modelos entrenados (‘.pkl‘, ‘.joblib‘), scalers, etc.

README.md: Documentación clara del proyecto.

requirements.txt: Listado de dependencias de Python.


Cada vez que queramos hacer actualizaciones hay que seguir estos pasos:
    1.- git status (revisamos lo que se está cambiando y que está o no dentro de la actualizacion)
    2.- git add . (añade todo lo que estamos actualizando o añadiendo) git add + "nombre archivo" en caso de una sola cosa
    3.- git commit -m "comentario" (importante para llevar un correcto versionamiento)
    4.- git push (finalizar la actualización)