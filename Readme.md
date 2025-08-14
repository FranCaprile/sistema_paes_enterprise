Sistema PAES Enterprise — MLOps

Proyecto de pipeline de datos, entrenamiento y predicción para estimar puntajes PAES por alumno usando variables SEPA, Indicadores, Bloom/Taxonomía, PCA y Dificultad.

Arquitectura
.
├── config.yml
├── data/
│   ├── processed/           # CSVs generados por data_pipeline
│   └── unprocessed/         # paes_encrypted.xlsx (origen)
├── ml_models/               # <dd-mm>_stacking_allvars_<prueba>.joblib
├── reports/
│   ├── metrics.csv          # métricas en formato ancho (RMSE)
│   └── feature_importances/ # importancias del stacking (CSV + PNG)
└── src/
    ├── data_pipeline/       # loaders, transformaciones, run_pipeline.py
    ├── training_pipeline/   # datasets, stacking, métricas, run_pipeline.py
    └── prediction_service/  # predictor.py y test_predictor.py


Flujo:

La idea del proyecto es automatizar el codigo para predecir el puntaje de la PAES a partir de resultados de la SEPA.
Para ello, se sigue el siguiente flujo:

- Data Pipeline → toma los datos no procesados que tenemos de la SEPA y de la PAES y los procesa, creando distintas variables y guardando los archivos procesados en data/processed. La idea es después usar los archivos procesados para entrenar el modelo.

Training Pipeline → entrena los modelos: se entrenan primero 8 modelos (que están en training_pipeline/models.py) y se miden sus métricas (que se guardan en reports/performance_metrics/metrics.csv). Además, se entrena el modelo Stacking, que es una union de otros modelos y que toma todas las variables para entrenarse (All Vars). El modelo Stacking se guarda en ml_models y su gráfico de importancias de las variables en reports/feature_importances.

Prediction Service → predice PAES por RUT usando el último modelo de Stacking disponible.

Uso: 

1) Descargar los paquetes necesarios:

python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2) Para correr Data Pipeline

La carpeta Data Pipeline genera y/o actualiza los datos (de la sepa y paes respectivamente)data/processed/*.csv y se ejecuta con el siguiente comando en el terminal:

python3 src/data_pipeline/run_pipeline.py

2) Training Pipeline

Entrena stacking por prueba, guarda modelos y métricas, se ejecuta así:

python3 src/training_pipeline/run_pipeline.py

Salida esperada:

Modelos: ml_models/<dd-mm>_stacking_allvars_<prueba>.joblib

Métricas (ancho): reports/metrics.csv

Importancias del stacking: reports/feature_importances/*

3) Predicción por RUT

predice PAES por RUT usando el último modelo de Stacking disponible, y se ejecuta escribiendo lo siguiente en el terminal:

# con RUT codificado
python src/prediction_service/test_predictor.py *rut_codificado*

Ejemplo:
python src/prediction_service/test_predictor.py 425047515d575059

# o sin argumento (usa el primer student_rut disponible)
python src/prediction_service/test_predictor.py


También, la predicción del rut se puede llamar como función:

from prediction_service.predictor import predecir_paes_por_rut
preds = predecir_paes_por_rut("12345678K")  # {'C. Lectora': 625.3, 'Matemática': 601.2, ...}

Información importante:

Los archivos de la SEPA se obtienen a partir de un serverless de la base de datos NEON, su link y clave están en el archivo config.yml (en sepa_uri). Por otro lado, los archivo de la paes están en data/unprocessed. Ambos archivos se limpian y analizan para obtener lso archivos de data/processed y a partir de ellos, se corre el modelo.

Los ruts de las base de datos de la paes y la sepa están codificados, la función de decodificación está en src/prediction_service/predictor.py

El training guarda solo el modelo Stacking (All Vars) por prueba.

Llave de unión: student_rut (normalizado sin puntos/guiones).

reports/metrics.csv está en formato ancho: fila = (Modelo, Variable), columnas = pruebas (RMSE).

Es importante ejecutar el codigo desde la raíz del repo; recomendable tener src/__init__.py y submódulos con __init__.py.

Licencia

Uso académico / práctica MLOps.

