# Sistema PAES Enterprise — MLOps

Proyecto para la construcción de un **pipeline de datos**, **entrenamiento** y **predicción** que estima los puntajes PAES por alumno a partir de:

- Resultados SEPA  
- Indicadores  
- Bloom / Taxonomía  
- PCA  
- Dificultad  

---

## 📂 Arquitectura del proyecto


├── config.yml
├── data/
│ ├── processed/ # CSVs generados por data_pipeline
│ └── unprocessed/ # paes_encrypted.xlsx (origen)
├── ml_models/ # <dd-mm>stacking_allvars<prueba>.joblib
├── reports/
│ ├── metrics.csv # métricas en formato ancho (RMSE)
│ └── feature_importances/ # importancias del stacking (CSV + PNG)
└── src/
├── data_pipeline/ # loaders, transformaciones, run_pipeline.py
├── training_pipeline/ # datasets, stacking, métricas, run_pipeline.py
└── prediction_service/ # predictor.py y test_predictor.py



---

## 🔄 Flujo de trabajo

El objetivo es **automatizar el código** para predecir el puntaje de la PAES a partir de resultados de la SEPA.  
El flujo es:

1. **Data Pipeline**  
   - Procesa datos brutos de SEPA y PAES, genera variables derivadas y guarda los resultados en `data/processed`.
   - Estos archivos procesados se usan posteriormente para el entrenamiento del modelo.

2. **Training Pipeline**  
   - Entrena 8 modelos base (definidos en `training_pipeline/models.py`) y evalúa métricas guardadas en `reports/metrics.csv`.
   - Entrena un modelo **Stacking** que combina algunos de los modelos entrenados anteriormente y combina todas las variables (**All Vars**).
   - Guarda:
     - Modelo Stacking entrenado en `ml_models/`
     - Importancia de variables en `reports/feature_importances/`

3. **Prediction Service**  
   - Predice puntajes PAES por **RUT** usando el último modelo de Stacking disponible.

---

## ⚙️ Uso

### 1️⃣ Instalación de dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate     # En Windows: .venv\Scripts\activate
pip install -r requirements.txt


### 2️⃣ Ejecutar Data Pipeline 


Genera y/o actualiza los datos, se ejecuta con el siguiente comando en el terminal:

```bash
python3 src/data_pipeline/run_pipeline.py

### 3️⃣ Ejecutar Training Pipeline


Entrena modelo Stacking por prueba, guarda modelos y métricas:

```bash
python3 src/training_pipeline/run_pipeline.py

Salida esperada:

Modelos: ml_models/<dd-mm>_stacking_allvars_<prueba>.joblib

Métricas: reports/metrics.csv

Importancias del stacking: reports/feature_importances/*

### 4️⃣ Predicción por RUT

Predice PAES por RUT usando el último modelo de Stacking disponible:

```bash
python src/prediction_service/test_predictor.py *rut_codificado*

Ejemplo:
```bash
python src/prediction_service/test_predictor.py 425047515d575059


También, la predicción del rut se puede llamar como función:

```bash
from prediction_service.predictor import predecir_paes_por_rut
preds = predecir_paes_por_rut("12345678K")  # {'C. Lectora': 625.3, 'Matemática': 601.2, ...}

### Información importante:

SEPA: Los datos se obtienen de un servidor NEON (credenciales en config.yml → sepa_uri).

PAES: Los archivos fuente están en data/unprocessed.

Procesamiento: Ambos conjuntos se limpian y transforman para generar data/processed, base del entrenamiento.

Codificación RUT: Los RUTs están codificados; la función de decodificación está en src/prediction_service/predictor.py.

Llave de unión: student_rut (sin puntos ni guiones).

Métricas: reports/metrics.csv en formato ancho (fila = (Modelo, Variable), columnas = pruebas (RMSE)).

Ejecución: Siempre correr desde la raíz del repositorio.

Estructura Python: src/__init__.py y __init__.py en submódulos recomendados.

Licencia

Uso académico / práctica MLOps.

