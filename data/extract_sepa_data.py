import os
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Cargar configuración adicional desde el archivo config.yml (si es necesario)
with open("config/config.yml", "r") as f:
    cfg = yaml.safe_load(f)

# Obtener la URI de la base de datos desde las variables de entorno, o usar la configuración de config.yml
sepa_uri = os.getenv('SEPA_DB_URI', cfg['database']['sepa_uri'])

# Verificar que la URI de la base de datos se ha cargado correctamente
if not sepa_uri:
    print("Error: La URI de la base de datos no está definida en las variables de entorno o en el archivo config.yml.")
else:
    print(f"URI de la base de datos cargada correctamente: {sepa_uri}")

# Crear la conexión con la base de datos SEPA
def create_connection():
    try:
        engine = create_engine(sepa_uri)
        print("Conexión exitosa con la base de datos SEPA.")
        return engine
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

# Función para ejecutar una consulta SQL y guardar el resultado como DataFrame
def execute_query(engine, query):
    try:
        return pd.read_sql(text(query), con=engine.connect())
    except Exception as e:
        print(f"Error al ejecutar la consulta: {e}")
        return None

# Función para extraer datos de la base de datos SEPA
def extract_sepa_data():
    queries = {
        "student": """
            SELECT s.id AS student_id, s.fullname AS student_name, s.rut AS student_rut
            FROM student s;
        """,
        "answer": """
            SELECT a.score, a.student_id, a.question_id
            FROM answer a;
        """,
        "indicator": """
            SELECT description, question_id
            FROM indicator;
        """
    }

    engine = create_connection()
    if engine is None:
        return

    for table_name, query in queries.items():
        df = execute_query(engine, query)
        if df is not None:
            output_path = f"data/unprocessed/{table_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"Datos de la tabla {table_name} guardados en {output_path}")
        else:
            print(f"No se pudo extraer los datos de la tabla {table_name}.")

# Ejecutar la extracción
if __name__ == "__main__":
    extract_sepa_data()
