import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la URI de la base de datos desde las variables de entorno
sepa_uri = os.getenv('SEPA_DB_URI')

if not sepa_uri:
    print("Error: La URI de la base de datos no está definida en las variables de entorno o en el archivo config.yml.")
else:
    print(f"URI de la base de datos cargada correctamente: {sepa_uri}")

# Crear la conexión con la base de datos SEPA
engine = create_engine(sepa_uri)

# Consulta de ejemplo para probar la conexión
sql_str = 'SELECT * FROM student LIMIT 10'
df = pd.read_sql(sql_str, engine)
print(df)
