import os
import yaml
import pandas as pd
from sqlalchemy import create_engine, text
import traceback

def cargar_configuracion():
    """
    Lee y devuelve todo el contenido de config/config.yml como dict.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def crear_engine():
    cfg = cargar_configuracion()
    uri = cfg["database"]["sepa_uri"]
    return create_engine(uri)

def cargar_datos_brutos():
    """
    Carga todos los archivos PAES y el archivo SEPA definidos en config.yml.
    Devuelve un dict con:
      - 'paes': lista de DataFrames (uno por cada archivo PAES)
      - 'sepa': DataFrame de SEPA
    """
    cfg = cargar_configuracion()
    rutas = cfg["paths"]

    resultados = {}


    # 1) Cargar datos PAES
    xl_path = rutas["paes_encrypted_xlsx"]
    if xl_path:
        try:
            df_xl = pd.read_excel(xl_path)
            print(f"[OK] PAES Excel cargado: {xl_path} ({len(df_xl)} filas)")
            resultados["paes_encrypted"] = df_xl
        except Exception as e:
            print(f"[ERROR] PAES Excel {xl_path}: {e}")
            resultados["paes_encrypted"] = None
    else:
        resultados["paes_encrypted"] = None
    resultados["paes"] = []
    engine = crear_engine()

    resultados["db_queries"] = {}
    for name, query in cfg["database"].get("db_queries", {}).items():
        try:
            df_q = pd.read_sql(text(query), engine)
            print(f"[OK] Query {name} → {len(df_q)} filas")
            resultados["db_queries"][name] = df_q
        except Exception as e:
            print(f"[ERROR] Query {name}: {e}")
            resultados["db_queries"][name] = None

    return resultados


if __name__ == "__main__":
    data = cargar_datos_brutos()
    # Pequeña verificación
    for i, df in enumerate(data["paes"], start=1):
        print(f"PAES #{i} columnas:", df.columns.tolist())
    for tbl, df in data["db_tables"].items():
        print(f"{tbl}: {len(df) if df is not None else 'ERROR'} filas")
    for tbl, df in data["db_tables"].items():
        print(f"  {tbl}: {len(df) if df is not None else 'ERROR'} rows")
    for name, df in data["db_queries"].items():
        print(f"  Query {name}: {len(df) if df is not None else 'ERROR'} rows")


"""""
    # 2) Cargar datos de la SEPA
    engine = crear_engine()
    resultados["db_tables"] = {}
    for tbl in cfg["database"].get("db_tables", []):
        try:
            # Lee toda la tabla
            df_tbl = pd.read_sql(text(f"SELECT * FROM public.{tbl}"), con=engine)
            print(f"[OK] Tabla {tbl}: {len(df_tbl)} filas")
            resultados["db_tables"][tbl] = df_tbl
        except Exception as e:
            print(f"[ERROR] Tabla {tbl}: {e}")
            traceback.print_exc()           # <-- Para ver el detalle del error
            resultados["db_tables"][tbl] = None
"""""