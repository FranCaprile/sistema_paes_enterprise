import os
import yaml
import pandas as pd
from sqlalchemy import create_engine, text

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

    # 1) Cargar archivos PAES
    paes_dfs = []
    for ruta_paes in rutas["paes"]:
        try:
            df = pd.read_csv(ruta_paes, delimiter=";")
            print(f"[OK] PAES cargado: {ruta_paes} ({len(df)} filas)")
            paes_dfs.append(df)
        except FileNotFoundError:
            print(f"[ERROR] PAES no encontrado: {ruta_paes}")
        except Exception as e:
            print(f"[ERROR] Leyendo PAES {ruta_paes}: {e}")
    resultados["paes"] = paes_dfs

    # 2) Cargar datos de la SEPA
    engine = crear_engine()
    resultados["db"] = {}
    for tbl in cfg["database"].get("db_tables", []):
        try:
            # Lee toda la tabla
            df_tbl = pd.read_sql(text(f"SELECT * FROM public.{tbl}"), con=engine)
            print(f"[OK] Tabla {tbl}: {len(df_tbl)} filas")
            resultados["db"][tbl] = df_tbl
        except Exception as e:
            print(f"[ERROR] Tabla {tbl}: {e}")
            resultados["db"][tbl] = None

    return resultados

if __name__ == "__main__":
    data = cargar_datos_brutos()
    # Pequeña verificación
    for i, df in enumerate(data["paes"], start=1):
        print(f"PAES #{i} columnas:", df.columns.tolist())
    for tbl, df in data["db"].items():
        print(f"{tbl}: {len(df) if df is not None else 'ERROR'} filas")

