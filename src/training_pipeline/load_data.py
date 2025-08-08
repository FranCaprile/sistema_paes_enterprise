import os
import yaml
import pandas as pd

def cargar_configuracion():
    """
    Lee y devuelve todo el contenido de config/config.yml como dict.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def cargar_dataset(ruta: str, tipo: str = 'csv'):
    """
    Carga un dataset desde una ruta dada, con formato 'csv' o 'excel'.
    """
    if tipo == 'csv':
        return pd.read_csv(ruta)
    elif tipo == 'excel':
        return pd.read_excel(ruta, skiprows=1)
    else:
        raise ValueError(f"Tipo de archivo {tipo} no soportado")


def normalize_rut(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas RUT, rut o student_id a student_rut para merges.
    Elimina columnas duplicadas resultantes.
    """
    mapping = {}
    if 'RUT' in df.columns:
        mapping['RUT'] = 'student_rut'
    if 'rut' in df.columns:
        mapping['rut'] = 'student_rut'
    if mapping:
        df = df.rename(columns=mapping)
    return df