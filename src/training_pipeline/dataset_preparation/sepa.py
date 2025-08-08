# src/training_pipeline/dataset_preparation/sepa.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
from load_data import normalize_rut as _normalize_rut

# Ejemplo: N = máximo número de tests en SEPA (ajustar según config)
N = 12  # o tomar dinámicamente

def dataset_sepa(dfs: Dict[str, pd.DataFrame], target_col: str):
    """
    Usa todas las columnas mat_* y leng_* de SEPA como X,
    y target_col (cualquiera de las notas PAES) como y.
    """

    # 1) Normalizar y limpiar RUT en SEPA y PAES
    df_sepa = _normalize_rut(dfs['df_pruebas_sepa'])
    df_paes = _normalize_rut(dfs['df_paes'])

    for df_tmp in (df_sepa, df_paes):
        df_tmp['student_rut'] = (
            df_tmp['student_rut']
            .astype(str)
            .str.replace(r'[^0-9A-Za-z]', '', regex=True)
            .str.strip()
        )
        df_tmp = df_tmp.loc[:, ~df_tmp.columns.duplicated()]

    # 2) Merge SEPA + PAES
    df = df_sepa.merge(df_paes, on='student_rut', how='inner')

    # 3) Reemplazar '-' por NaN y castear las columnas de score si existen
    df.replace('-', np.nan, inplace=True)
    for score in ['C. Lectora', 'Matemática', 'Historia', 'Ciencias']:
        if score in df.columns:
            df[score] = df[score].astype(float)

    # 4) Seleccionar features: todos los mat_* y leng_* 
    feat_cols = [c for c in df.columns if c.startswith('mat_') or c.startswith('leng_')]
    if not feat_cols:
        # no hay features SEPA
        return np.empty((0,0)), np.empty((0,0)), np.empty((0,)), np.empty((0,))

    # 5) Verificar que existe target_col
    if target_col not in df.columns:
        raise KeyError(f"Objetivo «{target_col}» no encontrado en PAES: {df.columns.tolist()}")

    # 6) Filtrar filas completas
    df_datos = df[feat_cols + [target_col]].dropna()

    # 7) Split
    X = df_datos[feat_cols]
    y = df_datos[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)