# src/training_pipeline/dataset_preparation/pca.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
from load_data import normalize_rut as _normalize_rut

# Columnas de PCA disponibles en df_pca: PC1..PC8
def dataset_pca(dfs: Dict[str, pd.DataFrame], target_col: str):
    """
    Prepara split usando componentes principales como features.
    """
    # 1) Normalizar y limpiar RUT en todos los DataFrames
    df_sepa = _normalize_rut(dfs['df_pruebas_sepa'])
    df_paes = _normalize_rut(dfs['df_paes'])
    df_pca  = _normalize_rut(dfs['df_pca'])
    for df_tmp in (df_sepa, df_paes, df_pca):
        df_tmp['student_rut'] = (
            df_tmp['student_rut']
            .astype(str)
            .str.replace(r'[^0-9A-Za-z]', '', regex=True)
            .str.strip()
        )
        df_tmp = df_tmp.loc[:, ~df_tmp.columns.duplicated()]

    # 2) Merge SEPA + PAES + PCA
    df = (
        df_sepa
        .merge(df_paes, on='student_rut', how='inner')
        .merge(df_pca,  on='student_rut', how='inner')
    )

    # 3) Reemplazar '-' por NaN y castear puntajes
    df.replace('-', np.nan, inplace=True)
    for score in ['C. Lectora', 'Matemática', 'Historia', 'Ciencias', 'M2']:
        if score in df.columns:
            df[score] = df[score].astype(float)

    # 4) Verificar objetivo
    if target_col not in df.columns:
        raise KeyError(f"Objetivo '{target_col}' no encontrado en PAES. Columnas: {df.columns.tolist()}")

    # 5) Seleccionar features de PCA
    pca_cols = [c for c in df_pca.columns if c.startswith('PC')]
    if not pca_cols:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,)), np.empty((0,))

    # 6) Preparar DataFrame y eliminar NaNs
    df_datos = df[pca_cols + [target_col]].dropna()

    X = df_datos[pca_cols]
    y = df_datos[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
