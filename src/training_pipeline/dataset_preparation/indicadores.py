# src/training_pipeline/dataset_preparation/indicadores.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
from load_data import normalize_rut as _normalize_rut

# Lista de todas las columnas de puntaje posibles
SCORE_COLS = ['C. Lectora', 'Matemática', 'Historia', 'Ciencias', 'M2']

def dataset_indicadores(dfs: Dict[str, pd.DataFrame], target_col: str):
    """
    Prepara X_train, X_test, y_train, y_test usando solo las columnas
    que vienen de df_indicadores, eliminando las demás columnas de puntaje.
    """

    # 1) Normalizar la clave de fusión en cada DataFrame
    df_sepa = _normalize_rut(dfs['df_pruebas_sepa'])
    df_paes = _normalize_rut(dfs['df_paes'])
    df_ind  = _normalize_rut(dfs['df_indicadores'])

    # 2) Merge SEPA + PAES + Indicadores
    df = df_sepa.merge(df_paes, on='student_rut', how='inner') \
                .merge(df_ind,  on='student_rut', how='inner')

    # 3) Reemplazar '-' por NaN y castear las columnas de score
    df.replace('-', np.nan, inplace=True)
    cast_cols = [c for c in SCORE_COLS if c in df.columns]
    if cast_cols:
        df = df.astype({c: 'float' for c in cast_cols})

    # 4) Comprobar que la columna objetivo existe
    if target_col not in df.columns:
        raise KeyError(
            f"No existe la columna objetivo '{target_col}'.\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # 5) Eliminar columnas irrelevantes:
    #    • Metadatos: id, Nombre, Admisión, student_rut
    #    • Todas las demás columnas de puntaje
    drop_cols = ['id', 'Nombre', 'Admisión', 'student_rut']
    drop_cols += [c for c in cast_cols if c != target_col]
    df = df.drop(columns=drop_cols, errors='ignore')

    # 6) Detectar las columnas de indicadores (las que venían en df_ind)
    ind_cols = [c for c in df_ind.columns if c != 'student_rut']
    # Asegurar que al menos una permanezca en el DataFrame
    valid_feats = [c for c in ind_cols if c in df.columns]
    if not valid_feats:
        raise ValueError(
            f"No se detectaron columnas de indicadores en:\n{df.columns.tolist()}"
        )

    # 7) Eliminar filas con NaN en features o en el target
    df = df.dropna(subset=valid_feats + [target_col])

    # 8) Separar X e y y hacer train/test split
    X = df[valid_feats]
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)
