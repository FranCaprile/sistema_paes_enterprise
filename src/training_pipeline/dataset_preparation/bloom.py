# training_pipeline/dataset_preparation/bloom.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
from load_data import normalize_rut as _normalize_rut

# Lista de todas las columnas de puntaje que existen en tu DF
SCORE_COLS = ['C. Lectora', 'Matemática', 'Historia', 'Ciencias', 'M2']


def dataset_bloom(dfs: Dict[str, pd.DataFrame], target_col: str):
    """
    Prepara un split para la variable de taxonomía 'bloom'.
    Extrae todas las columnas de df_taxonomia como features.
    """
    # 1) Normalizar clave primaria en cada DataFrame
    df_sepa = _normalize_rut(dfs['df_pruebas_sepa'])
    df_paes = _normalize_rut(dfs['df_paes'])
    df_taxo = _normalize_rut(dfs['df_taxonomia'])

    # 2) Merge SEPA + PAES
    df = df_sepa.merge(df_paes, on='student_rut', how='inner')
    df.replace('-', np.nan, inplace=True)
    # 3) Castear columnas de puntaje a float
    cast_cols = [c for c in SCORE_COLS if c in df.columns]
    if cast_cols:
        df = df.astype({c: 'float' for c in cast_cols})

    
    # 4) Merge con la taxonomía
    df_taxo = df_taxo.fillna(0)
    df = df.merge(df_taxo, on='student_rut', how='left')

    # 5) Identificar todas las columnas de taxonomía disponibles
    taxo_feats = [c for c in df_taxo.columns if c != 'student_rut']
    if not taxo_feats:
        raise ValueError(f"No se encontraron features de taxonomía en df_taxonomia: {df_taxo.columns.tolist()}")

    # 6) Validar que target_col exista
    if target_col not in df.columns:
        raise KeyError(f"Objetivo '{target_col}' no encontrado. Columnas: {df.columns.tolist()}")

    # 7) Eliminar columnas irrelevantes: metadatos y otros puntajes
    non_feat = ['id', 'Nombre', 'Admisión', 'student_rut']
    non_feat += [c for c in SCORE_COLS if c != target_col and c in df.columns]
    df = df.drop(columns=non_feat, errors='ignore')

    # 8) Eliminar filas con NaN en features de taxonomía o en el target
    df = df.dropna(subset=taxo_feats + [target_col])

    # 9) Separar X e y
    X = df[taxo_feats]
    y = df[target_col]

    # 10) Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)

