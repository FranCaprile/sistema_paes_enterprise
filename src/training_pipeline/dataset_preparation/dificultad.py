import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
import pandas as pd

# src/training_pipeline/dataset_preparation/dificultad.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
from load_data import normalize_rut as _normalize_rut

SCORE_COLS = ['C. Lectora', 'Matemática', 'Historia', 'Ciencias', 'M2']

def dataset_dificultad(dfs: Dict[str, pd.DataFrame], target_col: str):
    """
    Prepara X_train, X_test, y_train, y_test usando las columnas de dificultad
    (pivot por dimensión) como features, y la PAES como y.
    Une SEPA + PAES + DIFICULTAD por student_rut.
    """
    # 1) Normalizar y limpiar RUT
    df_sepa = _normalize_rut(dfs['df_pruebas_sepa'])
    df_paes = _normalize_rut(dfs['df_paes'])
    df_diff = _normalize_rut(dfs['df_avg_diff_agg'])

    # Alinear nombres en df_diff: rut/RUT -> student_rut
    df_diff = df_diff.rename(columns={'rut': 'student_rut', 'RUT': 'student_rut'})
    df_diff = df_diff.loc[:, ~df_diff.columns.duplicated()]

    # Unificar formato de student_rut (string sin símbolos) en todos
    for df_tmp in (df_sepa, df_paes, df_diff):
        if 'student_rut' in df_tmp.columns:
            df_tmp['student_rut'] = (
                df_tmp['student_rut']
                .astype(str)
                .str.replace(r'[^0-9A-Za-z]', '', regex=True)
                .str.strip()
            )

    # 2) Merge SEPA + PAES + DIFICULTAD
    df = (
        df_sepa
        .merge(df_paes, on='student_rut', how='inner')
        .merge(df_diff, on='student_rut', how='inner')
    )

    # 3) Cast de puntajes PAES
    df.replace('-', np.nan, inplace=True)
    for col in SCORE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # 4) Validar objetivo
    if target_col not in df.columns:
        raise KeyError(
            f"Objetivo '{target_col}' no encontrado. Columnas: {df.columns.tolist()}"
        )

    # 5) Seleccionar features de dificultad: todas las del pivot (excepto ids/rut)
    diff_feature_cols = [c for c in df_diff.columns if c not in ('student_rut', 'rut', 'id')]
    # Asegurar que existan en el DF unido (por si hubo columnas filtradas)
    diff_feature_cols = [c for c in diff_feature_cols if c in df.columns]

    if not diff_feature_cols:
        # No hay features -> devolver splits vacíos para que el pipeline lo salte
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,)), np.empty((0,))

    # 6) Filtrar filas completas en X e y
    df_datos = df[diff_feature_cols + [target_col]].dropna()
    if df_datos.empty:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,)), np.empty((0,))

    X = df_datos[diff_feature_cols]
    y = df_datos[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)
