# src/training_pipeline/dataset_preparation/all_vars.py
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.model_selection import train_test_split
from load_data import normalize_rut as _normalize_rut

SCORE_COLS = ['C. Lectora', 'Matemática', 'Historia', 'Ciencias', 'M2']

def _clean_rut(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'student_rut' not in df.columns:
        df = df.rename(columns={'RUT': 'student_rut', 'rut': 'student_rut'})
    df = df.loc[:, ~df.columns.duplicated()]
    if 'student_rut' in df.columns:
        df['student_rut'] = (
            df['student_rut'].astype(str)
            .str.replace(r'[^0-9A-Za-z]', '', regex=True)
            .str.strip()
        )
    return df

def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, how='left') -> pd.DataFrame:
    if 'student_rut' not in left.columns or 'student_rut' not in right.columns:
        return left
    return left.merge(right, on='student_rut', how=how)

def dataset_allvars(dfs: Dict[str, pd.DataFrame], target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Crea un dataset unificado con TODAS las variables:
    - SEPA (mat_* y leng_*)
    - Indicadores (todas las columnas de df_indicadores excepto ids)
    - PCA (PC1..)
    - Dificultad (pivot df_avg_diff o df_avg_diff_agg)
    - Bloom/Taxonomía (df_taxonomia)
    Devuelve: X_train, X_test, y_train, y_test
    """
    # 1) Base: SEPA + PAES (INNER -> asegura target)
    df_sepa = _clean_rut(_normalize_rut(dfs['df_pruebas_sepa']))
    df_paes = _clean_rut(_normalize_rut(dfs['df_paes']))

    base = df_sepa.merge(df_paes, on='student_rut', how='inner')
    base.replace('-', np.nan, inplace=True)
    for c in SCORE_COLS:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors='coerce')

    if target_col not in base.columns:
        # no target disponible
        return np.empty((0,0)), np.empty((0,0)), np.empty((0,)), np.empty((0,))

    # 2) Agregar variables (LEFT -> no pierdo filas base)
    # Indicadores
    if 'df_indicadores' in dfs:
        ind = _clean_rut(_normalize_rut(dfs['df_indicadores']))
        base = _safe_merge(base, ind, how='left')

    # PCA
    if 'df_pca' in dfs:
        pca = _clean_rut(_normalize_rut(dfs['df_pca']))
        base = _safe_merge(base, pca, how='left')

    # Dificultad (prefiere agg si existe)
    dif_key = 'df_avg_diff_agg' if 'df_avg_diff_agg' in dfs else ('df_avg_diff' if 'df_avg_diff' in dfs else None)
    if dif_key:
        diff = dfs[dif_key].copy()
        # normalizar clave
        diff = diff.rename(columns={'RUT': 'student_rut', 'rut': 'student_rut'})
        diff = _clean_rut(diff)
        base = _safe_merge(base, diff, how='left')

    # Bloom/Taxonomía
    if 'df_taxonomia' in dfs:
        blo = dfs['df_taxonomia'].copy()
        blo = blo.rename(columns={'RUT': 'student_rut', 'rut': 'student_rut'})
        blo = _clean_rut(blo)
        # quitar metadatos típicos
        blo = blo.drop(columns=['id', 'Nombre', 'Admisión'], errors='ignore')
        # quitar posibles scores PAES por seguridad
        blo = blo.drop(columns=[c for c in SCORE_COLS if c in blo.columns], errors='ignore')
        base = _safe_merge(base, blo, how='left')

    # 3) Seleccionar features numéricos
    #    - Quitar identificadores/targets
    drop_cols = set(['student_rut', 'id', 'Nombre', 'Admisión'] + SCORE_COLS)
    feature_cols: List[str] = [c for c in base.columns if c not in drop_cols]

    # Forzar numéricos
    X_all = base[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = base[target_col]

    # Opcional: quitar columnas con varianza cero o demasiados NaN
    # (aquí simple: rellenamos NaN con 0; si prefieres otra imputación, cámbiala)
    X_all = X_all.fillna(0.0)

    # 4) Filtrar filas sin target
    valid = y.notna()
    X_all = X_all.loc[valid]
    y = pd.to_numeric(y.loc[valid], errors='coerce')
    valid = y.notna()
    X_all = X_all.loc[valid]
    y = y.loc[valid]

    if X_all.shape[1] == 0 or X_all.shape[0] == 0:
        return np.empty((0,0)), np.empty((0,0)), np.empty((0,)), np.empty((0,))

    # 5) Split
    return train_test_split(X_all, y, test_size=test_size, random_state=random_state)
