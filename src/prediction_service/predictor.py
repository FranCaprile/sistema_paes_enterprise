# src/prediction_service/predictor.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd

# Asegurar que el proyecto raíz esté en el path (para importar load_data, etc.)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from training_pipeline.load_data import cargar_configuracion, cargar_dataset  # ya usados en tu pipeline

SCORE_COLS = ['C. Lectora', 'Matemática', 'Historia', 'Ciencias', 'M2']


# ---------------------------
# Utilidades de normalización
# ---------------------------

def _slug(s: str) -> str:
    """slug consistente con el usado al guardar modelos."""
    return (
        str(s)
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "-")
    )

def _clean_student_rut_series(s: pd.Series) -> pd.Series:
    """Deja RUT/RUN en formato alfanumérico (sin puntos, guiones, espacios)."""
    return (
        s.astype(str)
         .str.replace(r'[^0-9A-Za-z]', '', regex=True)
         .str.strip()
    )

def _normalize_key(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra RUT/rut -> student_rut, elimina duplicados de nombre y limpia formato."""
    df = df.copy()
    if 'student_rut' not in df.columns:
        mapping = {}
        if 'RUT' in df.columns: mapping['RUT'] = 'student_rut'
        if 'rut' in df.columns: mapping['rut'] = 'student_rut'
        if mapping:
            df = df.rename(columns=mapping)
    # columna única
    df = df.loc[:, ~df.columns.duplicated()]
    if 'student_rut' in df.columns:
        df['student_rut'] = _clean_student_rut_series(df['student_rut'])
    return df


# ---------------------------
# Carga de artefactos y datos
# ---------------------------

def _load_processed_datasets(cfg: Dict) -> Dict[str, pd.DataFrame]:
    """
    Carga los datasets procesados necesarios para armar features.
    NO usa df_paes (en inferencia no deberíamos depender de los puntajes reales).
    """
    rutas = cfg['output_paths'].copy()
    # Cargamos solo los que sirven como features (no target)
    keys_to_load = [
        'df_pruebas_sepa',     # SEPA agregada (mat_*, leng_*)
        'df_indicadores',      # indicadores por alumno
        'df_pca',              # PC1..PCn
        'df_avg_diff',         # dificultad (por tipo y año) - opcional
        'df_avg_diff_agg',     # dificultad (agg por tipo) - opcional
        'df_taxonomia',        # bloom/taxonomía
    ]
    dfs: Dict[str, pd.DataFrame] = {}
    for k in keys_to_load:
        if k in rutas and Path(rutas[k]).exists():
            dfs[k] = cargar_dataset(Path(rutas[k]), tipo='csv')
        else:
            # puede que alguno no exista; está bien, seguimos
            pass

    # normalizar claves de fusión
    for k, df in list(dfs.items()):
        dfs[k] = _normalize_key(df)

    return dfs


def _find_latest_stacking_models(ml_dir: Path, pruebas: List[str]) -> Dict[str, Path]:
    """
    Busca los últimos modelos de stacking por prueba (target), con naming:
    <date>_stacking_allvars_<slug_prueba>.joblib
    Devuelve dict {prueba -> path al más reciente}.
    """
    ml_dir = Path(ml_dir)
    if not ml_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de modelos: {ml_dir}")

    # indexamos por slug para emparejar
    slug_map = { _slug(p): p for p in pruebas }

    # recolectar candidatos
    paths = list(ml_dir.glob("*stacking_allvars_*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No se encontraron modelos stacking_allvars en {ml_dir}")

    latest: Dict[str, Tuple[float, Path]] = {}
    for p in paths:
        name = p.name
        # esperamos sufijo ..._<slug>.joblib
        try:
            slug_prueba = name.split("stacking_allvars_")[1].replace(".joblib", "")
        except Exception:
            continue
        if slug_prueba not in slug_map:
            # nombre desconocido, ignorar
            continue
        prueba = slug_map[slug_prueba]
        mtime = p.stat().st_mtime
        if prueba not in latest or mtime > latest[prueba][0]:
            latest[prueba] = (mtime, p)

    # convertir a {prueba: Path}
    resolved = {pr: tup[1] for pr, tup in latest.items()}
    if not resolved:
        raise FileNotFoundError("No se pudo resolver ningún modelo más reciente por prueba.")
    return resolved


# ---------------------------
# Armado de features (infer)
# ---------------------------

def _pick_diff_key(dfs: Dict[str, pd.DataFrame]) -> str | None:
    if 'df_avg_diff_agg' in dfs: return 'df_avg_diff_agg'
    if 'df_avg_diff' in dfs:     return 'df_avg_diff'
    return None

def _build_allvars_features_for_rut(dfs: Dict[str, pd.DataFrame], rut: str) -> pd.DataFrame:
    """
    Construye UNA fila de features para el alumno con ese RUT.
    Mismo orden de merges que en entrenamiento (sin PAES):
    base = SEPA  LEFT JOIN indicadores LEFT JOIN pca LEFT JOIN diff LEFT JOIN taxonomía
    """
    if 'df_pruebas_sepa' not in dfs:
        raise KeyError("Falta df_pruebas_sepa en los datasets procesados.")

    target_rut = _clean_student_rut_series(pd.Series([rut]))[0]

    # base: una fila por alumno en df_pruebas_sepa
    base = dfs['df_pruebas_sepa']
    base = base[ base['student_rut'] == target_rut ].copy()
    if base.empty:
        raise KeyError(f"No encontré el RUT {rut} (normalizado: {target_rut}) en df_pruebas_sepa.")

    # LEFT JOIN otros datasets (pueden faltar columnas, se llenarán en 0)
    if 'df_indicadores' in dfs:
        base = base.merge(dfs['df_indicadores'], on='student_rut', how='left')
    if 'df_pca' in dfs:
        base = base.merge(dfs['df_pca'], on='student_rut', how='left')
    diff_key = _pick_diff_key(dfs)
    if diff_key:
        # df_avg_diff*_ viene con 'rut' o 'student_rut' normalizado en _load_processed_datasets
        base = base.merge(dfs[diff_key], on='student_rut', how='left')
    if 'df_taxonomia' in dfs:
        # quitar metadatos redundantes antes de unir
        blo = dfs['df_taxonomia'].drop(columns=['id', 'Nombre', 'Admisión'], errors='ignore')
        base = base.merge(blo, on='student_rut', how='left')

    # quitar columnas objetivo, ids y metadatos
    drop_cols = set(['student_rut', 'id', 'Nombre', 'Admisión'] + SCORE_COLS)
    feature_cols = [c for c in base.columns if c not in drop_cols]

    # asegurar numéricos + rellenar faltantes
    X = base[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # devolver con índice 0 (una fila)
    return X.iloc[[0]]  # DataFrame 1xF


def _expected_feature_order_from_model(stack_model, candidate_cols: List[str]) -> List[str] | None:
    """
    Intenta recuperar el orden de features usado en entrenamiento desde
    algún estimador base (feature_names_in_). Si no existe, retorna None.
    """
    try:
        for est in stack_model.estimators_:
            # estimador puede venir como Pipeline; sklearn suele propagar feature_names_in_
            if hasattr(est, "feature_names_in_"):
                names = list(est.feature_names_in_)
                # nos aseguramos que son columnas existentes (intersección)
                names = [c for c in names if c in candidate_cols]
                if names:
                    return names
    except Exception:
        pass
    return None


def _align_X_to_expected(X: pd.DataFrame, expected_order: List[str] | None) -> pd.DataFrame:
    """
    - Si expected_order existe: crea un DF con esas columnas, agregando faltantes=0,
      y descarta extras.
    - Si no existe: retorna X como viene (se asume mismo orden determinista que training).
    """
    if not expected_order:
        return X

    cols = list(expected_order)
    X_aligned = pd.DataFrame(index=X.index, columns=cols, dtype=float)

    for c in cols:
        if c in X.columns:
            X_aligned[c] = pd.to_numeric(X[c], errors='coerce')
        else:
            X_aligned[c] = 0.0  # columna faltante

    # garantizar sin NaN
    X_aligned = X_aligned.fillna(0.0)
    return X_aligned


# ---------------------------
# API pública
# ---------------------------

def predecir_paes_por_rut(rut: str) -> Dict[str, float]:
    """
    Carga el último modelo de stacking por prueba y predice para el alumno (rut).
    Retorna dict { 'C. Lectora': pred, 'Matemática': pred, ... } para las pruebas disponibles.
    """
    cfg = cargar_configuracion()
    dfs = _load_processed_datasets(cfg)

    # Buscar modelos más recientes por prueba
    modelos_por_prueba = _find_latest_stacking_models(
        ml_dir=Path(cfg['paths']['ml_models_dir']),
        pruebas=cfg['pruebas']
    )

    # Construir features UNA vez (válidas para todas las pruebas)
    X_row = _build_allvars_features_for_rut(dfs, rut)

    # Predecir para cada prueba disponible
    resultados: Dict[str, float] = {}
    for prueba, path_model in modelos_por_prueba.items():
        stack = joblib.load(path_model)

        # reordenar columnas si el modelo expone feature_names_in_
        expected = _expected_feature_order_from_model(stack, list(X_row.columns))
        X_in = _align_X_to_expected(X_row, expected)

        pred = float(stack.predict(X_in)[0])
        resultados[prueba] = pred

    return resultados


# ---------------------------
# Helpers opcionales para scripts
# ---------------------------

def get_any_student_rut() -> str | None:
    """Devuelve un RUT cualquiera presente en df_pruebas_sepa, para tests manuales."""
    cfg = cargar_configuracion()
    dfs = _load_processed_datasets(cfg)
    if 'df_pruebas_sepa' not in dfs:
        return None
    s = dfs['df_pruebas_sepa']['student_rut'].dropna()
    if s.empty:
        return None
    return str(s.iloc[0])
