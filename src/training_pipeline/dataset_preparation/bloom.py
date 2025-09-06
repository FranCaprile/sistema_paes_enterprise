# src/training_pipeline/dataset_preparation/bloom.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
from load_data import normalize_rut as _normalize_rut

BLOOM_COLS = ["Conocer","Comprender","Aplicar","Analizar","Evaluar","Crear"]  # ajusta si difieren

def dataset_bloom(dfs: Dict[str, pd.DataFrame], target_col: str, test_size=0.2, random_state=42):
    # 1) Normalizar claves
    df_sepa = _normalize_rut(dfs["df_pruebas_sepa"])
    df_paes = _normalize_rut(dfs["df_paes"])
    df_tax  = _normalize_rut(dfs["df_taxonomia"])  # tiene 'rut' -> se renombra a student_rut más abajo

    # 2) Homologar nombre de RUT en taxonomía
    if "rut" in df_tax.columns and "student_rut" not in df_tax.columns:
        df_tax = df_tax.rename(columns={"rut": "student_rut"})

    # 3) Merge SEPA + PAES + TAXONOMÍA
    for d in (df_sepa, df_paes, df_tax):
        d["student_rut"] = (
            d["student_rut"].astype(str)
            .str.upper()
            .str.replace(r"[^0-9K]", "", regex=True)
            .str.strip()
        )
    df = (
        df_sepa.merge(df_paes, on="student_rut", how="inner")
               .merge(df_tax,  on="student_rut", how="inner")
    )

    # 4) Reemplazar '-' y castear targets
    df = df.replace("-", np.nan)
    if target_col not in df.columns:
        raise KeyError(f"Objetivo '{target_col}' no está en columnas: {df.columns.tolist()}")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # 5) Features Bloom presentes en el df
    bloom_feats = [c for c in BLOOM_COLS if c in df.columns]
    if not bloom_feats:
        raise ValueError(f"No encontré columnas Bloom en taxonomía. Esperaba {BLOOM_COLS}")

    # 6) No botar filas por NaN en Bloom: rellenar
    df[bloom_feats] = df[bloom_feats].astype(float).fillna(0.0)

    # 7) Botar solo filas sin target
    df = df.dropna(subset=[target_col])

    # 8) Armar X, y y split
    X = df[bloom_feats]
    y = df[target_col].astype(float)

    if len(X) == 0 or X.shape[1] == 0:
        # devolvemos splits vacíos para que el caller lo salte
        return (pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float))

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
