import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict

def dataset_pca(dfs: Dict[str, pd.DataFrame], prueba: str):
    df = dfs['df_pruebas_sepa'].merge(dfs['df_pca'], on='student_rut')
    X = df[[c for c in df.columns if c.startswith('pca_')]]
    y = df[prueba]
    return train_test_split(X, y, test_size=0.2, random_state=42)